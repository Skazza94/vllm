"""Bringing a topology up and down.

Containers start with ``--network none``: their only interfaces are the
veths the harness adds, configured from the host with ``nsenter`` (images
need no network tools). Every container carries the label
``e2e.topology=<name>``, which is how ``down`` finds them; deleting a
container deletes its network namespace and veths with it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import tempfile

from . import shell
from .spec import Spec


LABEL = "e2e.topology"
LOOSE_RP_FILTER = ("net.ipv4.conf.all.rp_filter=2", "net.ipv4.conf.default.rp_filter=2")
# A multi-link network's policy tables and rules: 1000 + 256 x the network's
# position in the spec + the link's slot on its node.
POLICY_TABLE_BASE = 1000
STATE_DIRECTORY = Path(__file__).resolve().parent.parent / "state"
# Where each container finds the topology's description.
CONTAINER_STATE = "/e2e/topology.json"


def up(spec: Spec) -> dict:
    """Create the topology's containers and links; returns its description
    (also in ``state/<name>.json`` and in every node at CONTAINER_STATE).
    Anything created is removed again if a step fails."""

    if containers(spec.name):
        raise RuntimeError(f"topology {spec.name} is already up: run down first")
    try:
        pids = {}
        for network in spec.networks:
            pids[spec.router(network)] = _start(
                spec, spec.router(network), spec.router_image, "router", spec.router_cpus, None, (),
                # The router forwards between its links.
                sysctls=("net.ipv4.ip_forward=1",))
        for node in spec.nodes.values():
            pids[spec.container(node.name)] = _start(
                spec, spec.container(node.name), node.image, "node", node.cpus, node.memory, node.docker_args,
                # Replies to link k's address arrive on link k, the subnet's
                # main route on link 0: a loose reverse-path check.
                sysctls=LOOSE_RP_FILTER if node.multi_link else ())
        for link in spec.links:
            _connect(spec, link, pids[spec.container(link.node)], pids[spec.router(link.network)])
        state = describe(spec)
        _publish(spec, state)
        if any(node.multi_link for node in spec.nodes.values()):
            state["link_check"] = check_links(spec, state)
            _publish(spec, state)
        return state
    except BaseException:
        down(spec.name)
        raise


def down(name: str) -> list[str]:
    """Remove the topology's containers and any veth left in the host's
    namespace by an interrupted ``up``; returns the containers removed."""

    removed = containers(name)
    if removed:
        shell.run(["docker", "rm", "-f", *removed])
    prefix = _veth_prefix(name)
    for line in shell.run(["ip", "-o", "link", "show"]).splitlines():
        interface = line.split(":", 2)[1].strip().split("@", 1)[0]
        if interface.startswith(prefix):
            shell.sudo("ip", "link", "delete", interface, check=False)
    (STATE_DIRECTORY / f"{name}.json").unlink(missing_ok=True)
    return removed


def containers(name: str) -> list[str]:
    """The topology's containers, running or not."""

    output = shell.run(["docker", "ps", "-a", "--filter", f"label={LABEL}={name}", "--format", "{{.Names}}"])
    return sorted(output.split())


def describe(spec: Spec) -> dict:
    """Each node's container and interfaces, and each network's router."""

    return {
        "name": spec.name,
        "nodes": {
            node.name: {
                "container": spec.container(node.name),
                "interfaces": {
                    link.interface: {"address": str(link.node_address), "router": str(link.router_address),
                                     "subnet": str(spec.networks[link.network].subnet), "network": link.network,
                                     "slot": link.slot, "router_port": link.router_interface}
                    for link in spec.links_of(node.name)
                },
            }
            for node in spec.nodes.values()
        },
        "networks": {
            network.name: {"router": spec.router(network.name), "subnet": str(network.subnet), "mtu": network.mtu,
                           "rate": network.rate, "delay": network.delay}
            for network in spec.networks.values()
        },
    }


def load_state(name: str) -> dict:
    path = STATE_DIRECTORY / f"{name}.json"
    if not path.is_file():
        raise RuntimeError(f"topology {name} is not up (no {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def exec_in(container: str, command: list[str], *, check: bool = True, timeout: float | None = 120,
            input: str | None = None) -> str:
    return shell.run(["docker", "exec", *(["-i"] if input is not None else []), container, *command],
                     check=check, timeout=timeout, input=input)


def _start(spec: Spec, container: str, image: str, role: str, cpus: str | None, memory: str | None,
           extra: tuple[str, ...], sysctls: tuple[str, ...] = ()) -> int:
    command = ["docker", "run", "-d", "--init", "--name", container, "--hostname", container,
               "--label", f"{LABEL}={spec.name}", "--label", f"e2e.role={role}",
               "--network", "none", "--sysctl", "net.ipv6.conf.all.disable_ipv6=1"]
    for sysctl in sysctls:
        command += ["--sysctl", sysctl]
    if cpus:
        command += ["--cpuset-cpus", cpus]
    if memory:
        command += ["--memory", memory]
    command += [*extra, "--entrypoint", "sleep", image, "infinity"]
    shell.run(command)
    return int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}", container]).strip())


def _connect(spec: Spec, link, node_pid: int, router_pid: int) -> None:
    """The veth pair between a node and its network's router, addressed,
    routed and shaped."""

    network = spec.networks[link.network]
    prefix = _veth_prefix(spec.name)
    node_end, router_end = (prefix + secrets.token_hex(4) for _ in range(2))
    shell.sudo("ip", "link", "add", node_end, "type", "veth", "peer", "name", router_end)
    shell.sudo("ip", "link", "set", node_end, "netns", str(node_pid))
    shell.sudo("ip", "link", "set", router_end, "netns", str(router_pid))
    for pid, temporary, interface, address in (
            (node_pid, node_end, link.interface, link.node_address),
            (router_pid, router_end, link.router_interface, link.router_address)):
        shell.in_netns(pid, "ip", "link", "set", temporary, "name", interface)
        shell.in_netns(pid, "ip", "link", "set", interface, "mtu", str(network.mtu), "up")
        shell.in_netns(pid, "ip", "address", "add", f"{address}/31", "dev", interface)
        if network.rate or network.delay:
            shell.in_netns(pid, "tc", "qdisc", "replace", "dev", interface, "root", "netem",
                           "limit", str(network.queue_limit),
                           *(["rate", network.rate] if network.rate else []),
                           *(["delay", network.delay] if network.delay else []))
    if link.links > 1:
        shell.in_netns(node_pid, "sysctl", "-q", "-w", f"net.ipv4.conf.{link.interface}.rp_filter=2")
    if link.slot == 0:
        # The subnet's main route (sockets not bound to a link's address).
        shell.in_netns(node_pid, "ip", "route", "add", str(network.subnet), "via", str(link.router_address),
                       "dev", link.interface)
    if link.links > 1:
        # Link k's own table: its /31, then the subnet via its router; and
        # the rule sending traffic from its address there.
        table = str(policy_table(spec, link))
        shell.in_netns(node_pid, "ip", "route", "add", f"{link.router_address}/31", "dev", link.interface,
                       "src", str(link.node_address), "table", table)
        shell.in_netns(node_pid, "ip", "route", "add", str(network.subnet), "via", str(link.router_address),
                       "dev", link.interface, "src", str(link.node_address), "table", table)
        shell.in_netns(node_pid, "ip", "rule", "add", "from", f"{link.node_address}/32", "lookup", table,
                       "priority", table)


def policy_table(spec: Spec, link) -> int:
    """A multi-link network's link's policy table (and rule priority), unique
    on its node: one block of 256 per network, the link's slot within it."""

    return POLICY_TABLE_BASE + 256 * list(spec.networks).index(link.network) + link.slot


def _publish(spec: Spec, state: dict) -> None:
    """The description on the host, and in every node with the other nodes'
    names in /etc/hosts as ``<node>.<network>``."""

    STATE_DIRECTORY.mkdir(exist_ok=True)
    path = STATE_DIRECTORY / f"{spec.name}.json"
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    for node in spec.nodes.values():
        container = spec.container(node.name)
        hosts = "".join(
            f"{link.node_address} {link.node}{f'-r{link.slot}' if link.links > 1 else ''}.{link.network}\n"
            for link in spec.links if link.network in node.networks
        )
        exec_in(container, ["sh", "-c", f"mkdir -p {Path(CONTAINER_STATE).parent} && cat >> /etc/hosts"],
                input=hosts)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as copy:
            copy.write(json.dumps({**state, "self": node.name}, indent=2) + "\n")
            copy.flush()
            shell.run(["docker", "cp", copy.name, f"{container}:{CONTAINER_STATE}"])


CHECK_RECEIVE = """
import socket, sys
server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((sys.argv[1], int(sys.argv[2])))
server.listen()
connection, _ = server.accept()
received = 0
while chunk := connection.recv(1 << 20):
    received += len(chunk)
connection.sendall(b"k")
print(received)
"""
CHECK_SEND = """
import socket, sys, time
source, host, port, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for _ in range(200):
    try:
        sender = socket.create_connection((host, port), timeout=10, source_address=(source, 0))
        break
    except ConnectionRefusedError:
        time.sleep(0.02)
sender.sendall(b"x" * size)
sender.shutdown(socket.SHUT_WR)
sender.recv(1)
"""
CHECK_BYTES = 256 << 10
CHECK_PORT = 47000


def check_links(spec: Spec, state: dict) -> list[dict]:
    """Every multi-link network's links, both ways between every two of its
    nodes, before any workload: the route from each link's address to each
    peer link leaves on that link, and a source-bound TCP exchange of
    CHECK_BYTES raises the source link's router port RX and the destination's
    TX by at least that, and no other port's by as much. Raises on the first
    failure; returns each exchange's evidence."""

    results = []
    for network in spec.networks.values():
        nodes = [node for node in spec.nodes.values() if network.name in node.networks]
        if not any(node.links_on(network.name) > 1 for node in nodes):
            continue
        router_pid = _pid(spec.router(network.name))
        for source in nodes:
            for target in nodes:
                if source is target:
                    continue
                for out in [link for link in spec.links_of(source.name) if link.network == network.name]:
                    for into in [link for link in spec.links_of(target.name) if link.network == network.name]:
                        results.append(_check_exchange(spec, network.name, router_pid, out, into))
    return results


def _check_exchange(spec: Spec, network: str, router_pid: int, out, into) -> dict:
    source, target = spec.container(out.node), spec.container(into.node)
    # Images carry no network tools: the host's ip, in the node's namespace.
    route = json.loads(shell.in_netns(_pid(source), "ip", "-j", "route", "get", str(into.node_address), "from",
                                      str(out.node_address)) or "[]")
    device = route[0].get("dev") if route else None
    if device != out.interface:
        raise RuntimeError(f"{out.node} link {out.slot} ({out.node_address}) routes to {into.node_address} via "
                           f"{device}, not {out.interface}")
    before = _router_ports(router_pid)
    port = CHECK_PORT + into.index
    shell.run(["docker", "exec", "-d", target, "python3", "-c", CHECK_RECEIVE, str(into.node_address), str(port)])
    exec_in(source, ["python3", "-c", CHECK_SEND, str(out.node_address), str(into.node_address), str(port),
                     str(CHECK_BYTES)])
    after = _router_ports(router_pid)
    moved = {name: {direction: after[name][direction] - before.get(name, {}).get(direction, 0)
                    for direction in ("rx", "tx")} for name in after}
    sent, delivered = moved[out.router_interface]["rx"], moved[into.router_interface]["tx"]
    others = {name: value for name, value in moved.items()
              if name not in (out.router_interface, into.router_interface)
              and max(value.values()) >= CHECK_BYTES}
    if sent < CHECK_BYTES or delivered < CHECK_BYTES or others:
        raise RuntimeError(f"{network}: {out.node} link {out.slot} -> {into.node} link {into.slot}: router port "
                           f"{out.router_interface} received {sent}, {into.router_interface} sent {delivered} of "
                           f"{CHECK_BYTES} bytes; other ports moving as much: {others}")
    return {"network": network, "from": [out.node, out.slot, str(out.node_address), out.router_interface],
            "to": [into.node, into.slot, str(into.node_address), into.router_interface], "route_dev": device,
            "router_rx": sent, "router_tx": delivered,
            # Every port's bytes over the exchange: the others carry less.
            "ports": dict(sorted(moved.items(), key=lambda item: int(item[0][1:]) if item[0][1:].isdigit()
                                 else -1))}


def _router_ports(router_pid: int) -> dict[str, dict[str, int]]:
    links = json.loads(shell.in_netns(router_pid, "ip", "-j", "-s", "link", "show"))
    return {link["ifname"]: {"rx": link["stats64"]["rx"]["bytes"], "tx": link["stats64"]["tx"]["bytes"]}
            for link in links if link["ifname"].startswith("p")}


def _pid(container: str) -> int:
    return int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}", container]).strip())


def _veth_prefix(name: str) -> str:
    """Host-side veth names (15 characters at most): a prefix per topology,
    so ``down`` can find what an interrupted ``up`` left behind."""

    return "e2e" + hashlib.sha256(name.encode()).hexdigest()[:4]
