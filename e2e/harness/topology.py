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
                spec, spec.container(node.name), node.image, "node", node.cpus, node.memory, node.docker_args)
        for link in spec.links:
            _connect(spec, link, pids[spec.container(link.node)], pids[spec.router(link.network)])
        state = describe(spec)
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
                    link.network: {"address": str(link.node_address), "router": str(link.router_address),
                                   "subnet": str(spec.networks[link.network].subnet)}
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
            (node_pid, node_end, link.network, link.node_address),
            (router_pid, router_end, link.router_interface, link.router_address)):
        shell.in_netns(pid, "ip", "link", "set", temporary, "name", interface)
        shell.in_netns(pid, "ip", "link", "set", interface, "mtu", str(network.mtu), "up")
        shell.in_netns(pid, "ip", "address", "add", f"{address}/31", "dev", interface)
        if network.rate or network.delay:
            shell.in_netns(pid, "tc", "qdisc", "replace", "dev", interface, "root", "netem",
                           "limit", str(network.queue_limit),
                           *(["rate", network.rate] if network.rate else []),
                           *(["delay", network.delay] if network.delay else []))
    shell.in_netns(node_pid, "ip", "route", "add", str(network.subnet), "via", str(link.router_address),
                   "dev", link.network)


def _publish(spec: Spec, state: dict) -> None:
    """The description on the host, and in every node with the other nodes'
    names in /etc/hosts as ``<node>.<network>``."""

    STATE_DIRECTORY.mkdir(exist_ok=True)
    path = STATE_DIRECTORY / f"{spec.name}.json"
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    for node in spec.nodes.values():
        container = spec.container(node.name)
        hosts = "".join(
            f"{link.node_address} {link.node}.{link.network}\n"
            for link in spec.links if link.network in node.networks
        )
        exec_in(container, ["sh", "-c", f"mkdir -p {Path(CONTAINER_STATE).parent} && cat >> /etc/hosts"],
                input=hosts)
        with tempfile.NamedTemporaryFile("w", suffix=".json") as copy:
            copy.write(json.dumps({**state, "self": node.name}, indent=2) + "\n")
            copy.flush()
            shell.run(["docker", "cp", copy.name, f"{container}:{CONTAINER_STATE}"])


def _veth_prefix(name: str) -> str:
    """Host-side veth names (15 characters at most): a prefix per topology,
    so ``down`` can find what an interrupted ``up`` left behind."""

    return "e2e" + hashlib.sha256(name.encode()).hexdigest()[:4]
