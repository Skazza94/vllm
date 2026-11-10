"""Topologies on this host's docker: set E2E_DOCKER=1 to run them."""

from __future__ import annotations

import json
import os
import textwrap
import time
import unittest

from harness import shell, spec, topology


SMALL_IMAGE = "python:3.12-slim"
NEEDS_DOCKER = unittest.skipUnless(os.environ.get("E2E_DOCKER") == "1", "set E2E_DOCKER=1 to run docker topologies")

RECEIVE = textwrap.dedent("""
    import socket, sys
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", int(sys.argv[1])))
    server.listen()
    connection, _ = server.accept()
    received = 0
    while chunk := connection.recv(1 << 20):
        received += len(chunk)
    open("/tmp/received", "w").write(str(received))
""")
SEND = textwrap.dedent("""
    import socket, sys, time
    host, port, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    for _ in range(100):
        try:
            sender = socket.create_connection((host, port), timeout=10)
            break
        except ConnectionRefusedError:
            time.sleep(0.05)
    chunk = b"x" * (1 << 20)
    start = time.monotonic()
    sent = 0
    while sent < size:
        sender.sendall(chunk[: size - sent])
        sent += min(len(chunk), size - sent)
    sender.shutdown(socket.SHUT_WR)
    sender.recv(1)
    print(time.monotonic() - start)
""")
REACH = textwrap.dedent("""
    import socket, sys
    try:
        socket.create_connection((sys.argv[1], 9), timeout=2)
        print("connected")
    except ConnectionRefusedError:
        print("refused")  # reached the host, nothing listening
    except OSError as error:
        print(type(error).__name__, error.errno)
""")


def pid(container: str) -> int:
    return int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}", container]).strip())


def interfaces(container: str) -> dict[str, dict]:
    links = json.loads(shell.in_netns(pid(container), "ip", "-j", "-s", "address", "show"))
    return {link["ifname"]: link for link in links}


def routes(container: str) -> list[str]:
    return sorted(f"{r['dst']} via {r.get('gateway', '-')} dev {r['dev']}"
                  for r in json.loads(shell.in_netns(pid(container), "ip", "-j", "route", "show")))


def transfer(parsed: spec.Spec, sender: str, receiver: str, network: str, size: int, port: int) -> float:
    """Send ``size`` bytes from one node to another's address on ``network``;
    returns the seconds it took."""

    target = parsed.container(receiver)
    shell.run(["docker", "exec", "-d", target, "python3", "-c", RECEIVE, str(port)])
    seconds = float(topology.exec_in(parsed.container(sender), [
        "python3", "-c", SEND, str(parsed.address(receiver, network)), str(port), str(size)]))
    for _ in range(50):
        received = topology.exec_in(target, ["cat", "/tmp/received"], check=False).strip()
        if received:
            break
        time.sleep(0.1)
    topology.exec_in(target, ["rm", "-f", "/tmp/received"])
    if int(received) != size:
        raise AssertionError(f"{receiver} received {received} of {size} bytes")
    return seconds


def router_bytes(parsed: spec.Spec, network: str) -> int:
    """Bytes the network's router received on all its links."""

    return sum(link["stats64"]["rx"]["bytes"] for name, link in interfaces(parsed.router(network)).items()
               if name != "lo")


@NEEDS_DOCKER
class StarTopologyTests(unittest.TestCase):
    SPEC = spec.parse({
        "name": "e2e-selftest",
        "image": SMALL_IMAGE,
        "networks": {"control": {"subnet": "10.250.0.0/24"},
                     "data0": {"subnet": "10.100.0.0/24", "mtu": 9000},
                     "data1": {"subnet": "10.101.0.0/24", "mtu": 9000}},
        "nodes": {"prefill0": {"networks": ["control", "data0", "data1"]},
                  "decode0": {"networks": ["control", "data0", "data1"]},
                  "client": {"networks": ["control"]}},
    })

    @classmethod
    def setUpClass(cls) -> None:
        topology.down(cls.SPEC.name)
        cls.state = topology.up(cls.SPEC)

    @classmethod
    def tearDownClass(cls) -> None:
        topology.down(cls.SPEC.name)

    def test_nodes_have_only_their_veths_and_subnet_routes(self) -> None:
        for node in self.SPEC.nodes.values():
            with self.subTest(node=node.name):
                found = interfaces(self.SPEC.container(node.name))
                self.assertEqual({"lo", *node.networks} - set(found), set())
                # Beyond its veths, only the kernel's fallback tunnel devices
                # (ip6tnl0 and the like, when their modules are loaded): down,
                # without addresses.
                for name in set(found) - {"lo", *node.networks}:
                    self.assertNotIn("UP", found[name]["flags"], name)
                    self.assertEqual(found[name]["addr_info"], [], name)
                for link in self.SPEC.links_of(node.name):
                    interface = found[link.network]
                    self.assertEqual(interface["mtu"], self.SPEC.networks[link.network].mtu)
                    self.assertEqual([(a["local"], a["prefixlen"]) for a in interface["addr_info"]],
                                     [(str(link.node_address), 31)])
                self.assertEqual(routes(self.SPEC.container(node.name)), sorted(
                    [f"{self.SPEC.networks[link.network].subnet} via {link.router_address} dev {link.network}"
                     for link in self.SPEC.links_of(node.name)]
                    # The link's own /31, which starts at the router's address.
                    + [f"{link.router_address}/31 via - dev {link.network}" for link in self.SPEC.links_of(node.name)]))

    def test_nodes_reach_each_other_only_on_shared_networks(self) -> None:
        def reach(source: str, target: str, network: str) -> str:
            return topology.exec_in(self.SPEC.container(source), [
                "python3", "-c", REACH, str(self.SPEC.address(target, network))]).strip()

        for network in ("control", "data0", "data1"):
            self.assertEqual(reach("prefill0", "decode0", network), "refused")
        self.assertEqual(reach("client", "prefill0", "control"), "refused")
        # The client has no route to a data network (ENETUNREACH).
        self.assertEqual(reach("client", "decode0", "data0"), "OSError 101")

    def test_names_resolve_to_the_network_address(self) -> None:
        resolved = topology.exec_in(self.SPEC.container("prefill0"), [
            "python3", "-c", "import socket; print(socket.gethostbyname('decode0.data1'))"]).strip()
        self.assertEqual(resolved, str(self.SPEC.address("decode0", "data1")))
        described = json.loads(topology.exec_in(self.SPEC.container("client"), ["cat", topology.CONTAINER_STATE]))
        self.assertEqual(described["self"], "client")
        self.assertEqual(described["nodes"]["decode0"]["interfaces"]["data1"]["address"],
                         str(self.SPEC.address("decode0", "data1")))

    def test_traffic_crosses_only_its_networks_router(self) -> None:
        before = {network: router_bytes(self.SPEC, network) for network in self.SPEC.networks}
        size = 200 << 20
        transfer(self.SPEC, "prefill0", "decode0", "data1", size, 5001)
        grown = {network: router_bytes(self.SPEC, network) - before[network] for network in self.SPEC.networks}
        self.assertGreaterEqual(grown["data1"], size)
        self.assertLess(grown["data0"], 1 << 20)
        self.assertLess(grown["control"], 1 << 20)


@NEEDS_DOCKER
class ShapingTests(unittest.TestCase):
    SPEC = spec.parse({
        "name": "e2e-selftest-shaped",
        "image": SMALL_IMAGE,
        "networks": {"data0": {"subnet": "10.100.0.0/24", "rate": "1gbit", "delay": "1ms"}},
        "nodes": {"a": {"networks": ["data0"]}, "b": {"networks": ["data0"]}},
    })

    def test_links_are_shaped_and_down_leaves_nothing(self) -> None:
        topology.down(self.SPEC.name)
        topology.up(self.SPEC)
        try:
            rtt = topology.exec_in(self.SPEC.container("a"), ["python3", "-c", textwrap.dedent(f"""
                import socket, time
                start = time.monotonic()
                try:
                    socket.create_connection(("{self.SPEC.address('b', 'data0')}", 9), timeout=2)
                except ConnectionRefusedError:
                    pass
                print(time.monotonic() - start)
            """)])
            # SYN and RST each cross two links of 1 ms.
            self.assertGreaterEqual(float(rtt), 0.004)
            size = 50 << 20
            seconds = transfer(self.SPEC, "a", "b", "data0", size, 5002)
            gbits = size * 8 / seconds / 1e9
            self.assertLess(gbits, 1.05)
            self.assertGreater(gbits, 0.5)
        finally:
            topology.down(self.SPEC.name)
        self.assertEqual(topology.containers(self.SPEC.name), [])
        self.assertFalse((topology.STATE_DIRECTORY / f"{self.SPEC.name}.json").exists())
        leftovers = [line for line in shell.run(["ip", "-o", "link", "show"]).splitlines()
                     if topology._veth_prefix(self.SPEC.name) in line]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()


@NEEDS_DOCKER
class MultiLinkTopologyTests(unittest.TestCase):
    """A switch with a link per rank: a node's sockets bound to link k's
    address leave and return on link k."""

    SPEC = spec.parse({
        "name": "e2e-multilink",
        "image": SMALL_IMAGE,
        "networks": {"control": {"subnet": "10.251.0.0/24"},
                     "kv": {"subnet": "10.102.0.0/24", "mtu": 9000}},
        "nodes": {"prefill0": {"networks": ["control", {"kv": 3}]},
                  "decode0": {"networks": ["control", {"kv": 2}]},
                  "client": {"networks": ["control"]}},
    })

    @classmethod
    def setUpClass(cls) -> None:
        topology.down(cls.SPEC.name)
        cls.state = topology.up(cls.SPEC)

    @classmethod
    def tearDownClass(cls) -> None:
        topology.down(cls.SPEC.name)

    def test_links_were_checked_both_ways_before_use(self) -> None:
        checks = self.state["link_check"]
        # 3 x 2 one way, 2 x 3 the other.
        self.assertEqual(len(checks), 12)
        self.assertEqual({tuple(c["from"][:2]) for c in checks},
                         {("prefill0", 0), ("prefill0", 1), ("prefill0", 2), ("decode0", 0), ("decode0", 1)})
        for check in checks:
            self.assertGreaterEqual(min(check["router_rx"], check["router_tx"]), topology.CHECK_BYTES)
            # Every router port is recorded; only the exchange's two carry it.
            self.assertEqual(set(check["ports"]), {f"p{index}" for index in range(5)})
            self.assertEqual(check["ports"][check["from"][3]]["rx"], check["router_rx"])
            self.assertEqual(check["ports"][check["to"][3]]["tx"], check["router_tx"])
            others = [value for name, value in check["ports"].items() if name not in (check["from"][3], check["to"][3])]
            self.assertTrue(all(max(value.values()) < topology.CHECK_BYTES for value in others), check)
        self.assertEqual(json.loads((topology.STATE_DIRECTORY / "e2e-multilink.json").read_text())["link_check"],
                         checks)

    def test_each_link_has_its_interface_table_and_rule(self) -> None:
        node = pid(self.SPEC.container("prefill0"))
        names = set(interfaces(self.SPEC.container("prefill0")))
        self.assertTrue({"control", "kvr0", "kvr1", "kvr2"} <= names)
        rules = json.loads(shell.in_netns(node, "ip", "-j", "rule", "show"))
        for link in [l for l in self.SPEC.links_of("prefill0") if l.network == "kv"]:
            table = topology.policy_table(self.SPEC, link)
            self.assertTrue(any(r.get("src") == str(link.node_address) and str(r.get("table")) == str(table)
                                for r in rules), rules)
            for peer in [l for l in self.SPEC.links_of("decode0") if l.network == "kv"]:
                route = json.loads(shell.in_netns(node, "ip", "-j", "route", "get", str(peer.node_address),
                                                  "from", str(link.node_address)))
                self.assertEqual(route[0]["dev"], link.interface)
        # Unbound sockets keep the subnet's main route on link 0, and the
        # control network its own.
        self.assertIn("10.102.0.0/24 via 10.102.0.0 dev kvr0", routes(self.SPEC.container("prefill0")))
        self.assertIn("10.251.0.0/24 via 10.251.0.0 dev control", routes(self.SPEC.container("prefill0")))
        self.assertEqual(topology.exec_in(self.SPEC.container("client"), [
            "python3", "-c", REACH, str(self.SPEC.address("decode0", "control"))]).strip(), "refused")
        hosts = topology.exec_in(self.SPEC.container("decode0"), ["cat", "/etc/hosts"])
        self.assertIn("10.102.0.3 prefill0-r1.kv", hosts)
        self.assertIn("prefill0.control", hosts)
