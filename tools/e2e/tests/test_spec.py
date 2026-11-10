from __future__ import annotations

import copy
import ipaddress
from pathlib import Path
import unittest

from harness import spec


TOPOLOGIES = Path(__file__).resolve().parent.parent / "topologies"
RAW = {
    "name": "t",
    "networks": {"control": {"subnet": "10.250.0.0/24"},
                 "data0": {"subnet": "10.100.0.0/24", "mtu": 9000, "rate": "100gbit", "delay": "5us"}},
    "nodes": {"a": {"networks": ["control", "data0"]},
              "b": {"networks": ["control", "data0"]},
              "client": {"networks": ["control"]}},
}


def with_changes(**changes) -> dict:
    raw = copy.deepcopy(RAW)
    raw.update(changes)
    return raw


class SpecTests(unittest.TestCase):
    def test_every_example_topology_parses(self) -> None:
        paths = sorted(TOPOLOGIES.glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                spec.load(path)

    def test_each_link_is_a_31_with_the_router_on_the_even_address(self) -> None:
        parsed = spec.parse(RAW)
        plan = {(link.network, link.node): (str(link.router_address), str(link.node_address), link.index)
                for link in parsed.links}
        self.assertEqual(plan, {
            ("control", "a"): ("10.250.0.0", "10.250.0.1", 0),
            ("control", "b"): ("10.250.0.2", "10.250.0.3", 1),
            ("control", "client"): ("10.250.0.4", "10.250.0.5", 2),
            ("data0", "a"): ("10.100.0.0", "10.100.0.1", 0),
            ("data0", "b"): ("10.100.0.2", "10.100.0.3", 1),
        })
        self.assertEqual(parsed.address("b", "data0"), ipaddress.IPv4Address("10.100.0.3"))
        self.assertEqual(parsed.container("a"), "t-a")
        self.assertEqual(parsed.router("data0"), "t-router-data0")
        self.assertEqual(parsed.networks["data0"].rate, "100gbit")
        self.assertEqual(parsed.nodes["a"].image, spec.DEFAULT_IMAGE)

    def test_a_node_may_join_a_network_with_several_links(self) -> None:
        parsed = spec.parse(with_changes(nodes={"a": {"networks": ["control", {"data0": 3}]},
                                                "b": {"networks": ["control", {"data0": 2}]},
                                                "client": {"networks": ["control"]}}))
        plan = [(link.node, link.slot, link.interface, str(link.node_address), link.router_interface)
                for link in parsed.links if link.network == "data0"]
        self.assertEqual(plan, [
            ("a", 0, "data0r0", "10.100.0.1", "p0"), ("a", 1, "data0r1", "10.100.0.3", "p1"),
            ("a", 2, "data0r2", "10.100.0.5", "p2"), ("b", 0, "data0r0", "10.100.0.7", "p3"),
            ("b", 1, "data0r1", "10.100.0.9", "p4")])
        self.assertEqual(parsed.address("b", "data0", 1), ipaddress.IPv4Address("10.100.0.9"))
        self.assertTrue(parsed.nodes["a"].multi_link)
        self.assertFalse(parsed.nodes["client"].multi_link)
        # One link keeps the network's name, as before.
        self.assertEqual([link.interface for link in parsed.links if link.network == "control"],
                         ["control"] * 3)
        self.assertEqual(spec.parse(with_changes(nodes={"a": {"networks": [{"data0": 1}]}})).links[0].interface,
                         "data0")

    def test_invalid_specs_are_rejected(self) -> None:
        cases = [
            (with_changes(name="Bad_Name"), "name must be"),
            (with_changes(extra=1), "unknown spec key"),
            (with_changes(networks={}), "at least one network"),
            (with_changes(networks={**RAW["networks"], "data1": {"subnet": "10.100.0.128/25"}}), "overlap"),
            (with_changes(networks={**RAW["networks"], "data-1": {"subnet": "10.1.0.0/24"}}), "interface names"),
            (with_changes(networks={**RAW["networks"], "averyveryverylongname": {"subnet": "10.1.0.0/24"}}),
             "interface names"),
            (with_changes(networks={"control": {"subnet": "10.250.0.0/31"}, "data0": RAW["networks"]["data0"]}),
             "too small"),
            (with_changes(networks={**RAW["networks"], "data1": {"subnet": "nope"}}), "subnet"),
            (with_changes(networks={**RAW["networks"], "data1": {"subnet": "10.1.0.0/24", "rate": "fast"}}),
             "rate is"),
            (with_changes(networks={**RAW["networks"], "data1": {"subnet": "10.1.0.0/24", "delay": "5"}}),
             "delay is"),
            (with_changes(networks={**RAW["networks"], "data1": {"subnet": "10.1.0.0/24", "mtu": 10}}), "mtu"),
            (with_changes(nodes={"a": {"networks": ["data9"]}}), "unknown network"),
            (with_changes(nodes={"a": {"networks": []}}), "needs a list"),
            (with_changes(nodes={"a": {"networks": ["data0", "data0"]}}), "listed twice"),
            (with_changes(nodes={"router-x": {"networks": ["data0"]}}), "router-"),
            (with_changes(nodes={"a": {"networks": ["data0"], "gpu": 1}}), "unknown key"),
            (with_changes(routers={"memory": "1g"}), "routers takes only cpus"),
            (with_changes(nodes={"a": {"networks": [{"data0": 0}]}}), "positive number of links"),
            (with_changes(nodes={"a": {"networks": [{"data0": True}]}}), "positive number of links"),
            (with_changes(nodes={"a": {"networks": [{"data0": 2, "control": 2}]}}), "one {network: links}"),
            (with_changes(nodes={"a": {"networks": ["data0", {"data0": 2}]}}), "listed twice"),
            # data0r10 fits 15 characters; a 13-character network's link 10 does not.
            (with_changes(networks={"averylongnet1": {"subnet": "10.1.0.0/24"}},
                          nodes={"a": {"networks": [{"averylongnet1": 11}]}}), "longer than 15"),
            (with_changes(networks={"data0": {"subnet": "10.100.0.0/29"}},
                          nodes={"a": {"networks": [{"data0": 4}]}, "b": {"networks": ["data0"]}}),
             "too small for 5 link"),
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                spec.parse(raw)


if __name__ == "__main__":
    unittest.main()
