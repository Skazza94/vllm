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
        ]
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                spec.parse(raw)


if __name__ == "__main__":
    unittest.main()
