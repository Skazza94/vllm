"""Topology specs: nodes (containers), networks, and the address plan.

Every network is a star: one router container in the middle and one veth
pair from each of the network's nodes to it. Link ``k`` of a network (its
``k``-th link, in spec order) is a /31 taken from the network's subnet: the
router holds the even address, the node the odd one. A node routes the whole
subnet through its router, and nothing else: no default route.

A node may join a network with several links (``{kv: 8}``: a NIC per rank on
one switch), each its own /31 and interface ``<network>r<k>``; a socket bound
to link k's address leaves through link k (source-policy routing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
from pathlib import Path
import re

import yaml


NAME = re.compile(r"^[a-z][a-z0-9-]*$")
# Linux interface names: at most 15 characters. A node's interface is
# named after its network.
INTERFACE = re.compile(r"^[a-z][a-z0-9]{0,14}$")
# Linux's limit on an interface name's length.
MAX_INTERFACE = 15
RATE = re.compile(r"^\d+(\.\d+)?(bit|kbit|mbit|gbit|tbit)$")
DELAY = re.compile(r"^\d+(\.\d+)?(us|ms|s)$")
DEFAULT_IMAGE = "vllm:cpu"
DEFAULT_ROUTER_IMAGE = "python:3.12-slim"


@dataclass(frozen=True)
class Network:
    name: str
    subnet: ipaddress.IPv4Network
    mtu: int = 1500
    # Each link's egress shaping (tc netem), both ways: a path between two
    # nodes crosses two links, so it sees the rate once and the delay twice.
    rate: str | None = None
    delay: str | None = None
    queue_limit: int = 10000  # netem's queue, in packets


@dataclass(frozen=True)
class Node:
    name: str
    networks: tuple[str, ...]
    image: str
    cpus: str | None = None
    memory: str | None = None
    docker_args: tuple[str, ...] = ()
    link_counts: tuple[tuple[str, int], ...] = ()
    """Networks the node joins with several links, and how many."""

    def links_on(self, network: str) -> int:
        return dict(self.link_counts).get(network, 1)

    @property
    def multi_link(self) -> bool:
        return any(count > 1 for _, count in self.link_counts)


@dataclass(frozen=True)
class Link:
    """One node's veth pair to its network's router."""

    network: str
    node: str
    index: int  # the link's position among the network's links
    node_address: ipaddress.IPv4Address
    router_address: ipaddress.IPv4Address
    slot: int = 0  # among the node's links on the network
    links: int = 1  # the node's links on the network

    @property
    def router_interface(self) -> str:
        return f"p{self.index}"

    @property
    def interface(self) -> str:
        """The node's interface: the network's name, or with several links
        ``<network>r<slot>``."""
        return self.network if self.links == 1 else f"{self.network}r{self.slot}"


@dataclass(frozen=True)
class Spec:
    name: str
    networks: dict[str, Network]
    nodes: dict[str, Node]
    router_image: str = DEFAULT_ROUTER_IMAGE
    router_cpus: str | None = None
    links: tuple[Link, ...] = field(default=())

    def container(self, node: str) -> str:
        return f"{self.name}-{node}"

    def router(self, network: str) -> str:
        return f"{self.name}-router-{network}"

    def links_of(self, node: str) -> list[Link]:
        return [link for link in self.links if link.node == node]

    def address(self, node: str, network: str, slot: int = 0) -> ipaddress.IPv4Address:
        for link in self.links:
            if (link.node, link.network, link.slot) == (node, network, slot):
                return link.node_address
        raise KeyError(f"node {node} has no link {slot} on network {network}")


def load(path: str | Path) -> Spec:
    with open(path, encoding="utf-8") as handle:
        return parse(yaml.safe_load(handle) or {})


def parse(raw: dict) -> Spec:
    """Validate a spec and build its address plan; raises ValueError."""

    if not isinstance(raw, dict):
        raise ValueError("a spec is a mapping")
    unknown = set(raw) - {"name", "image", "router_image", "routers", "networks", "nodes"}
    if unknown:
        raise ValueError(f"unknown spec key(s): {', '.join(sorted(unknown))}")
    name = raw.get("name")
    if not isinstance(name, str) or not NAME.match(name):
        raise ValueError("name must be lowercase letters, digits and dashes, starting with a letter")
    image = raw.get("image", DEFAULT_IMAGE)
    routers = raw.get("routers") or {}
    if not isinstance(routers, dict) or set(routers) - {"cpus"}:
        raise ValueError("routers takes only cpus")

    networks: dict[str, Network] = {}
    for net_name, options in (raw.get("networks") or {}).items():
        networks[net_name] = _network(net_name, options or {})
    if not networks:
        raise ValueError("a spec needs at least one network")
    subnets = list(networks.values())
    for i, first in enumerate(subnets):
        for second in subnets[i + 1:]:
            if first.subnet.overlaps(second.subnet):
                raise ValueError(f"networks {first.name} and {second.name} overlap")

    nodes: dict[str, Node] = {}
    for node_name, options in (raw.get("nodes") or {}).items():
        nodes[node_name] = _node(node_name, options or {}, networks, image)
    if not nodes:
        raise ValueError("a spec needs at least one node")
    for node_name in nodes:
        if node_name.startswith("router-"):
            raise ValueError(f"node {node_name}: names starting with router- are the routers'")

    links = []
    for network in networks.values():
        members = [(node, slot, node.links_on(network.name)) for node in nodes.values()
                   if network.name in node.networks for slot in range(node.links_on(network.name))]
        if 2 * len(members) > network.subnet.num_addresses:
            raise ValueError(f"network {network.name}: {network.subnet} is too small for {len(members)} link(s)")
        for index, (node, slot, count) in enumerate(members):
            base = network.subnet.network_address + 2 * index
            link = Link(network.name, node.name, index, base + 1, base, slot, count)
            if len(link.interface) > MAX_INTERFACE:
                raise ValueError(f"node {node.name}: interface {link.interface} is longer than {MAX_INTERFACE} "
                                 "characters (shorten the network's name)")
            links.append(link)
    return Spec(name=name, networks=networks, nodes=nodes,
                router_image=raw.get("router_image", DEFAULT_ROUTER_IMAGE),
                router_cpus=routers.get("cpus"), links=tuple(links))


def _network(name: str, options: dict) -> Network:
    if not INTERFACE.match(name):
        raise ValueError(f"network {name}: names are interface names (a letter, then up to 14 letters or digits)")
    unknown = set(options) - {"subnet", "mtu", "rate", "delay", "queue_limit"}
    if unknown:
        raise ValueError(f"network {name}: unknown key(s) {', '.join(sorted(unknown))}")
    try:
        subnet = ipaddress.IPv4Network(options["subnet"])
    except KeyError:
        raise ValueError(f"network {name} needs a subnet") from None
    except (ValueError, TypeError) as error:
        raise ValueError(f"network {name}: subnet {options['subnet']!r}: {error}") from None
    mtu = options.get("mtu", 1500)
    if not isinstance(mtu, int) or not 68 <= mtu <= 65535:
        raise ValueError(f"network {name}: mtu must be an integer from 68 to 65535")
    rate, delay = options.get("rate"), options.get("delay")
    if rate is not None and not RATE.match(str(rate)):
        raise ValueError(f"network {name}: rate is a tc rate such as 100gbit")
    if delay is not None and not DELAY.match(str(delay)):
        raise ValueError(f"network {name}: delay is a tc time such as 50us")
    limit = options.get("queue_limit", 10000)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError(f"network {name}: queue_limit must be a positive integer")
    return Network(name, subnet, mtu, rate and str(rate), delay and str(delay), limit)


def _node(name: str, options: dict, networks: dict[str, Network], image: str) -> Node:
    if not NAME.match(name):
        raise ValueError(f"node {name}: names are lowercase letters, digits and dashes")
    unknown = set(options) - {"networks", "image", "cpus", "memory", "docker_args"}
    if unknown:
        raise ValueError(f"node {name}: unknown key(s) {', '.join(sorted(unknown))}")
    listed = options.get("networks") or []
    if not listed or not isinstance(listed, list):
        raise ValueError(f"node {name} needs a list of networks")
    # A network's name, or {name: links}.
    on, counts = [], []
    for entry in listed:
        if isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(f"node {name}: a network with links is one {{network: links}} mapping")
            (net, count), = entry.items()
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"node {name}: network {net} takes a positive number of links")
            if count > 1:
                counts.append((net, count))
            on.append(net)
        else:
            on.append(entry)
    missing = [net for net in on if net not in networks]
    if missing:
        raise ValueError(f"node {name}: unknown network(s) {', '.join(missing)}")
    if len(set(on)) != len(on):
        raise ValueError(f"node {name}: a network is listed twice")
    args = options.get("docker_args") or []
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError(f"node {name}: docker_args is a list of strings")
    return Node(name, tuple(on), options.get("image", image), options.get("cpus"),
                options.get("memory"), tuple(args), tuple(counts))
