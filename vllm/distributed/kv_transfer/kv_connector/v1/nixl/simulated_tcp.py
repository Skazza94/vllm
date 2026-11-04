# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Managed CPU UCX/TCP binding, before the NIXL agent creates its context."""

import os
import socket

import psutil


def bind_ucx_tcp(address: str, extra: dict) -> dict:
    """Select exactly one Ethernet interface, without UCX device-list rewriting.

    UCX restricts devices rather than source-IP aliases. Requiring one IPv4
    address on the selected interface avoids claiming that an alias pins the
    source address. NIXL's device_list parameter appends ':1' and overrides
    NET_DEVICES unconditionally; it must never enter this CPU TCP path.
    """
    forbidden = {"device_list", "engine_config", "backend_params", "backend_options"}
    if forbidden.intersection(extra):
        raise ValueError(
            "Managed UCX/TCP does not accept backend/device_list overrides"
        )
    # The pinned Python agent forwards only num_threads during construction.
    # Reject plugin/environment override names too, rather than silently
    # accepting a setting that a future binding could forward.
    if any(
        k.startswith("NIXL_UCX_") and k != "NIXL_UCX_WARNING_TIMEOUT"
        for k in os.environ
    ):
        raise ValueError("Managed UCX/TCP does not accept NIXL_UCX overrides")
    interfaces = psutil.net_if_addrs()
    matches = [
        name
        for name, addrs in interfaces.items()
        if any(a.family == socket.AF_INET and a.address == address for a in addrs)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"UCX/TCP address {address} needs one unique local interface: {matches}"
        )
    interface = matches[0]
    ipv4 = [a.address for a in interfaces[interface] if a.family == socket.AF_INET]
    if ipv4 != [address] or address.startswith("127."):
        raise ValueError(
            f"UCX/TCP interface {interface} has ambiguous/loopback IPv4 addresses: {ipv4}"
        )
    expected = {"UCX_TLS": "tcp,self", "UCX_NET_DEVICES": interface}
    for key, value in expected.items():
        if key in os.environ and os.environ[key] != value:
            raise ValueError(
                f"Managed UCX/TCP needs {key}={value}, not {os.environ[key]}"
            )
    if "UCX_CONFIG_FILE" in os.environ:
        raise ValueError("Managed UCX/TCP does not accept UCX_CONFIG_FILE")
    os.environ.update(expected)
    return {"address": address, "interface": interface, **expected}


def agent_evidence(agent, requested: list[str], binding: dict | None) -> dict:
    """Observe initialized backend handles, not available plugins or config."""
    initialized = getattr(agent, "backends", None)
    if not isinstance(initialized, dict) or set(initialized) != set(requested):
        raise ValueError(
            f"NIXL initialized backends {initialized}, requested {requested}"
        )
    if binding is not None:
        # Config and engine_config modifications defer to UCX_*; device_list
        # alone bypasses them. Check observed options for that override too.
        options = agent.get_backend_params("UCX")
        if options.get("device_list"):
            raise ValueError("Managed UCX/TCP agent has a device_list override")
        for key in ("UCX_TLS", "UCX_NET_DEVICES"):
            if os.environ.get(key) != binding[key]:
                raise ValueError(
                    f"UCX/TCP setting changed during agent initialization: {key}"
                )
    return {"backends": sorted(initialized), "binding": binding}
