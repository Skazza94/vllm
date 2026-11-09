# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Managed CPU TCP binding, before NIXL initializes its selected backend."""

import os
import socket
import threading

import psutil

from vllm.logger import init_logger

logger = init_logger(__name__)

# How long a managed TCP backend's add_remote_agent may run before it is
# reported as stalled (a healthy one returns in well under a second).
REMOTE_AGENT_STALL_S = 30.0


def tcp_interface(address: str, backend: str) -> str:
    """Resolve one non-loopback IPv4 address without accepting aliases."""
    interfaces = psutil.net_if_addrs()
    matches = [
        name
        for name, addrs in interfaces.items()
        if any(a.family == socket.AF_INET and a.address == address for a in addrs)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{backend}/TCP address {address} needs one unique local interface: "
            f"{matches}"
        )
    interface = matches[0]
    ipv4 = [a.address for a in interfaces[interface] if a.family == socket.AF_INET]
    if ipv4 != [address] or address.startswith("127."):
        raise ValueError(
            f"{backend}/TCP interface {interface} has ambiguous/loopback "
            f"IPv4 addresses: {ipv4}"
        )
    return interface


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
    interface = tcp_interface(address, "UCX")
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
        if requested == ["LIBFABRIC"]:
            from .simulated_libfabric import validate_libfabric_binding

            validate_libfabric_binding(binding)
            options = agent.get_backend_params("LIBFABRIC")
            if options != {"num_threads": "4"}:
                raise ValueError(f"Unexpected managed LIBFABRIC options: {options}")
            defaults = agent.get_plugin_params("LIBFABRIC")
            if defaults != {"num_threads": "0", "split_batch_size": "1024"}:
                raise ValueError(f"Unexpected LIBFABRIC plugin defaults: {defaults}")
            return {
                "backends": sorted(initialized),
                "binding": binding,
                "backend_options": options,
                "plugin_defaults": defaults,
                "thread_policy_source": "agent_config",
                "native_provider_observed": False,
                "native_threads_observed": False,
            }
        # Config and engine_config modifications defer to UCX_*; device_list
        # alone bypasses them. Check observed options for that override too.
        options = agent.get_backend_params("UCX")
        if options.get("device_list"):
            raise ValueError("Managed UCX/TCP agent has a device_list override")
        # The binding is the environment set before construction; UCX's own
        # transport and device selection is not observed from Python.
        return {
            "backends": sorted(initialized),
            "binding": binding,
            "native_transport_observed": False,
        }
    return {"backends": sorted(initialized), "binding": binding}


def report_stalled_remote_agents(
    agent, backend: str, after_s: float = REMOTE_AGENT_STALL_S
) -> None:
    """Log an error when the agent's add_remote_agent has not returned after
    `after_s`. Over managed TCP it can block indefinitely when the peer's KV
    address is unreachable (its side channel answering); it cannot be
    interrupted, and it holds NIXL's single handshake thread, so later
    handshakes and the process's exit wait behind it. The report makes such a
    hang visible; it does not end it."""
    add = agent.add_remote_agent

    def add_remote_agent(*args, **kwargs):
        timer = threading.Timer(
            after_s,
            logger.error,
            args=(
                "NIXL %s add_remote_agent has not returned after %.0f s: the "
                "peer's KV address (rank_addresses) may be unreachable. This "
                "handshake, those queued behind it and the process's exit "
                "wait until it returns or the process is killed.",
                backend,
                after_s,
            ),
        )
        timer.daemon = True
        timer.start()
        try:
            return add(*args, **kwargs)
        finally:
            timer.cancel()

    agent.add_remote_agent = add_remote_agent
