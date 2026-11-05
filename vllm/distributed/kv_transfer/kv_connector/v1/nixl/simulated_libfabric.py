# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Managed libfabric TCP policy, before process-wide provider initialization."""

import os
import threading
from pathlib import Path

import psutil

from .simulated_tcp import tcp_interface

# Matches --sysconfdir in docker/Dockerfile.cpu, not the host's /etc.
LIBFABRIC_CONFIG = Path("/opt/libfabric/etc/libfabric.conf")
LIBFABRIC_DIAGNOSTICS = {"FI_LOG_LEVEL", "FI_LOG_PROV", "FI_LOG_SUBSYS"}
_binding: dict | None = None
_lock = threading.Lock()


def libfabric_settings(interface: str) -> dict[str, str]:
    return {
        "FI_PROVIDER": "tcp",
        "FI_TCP_IFACE": interface,
        "FI_ADDR_FORMAT": "FI_SOCKADDR_IN",
        "FI_PREFER_SYSCONFIG": "0",
    }


def _check_policy(interface: str, require_env: bool = False) -> None:
    expected = libfabric_settings(interface)
    for key, value in os.environ.items():
        if key.startswith("NIXL_LIBFABRIC_") or (
            key.startswith("FI_") and key not in set(expected) | LIBFABRIC_DIAGNOSTICS
        ):
            raise ValueError(f"Managed LIBFABRIC/TCP does not accept {key}")
        if key in expected and value != expected[key]:
            raise ValueError(f"Managed LIBFABRIC/TCP needs {key}={expected[key]}")
    try:
        config = LIBFABRIC_CONFIG.read_bytes()
    except FileNotFoundError:
        config = b""
    except OSError as error:
        raise ValueError(f"Cannot check LIBFABRIC config {LIBFABRIC_CONFIG}") from error
    if config:
        raise ValueError(
            f"Managed LIBFABRIC config must be absent/empty: {LIBFABRIC_CONFIG}"
        )
    if require_env and any(os.environ.get(k) != v for k, v in expected.items()):
        raise ValueError("Managed LIBFABRIC/TCP settings changed or are missing")


def validate_libfabric_binding(binding: dict) -> None:
    """Check policy after construction without querying native providers."""
    if binding != _binding:
        raise ValueError("LIBFABRIC/TCP has no matching prepared process binding")
    _check_policy(binding["interface"], require_env=True)


def bind_libfabric_tcp(address: str, extra: dict) -> dict:
    """Bind once per process; a loaded native library needs prior preparation."""
    global _binding
    forbidden = {
        "device_list",
        "engine_config",
        "backend_params",
        "backend_options",
        "split_batch_size",
        "post_queue_size",
        "striping_threshold",
        "max_bw_per_dram_seg",
        "enable_pthread",
        "enableProgTh",
    }
    if forbidden.intersection(extra):
        raise ValueError("Managed LIBFABRIC/TCP does not accept backend overrides")
    threads = extra.get("num_threads", 4)
    if type(threads) is not int or threads != 4:
        raise ValueError("Managed LIBFABRIC/TCP needs num_threads=4")
    with _lock:
        interface = tcp_interface(address, "LIBFABRIC")
        stats = psutil.net_if_stats().get(interface)
        if (
            stats is None
            or not stats.isup
            or "running" not in getattr(stats, "flags", "").split(",")
        ):
            raise ValueError(
                f"LIBFABRIC/TCP interface {interface} must be UP and RUNNING"
            )
        binding = {
            "address": address,
            "interface": interface,
            **libfabric_settings(interface),
            "config_file": str(LIBFABRIC_CONFIG),
            "config_policy": "absent_or_empty",
        }
        if _binding is not None and binding != _binding:
            raise ValueError("Managed LIBFABRIC/TCP cannot change its process binding")
        if _binding is None and "/libfabric.so" in Path("/proc/self/maps").read_text():
            raise ValueError("LIBFABRIC was loaded before managed TCP binding")
        _check_policy(interface)
        expected = libfabric_settings(interface)
        os.environ.update(expected)
        _binding = binding
        return binding.copy()
