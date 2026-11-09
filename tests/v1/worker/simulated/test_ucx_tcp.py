# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Managed TCP config: no UCX context or transport is started here."""

import os
import socket
from types import SimpleNamespace as NS

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.simulated_tcp import (
    agent_evidence,
    bind_ucx_tcp,
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    # The binders write os.environ directly; restore it after each test.
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(("UCX_", "NIXL_UCX_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "psutil.net_if_addrs",
        lambda: {
            "control": [NS(family=socket.AF_INET, address="10.250.0.1")],
            "kv0": [NS(family=socket.AF_INET, address="10.110.0.1")],
            "kv1": [NS(family=socket.AF_INET, address="10.110.0.3")],
        },
    )
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_tcp_interface_and_repeat_initialization():
    a = bind_ucx_tcp("10.110.0.1", {})
    assert a == bind_ucx_tcp("10.110.0.1", {})
    assert a == {
        "address": "10.110.0.1",
        "interface": "kv0",
        "UCX_TLS": "tcp,self",
        "UCX_NET_DEVICES": "kv0",
    }
    agent = NS(
        backends={"UCX": object()},
        get_backend_params=lambda _: {"engine_config": "TLS=all,NET_DEVICES=all"},
    )
    # Pinned config.modify defers to the present UCX_* environment variables.
    assert agent_evidence(agent, ["UCX"], a)["binding"] == a
    with pytest.raises(ValueError, match="UCX_NET_DEVICES"):
        bind_ucx_tcp("10.110.0.3", {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("UCX_TLS", "all"),
        ("UCX_NET_DEVICES", "control"),
        ("UCX_CONFIG_FILE", "/custom"),
        ("NIXL_UCX_DEVICE_LIST", "kv0"),
    ],
)
def test_environment_override_refused(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        bind_ucx_tcp("10.110.0.1", {})


@pytest.mark.parametrize(
    "key", ["device_list", "engine_config", "backend_params", "backend_options"]
)
def test_plugin_override_refused(key):
    with pytest.raises(ValueError, match="overrides"):
        bind_ucx_tcp("10.110.0.1", {key: "kv0"})


@pytest.mark.parametrize("backend", ["UCX", "LIBFABRIC"])
def test_missing_duplicate_and_alias_addresses_fail(monkeypatch, libfabric, backend):
    bind = bind_ucx_tcp if backend == "UCX" else libfabric.bind_libfabric_tcp
    for mapping in (
        {},
        {"kv0": ["10.110.0.1"], "kv1": ["10.110.0.1"]},
        {"kv0": ["10.110.0.1", "10.110.0.2"]},
    ):
        monkeypatch.setattr(
            "psutil.net_if_addrs",
            lambda mapping=mapping: {
                k: [NS(family=socket.AF_INET, address=a) for a in v]
                for k, v in mapping.items()
            },
        )
        with pytest.raises(ValueError):
            bind("10.110.0.1", {})


def test_backend_identity_and_device_override_are_observed():
    binding = bind_ucx_tcp("10.110.0.1", {})
    for backends, options in (
        ({}, {}),
        ({"Mooncake": 1}, {}),
        ({"UCX": 1, "Mooncake": 2}, {}),
        ({"UCX": 1}, {"device_list": "kv0"}),
    ):
        agent = NS(
            backends=backends, get_backend_params=lambda _, options=options: options
        )
        with pytest.raises(ValueError):
            agent_evidence(agent, ["UCX"], binding)


def test_ucx_evidence_does_not_claim_native_observation():
    binding = bind_ucx_tcp("10.110.0.1", {})
    evidence = agent_evidence(
        NS(backends={"UCX": 1}, get_backend_params=lambda _: {}), ["UCX"], binding
    )
    assert evidence["binding"] == binding
    assert evidence["native_transport_observed"] is False


@pytest.mark.parametrize("backend", ["UCX", "LIBFABRIC"])
@pytest.mark.parametrize(
    "tp_rank,tp_size,dp_rank,dp_size", [(0, 1, 0, 1), (1, 2, 0, 1), (0, 1, 1, 2)]
)
def test_worker_uses_tp_and_local_dp_rank(
    monkeypatch, tp_rank, tp_size, dp_rank, dp_size, libfabric, backend
):
    from vllm.distributed.kv_transfer.kv_connector.v1 import rank_kv
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        _bind_rank_kv_address,
    )

    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_rank", lambda: tp_rank)
    monkeypatch.setattr(
        rank_kv, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    config = NS(
        simulation_config=NS(simulate_forward=True, kv_transfer_mode="real"),
        parallel_config=NS(
            data_parallel_size=dp_size,
            data_parallel_rank_local=dp_rank,
            data_parallel_size_local=dp_size,
        ),
        kv_transfer_config=NS(
            kv_connector_extra_config={
                "rank_addresses": ["10.110.0.1", "10.110.0.3"][: tp_size * dp_size]
            }
        ),
    )
    evidence = _bind_rank_kv_address(config, [backend])
    assert evidence["interface"] == f"kv{dp_rank * tp_size + tp_rank}"
    with pytest.raises(ValueError, match="Mooncake backend only"):
        _bind_rank_kv_address(config, ["Mooncake", "UCX"])
    config.kv_transfer_config.kv_connector_extra_config = {}
    with pytest.raises(ValueError, match="rank_addresses"):
        _bind_rank_kv_address(config, [backend])


@pytest.fixture
def libfabric(monkeypatch, tmp_path):
    import importlib

    import psutil

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import simulated_libfabric

    module = importlib.reload(simulated_libfabric)
    # Undone at teardown: no process binding outlives the test.
    monkeypatch.setattr(module, "_binding", None)
    for key in list(os.environ):
        if key.startswith(("FI_", "NIXL_LIBFABRIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(module, "LIBFABRIC_CONFIG", tmp_path / "libfabric.conf")
    nicstat = next(iter(psutil.net_if_stats().values()))._replace(
        isup=True, flags="up,running"
    )
    monkeypatch.setattr(
        "psutil.net_if_stats",
        lambda: {name: nicstat for name in ("kv0", "kv1", "control")},
    )
    return module


def test_libfabric_policy_reuse_and_process_binding(libfabric, monkeypatch):
    binding = libfabric.bind_libfabric_tcp("10.110.0.1", {})
    assert binding == libfabric.bind_libfabric_tcp("10.110.0.1", {})
    assert binding["FI_PROVIDER"] == "tcp"
    assert binding["FI_TCP_IFACE"] == "kv0"
    assert binding["FI_ADDR_FORMAT"] == "FI_SOCKADDR_IN"
    assert binding["FI_PREFER_SYSCONFIG"] == "0"
    assert binding["config_policy"] == "absent_or_empty"
    agent = NS(
        backends={"LIBFABRIC": object()},
        get_backend_params=lambda _: {"num_threads": "4"},
        get_plugin_params=lambda _: {"num_threads": "0", "split_batch_size": "1024"},
    )
    evidence = agent_evidence(agent, ["LIBFABRIC"], binding)
    assert evidence["backend_options"] == {"num_threads": "4"}
    assert evidence["plugin_defaults"] == {
        "num_threads": "0",
        "split_batch_size": "1024",
    }
    assert evidence["thread_policy_source"] == "agent_config"
    assert evidence["native_provider_observed"] is False
    assert evidence["native_threads_observed"] is False
    # Even clearing the environment cannot rebind process-wide native state.
    for key in list(os.environ):
        if key.startswith("FI_"):
            monkeypatch.delenv(key)
    with pytest.raises(ValueError, match="process binding"):
        libfabric.bind_libfabric_tcp("10.110.0.3", {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("FI_PROVIDER", "sockets"),
        ("FI_TCP_IFACE", "control"),
        ("FI_ADDR_FORMAT", "FI_SOCKADDR_IN6"),
        ("FI_PREFER_SYSCONFIG", "1"),
        ("FI_PROVIDER_PATH", ""),
        ("FI_TCP_PROV_NAME", "tcp"),
        ("FI_TCP_TX_SIZE", "100"),
        ("FI_HOOK", "debug"),
        ("NIXL_LIBFABRIC_MAX_BW_PER_DRAM_SEG", "100"),
    ],
)
def test_libfabric_conflicting_settings_fail_before_binding(
    libfabric, monkeypatch, key, value
):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        libfabric.bind_libfabric_tcp("10.110.0.1", {})
    assert "FI_ADDR_FORMAT" not in os.environ or key == "FI_ADDR_FORMAT"


@pytest.mark.parametrize(
    "override",
    [
        {"device_list": "kv0"},
        {"engine_config": "test"},
        {"backend_params": {}},
        {"backend_options": {}},
        {"split_batch_size": 2048},
        {"post_queue_size": 65536},
        {"striping_threshold": 1},
        {"enable_pthread": False},
        {"num_threads": 0},
        {"num_threads": 8},
        {"num_threads": True},
    ],
)
def test_libfabric_plugin_overrides_fail(libfabric, override):
    with pytest.raises(ValueError):
        libfabric.bind_libfabric_tcp("10.110.0.1", override)


@pytest.mark.parametrize(
    "isup,flags", [(False, "up,running"), (True, "up"), (True, "")]
)
def test_libfabric_requires_up_and_running(libfabric, monkeypatch, isup, flags):
    import psutil

    nicstat = next(iter(psutil.net_if_stats().values()))._replace(
        isup=isup, flags=flags
    )
    monkeypatch.setattr("psutil.net_if_stats", lambda: {"kv0": nicstat})
    with pytest.raises(ValueError, match="UP and RUNNING"):
        libfabric.bind_libfabric_tcp("10.110.0.1", {})


def test_libfabric_checks_compiled_config_before_and_after_agent(libfabric):
    libfabric.LIBFABRIC_CONFIG.write_text("FI_TCP_TX_SIZE=1\n")
    with pytest.raises(ValueError, match="absent/empty"):
        libfabric.bind_libfabric_tcp("10.110.0.1", {})
    assert "FI_PROVIDER" not in os.environ
    libfabric.LIBFABRIC_CONFIG.write_text("")
    binding = libfabric.bind_libfabric_tcp("10.110.0.1", {})
    libfabric.LIBFABRIC_CONFIG.write_text("FI_PREFER_SYSCONFIG=1\n")
    with pytest.raises(ValueError, match="absent/empty"):
        libfabric.validate_libfabric_binding(binding)


def test_libfabric_rejects_config_read_errors(libfabric):
    libfabric.LIBFABRIC_CONFIG.mkdir()
    with pytest.raises(ValueError, match="Cannot check"):
        libfabric.bind_libfabric_tcp("10.110.0.1", {})


def test_libfabric_loaded_library_requires_prior_preparation(libfabric, monkeypatch):
    from pathlib import Path

    read_text = Path.read_text

    def maps(path, *args, **kwargs):
        return (
            "/opt/libfabric/lib/libfabric.so.1"
            if str(path) == "/proc/self/maps"
            else read_text(path, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "read_text", maps)
    with pytest.raises(ValueError, match="loaded before"):
        libfabric.bind_libfabric_tcp("10.110.0.1", {})
    assert "FI_PROVIDER" not in os.environ

    monkeypatch.setattr(Path, "read_text", read_text)
    binding = libfabric.bind_libfabric_tcp("10.110.0.1", {})
    monkeypatch.setattr(Path, "read_text", maps)
    assert libfabric.bind_libfabric_tcp("10.110.0.1", {}) == binding


def test_libfabric_diagnostics_and_post_construction_tampering(libfabric, monkeypatch):
    for key in libfabric.LIBFABRIC_DIAGNOSTICS:
        monkeypatch.setenv(key, "tcp" if key == "FI_LOG_PROV" else "info")
    binding = libfabric.bind_libfabric_tcp("10.110.0.1", {})
    libfabric.validate_libfabric_binding(binding)
    monkeypatch.setenv("FI_TCP_IFACE", "control")
    with pytest.raises(ValueError):
        libfabric.validate_libfabric_binding(binding)


@pytest.mark.parametrize(
    "options",
    [
        {"num_threads": "0", "split_batch_size": "1024"},
        {"num_threads": "4", "split_batch_size": "2048"},
        {},
    ],
)
def test_libfabric_observes_backend_options(libfabric, options):
    binding = libfabric.bind_libfabric_tcp("10.110.0.1", {})
    with pytest.raises(ValueError, match="options"):
        agent_evidence(
            NS(
                backends={"LIBFABRIC": object()},
                get_backend_params=lambda _, options=options: options,
            ),
            ["LIBFABRIC"],
            binding,
        )


def test_a_stalled_remote_agent_is_reported_and_a_prompt_one_is_not(monkeypatch):
    import threading

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import simulated_tcp

    errors = []
    monkeypatch.setattr(simulated_tcp.logger, "error", lambda *a: errors.append(a))
    release = threading.Event()
    agent = NS(add_remote_agent=lambda meta: release.wait(5) and f"agent-{meta}")
    simulated_tcp.report_stalled_remote_agents(agent, "LIBFABRIC", after_s=0.05)
    worker = threading.Thread(target=lambda: errors.append(agent.add_remote_agent("p")))
    worker.start()
    deadline = threading.Event()
    for _ in range(100):
        if errors:
            break
        deadline.wait(0.02)
    assert errors and errors[0][1] == "LIBFABRIC"
    release.set()
    worker.join(5)
    assert errors[-1] == "agent-p"
    # A call that returns before the deadline reports nothing.
    errors.clear()
    prompt = NS(add_remote_agent=lambda m: m)
    simulated_tcp.report_stalled_remote_agents(prompt, "UCX", after_s=0.05)
    assert prompt.add_remote_agent("x") == "x"
    deadline.wait(0.15)
    assert errors == []
