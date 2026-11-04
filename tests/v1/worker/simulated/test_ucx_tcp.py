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


def test_missing_duplicate_and_alias_addresses_fail(monkeypatch):
    for mapping in (
        {},
        {"kv0": ["10.110.0.1"], "kv1": ["10.110.0.1"]},
        {"kv0": ["10.110.0.1", "10.110.0.2"]},
    ):
        monkeypatch.setattr(
            "psutil.net_if_addrs",
            lambda: {
                k: [NS(family=socket.AF_INET, address=a) for a in v]
                for k, v in mapping.items()
            },
        )
        with pytest.raises(ValueError):
            bind_ucx_tcp("10.110.0.1", {})


def test_backend_identity_and_device_override_are_observed():
    binding = bind_ucx_tcp("10.110.0.1", {})
    for backends, options in (
        ({}, {}),
        ({"Mooncake": 1}, {}),
        ({"UCX": 1, "Mooncake": 2}, {}),
        ({"UCX": 1}, {"device_list": "kv0"}),
    ):
        agent = NS(backends=backends, get_backend_params=lambda _: options)
        with pytest.raises(ValueError):
            agent_evidence(agent, ["UCX"], binding)


def test_settings_cannot_change_during_agent_construction(monkeypatch):
    binding = bind_ucx_tcp("10.110.0.1", {})
    monkeypatch.setenv("UCX_TLS", "all")
    with pytest.raises(ValueError, match="changed"):
        agent_evidence(
            NS(backends={"UCX": 1}, get_backend_params=lambda _: {}), ["UCX"], binding
        )


@pytest.mark.parametrize(
    "tp_rank,tp_size,dp_rank,dp_size", [(0, 1, 0, 1), (1, 2, 0, 1), (0, 1, 1, 2)]
)
def test_worker_uses_tp_and_local_dp_rank(
    monkeypatch, tp_rank, tp_size, dp_rank, dp_size
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
    evidence = _bind_rank_kv_address(config, ["UCX"])
    assert evidence["interface"] == f"kv{dp_rank * tp_size + tp_rank}"
    with pytest.raises(ValueError, match="Mooncake backend only"):
        _bind_rank_kv_address(config, ["Mooncake", "UCX"])
    config.kv_transfer_config.kv_connector_extra_config = {}
    with pytest.raises(ValueError, match="rank_addresses"):
        _bind_rank_kv_address(config, ["UCX"])
