# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker integration using a stand-in agent; no native UCX is started."""

import importlib
import os
import socket
from dataclasses import dataclass
from types import SimpleNamespace

import psutil
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1 import rank_kv
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector

from .test_nixl_connector import FakeNixlWrapper
from .utils import create_vllm_config, make_kv_cache_config

pytestmark = pytest.mark.cpu_test


@pytest.fixture(
    autouse=True, params=[False, True], ids=["sdk-config", "stand-in-config"]
)
def external_agent_config(request, monkeypatch):
    if request.param or base_worker.nixl_agent_config is None:
        # Only the external SDK boundary is replaced when unavailable.
        @dataclass
        class AgentConfig:
            backends: list[str]
            capture_telemetry: bool
            num_threads: int = 0
            enable_pthread: bool = True

        monkeypatch.setattr(base_worker, "nixl_agent_config", AgentConfig)


@pytest.fixture
def restored_env():
    # The managed binders write os.environ directly.
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.usefixtures("restored_env")
@pytest.mark.parametrize("backend,threads", [("UCX", 3), ("LIBFABRIC", 4)])
def test_tcp_worker_binds_before_explicit_agent_creation(
    monkeypatch, dist_init, default_vllm_config, tmp_path, backend, threads
):
    for key in list(os.environ):
        if key.startswith(("UCX_", "NIXL_UCX_", "FI_", "NIXL_LIBFABRIC_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "psutil.net_if_addrs",
        lambda: {"kv0": [SimpleNamespace(family=socket.AF_INET, address="10.110.0.1")]},
    )
    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import simulated_libfabric

    libfabric = importlib.reload(simulated_libfabric)
    monkeypatch.setattr(libfabric, "_binding", None)
    monkeypatch.setattr(libfabric, "LIBFABRIC_CONFIG", tmp_path / "libfabric.conf")
    nicstat = next(iter(psutil.net_if_stats().values()))._replace(
        isup=True, flags="up,running"
    )
    monkeypatch.setattr("psutil.net_if_stats", lambda: {"kv0": nicstat})
    created = []

    class Agent(FakeNixlWrapper):
        def __init__(self, name, config):
            # Observe the environment at the moment the real worker constructs
            # its agent, and the actual pinned Python configuration object.
            if backend == "UCX":
                assert os.environ["UCX_TLS"] == "tcp,self"
                assert os.environ["UCX_NET_DEVICES"] == "kv0"
            else:
                assert os.environ["FI_PROVIDER"] == "tcp"
                assert os.environ["FI_TCP_IFACE"] == "kv0"
                assert os.environ["FI_ADDR_FORMAT"] == "FI_SOCKADDR_IN"
                assert os.environ["FI_PREFER_SYSCONFIG"] == "0"
            assert config.backends == [backend]
            assert config.num_threads == threads
            assert config.enable_pthread is True
            created.append(name)
            super().__init__(name)
            self.backends = {backend: object()}

        def get_backend_params(self, name):
            assert name == backend
            return {"num_threads": str(threads)}

        def get_plugin_params(self, name):
            assert name == "LIBFABRIC"
            return {"num_threads": "0", "split_batch_size": "1024"}

    monkeypatch.setattr(base_worker, "NixlWrapper", Agent)
    for _ in range(2):
        config = create_vllm_config()
        config.simulation_config.simulate_forward = True
        config.simulation_config.kv_transfer_mode = "real"
        config.kv_transfer_config.kv_connector_extra_config.update(
            backends=[backend], num_threads=threads, rank_addresses=["10.110.0.1"]
        )
        connector = NixlConnector(
            config,
            KVConnectorRole.WORKER,
            make_kv_cache_config(block_size=config.cache_config.block_size),
        )
        evidence = connector.connector_worker.simulated_transport
        assert evidence["backends"] == [backend]
        assert evidence["binding"]["interface"] == "kv0"
        assert evidence["num_threads"] == threads
        connector.shutdown()
    assert len(created) == 2
    if backend == "LIBFABRIC":
        assert evidence["thread_policy_source"] == "agent_config"
        assert evidence["native_provider_observed"] is False
        monkeypatch.setenv("FI_PROVIDER", "sockets")
        with pytest.raises(ValueError, match="FI_PROVIDER"):
            NixlConnector(
                config,
                KVConnectorRole.WORKER,
                make_kv_cache_config(block_size=config.cache_config.block_size),
            )
        assert len(created) == 2


def test_unmanaged_empty_backend_list_keeps_the_ucx_default(
    monkeypatch, dist_init, default_vllm_config
):
    config = create_vllm_config()
    assert not config.simulation_config.simulate_forward
    config.kv_transfer_config.kv_connector_extra_config["backends"] = []
    seen = []

    class Agent(FakeNixlWrapper):
        def __init__(self, name, agent_config):
            assert agent_config.backends == ["UCX"]
            seen.append(agent_config.num_threads)
            super().__init__(name)

    monkeypatch.setattr(base_worker, "NixlWrapper", Agent)
    connector = NixlConnector(
        config,
        KVConnectorRole.WORKER,
        make_kv_cache_config(block_size=config.cache_config.block_size),
    )
    connector.shutdown()
    assert seen == [4]
