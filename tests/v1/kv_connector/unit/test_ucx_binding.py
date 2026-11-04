# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker integration using a stand-in agent; no native UCX is started."""

import os
import socket
from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1 import rank_kv
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector

from .test_nixl_connector import FakeNixlWrapper
from .utils import create_vllm_config, make_kv_cache_config


def test_ucx_worker_binds_before_explicit_agent_creation(
    monkeypatch, dist_init, default_vllm_config
):
    for key in list(os.environ):
        if key.startswith(("UCX_", "NIXL_UCX_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(
        "psutil.net_if_addrs",
        lambda: {"kv0": [SimpleNamespace(family=socket.AF_INET, address="10.110.0.1")]},
    )
    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_world_size", lambda: 1)
    created = []

    class Agent(FakeNixlWrapper):
        def __init__(self, name, config):
            # Observe the environment at the moment the real worker constructs
            # its agent, and the actual pinned Python configuration object.
            assert os.environ["UCX_TLS"] == "tcp,self"
            assert os.environ["UCX_NET_DEVICES"] == "kv0"
            assert config.backends == ["UCX"]
            assert config.num_threads == 3
            assert config.enable_pthread is True
            created.append(name)
            super().__init__(name)
            self.backends = {"UCX": object()}

        def get_backend_params(self, name):
            assert name == "UCX"
            return {"num_threads": "3"}

    monkeypatch.setattr(base_worker, "NixlWrapper", Agent)
    for _ in range(2):
        config = create_vllm_config()
        config.simulation_config.simulate_forward = True
        config.simulation_config.kv_transfer_mode = "real"
        config.kv_transfer_config.kv_connector_extra_config.update(
            backends=["UCX"], num_threads=3, rank_addresses=["10.110.0.1"]
        )
        connector = NixlConnector(
            config,
            KVConnectorRole.WORKER,
            make_kv_cache_config(block_size=config.cache_config.block_size),
        )
        evidence = connector.connector_worker.simulated_transport
        assert evidence["backends"] == ["UCX"]
        assert evidence["binding"]["interface"] == "kv0"
        assert evidence["num_threads"] == 3
        connector.shutdown()
    assert len(created) == 2
