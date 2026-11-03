# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank KV transfer settings shared by the native connectors under
simulated execution with real KV transfers (MooncakeConnector, and the
NixlConnector over NIXL's Mooncake backend): each rank's own KV address
(``rank_addresses``), TENT's hp_tcp bound to it, and the tensor-parallel
sizes real transfers pair."""

import ipaddress
import os

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def local_kv_rank(vllm_config: VllmConfig, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """This rank's place in ``rank_addresses``, and their count. An engine
    of its own (no DP, or a dense model's independent DP engine, which runs
    as DP=1) lists its TP ranks; an MoE model's DP engines on one host list
    theirs in turn, local DP rank by TP rank."""
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size == 1:
        return tp_rank, tp_size
    dp_local_rank = parallel_config.data_parallel_rank_local or 0
    dp_local_size = parallel_config.data_parallel_size_local
    if not 0 <= dp_local_rank < dp_local_size:
        # An engine launched on its own (offline MoE DP) keeps its host-local
        # rank among engines its configuration does not count.
        raise ValueError(
            f"rank_addresses needs this host's MoE DP engines launched "
            f"together: local DP rank {dp_local_rank} is not one of its "
            f"{dp_local_size} local engines."
        )
    return dp_local_rank * tp_size + tp_rank, dp_local_size * tp_size


def this_rank_kv_address(vllm_config: VllmConfig) -> str | None:
    """`rank_kv_address` of this worker's rank, None without
    ``rank_addresses``."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    if kv_transfer_config.kv_connector_extra_config.get("rank_addresses") is None:
        return None
    return rank_kv_address(
        kv_transfer_config,
        *local_kv_rank(
            vllm_config,
            get_tensor_model_parallel_rank(),
            get_tensor_model_parallel_world_size(),
        ),
    )


def rank_kv_address(
    kv_transfer_config: KVTransferConfig, rank: int, ranks: int
) -> str | None:
    """This rank's own address, from ``rank_addresses`` in the connector's
    extra config (one IPv4 address per rank of the instance on this host,
    as `local_kv_rank` orders them, the same for every PP stage): its
    handshakes and TENT's hp_tcp use it, so each rank can have a NIC of
    its own. None without it: the host's."""
    addresses = kv_transfer_config.kv_connector_extra_config.get("rank_addresses")
    if addresses is None:
        return None
    if not isinstance(addresses, list) or len(addresses) != ranks:
        raise ValueError(
            "rank_addresses needs one address per rank on this host (local "
            f"DP ranks x TP ranks: {ranks}), got {addresses!r}."
        )
    for address in addresses:
        # IPv4Address() also takes integers.
        try:
            if not isinstance(address, str):
                raise ValueError
            ipaddress.IPv4Address(address)
        except ValueError:
            raise ValueError(
                f"rank_addresses: {address!r} is not an IPv4 address."
            ) from None
    return addresses[rank]


def bind_hp_tcp(address: str) -> None:
    """Make TENT in this process use `address` only: its RPC server (control
    metadata) and hp_tcp's listening, advertised and sending address (its
    MC_TENT_CONF; other settings kept). TENT does not derive its RPC host
    from initialize()'s hostname, and MOONCAKE_LOCAL_HOSTNAME overrides the
    config's."""
    conf = tent_config.load() or {}
    if "MOONCAKE_LOCAL_HOSTNAME" in os.environ:
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = address
    for key, value in (
        ("rpc_server_hostname", address),
        ("transports/hp_tcp/bind_address", address),
        ("transports/hp_tcp/advertise_address", address),
        ("transports/hp_tcp/rail_addresses", [address]),
    ):
        tent_config.store(conf, key, value)
    tent_config.save(conf)


def simulated_tp_mismatch(
    vllm_config: VllmConfig,
    local_tp_size: int,
    remote_tp_size: int,
    *,
    nixl_consumer: bool = False,
) -> str | None:
    """Why a peer's tensor-parallel size cannot pair with ours, if it cannot:
    real KV transfers under simulated compute pair equal sizes, or unequal
    ones (one dividing the other) for a model of plain attention only (no
    MLA, sliding window, recurrent state or draft), whose pattern is each KV
    head's (``kv_pattern.head_pattern``) wherever the connector re-shards it.
    NIXL's decode (`nixl_consumer`, the local side) refuses, as NIXL itself
    does, a larger producer whose KV heads are replicated."""
    sim = vllm_config.simulation_config
    if not (sim.simulate_forward and sim.kv_transfer_mode == "real"):
        return None
    if local_tp_size == remote_tp_size:
        return None
    sizes = f"local TP {local_tp_size} and remote TP {remote_tp_size}"
    larger, smaller = max(local_tp_size, remote_tp_size), min(local_tp_size, remote_tp_size)
    if larger % smaller:
        return f"Simulated real KV transfers pair TP sizes one dividing the other, not {sizes}."
    model = vllm_config.model_config
    layer_types = getattr(model.hf_text_config, "layer_types", None) or ()
    if (
        model.use_mla
        or model.is_hybrid
        or model.get_sliding_window() is not None
        or any("sliding" in str(kind) for kind in layer_types)
        or vllm_config.speculative_config is not None
    ):
        return (
            "Simulated real KV transfers pair unequal TP sizes for plain attention "
            f"only (no MLA, sliding window, recurrent state or draft), not {sizes}."
        )
    if (
        nixl_consumer
        and remote_tp_size > local_tp_size
        and remote_tp_size > model.get_total_num_kv_heads()
    ):
        # NIXL asserts the same (nixl/base_worker.py, before reading).
        return (
            f"NIXL does not read from a larger producer whose KV heads are "
            f"replicated ({sizes}, {model.get_total_num_kv_heads()} KV heads)."
        )
    return None
