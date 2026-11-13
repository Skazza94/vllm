# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank KV transfer settings shared by the native connectors under
simulated execution with real KV transfers (MooncakeConnector, and the
NixlConnector over NIXL's Mooncake backend): each rank's own KV address
(``rank_addresses``), TENT's hp_tcp bound to it, and the tensor-parallel
sizes real transfers pair."""

import ipaddress
import os

from vllm import envs
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def host_dp_ranks(parallel_config) -> tuple[int, int]:
    """This DP engine's index among, and the count of, its model's DP engines
    on this host. Engines launched together by one server say so in their
    configuration (their local rank and local size). A multi-port external-LB
    supervisor launches a server for each of the host's ranks instead, each
    configured with one local rank; it exports their place on the host
    (VLLM_DP_SUPERVISED_RANK_LOCAL, VLLM_DP_SUPERVISED_SIZE_LOCAL)."""
    rank, size = envs.VLLM_DP_SUPERVISED_RANK_LOCAL, envs.VLLM_DP_SUPERVISED_SIZE_LOCAL
    if rank is None and size is None:
        return (
            parallel_config.data_parallel_rank_local or 0,
            parallel_config.data_parallel_size_local,
        )
    if (
        rank is None
        or size is None
        or not parallel_config.data_parallel_external_lb
        or parallel_config.data_parallel_size_local != 1
        or not 0 <= rank < size <= parallel_config.data_parallel_size
    ):
        raise ValueError(
            "VLLM_DP_SUPERVISED_RANK_LOCAL and VLLM_DP_SUPERVISED_SIZE_LOCAL are "
            "set together by the multi-port external-LB supervisor, for a server "
            "of one local DP rank in external LB mode: got rank "
            f"{rank} of {size} (data_parallel_size "
            f"{parallel_config.data_parallel_size}, data_parallel_size_local "
            f"{parallel_config.data_parallel_size_local}, external LB "
            f"{parallel_config.data_parallel_external_lb})."
        )
    return rank, size


def local_kv_rank(vllm_config: VllmConfig, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """This rank's place in ``rank_addresses``, and their count. An engine
    of its own (no DP, or a dense model's independent DP engine, which runs
    as DP=1) lists its TP ranks; an MoE model's DP engines on one host list
    theirs in turn, host-local DP rank by TP rank (`host_dp_ranks`)."""
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size == 1:
        return tp_rank, tp_size
    dp_local_rank, dp_local_size = host_dp_ranks(parallel_config)
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


def plain_mla(model) -> bool:
    """MLA with one latent cache a layer, as DeepSeek-V3 and Kimi-K2 keep it:
    not DeepSeek-V3.2's or GLM-5's sparse attention (an indexer's cache
    beside the latent, which DeepSeek's code selects by the config having
    `index_topk`, set or not), not DeepSeek-V4's compressed caches, not a
    hybrid's (Kimi-K3's) recurrent states."""
    configs = (model.hf_config, model.hf_text_config)
    return (
        model.use_mla
        and not model.is_hybrid
        and not any(hasattr(config, "index_topk") for config in configs)
        and not any(
            getattr(config, "model_type", None) == "deepseek_v4" for config in configs
        )
    )


def sparse_mla(model) -> bool:
    """DeepSeek-V3.2's or GLM-5's sparse MLA (DSA): a latent and, on the
    layers that have one, an indexer's key cache, each one cache held whole
    by every rank, as plain MLA's latent is."""
    from vllm.config.simulation import DSA_MODEL_TYPES

    return (
        model.use_mla
        and not model.is_hybrid
        and getattr(model.hf_config, "model_type", None) in DSA_MODEL_TYPES
        and getattr(model.hf_text_config, "index_topk", None) is not None
    )


def dcp_scope_problems(vllm_config: VllmConfig, *, transfers: bool) -> list[str]:
    """What of an instance's configuration falls outside the scope of decode
    context parallelism under simulated compute, either endpoint's when its
    KV moves (`transfers`): dense MLA, or sparse MLA with AISimulate in one
    instance. Both use tensor parallelism alone (no pipeline, prefill context
    or data parallelism, no expert parallelism) and no speculative decoding.
    Prefix caching is admitted with transfers too (its byte checks need the
    same setting on both endpoints: their pages' patterns are content-seeded
    exactly when it is on)."""
    parallel = vllm_config.parallel_config
    problems = []
    model = vllm_config.model_config
    if not plain_mla(model):
        if not transfers and sparse_mla(model):
            if vllm_config.simulation_config.timing != "aisimulate":
                problems.append(
                    "a model other than dense MLA without AISimulate timing "
                    "(sparse MLA DCP requires timing aisimulate)"
                )
        elif transfers:
            problems.append(
                "a model other than dense MLA (sparse or compressed MLA, a "
                "hybrid, or attention without MLA)"
            )
        else:
            problems.append(
                "a model other than dense MLA or DeepSeek-V3.2-style sparse MLA"
            )
    for name, size in (
        ("pipeline parallelism", parallel.pipeline_parallel_size),
        ("prefill context parallelism", parallel.prefill_context_parallel_size),
        ("data parallelism", parallel.data_parallel_size),
    ):
        if size > 1:
            problems.append(f"{name} {size}")
    if parallel.enable_expert_parallel:
        problems.append("expert parallelism")
    if vllm_config.speculative_config is not None:
        problems.append("speculative decoding")
    return problems


def dcp_pair_refusal(local_dcp_size: int, remote_dcp_size: int, problems) -> str:
    return (
        "Simulated real KV transfers pair decode context parallelism (local DCP "
        f"{local_dcp_size}, remote DCP {remote_dcp_size}) for a dense MLA model "
        "over tensor parallelism alone on both sides, TP sizes one dividing the "
        "other, without speculative decoding, not with "
        f"{', '.join(problems)}."
    )


def simulated_dcp_scope(vllm_config: VllmConfig) -> list[str] | None:
    """An instance's `dcp_scope_problems` with real KV transfers under
    simulated compute, which NIXL advertises for its peer to check whichever
    side has decode context parallelism; None otherwise."""
    sim = vllm_config.simulation_config
    if not (sim.simulate_forward and sim.kv_transfer_mode == "real"):
        return None
    return dcp_scope_problems(vllm_config, transfers=True)


def simulated_tp_mismatch(
    vllm_config: VllmConfig,
    local_tp_size: int,
    remote_tp_size: int,
    *,
    nixl_consumer: bool = False,
    remote_dcp_size: int = 1,
) -> str | None:
    """Why a peer's tensor-parallel size cannot pair with ours, if it cannot:
    real KV transfers under simulated compute pair equal sizes, or unequal
    ones (one dividing the other) for a model of plain attention (sliding
    window layers beside full ones included; no recurrent state or draft),
    or, over NIXL only, a hybrid of Mamba2 or GDN states or Kimi-K3 at
    decode TP4 from prefill TP8 (`state_layout.pairing_problem`), whose
    pattern is each KV head's
    (``kv_pattern.head_pattern``) wherever the connector re-shards it, or of
    plain MLA (`plain_mla`: one latent, the same on every rank) or sparse MLA
    (`sparse_mla`: an indexer's keys beside it, the same on every rank too). NIXL's
    decode (`nixl_consumer`, the local side) refuses, as NIXL itself does, a
    larger producer whose KV heads are replicated; MLA's latent is exempt.
    Decode context parallelism on either side (`remote_dcp_size` the peer's,
    from NIXL's handshake) pairs TP sizes one dividing the other (NIXL maps
    each decode rank's DCP slice to the producer ranks holding it,
    `TransferTopology.dcp_source_ranks`), this instance in
    `dcp_scope_problems`'s scope whatever its own DCP size (the peer's
    scope is checked from its NIXL metadata, `simulated_scope`)."""
    sim = vllm_config.simulation_config
    if not (sim.simulate_forward and sim.kv_transfer_mode == "real"):
        return None
    local_dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    if local_dcp_size > 1 or remote_dcp_size > 1:
        problems = dcp_scope_problems(vllm_config, transfers=True)
        larger, smaller = max(local_tp_size, remote_tp_size), min(
            local_tp_size, remote_tp_size
        )
        if larger % smaller:
            problems.append(f"local TP {local_tp_size} and remote TP {remote_tp_size}")
        if problems:
            return dcp_pair_refusal(local_dcp_size, remote_dcp_size, problems)
        return None
    if local_tp_size == remote_tp_size:
        return None
    sizes = f"local TP {local_tp_size} and remote TP {remote_tp_size}"
    larger, smaller = max(local_tp_size, remote_tp_size), min(local_tp_size, remote_tp_size)
    if larger % smaller:
        return f"Simulated real KV transfers pair TP sizes one dividing the other, not {sizes}."
    model = vllm_config.model_config
    # Layers of different KV heads (Gemma4's global layers beside its
    # sliding ones): the connectors count one head total, so only sizes that
    # replicate no layer and split every layer evenly on both sides are
    # admitted (each total a positive integer, never a bool, at least the
    # larger size and divisible by both).
    per_layer = getattr(model.hf_text_config, "per_layer_config", None) or ()
    totals = [getattr(layer, "num_key_value_heads", None) for layer in per_layer]
    if len(set(totals)) > 1:
        if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in totals):
            return (
                "Simulated real KV transfers pair unequal TP sizes for layers of "
                f"valid KV head counts only (this model's layers differ: {sorted(map(str, set(totals)))}), "
                f"not {sizes}."
            )
        if any(n < larger or n % local_tp_size or n % remote_tp_size for n in totals):
            return (
                "Simulated real KV transfers pair unequal TP sizes for layers of "
                "differing KV heads only where none replicates and each splits "
                f"evenly on both sides (this model's layers differ in KV heads: "
                f"{sorted(set(totals))}), not {sizes}."
            )
    if model.is_hybrid and nixl_consumer:
        from vllm.v1.worker.simulated.state_layout import pairing_problem

        problem = pairing_problem(vllm_config, local_tp_size, remote_tp_size)
        if problem is not None:
            return (
                "Simulated real KV transfers pair unequal TP sizes for a hybrid model "
                "over NIXL only for Mamba2 or GDN states that split on both sides, or "
                f"Kimi-K3 from prefill TP8 to decode TP4, not {sizes}: {problem}."
            )
        return None
    if (
        (model.use_mla and not (plain_mla(model) or sparse_mla(model)))
        or model.is_hybrid
        or vllm_config.speculative_config is not None
    ):
        return (
            "Simulated real KV transfers pair unequal TP sizes for plain attention, "
            "plain or sparse MLA only (no compressed MLA, recurrent state "
            f"or draft), not {sizes}."
        )
    if (
        nixl_consumer
        and not model.use_mla
        and remote_tp_size > local_tp_size
        and remote_tp_size > model.get_total_num_kv_heads()
    ):
        # NIXL asserts the same (nixl/base_worker.py, before reading).
        return (
            f"NIXL does not read from a larger producer whose KV heads are "
            f"replicated ({sizes}, {model.get_total_num_kv_heads()} KV heads)."
        )
    return None
