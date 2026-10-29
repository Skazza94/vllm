# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step timing for simulated execution: a `TimingProvider` prices a
`StepDescriptor` as a `StepCost`."""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import itertools
import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Recorded steps a k-NN lookup interpolates.
_DEFAULT_KNN_K = 4
# Relative error below which a smaller host-time fit wins a tie.
_FIT_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SimRequestStep:
    """One request's share of a simulated step."""

    req_id: str
    num_scheduled_tokens: int
    """Tokens computed for this request in this step: a prefill chunk, or
    ``1 + num_spec_tokens`` for decode."""
    num_computed_tokens: int
    """KV length before this step (prompt tokens already computed + outputs)."""
    num_prompt_tokens: int
    num_cached_tokens: int
    """Prefix-cache hits credited at admission (0 for already-running reqs)."""
    num_spec_tokens: int
    is_prefill: bool
    dp_rank: int = 0
    """The data-parallel rank serving it: in an MoE-DP joint step, which
    rank's batch it belongs to."""


@dataclass(frozen=True)
class ParallelSummary:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


def parallel_summary(vllm_config: VllmConfig) -> ParallelSummary:
    pc = vllm_config.parallel_config
    return ParallelSummary(
        tensor_parallel_size=pc.tensor_parallel_size,
        pipeline_parallel_size=pc.pipeline_parallel_size,
        data_parallel_size=pc.data_parallel_size,
    )


@dataclass(frozen=True)
class StepDescriptor:
    """Everything a timing model may need to price one scheduler step."""

    step_index: int
    requests: tuple[SimRequestStep, ...]
    num_prefill_tokens: int
    num_decode_tokens: int
    num_prefill_reqs: int
    num_decode_reqs: int
    num_common_prefix_blocks: int
    num_kv_tokens: int
    """Sum over scheduled requests of KV length after this step."""
    parallel: ParallelSummary = field(default_factory=ParallelSummary)

    @property
    def num_reqs(self) -> int:
        return len(self.requests)

    @property
    def num_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_decode_only(self) -> bool:
        return self.num_prefill_reqs == 0

    def replay_key(self) -> str:
        """Batch composition key shared with recorded step logs."""
        pairs = sorted(
            (r.num_scheduled_tokens, r.num_computed_tokens) for r in self.requests
        )
        return json.dumps(pairs, separators=(",", ":"))

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["requests"] = [asdict(r) for r in self.requests]
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> StepDescriptor:
        return cls(
            **d
            | {
                "requests": tuple(SimRequestStep(**r) for r in d["requests"]),
                "parallel": ParallelSummary(**d.get("parallel", {})),
            }
        )


def accepted_in_step_order(
    step: StepDescriptor, accepted: Mapping[str, int]
) -> tuple[int, ...] | None:
    """Accepted drafts of the step's requests, in its order; None for a joint
    MoE-DP step, whose other ranks' requests are not in `accepted`."""
    if not all(r.req_id in accepted for r in step.requests):
        return None
    return tuple(accepted[r.req_id] for r in step.requests)


def build_step_descriptor(
    scheduler_output: SchedulerOutput,
    prompt_lens: dict[str, int],
    step_index: int,
    parallel: ParallelSummary,
) -> StepDescriptor:
    """Describe one scheduled step from the `SchedulerOutput`, as the simulated
    runner and a recording worker both do. `prompt_lens` maps request
    ids to prompt lengths across steps."""
    for req_id in scheduler_output.finished_req_ids:
        prompt_lens.pop(req_id, None)
    num_scheduled = scheduler_output.num_scheduled_tokens
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    computed: dict[str, int] = {}
    cached_at_admission: dict[str, int] = {}
    for req in scheduler_output.scheduled_new_reqs:
        prompt_lens[req.req_id] = req.prompt_len
        computed[req.req_id] = req.num_computed_tokens
        cached_at_admission[req.req_id] = req.num_computed_tokens
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens):
        computed[req_id] = num_computed

    requests: list[SimRequestStep] = []
    num_prefill_tokens = num_decode_tokens = 0
    num_prefill_reqs = num_decode_reqs = 0
    num_kv_tokens = 0
    for req_id in sorted(num_scheduled):
        scheduled = num_scheduled[req_id]
        num_computed = computed.get(req_id, 0)
        prompt_len = prompt_lens.get(req_id, 0)
        is_prefill = num_computed < prompt_len
        requests.append(
            SimRequestStep(
                req_id=req_id,
                num_scheduled_tokens=scheduled,
                num_computed_tokens=num_computed,
                num_prompt_tokens=prompt_len,
                num_cached_tokens=cached_at_admission.get(req_id, 0),
                num_spec_tokens=len(spec_tokens.get(req_id, ())),
                is_prefill=is_prefill,
            )
        )
        if is_prefill:
            num_prefill_tokens += scheduled
            num_prefill_reqs += 1
        else:
            num_decode_tokens += scheduled
            num_decode_reqs += 1
        num_kv_tokens += num_computed + scheduled
    common = scheduler_output.num_common_prefix_blocks
    return StepDescriptor(
        step_index=step_index,
        requests=tuple(requests),
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_reqs=num_prefill_reqs,
        num_decode_reqs=num_decode_reqs,
        num_common_prefix_blocks=max(common) if common else 0,
        num_kv_tokens=num_kv_tokens,
        parallel=parallel,
    )


def empty_step_descriptor(step_index: int, parallel: ParallelSummary) -> StepDescriptor:
    """A rank's share of a lockstep step in which it has no requests (an idle
    data-parallel rank's dummy batch)."""
    return StepDescriptor(step_index, (), 0, 0, 0, 0, 0, 0, parallel)


def merge_dp_descriptors(descriptors: list[StepDescriptor]) -> StepDescriptor:
    """All MoE data-parallel ranks' shares of one lockstep step, as one step.
    Their forward passes meet in all-to-all collectives, so the step takes as
    long as the whole joint batch dictates on every rank."""
    own = descriptors[0]
    return StepDescriptor(
        step_index=own.step_index,
        requests=tuple(
            sorted((r for d in descriptors for r in d.requests), key=lambda r: r.req_id)
        ),
        num_prefill_tokens=sum(d.num_prefill_tokens for d in descriptors),
        num_decode_tokens=sum(d.num_decode_tokens for d in descriptors),
        num_prefill_reqs=sum(d.num_prefill_reqs for d in descriptors),
        num_decode_reqs=sum(d.num_decode_reqs for d in descriptors),
        num_common_prefix_blocks=max(d.num_common_prefix_blocks for d in descriptors),
        num_kv_tokens=sum(d.num_kv_tokens for d in descriptors),
        parallel=own.parallel,
    )


def gather_dp_step(descriptor: StepDescriptor) -> StepDescriptor:
    """The joint step of all MoE data-parallel ranks. Must be called exactly
    where the real runner synchronizes the DP ranks: every step that runs the
    model, and every dummy batch of an idle rank."""
    import torch.distributed as dist

    from vllm.distributed.parallel_state import get_dp_group

    group = get_dp_group()
    descriptor = dataclasses.replace(
        descriptor,
        requests=tuple(
            dataclasses.replace(r, dp_rank=group.rank_in_group)
            for r in descriptor.requests
        ),
    )
    gathered: list[StepDescriptor | None] = [None] * group.world_size
    dist.all_gather_object(gathered, descriptor, group=group.cpu_group)
    others = [d for i, d in enumerate(gathered) if i != group.rank_in_group]
    return merge_dp_descriptors([descriptor, *others])  # type: ignore[list-item]


@dataclass(frozen=True)
class StepCost:
    """Predicted cost of one step: device time, and the host time the step's
    `execute_model`/`sample_tokens` calls take."""

    device_s: float
    host_s: float = 0.0
    num_accepted: tuple[int, ...] | None = None
    """Per-request accepted draft tokens (speculative decoding), or None."""
    breakdown: dict[str, float] | None = None
    """Optional named components (e.g. per kernel class), for logging."""


@runtime_checkable
class TimingProvider(Protocol):
    """Prices one simulated step. Implementations are constructed with the
    engine configuration plus the user's `timing_args` as keyword arguments."""

    def estimate(self, step: StepDescriptor) -> StepCost: ...


class ZeroTimingProvider:
    """Every step takes no device time (pure scheduler / KV-cache simulation)."""

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any) -> None:
        if kwargs:
            raise ValueError(f"zero timing provider takes no arguments, got {kwargs}")

    def estimate(self, step: StepDescriptor) -> StepCost:
        return StepCost(device_s=0.0)


@dataclass(frozen=True)
class HostTimeModel:
    """A step's host time: ``fixed_us + per_req_us * R + per_prefill_token_us *
    P`` for R scheduled requests and P prefill tokens."""

    fixed_us: float = 0.0
    per_req_us: float = 0.0
    per_prefill_token_us: float = 0.0

    @classmethod
    def from_log(cls, path: str) -> HostTimeModel:
        """Non-negative least-squares fit to a recorded step log; terms the log
        cannot determine stay at zero."""
        rows, host_us = [], []
        for rec in load_step_log(path):
            if "step" not in rec:
                raise ValueError(f"{path}: host time needs a recorded step log")
            step = StepDescriptor.from_json_dict(rec["step"])
            rows.append((1.0, step.num_reqs, step.num_prefill_tokens))
            host_us.append(_record_cost(rec)[1] * 1e6)
        if not rows:
            raise ValueError(f"{path} has no recorded steps")
        a, y = np.asarray(rows, dtype=np.float64), np.asarray(host_us)
        if not a[:, 2].any():
            logger.warning("%s has no prefill steps: no per-token host time", path)
        # NNLS over the determined subsets of terms; on a tie the smaller
        # subset, preferring the fixed term, wins.
        best, best_err = np.zeros(3), np.inf
        subsets = sorted(
            (c for k in (1, 2, 3) for c in itertools.combinations(range(3), k)),
            key=lambda c: (len(c), c[0] != 0),
        )
        for cols in subsets:
            if np.linalg.matrix_rank(a[:, cols]) < len(cols):
                continue
            fit, *_ = np.linalg.lstsq(a[:, cols], y, rcond=None)
            err = float(np.sum((a[:, cols] @ fit - y) ** 2))
            if (fit >= 0).all() and err < best_err * (1 - _FIT_TIE_TOLERANCE):
                best, best_err = np.zeros(3), err
                best[list(cols)] = fit
        model = cls(*(float(c) for c in best))
        logger.info("Host time fitted to %s: %s", path, model)
        return model

    def host_s(self, step: StepDescriptor) -> float:
        return 1e-6 * (
            self.fixed_us
            + self.per_req_us * step.num_reqs
            + self.per_prefill_token_us * step.num_prefill_tokens
        )


class LinearTimingProvider:
    """Linear model: ``fixed_us + prefill_us_per_token * P + decode_us_per_req * D
    + decode_us_per_kv_token * K`` where P is prefill tokens, D decode
    requests and K the KV tokens read by decode requests. Host time is
    `host_us` per step, or fitted to the recorded log `host_fit_log`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        fixed_us: float = 0.0,
        prefill_us_per_token: float = 0.0,
        decode_us_per_req: float = 0.0,
        decode_us_per_kv_token: float = 0.0,
        host_us: float = 0.0,
        host_fit_log: str | None = None,
    ) -> None:
        self.fixed_us = fixed_us
        self.prefill_us_per_token = prefill_us_per_token
        self.decode_us_per_req = decode_us_per_req
        self.decode_us_per_kv_token = decode_us_per_kv_token
        if host_fit_log is not None and host_us:
            raise ValueError("Pass host_us or host_fit_log, not both.")
        self.host = (
            HostTimeModel.from_log(host_fit_log)
            if host_fit_log is not None
            else HostTimeModel(fixed_us=host_us)
        )

    def estimate(self, step: StepDescriptor) -> StepCost:
        decode_kv_tokens = sum(
            r.num_computed_tokens + r.num_scheduled_tokens
            for r in step.requests
            if not r.is_prefill
        )
        prefill = self.prefill_us_per_token * step.num_prefill_tokens
        decode = (
            self.decode_us_per_req * step.num_decode_reqs
            + self.decode_us_per_kv_token * decode_kv_tokens
        )
        device_us = self.fixed_us + prefill + decode
        return StepCost(
            device_s=device_us * 1e-6,
            host_s=self.host.host_s(step),
            breakdown={"prefill_us": prefill, "decode_us": decode},
        )


# Dense bf16 tensor-core peak and HBM bandwidth per GPU (SXM parts).
GPU_PRESETS: dict[str, dict[str, float]] = {
    "a100": {"peak_tflops": 312.0, "hbm_tbps": 2.039},
    "h100": {"peak_tflops": 989.0, "hbm_tbps": 3.35},
    "h200": {"peak_tflops": 989.0, "hbm_tbps": 4.8},
    "b200": {"peak_tflops": 2250.0, "hbm_tbps": 8.0},
}
# Each GPU's link to the others of its NVLink domain (SXM parts): the
# published bidirectional bandwidth halved, GB/s per direction. Physical
# peaks: `interconnect_efficiency` scales them to what collectives achieve.
INTERCONNECT_PRESETS: dict[str, dict[str, Any]] = {
    "a100": {"interconnect": "NVLink 3", "interconnect_gbs": 300.0},
    "h100": {"interconnect": "NVLink 4", "interconnect_gbs": 450.0},
    "h200": {"interconnect": "NVLink 4", "interconnect_gbs": 450.0},
    "b200": {"interconnect": "NVLink 5", "interconnect_gbs": 900.0},
}
# GPUs one NVLink domain joins in each preset's usual (HGX) node: the widest
# expert-parallel group modeled so far.
NVLINK_DOMAIN_GPUS = 8
# Expert-parallel all-to-all backends modeled, by vLLM's all2all_backend name:
# a dispatched token copy is its hidden state plus routing metadata, through
# the modular interface its top_k weights (float32) and ids (int32 gathered,
# int64 for DeepEP).
EXPERT_PARALLEL_BACKENDS = {
    "allgather_reducescatter": {"id_bytes": 4, "weight_bytes": 4},
    "deepep_high_throughput": {"id_bytes": 8, "weight_bytes": 4},
}
# How the allgather_reducescatter backend dispatches: through a modular MoE
# kernel, a token's top-k ids and weights; through a monolithic one (FlashInfer
# TRTLLM's, Blackwell only), every expert's router logit.
MOE_INTERFACES = ("modular", "monolithic")
# Each preset's CUDA compute capability, which vLLM's backend choice follows.
COMPUTE_CAPABILITIES: dict[str, tuple[int, int]] = {
    "a100": (8, 0),
    "h100": (9, 0),
    "h200": (9, 0),
    "b200": (10, 0),
}
# KV cache dtypes packing elements below a byte, or with per-token scales:
# their size is not their storage dtype's (fp8_ds_mla: 656 bytes per token).
PACKED_KV_DTYPES = (
    "int4",
    "nvfp4",
    "turboquant",
    "int8_per_token_head",
    "fp8_per_token_head",
    "fp8_ds_mla",
)
_MLA_KV_DTYPES_16_BIT = ("auto", "float16", "bfloat16")
_MLA_KV_DTYPES = (*_MLA_KV_DTYPES_16_BIT, "fp8", "fp8_e4m3")
_MLA_LATENT_SIZES = (320, 512, 576)


@dataclass(frozen=True)
class ExpertGeometry:
    """One layer's routed experts on this rank."""

    local_experts: int
    global_experts: int
    top_k: int
    params_per_expert: int
    """Weights of one expert's gate/up and down projections, logical
    (unpadded)."""

    def expected_read(self, num_tokens: int) -> float:
        """Local experts a step's tokens are expected to route to, each token
        to `top_k` of the global experts, independently and uniformly."""
        return self.local_experts * (
            1.0 - (1.0 - self.top_k / self.global_experts) ** num_tokens
        )


@dataclass(frozen=True)
class LatentAttentionGeometry:
    """One layer's multi-head latent attention (MLA) on this rank: a cached
    latent of `kv_lora_rank` plus a shared rope key per token, which
    `kv_b_proj` up-projects into the heads' non-rope keys and values."""

    num_heads: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_b_params: int
    """Weights of `kv_b_proj` on this rank, logical (unpadded)."""

    @property
    def latent_size(self) -> int:
        """Elements one token caches in this layer, on every rank."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    def attention_flops(self, pairs: int, absorbed: bool) -> int:
        """QK^T and AV over `pairs` query-key pairs: on the decode path MQA
        over the latent (keys `latent_size`, values `kv_lora_rank` wide),
        on the prefill path MHA over the up-projected keys and values."""
        if absorbed:
            return 2 * self.num_heads * pairs * (self.latent_size + self.kv_lora_rank)
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        return 2 * self.num_heads * pairs * (qk_head_dim + self.v_head_dim)

    def kv_b_flops(self, request: SimRequestStep, absorbed: bool) -> int:
        """`kv_b_proj`'s share of a request's step. The decode path absorbs it
        into each query token (keys before QK^T, values after AV: one
        application per token); the prefill path up-projects the latent of
        every scheduled and cached context token."""
        tokens = request.num_scheduled_tokens
        if not absorbed:
            tokens += request.num_computed_tokens
        return 2 * self.kv_b_params * tokens


@dataclass(frozen=True)
class LatentAttentionBackend:
    """One of vLLM's CUDA MLA backends: what it serves (as its
    `validate_configuration`), and how it splits a batch between the decode
    path and the prefill path (its metadata builder's reorder threshold, and
    whether its decode queries must be of one length)."""

    name: str
    decode_threshold: int
    uniform_decode: bool
    compute_capabilities: tuple[int, ...] | None
    """Majors it runs on; None: any."""
    kv_cache_dtypes: tuple[str, ...]
    kernel_block_sizes: tuple[int, ...]
    """A block size must be a multiple of one."""
    latent_sizes: tuple[int, ...] | None = _MLA_LATENT_SIZES
    """Head sizes (`latent_size`) it serves; None: any."""
    qk_nope_head_dims: tuple[int, ...] | None = None
    fp8_min_capability: tuple[int, int] | None = None

    def unsupported(
        self,
        latent: LatentAttentionGeometry,
        kv_cache_dtype: str,
        capability: tuple[int, int] | None,
        block_size: int | None,
    ) -> list[str]:
        """Why vLLM would not run this backend for the layer, cache and GPU
        (`capability` None: not checked); empty if it would."""
        reasons = []
        if self.latent_sizes is not None and latent.latent_size not in self.latent_sizes:
            reasons.append(f"latent width {latent.latent_size} not in {self.latent_sizes}")
        if (
            self.qk_nope_head_dims is not None
            and latent.qk_nope_head_dim not in self.qk_nope_head_dims
        ):
            reasons.append(
                f"qk_nope_head_dim {latent.qk_nope_head_dim} not in {self.qk_nope_head_dims}"
            )
        if kv_cache_dtype not in self.kv_cache_dtypes:
            reasons.append(f"KV cache dtype {kv_cache_dtype} not in {self.kv_cache_dtypes}")
        if block_size is not None and not any(
            block_size % size == 0 for size in self.kernel_block_sizes
        ):
            reasons.append(
                f"block size {block_size} not a multiple of one of {self.kernel_block_sizes}"
            )
        if capability is not None:
            if (
                self.compute_capabilities is not None
                and capability[0] not in self.compute_capabilities
            ):
                reasons.append(f"compute capability {capability} not supported")
            elif (
                self.fp8_min_capability is not None
                and kv_cache_dtype.startswith("fp8")
                and capability < self.fp8_min_capability
            ):
                reasons.append(
                    f"an fp8 KV cache needs compute capability {self.fp8_min_capability}"
                )
        return reasons

    def absorbed(self, requests: tuple[SimRequestStep, ...]) -> tuple[bool, ...]:
        """Which requests take the decode path. The runner orders the batch
        as decodes, short extends, long extends, then first chunks
        (`reorder_batch_to_split_decodes_and_prefills`; within each region
        the step's order is kept here), and the builder splits it after
        its leading decodes (`split_decodes_and_prefills`)."""
        if not requests:
            return ()
        threshold = self.decode_threshold

        def region(r: SimRequestStep) -> int:
            if r.num_computed_tokens == 0:
                return 3
            if r.num_scheduled_tokens > threshold:
                return 2
            return 1 if r.num_computed_tokens < r.num_prompt_tokens else 0

        order = sorted(range(len(requests)), key=lambda i: region(requests[i]))
        lens = [requests[i].num_scheduled_tokens for i in order]
        num_decodes = len(lens)
        all_short = max(lens, default=0) <= threshold
        if all_short and (not self.uniform_decode or threshold <= 1):
            pass  # every request a decode, without looking at lengths
        elif lens[0] > threshold:
            num_decodes = 0
        elif self.uniform_decode:
            num_decodes = next((i for i, n in enumerate(lens) if n != lens[0]), len(lens))
        else:
            num_decodes = next((i for i, n in enumerate(lens) if n > threshold), len(lens))
        decodes = set(order[:num_decodes])
        return tuple(i in decodes for i in range(len(requests)))


# vLLM's CUDA MLA backends that serve dense latent attention: their metadata
# builders' reorder_batch_threshold and query_len_support (UNIFORM and
# SINGLE_ONLY require one decode query length), without speculative
# decoding; what each serves (FlashMLA's dense kernels: Hopper only).
LATENT_ATTENTION_BACKENDS = {
    backend.name: backend
    for backend in (
        LatentAttentionBackend(
            "TRITON_MLA", 1, True, None, _MLA_KV_DTYPES, (16,),
            latent_sizes=None, fp8_min_capability=(8, 9),
        ),
        LatentAttentionBackend("FLASHMLA", 128, True, (9,), _MLA_KV_DTYPES, (64,)),
        LatentAttentionBackend(
            "FLASH_ATTN_MLA", 512, False, (9,), _MLA_KV_DTYPES_16_BIT, (16,)
        ),
        LatentAttentionBackend(
            "FLASHINFER_MLA", 1, False, (10,), _MLA_KV_DTYPES, (32, 64),
            qk_nope_head_dims=(64, 128, 192),
        ),
        LatentAttentionBackend("CUTLASS_MLA", 1, True, (10,), _MLA_KV_DTYPES, (128,)),
    )
}  # fmt: skip
# vLLM's CUDA MLA priorities by compute capability major
# (`_get_backend_priorities`), the first that supports the model, cache and
# GPU chosen. TOKENSPEED_MLA, behind FLASHINFER_MLA on SM100, serves a
# subset of its cases (fp8 KV, DeepSeek-R1's dimensions, 32/64 blocks), so
# never wins here.
LATENT_ATTENTION_PRIORITIES: dict[int, tuple[str, ...]] = {
    10: ("FLASHINFER_MLA", "CUTLASS_MLA", "FLASH_ATTN_MLA", "FLASHMLA", "TRITON_MLA"),
}
_DEFAULT_LATENT_ATTENTION_PRIORITY = (
    "FLASH_ATTN_MLA",
    "FLASHMLA",
    "FLASHINFER_MLA",
    "TRITON_MLA",
)


@dataclass(frozen=True)
class LayerGeometry:
    """One decoder layer's matrix multiplications on this rank."""

    dense_params: int
    """Weights every token goes through: attention projections, dense MLP,
    shared experts, router; not an MLA layer's `kv_b_proj` (`latent`)."""
    experts: tuple[ExpertGeometry, ...] = ()
    latent: LatentAttentionGeometry | None = None


# How an MoE step's expert reads are estimated, as recorded with the geometry.
ROUTING_MODEL = {
    "model": "independent_uniform_top_k",
    "experts_read_per_layer": (
        "local_experts * (1 - (1 - top_k / global_experts) ** step_tokens)"
    ),
    "flops_per_token_per_layer": "2 * (dense_params + top_k * params_per_expert)",
    "not_modeled": [
        "routing skew (error direction unknown)",
        "grouped (node-limited) routing",
    ],
}


def rank_reach_probability(global_experts: int, local_experts: int, top_k: int) -> float:
    """The probability that a token's top_k distinct experts, chosen
    uniformly, include one of a rank's local_experts:
    1 - C(E - m, k) / C(E, k)."""
    return 1.0 - math.comb(global_experts - local_experts, top_k) / math.comb(
        global_experts, top_k
    )


class UnsupportedModelLayout(ValueError):
    """The built model's decoder layers cannot be told apart."""


def _group_layers(layers: tuple[LayerGeometry, ...]) -> list[dict[str, Any]]:
    """Consecutive layers of one geometry, as ``first``-``last`` ranges."""
    groups: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        if groups and layers[groups[-1]["last"]] == layer:
            groups[-1]["last"] = index
        else:
            group = {
                "first": index,
                "last": index,
                "dense_params": layer.dense_params,
                "experts": [asdict(e) for e in layer.experts],
            }
            if layer.latent is not None:
                group["latent"] = asdict(layer.latent)
            groups.append(group)
    return groups


@dataclass(frozen=True)
class ModelGeometry:
    """What a roofline needs of a model, per tensor-parallel rank."""

    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    intermediate_size: int
    vocab_size: int
    weight_bytes: float
    """Bytes per weight (dtype size)."""
    kv_bytes: float
    """Bytes per KV cache element."""
    activation_bytes: float = 2
    """Bytes per activation element (the model's dtype, whatever its weights'
    or KV's)."""
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel: bool = False
    """Routed experts split across the data-parallel ranks (EP), each rank
    running attention on its own requests (TP=1 per rank)."""
    routing_method: str | None = None
    """The MoE layers' routing (vLLM's RoutingMethodType name)."""
    router_logits_bytes: int | None = None
    """Bytes per router logit (the gate's output dtype)."""
    layers: tuple[LayerGeometry, ...] | None = None
    """Each decoder layer's matrix multiplications, from the built model;
    None: `layer_params` for every layer (a dense model's config)."""

    @classmethod
    def from_config(cls, vllm_config: VllmConfig) -> ModelGeometry:
        """A dense model's geometry from its config."""
        if vllm_config.model_config.is_moe or vllm_config.model_config.use_mla:
            raise ValueError(
                "The roofline timing provider needs an MoE or MLA model's "
                "built layers: ModelGeometry.from_model."
            )
        return cls._from_config(vllm_config)

    @classmethod
    def from_model(cls, vllm_config: VllmConfig, model: Any) -> ModelGeometry:
        """The geometry of the built model (simulation builds it on the meta
        device, with this rank's shapes): each decoder layer's matrix
        multiplications at their logical sizes, the routed experts' and an
        MLA layer's `kv_b_proj` apart. Embeddings are lookups and norms no
        matrix multiplications; the LM head stays `vocab_size`, run for the
        sampled tokens only."""
        import torch

        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase

        def linear_params(module: torch.nn.Module) -> int:
            if isinstance(module, LinearBase):
                inputs = getattr(module, "input_size_per_partition", module.input_size)
                outputs = getattr(module, "output_partition_sizes", None) or [
                    getattr(module, "output_size_per_partition", module.output_size)
                ]
                return inputs * sum(outputs)
            assert isinstance(module, torch.nn.Linear)
            return module.in_features * module.out_features

        geometry = cls._from_config(vllm_config)
        num_layers = vllm_config.model_config.get_total_num_hidden_layers()
        # The decoder: the list of the model's layer count named "layers", or
        # the only such list under any name (EXAONE's transformer.h).
        lists = [
            (name.rpartition(".")[2], module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.ModuleList) and len(module) == num_layers
        ]
        named = [module for name, module in lists if name == "layers"]
        candidates = named if named else [module for _, module in lists]
        if len(candidates) != 1:
            raise UnsupportedModelLayout(
                f"The roofline timing provider cannot tell the decoder layers "
                f"apart: {len(lists)} module lists of {num_layers} modules, "
                f"{len(named)} named 'layers'."
            )
        layers = []
        for layer in candidates[0]:
            dense, experts, latent = 0, [], None
            attentions = [m for m in layer.modules() if isinstance(m, MLAAttention)]
            if len(attentions) > 1:
                raise UnsupportedModelLayout(
                    f"The roofline timing provider models one MLA attention "
                    f"per decoder layer, not {len(attentions)}."
                )
            for attention in attentions:
                if (
                    attention.indexer is not None
                    or getattr(attention, "use_sparse", False)
                    or attention.sliding_window is not None
                ):
                    raise ValueError(
                        "The roofline timing provider models dense MLA (no "
                        "sparse or sliding-window MLA)."
                    )
                latent = LatentAttentionGeometry(
                    num_heads=attention.num_heads,
                    kv_lora_rank=attention.kv_lora_rank,
                    qk_nope_head_dim=attention.qk_nope_head_dim,
                    qk_rope_head_dim=attention.qk_rope_head_dim,
                    v_head_dim=attention.v_head_dim,
                    kv_b_params=linear_params(attention.kv_b_proj),
                )
            # modules() yields a module referenced twice (a shared expert's
            # gate, MLA's kv_b_proj, say) once.
            for module in layer.modules():
                if attentions and module is attentions[0].kv_b_proj:
                    continue
                if isinstance(module, RoutedExperts):
                    moe = module.moe_config
                    hidden = moe.hidden_dim_unpadded or moe.hidden_dim
                    intermediate = (
                        moe.intermediate_size_per_partition_unpadded
                        or moe.intermediate_size_per_partition
                    )
                    projections = 3 if moe.is_act_and_mul else 2
                    experts.append(
                        ExpertGeometry(
                            local_experts=moe.num_local_experts,
                            global_experts=moe.num_experts,
                            top_k=moe.experts_per_token,
                            params_per_expert=projections * hidden * intermediate,
                        )
                    )
                elif isinstance(module, (LinearBase, torch.nn.Linear)):
                    dense += linear_params(module)
            layers.append(LayerGeometry(dense, tuple(experts), latent))
        routers = [
            module.moe_config
            for layer in candidates[0]
            for module in layer.modules()
            if isinstance(module, RoutedExperts)
        ]
        if routers:
            method = getattr(routers[0], "routing_method", None)
            dtype = getattr(routers[0], "router_logits_dtype", None)
            geometry = dataclasses.replace(
                geometry,
                routing_method=method.name if method is not None else None,
                router_logits_bytes=dtype.itemsize if dtype is not None else None,
            )
        if any(layer.experts for layer in layers):
            cls._check_moe(vllm_config)
            if geometry.expert_parallel:
                ranks = geometry.data_parallel_size
                for layer in layers:
                    for experts in layer.experts:
                        if experts.local_experts * ranks != experts.global_experts:
                            raise ValueError(
                                "The roofline timing provider models experts "
                                f"placed evenly: {experts.local_experts} of "
                                f"{experts.global_experts} on one of {ranks} ranks."
                            )
        if any(layer.latent for layer in layers):
            cls._check_latent(vllm_config)
        elif vllm_config.model_config.use_mla:
            raise UnsupportedModelLayout(
                "The roofline timing provider found no MLA attention in the "
                "MLA model's decoder layers."
            )
        return dataclasses.replace(geometry, layers=tuple(layers))

    @staticmethod
    def _check_latent(vllm_config: VllmConfig) -> None:
        parallel = vllm_config.parallel_config
        # Speculative decoding raises the backends' decode thresholds.
        if (
            parallel.decode_context_parallel_size > 1
            or parallel.prefill_context_parallel_size > 1
            or vllm_config.speculative_config is not None
        ):
            raise ValueError(
                "The roofline timing provider models MLA models without "
                "context parallelism (DCP, PCP) or speculative decoding."
            )
        ModelGeometry._check_unquantized(vllm_config, "MLA")

    @staticmethod
    def _check_unquantized(vllm_config: VllmConfig, kind: str) -> None:
        import torch

        model = vllm_config.model_config
        if model.quantization is not None or model.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError(
                f"The roofline timing provider models {kind} models with "
                f"unquantized bf16 or fp16 weights, not {model.quantization or model.dtype}."
            )

    @staticmethod
    def _check_moe(vllm_config: VllmConfig) -> None:
        parallel = vllm_config.parallel_config
        tp = parallel.tensor_parallel_size
        # Pure TP keeps every expert on each rank, its matrices split. Data
        # parallelism with EP splits the experts across the DP ranks, each
        # running attention for its own requests. DCP reuses the TP ranks, so
        # the world size does not show it.
        pure_tp = (
            parallel.world_size_across_dp == tp and not parallel.enable_expert_parallel
        )
        dp_attention_ep = (
            parallel.enable_expert_parallel
            and tp == 1
            and parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size > 1
        )
        if (
            not (pure_tp or dp_attention_ep)
            or parallel.decode_context_parallel_size > 1
            or parallel.prefill_context_parallel_size > 1
        ):
            raise ValueError(
                "The roofline timing provider models MoE models under tensor "
                "parallelism, or data-parallel attention with expert "
                "parallelism (one rank per DP rank), without PP, PCP or DCP "
                "so far."
            )
        if parallel.enable_eplb:
            raise ValueError(
                "The roofline timing provider does not model EPLB's redundant "
                "experts."
            )
        ModelGeometry._check_unquantized(vllm_config, "MoE")

    @classmethod
    def _from_config(cls, vllm_config: VllmConfig) -> ModelGeometry:
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        if model.is_hybrid or model.is_attention_free:
            raise ValueError(
                "The roofline timing provider models dense, MoE and MLA "
                "attention models (no hybrid or attention-free ones)."
            )
        tp = parallel.tensor_parallel_size
        text = model.hf_text_config
        cache_dtype = vllm_config.cache_config.cache_dtype
        if cache_dtype.startswith(PACKED_KV_DTYPES):
            raise ValueError(f"The roofline timing provider does not model {cache_dtype} KV.")
        weight_bytes = model.dtype.itemsize
        return cls(
            num_layers=model.get_num_layers(parallel),
            hidden_size=model.get_hidden_size(),
            num_heads=model.get_num_attention_heads(parallel),
            num_kv_heads=model.get_num_kv_heads(parallel),
            head_size=model.get_head_size(),
            intermediate_size=getattr(text, "intermediate_size", 0) // tp,
            vocab_size=model.get_vocab_size() // tp,
            weight_bytes=weight_bytes,
            kv_bytes=kv_cache_dtype_str_to_dtype(cache_dtype, model).itemsize,
            activation_bytes=model.dtype.itemsize,
            kv_cache_dtype=cache_dtype,
            tensor_parallel_size=tp,
            data_parallel_size=parallel.data_parallel_size,
            expert_parallel=parallel.enable_expert_parallel,
        )

    @property
    def layer_params(self) -> int:
        """Linear-layer weights of one decoder layer on this rank."""
        q_and_o = 2 * self.hidden_size * self.num_heads * self.head_size
        k_and_v = 2 * self.hidden_size * self.num_kv_heads * self.head_size
        mlp = 3 * self.hidden_size * self.intermediate_size
        return q_and_o + k_and_v + mlp

    @property
    def token_params(self) -> int:
        """Decoder weights a token goes through: every layer's dense ones and
        `top_k` of its experts (not `kv_b_proj`: `attention_flops`)."""
        if self.layers is None:
            return self.num_layers * self.layer_params
        return sum(
            layer.dense_params
            + sum(e.top_k * e.params_per_expert for e in layer.experts)
            for layer in self.layers
        )

    def weights_read(self, num_tokens: int, latent_paths: int = 1) -> float:
        """Decoder weights a step of `num_tokens` tokens reads: every layer's
        dense ones, the experts its tokens are expected to route to, and an
        MLA layer's `kv_b_proj` once per path the step takes (the decode
        path reads the absorbed copies vLLM makes of it at load)."""
        if self.layers is None:
            return self.num_layers * self.layer_params
        return sum(
            layer.dense_params
            + sum(e.expected_read(num_tokens) * e.params_per_expert for e in layer.experts)
            + (layer.latent.kv_b_params * latent_paths if layer.latent else 0)
            for layer in self.layers
        )

    @functools.cached_property
    def _attention_layers(
        self,
    ) -> tuple[int, tuple[tuple[LatentAttentionGeometry, int], ...]]:
        """Layers of multi-head attention, and of each MLA geometry."""
        if self.layers is None:
            return self.num_layers, ()
        latent = collections.Counter(layer.latent for layer in self.layers)
        plain = latent.pop(None, 0)
        return plain, tuple(latent.items())  # type: ignore[arg-type]

    @property
    def ep_size(self) -> int:
        """Ranks the routed experts are split across, synchronizing at every
        MoE layer: the DP ranks under EP, else 1."""
        if self.expert_parallel and any(
            layer.experts for layer in self.layers or ()
        ):
            return self.data_parallel_size
        return 1

    @property
    def has_latent_attention(self) -> bool:
        return bool(self._attention_layers[1])

    def layer_attention_flops(
        self, layer: LayerGeometry, request: SimRequestStep, absorbed: bool
    ) -> int:
        """One request's QK^T and AV in one layer, and its MLA `kv_b_proj`."""
        scheduled = request.num_scheduled_tokens
        pairs = scheduled * request.num_computed_tokens + scheduled * (scheduled + 1) // 2
        if layer.latent is None:
            return 4 * self.num_heads * self.head_size * pairs
        return layer.latent.attention_flops(pairs, absorbed) + layer.latent.kv_b_flops(
            request, absorbed
        )

    def layer_kv_bytes_per_token(self, layer: LayerGeometry) -> float:
        """K and V, or the MLA latent, of one token in one layer."""
        if layer.latent is None:
            return 2 * self.num_kv_heads * self.head_size * self.kv_bytes
        return layer.latent.latent_size * self.kv_bytes

    def attention_flops(
        self, requests: tuple[SimRequestStep, ...], absorbed: tuple[bool, ...]
    ) -> int:
        """A step's QK^T and AV in every layer, each new token attending to
        its request's context and to the new tokens before it, plus each
        MLA layer's `kv_b_proj`; `absorbed[i]`: whether request i takes the
        MLA decode path."""
        plain, latent = self._attention_layers
        flops = 0
        for request, decode_path in zip(requests, absorbed):
            scheduled = request.num_scheduled_tokens
            pairs = (
                scheduled * request.num_computed_tokens
                + scheduled * (scheduled + 1) // 2
            )
            flops += 4 * plain * self.num_heads * self.head_size * pairs
            for geometry, count in latent:
                flops += count * (
                    geometry.attention_flops(pairs, decode_path)
                    + geometry.kv_b_flops(request, decode_path)
                )
        return flops

    def describe(self) -> dict[str, Any]:
        """The geometry as recorded: consecutive layers of one geometry
        grouped."""
        described = asdict(self)
        described["source"] = "config" if self.layers is None else "built model"
        if self.layers is not None:
            described["layers"] = _group_layers(self.layers)
            if any(layer.experts for layer in self.layers):
                described["routing"] = ROUTING_MODEL
        return described

    @property
    def kv_bytes_per_token(self) -> float:
        """K and V of one token across all layers on this rank; an MLA
        layer's latent, which every rank holds whole."""
        plain, latent = self._attention_layers
        return (
            plain * 2 * self.num_kv_heads * self.head_size
            + sum(geometry.latent_size * count for geometry, count in latent)
        ) * self.kv_bytes


class RooflineTimingProvider:
    """Roofline model of a dense or MoE transformer, one GPU per
    tensor-parallel rank: a step takes ``fixed_us`` plus the longer of its
    FLOPs at ``peak_tflops * compute_efficiency`` and its bytes at
    ``hbm_tbps * bandwidth_efficiency``, plus its communication.

    FLOPs: 2 per weight per token through the decoder layers (an MoE
    layer's dense weights and ``top_k`` of its experts), the LM head for the
    tokens sampled, and attention (QK^T and AV) over each request's context.
    Bytes: the dense weights once per step, the experts the step's tokens
    are expected to route to (each token independently and uniformly), the
    KV each request reads (its context) and writes. Under tensor
    parallelism, each layer's two all-reduces are added to the step (a
    no-overlap approximation: `communication_s`) at the interconnect's
    peak (`INTERCONNECT_PRESETS`, or ``interconnect_gbs``) times
    ``interconnect_efficiency``, plus ``collective_latency_us`` each. Not
    modeled: activations, embedding and logits collectives, kernel
    efficiency varying with batch shape, routing skew (fewer experts read,
    but uneven work: the error's direction is unknown). ``gpu`` picks a
    preset of `GPU_PRESETS`; ``peak_tflops`` and ``hbm_tbps`` override it.
    An MoE or MLA model's geometry comes from its built layers
    (`bind_model`).

    MLA: a request takes the decode path (MQA over the latent) or the
    prefill path (MHA over its up-projected context) as the modeled
    backend splits the batch (`LatentAttentionBackend`): ``mla_backend``
    (a `LATENT_ATTENTION_BACKENDS` name), else vLLM's default on ``gpu``.
    Every rank reads and writes the whole latent.

    Data-parallel attention with expert parallelism (TP=1 per DP rank, up to
    `NVLINK_DOMAIN_GPUS` ranks): a joint step's segments between the MoE
    layers' synchronizations each take the slowest rank's own work; every
    rank's share of the experts and the all-to-all (``ep_all2all``, an
    `EXPERT_PARALLEL_BACKENDS` name, else ``all2all_backend``; its dispatch
    through ``ep_moe_interface``, else vLLM's default MoE kernel on ``gpu``)
    follow without overlap."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        gpu: str | None = None,
        peak_tflops: float | None = None,
        hbm_tbps: float | None = None,
        compute_efficiency: float = 0.6,
        bandwidth_efficiency: float = 0.8,
        fixed_us: float = 20.0,
        host_us: float = 0.0,
        geometry: ModelGeometry | None = None,
        interconnect_gbs: float | None = None,
        interconnect_efficiency: float = 0.8,
        collective_latency_us: float = 10.0,
        mla_backend: str | None = None,
        ep_all2all: str | None = None,
        ep_moe_interface: str | None = None,
    ) -> None:
        if gpu is not None and gpu not in GPU_PRESETS:
            raise ValueError(f"Unknown GPU {gpu!r}; presets: {sorted(GPU_PRESETS)}.")
        preset = GPU_PRESETS[gpu] if gpu else {}
        peak = peak_tflops if peak_tflops is not None else preset.get("peak_tflops")
        bandwidth = hbm_tbps if hbm_tbps is not None else preset.get("hbm_tbps")
        if not peak or not bandwidth:
            raise ValueError("The roofline timing provider needs gpu, or peak_tflops and hbm_tbps.")
        self.flops_per_s = peak * 1e12 * compute_efficiency
        self.bytes_per_s = bandwidth * 1e12 * bandwidth_efficiency
        self.fixed_us = fixed_us
        self.host = HostTimeModel(fixed_us=host_us)
        for name, value, low in (
            ("interconnect_gbs", interconnect_gbs, None),
            ("interconnect_efficiency", interconnect_efficiency, None),
            ("collective_latency_us", collective_latency_us, 0.0),
        ):
            if value is None:
                continue
            if not math.isfinite(value) or (value < low if low is not None else value <= 0):
                bound = "a finite nonnegative" if low is not None else "a finite positive"
                raise ValueError(f"{name} must be {bound} number, got {value}.")
        link = INTERCONNECT_PRESETS.get(gpu, {}) if gpu else {}
        link_gbs = interconnect_gbs if interconnect_gbs is not None else link.get("interconnect_gbs")
        self.collective_bytes_per_s = (
            link_gbs * 1e9 * interconnect_efficiency if link_gbs else None
        )
        self.collective_latency_s = collective_latency_us * 1e-6
        self._communication = {
            "model": "no_overlap_approximation",
            "collectives": (
                "2 all-reduces per decoder layer (attention output, MLP or MoE "
                "output) of scheduled_tokens x hidden_size x activation_bytes"
            ),
            "time": (
                "2 * layers * (latency + 2 * (N - 1) / N * payload / "
                "effective_bandwidth), 0 for N = 1; added to the step"
            ),
            "algorithm_factor": "2 (N - 1) / N: a ring all-reduce's bus bandwidth (NCCL)",
            "omitted": [
                "embedding and LM-head (logits) collectives",
                "overlap with compute",
                "expert parallelism's dispatch and combine",
            ],
            "topology": "all tensor-parallel ranks in one NVLink domain",
            "interconnect": link.get("interconnect") if interconnect_gbs is None else None,
            "interconnect_gbs": link_gbs,
            "interconnect_efficiency": interconnect_efficiency,
            "effective_gbs": link_gbs * interconnect_efficiency if link_gbs else None,
            "collective_latency_us": collective_latency_us,
            "latency_rank_dependence": "none: one latency per collective",
        }
        if mla_backend is not None and mla_backend not in LATENT_ATTENTION_BACKENDS:
            raise ValueError(
                f"Unknown MLA backend {mla_backend!r}; modeled: "
                f"{sorted(LATENT_ATTENTION_BACKENDS)}."
            )
        if ep_all2all is not None and ep_all2all not in EXPERT_PARALLEL_BACKENDS:
            raise ValueError(
                f"The roofline timing provider models the expert-parallel "
                f"all-to-all as {sorted(EXPERT_PARALLEL_BACKENDS)}, not "
                f"{ep_all2all!r}."
            )
        self._gpu = gpu
        self._mla_backend = mla_backend
        if ep_moe_interface is not None and ep_moe_interface not in MOE_INTERFACES:
            raise ValueError(
                f"ep_moe_interface is one of {MOE_INTERFACES}, not {ep_moe_interface!r}."
            )
        self._ep_all2all = ep_all2all
        self._ep_moe_interface = ep_moe_interface
        # Resolved with an expert-parallel model's layers (_check_geometry).
        self.expert_all2all: str | None = None
        self.moe_interface: str | None = None
        self._expert_parallel: dict[str, Any] | None = None
        # Resolved with an MLA model's layers (_check_geometry).
        self.latent_backend: LatentAttentionBackend | None = None
        self._latent_attention: dict[str, Any] | None = None
        self._vllm_config = vllm_config
        # An MoE or MLA model's geometry comes from its built layers
        # (bind_model).
        self.geometry: ModelGeometry | None = geometry
        if geometry is None and not (
            vllm_config.model_config.is_moe or vllm_config.model_config.use_mla
        ):
            self.geometry = ModelGeometry.from_config(vllm_config)
        if self.geometry is not None:
            self._check_geometry(self.geometry)
        self._assumptions = {
            "gpu": gpu,
            "peak_tflops": peak,
            "hbm_tbps": bandwidth,
            "compute_efficiency": compute_efficiency,
            "bandwidth_efficiency": bandwidth_efficiency,
            "fixed_us": fixed_us,
            "host_us": host_us,
        }

    def bind_model(self, model: Any) -> None:
        """Take each layer's geometry from the built model, before any step
        is timed. A dense model whose decoder layers cannot be told apart
        keeps its config's; an MoE or MLA model needs its layers."""
        try:
            self.geometry = ModelGeometry.from_model(self._vllm_config, model)
        except UnsupportedModelLayout as error:
            config = self._vllm_config.model_config
            if config.is_moe or config.use_mla:
                raise
            logger.warning("%s Timing the dense model from its config.", error)
            self.geometry = ModelGeometry.from_config(self._vllm_config)
        self._check_geometry(self.geometry)

    def _check_geometry(self, geometry: ModelGeometry) -> None:
        if geometry.tensor_parallel_size > 1 and self.collective_bytes_per_s is None:
            raise ValueError(
                "The roofline timing provider needs an interconnect for tensor "
                "parallelism: a gpu preset, or interconnect_gbs."
            )
        if geometry.has_latent_attention:
            self._select_latent_backend(geometry)
        if geometry.ep_size > 1:
            self._select_expert_all2all(geometry)

    def _select_expert_all2all(self, geometry: ModelGeometry) -> None:
        """The modeled expert-parallel all-to-all: ``ep_all2all``, else the
        backend vLLM constructs (``all2all_backend``), which on the CPU
        platform need not be the one modeled."""
        ranks = geometry.ep_size
        if ranks > NVLINK_DOMAIN_GPUS:
            raise ValueError(
                f"The roofline timing provider models expert parallelism within "
                f"one NVLink domain ({NVLINK_DOMAIN_GPUS} GPUs), not {ranks} ranks."
            )
        if self.collective_bytes_per_s is None:
            raise ValueError(
                "The roofline timing provider needs an interconnect for expert "
                "parallelism: a gpu preset, or interconnect_gbs."
            )
        configured = (
            self._vllm_config.parallel_config.all2all_backend
            if self._vllm_config is not None
            else None
        )
        modeled = self._ep_all2all or configured or "allgather_reducescatter"
        if modeled not in EXPERT_PARALLEL_BACKENDS:
            raise ValueError(
                f"The roofline timing provider models the expert-parallel "
                f"all-to-all as {sorted(EXPERT_PARALLEL_BACKENDS)}, not "
                f"{modeled!r}: pass ep_all2all."
            )
        self.expert_all2all = modeled
        metadata = EXPERT_PARALLEL_BACKENDS[modeled]
        interface, interface_selected_by = self._ep_moe_interface, "ep_moe_interface"
        if modeled != "allgather_reducescatter":
            # DeepEP dispatches top-k routed tokens only.
            if interface == "monolithic":
                raise ValueError(
                    f"{modeled} dispatches through the modular MoE interface, "
                    "not the monolithic one."
                )
            interface, interface_selected_by = "modular", modeled
        elif interface is None:
            if self._gpu is None:
                raise ValueError(
                    "The roofline timing provider needs the MoE interface the "
                    "allgather_reducescatter backend dispatches through: a gpu "
                    "preset, or ep_moe_interface."
                )
            if COMPUTE_CAPABILITIES[self._gpu][0] == 10:
                # TRTLLM's monolithic kernel comes first on Blackwell where its
                # routing, activation and expert-count limits allow: not
                # modeled, so the interface must be named.
                raise ValueError(
                    f"The roofline timing provider needs ep_moe_interface on "
                    f"{self._gpu}: vLLM's default MoE kernel there may be "
                    "monolithic or modular."
                )
            interface = "modular"
            interface_selected_by = (
                f"vLLM's default unquantized MoE kernel on {self._gpu}: a "
                "modular one (Triton first, no monolithic candidate)"
            )
        if interface == "monolithic" and geometry.router_logits_bytes is None:
            raise ValueError(
                "The roofline timing provider needs the router logits' dtype "
                "for the monolithic MoE interface."
            )
        self.moe_interface = interface
        if interface == "monolithic":
            routing = (
                f"global_experts * {geometry.router_logits_bytes} (every "
                "expert's router logit)"
            )
        else:
            routing = (
                f"top_k * ({metadata['id_bytes']} + {metadata['weight_bytes']}) "
                "(top-k ids and weights)"
            )
        if modeled == "allgather_reducescatter":
            communication = (
                "per MoE layer, a variable-size all-gather of the hidden states "
                "and routing metadata, then a variable-size reduce-scatter of "
                "the outputs (vLLM: one grouped NCCL broadcast or reduce per "
                "rank): each phase moves the other ranks' tokens, T - T_r, "
                "each way, priced at the busiest rank"
            )
        else:
            communication = (
                "per MoE layer, dispatch sends each token once to every other "
                "rank holding one of its top_k experts, combine returns one "
                "reduced output per such rank: rank r sends T_r (N - 1) p "
                "copies and receives (T - T_r) p, p = 1 - C(E - m, k) / C(E, k) "
                "for m local experts; each phase priced at the busiest rank's "
                "send or receive demand"
            )
        self._expert_parallel = {
            "model": (
                "data-parallel attention with expert parallelism: each rank runs "
                "its own requests' attention, dense weights and LM head; the "
                "routed experts are split evenly; the ranks synchronize at each "
                "MoE layer's dispatch and combine"
            ),
            "step_time": (
                "fixed + the sum, over the segments between synchronizations, "
                "of the slowest rank's max(compute, memory), + all-to-all time "
                "(no overlap)"
            ),
            "experts": (
                "every rank: 2 * T * top_k * local / global experts' FLOPs, and "
                "the local experts the joint step's T tokens are expected to "
                "route to read"
            ),
            "ranks": ranks,
            "all2all_modeled": modeled,
            "all2all_configured": configured,
            "all2all_configured_note": (
                "vLLM's all2all_backend setting; the CPU build may construct "
                "another (it cannot build DeepEP)"
            ),
            "moe_interface": interface,
            "moe_interface_selected_by": interface_selected_by,
            "routing_method": geometry.routing_method,
            "communication": communication,
            "bytes_per_token_copy": (
                f"dispatch: hidden_size * activation_bytes + {routing}; "
                "combine: hidden_size * activation_bytes"
            ),
            "dispatch_dtype_bytes": geometry.activation_bytes,
            "bandwidth": (
                "each GPU's interconnect bandwidth per direction times "
                "interconnect_efficiency, sending and receiving concurrently, "
                "plus collective_latency_us per phase (uncalibrated, shared "
                "with tensor parallelism)"
            ),
            "baseline": (
                "idealized: balanced uniform routing, no overlap; skew may "
                "lengthen steps (imbalance) or shorten them (fewer experts "
                "read), overlap shortens them"
            ),
            "not_modeled": [
                "routing skew",
                "grouped (node-limited) routing, which changes which ranks a "
                "token reaches",
                "EPLB's redundant experts",
                "overlap of the all-to-all with compute",
                "idle ranks' dummy batches and padding",
                "expert parallelism across NVLink domains",
                "DeepEP low-latency",
            ],
        }

    def _select_latent_backend(self, geometry: ModelGeometry) -> None:
        """The MLA backend vLLM would run for the model's layers, KV cache,
        block size and GPU: ``mla_backend`` if it supports them, else the
        first of vLLM's priorities on ``gpu`` that does."""
        capability = COMPUTE_CAPABILITIES[self._gpu] if self._gpu else None
        block_size = (
            self._vllm_config.cache_config.block_size
            if self._vllm_config is not None
            else None
        )
        latents = [latent for latent, _ in geometry._attention_layers[1]]

        def unsupported(name: str) -> list[str]:
            backend = LATENT_ATTENTION_BACKENDS[name]
            return sorted(
                {
                    reason
                    for latent in latents
                    for reason in backend.unsupported(
                        latent, geometry.kv_cache_dtype, capability, block_size
                    )
                }
            )

        rejected: dict[str, list[str]] = {}
        if self._mla_backend is not None:
            name = self._mla_backend
            if reasons := unsupported(name):
                raise ValueError(
                    f"vLLM would not run the MLA backend {name} here: "
                    + "; ".join(reasons)
                    + "."
                )
            selected_by = "mla_backend"
        elif capability is None:
            raise ValueError(
                "The roofline timing provider needs an MLA backend for an MLA "
                "model: a gpu preset, or mla_backend."
            )
        else:
            priorities = LATENT_ATTENTION_PRIORITIES.get(
                capability[0], _DEFAULT_LATENT_ATTENTION_PRIORITY
            )
            for name in priorities:
                if not (reasons := unsupported(name)):
                    break
                rejected[name] = reasons
            else:
                raise ValueError(
                    f"No MLA backend vLLM would run on {self._gpu}: {rejected}."
                )
            major, minor = capability
            selected_by = (
                f"vLLM's CUDA priorities on {self._gpu} (SM{major}{minor}): the "
                "first serving the layers, KV cache and block size"
            )
        backend = LATENT_ATTENTION_BACKENDS[name]
        self.latent_backend = backend
        self._latent_attention = {
            "backend": backend.name,
            "selected_by": selected_by,
            "rejected": rejected,
            "checked": {
                "kv_cache_dtype": geometry.kv_cache_dtype,
                "block_size": block_size,
                "compute_capability": capability,
            },
            "decode_threshold": backend.decode_threshold,
            "uniform_decode": backend.uniform_decode,
            "split": (
                "batch ordered as decodes, short extends, long extends, first "
                "chunks (each region in the step's order); the leading requests "
                "of at most decode_threshold tokens (with uniform_decode, of the "
                "first one's length) take the decode path"
            ),
            "decode_path": (
                "MQA over the latent: QK^T kv_lora_rank + rope wide, AV "
                "kv_lora_rank wide; kv_b_proj absorbed, once per query token"
            ),
            "prefill_path": (
                "MHA: QK^T qk_nope + rope wide, AV v_head wide; kv_b_proj "
                "up-projects every scheduled and cached context token"
            ),
            "kv_b_proj_read": "once per path a step takes",
        }

    def communication_s(self, num_tokens: int) -> float:
        """A step's tensor-parallel all-reduces, under the no-overlap
        approximation: 0 on one rank."""
        g = self._bound()
        n = g.tensor_parallel_size
        if n == 1:
            return 0.0
        assert self.collective_bytes_per_s is not None
        payload = num_tokens * g.hidden_size * g.activation_bytes
        per_collective = (
            self.collective_latency_s
            + 2 * (n - 1) / n * payload / self.collective_bytes_per_s
        )
        return 2 * g.num_layers * per_collective

    def _bound(self) -> ModelGeometry:
        if self.geometry is None:
            raise RuntimeError(
                "The roofline timing provider has no geometry: bind_model() "
                "the built model first."
            )
        return self.geometry

    def assumptions(self) -> dict[str, Any]:
        """What the step times rest on, resolved: the GPU's figures (preset or
        overridden), efficiencies, fixed costs and the model's geometry."""
        geometry = self._bound()
        assumptions = {
            **self._assumptions,
            "communication": self._communication,
            "geometry": geometry.describe(),
        }
        if geometry.has_latent_attention:
            assumptions["latent_attention"] = self._latent_attention
        if geometry.ep_size > 1:
            assumptions["expert_parallel"] = self._expert_parallel
        return assumptions

    def estimate(self, step: StepDescriptor) -> StepCost:
        g = self._bound()
        if g.ep_size > 1:
            return self._estimate_expert_parallel(step, g)
        sampled = sum(
            1
            for r in step.requests
            if r.num_computed_tokens + r.num_scheduled_tokens >= r.num_prompt_tokens
        )
        absorbed = (False,) * len(step.requests)
        if g.has_latent_attention:
            assert self.latent_backend is not None
            absorbed = self.latent_backend.absorbed(step.requests)
        latent_paths = any(absorbed) + (not all(absorbed))
        context = sum(r.num_computed_tokens + r.num_scheduled_tokens for r in step.requests)
        lm_head = g.hidden_size * g.vocab_size
        flops = (
            2 * g.token_params * step.num_tokens
            + 2 * lm_head * sampled
            + g.attention_flops(step.requests, absorbed)
        )
        memory = (g.weights_read(step.num_tokens, latent_paths) + lm_head) * g.weight_bytes
        memory += (context + step.num_tokens) * g.kv_bytes_per_token  # read and write
        compute_us = flops / self.flops_per_s * 1e6
        memory_us = memory / self.bytes_per_s * 1e6
        communication_us = self.communication_s(step.num_tokens) * 1e6
        breakdown = {
            "compute_us": compute_us,
            "memory_us": memory_us,
            "communication_us": communication_us,
        }
        if g.has_latent_attention:
            breakdown["mla_decode_path_reqs"] = float(sum(absorbed))
        return StepCost(
            device_s=(self.fixed_us + max(compute_us, memory_us) + communication_us) * 1e-6,
            host_s=self.host.host_s(step),
            breakdown=breakdown,
        )

    def _estimate_expert_parallel(self, step: StepDescriptor, g: ModelGeometry) -> StepCost:
        """A joint step of data-parallel ranks with expert-parallel MoE
        layers: each rank's own work between synchronizations, the slowest
        rank's setting the pace, then the experts' and the all-to-all's."""
        ranks = g.ep_size
        batches: list[list[SimRequestStep]] = [[] for _ in range(ranks)]
        for request in step.requests:
            batches[request.dp_rank].append(request)
        tokens = [sum(r.num_scheduled_tokens for r in batch) for batch in batches]
        total = sum(tokens)
        # Each rank splits its own batch between the MLA paths.
        absorbed = [
            self.latent_backend.absorbed(tuple(batch))
            if self.latent_backend is not None
            else (False,) * len(batch)
            for batch in batches
        ]
        flops = [0.0] * ranks
        memory = [0.0] * ranks
        local_s = experts_s = communication_s = 0.0

        def segment() -> float:
            slowest = max(
                max(f / self.flops_per_s, m / self.bytes_per_s)
                for f, m in zip(flops, memory)
            )
            flops[:] = [0.0] * ranks
            memory[:] = [0.0] * ranks
            return slowest

        for layer in g.layers or ():
            for rank, batch in enumerate(batches):
                if not batch:
                    continue  # an idle rank's dummy batch is not modeled
                paths = any(absorbed[rank]) + (not all(absorbed[rank]))
                weights = layer.dense_params
                if layer.latent is not None:
                    weights += layer.latent.kv_b_params * paths
                context = sum(r.num_computed_tokens + r.num_scheduled_tokens for r in batch)
                flops[rank] += 2 * layer.dense_params * tokens[rank] + sum(
                    g.layer_attention_flops(layer, r, a)
                    for r, a in zip(batch, absorbed[rank])
                )
                memory[rank] += weights * g.weight_bytes + (
                    context + tokens[rank]
                ) * g.layer_kv_bytes_per_token(layer)
            for experts in layer.experts:
                local_s += segment()  # dispatch
                share = experts.local_experts / experts.global_experts
                expert_flops = 2 * total * experts.top_k * share * experts.params_per_expert
                expert_bytes = (
                    experts.expected_read(total) * experts.params_per_expert * g.weight_bytes
                )
                experts_s += max(
                    expert_flops / self.flops_per_s, expert_bytes / self.bytes_per_s
                )
                communication_s += self.expert_all2all_s(tokens, experts, g)
        lm_head = g.hidden_size * g.vocab_size
        for rank, batch in enumerate(batches):
            if batch:
                sampled = sum(
                    1
                    for r in batch
                    if r.num_computed_tokens + r.num_scheduled_tokens >= r.num_prompt_tokens
                )
                flops[rank] += 2 * lm_head * sampled
                memory[rank] += lm_head * g.weight_bytes
        local_s += segment()
        breakdown = {
            "local_us": local_s * 1e6,
            "experts_us": experts_s * 1e6,
            "communication_us": communication_s * 1e6,
            "idle_dp_ranks": float(tokens.count(0)),
        }
        if g.has_latent_attention:
            breakdown["mla_decode_path_reqs"] = float(sum(map(sum, absorbed)))
        return StepCost(
            device_s=self.fixed_us * 1e-6 + local_s + experts_s + communication_s,
            host_s=self.host.host_s(step),
            breakdown=breakdown,
        )

    def expert_all2all_s(
        self, tokens: list[int], experts: ExpertGeometry, g: ModelGeometry
    ) -> float:
        """One MoE layer's dispatch and combine, for the ranks' token counts."""
        assert self.expert_all2all is not None and self.collective_bytes_per_s
        total = sum(tokens)
        metadata = EXPERT_PARALLEL_BACKENDS[self.expert_all2all]
        hidden = g.hidden_size * g.activation_bytes
        if self.moe_interface == "monolithic":
            assert g.router_logits_bytes is not None
            routing = experts.global_experts * g.router_logits_bytes
        else:
            routing = experts.top_k * (metadata["id_bytes"] + metadata["weight_bytes"])
        if self.expert_all2all == "allgather_reducescatter":
            copies = max(total - t for t in tokens)
        else:
            p = rank_reach_probability(
                experts.global_experts, experts.local_experts, experts.top_k
            )
            ranks = len(tokens)
            copies = max(max(t * (ranks - 1) * p, (total - t) * p) for t in tokens)
        dispatch = copies * (hidden + routing) / self.collective_bytes_per_s
        combine = copies * hidden / self.collective_bytes_per_s
        return 2 * self.collective_latency_s + dispatch + combine


def _step_features(step: StepDescriptor) -> tuple[float, float, float]:
    """Coarse batch features used for nearest-neighbour lookups."""
    past = sum(r.num_computed_tokens for r in step.requests)
    return (float(step.num_reqs), float(step.num_tokens), float(past))


# Lookups of this process's k-NN tables, and those outside the recorded range.
_knn_lookups = [0, 0]


def knn_lookup_counts() -> tuple[int, int]:
    """(lookups, lookups outside the recorded range) of this process's k-NN
    timing tables so far."""
    return _knn_lookups[0], _knn_lookups[1]


class _KNNTable:
    """Inverse-distance weighted k-NN over z-scored (num_reqs, num_tokens,
    sum_past_kv) features of recorded steps. A lookup outside the
    recorded range, which k-NN underestimates, is counted and warned
    about."""

    def __init__(
        self,
        points: list[tuple[tuple[float, float, float], tuple[float, float]]],
        k: int,
    ):
        if not points:
            raise ValueError("timing table is empty")
        feats = np.asarray([p for p, _ in points], dtype=np.float64)
        self.values = np.asarray([v for _, v in points], dtype=np.float64)
        self.mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        # One contiguous array per feature: summing three columns is several
        # times faster than reducing over a length-3 axis, with the same result.
        self.columns = np.ascontiguousarray(((feats - self.mean) / self.std).T)
        self.k = min(max(1, k), len(points))
        self.lo = feats.min(axis=0).tolist()
        self.hi = feats.max(axis=0).tolist()
        self.warned = False

    def query(self, f: tuple[float, float, float]) -> StepCost:
        _knn_lookups[0] += 1
        if any(not lo <= v <= hi for v, lo, hi in zip(f, self.lo, self.hi)):
            if not self.warned:
                self.warned = True
                logger.warning(
                    "Step (num_reqs, num_tokens, sum_past_kv) = %s is outside "
                    "the recorded range %s to %s: its time is that of the "
                    "nearest recorded steps. Record under a load that covers "
                    "the simulated one.",
                    f,
                    self.lo,
                    self.hi,
                )
            _knn_lookups[1] += 1
        q = (np.asarray(f, dtype=np.float64) - self.mean) / self.std
        x, y, z = self.columns
        dist = np.sqrt((x - q[0]) ** 2 + (y - q[1]) ** 2 + (z - q[2]) ** 2)
        idx = np.argpartition(dist, self.k - 1)[: self.k]
        d = dist[idx]
        if (d == 0.0).any():
            device_s, host_s = self.values[idx[d == 0.0]].mean(axis=0)
        else:
            w = 1.0 / d
            device_s, host_s = (w[:, None] * self.values[idx]).sum(axis=0) / w.sum()
        return StepCost(device_s=float(device_s), host_s=float(host_s))


def load_step_log(
    path: str, kind: Literal["step", "kv_transfers", "kv_post"] = "step"
) -> list[dict[str, Any]]:
    """The step records of a JSONL step log, or its records of completed KV
    transfers (`kv_transfers`) or NIXL read postings (`kv_post`)."""
    lines = Path(path).read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    if kind == "step":
        # Every other record is a kv_ one (kv_transfers, kv_post, kv_check, ...),
        # a `threads` one or the `timing` one.
        return [
            r
            for r in records
            if "threads" not in r
            and "timing" not in r
            and not any(key.startswith("kv_") for key in r)
        ]
    return [r for r in records if kind in r]


def _record_cost(rec: dict[str, Any]) -> tuple[float, float]:
    """(device_s, host_s) of a step record, or of a hand-written one."""
    cost = rec.get("cost", rec)
    return float(cost["device_s"]), float(cost.get("host_s") or 0.0)


def _record_features(rec: dict[str, Any]) -> tuple[float, float, float]:
    """`_step_features` of a step record, or of a hand-written one:
    {"num_reqs", "num_tokens", "sum_past_kv", "device_s"[, "host_s"]}."""
    if "step" in rec:
        return _step_features(StepDescriptor.from_json_dict(rec["step"]))
    return (
        float(rec["num_reqs"]),
        float(rec["num_tokens"]),
        float(rec["sum_past_kv"]),
    )


def acceptance_rates_from_log(path: str) -> list[float]:
    """Per-position draft acceptance rates of a step log recorded with
    speculative decoding, each conditional on the earlier drafts being
    accepted."""
    reached = np.zeros(0)
    accepted = np.zeros(0)
    for rec in load_step_log(path):
        num_accepted = rec.get("cost", {}).get("num_accepted")
        if not num_accepted:
            continue
        for req, n in zip(rec["step"]["requests"], num_accepted):
            drafts = req["num_spec_tokens"]
            if drafts > len(reached):
                reached = np.pad(reached, (0, drafts - len(reached)))
                accepted = np.pad(accepted, (0, drafts - len(accepted)))
            n = min(n, drafts)
            reached[: min(n + 1, drafts)] += 1
            accepted[:n] += 1
    if not reached.any():
        raise ValueError(f"{path} has no speculative decoding steps")
    return (accepted / np.maximum(reached, 1)).tolist()


class TableTimingProvider:
    """Interpolates recorded steps (`path`, a step log or a hand-written JSONL
    table)."""

    def __init__(
        self, vllm_config: VllmConfig, *, path: str, k: int = _DEFAULT_KNN_K
    ) -> None:
        records = load_step_log(path)
        self.table = _KNNTable(
            [(_record_features(r), _record_cost(r)) for r in records], k
        )
        logger.info("Loaded %d timing records from %s", len(records), path)

    def estimate(self, step: StepDescriptor) -> StepCost:
        return self.table.query(_step_features(step))


class ReplayTimingProvider:
    """Replays the mean recorded time of a step's exact batch composition; on a
    miss, interpolates (``fallback="knn"``) or returns
    `miss_fallback_s`."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        path: str,
        fallback: str = "knn",
        k: int = _DEFAULT_KNN_K,
        miss_fallback_s: float = 0.0,
    ) -> None:
        records = load_step_log(path)
        # Steps of one composition, recorded more than once, vary run to run:
        # replay their mean.
        same_key: dict[str, list[tuple[float, float]]] = {}
        points = []
        for rec in records:
            cost = _record_cost(rec)
            if "step" in rec:
                key = StepDescriptor.from_json_dict(rec["step"]).replay_key()
                same_key.setdefault(key, []).append(cost)
            points.append((_record_features(rec), cost))
        self.exact: dict[str, tuple[float, float]] = {
            key: (
                math.fsum(d for d, _ in costs) / len(costs),
                math.fsum(h for _, h in costs) / len(costs),
            )
            for key, costs in same_key.items()
        }
        if fallback not in ("knn", "zero"):
            raise ValueError(f"unknown replay fallback {fallback!r}")
        self.fallback = fallback
        self.miss_fallback_s = miss_fallback_s
        self.table = _KNNTable(points, k) if fallback == "knn" else None
        logger.info(
            "Loaded %d replay records (%d exact keys) from %s",
            len(records),
            len(self.exact),
            path,
        )

    def estimate(self, step: StepDescriptor) -> StepCost:
        cost = self.exact.get(step.replay_key())
        if cost is not None:
            device_s, host_s = cost
            return StepCost(
                device_s=device_s, host_s=host_s, breakdown={"replay_exact": 1.0}
            )
        if self.table is not None:
            return dataclasses.replace(
                self.table.query(_step_features(step)),
                breakdown={"replay_exact": 0.0},
            )
        return StepCost(device_s=self.miss_fallback_s, breakdown={"replay_exact": 0.0})


BUILTIN_TIMING_PROVIDERS: dict[str, type] = {
    "zero": ZeroTimingProvider,
    "linear": LinearTimingProvider,
    "roofline": RooflineTimingProvider,
    "table": TableTimingProvider,
    "replay": ReplayTimingProvider,
}


def get_timing_provider(
    name: str, args: dict[str, Any], vllm_config: VllmConfig
) -> TimingProvider:
    cls = BUILTIN_TIMING_PROVIDERS.get(name)
    if cls is None:
        if "." not in name:
            raise ValueError(
                f"Unknown timing provider {name!r}. Built-ins: "
                f"{sorted(BUILTIN_TIMING_PROVIDERS)}; otherwise pass a "
                "fully-qualified class name."
            )
        cls = resolve_obj_by_qualname(name)
    provider = cls(vllm_config, **args)
    if not isinstance(provider, TimingProvider):
        raise TypeError(f"{name} does not implement TimingProvider.estimate")
    return provider


class Clock(Protocol):
    horizon: float
    """When the driver of an in-process engine next has work for it (a
    request arrival); an engine that is only waiting may skip ahead to it."""

    def now(self) -> float: ...

    def sleep_until(self, deadline: float) -> None: ...


class RealClock:
    """Wall clock: simulated steps take real time, so every existing metric,
    benchmark client and router sees realistic latencies."""

    horizon = math.inf

    def now(self) -> float:
        return time.monotonic()

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class VirtualClock:
    """Simulated time: waiting for a deadline moves the clock to it."""

    def __init__(self) -> None:
        self._now = 0.0
        self.horizon = math.inf

    def now(self) -> float:
        return self._now

    def sleep_until(self, deadline: float) -> None:
        self._now = max(self._now, deadline)


class FastForwardClock:
    """The wall clock with waits skipped: waiting for a deadline shifts the
    clock forward instead of sleeping."""

    def __init__(self) -> None:
        self._offset = 0.0
        self.horizon = math.inf

    def now(self) -> float:
        return time.monotonic() + self._offset

    def sleep_until(self, deadline: float) -> None:
        self._offset += max(0.0, deadline - self.now())


_CLOCKS: dict[str, type[Clock]] = {
    "real": RealClock,
    "virtual": VirtualClock,
    "fast_forward": FastForwardClock,
}
# The clock of the engine being built in the process of its driver.
_driven_engine_clock: Clock | None = None


@contextlib.contextmanager
def building_driven_engine(clock: Clock) -> Iterator[None]:
    """Engines built in this context share `clock` with their driver."""
    global _driven_engine_clock
    _driven_engine_clock = clock
    try:
        yield
    finally:
        _driven_engine_clock = None


def driven_engine_clock() -> Clock | None:
    """The clock of the driven engine being built, if any."""
    return _driven_engine_clock


def new_clock(kind: str) -> Clock:
    """A new clock of `SimulationConfig.clock` kind `kind`."""
    return _CLOCKS[kind]()


class SimulatedAsyncOutput(AsyncModelRunnerOutput):
    """Defers a step's completion to `get_output()`, so the engine core
    schedules the next step meanwhile, as under async scheduling."""

    def __init__(self, output: ModelRunnerOutput, deadline: float, clock: Clock):
        self.output = output
        self.deadline = deadline
        self.clock = clock
        self.resolved = False

    def get_output(self) -> ModelRunnerOutput:
        self.clock.sleep_until(self.deadline)
        self.resolved = True
        return self.output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        return self.output.kv_connector_output


def is_output_rank() -> bool:
    """Whether this rank returns the step output: the last pipeline stage's
    first tensor-parallel rank."""
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return get_pp_group().is_last_rank and get_tp_group().rank_in_group == 0


def step_log_path(vllm_config: VllmConfig) -> str | None:
    """This rank's step log, if it writes one: the output rank's, of DP rank 0
    under MoE DP (which logs joint steps), else `<log_path>.dp<i>` for an
    independent DP rank i > 0. Under pipeline parallelism a recorded step is
    timed on the last stage, so its device time spans the whole pipeline.
    With real KV transfers every tensor-parallel rank moves and checks its
    own KV heads: rank r > 0 logs into `<log_path>.tp<r>`."""
    from vllm.distributed.parallel_state import get_tp_group

    path = vllm_config.simulation_config.log_path
    if path is None:
        return None
    if not is_output_rank():
        if vllm_config.simulation_config.kv_transfer_mode != "real":
            return None
        from vllm.distributed.parallel_state import get_pp_group

        if not get_pp_group().is_last_rank:
            return None
        root, ext = os.path.splitext(path)
        return f"{root}.tp{get_tp_group().rank_in_group}{ext}"
    parallel_config = vllm_config.parallel_config
    dp_index = parallel_config.data_parallel_index
    if parallel_config.data_parallel_size > 1:
        return path if parallel_config.data_parallel_rank == 0 else None
    if dp_index > 0:
        root, ext = os.path.splitext(path)
        path = f"{root}.dp{dp_index}{ext}"
    return path


class StepLogWriter:
    """Appends one JSON line per simulated step, plus one per step in which
    KV connector transfers completed. The step records feed the `table` and
    `replay` providers, the transfer records `kv_transfer_timing`."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def write(
        self,
        step: StepDescriptor,
        cost: StepCost,
        t_start: float,
        t_deadline: float,
        finished_req_ids: list[str],
    ) -> None:
        rec = {
            "step": step.to_json_dict(),
            "cost": asdict(cost),
            "t_start": t_start,
            "t_deadline": t_deadline,
            "finished_req_ids": finished_req_ids,
        }
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_transfers(self, transfers: dict[str, dict[str, float]], t: float) -> None:
        """Append the KV transfers that completed in a step, per direction:
        their count, bytes and summed duration."""
        rec = {"kv_transfers": transfers, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_kv_checks(self, results: list[dict], t: float) -> None:
        """Append the byte checks of received KV blocks (one record each)."""
        for result in results:
            rec = {"kv_check": result, "t": t}
            self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if results:
            self._f.flush()

    def write_kv_events(
        self, events: dict[str, object], t: float, t_scheduled: float
    ) -> None:
        """Append a step's request lifecycle events (`scheduled_kv_events`,
        `ReadsInFlight.events`), leaving out empty ones. The scheduler's took
        effect by `t_scheduled`, the step's start; the connector's (reads
        posted, transfers ended, notifications sent) by `t`, its end."""
        events = {kind: ids for kind, ids in events.items() if ids}
        if events:
            rec = {"kv_events": events, "t_scheduled": t_scheduled, "t": t}
            self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._f.flush()

    def write_threads(self, threads: dict[int, str], t: float) -> None:
        """Append the process's Python threads that started since the last
        such record, by native id: a profiler sees only their ids."""
        rec = {"threads": {str(tid): name for tid, name in threads.items()}, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_timing(self, timing: dict[str, object], t: float) -> None:
        """Append the timing provider and what its step times rest on
        (`timing`), once, as the runner starts."""
        rec = {"timing": timing, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_kv_layout(self, layout: dict[str, object], t: float) -> None:
        """Append the KV cache as the connector registers it (`kv_layout`)."""
        rec = {"kv_layout": layout, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def write_kv_post(self, reads: int, runs: int, seconds: float, t: float) -> None:
        """Append the NIXL reads a decode rank posted in a step: their count,
        runs of consecutive blocks (see `PostCost`) and the time posting took,
        which the step's own record leaves out."""
        rec = {"kv_post": {"reads": reads, "runs": runs, "seconds": seconds}, "t": t}
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
