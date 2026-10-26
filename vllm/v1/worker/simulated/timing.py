# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step timing for simulated execution: a `TimingProvider` prices a
`StepDescriptor` as a `StepCost`."""

from __future__ import annotations

import contextlib
import dataclasses
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
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

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
# KV cache dtypes packing elements below a byte, or with per-token scales:
# their size is not their storage dtype's.
PACKED_KV_DTYPES = ("int4", "nvfp4", "turboquant", "int8_per_token_head", "fp8_per_token_head")


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
class LayerGeometry:
    """One decoder layer's matrix multiplications on this rank."""

    dense_params: int
    """Weights every token goes through: attention projections, dense MLP,
    shared experts, router."""
    experts: tuple[ExpertGeometry, ...] = ()


# How an MoE step's expert reads are estimated, as recorded with the geometry.
ROUTING_MODEL = {
    "model": "independent_uniform_top_k",
    "experts_read_per_layer": (
        "local_experts * (1 - (1 - top_k / global_experts) ** step_tokens)"
    ),
    "flops_per_token_per_layer": "2 * (dense_params + top_k * params_per_expert)",
    "not_modeled": ["routing skew (error direction unknown)", "expert parallelism"],
}


class UnsupportedModelLayout(ValueError):
    """The built model's decoder layers cannot be told apart."""


def _group_layers(layers: tuple[LayerGeometry, ...]) -> list[dict[str, Any]]:
    """Consecutive layers of one geometry, as ``first``-``last`` ranges."""
    groups: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        if groups and layers[groups[-1]["last"]] == layer:
            groups[-1]["last"] = index
        else:
            groups.append(
                {
                    "first": index,
                    "last": index,
                    "dense_params": layer.dense_params,
                    "experts": [asdict(e) for e in layer.experts],
                }
            )
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
    layers: tuple[LayerGeometry, ...] | None = None
    """Each decoder layer's matrix multiplications, from the built model;
    None: `layer_params` for every layer (a dense model's config)."""

    @classmethod
    def from_config(cls, vllm_config: VllmConfig) -> ModelGeometry:
        """A dense model's geometry from its config."""
        if vllm_config.model_config.is_moe:
            raise ValueError(
                "The roofline timing provider needs an MoE model's built "
                "layers: ModelGeometry.from_model."
            )
        return cls._from_config(vllm_config)

    @classmethod
    def from_model(cls, vllm_config: VllmConfig, model: Any) -> ModelGeometry:
        """The geometry of the built model (simulation builds it on the meta
        device, with this rank's shapes): each decoder layer's matrix
        multiplications at their logical sizes, the routed experts' apart.
        Embeddings are lookups and norms no matrix multiplications; the LM
        head stays `vocab_size`, run for the sampled tokens only."""
        import torch

        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase

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
            dense, experts = 0, []
            # modules() yields a module referenced twice (a shared expert's
            # gate, say) once.
            for module in layer.modules():
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
                elif isinstance(module, LinearBase):
                    inputs = getattr(
                        module, "input_size_per_partition", module.input_size
                    )
                    outputs = getattr(module, "output_partition_sizes", None) or [
                        getattr(module, "output_size_per_partition", module.output_size)
                    ]
                    dense += inputs * sum(outputs)
                elif isinstance(module, torch.nn.Linear):
                    dense += module.in_features * module.out_features
            layers.append(LayerGeometry(dense, tuple(experts)))
        if any(layer.experts for layer in layers):
            cls._check_moe(vllm_config)
        return dataclasses.replace(geometry, layers=tuple(layers))

    @staticmethod
    def _check_moe(vllm_config: VllmConfig) -> None:
        import torch

        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        if parallel.world_size_across_dp > 1 or parallel.enable_expert_parallel:
            raise ValueError(
                "The roofline timing provider models MoE models on one rank "
                "(no TP, PP, DP or EP) so far."
            )
        if model.quantization is not None or model.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError(
                "The roofline timing provider models MoE models with "
                f"unquantized bf16 or fp16 weights, not {model.quantization or model.dtype}."
            )

    @classmethod
    def _from_config(cls, vllm_config: VllmConfig) -> ModelGeometry:
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        if model.use_mla or model.is_hybrid or model.is_attention_free:
            raise ValueError(
                "The roofline timing provider models dense and MoE attention "
                "models (no MLA, hybrid or attention-free ones)."
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
        `top_k` of its experts."""
        if self.layers is None:
            return self.num_layers * self.layer_params
        return sum(
            layer.dense_params
            + sum(e.top_k * e.params_per_expert for e in layer.experts)
            for layer in self.layers
        )

    def weights_read(self, num_tokens: int) -> float:
        """Decoder weights a step of `num_tokens` tokens reads: every layer's
        dense ones and the experts its tokens are expected to route to."""
        if self.layers is None:
            return self.num_layers * self.layer_params
        return sum(
            layer.dense_params
            + sum(e.expected_read(num_tokens) * e.params_per_expert for e in layer.experts)
            for layer in self.layers
        )

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
        """K and V of one token across all layers on this rank."""
        return self.num_layers * 2 * self.num_kv_heads * self.head_size * self.kv_bytes


class RooflineTimingProvider:
    """Roofline model of a dense or MoE transformer, one GPU per
    tensor-parallel rank (MoE: one rank): a step takes ``fixed_us`` plus the
    longer of its FLOPs at ``peak_tflops * compute_efficiency`` and its bytes
    at ``hbm_tbps * bandwidth_efficiency``.

    FLOPs: 2 per weight per token through the decoder layers (an MoE
    layer's dense weights and ``top_k`` of its experts), the LM head for the
    tokens sampled, and attention (QK^T and AV) over each request's context.
    Bytes: the dense weights once per step, the experts the step's tokens
    are expected to route to (each token independently and uniformly), the
    KV each request reads (its context) and writes. Not modeled:
    activations, communication between ranks, kernel efficiency varying with
    batch shape, routing skew (fewer experts read, but uneven work: the
    error's direction is unknown). ``gpu`` picks a preset of `GPU_PRESETS`;
    ``peak_tflops`` and ``hbm_tbps`` override it. An MoE model's geometry
    comes from its built layers (`bind_model`)."""

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
        self._vllm_config = vllm_config
        # An MoE model's geometry comes from its built layers (bind_model).
        self.geometry: ModelGeometry | None = geometry
        if geometry is None and not vllm_config.model_config.is_moe:
            self.geometry = ModelGeometry.from_config(vllm_config)
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
        keeps its config's; an MoE model needs its layers."""
        try:
            self.geometry = ModelGeometry.from_model(self._vllm_config, model)
        except UnsupportedModelLayout as error:
            if self._vllm_config.model_config.is_moe:
                raise
            logger.warning("%s Timing the dense model from its config.", error)
            self.geometry = ModelGeometry.from_config(self._vllm_config)

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
        return {**self._assumptions, "geometry": self._bound().describe()}

    def estimate(self, step: StepDescriptor) -> StepCost:
        g = self._bound()
        sampled = sum(
            1
            for r in step.requests
            if r.num_computed_tokens + r.num_scheduled_tokens >= r.num_prompt_tokens
        )
        # Query-key pairs: each new token attends to the context and to the
        # new tokens before it.
        pairs = sum(
            r.num_scheduled_tokens * r.num_computed_tokens
            + r.num_scheduled_tokens * (r.num_scheduled_tokens + 1) // 2
            for r in step.requests
        )
        context = sum(r.num_computed_tokens + r.num_scheduled_tokens for r in step.requests)
        lm_head = g.hidden_size * g.vocab_size
        flops = (
            2 * g.token_params * step.num_tokens
            + 2 * lm_head * sampled
            + 4 * g.num_layers * g.num_heads * g.head_size * pairs
        )
        memory = (g.weights_read(step.num_tokens) + lm_head) * g.weight_bytes
        memory += (context + step.num_tokens) * g.kv_bytes_per_token  # read and write
        compute_us = flops / self.flops_per_s * 1e6
        memory_us = memory / self.bytes_per_s * 1e6
        return StepCost(
            device_s=(self.fixed_us + max(compute_us, memory_us)) * 1e-6,
            host_s=self.host.host_s(step),
            breakdown={"compute_us": compute_us, "memory_us": memory_us},
        )


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


def is_output_rank() -> bool:
    """Whether this rank returns the step output: the last pipeline stage's
    first tensor-parallel rank."""
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    return get_pp_group().is_last_rank and get_tp_group().rank_in_group == 0


def step_log_path(vllm_config: VllmConfig) -> str | None:
    """This rank's step log, if it writes one: the output rank's, of DP rank 0
    under MoE DP (which logs joint steps), else `<log_path>.dp<i>` for an
    independent DP rank i > 0. Under pipeline parallelism a recorded step is
    timed on the last stage, so its device time spans the whole pipeline."""
    path = vllm_config.simulation_config.log_path
    if path is None or not is_output_rank():
        return None
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
