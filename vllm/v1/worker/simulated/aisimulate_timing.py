# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step times from AISimulate (https://github.com/ai-dynamo/aisimulate).

AISimulate's forward-pass performance model prices one step of a published
model on a GPU, from kernels measured on that GPU under a vLLM release. This
provider asks it for every simulated step: the scheduler's real batch, one
workload per attention data-parallel rank. The simulated deployment it
describes is the published one (`target_model`, with its published weight
precision); the local model serves the KV cache and the scheduler only, so
their KV and state geometry must agree.

The package is optional and imported only when this provider is chosen.
"""

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.config.simulation import DSA_FP8_CACHE_DTYPES
from vllm.logger import init_logger
from vllm.v1.worker.simulated.timing import (
    HostTimeModel,
    SimRequestStep,
    StepCost,
    StepDescriptor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# Our GPU presets as AISimulate systems (SXM parts).
GPU_SYSTEMS = {
    "a100": "a100_sxm",
    "h100": "h100_sxm",
    "h200": "h200_sxm",
    "b200": "b200_sxm",
}
# Tensor-parallel sizes of one modeled NVLink domain, and its GPUs: the
# data-parallel ranks times each one's tensor-parallel size.
TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)
MAX_LOCAL_GPUS = 8
# A published checkpoint's quantization (AISimulate's own inference from its
# config) as the estimator's modes. Attention runs in BF16 in each.
QUANT_MODES: dict[str | None, dict[str, str]] = {
    None: {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
    },
    "fp8_block": {
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": "bfloat16",
    },
    "mxfp4": {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "w4a16_mxfp4",
        "fmha_quant_mode": "bfloat16",
    },
}
# Where vLLM's online quantization methods live (weights quantized at load).
ONLINE_QUANT_MODULES = "vllm.model_executor.layers.quantization.online."
# Online quantization (`--quantization`) AISimulate times, on an unquantized
# checkpoint: per-tensor FP8 weights with dynamic activations (its `fp8`),
# 128-block FP8 (`fp8_block`); the methods the layers are then built with.
ONLINE_QUANT_MODES: dict[str, tuple[dict[str, str], frozenset[str]]] = {
    **dict.fromkeys(
        ("fp8", "fp8_per_tensor"),
        (
            {"gemm_quant_mode": "fp8", "moe_quant_mode": "fp8"},
            frozenset(
                {"Fp8PerTensorOnlineLinearMethod", "Fp8PerTensorOnlineMoEMethod"}
            ),
        ),
    ),
    "fp8_per_block": (
        {"gemm_quant_mode": "fp8_block", "moe_quant_mode": "fp8_block"},
        frozenset({"Fp8PerBlockOnlineLinearMethod", "Fp8PerBlockOnlineMoEMethod"}),
    ),
}
# Quantizations whose modes are the checkpoint's own (`_inferred_modes`).
INFERRED_QUANT_ALGOS = ("compressed-tensors", "nvfp4", "mixed_precision", "fp8")
QUANT_MODE_FIELDS = (
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
)
# A KV cache dtype as AISimulate's KV mode ("auto": the model dtype's).
KV_CACHE_MODES = {
    "bfloat16": "bfloat16",
    "float16": "bfloat16",  # two bytes, as BF16
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
}
# Architectures whose KV and state geometry is compared field by field;
# others are refused, their compatibility not established.
ATTENTION_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "sliding_window",
    "layer_types",
)
# MLA caches one latent per token and layer: its widths, not a head size.
MLA_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
)
# Kimi-K3 (Kimi-Linear): MLA layers' latent widths (without RoPE when
# mla_use_nope) beside KDA layers, by its linear attention config (the KDA
# layers, their heads, head width and short convolution).
KIMI_K3_FIELDS = MLA_FIELDS + ("mla_use_nope", "linear_attn_config")
# DSA sparse MLA (DeepSeek-V3.2, GLM-5): the latent's widths and its
# lightning indexer's (heads, key width, top-k, which layers hold one).
DSA_FIELDS = MLA_FIELDS + (
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "index_topk_freq",
    "index_skip_topk_offset",
    "index_topk_pattern",
)
MAMBA2_FIELDS = (
    "hybrid_override_pattern",
    "mamba_num_heads",
    "mamba_head_dim",
    "n_groups",
    "ssm_state_size",
    "conv_kernel",
)
# Gemma4: sliding layers of head_dim x num_key_value_heads, global ones of
# global_head_dim x num_global_key_value_heads (one K for V when
# attention_k_eq_v), no layer sharing another's KV.
GEMMA4_FIELDS = ATTENTION_FIELDS + (
    "global_head_dim",
    "num_global_key_value_heads",
    "attention_k_eq_v",
    "num_kv_shared_layers",
    # Each layer's own (head_dim, num_key_value_heads), as a heterogeneous
    # config resolves them (transformers keeps the global ones per layer).
    "per_layer_kv",
)
# MiMo-V2-Flash: full and sliding layers of their own KV heads and K/V widths,
# by its hybrid pattern, the sliding ones with sink biases.
MIMO_V2_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "v_head_dim",
    "hybrid_layer_pattern",
    "sliding_window",
    "swa_num_key_value_heads",
    "swa_head_dim",
    "swa_v_head_dim",
    "add_swa_attention_sink_bias",
    "add_full_attention_sink_bias",
)
# DeciLM (Llama-3.3-Nemotron-Super): each layer's attention its own, by its
# block config: none (no_op), a linear map, or attention of
# num_attention_heads / n_heads_in_group KV heads (a window, sinks).
DECILM_FIELDS = ("num_hidden_layers", "num_attention_heads", "hidden_size", "block_attention")
# Qwen's gated delta-net: its state's widths (the convolution's and the
# recurrent state's).
GDN_FIELDS = (
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)
# DeepSeek-V4: its sparse attention's widths, and the compress ratio of each
# decoder layer (compared apart, as the native helper resolves them).
V4_FIELDS = (
    "num_hidden_layers",
    "num_attention_heads",
    "head_dim",
    "qk_rope_head_dim",
    "sliding_window",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "o_groups",
)
V4_COMPRESS_RATIOS = (0, 1, 4, 128)
# DeepSeek-V4's cache is the packed FP8 latent (`fp8_ds_mla`), which vLLM's
# supported path resolves `auto` to (and `fp8`); AISimulate's V4 tables are
# FP8 KV only.
V4_KV_CACHE_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla")
V4_APPROXIMATION = (
    "AISimulate prices DeepSeek-V4's pure sliding-window layers (compress "
    "ratio 0) with its HCA (ratio 128) module tables; its capacity model keeps "
    "the real ratios"
)
# Speculative decoding: DeepSeek-V3's and DeepSeek-V4's native single-module
# MTP only (each target with its own draft architecture), as AISimulate's MTP
# depth (`nextn`, 1 to 5).
MTP_TARGETS = {
    "DeepseekV3ForCausalLM": "DeepSeekMTPModel",
    "DeepseekV4ForCausalLM": "DeepSeekV4MTPModel",
    "Qwen3_5ForConditionalGeneration": "Qwen3_5MTP",
    "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeMTP",
    "DeepseekV32ForCausalLM": "DeepseekV32MTPModel",
    "GlmMoeDsaForCausalLM": "DeepseekV32MTPModel",
}
# Each draft's model type as the speculative config builds it.
MTP_DRAFT_TYPES = {
    "DeepSeekMTPModel": "deepseek_mtp",
    "DeepSeekV4MTPModel": "deepseek_mtp",
    "Qwen3_5MTP": "qwen3_5_mtp",
    "Qwen3_5MoeMTP": "qwen3_5_mtp",
    "DeepseekV32MTPModel": "deepseek_mtp",
}
MTP_MAX_DEPTH = 5
MTP_APPROXIMATIONS = [
    (
        "a decode step verifies k + 1 tokens a request (the configured depth k, "
        "whatever drafts the step actually scheduled: a step with fewer is "
        "charged the full width)"
    ),
    (
        "the MTP drafter is k target-shaped layers of operations, not a measured "
        "drafter; its weights and KV, prefill-side draft work and host glue are "
        "omitted"
    ),
    (
        "MLA verification lacks the sequence-basis width treatment AISimulate "
        "gives other attention"
    ),
    "acceptance does not enter the cost",
    "graph-based kernel tables: an off-graph deployment is not modeled",
]
# A hybrid target's MTP (Qwen3.5): its native drafter is one full-attention
# layer, priced as k layers of the target's mix of attention and delta-nets.
# Plain-MLA context steps over a cached prefix (request 236's probe): in
# DeepSeek-V3 and Kimi-K3 the estimates fall as a positive prefix grows with
# the new tokens fixed, while generation estimates rise. Not corrected.
MLA_PREFIX_APPROXIMATION = (
    "the pinned AISimulate version has a cached-prefix timing limitation for "
    "plain-MLA context steps: in DeepSeek-V3 and Kimi-K3 probes, estimates "
    "decrease as an already-positive prefix grows with the new-token count "
    "fixed; prefix-cache hits, later prefill chunks and decode's resumed "
    "context step use this path; no timing correction is applied"
)
HYBRID_MTP_APPROXIMATION = (
    "the MTP drafter, a full-attention layer, is priced as the hybrid target's "
    "layers ((L + k) / L of its graph): its attention and delta-net mix, not "
    "the drafter's own"
)
GEOMETRY_FIELDS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        (
            "LlamaForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen3ForCausalLM",
            "Qwen2MoeForCausalLM",
            "Qwen3MoeForCausalLM",
            "MixtralForCausalLM",
            "GptOssForCausalLM",
        ),
        ATTENTION_FIELDS,
    ),
    **dict.fromkeys(("DeepseekV3ForCausalLM", "KimiK25ForConditionalGeneration"), MLA_FIELDS),
    "KimiK3ForConditionalGeneration": KIMI_K3_FIELDS,
    **dict.fromkeys(("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"), DSA_FIELDS),
    "DeepseekV4ForCausalLM": V4_FIELDS,
    "NemotronHForCausalLM": ATTENTION_FIELDS + MAMBA2_FIELDS,
    "Gemma4ForConditionalGeneration": GEMMA4_FIELDS,
    "MiMoV2FlashForCausalLM": MIMO_V2_FIELDS,
    "MuseGlimmerForConditionalGeneration": ATTENTION_FIELDS,
    "MiniMaxM2ForCausalLM": ATTENTION_FIELDS,
    "DeciLMForCausalLM": DECILM_FIELDS,
    **dict.fromkeys(
        ("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"),
        ATTENTION_FIELDS,
    ),
    **dict.fromkeys(
        ("Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration"),
        ATTENTION_FIELDS + GDN_FIELDS,
    ),
}
APPROXIMATIONS = [
    (
        "a data-parallel step takes its slowest rank's estimate, not per-layer "
        "synchronization or exact expert routing under uneven load"
    ),
    "the op-level estimator reduces a rank's requests to aggregate and average lengths",
]


def _mimo_v2_flash_timing_input(config: dict[str, Any]) -> dict[str, Any]:
    """MiMo-V2-Flash's hybrid_layer_pattern marks its sliding layers 1 (as its
    model reads it), AISimulate's hybrid graph its global ones: inverted, the
    estimator prices 9 full and 39 sliding layers, the model's own."""
    return {
        **config,
        "hybrid_layer_pattern": [1 - v for v in config["hybrid_layer_pattern"]],
    }


# A bundled config AISimulate would misread, by architecture: what it is
# given instead (a timing input only; the served model and the geometry
# compared are the published config's), and what was changed.
TIMING_INPUTS: dict[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "MiMoV2FlashForCausalLM": (
        _mimo_v2_flash_timing_input,
        "hybrid_layer_pattern inverted (MiMo marks sliding layers 1, AISimulate "
        "global ones)",
    ),
}


def _offline_model(
    target: str, bundled: str, timing_input: Callable | None = None
) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """The model AISimulate is given: its name when the package lists it as
    pre-downloaded; else a private snapshot of the package's own bundled
    config (with its quantization sidecar, or none when it bundles none),
    which it reads locally where it would otherwise download the published
    config. A config it would misread (`timing_input`) is always given as a
    snapshot of that function's copy. The snapshot is this caller's alone:
    keep the returned directory alive while AISimulate may read it."""
    from aisimulate_core.sdk.common import DefaultHFModels

    if target in DefaultHFModels and timing_input is None:
        return target, None
    snapshot = tempfile.TemporaryDirectory(prefix="aisimulate-")
    config = os.path.join(snapshot.name, "config.json")
    if timing_input is None:
        shutil.copyfile(bundled, config)
    else:
        with open(bundled) as f:
            normalized = timing_input(json.load(f))
        with open(config, "w") as f:
            json.dump(normalized, f, indent=2)
    sidecar = bundled.removesuffix("_config.json") + "_hf_quant_config.json"
    if os.path.isfile(sidecar):
        shutil.copyfile(sidecar, os.path.join(snapshot.name, "hf_quant_config.json"))
    return snapshot.name, snapshot


def _full_config(target: str) -> dict[str, Any]:
    """The target's config as AISimulate loads it: with its quantization
    sidecar (`hf_quant_config.json`), where a ModelOpt checkpoint keeps what
    its `config.json` may not say."""
    from aisimulate_core.sdk.utils import _load_model_config_from_model_path

    return _load_model_config_from_model_path(target)


def _inferred_modes(target: str) -> dict[str, str]:
    """A checkpoint's GEMM and expert modes as AISimulate infers them from its
    full config (BF16 where a layer kind is left unquantized): a
    compressed-tensors one's groups and ignore list, a ModelOpt one's NVFP4,
    per-tensor FP8 or mixed per-layer precision."""
    from aisimulate_core.sdk.models.helpers import _infer_quant_modes_from_raw_config

    inferred = _infer_quant_modes_from_raw_config(_full_config(target))

    def name(key: str) -> str:
        value = inferred.get(key)
        return getattr(value, "name", value) or "bfloat16"

    return {
        "gemm_quant_mode": name("gemm_quant_mode"),
        "moe_quant_mode": name("moe_quant_mode"),
        "fmha_quant_mode": "bfloat16",
    }


def _resolved(config: Any, name: str) -> Any:
    """A config field as its class resolves it; a heterogeneous config's
    per-layer field (Gemma4's KV heads), which it refuses to read globally or
    keeps per layer only, as the config holds it (the published value)."""
    try:
        value = getattr(config, name, None)
    except Exception as e:  # transformers' AmbiguousGlobalPerLayerAttributeError
        if type(e).__name__ != "AmbiguousGlobalPerLayerAttributeError":
            raise
        value = None
    if value is None and hasattr(config, "to_dict"):
        value = config.to_dict().get(name)
    return value


def _geometry(config: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """A config object's KV and state geometry, as its class resolves it. An
    attention head size its class leaves unset is the model's own fallback,
    hidden size over attention heads (Mixtral's, say)."""
    derived = ("per_layer_kv", "block_attention")
    geometry = {name: _resolved(config, name) for name in fields if name not in derived}
    if "block_attention" in fields:
        blocks = getattr(config, "block_configs", None)
        geometry["block_attention"] = (
            [
                tuple(
                    _resolved(block.attention, name)
                    for name in (
                        "no_op",
                        "replace_with_linear",
                        "n_heads_in_group",
                        "window_length",
                        "num_sink_tokens",
                    )
                )
                for block in blocks
            ]
            if blocks is not None
            else None
        )
    if "per_layer_kv" in fields:
        layers = getattr(config, "per_layer_config", None)
        geometry["per_layer_kv"] = (
            [
                (layer.head_dim, layer.num_key_value_heads)
                for layer in (layers[i] for i in range(config.num_hidden_layers))
            ]
            if layers is not None
            else None
        )
    if fields is V4_FIELDS:
        # No fallback: each of V4's fields must be resolved.
        if missing := [name for name, value in geometry.items() if value is None]:
            raise ValueError(f"DeepSeek-V4 geometry not resolved: {missing}.")
    elif "head_dim" in geometry and geometry["head_dim"] is None:
        geometry["head_dim"] = config.hidden_size // config.num_attention_heads
    return geometry


def _compress_ratios(config: Any) -> list[int]:
    """DeepSeek-V4's compress ratio of each decoder layer, as the native
    helper resolves them from either config spelling (MTP layers left out)."""
    from vllm.transformers_utils.deepseek_v4_utils import get_compress_ratios

    ratios = list(get_compress_ratios(config))[: config.num_hidden_layers]
    if len(ratios) != config.num_hidden_layers or any(
        r not in V4_COMPRESS_RATIOS for r in ratios
    ):
        raise ValueError(f"DeepSeek-V4 compress ratios not resolved: {ratios}.")
    return ratios


def _probe_request(
    scheduled: int, computed: int, prompt: int, prefill: bool
) -> SimRequestStep:
    return SimRequestStep(
        req_id="probe",
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_cached_tokens=0,
        num_spec_tokens=0,
        is_prefill=prefill,
    )


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _variance(values: list[int]) -> float:
    """Population variance; 0 for no values."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def rank_metrics(requests: list[SimRequestStep]) -> dict[str, Any]:
    """One attention data-parallel rank's scheduled work, as AISimulate's
    forward-pass metrics. A prefill request (a chunk, or the native N - 1
    handoff's single token) brings its scheduled tokens on top of its
    computed ones; a decode request one token on top of its context."""
    prefill = [r for r in requests if r.is_prefill]
    decode = [r for r in requests if not r.is_prefill]
    return {
        "scheduled_requests": {
            "num_prefill_requests": len(prefill),
            "sum_prefill_tokens": sum(r.num_scheduled_tokens for r in prefill),
            "var_prefill_length": _variance([r.num_prompt_tokens for r in prefill]),
            "sum_prefill_kv_tokens": sum(r.num_computed_tokens for r in prefill),
            "num_decode_requests": len(decode),
            "sum_decode_kv_tokens": sum(r.num_computed_tokens for r in decode),
            "var_decode_kv_tokens": _variance([r.num_computed_tokens for r in decode]),
            "extend_lengths": [r.num_scheduled_tokens for r in requests],
            "past_kv_lengths": [r.num_computed_tokens for r in requests],
        }
    }


class AisimulateTimingProvider:
    """Step times from AISimulate's forward-pass performance model (op-level,
    measured kernels; no analytical fallback) for the published deployment
    `target_model` on `gpu`. Host time is `host_us` per step.

    Supported: tensor parallelism within one node (TP 1, 2, 4 or 8), or
    an MoE model's local data-parallel attention with expert parallelism
    (each rank tensor-parallel or not, up to 8 GPUs); decode context
    parallelism of a dense MLA model inside its tensor-parallel group,
    without data parallelism, with vLLM's default merge (`ag_rs`, no Q
    replication). Anything else is refused, as are pipeline and prefill
    context parallelism, EPLB and speculative decoding other than
    DeepSeek-V3's or DeepSeek-V4's native single-module MTP. A multimodal
    model is timed as its language model, served without multimodal
    inputs."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        *,
        gpu: str = "h100",
        target_model: str | None = None,
        backend_version: str | None = None,
        quant_modes: dict[str, str] | None = None,
        host_us: float = 0.0,
    ) -> None:
        if not math.isfinite(host_us) or host_us < 0:
            raise ValueError(
                f"host_us must be a finite nonnegative number, got {host_us}."
            )
        self.host = HostTimeModel(fixed_us=host_us)
        system = GPU_SYSTEMS.get(gpu)
        if system is None:
            raise ValueError(
                f"The aisimulate timing provider models GPUs {sorted(GPU_SYSTEMS)}, "
                f"not {gpu!r}."
            )
        try:
            import aisimulate_core
            from aisimulate_core.sdk import (
                ForwardPassPerfModelConfig,
                RustForwardPassPerfModel,
            )
            from aisimulate_core.sdk.utils import _infer_quantization_fields
        except ImportError as e:
            raise ImportError(
                "The aisimulate timing provider needs the aisimulate package "
                "(pip install --no-deps aisimulate from pypi.nvidia.com)."
            ) from e

        # A multimodal model is timed as its language model: no input may
        # reach the encoders AISimulate does not time.
        if vllm_config.model_config.supports_multimodal_inputs:
            raise ValueError(
                "The aisimulate timing provider times a multimodal model's "
                "language model only: serve it with language_model_only (no "
                "modality, no multimodal embeddings)."
            )
        topology = self._topology(vllm_config)
        dcp = self._decode_context_parallel(vllm_config)
        target, local_config = self._target(vllm_config, target_model)
        bundled = os.path.join(
            os.path.dirname(aisimulate_core.__file__),
            "model_configs",
            f"{target.replace('/', '--')}_config.json",
        )
        if not os.path.isfile(bundled):
            raise ValueError(
                f"The aisimulate timing provider needs AISimulate's bundled config "
                f"of {target}, which this package does not have."
            )
        with open(bundled) as f:
            target_config = json.load(f)
        geometry = self._check_geometry(vllm_config, target_config, target)

        # One snapshot for the provider's lifetime: quant inference and the
        # estimator read the same files.
        timing_input, transformation = TIMING_INPUTS.get(
            (target_config.get("architectures") or [None])[0], (None, None)
        )
        model, self._snapshot = _offline_model(target, bundled, timing_input)
        v4 = target_config.get("architectures") == ["DeepseekV4ForCausalLM"]
        # Plain MLA: a latent (`kv_lora_rank`), no indexer, not V4's caches.
        text = target_config.get("text_config") or target_config
        plain_mla = (
            bool(text.get("kv_lora_rank")) and "index_topk" not in text and not v4
        )
        # The quantization of the full config: a ModelOpt checkpoint may
        # declare it in its sidecar only, and would read as unquantized.
        quant = _infer_quantization_fields(_full_config(model)).get("quant_algo")
        # V4's experts follow the raw config's `expert_dtype` (AISimulate's own
        # V4 override): FP4 experts with FP8 activations.
        expert_dtype = target_config.get("expert_dtype") if v4 else None
        modes = QUANT_MODES.get(quant) if not v4 else None
        if quant in INFERRED_QUANT_ALGOS:
            # Per-checkpoint precision (Kimi-K2.5's int4 experts beside BF16
            # GEMMs, a ModelOpt checkpoint's NVFP4 or FP8 layers): AISimulate's
            # own reading of the checkpoint's groups, layers and ignore list,
            # sidecars included; attention BF16 as elsewhere.
            modes = _inferred_modes(model)
        if v4 and (quant, expert_dtype) == ("fp8_block", "fp4"):
            modes = {
                "gemm_quant_mode": "fp8_block",
                "moe_quant_mode": "w4a8_mxfp4_mxfp8",
                "fmha_quant_mode": "bfloat16",
            }
        if modes is None:
            raise ValueError(
                f"The aisimulate timing provider does not map {target}'s "
                f"quantization {quant!r} (experts {expert_dtype!r}) to estimator "
                "modes."
            )
        modes = {**modes, "comm_quant_mode": "half"}
        # Weights this server quantizes itself as it loads them, of an
        # unquantized checkpoint: timed as that precision of the published
        # model. (A quantized checkpoint names its own method in the same
        # field; layers quantized online beside it are refused when built.)
        online = vllm_config.model_config.quantization
        self._online_methods: frozenset[str] = frozenset()
        if quant is None and online in ONLINE_QUANT_MODES:
            self._check_plain_online_quantization(vllm_config.model_config, online)
            online_modes, self._online_methods = ONLINE_QUANT_MODES[online]
            modes.update(online_modes)
        else:
            online = None
        if quant_modes:
            unknown = set(quant_modes) - set(QUANT_MODE_FIELDS)
            if unknown:
                raise ValueError(f"Unknown quant_modes {sorted(unknown)}.")
            modes.update(quant_modes)
        dsa = target_config.get("architectures") in (
            ["DeepseekV32ForCausalLM"],
            ["GlmMoeDsaForCausalLM"],
        )
        # GLM-5's `auto` cache is FP8 on a Blackwell GPU (SM100) and BF16 on
        # this host: which one is served, and timed, has to be said.
        if (
            target_config.get("architectures") == ["GlmMoeDsaForCausalLM"]
            and system == "b200_sxm"
            and vllm_config.cache_config.cache_dtype == "auto"
        ):
            raise ValueError(
                "GLM-5 on B200 resolves kv_cache_dtype auto to its packed FP8 "
                "sparse-MLA cache on the GPU and to BF16 on this host: pass "
                "kv_cache_dtype fp8 (the deployment's default) or bfloat16."
            )
        kv_format, kv_mode = self._kv_cache_mode(vllm_config, v4, dsa)

        nextn, speculation = self._speculation(vllm_config, target_config, topology)

        kv_role = getattr(vllm_config.kv_transfer_config, "kv_role", None)
        worker_type = {"kv_producer": "prefill", "kv_consumer": "decode"}.get(
            kv_role, "aggregated"
        )
        self.num_ranks = topology["attention_dp"]
        self.config = {
            "model": model,
            "system": system,
            "backend": "vllm",
            "worker_type": worker_type,
            "backend_version": backend_version,
            **topology,
            # None, not 1, without DCP: another identity of the estimator's.
            "dcp": dcp["size"] if dcp else None,
            "cp_size": 1,
            "enable_eplb": False,
            "nextn": nextn,
            **modes,
            "kvcache_quant_mode": kv_mode,
            "estimation_mode": "op_level",
            "database_mode": "SILICON",
            "fallback_policy": "deny",
        }
        self.model = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(**self.config)
        )
        diagnostics = self.model.diagnostics()
        if (
            diagnostics.get("readiness") != "ready"
            or diagnostics.get("source") == "fallback_regression"
        ):
            raise ValueError(
                f"AISimulate has no timing for {target} on {system} with "
                f"{topology}: {diagnostics.get('readiness')} "
                f"({diagnostics.get('last_warning')})."
            )
        # Op-level estimates only: AISimulate's interpolation and regression
        # modes are not taken, even when they would answer.
        mode = (diagnostics.get("provenance") or {}).get("config", {}).get(
            "estimation_mode"
        )
        if mode != "op_level":
            raise ValueError(
                f"AISimulate resolved {target} on {system} with {topology} to "
                f"estimation mode {mode!r}, not op_level."
            )
        probes = self._startup_probes(target, system, topology)
        try:
            version = importlib.metadata.version("aisimulate")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(aisimulate_core, "__version__", None)
        from vllm import __version__ as vllm_version

        resolved = (diagnostics.get("provenance") or {}).get("config", {})
        modified = bool(quant_modes) or online is not None
        self._assumptions = {
            "provider": "aisimulate",
            "package_version": version,
            "source": diagnostics.get("source"),
            "readiness": diagnostics.get("readiness"),
            "config": self.config,
            "resolved_config": resolved,
            "target_model": target,
            # The estimator's own bundled config, read offline.
            "target_config_source": (
                "bundled"
                if self.config["model"] == target
                else "bundled, normalized local copy"
                if transformation
                else "bundled, local copy"
            ),
            "target_config_sha256": _file_sha256(bundled),
            # What the estimator read instead of the bundled config, when it
            # would misread that: the change and the file it read.
            "timing_input": (
                {
                    "transformation": transformation,
                    "config_sha256": _file_sha256(
                        os.path.join(self.config["model"], "config.json")
                    ),
                }
                if transformation
                else None
            ),
            "target_quantization": quant,
            "online_quantization": online,
            "target_modified": modified,
            "local_config": local_config,
            "local_dtype": str(vllm_config.model_config.dtype).removeprefix("torch."),
            "kv_cache_dtype": vllm_config.cache_config.cache_dtype,
            "kv_cache_format": kv_format,
            "expert_dtype": expert_dtype,
            "startup_probes": probes,
            "decode_context_parallel": dcp,
            "speculation": speculation,
            "kv_cache_mode_note": (
                "an FP16 KV cache is timed as BF16 (two bytes; an approximation)"
                if kv_mode == "bfloat16" and self._kv_dtype(vllm_config) == "float16"
                else None
            ),
            "recurrent_state": self._recurrent_state(vllm_config),
            "compared_geometry": geometry,
            "deployment": (
                f"{'modified ' if modified else ''}published {target} "
                f"({quant or 'unquantized'} weights) on {system}, AISimulate "
                f"vLLM {resolved.get('backend_version') or backend_version} "
                f"data; the simulator runs vLLM {vllm_version}"
            ),
            "approximations": APPROXIMATIONS
            + ([V4_APPROXIMATION] if v4 else [])
            + ([MLA_PREFIX_APPROXIMATION] if plain_mla else [])
            + (MTP_APPROXIMATIONS if nextn else [])
            + (
                [HYBRID_MTP_APPROXIMATION]
                if nextn and getattr(vllm_config.model_config, "is_hybrid", False)
                else []
            ),
            "host_us": host_us,
        }
        logger.info("AISimulate timing: %s", self._assumptions["deployment"])

    @staticmethod
    def _check_plain_online_quantization(model_config: Any, online: str) -> None:
        """The modes are the shorthand's as vLLM resolves it alone, every
        linear layer and every expert quantized its way: a
        `--quantization-config` beside it (layers ignored, per-layer targets,
        another weight or activation precision) is another deployment."""
        from vllm.config.quantization import (
            QuantizationConfigArgs,
            resolve_quantization_config,
        )

        resolved = getattr(model_config, "quantization_config", None)
        # `fp8` resolves to no config; one that sets nothing is the same.
        plain = resolve_quantization_config(online, None) or QuantizationConfigArgs()
        if (resolved or QuantizationConfigArgs()) == plain:
            return
        raise ValueError(
            f"The aisimulate timing provider times --quantization {online} as "
            "vLLM resolves it alone, not with a quantization_config of its own "
            f"(ignored layers, targets or other precisions): {resolved}."
        )

    @staticmethod
    def _decode_context_parallel(vllm_config: "VllmConfig") -> dict[str, Any] | None:
        """Decode context parallelism as AISimulate prices it, None without:
        a dense MLA model's, inside its tensor-parallel group. The estimator
        takes the requests' global context lengths and stripes the decode
        KV, adding the merge collectives; its forward-pass config sets the
        DCP size alone, so it prices vLLM's default merge (all-gather and
        reduce-scatter, the query all-gathered), and no other."""
        from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import plain_mla

        pc = vllm_config.parallel_config
        dcp = pc.decode_context_parallel_size
        if dcp == 1:
            return None
        problems = []
        if not plain_mla(vllm_config.model_config):
            problems.append("a model other than dense MLA")
        if pc.data_parallel_size > 1:
            problems.append(f"data parallelism {pc.data_parallel_size}")
        if pc.dcp_comm_backend != "ag_rs":
            problems.append(f"dcp_comm_backend {pc.dcp_comm_backend}")
        if pc.dcp_q_replicate:
            problems.append("dcp_q_replicate")
        if problems:
            raise ValueError(
                f"The aisimulate timing provider times decode context parallelism "
                f"(decode_context_parallel_size {dcp}) for a dense MLA model "
                "with dcp_comm_backend ag_rs and dcp_q_replicate off, the "
                "estimator's forward-pass config setting the DCP size alone, "
                f"not with {', '.join(problems)}."
            )
        # The interleave is resolved after this record is taken (NIXL sets
        # it to the page); the runner records it (`_resolved_dcp_geometry`).
        return {
            "size": dcp,
            "comm_backend": pc.dcp_comm_backend,
            "q_replicate": bool(pc.dcp_q_replicate),
        }

    @staticmethod
    def _topology(vllm_config: "VllmConfig") -> dict[str, Any]:
        """The parallel layout, refused outside the supported scope."""
        pc = vllm_config.parallel_config
        problems = []
        if pc.pipeline_parallel_size > 1:
            problems.append("pipeline parallelism")
        if pc.prefill_context_parallel_size > 1:
            problems.append("prefill context parallelism")
        if pc.enable_eplb:
            problems.append("EPLB")
        if pc.nnodes > 1:
            problems.append("several nodes")
        tp, dp, ep = (
            pc.tensor_parallel_size,
            pc.data_parallel_size,
            pc.enable_expert_parallel,
        )
        moe = vllm_config.model_config.is_moe
        if dp > 1:
            if not (moe and ep):
                problems.append(
                    "data parallelism without an MoE model's expert parallelism"
                )
            # All on this host, whether one server launched them or a
            # multi-port supervisor a server for each (`host_dp_ranks`).
            from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
                host_dp_ranks,
            )

            local = host_dp_ranks(pc)[1]
            if local != dp or tp * dp > MAX_LOCAL_GPUS:
                problems.append(
                    f"{dp} data-parallel ranks of tensor parallelism {tp} "
                    f"({local} local; up to {MAX_LOCAL_GPUS} GPUs, all local)"
                )
        elif ep:
            problems.append("expert parallelism without data parallelism")
        if tp not in TENSOR_PARALLEL_SIZES:
            problems.append(f"tensor parallelism {tp} (not {TENSOR_PARALLEL_SIZES})")
        if problems:
            raise ValueError(
                "The aisimulate timing provider models tensor parallelism, alone "
                "or inside an MoE model's local data-parallel attention with "
                f"expert parallelism, not {', '.join(problems)}."
            )
        if dp > 1:
            # As vLLM lays the experts out: over every GPU of every rank.
            return {
                "tp": tp,
                "pp": 1,
                "attention_dp": dp,
                "moe_tp_size": 1,
                "moe_ep_size": tp * dp,
            }
        if moe:
            return {
                "tp": tp,
                "pp": 1,
                "attention_dp": 1,
                "moe_tp_size": tp,
                "moe_ep_size": 1,
            }
        return {
            "tp": tp,
            "pp": 1,
            "attention_dp": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
        }

    @staticmethod
    def _speculation(
        vllm_config: "VllmConfig", target_config: dict[str, Any], topology: dict
    ) -> tuple[int, dict[str, Any] | None]:
        """AISimulate's MTP depth for the speculative configuration: DeepSeek-V3's,
        DeepSeek-V4's or Qwen3.5's native single-module MTP (its own
        checkpoint's one MTP layer, the target's parallelism and precision)
        only."""
        spec = vllm_config.speculative_config
        if spec is None:
            return 0, None
        draft = getattr(spec.draft_model_config, "hf_config", None)
        k = spec.num_speculative_tokens
        problems = []
        if spec.method not in ("mtp", "deepseek_mtp"):
            problems.append(f"method {spec.method}")
        target = target_config.get("architectures", [None])[0]
        if target not in MTP_TARGETS:
            problems.append(f"target {target_config.get('architectures')}")
        # One MTP layer, which every step reuses (a checkpoint with several
        # is refused even at k = 1).
        # DeepSeek counts them as `num_nextn_predict_layers`, Qwen3.5 (in its
        # text config) as `mtp_num_hidden_layers`.
        text = target_config.get("text_config") or target_config

        def mtp_layers(config) -> int | None:
            return config.get("num_nextn_predict_layers") or config.get(
                "mtp_num_hidden_layers"
            )

        to_dict = getattr(draft, "to_dict", None)
        draft_config = to_dict() if to_dict else dict(vars(draft or object()))
        layers = {
            mtp_layers(text),
            mtp_layers(draft_config.get("text_config") or draft_config),
        }
        if layers != {1}:
            problems.append(f"{sorted(layers, key=str)} MTP layers (one)")
        draft_type = MTP_DRAFT_TYPES.get(MTP_TARGETS.get(target))
        if getattr(draft, "model_type", None) != draft_type:
            problems.append(f"draft {getattr(draft, 'model_type', None)}")
        architectures = list(
            getattr(spec.draft_model_config, "architectures", None) or []
        )
        if target in MTP_TARGETS and architectures != [MTP_TARGETS[target]]:
            problems.append(f"draft architectures {architectures}")
        if spec.model not in (None, vllm_config.model_config.model):
            problems.append(f"draft checkpoint {spec.model}")
        if spec.use_multi_module_mtp():
            problems.append("multi-module MTP")
        if not 1 <= k <= MTP_MAX_DEPTH:
            problems.append(f"{k} speculative tokens (1 to {MTP_MAX_DEPTH})")
        if spec.draft_tensor_parallel_size not in (None, topology["tp"]):
            problems.append(
                f"draft tensor parallelism {spec.draft_tensor_parallel_size}"
            )
        if spec.quantization not in (None, vllm_config.model_config.quantization):
            problems.append(f"draft quantization {spec.quantization}")
        # The draft's own precision choices: its KV cache (None inherits the
        # target's) and its copy of the target's LM head.
        if spec.kv_cache_dtype not in (None, vllm_config.cache_config.cache_dtype):
            problems.append(f"draft KV cache dtype {spec.kv_cache_dtype}")
        if getattr(spec, "draft_lm_head_quantization", None) is not None:
            problems.append(
                f"draft LM head quantization {spec.draft_lm_head_quantization}"
            )
        if problems:
            raise ValueError(
                "The aisimulate timing provider times DeepSeek-V3's, "
                "DeepSeek-V4's or Qwen3.5's native single-module MTP only, not "
                f"{', '.join(problems)}."
            )
        return k, {
            "method": spec.method,
            "num_speculative_tokens": k,
            "verify_width": k + 1,
            "draft_architectures": architectures,
        }

    @staticmethod
    def _target(
        vllm_config: "VllmConfig", target_model: str | None
    ) -> tuple[str, dict[str, Any]]:
        """The published model timed, and the local config's identity. A
        model directory `<org>--<name>/<commit>` names its own; a derived one
        (with `DERIVED.md`) must name its target."""
        model = vllm_config.model_config.model
        local: dict[str, Any] = {"model": model}
        config_path = os.path.join(model, "config.json")
        if os.path.isfile(config_path):
            local["config_sha256"] = _file_sha256(config_path)
        if target_model is not None:
            return target_model, local
        if not os.path.isdir(model):
            return model, local
        if os.path.exists(os.path.join(model, "DERIVED.md")):
            raise ValueError(
                f"{model} is derived from a published model (DERIVED.md): pass "
                "timing_args.target_model, the published model to time."
            )
        org, sep, name = os.path.basename(
            os.path.dirname(os.path.abspath(model))
        ).partition("--")
        if not sep or not org or not name:
            raise ValueError(
                f"Cannot tell which published model {model} is: pass "
                "timing_args.target_model."
            )
        return f"{org}/{name}", local

    @staticmethod
    def _check_geometry(
        vllm_config: "VllmConfig", target_config: dict[str, Any], target: str
    ) -> dict[str, Any]:
        """The local model's KV and state geometry (the engine's effective
        config) must be the target's."""
        architectures = target_config.get("architectures") or []
        local_architectures = vllm_config.model_config.architectures
        fields = GEOMETRY_FIELDS.get(architectures[0] if architectures else "")
        if fields is None or list(local_architectures) != list(architectures):
            raise ValueError(
                f"The aisimulate timing provider cannot establish KV compatibility "
                f"of the local {local_architectures} with {target} ({architectures})."
            )
        # The target's raw config through the local config's class: both
        # sides get the same defaults (a head size, layer types, KV heads).
        local_config = vllm_config.model_config.hf_text_config
        local = _geometry(local_config, fields)
        # A multimodal wrapper's language model is its text config.
        target_text = target_config.get("text_config", target_config)
        published = _geometry(type(local_config).from_dict(target_text), fields)
        if fields is V4_FIELDS:
            target_class_config = type(local_config).from_dict(target_config)
            for name, config in (
                ("local", local_config),
                ("target", target_class_config),
            ):
                ratios = _compress_ratios(config)
                (local if name == "local" else published)["compress_ratios"] = ratios
            fields = (*fields, "compress_ratios")
        differences = {
            name: (local[name], published[name])
            for name in fields
            if local[name] != published[name]
        }
        if differences:
            raise ValueError(
                f"The local model's KV or state geometry differs from {target}'s "
                f"(local, target): {differences}."
            )
        return local

    @staticmethod
    def _kv_dtype(vllm_config: "VllmConfig") -> str:
        dtype = vllm_config.cache_config.cache_dtype
        if dtype == "auto":
            dtype = str(vllm_config.model_config.dtype).removeprefix("torch.")
        return dtype

    @staticmethod
    def _recurrent_state(vllm_config: "VllmConfig") -> dict[str, Any] | None:
        """A hybrid model's recurrent state precision, which AISimulate's
        configuration does not take: recorded, not timed as such."""
        if not getattr(vllm_config.model_config, "is_hybrid", False):
            return None
        cache = vllm_config.cache_config
        return {
            "mamba_cache_dtype": getattr(cache, "mamba_cache_dtype", None),
            "mamba_ssm_cache_dtype": getattr(cache, "mamba_ssm_cache_dtype", None),
            "note": (
                "configuration flags (possibly auto), not resolved tensor dtypes; "
                "AISimulate takes no state precision and times its own"
            ),
        }

    @classmethod
    def _kv_cache_mode(
        cls, vllm_config: "VllmConfig", v4: bool, dsa: bool = False
    ) -> tuple[str, str]:
        """The KV cache's effective format and AISimulate's KV mode for it.
        The provider is built before the model resolves `auto`: DeepSeek-V4's
        is resolved here as its supported packed path does, and a DSA
        model's FP8 cache, under any of its names, is FlashMLA's packed one
        (a dense MLA's FP8 cache stays a byte an element)."""
        if v4:
            configured = vllm_config.cache_config.cache_dtype
            if configured not in V4_KV_CACHE_DTYPES:
                raise ValueError(
                    f"The aisimulate timing provider times DeepSeek-V4 with its "
                    f"packed FP8 cache (cache dtype {V4_KV_CACHE_DTYPES}), not "
                    f"{configured}."
                )
            return "fp8_ds_mla", "fp8"
        if dsa and cls._kv_dtype(vllm_config) in DSA_FP8_CACHE_DTYPES:
            return "fp8_ds_mla", "fp8"
        dtype = cls._kv_dtype(vllm_config)
        mode = KV_CACHE_MODES.get(dtype)
        if mode is None:
            raise ValueError(
                f"The aisimulate timing provider maps KV cache dtypes "
                f"{sorted(KV_CACHE_MODES)}, not {dtype}."
            )
        return dtype, mode

    def _startup_probes(
        self, target: str, system: str, topology: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """A prefill and a decode step at startup, on rank 0 (the others
        idle): readiness alone does not show the estimator has data for the
        target's shapes. Only these shapes are established."""
        probes = []
        for kind, requests in (
            ("prefill", [_probe_request(1024, 0, 1024, True)]),
            ("decode", [_probe_request(1, 1024, 1024, False)] * 4),
        ):
            metrics = [rank_metrics(requests)] + [
                rank_metrics([]) for _ in range(self.num_ranks - 1)
            ]
            sent = metrics[0] if self.num_ranks == 1 else metrics
            try:
                ms = self.model.estimate_forward_pass_time_ms(sent)
            except Exception as e:
                raise ValueError(
                    f"AISimulate cannot time a {kind} step of {target} on {system} "
                    f"with {topology}: {e}"
                ) from e
            if ms is None or not math.isfinite(ms) or ms <= 0:
                raise ValueError(
                    f"AISimulate returned {ms!r} ms for a {kind} step of {target} "
                    f"on {system} with {topology}."
                )
            probes.append({"kind": kind, "metrics": sent, "ms": ms})
        return probes

    def bind_model(self, model: Any) -> None:
        """The weights timed are the built layers': the published
        checkpoint's, or those of the online quantization the estimator was
        configured for (`--quantization fp8`, `fp8_per_block`). Layers
        quantized online any other way (another method, an online
        `--quantization-config`, beside a quantized checkpoint) are another
        deployment, which is not timed as this one."""
        online = {
            type(method).__name__
            for module in model.modules()
            if (method := getattr(module, "quant_method", None)) is not None
            and type(method).__module__.startswith(ONLINE_QUANT_MODULES)
        }
        if online - self._online_methods:
            record = self._assumptions
            timed = record["online_quantization"] or (
                f"{record['target_quantization'] or 'unquantized'} weights"
            )
            raise ValueError(
                "The aisimulate timing provider times the published "
                f"{record['target_model']} ({timed}), not "
                "layers this server quantizes online "
                f"({', '.join(sorted(online - self._online_methods))}): it times "
                f"--quantization {', '.join(sorted(ONLINE_QUANT_MODES))} of an "
                "unquantized checkpoint only."
            )
        if self._online_methods and not online:
            raise ValueError(
                "The aisimulate timing provider was configured for online "
                f"quantization ({self._assumptions['online_quantization']}), but "
                "no layer of the built model is quantized online."
            )

    def assumptions(self) -> dict[str, Any]:
        """What the step times rest on, for the step log."""
        return {"aisimulate": self._assumptions}

    def estimate(self, step: StepDescriptor) -> StepCost:
        if not step.requests:
            return StepCost(device_s=0.0, host_s=self.host.host_s(step))
        ranks: list[list[SimRequestStep]] = [[] for _ in range(self.num_ranks)]
        for request in step.requests:
            if not 0 <= request.dp_rank < self.num_ranks:
                raise ValueError(
                    f"Request {request.req_id} is on data-parallel rank "
                    f"{request.dp_rank} of {self.num_ranks}."
                )
            ranks[request.dp_rank].append(request)
        metrics = [rank_metrics(r) for r in ranks]
        ms = self.model.estimate_forward_pass_time_ms(
            metrics[0] if self.num_ranks == 1 else metrics
        )
        if ms is None or not math.isfinite(ms) or ms < 0:
            raise ValueError(
                f"AISimulate returned {ms!r} ms for step {step.step_index}."
            )
        breakdown = {"aisimulate_ms": ms}
        if nextn := self.config["nextn"]:
            # The configured width against the drafts this step scheduled.
            decode = [r for r in step.requests if not r.is_prefill]
            breakdown["spec_draft_tokens"] = float(
                sum(r.num_spec_tokens for r in decode)
            )
            breakdown["spec_requests_below_width"] = float(
                sum(r.num_spec_tokens < nextn for r in decode)
            )
        return StepCost(
            device_s=ms / 1000.0,
            host_s=self.host.host_s(step),
            breakdown=breakdown,
        )
