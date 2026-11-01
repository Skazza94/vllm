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
from typing import TYPE_CHECKING, Any

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
# Tensor-parallel sizes of one modeled NVLink domain.
TENSOR_PARALLEL_SIZES = (1, 2, 4, 8)
MAX_LOCAL_DATA_PARALLEL = 8
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
    "DeepseekV3ForCausalLM": MLA_FIELDS,
    **dict.fromkeys(("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM"), DSA_FIELDS),
    "DeepseekV4ForCausalLM": V4_FIELDS,
    "NemotronHForCausalLM": ATTENTION_FIELDS + MAMBA2_FIELDS,
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


def _geometry(config: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """A config object's KV and state geometry, as its class resolves it. An
    attention head size its class leaves unset is the model's own fallback,
    hidden size over attention heads (Mixtral's, say)."""
    geometry = {name: getattr(config, name, None) for name in fields}
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

    Supported: pure tensor parallelism within one node (TP 1, 2, 4 or 8), or
    an MoE model's local data-parallel attention with expert parallelism
    (TP 1, 2 to 8 local DP ranks). Anything else is refused, as are pipeline
    and context parallelism, EPLB and speculative decoding other than
    DeepSeek-V3's or DeepSeek-V4's native single-module MTP. A multimodal
    model is timed as its language model, served without multimodal inputs."""

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

        v4 = target_config.get("architectures") == ["DeepseekV4ForCausalLM"]
        quant = _infer_quantization_fields(target_config).get("quant_algo")
        # V4's experts follow the raw config's `expert_dtype` (AISimulate's own
        # V4 override): FP4 experts with FP8 activations.
        expert_dtype = target_config.get("expert_dtype") if v4 else None
        modes = QUANT_MODES.get(quant) if not v4 else None
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
        if quant_modes:
            unknown = set(quant_modes) - set(QUANT_MODE_FIELDS)
            if unknown:
                raise ValueError(f"Unknown quant_modes {sorted(unknown)}.")
            modes.update(quant_modes)
        kv_format, kv_mode = self._kv_cache_mode(vllm_config, v4)
        # GLM-5's deployment on Blackwell (SM100) resolves `auto` to an FP8
        # sparse-MLA cache (`fp8_ds_mla`), which AISimulate does not time:
        # only an explicit BF16 cache is that GPU's timed deployment.
        if (
            target_config.get("architectures") == ["GlmMoeDsaForCausalLM"]
            and system == "b200_sxm"
            and vllm_config.cache_config.cache_dtype == "auto"
        ):
            raise ValueError(
                "GLM-5 on B200 resolves kv_cache_dtype auto to an FP8 sparse-MLA "
                "cache, which the aisimulate timing provider does not time; pass "
                "kv_cache_dtype bfloat16."
            )

        nextn, speculation = self._speculation(vllm_config, target_config, topology)

        kv_role = getattr(vllm_config.kv_transfer_config, "kv_role", None)
        worker_type = {"kv_producer": "prefill", "kv_consumer": "decode"}.get(
            kv_role, "aggregated"
        )
        self.num_ranks = topology["attention_dp"]
        self.config = {
            "model": target,
            "system": system,
            "backend": "vllm",
            "worker_type": worker_type,
            "backend_version": backend_version,
            **topology,
            "dcp": None,
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
        modified = bool(quant_modes)
        self._assumptions = {
            "provider": "aisimulate",
            "package_version": version,
            "source": diagnostics.get("source"),
            "readiness": diagnostics.get("readiness"),
            "config": self.config,
            "resolved_config": resolved,
            "target_model": target,
            "target_config_sha256": _file_sha256(bundled),
            "target_quantization": quant,
            "target_modified": modified,
            "local_config": local_config,
            "local_dtype": str(vllm_config.model_config.dtype).removeprefix("torch."),
            "kv_cache_dtype": vllm_config.cache_config.cache_dtype,
            "kv_cache_format": kv_format,
            "expert_dtype": expert_dtype,
            "startup_probes": probes,
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
    def _topology(vllm_config: "VllmConfig") -> dict[str, Any]:
        """The parallel layout, refused outside the supported scope."""
        pc = vllm_config.parallel_config
        problems = []
        if pc.pipeline_parallel_size > 1:
            problems.append("pipeline parallelism")
        if pc.prefill_context_parallel_size > 1 or pc.decode_context_parallel_size > 1:
            problems.append("context parallelism")
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
            if tp > 1:
                problems.append("tensor parallelism inside data parallelism")
            if pc.data_parallel_size_local != dp or dp > MAX_LOCAL_DATA_PARALLEL:
                problems.append(
                    f"{dp} data-parallel ranks ({pc.data_parallel_size_local} local; "
                    f"up to {MAX_LOCAL_DATA_PARALLEL}, all local)"
                )
        elif ep:
            problems.append("expert parallelism without data parallelism")
        if tp not in TENSOR_PARALLEL_SIZES:
            problems.append(f"tensor parallelism {tp} (not {TENSOR_PARALLEL_SIZES})")
        if problems:
            raise ValueError(
                "The aisimulate timing provider models pure tensor parallelism "
                "or an MoE model's local data-parallel attention with expert "
                f"parallelism, not {', '.join(problems)}."
            )
        if dp > 1:
            return {
                "tp": 1,
                "pp": 1,
                "attention_dp": dp,
                "moe_tp_size": 1,
                "moe_ep_size": dp,
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
    def _kv_cache_mode(cls, vllm_config: "VllmConfig", v4: bool) -> tuple[str, str]:
        """The KV cache's effective format and AISimulate's KV mode for it.
        The provider is built before the model resolves `auto`: DeepSeek-V4's
        is resolved here as its supported packed path does."""
        if v4:
            configured = vllm_config.cache_config.cache_dtype
            if configured not in V4_KV_CACHE_DTYPES:
                raise ValueError(
                    f"The aisimulate timing provider times DeepSeek-V4 with its "
                    f"packed FP8 cache (cache dtype {V4_KV_CACHE_DTYPES}), not "
                    f"{configured}."
                )
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
