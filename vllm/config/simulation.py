# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for simulated execution."""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config, hash_factors

if TYPE_CHECKING:
    from vllm.config.vllm import VllmConfig

_SIMULATED_WORKER = "vllm.v1.worker.simulated.worker.SimulatedWorker"
_RECORDING_WORKER = "vllm.v1.worker.simulated.recording.RecordingWorker"
# Native connectors that move real KV under kv_transfer_mode real: NIXL's
# decode-pulled READs, or Mooncake's prefill-written transfers.
NIXL_CONNECTORS = ("NixlConnector", "NixlPullConnector")
REAL_KV_CONNECTORS = (*NIXL_CONNECTORS, "MooncakeConnector")
# Speculative decoding with real KV transfers: a model's native single-module
# MTP (its draft architecture), whose one MTP layer's KV joins the target's.
REAL_KV_MTP_DRAFTS = {
    "deepseek_v3": "DeepSeekMTPModel",
    "deepseek_v4": "DeepSeekV4MTPModel",
    "qwen3_5": "Qwen3_5MTP",
    "qwen3_5_moe": "Qwen3_5MoeMTP",
    "deepseek_v32": "DeepseekV32MTPModel",
    "glm_moe_dsa": "DeepseekV32MTPModel",
}
REAL_KV_MTP_MAX_DEPTH = 5
# Those Mooncake moves too: its draft one more MLA layer of the target's
# groups. Mooncake has no speculative-compatibility check (it refuses only a
# producer layer the consumer lacks); both instances must share the config.
# A hybrid's MTP is NIXL's: Mooncake would also move each state's k scratch
# slots, which NIXL leaves.
REAL_KV_MTP_MOONCAKE = ("deepseek_v3", "deepseek_v32", "glm_moe_dsa")
# Sparse MLA with a lightning indexer (DSA): DeepSeek-V3.2, GLM-5.
DSA_MODEL_TYPES = ("deepseek_v32", "glm_moe_dsa")
# Cache dtypes a DSA latent is kept packed for (FlashMLA's `fp8_ds_mla`).
DSA_FP8_CACHE_DTYPES = ("fp8", "fp8_e4m3", "fp8_ds_mla")


@config
class KVTransferTiming:
    """Duration of simulated KV transfers: `latency + bytes / bandwidth`."""

    load_gbps: float | None = Field(default=None, gt=0)
    """Load bandwidth, 1e9 bytes/s (required): from the CPU cache when
    offloading, from a prefill instance under disaggregated prefill."""

    load_latency_us: float = Field(default=0.0, ge=0)
    """Latency of a load."""

    store_gbps: float | None = Field(default=None, gt=0)
    """Store bandwidth to the CPU cache, 1e9 bytes/s (offloading only)."""

    store_latency_us: float = Field(default=0.0, ge=0)
    """Latency of a store."""

    post_us_per_read: float = Field(default=0.0, ge=0)
    """Host time a NIXL decode rank spends posting a read."""

    post_us_per_run: float = Field(default=0.0, ge=0)
    """Additional posting time per run of blocks consecutive on both sides."""

    def __post_init__(self) -> None:
        if self.load_gbps is None:
            raise ValueError("kv_transfer_timing needs load_gbps.")


@config
class SimulationConfig:
    """Simulated execution: the real engine runs, and each step takes the time
    a timing model predicts instead of running the model."""

    simulate_forward: bool = False
    """Replace model forward and sampling with simulated execution."""

    timing: str = "zero"
    """Timing provider: `zero`, `linear`, `table`, `replay`, or the qualified
    name of a `vllm.v1.worker.simulated.timing.TimingProvider` class."""

    timing_args: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments of the timing provider."""

    clock: Literal["real", "virtual", "fast_forward"] = "real"
    """`real`: steps take their time on the wall clock. `fast_forward`: waits
    are skipped. `virtual`: waits are skipped and only predicted host time
    counts (deterministic). The last two need an engine built by
    `vllm.v1.worker.simulated.offline.make_engine`."""

    log_path: str | None = None
    """Step log to append to: predicted steps when simulating, measured ones
    with `record_step_timing`."""

    record_step_timing: bool = False
    """Measure a real deployment's steps and write them to `log_path`."""

    spec_acceptance_rates: list[float] | str | None = None
    """Per-position draft acceptance probabilities (the last one repeats), or
    a step log recorded with speculative decoding to estimate them from."""

    kv_transfer_timing: KVTransferTiming | str | None = None
    """Simulated KV transfer timing, or a step log recorded with the
    connector to fit it from."""

    kv_transfer_mode: Literal["simulated", "real"] = "simulated"
    """`simulated`: the NIXL connector's worker side is modeled and moves no
    data; `kv_transfer_timing` gives the transfer times. `real`: a native
    connector (`REAL_KV_CONNECTORS`) registers real KV cache tensors and moves
    them between instances, while compute stays simulated."""

    kv_pattern_fill: bool = False
    """Real KV transfers, prefill instance: write a deterministic pattern into
    each request's KV blocks as they are allocated, so that a decode instance
    can check the bytes it receives (`kv_pattern_check`). Costs a write of
    every block; leave it off when measuring performance."""

    kv_pattern_check: bool = False
    """Real KV transfers, decode instance: fill the blocks a read lands in
    with a sentinel, and compare them with the prefill instance's pattern when
    the read finishes. Results go to the step log (`kv_check` records)."""

    fallback_token_id: int = Field(default=0, ge=0)
    """Output token of requests without `trace_decode_token_ids`."""

    @field_validator("spec_acceptance_rates")
    @classmethod
    def _check_rates(cls, rates: list[float] | str | None) -> list[float] | str | None:
        if isinstance(rates, list) and (
            not rates or not all(0.0 <= r <= 1.0 for r in rates)
        ):
            raise ValueError("spec_acceptance_rates must be probabilities in [0, 1].")
        return rates

    def compute_hash(self) -> str:
        """Simulation never compiles a graph; only the mode enters the hash."""
        return hash_factors({"simulate_forward": self.simulate_forward})

    def verify_and_update_config(self, vllm_config: "VllmConfig") -> None:
        """Check `vllm_config` against simulated execution or recording, and
        set what they imply."""
        from vllm.platforms import current_platform

        parallel = vllm_config.parallel_config
        if self.record_step_timing:
            if self.simulate_forward:
                raise ValueError(
                    "record_step_timing measures a real deployment; it cannot "
                    "be combined with simulate_forward."
                )
            if self.log_path is None:
                raise ValueError("record_step_timing needs log_path.")
            if not current_platform.is_cuda_alike():
                raise ValueError("record_step_timing needs a CUDA-alike platform.")
            if parallel.worker_cls == "auto":
                parallel.worker_cls = _RECORDING_WORKER
        if (self.kv_pattern_fill or self.kv_pattern_check) and (
            self.kv_transfer_mode != "real"
        ):
            raise ValueError(
                "kv_pattern_fill and kv_pattern_check need kv_transfer_mode real."
            )
        if not self.simulate_forward:
            if self.kv_transfer_mode == "real":
                raise ValueError(
                    "kv_transfer_mode real applies to simulated execution; it "
                    "needs simulate_forward."
                )
            return

        if not (current_platform.is_cpu() or current_platform.is_cuda_alike()):
            raise ValueError("Simulated forward needs a CUDA-alike or CPU platform.")
        if current_platform.is_cpu() and parallel.decode_context_parallel_size > 1:
            self._verify_cpu_decode_context_parallel(vllm_config)
        if vllm_config.lora_config is not None:
            raise ValueError("Simulated forward does not support LoRA.")
        if vllm_config.weight_transfer_config is not None:
            raise ValueError("Simulated forward does not support weight transfer.")
        if vllm_config.model_config.runner_type != "generate":
            raise ValueError("Simulated forward supports generation runners only.")
        if self.clock != "real" and parallel.world_size_across_dp > 1:
            raise ValueError(
                f"The {self.clock} simulation clock supports one rank, not "
                f"{parallel.world_size_across_dp}; use the real clock."
            )
        self._verify_kv_transfer(vllm_config)

        if parallel.worker_cls == "auto":
            parallel.worker_cls = _SIMULATED_WORKER
        # Nothing is reduced across ranks.
        parallel.disable_custom_all_reduce = True
        # Requests may carry their output tokens (trace_decode_token_ids).
        vllm_config.model_config.enable_trace_replay = True
        load_configs = [vllm_config.load_config]
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.draft_load_config is not None:
            load_configs.append(speculative.draft_load_config)
        for load_config in load_configs:
            load_config.load_format = "dummy"
            load_config.device = "meta"

    def _verify_cpu_decode_context_parallel(self, vllm_config: "VllmConfig") -> None:
        """Decode context parallelism on CPU over tensor parallelism alone.
        Dense MLA can run in one instance or with real NIXL pull transfers,
        prefix caching on or off; the handshake checks the peer's DCP. Sparse MLA
        requires AISimulate and one instance. Neither backend runs attention
        kernels under a simulated forward. CPU GQA has no DCP path."""
        from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
            dcp_scope_problems,
        )

        parallel = vllm_config.parallel_config
        transfer = vllm_config.kv_transfer_config
        problems = dcp_scope_problems(vllm_config, transfers=transfer is not None)
        if transfer is not None:
            if transfer.kv_connector != "NixlConnector":
                problems.append(f"the {transfer.kv_connector} connector")
            if self.kv_transfer_mode != "real":
                problems.append(f"kv_transfer_mode {self.kv_transfer_mode}")
        if problems:
            raise ValueError(
                "Simulated forward on CPU supports decode context parallelism "
                f"(decode_context_parallel_size "
                f"{parallel.decode_context_parallel_size}) over tensor parallelism "
                "alone: dense MLA in one instance or with real NixlConnector "
                "pull transfers; sparse MLA only in one "
                f"instance with AISimulate, not with {', '.join(problems)}."
            )

    @staticmethod
    def _real_kv_mtp_problems(vllm_config: "VllmConfig") -> list[str]:
        """Speculative decoding moves its draft's KV with the target's only as
        a model's native single-module MTP (`REAL_KV_MTP_DRAFTS`): the
        checkpoint's one MTP layer, at the target's parallelism and
        precision; over NIXL, whose handshake checks the instances' transfer
        compatibility, the draft among it, or over Mooncake for
        `REAL_KV_MTP_MOONCAKE` (no such check: the instances' configs must
        match)."""
        spec = vllm_config.speculative_config
        model = vllm_config.model_config
        kv_transfer = vllm_config.kv_transfer_config
        k = spec.num_speculative_tokens
        draft = REAL_KV_MTP_DRAFTS.get(getattr(model.hf_config, "model_type", None))
        architectures = list(
            getattr(spec.draft_model_config, "architectures", None) or []
        )
        text = model.hf_text_config
        layers = getattr(text, "num_nextn_predict_layers", None) or getattr(
            text, "mtp_num_hidden_layers", None
        )
        mismatches = []
        if spec.method != "mtp":
            mismatches.append(f"method {spec.method}")
        if draft is None or architectures != [draft]:
            mismatches.append(
                f"draft architectures {architectures} for "
                f"{getattr(model.hf_config, 'model_type', None)}"
            )
        if spec.model != model.model:
            mismatches.append(f"draft checkpoint {spec.model}")
        if layers != 1 or spec.use_multi_module_mtp():
            mismatches.append(f"{layers} MTP layers")
        if not 1 <= k <= REAL_KV_MTP_MAX_DEPTH:
            mismatches.append(f"{k} speculative tokens")
        tp = vllm_config.parallel_config.tensor_parallel_size
        if spec.draft_tensor_parallel_size not in (None, tp):
            mismatches.append(
                f"draft tensor parallelism {spec.draft_tensor_parallel_size}"
            )
        if spec.quantization not in (None, model.quantization):
            mismatches.append(f"draft quantization {spec.quantization}")
        if spec.kv_cache_dtype not in (None, vllm_config.cache_config.cache_dtype):
            mismatches.append(f"draft KV cache dtype {spec.kv_cache_dtype}")
        if getattr(spec, "draft_lm_head_quantization", None) is not None:
            mismatches.append(
                f"draft LM head quantization {spec.draft_lm_head_quantization}"
            )
        model_type = getattr(model.hf_config, "model_type", None)
        if kv_transfer.kv_connector not in NIXL_CONNECTORS and not (
            kv_transfer.kv_connector == "MooncakeConnector"
            and model_type in REAL_KV_MTP_MOONCAKE
        ):
            mismatches.append(f"{kv_transfer.kv_connector} for {model_type}")
        if not mismatches:
            return []
        return [
            "speculative decoding only as a native single-module MTP (DeepSeek-V3, "
            f"DeepSeek-V3.2, GLM-5, DeepSeek-V4, Qwen3.5; 1 to {REAL_KV_MTP_MAX_DEPTH} "
            "tokens) over NIXL, or DeepSeek-V3's, DeepSeek-V3.2's or GLM-5's over "
            f"Mooncake, not {', '.join(mismatches)}"
        ]

    def _verify_kv_transfer(self, vllm_config: "VllmConfig") -> None:
        if self.kv_transfer_mode == "real":
            # Before any simulated-only branch, so none of them can skip it.
            self._verify_real_kv_transfer(vllm_config)
            return
        kv_transfer = vllm_config.kv_transfer_config
        if kv_transfer is None:
            return
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory
        from vllm.v1.worker.simulated import kv_nixl

        connector = kv_transfer.kv_connector
        if connector == "OffloadingConnector":
            spec = OffloadingSpecFactory.get_spec_cls(
                kv_transfer.kv_connector_extra_config
            )
            if not issubclass(spec, CPUOffloadingSpec):
                raise ValueError(
                    f"Simulated KV offloading supports CPU offloading, not "
                    f"{spec.__name__}."
                )
            return
        if connector not in kv_nixl.SIMULATED_CONNECTORS:
            raise ValueError(
                f"Simulated forward does not support the {connector} connector."
            )
        if kv_transfer.kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "Simulated disaggregated prefill needs kv_role kv_producer or "
                f"kv_consumer, not {kv_transfer.kv_role}."
            )
        if vllm_config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError(
                "Simulated disaggregated prefill does not support decode "
                "context parallelism."
            )
        if self.clock == "fast_forward" or (
            connector == kv_nixl.NixlPushConnector.__name__ and self.clock != "real"
        ):
            raise ValueError(
                f"Simulated {connector} does not support the {self.clock} clock."
            )
        module_path = kv_transfer.kv_connector_module_path
        if module_path not in (None, kv_nixl.__name__):
            raise ValueError(
                "Simulated forward replaces the NIXL connector's worker side; "
                f"kv_connector_module_path {module_path} cannot be set."
            )
        kv_transfer.kv_connector_module_path = kv_nixl.__name__

    def check_pattern_request(
        self,
        enable_prefix_caching: bool,
        *,
        lora_request: object,
        cache_salt: str | None,
        prompt_embeds: object,
        mm_features: object,
    ) -> None:
        """With prefix caching, the pattern checks seed a block by its tokens:
        a request whose cache key adds more (a LoRA adapter, a salt, embedded
        or multimodal input) is refused at intake, before it is scheduled."""
        if not (self.kv_pattern_fill or self.kv_pattern_check) or not (
            enable_prefix_caching
        ):
            return
        extra = [
            name
            for name, present in (
                ("a LoRA adapter", lora_request is not None),
                ("a cache_salt", bool(cache_salt)),
                ("prompt embeddings", prompt_embeds is not None),
                ("multimodal inputs", bool(mm_features)),
            )
            if present
        ]
        if extra:
            raise ValueError(
                "KV pattern checks with prefix caching take token prompts only, "
                f"not {', '.join(extra)}: a block's pattern is its tokens'."
            )

    def _verify_real_kv_transfer(self, vllm_config: "VllmConfig") -> None:
        """Check what real transfers under simulated compute support so far:
        CPU KV buffers moved by a native connector (NIXL or Mooncake) between
        instances of a full-attention model, dense or MoE (its experts hold
        no KV), or of a dense MLA one (one latent per token and layer, which
        every tensor-parallel rank holds whole), on one rank, tensor-parallel
        ones, or an MoE model's data-parallel ranks with expert parallelism.
        Sliding-window layers, alone or beside full-attention ones, of which
        either connector moves each window's blocks only; Qwen gated
        delta-net or Mamba2 layers (a hybrid model's), each request's state
        one block or, with prefix caching, a state a block (the align mode
        vLLM then sets), over Mooncake or over NIXL with the DS conv-state
        layout NIXL's state descriptors need. The native connector stays in
        place."""
        from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import plain_mla
        from vllm.platforms import current_platform

        kv_transfer = vllm_config.kv_transfer_config
        if kv_transfer is None:
            raise ValueError(
                "kv_transfer_mode real needs a kv_transfer_config with one of "
                + ", ".join(REAL_KV_CONNECTORS)
                + "."
            )
        problems = []
        if vllm_config.cache_config.kv_offloading_size is not None:
            # Validated before VllmConfig turns it into an offloading
            # connector (kv_role kv_both), which would replace the native one.
            problems.append(
                f"no KV offloading (kv_offloading_size "
                f"{vllm_config.cache_config.kv_offloading_size})"
            )
        if kv_transfer.kv_connector not in REAL_KV_CONNECTORS:
            problems.append(
                f"one of {', '.join(REAL_KV_CONNECTORS)}, not {kv_transfer.kv_connector}"
            )
        if kv_transfer.kv_connector_module_path is not None:
            problems.append(
                "the native connector, without kv_connector_module_path "
                f"({kv_transfer.kv_connector_module_path})"
            )
        if not current_platform.is_cpu() or kv_transfer.kv_buffer_device != "cpu":
            problems.append(
                "CPU KV buffers (the CPU platform and kv_buffer_device cpu), not "
                f"{current_platform.device_type} with {kv_transfer.kv_buffer_device}"
            )
        if self.clock != "real":
            problems.append(f"the real clock, not {self.clock}")
        if self.kv_transfer_timing is not None:
            problems.append(
                "no kv_transfer_timing: transfers take their actual time"
            )
        cache = vllm_config.cache_config
        if cache.kv_cache_memory_bytes is None:
            problems.append("an explicit kv_cache_memory_bytes")
        parallel = vllm_config.parallel_config
        sizes = {
            "TP": parallel.tensor_parallel_size,
            "PP": parallel.pipeline_parallel_size,
            "DP": parallel.data_parallel_size,
            "PCP": parallel.prefill_context_parallel_size,
            "DCP": parallel.decode_context_parallel_size,
        }
        model = vllm_config.model_config
        # Tensor parallelism: each rank moves its own KV heads, to the
        # ranks of the other instance that hold them; an instance of another
        # TP size pairs for plain attention only, the connector re-sharding
        # heads (`simulated_tp_mismatch`; each head's pattern is its own).
        # Data-parallel attention (an MoE model's DP
        # ranks, with expert parallelism): each DP rank holds its own
        # requests' KV, paired with whichever DP rank serves the request on
        # the other instance.
        # Decode context parallelism: a dense MLA model over NIXL's pull
        # connector, the scope `_verify_cpu_decode_context_parallel` checks
        # (each rank's blocks are global pages, `kv_pattern.dcp_page`).
        dcp_ok = (
            sizes["DCP"] > 1
            and kv_transfer.kv_connector == "NixlConnector"
            and plain_mla(model)
        )
        if any(sizes[name] != 1 for name in ("PP", "PCP")) or (
            sizes["DCP"] != 1 and not dcp_ok
        ):
            parallel_sizes = ", ".join(f"{k}={v}" for k, v in sizes.items())
            problems.append(
                "tensor parallelism and data-parallel attention at most, not "
                f"{parallel_sizes}"
            )
        elif sizes["DP"] > 1 and not (model.is_moe and parallel.enable_expert_parallel):
            problems.append(
                "data parallelism only as an MoE model's data-parallel attention, "
                "with expert parallelism"
            )
        # DeepSeek-V3.2's MLA is sparse: an indexer picks the tokens. V4's is
        # too; its caches (packed FP8 latents, sliding windows, compressor
        # states) move over NIXL, which keeps each group's own blocks.
        v4 = getattr(model.hf_config, "model_type", None) == "deepseek_v4"
        if v4:
            if kv_transfer.kv_connector not in ("NixlConnector", "NixlPullConnector"):
                problems.append(
                    f"NixlConnector for DeepSeek-V4, not {kv_transfer.kv_connector}"
                )
            # Its caches are one replicated shard: each TP rank moves them
            # whole, inside a DP rank too.
            if cache.cache_dtype not in ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
                problems.append(
                    f"DeepSeek-V4's packed FP8 cache, not {cache.cache_dtype}"
                )
        sparse_mla = (
            model.use_mla
            and not v4
            and getattr(model.hf_text_config, "index_topk", None) is not None
        )
        # DeepSeek-V3.2's sparse attention (DSA, GLM-5's too): a latent, BF16
        # or FlashMLA's packed FP8 (any FP8 dtype is served as `fp8_ds_mla`),
        # and an indexer key cache a layer, as the deployment keeps them.
        dsa = sparse_mla and getattr(model.hf_config, "model_type", None) in (
            DSA_MODEL_TYPES
        )
        if dsa:
            sparse_mla = False
            # `auto` takes the model's dtype: BF16 only when the model is.
            resolved = model.dtype if cache.cache_dtype == "auto" else cache.cache_dtype
            if str(resolved).removeprefix("torch.") not in (
                "bfloat16",
                *DSA_FP8_CACHE_DTYPES,
            ):
                problems.append(
                    "a BF16 sparse-MLA latent cache (kv_cache_dtype auto of a "
                    "BF16 model, or bfloat16) or its packed FP8 one "
                    f"({', '.join(DSA_FP8_CACHE_DTYPES)}), not {cache.cache_dtype} "
                    f"({resolved})"
                )
        if cache.enable_prefix_caching and (self.kv_pattern_fill or self.kv_pattern_check):
            # The pattern is then a block's content (its tokens and every
            # earlier one, as prefix caching keys it): attention caches, a
            # hybrid's states (the tokens a state has absorbed), DSA's
            # latents and indexer keys and DeepSeek-V4's packed groups. With
            # speculative decoding a page is written once its tokens are known
            # and their KV computed (drafts only once accepted). A hybrid's
            # speculative state blocks, which align mode relocates, are not
            # validated with it.
            if vllm_config.speculative_config is not None and model.is_hybrid:
                problems.append(
                    "no speculative decoding with prefix caching for "
                    "kv_pattern_fill and kv_pattern_check on a hybrid model"
                )
        if sparse_mla or model.is_attention_free:
            problems.append(
                "an attention model, dense, MoE or dense MLA, its layers full "
                "or sliding-window attention, Qwen gated delta-nets or Mamba2 "
                "mixers (no sparse MLA or attention-free models)"
            )
        elif model.is_hybrid and kv_transfer.kv_connector in NIXL_CONNECTORS:
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            # NIXL moves a state as its conv sub-projections and SSM state,
            # which it locates in the DS layout only.
            if not is_conv_state_dim_first():
                problems.append(
                    "the DS conv-state layout for a hybrid model over NIXL "
                    "(VLLM_SSM_CONV_STATE_LAYOUT=DS)"
                )
        if vllm_config.speculative_config is not None:
            problems.extend(self._real_kv_mtp_problems(vllm_config))
        if self.kv_pattern_fill and kv_transfer.kv_role != "kv_producer":
            problems.append("kv_role kv_producer for kv_pattern_fill")
        if self.kv_pattern_check and kv_transfer.kv_role != "kv_consumer":
            problems.append("kv_role kv_consumer for kv_pattern_check")
        if self.kv_pattern_check and self.log_path is None:
            problems.append("log_path for kv_pattern_check's results")
        if problems:
            raise ValueError(
                "kv_transfer_mode real currently needs " + "; ".join(problems) + "."
            )
