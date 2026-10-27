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
REAL_KV_CONNECTORS = ("NixlConnector", "NixlPullConnector", "MooncakeConnector")


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

    def _verify_real_kv_transfer(self, vllm_config: "VllmConfig") -> None:
        """Check what real transfers under simulated compute support so far:
        CPU KV buffers moved by a native connector (NIXL or Mooncake) between
        instances of a full-attention model, dense or MoE (its experts hold
        no KV), on one rank or tensor-parallel ones. The native connector
        stays in place."""
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
        if cache.enable_prefix_caching and (self.kv_pattern_fill or self.kv_pattern_check):
            # The pattern is the request's: a block cached for another
            # request holds another pattern.
            problems.append("prefix caching disabled for kv_pattern_fill and kv_pattern_check")
        parallel = vllm_config.parallel_config
        sizes = {
            "TP": parallel.tensor_parallel_size,
            "PP": parallel.pipeline_parallel_size,
            "DP": parallel.data_parallel_size,
            "PCP": parallel.prefill_context_parallel_size,
            "DCP": parallel.decode_context_parallel_size,
        }
        # Tensor parallelism: each rank moves its own KV heads, to the
        # matching rank of an instance of the same TP size (the connector
        # checks the peer's).
        if any(size != 1 for name, size in sizes.items() if name != "TP"):
            parallel_sizes = ", ".join(f"{k}={v}" for k, v in sizes.items())
            problems.append(f"tensor parallelism at most, not {parallel_sizes}")
        elif sizes["TP"] > 1 and kv_transfer.kv_connector != "MooncakeConnector":
            # Only Mooncake checks the peer's TP size (simulated_tp_mismatch):
            # NIXL would pair unequal ranks, whose patterns differ.
            problems.append(
                f"MooncakeConnector for tensor parallelism, not {kv_transfer.kv_connector}"
            )
        model = vllm_config.model_config
        if (
            model.use_mla
            or model.is_hybrid
            or model.is_attention_free
            or model.get_sliding_window() is not None
        ):
            problems.append(
                "a full-attention model, dense or MoE (no MLA, hybrid, "
                "attention-free or sliding-window models)"
            )
        if vllm_config.speculative_config is not None:
            problems.append("no speculative decoding")
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
