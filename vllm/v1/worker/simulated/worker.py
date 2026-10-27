# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker for simulated execution: no weights, no KV tensors, no forward."""

# ruff: noqa: E402
import os

import torch

from vllm.platforms import current_platform

if current_platform.is_cpu():
    # Stubs torch's CUDA stream/event APIs for the GPU runner, as in
    # cpu_worker.py; must precede its import.
    import vllm.v1.worker.cpu.shm  # noqa: F401

from vllm.config.vllm import set_current_vllm_config
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.simulated.model_runner import SimulatedModelRunner
from vllm.v1.worker.worker_base import CompilationTimes

logger = init_logger(__name__)


_TENT_METRICS_KEY = "metrics/http_port"
_TENT_METRICS_PORT = "TENT_METRICS_HTTP_PORT"
_TENT_METRICS_BASE_PORT = "VLLM_SIM_TENT_METRICS_BASE_PORT"
# TENT's MetricsConfig::http_port.
_TENT_DEFAULT_METRICS_PORT = 9100


def _thread_ids() -> set[int]:
    try:
        return {int(tid) for tid in os.listdir("/proc/self/task")}
    except OSError:
        return set()


def _rank_tent_metrics_port() -> None:
    """Every rank's transfer engine serves TENT's metrics on its own port:
    the base port plus its tensor-parallel rank. TENT takes the port from
    MC_TENT_CONF's metrics/http_port (nested, else the flat key), else
    TENT_METRICS_HTTP_PORT, else its default; whichever it would use is
    offset."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config
    from vllm.distributed.parallel_state import get_tp_group

    def number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    conf = tent_config.load()
    found = tent_config.lookup(conf, _TENT_METRICS_KEY, number) if conf else None
    base = os.environ.get(_TENT_METRICS_BASE_PORT) or str(
        (found and int(found[1]))
        or os.environ.get(_TENT_METRICS_PORT)
        or _TENT_DEFAULT_METRICS_PORT
    )
    os.environ[_TENT_METRICS_BASE_PORT] = base
    port = int(base) + get_tp_group().rank_in_group
    os.environ[_TENT_METRICS_PORT] = str(port)
    if found is not None and int(found[1]) != port:
        assert conf is not None
        tent_config.store(conf, _TENT_METRICS_KEY, port, nested=found[0])
        tent_config.save(conf)


class SimulatedWorker(Worker):
    """Worker for simulated execution: no weights, no KV tensors, no forward."""

    _kv_connector: KVConnectorBaseType | None = None

    def init_device(self) -> None:
        if current_platform.is_cuda_alike():
            self._bind_device()
        else:
            from vllm.v1.worker.cpu_worker import set_dist_ident

            self.device = torch.device("cpu")
            set_dist_ident(self.distributed_init_method)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        set_random_seed(self.model_config.seed)
        logger.info(
            "Simulated execution on %s (timing=%s)",
            self.device,
            self.vllm_config.simulation_config.timing,
        )
        self._init_workspace_and_model_runner()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # Engines simulated in one process (disaggregated instances) each
        # build their own connector, which is a process global.
        kv_transfer_state.set_kv_transfer_group(None)
        if self.vllm_config.simulation_config.kv_transfer_mode == "real":
            _rank_tent_metrics_port()
        before = _thread_ids()
        # The attention backend's kernel block sizes depend on the simulation.
        with set_current_vllm_config(self.vllm_config):
            super().initialize_from_config(kv_cache_config)
        if kv_transfer_state.has_kv_transfer_group():
            self._kv_connector = kv_transfer_state.get_kv_transfer_group()
            # Native threads the connector started (its transfer engine),
            # which a profiler cannot tell apart by name.
            self.model_runner.connector_threads = _thread_ids() - before

    def shutdown(self) -> None:
        # Shut down this engine's connector, not the last one built.
        kv_transfer_state.set_kv_transfer_group(self._kv_connector)
        super().shutdown()

    def execute_model(self, scheduler_output):  # type: ignore[override]
        """No hidden states flow between pipeline stages: every stage runs
        its own bookkeeping and timing from the same scheduler output."""
        with self.annotate_profile(scheduler_output):
            return self.model_runner.execute_model(scheduler_output)

    def _make_model_runner(self) -> SimulatedModelRunner:  # type: ignore[override]
        return SimulatedModelRunner(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        """Virtual KV capacity. Nothing is profiled: the KV cache is sized from
        `kv_cache_memory_bytes`, or from the device's total memory and
        `gpu_memory_utilization` when that is unset."""
        kv_bytes = self.cache_config.kv_cache_memory_bytes
        if kv_bytes is None:
            assert self.device is not None
            if self.device.type == "cpu":
                import psutil

                total = psutil.virtual_memory().total
            else:
                total = torch.accelerator.get_device_properties(
                    self.device
                ).total_memory  # type: ignore[attr-defined]
            kv_bytes = int(total * self.cache_config.gpu_memory_utilization)
            logger.warning(
                "kv_cache_memory_bytes is unset; simulating %s GiB of KV cache "
                "from the local device. Pass --kv-cache-memory-bytes to model "
                "the target deployment.",
                format_gib(kv_bytes),
            )
        logger.info("Using %s GiB of virtual KV cache.", format_gib(kv_bytes))
        return kv_bytes

    def compile_or_warm_up_model(self) -> CompilationTimes:
        assert isinstance(self.model_runner, SimulatedModelRunner)
        self.model_runner.warm_up_step_kernels()
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("Sleep mode is a no-op under simulated execution.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("Wake-up is a no-op under simulated execution.")
