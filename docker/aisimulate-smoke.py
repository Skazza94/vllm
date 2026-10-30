"""Build-time check of the AISimulate layer, offline: a ready estimator for
a bundled target gives a finite positive step time, and none of AISimulate's
search-only dependencies (JAX) came along."""

import importlib.metadata
import importlib.util
import math

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

assert importlib.util.find_spec("jax") is None, "JAX must not be installed"
model = RustForwardPassPerfModel.best_available(
    ForwardPassPerfModelConfig(
        model="Qwen/Qwen3-30B-A3B", system="h100_sxm", backend="vllm",
        worker_type="decode", tp=1, moe_tp_size=1, moe_ep_size=1,
        kvcache_quant_mode="bfloat16", database_mode="SILICON",
        fallback_policy="deny",
    )
)
diagnostics = model.diagnostics()
assert (diagnostics["source"], diagnostics["readiness"]) == ("aic", "ready"), diagnostics
ms = model.estimate_forward_pass_time_ms({"scheduled_requests": {
    "num_prefill_requests": 0, "sum_prefill_tokens": 0, "var_prefill_length": 0.0,
    "sum_prefill_kv_tokens": 0, "num_decode_requests": 8,
    "sum_decode_kv_tokens": 8192, "var_decode_kv_tokens": 0.0,
}})
assert ms is not None and math.isfinite(ms) and ms > 0, ms
print(f"aisimulate {importlib.metadata.version('aisimulate')}: decode {ms:.3f} ms")
