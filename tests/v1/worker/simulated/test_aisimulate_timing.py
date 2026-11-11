# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AISimulate timing provider: its target deployment, feature gate and
step metrics, against a stand-in AISimulate package, and against the real
pinned package when it is installed."""

import dataclasses
import importlib.util
import json
import math
import os
import sys
import types
from types import SimpleNamespace

import pytest

from tests.v1.worker.simulated.utils import sim_req, sim_step
from vllm.v1.worker.simulated.timing import get_timing_provider

REAL = importlib.util.find_spec("aisimulate_core") is not None

QWEN3_MOE = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
}
DEEPSEEK_V3 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    "num_nextn_predict_layers": 1,
}
GPT_OSS = {
    "architectures": ["GptOssForCausalLM"],
    "hidden_size": 2880,
    "num_hidden_layers": 24,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 12,
    "quantization_config": {"quant_method": "mxfp4"},
}
NEMOTRON_H = {
    "architectures": ["NemotronHForCausalLM"],
    "hidden_size": 4480,
    "num_hidden_layers": 4,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hybrid_override_pattern": "M-*M",
    "mamba_num_heads": 128,
    "mamba_head_dim": 80,
    "n_groups": 8,
    "ssm_state_size": 128,
    "conv_kernel": 4,
}
BUNDLED = {
    "Qwen/Qwen3-30B-A3B": QWEN3_MOE,
    "deepseek-ai/DeepSeek-V3": DEEPSEEK_V3,
    "openai/gpt-oss-20b": GPT_OSS,
    "nvidia/Nemotron-H-Test": NEMOTRON_H,
    # ModelOpt checkpoints: their quantization is in a sidecar only.
    "nvidia/Qwen3-30B-A3B-NVFP4": QWEN3_MOE,
    "nvidia/Qwen3-30B-A3B-FP8": QWEN3_MOE,
    "nvidia/Qwen3-30B-A3B-Mixed": QWEN3_MOE,
    "nvidia/Qwen3-30B-A3B-Other": QWEN3_MOE,
}
SIDECARS = {
    "nvidia/Qwen3-30B-A3B-NVFP4": {"quantization": {"quant_algo": "NVFP4"}},
    "nvidia/Qwen3-30B-A3B-FP8": {"quantization": {"quant_algo": "FP8"}},
    "nvidia/Qwen3-30B-A3B-Mixed": {"quantization": {"quant_algo": "MIXED_PRECISION"}},
    "nvidia/Qwen3-30B-A3B-Other": {"quantization": {"quant_algo": "W4A8_AWQ"}},
}


class FakeModel:
    """Stands in for RustForwardPassPerfModel: records its config and
    queries, returns `ms` per query."""

    instances: list["FakeModel"] = []
    readiness = "ready"
    source = "aic"
    ms: float | None = 7.0

    def __init__(self, config):
        self.config = config
        self.queries = []

    @classmethod
    def best_available(cls, config):
        model = cls(config)
        cls.instances.append(model)
        return model

    def diagnostics(self):
        return {
            "source": self.source,
            "readiness": self.readiness,
            "last_warning": None if self.readiness == "ready" else "unsupported",
            "provenance": {
                "config": {**self.config.kwargs, "backend_version": "0.24.0"}
            },
        }

    def estimate_forward_pass_time_ms(self, metrics):
        self.queries.append(metrics)
        return self.ms


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def infer_quantization(raw):
    quant = raw.get("quantization_config") or {}
    if quant.get("quant_method") == "fp8" and quant.get("weight_block_size"):
        return {"quant_algo": "fp8_block", "quant_dynamic": True}
    if quant.get("quant_method") == "mxfp4":
        return {"quant_algo": "mxfp4"}
    # A ModelOpt checkpoint's sidecar, which only the full config holds.
    sidecar = (raw.get("hf_quant_config") or {}).get("quantization") or {}
    if sidecar.get("quant_algo"):
        return {"quant_algo": sidecar["quant_algo"].lower()}
    return {}


def load_full_config(bundle):
    """Stands in for AISimulate's loader: a bundled or snapshot config with
    its quantization sidecar, as `hf_quant_config` and `quant_algo`."""

    def load(model):
        if os.path.isdir(model):
            config, sidecar = (
                os.path.join(model, "config.json"),
                os.path.join(model, "hf_quant_config.json"),
            )
        else:
            stem = os.path.join(bundle, model.replace("/", "--"))
            config, sidecar = f"{stem}_config.json", f"{stem}_hf_quant_config.json"
        with open(config) as f:
            raw = json.load(f)
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                raw["hf_quant_config"] = json.load(f)
        raw.update(infer_quantization(raw))
        return raw

    return load


def infer_quant_modes(raw):
    """Stands in for AISimulate's reading of a full config's modes."""
    algo = raw.get("quant_algo")
    if algo == "nvfp4":
        return {"gemm_quant_mode": "nvfp4", "moe_quant_mode": "nvfp4"}
    if algo == "fp8":
        return {"gemm_quant_mode": "fp8_static", "moe_quant_mode": "fp8"}
    if algo == "mixed_precision":
        return {"gemm_quant_mode": "fp8_block", "moe_quant_mode": "nvfp4"}
    raise ValueError(f"Unsupported quant algorithm: {algo}")


@pytest.fixture
def fake_aisimulate(tmp_path, monkeypatch):
    """A stand-in AISimulate package with the bundled configs above."""
    configs = tmp_path / "aisimulate_core" / "model_configs"
    configs.mkdir(parents=True)
    for name, config in BUNDLED.items():
        (configs / f"{name.replace('/', '--')}_config.json").write_text(
            json.dumps(config)
        )
    core = types.ModuleType("aisimulate_core")
    core.__file__ = str(tmp_path / "aisimulate_core" / "__init__.py")
    sdk = types.ModuleType("aisimulate_core.sdk")
    sdk.ForwardPassPerfModelConfig = FakeConfig
    sdk.RustForwardPassPerfModel = FakeModel
    utils = types.ModuleType("aisimulate_core.sdk.utils")
    utils._infer_quantization_fields = infer_quantization
    utils._load_model_config_from_model_path = load_full_config(str(configs))
    models = types.ModuleType("aisimulate_core.sdk.models")
    helpers = types.ModuleType("aisimulate_core.sdk.models.helpers")
    helpers._infer_quant_modes_from_raw_config = infer_quant_modes
    # Every bundled config here is one the package lists as pre-downloaded.
    common = types.ModuleType("aisimulate_core.sdk.common")
    common.DefaultHFModels = set(BUNDLED)
    for name, module in (
        ("aisimulate_core", core),
        ("aisimulate_core.sdk", sdk),
        ("aisimulate_core.sdk.utils", utils),
        ("aisimulate_core.sdk.common", common),
        ("aisimulate_core.sdk.models", models),
        ("aisimulate_core.sdk.models.helpers", helpers),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    for name, sidecar in SIDECARS.items():
        (configs / f"{name.replace('/', '--')}_hf_quant_config.json").write_text(
            json.dumps(sidecar)
        )
    FakeModel.instances = []
    monkeypatch.setattr(FakeModel, "readiness", "ready")
    monkeypatch.setattr(FakeModel, "source", "aic")
    monkeypatch.setattr(FakeModel, "ms", 7.0)
    return FakeModel


class HfConfig(dict):
    """A stand-in config class resolving no defaults: the tests' configs
    spell out their geometry."""

    __getattr__ = dict.get

    @classmethod
    def from_dict(cls, config):
        return cls(config)

    def to_dict(self):
        return dict(self)


def engine_config(
    model="Qwen/Qwen3-30B-A3B",
    local=None,
    *,
    is_moe=None,
    dtype="bfloat16",
    cache_dtype="auto",
    role=None,
    speculative=None,
    **parallel,
):
    """The engine configuration the provider reads."""
    local = local or BUNDLED.get(model, QWEN3_MOE)
    defaults = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "data_parallel_size_local": parallel.get("data_parallel_size", 1),
        "data_parallel_rank_local": None,
        "data_parallel_external_lb": False,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": False,
        "enable_eplb": False,
        "nnodes": 1,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            dtype=dtype,
            architectures=local["architectures"],
            is_moe=(
                "Moe" in local["architectures"][0]
                or "Deepseek" in local["architectures"][0]
                or "GptOss" in local["architectures"][0]
            )
            if is_moe is None
            else is_moe,
            hf_text_config=HfConfig(local),
            quantization=None,
            supports_multimodal_inputs=False,
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        parallel_config=SimpleNamespace(**{**defaults, **parallel}),
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
        speculative_config=speculative,
    )


def provider(config, **timing_args):
    from vllm.v1.worker.simulated.aisimulate_timing import AisimulateTimingProvider

    return AisimulateTimingProvider(config, **timing_args)


def model_dir(tmp_path, name, config, derived=False):
    directory = tmp_path / name / "0123abcd"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config))
    if derived:
        (directory / "DERIVED.md").write_text("weights converted to BF16")
    return str(directory)


def test_a_native_moe_target_is_timed_with_its_own_modes(fake_aisimulate):
    p = provider(engine_config(role="kv_consumer"), gpu="h200")
    config = fake_aisimulate.instances[0].config.kwargs
    assert config["model"] == "Qwen/Qwen3-30B-A3B"
    assert config["system"] == "h200_sxm"
    assert config["worker_type"] == "decode"
    assert (
        config["tp"],
        config["attention_dp"],
        config["moe_tp_size"],
        config["moe_ep_size"],
    ) == (1, 1, 1, 1)
    assert (config["gemm_quant_mode"], config["moe_quant_mode"]) == (
        "bfloat16",
        "bfloat16",
    )
    assert config["kvcache_quant_mode"] == "bfloat16"
    assert config["comm_quant_mode"] == "half"
    assert (config["database_mode"], config["fallback_policy"]) == ("SILICON", "deny")
    assert (config["dcp"], config["cp_size"], config["enable_eplb"]) == (None, 1, False)
    record = p.assumptions()["aisimulate"]
    assert record["target_quantization"] is None
    assert not record["target_modified"]
    assert record["resolved_config"]["backend_version"] == "0.24.0"
    assert record["compared_geometry"]["num_key_value_heads"] == 4


@pytest.mark.parametrize(
    "role, worker_type",
    [("kv_producer", "prefill"), ("kv_consumer", "decode"), (None, "aggregated")],
)
def test_the_worker_type_follows_the_kv_role(fake_aisimulate, role, worker_type):
    provider(engine_config(role=role))
    assert fake_aisimulate.instances[0].config.kwargs["worker_type"] == worker_type


def test_a_dense_target_maps_tensor_parallelism_only(fake_aisimulate, tmp_path):
    dense = {**QWEN3_MOE, "architectures": ["Qwen3ForCausalLM"]}
    configs = tmp_path / "aisimulate_core" / "model_configs"
    (configs / "Qwen--Qwen3-8B-Test_config.json").write_text(json.dumps(dense))
    provider(engine_config("Qwen/Qwen3-8B-Test", dense, tensor_parallel_size=2))
    config = fake_aisimulate.instances[0].config.kwargs
    assert (config["tp"], config["moe_tp_size"], config["moe_ep_size"]) == (
        2,
        None,
        None,
    )


def test_derived_and_native_published_targets(fake_aisimulate, tmp_path):
    # A canonical directory names its own published model.
    native = model_dir(tmp_path, "openai--gpt-oss-20b", GPT_OSS)
    p = provider(engine_config(native, GPT_OSS))
    config = fake_aisimulate.instances[-1].config.kwargs
    assert config["model"] == "openai/gpt-oss-20b"
    assert (config["gemm_quant_mode"], config["moe_quant_mode"]) == (
        "bfloat16",
        "w4a16_mxfp4",
    )
    assert p.assumptions()["aisimulate"]["local_config"]["config_sha256"]
    # A BF16 copy must name its target; `-bf16` is never stripped.
    bf16 = {k: v for k, v in DEEPSEEK_V3.items() if k != "quantization_config"}
    derived = model_dir(tmp_path, "deepseek-ai--DeepSeek-V3-bf16", bf16, derived=True)
    with pytest.raises(ValueError, match="DERIVED.md"):
        provider(engine_config(derived, bf16))
    p = provider(engine_config(derived, bf16), target_model="deepseek-ai/DeepSeek-V3")
    config = fake_aisimulate.instances[-1].config.kwargs
    assert (config["gemm_quant_mode"], config["moe_quant_mode"]) == (
        "fp8_block",
        "fp8_block",
    )
    record = p.assumptions()["aisimulate"]
    assert record["target_quantization"] == "fp8_block"
    assert (
        "published deepseek-ai/DeepSeek-V3 (fp8_block weights)" in record["deployment"]
    )
    # A directory without the canonical name names nothing.
    plain = str(tmp_path / "plain")
    (tmp_path / "plain").mkdir()
    with pytest.raises(ValueError, match="target_model"):
        provider(engine_config(plain, QWEN3_MOE))


def test_an_override_labels_a_modified_target(fake_aisimulate):
    p = provider(engine_config(), quant_modes={"moe_quant_mode": "fp8_block"})
    assert fake_aisimulate.instances[0].config.kwargs["moe_quant_mode"] == "fp8_block"
    assert p.assumptions()["aisimulate"]["deployment"].startswith("modified published")
    with pytest.raises(ValueError, match="Unknown quant_modes"):
        provider(engine_config(), quant_modes={"kv_mode": "fp8"})


def test_kv_or_state_geometry_must_match(fake_aisimulate, tmp_path):
    # (Defaults resolve through the real config classes: the real-config
    # test below.)
    configs = tmp_path / "aisimulate_core" / "model_configs"
    with pytest.raises(ValueError, match="num_key_value_heads"):
        provider(
            engine_config("Qwen/Qwen3-30B-A3B", {**QWEN3_MOE, "num_key_value_heads": 8})
        )
    # A Mamba2 state dimension differs while attention is unchanged.
    provider(engine_config("nvidia/Nemotron-H-Test", NEMOTRON_H, is_moe=False))
    with pytest.raises(ValueError, match="ssm_state_size"):
        provider(
            engine_config(
                "nvidia/Nemotron-H-Test",
                {**NEMOTRON_H, "ssm_state_size": 64},
                is_moe=False,
            )
        )
    # An architecture whose compatibility is not established.
    configs.joinpath("org--Other_config.json").write_text(
        json.dumps({**QWEN3_MOE, "architectures": ["OtherForCausalLM"]})
    )
    with pytest.raises(ValueError, match="cannot establish KV compatibility"):
        provider(
            engine_config(
                "org/Other", {**QWEN3_MOE, "architectures": ["OtherForCausalLM"]}
            )
        )
    with pytest.raises(ValueError, match="bundled config"):
        provider(engine_config("org/Missing", QWEN3_MOE))


@pytest.mark.parametrize(
    "cache_dtype, dtype, mode",
    [
        ("auto", "bfloat16", "bfloat16"),
        ("auto", "float16", "bfloat16"),
        ("fp8", "bfloat16", "fp8"),
        ("fp8_e4m3", "bfloat16", "fp8"),
    ],
)
def test_the_kv_cache_dtype_is_passed_explicitly(
    fake_aisimulate, cache_dtype, dtype, mode
):
    provider(engine_config(cache_dtype=cache_dtype, dtype=dtype))
    assert fake_aisimulate.instances[0].config.kwargs["kvcache_quant_mode"] == mode


def test_an_unmapped_kv_cache_dtype_is_refused(fake_aisimulate):
    with pytest.raises(ValueError, match="KV cache dtypes"):
        provider(engine_config(dtype="float32"))
    with pytest.raises(ValueError, match="KV cache dtypes"):
        provider(engine_config(cache_dtype="fp8_ds_mla"))


@pytest.mark.parametrize(
    "parallel, problem",
    [
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
        ({"decode_context_parallel_size": 2}, "context parallelism"),
        ({"enable_eplb": True}, "EPLB"),
        ({"nnodes": 2}, "several nodes"),
        ({"tensor_parallel_size": 3}, "tensor parallelism 3"),
        (
            {"enable_expert_parallel": True},
            "expert parallelism without data parallelism",
        ),
        ({"data_parallel_size": 2}, "without an MoE model's expert parallelism"),
        (
            {
                "data_parallel_size": 2,
                "enable_expert_parallel": True,
                "tensor_parallel_size": 2,
            },
            "tensor parallelism inside data parallelism",
        ),
        (
            {
                "data_parallel_size": 4,
                "data_parallel_size_local": 2,
                "enable_expert_parallel": True,
            },
            "2 local",
        ),
        ({"data_parallel_size": 16, "enable_expert_parallel": True}, "up to 8"),
    ],
)
def test_unsupported_layouts_are_refused(fake_aisimulate, parallel, problem):
    # No real connector: the provider's own gate refuses them.
    with pytest.raises(ValueError, match=problem):
        provider(engine_config(**parallel))
    assert not fake_aisimulate.instances


def mtp(
    k=1, method="mtp", draft="deepseek_mtp", modules=1, arch="DeepSeekMTPModel", **fields
):
    """A speculative configuration: DeepSeek MTP of depth k by default."""
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=k,
        kv_cache_dtype=None,
        draft_lm_head_quantization=None,
        draft_model_config=SimpleNamespace(
            architectures=[arch],
            hf_config=SimpleNamespace(
                model_type=draft,
                architectures=[arch],
                num_nextn_predict_layers=modules,
            ),
        ),
        model=None,
        draft_tensor_parallel_size=None,
        quantization=None,
    )
    spec.__dict__.update(fields)
    spec.use_multi_module_mtp = lambda: method == "mtp" and min(modules, k) > 1
    return spec


V3 = "deepseek-ai/DeepSeek-V3"


def test_deepseek_v3_mtp_is_timed_as_aisimulates_mtp_depth(fake_aisimulate):
    p = provider(engine_config(V3, DEEPSEEK_V3, speculative=mtp(k=2)))
    assert fake_aisimulate.instances[0].config.kwargs["nextn"] == 2
    record = p.assumptions()["aisimulate"]
    assert record["speculation"]["verify_width"] == 3
    assert any("k + 1 tokens" in a for a in record["approximations"])
    # Without speculation: no depth, no speculation record or step fields.
    plain = provider(engine_config(V3, DEEPSEEK_V3))
    assert plain.config["nextn"] == 0
    assert plain.assumptions()["aisimulate"]["speculation"] is None
    cost = plain.estimate(sim_step(0, sim_req("d", 1, 10, 5)))
    assert cost.breakdown == {"aisimulate_ms": 7.0}


def test_a_step_records_its_drafts_against_the_width(fake_aisimulate):
    p = provider(engine_config(V3, DEEPSEEK_V3, speculative=mtp(k=2)))
    full = dataclasses.replace(sim_req("a", 3, 100, 50), num_spec_tokens=2)
    fewer = dataclasses.replace(sim_req("b", 2, 100, 50), num_spec_tokens=1)
    none = sim_req("c", 1, 100, 50)
    cost = p.estimate(sim_step(0, full, fewer, none, sim_req("p", 64, 0, 64)))
    assert cost.breakdown["spec_draft_tokens"] == 3.0
    assert cost.breakdown["spec_requests_below_width"] == 2.0
    # The decode requests stay logical requests (the estimator applies k + 1).
    work = fake_aisimulate.instances[0].queries[-1]["scheduled_requests"]
    assert work["num_decode_requests"] == 3
    assert work["extend_lengths"] == [3, 2, 1, 64]


@pytest.mark.parametrize(
    "target, spec, problem",
    [
        (V3, mtp(method="ngram"), "method ngram"),
        (V3, mtp(method="eagle3", draft="llama"), "method eagle3"),
        (V3, mtp(k=2, modules=2), "multi-module MTP"),
        # A checkpoint of several MTP layers, even at depth 1.
        (V3, mtp(modules=2), r"\[1, 2\] MTP layers"),
        (V3, mtp(arch="DeepSeekV4MTPModel"), "draft architectures"),
        (V3, mtp(k=6), "6 speculative tokens"),
        (V3, mtp(model="org/other-mtp"), "draft checkpoint"),
        (V3, mtp(draft_tensor_parallel_size=2), "draft tensor parallelism 2"),
        (V3, mtp(quantization="fp8"), "draft quantization fp8"),
        ("Qwen/Qwen3-30B-A3B", mtp(), "target"),
    ],
)
def test_other_speculative_configurations_are_refused(
    fake_aisimulate, target, spec, problem
):
    local = DEEPSEEK_V3 if target == V3 else QWEN3_MOE
    with pytest.raises(ValueError, match=problem):
        provider(engine_config(target, local, speculative=spec))


def test_dp_attention_with_expert_parallelism_is_mapped(fake_aisimulate):
    provider(engine_config(data_parallel_size=8, enable_expert_parallel=True))
    config = fake_aisimulate.instances[0].config.kwargs
    assert (
        config["tp"],
        config["attention_dp"],
        config["moe_tp_size"],
        config["moe_ep_size"],
    ) == (1, 8, 1, 8)


def supervised(monkeypatch, rank, size):
    """As the multi-port external-LB supervisor exports a server's place on
    its host."""
    monkeypatch.setenv("VLLM_DP_SUPERVISED_RANK_LOCAL", str(rank))
    monkeypatch.setenv("VLLM_DP_SUPERVISED_SIZE_LOCAL", str(size))


def supervised_rank_config(dp, **overrides):
    """A DP server launched by the multi-port supervisor: external LB, one
    local rank."""
    return engine_config(
        **{
            "data_parallel_size": dp,
            "data_parallel_size_local": 1,
            "data_parallel_external_lb": True,
            "enable_expert_parallel": True,
            **overrides,
        }
    )


def test_ranks_a_supervisor_launched_on_one_host_are_local(fake_aisimulate, monkeypatch):
    """A server for each DP rank on one host is the layout of the ranks one
    server launches: the same model configuration, the same joint step."""
    together = provider(engine_config(data_parallel_size=8, enable_expert_parallel=True))
    supervised(monkeypatch, 3, 8)
    apart = provider(supervised_rank_config(8))
    one, other = fake_aisimulate.instances
    assert one.config.kwargs == other.config.kwargs
    step = sim_step(
        0,
        on_rank(sim_req("p", 1024, 0, 1024), 0),
        on_rank(sim_req("d", 1, 4000, 3000), 5),
    )
    assert together.estimate(step) == apart.estimate(step)
    assert one.queries[-1] == other.queries[-1]
    assert len(one.queries[-1]) == 8


@pytest.mark.parametrize(
    "exported, match",
    [
        (None, "1 local"),  # a rank on a host of its own
        ((1, 4), "4 local"),  # half of the ranks on this host
    ],
)
def test_ranks_on_other_hosts_stay_refused(fake_aisimulate, monkeypatch, exported, match):
    if exported:
        supervised(monkeypatch, *exported)
    with pytest.raises(ValueError, match=match):
        provider(supervised_rank_config(8))


def test_a_supervised_place_needs_an_external_lb_rank_server(fake_aisimulate, monkeypatch):
    supervised(monkeypatch, 0, 8)
    with pytest.raises(ValueError, match="multi-port external-LB supervisor"):
        provider(engine_config(data_parallel_size=8, enable_expert_parallel=True))


def test_startup_probes_cover_a_prefill_and_a_decode_step(fake_aisimulate):
    p = provider(engine_config())
    prefill, decode = fake_aisimulate.instances[0].queries
    assert prefill["scheduled_requests"]["sum_prefill_tokens"] == 1024
    assert decode["scheduled_requests"]["num_decode_requests"] == 4
    record = p.assumptions()["aisimulate"]["startup_probes"]
    assert [(r["kind"], r["ms"]) for r in record] == [("prefill", 7.0), ("decode", 7.0)]
    saved = record[0]["metrics"]["scheduled_requests"]
    assert (saved["extend_lengths"], saved["past_kv_lengths"]) == ([1024], [0])
    saved = record[1]["metrics"]["scheduled_requests"]
    assert (saved["extend_lengths"], saved["past_kv_lengths"]) == ([1] * 4, [1024] * 4)
    # Data-parallel attention: rank 0 busy, the others idle, in order.
    p = provider(engine_config(data_parallel_size=4, enable_expert_parallel=True))
    saved = p.assumptions()["aisimulate"]["startup_probes"][0]["metrics"]
    busy = [r["scheduled_requests"]["num_prefill_requests"] for r in saved]
    assert busy == [1, 0, 0, 0]
    assert fake_aisimulate.instances[1].queries[0] == saved


def test_unresolved_v4_geometry_has_no_fallback(fake_aisimulate):
    from vllm.v1.worker.simulated.aisimulate_timing import V4_FIELDS, _geometry

    config = HfConfig({name: 1 for name in V4_FIELDS if name != "head_dim"})
    config["hidden_size"] = 4096
    with pytest.raises(ValueError, match="head_dim"):
        _geometry(config, V4_FIELDS)


def test_decilm_geometry_is_each_layers_attention(fake_aisimulate):
    """Llama-3.3-Nemotron-Super (DeciLM): each layer's attention by its block
    config, none, linear or of its own KV grouping (a window, sinks), so a
    layer of another kind or grouping differs."""
    from vllm.v1.worker.simulated.aisimulate_timing import DECILM_FIELDS, _geometry

    def attention(no_op=False, linear=False, group=8, window=None, sinks=None):
        return SimpleNamespace(
            attention=SimpleNamespace(
                no_op=no_op,
                replace_with_linear=linear,
                n_heads_in_group=None if no_op or linear else group,
                window_length=window,
                num_sink_tokens=sinks,
            )
        )

    def config(blocks):
        return SimpleNamespace(
            num_hidden_layers=len(blocks),
            num_attention_heads=64,
            hidden_size=8192,
            block_configs=blocks,
        )

    published = [attention(), attention(no_op=True), attention(), attention(linear=True)]
    geometry = _geometry(config(published), DECILM_FIELDS)
    assert geometry["block_attention"] == [
        (False, False, 8, None, None),
        (True, False, None, None, None),
        (False, False, 8, None, None),
        (False, True, None, None, None),
    ]
    for changed in (attention(group=4), attention(no_op=True), attention(window=1024)):
        other = _geometry(config([changed] + published[1:]), DECILM_FIELDS)
        assert other["block_attention"] != geometry["block_attention"]


@pytest.mark.parametrize("ms", [None, math.nan, 0.0])
def test_a_failing_startup_probe_is_refused(fake_aisimulate, monkeypatch, ms):
    monkeypatch.setattr(fake_aisimulate, "ms", ms)
    with pytest.raises(ValueError, match="for a prefill step"):
        provider(engine_config())


def test_an_estimator_without_data_for_the_shapes_is_refused(
    fake_aisimulate, monkeypatch
):
    def missing(self, metrics):
        raise RuntimeError("perf database error: MoE data missing")

    monkeypatch.setattr(fake_aisimulate, "estimate_forward_pass_time_ms", missing)
    with pytest.raises(
        ValueError, match="cannot time a prefill step.*MoE data missing"
    ):
        provider(engine_config())


def test_an_unready_estimator_is_refused(fake_aisimulate, monkeypatch):
    monkeypatch.setattr(fake_aisimulate, "readiness", "unsupported_config")
    monkeypatch.setattr(fake_aisimulate, "source", "fallback_regression")
    with pytest.raises(ValueError, match="no timing"):
        provider(engine_config())


@pytest.mark.parametrize("mode", ["interpolation", "regression", None])
def test_only_op_level_estimates_are_taken(fake_aisimulate, monkeypatch, mode):
    """The provider asks for op-level estimates and refuses any other
    resolution, or none recorded."""
    diagnostics = fake_aisimulate.diagnostics

    def resolved(self):
        d = diagnostics(self)
        d["provenance"]["config"]["estimation_mode"] = mode
        return d

    monkeypatch.setattr(fake_aisimulate, "diagnostics", resolved)
    with pytest.raises(ValueError, match=f"estimation mode {mode!r}, not op_level"):
        provider(engine_config())
    assert fake_aisimulate.instances[-1].config.kwargs["estimation_mode"] == "op_level"


def test_a_missing_package_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "aisimulate_core", None)
    with pytest.raises(ImportError, match="aisimulate package"):
        provider(engine_config())


@pytest.mark.parametrize(
    "args, match",
    [
        ({"host_us": -1.0}, "host_us"),
        ({"host_us": math.inf}, "host_us"),
        ({"gpu": "l4"}, "GPUs"),
    ],
)
def test_bad_arguments_are_refused(fake_aisimulate, args, match):
    with pytest.raises(ValueError, match=match):
        provider(engine_config(), **args)


def on_rank(request, rank):
    return dataclasses.replace(request, dp_rank=rank)


def test_a_step_becomes_one_workload(fake_aisimulate):
    p = provider(engine_config(), host_us=50.0)
    step = sim_step(
        3,
        sim_req("fresh", 512, 0, 1000),  # a first chunk
        sim_req("chunk", 488, 512, 1000),  # its continuation
        sim_req("handoff", 1, 1023, 1024),  # the native N - 1 handoff
        sim_req("decode", 1, 2000, 1500),  # decode: context 2000
        sim_req("decode2", 1, 1000, 900),
    )
    cost = p.estimate(step)
    _, _, query = fake_aisimulate.instances[0].queries  # two startup probes first
    work = query["scheduled_requests"]
    assert (work["num_prefill_requests"], work["sum_prefill_tokens"]) == (3, 1001)
    assert work["sum_prefill_kv_tokens"] == 512 + 1023
    prompts = [1000, 1000, 1024]
    mean = sum(prompts) / 3
    assert work["var_prefill_length"] == pytest.approx(
        sum((x - mean) ** 2 for x in prompts) / 3
    )
    assert (work["num_decode_requests"], work["sum_decode_kv_tokens"]) == (2, 3000)
    assert work["var_decode_kv_tokens"] == pytest.approx(250000.0)
    assert work["extend_lengths"] == [512, 488, 1, 1, 1]
    assert work["past_kv_lengths"] == [0, 512, 1023, 2000, 1000]
    assert cost.device_s == pytest.approx(0.007)
    assert cost.host_s == pytest.approx(50e-6)
    assert cost.breakdown == {"aisimulate_ms": 7.0}


def test_data_parallel_ranks_are_passed_in_order_idle_included(fake_aisimulate):
    p = provider(engine_config(data_parallel_size=4, enable_expert_parallel=True))
    step = sim_step(
        0,
        on_rank(sim_req("a", 1, 100, 50), 2),
        on_rank(sim_req("b", 256, 0, 256), 0),
        on_rank(sim_req("c", 1, 300, 50), 2),
    )
    p.estimate(step)
    _, _, query = fake_aisimulate.instances[0].queries  # two startup probes first
    assert isinstance(query, list) and len(query) == 4
    counts = [
        (
            q["scheduled_requests"]["num_prefill_requests"],
            q["scheduled_requests"]["num_decode_requests"],
        )
        for q in query
    ]
    assert counts == [(1, 0), (0, 0), (0, 2), (0, 0)]
    with pytest.raises(ValueError, match="rank 4 of 4"):
        p.estimate(sim_step(1, on_rank(sim_req("d", 1, 10, 5), 4)))


def test_an_empty_step_costs_nothing_without_a_query(fake_aisimulate):
    p = provider(engine_config())
    assert p.estimate(sim_step(0)).device_s == 0.0
    assert len(fake_aisimulate.instances[0].queries) == 2  # the startup probes only


@pytest.mark.parametrize("ms", [None, math.nan, math.inf, -1.0])
def test_an_invalid_estimate_raises(fake_aisimulate, monkeypatch, ms):
    p = provider(engine_config())
    monkeypatch.setattr(fake_aisimulate, "ms", ms)
    with pytest.raises(ValueError, match="AISimulate returned"):
        p.estimate(sim_step(0, sim_req("r", 1, 10, 5)))


def test_it_is_chosen_by_name(fake_aisimulate):
    p = get_timing_provider("aisimulate", {"gpu": "h100"}, engine_config())
    assert type(p).__name__ == "AisimulateTimingProvider"


# --- The real pinned package -------------------------------------------------


def bundled_config(target):
    import os

    import aisimulate_core

    path = os.path.join(
        os.path.dirname(aisimulate_core.__file__),
        "model_configs",
        f"{target.replace('/', '--')}_config.json",
    )
    with open(path) as f:
        return json.load(f)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_the_real_package_prices_handoff_mixed_and_uneven_dp_steps():
    import importlib.metadata

    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    # Qwen3-30B-A3B, the decode role: the consumer's first step is the
    # native N - 1 handoff (a one-token prefill), then mixed/chunked work.
    qwen = bundled_config("Qwen/Qwen3-30B-A3B")
    p = provider(engine_config("Qwen/Qwen3-30B-A3B", qwen, role="kv_consumer"))
    record = p.assumptions()["aisimulate"]
    assert record["package_version"] == importlib.metadata.version("aisimulate")
    assert (record["source"], record["readiness"]) == ("aic", "ready")
    handoff = p.estimate(sim_step(0, sim_req("h", 1, 1023, 1024))).device_s
    mixed = p.estimate(
        sim_step(
            1,
            sim_req("h", 1, 1023, 1024),
            sim_req("c", 512, 512, 2048),
            sim_req("d", 1, 3000, 2000),
        )
    ).device_s
    assert 0 < handoff < mixed < 1.0

    # DeepSeek-V3 on 8 local DP ranks with EP: uneven ranks, idle ones.
    v3 = bundled_config("deepseek-ai/DeepSeek-V3")
    p = provider(
        engine_config(
            "deepseek-ai/DeepSeek-V3",
            v3,
            data_parallel_size=8,
            enable_expert_parallel=True,
        )
    )
    busy = p.estimate(
        sim_step(
            0,
            on_rank(sim_req("p", 1024, 0, 1024), 0),
            on_rank(sim_req("d", 1, 4000, 3000), 5),
        )
    ).device_s
    light = p.estimate(sim_step(1, on_rank(sim_req("d", 1, 4000, 3000), 5))).device_s
    # The same ranks as a server each (a multi-port supervisor's): the same costs.
    with pytest.MonkeyPatch.context() as patch:
        supervised(patch, 5, 8)
        apart = provider(
            supervised_rank_config(8, model="deepseek-ai/DeepSeek-V3", local=v3)
        )
        assert apart.assumptions() == p.assumptions()
        assert (
            apart.estimate(
                sim_step(
                    0,
                    on_rank(sim_req("p", 1024, 0, 1024), 0),
                    on_rank(sim_req("d", 1, 4000, 3000), 5),
                )
            ).device_s,
            apart.estimate(
                sim_step(1, on_rank(sim_req("d", 1, 4000, 3000), 5))
            ).device_s,
        ) == (busy, light)
    assert 0 < light < busy

    # The weight and attention modes passed are the ones AISimulate infers
    # for the published checkpoints: the same estimate with and without them,
    # at the local KV cache's precision (ours, passed explicitly).
    for target, kwargs in (
        ("deepseek-ai/DeepSeek-V3", {"tp": 8, "moe_tp_size": 8, "moe_ep_size": 1}),
        ("openai/gpt-oss-20b", {"tp": 1}),
        ("Qwen/Qwen3-30B-A3B", {"tp": 1, "moe_tp_size": 1, "moe_ep_size": 1}),
    ):
        local = bundled_config(target)
        parallel = {"tensor_parallel_size": kwargs["tp"]}
        explicit = provider(engine_config(target, local, **parallel))
        default = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=target,
                system="h100_sxm",
                backend="vllm",
                worker_type="aggregated",
                estimation_mode="auto",
                fallback_policy="deny",
                kvcache_quant_mode=explicit.config["kvcache_quant_mode"],
                **kwargs,
            )
        )
        step = sim_step(0, sim_req("p", 1024, 0, 1024), sim_req("d", 1, 2048, 1024))
        from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics

        assert explicit.estimate(step).device_s * 1000 == pytest.approx(
            default.estimate_forward_pass_time_ms(rank_metrics(list(step.requests))),
            rel=1e-9,
        ), target
    # AISimulate's own default for DeepSeek-V3 is an FP8 KV cache; the local
    # BF16 cache is what is timed.
    fp8_kv = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="deepseek-ai/DeepSeek-V3",
            system="h100_sxm",
            backend="vllm",
            worker_type="aggregated",
            tp=8,
            moe_tp_size=8,
            moe_ep_size=1,
            kvcache_quant_mode="fp8",
        )
    )
    v3_default = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="deepseek-ai/DeepSeek-V3",
            system="h100_sxm",
            backend="vllm",
            worker_type="aggregated",
            tp=8,
            moe_tp_size=8,
            moe_ep_size=1,
        )
    )
    work = rank_metrics(list(step.requests))
    assert v3_default.estimate_forward_pass_time_ms(
        work
    ) == fp8_kv.estimate_forward_pass_time_ms(work)
    v3_modes = provider(
        engine_config("deepseek-ai/DeepSeek-V3", v3, tensor_parallel_size=8)
    )
    assert v3_modes.config["kvcache_quant_mode"] == "bfloat16"
    assert v3_modes.config["moe_quant_mode"] == "fp8_block"
    oss = provider(
        engine_config("openai/gpt-oss-20b", bundled_config("openai/gpt-oss-20b"))
    )
    assert oss.config["moe_quant_mode"] == "w4a16_mxfp4"


def real_engine_config(tmp_path, name, raw, derived=False, **kwargs):
    """A real engine configuration of a model directory holding `raw`."""
    from vllm.engine.arg_utils import EngineArgs

    directory = model_dir(tmp_path, name, raw, derived=derived)
    return EngineArgs(
        model=directory,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={"simulate_forward": True},
        **kwargs,
    ).create_engine_config()


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_real_configs_are_compared_through_their_own_classes(tmp_path):
    # Positive controls: Qwen3 (defaults spelled out) and GPT-oss, a BF16
    # copy of its MXFP4 checkpoint, layer types generated by its class.
    qwen = bundled_config("Qwen/Qwen3-30B-A3B")
    provider(real_engine_config(tmp_path, "Qwen--Qwen3-30B-A3B", qwen))
    oss = {
        k: v
        for k, v in bundled_config("openai/gpt-oss-20b").items()
        if k not in ("quantization_config", "layer_types")
    }
    p = provider(
        real_engine_config(tmp_path, "openai--gpt-oss-20b-bf16", oss, derived=True),
        target_model="openai/gpt-oss-20b",
    )
    assert p.config["moe_quant_mode"] == "w4a16_mxfp4"
    # DeepSeek-V3, native and the BF16 derivative: its class gives the MLA
    # layout (the raw JSON has no head size; none is invented).
    v3 = bundled_config("deepseek-ai/DeepSeek-V3")
    native = provider(
        real_engine_config(tmp_path, "deepseek-ai--DeepSeek-V3", v3),
        gpu="h200",
    )
    assert (
        native.assumptions()["aisimulate"]["compared_geometry"]["kv_lora_rank"] == 512
    )
    bf16 = {k: v for k, v in v3.items() if k != "quantization_config"}
    p = provider(
        real_engine_config(
            tmp_path, "deepseek-ai--DeepSeek-V3-bf16", bf16, derived=True
        ),
        target_model="deepseek-ai/DeepSeek-V3",
    )
    assert p.config["gemm_quant_mode"] == "fp8_block"
    # Mixtral leaves its head size unset; the model's fallback, hidden size
    # over heads, is compared: implicit and explicit 128 pass, a hidden size
    # giving 256 does not.
    mixtral = bundled_config("mistralai/Mixtral-8x7B-v0.1")
    assert "head_dim" not in mixtral or mixtral["head_dim"] is None
    for name, local in (
        ("implicit", mixtral),
        ("explicit", {**mixtral, "head_dim": 128}),
    ):
        p = provider(
            real_engine_config(
                tmp_path, f"mistralai--Mixtral-{name}", local, derived=True
            ),
            target_model="mistralai/Mixtral-8x7B-v0.1",
        )
        assert p.assumptions()["aisimulate"]["compared_geometry"]["head_dim"] == 128
    with pytest.raises(ValueError, match="head_dim"):
        provider(
            real_engine_config(
                tmp_path,
                "mistralai--Mixtral-wide",
                {**mixtral, "hidden_size": 8192},
                derived=True,
            ),
            target_model="mistralai/Mixtral-8x7B-v0.1",
        )
    # A changed latent or rope width is not the target's KV.
    for field, value in (("kv_lora_rank", 256), ("qk_rope_head_dim", 32)):
        changed = {**bf16, field: value}
        with pytest.raises(ValueError, match=field):
            provider(
                real_engine_config(
                    tmp_path, f"deepseek-ai--DeepSeek-V3-{field}", changed, derived=True
                ),
                target_model="deepseek-ai/DeepSeek-V3",
            )


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize(
    "target, gpu", [("Qwen/Qwen3.5-27B", "h100"), ("Qwen/Qwen3.5-35B-A3B", "h200")]
)
def test_a_multimodal_model_is_timed_as_its_language_model_only(tmp_path, target, gpu):
    """Qwen3.5, published as a vision-language model: its language model's
    geometry (the nested text config) is compared, and it is timed only
    when no multimodal input can reach the encoders AISimulate does not
    time."""
    raw = bundled_config(target)
    text = raw["text_config"]
    name = target.replace("/", "--")

    def config(suffix, local=raw, **kwargs):
        return real_engine_config(
            tmp_path, f"{name}-{suffix}", local, derived=True, **kwargs
        )

    for suffix, kwargs in (
        ("text", {"language_model_only": True}),
        ("zero", {"limit_mm_per_prompt": {"image": 0, "video": 0}}),
    ):
        p = provider(config(suffix, **kwargs), gpu=gpu, target_model=target)
        geometry = p.assumptions()["aisimulate"]["compared_geometry"]
        assert geometry["linear_num_value_heads"] == text["linear_num_value_heads"]
        assert geometry["num_key_value_heads"] == text["num_key_value_heads"]
    for suffix, kwargs in (
        ("default", {}),
        ("image-off", {"limit_mm_per_prompt": {"image": 0}}),
        (
            "embeds",
            {"limit_mm_per_prompt": {"image": 0, "video": 0}, "enable_mm_embeds": True},
        ),
    ):
        with pytest.raises(ValueError, match="language model only"):
            provider(config(suffix, **kwargs), gpu=gpu, target_model=target)
    # A changed delta-net width is not the target's state.
    wide = {**raw, "text_config": {**text, "linear_num_value_heads": 64}}
    with pytest.raises(ValueError, match="linear_num_value_heads"):
        provider(
            config("wide", wide, language_model_only=True), gpu=gpu, target_model=target
        )


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_an_engine_runs_on_aisimulate_timing(tmp_path, monkeypatch):
    """`timing: aisimulate` through the engine: Qwen3-30B-A3B built on the
    meta device, one request completed, each step priced by AISimulate."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    directory = model_dir(
        tmp_path, "Qwen--Qwen3-30B-A3B", bundled_config("Qwen/Qwen3-30B-A3B")
    )
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=directory,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        kv_cache_memory_bytes=1 << 30,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": {"gpu": "h100"},
            "log_path": str(log),
        },
    )
    try:
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 1024}],
            SamplingParams(max_tokens=4, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 4
        records = [json.loads(line) for line in log.read_text().splitlines()]
        (timing,) = [r["timing"] for r in records if "timing" in r]
        assert timing["aisimulate"]["target_model"] == "Qwen/Qwen3-30B-A3B"
        assert timing["aisimulate"]["source"] == "aic"
        steps = [r for r in records if "step" in r]
        assert steps and all(r["cost"]["breakdown"]["aisimulate_ms"] > 0 for r in steps)
    finally:
        del llm


V4 = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_deepseek_v4_ratios_are_compared_as_the_native_helper_resolves_them(
    tmp_path,
):
    published = bundled_config(V4)
    p = provider(
        real_engine_config(tmp_path, "deepseek-ai--DeepSeek-V4-Flash", published),
        gpu="b200",
    )
    record = p.assumptions()["aisimulate"]
    assert record["compared_geometry"]["compress_ratios"][:4] == [0, 0, 4, 128]
    assert (record["expert_dtype"], p.config["moe_quant_mode"]) == (
        "fp4",
        "w4a8_mxfp4_mxfp8",
    )
    assert (record["kv_cache_format"], p.config["kvcache_quant_mode"]) == (
        "fp8_ds_mla",
        "fp8",
    )
    # The legacy spelling: layer 2's ratio 4 -> 128.
    legacy = {**published, "compress_ratios": list(published["compress_ratios"])}
    legacy["compress_ratios"][2] = 128
    with pytest.raises(ValueError, match="compress_ratios"):
        provider(
            real_engine_config(tmp_path, "deepseek-ai--DeepSeek-V4-legacy", legacy),
            gpu="b200",
            target_model=V4,
        )
    # The class's own spelling (layer types and their rates), one rate changed.
    from vllm.engine.arg_utils import EngineArgs  # noqa: F401

    native = (
        type(
            real_engine_config(
                tmp_path, "deepseek-ai--DeepSeek-V4-x", published
            ).model_config.hf_text_config
        )
        .from_dict(published)
        .to_dict()
    )
    native.pop("compress_ratios", None)
    rate = next(k for k, v in native["compress_rates"].items() if v == 4)
    native["compress_rates"] = {**native["compress_rates"], rate: 128}
    with pytest.raises(ValueError, match="compress_ratios"):
        provider(
            real_engine_config(tmp_path, "deepseek-ai--DeepSeek-V4-rates", native),
            gpu="b200",
            target_model=V4,
        )


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_deepseek_v4_modes_and_cache_precision_with_the_real_package():
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics

    published = HfConfig(bundled_config(V4))
    tp4 = engine_config(V4, published, tensor_parallel_size=4)
    explicit = provider(tp4, gpu="b200")
    # BF16 attention over the packed FP8 cache resolves to op-level estimates.
    resolved = explicit.assumptions()["aisimulate"]["resolved_config"]
    assert (resolved["estimation_mode"], resolved["kvcache_quant_mode"]) == (
        "op_level",
        "fp8",
    )
    default = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=V4,
            system="b200_sxm",
            backend="vllm",
            worker_type="aggregated",
            tp=4,
            moe_tp_size=4,
            moe_ep_size=1,
            fallback_policy="deny",
        )
    )
    step = sim_step(0, sim_req("p", 1024, 0, 1024), sim_req("d", 1, 2048, 1024))
    assert explicit.estimate(step).device_s * 1000 == pytest.approx(
        default.estimate_forward_pass_time_ms(rank_metrics(list(step.requests))),
        rel=1e-9,
    )
    # An explicit FP8 cache is the same; BF16 has no V4 tables.
    provider(engine_config(V4, published, cache_dtype="fp8"), gpu="b200")
    with pytest.raises(ValueError, match="packed FP8 cache"):
        provider(engine_config(V4, published, cache_dtype="bfloat16"), gpu="b200")
    # Ready on H200 at TP1, but no MoE data for its shapes: refused at startup.
    with pytest.raises(ValueError, match="cannot time a"):
        provider(engine_config(V4, published), gpu="h200")


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_a_deepseek_v4_engine_runs_on_aisimulate_timing(
    tmp_path, monkeypatch, cache_dtype
):
    """The provider is built before the model (cache dtype still `auto`):
    V4-Flash on the meta device, one request, every step priced."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    directory = model_dir(
        tmp_path, "deepseek-ai--DeepSeek-V4-Flash", bundled_config(V4)
    )
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=directory,
        skip_tokenizer_init=True,
        max_model_len=4096,
        kv_cache_memory_bytes=4 << 30,
        kv_cache_dtype=cache_dtype,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": {"gpu": "b200"},
            "log_path": str(log),
        },
    )
    try:
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 1024}],
            SamplingParams(max_tokens=4, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 4
        records = [json.loads(line) for line in log.read_text().splitlines()]
        (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
        assert (timing["kv_cache_dtype"], timing["kv_cache_format"]) == (
            cache_dtype,
            "fp8_ds_mla",
        )
        steps = [r for r in records if "step" in r and r["step"]["requests"]]
        costs = [r["cost"]["breakdown"]["aisimulate_ms"] for r in steps]
        assert steps and all(math.isfinite(c) and c > 0 for c in costs)
    finally:
        del llm


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize("k", [1, 3])
def test_deepseek_v3_mtp_depths_match_the_pinned_estimator(k):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics

    local = HfConfig(bundled_config(V3))
    p = provider(
        engine_config(V3, local, tensor_parallel_size=8, speculative=mtp(k=k)),
        gpu="h200",
    )
    direct = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=V3,
            system="h200_sxm",
            backend="vllm",
            worker_type="aggregated",
            tp=8,
            moe_tp_size=8,
            moe_ep_size=1,
            nextn=k,
            kvcache_quant_mode="bfloat16",
            fallback_policy="deny",
        )
    )
    drafted = dataclasses.replace(sim_req("d", 1 + k, 1024, 512), num_spec_tokens=k)
    step = sim_step(
        0, *[dataclasses.replace(drafted, req_id=f"d{i}") for i in range(8)]
    )
    ms = p.estimate(step).device_s * 1000
    assert ms == pytest.approx(
        direct.estimate_forward_pass_time_ms(rank_metrics(list(step.requests))),
        rel=1e-9,
    )
    # A partial-width step is priced at the configured width.
    partial = sim_step(1, *[sim_req(f"e{i}", 1, 1024, 512) for i in range(8)])
    assert p.estimate(partial).device_s * 1000 == pytest.approx(ms, rel=1e-9)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize(
    "target, local",
    [
        ("deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2"),
        ("zai-org/GLM-5.2-FP8", "zai-org/GLM-5.2"),
    ],
)
def test_dsa_sparse_mla_is_timed_with_its_indexer_geometry(tmp_path, target, local):
    """DeepSeek-V3.2's and GLM-5's sparse MLA (DSA), timed as the published
    FP8 deployment from a BF16 local copy: the latent's and the indexer's
    geometry compared (GLM-5's indexer layers included), op-level estimates
    over a BF16 cache."""
    raw = {k: v for k, v in bundled_config(local).items() if k != "quantization_config"}
    name = local.replace("/", "--")

    def config(suffix, changed=raw, **kwargs):
        return real_engine_config(
            tmp_path, f"{name}-{suffix}", changed, derived=True,
            tensor_parallel_size=8, **kwargs,
        )

    p = provider(config("bf16"), gpu="h200", target_model=target)
    a = p.assumptions()["aisimulate"]
    assert (a["resolved_config"]["estimation_mode"], a["config"]["kvcache_quant_mode"]) == (
        "op_level", "bfloat16",
    )
    for field in ("index_n_heads", "index_head_dim", "index_topk", "index_topk_freq"):
        assert a["compared_geometry"][field] == raw.get(field)
    # Another indexer is not the target's.
    with pytest.raises(ValueError, match="index_n_heads"):
        provider(
            config("indexer", {**raw, "index_n_heads": raw["index_n_heads"] * 2}),
            gpu="h200", target_model=target,
        )
    if local == "zai-org/GLM-5.2":
        # On B200 its `auto` cache is FP8 sparse MLA, not timed; BF16 is.
        with pytest.raises(ValueError, match="kv_cache_dtype bfloat16"):
            provider(config("b200"), gpu="b200", target_model=target)
        provider(config("b200-bf16", kv_cache_dtype="bfloat16"), gpu="b200",
                 target_model=target)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_a_bundled_config_the_package_would_download_is_read_locally(tmp_path):
    """Kimi-K2-Instruct's and Mixtral's configs ship with the package, which
    lists them as not pre-downloaded and would fetch the published ones:
    the provider hands it a directory of its own bundled config instead."""
    import json as json_module
    import os

    from aisimulate_core.sdk.common import DefaultHFModels

    for target, gpu, tp in (
        ("moonshotai/Kimi-K2-Instruct", "b200", 8),
        ("mistralai/Mixtral-8x7B-v0.1", "h100", 2),
    ):
        assert target not in DefaultHFModels
        raw = {k: v for k, v in bundled_config(target).items() if k != "quantization_config"}
        p = provider(
            real_engine_config(tmp_path, target.replace("/", "--") + "-local", raw,
                               derived=True, tensor_parallel_size=tp),
            gpu=gpu, target_model=target,
        )
        a = p.assumptions()["aisimulate"]
        directory = p.config["model"]
        assert os.path.isdir(directory) and a["target_config_source"] == "bundled, local copy"
        with open(os.path.join(directory, "config.json")) as f:
            assert json_module.load(f) == bundled_config(target)
        assert a["resolved_config"]["estimation_mode"] == "op_level"
    # A listed target is named as before.
    p = provider(engine_config(), gpu="h100")
    assert p.config["model"] == "Qwen/Qwen3-30B-A3B"


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize(
    "target, gpu, kwargs, field, changed",
    [
        # Gemma4: its global layers' KV heads (a per-layer field its
        # heterogeneous config reads only as published).
        ("google/gemma-4-26B-A4B", "h100", {"language_model_only": True},
         "num_global_key_value_heads", 4),
        ("XiaomiMiMo/MiMo-V2-Flash", "h200", {}, "swa_num_key_value_heads", 16),
        ("meta-models/Muse-Glimmer-30B", "h200", {"language_model_only": True},
         "sliding_window", 4096),
    ],
)
def test_sliding_window_families_are_timed_with_their_layer_kinds(
    tmp_path, target, gpu, kwargs, field, changed
):
    """Gemma4, MiMo-V2-Flash and Muse-Glimmer: sliding and full layers of
    their own KV heads and widths, compared kind by kind (a changed one is not
    the target's); op-level estimates over a BF16 cache."""
    raw = {k: v for k, v in bundled_config(target).items() if k != "quantization_config"}
    name = target.replace("/", "--")

    def config(suffix, local):
        return real_engine_config(tmp_path, f"{name}-{suffix}", local, derived=True, **kwargs)

    p = provider(config("bf16", raw), gpu=gpu, target_model=target)
    a = p.assumptions()["aisimulate"]
    assert (a["resolved_config"]["estimation_mode"], a["config"]["kvcache_quant_mode"]) == (
        "op_level", "bfloat16",
    )
    text = raw.get("text_config", raw)
    if field == "num_global_key_value_heads":
        # Held per layer: the global layers' own heads and width.
        per_layer = a["compared_geometry"]["per_layer_kv"]
        full = [i for i, kind in enumerate(text["layer_types"]) if kind == "full_attention"]
        assert {per_layer[i] for i in full} == {(text["global_head_dim"], text[field])}
        assert {per_layer[i] for i in range(len(per_layer)) if i not in full} == {
            (text["head_dim"], text["num_key_value_heads"])
        }
        field = "per_layer_kv"
    else:
        assert a["compared_geometry"][field] == text[field]
    changed_text = {**text, ("num_global_key_value_heads" if field == "per_layer_kv" else field): changed}
    local = {**raw, "text_config": changed_text} if "text_config" in raw else changed_text
    with pytest.raises(ValueError, match=field):
        provider(config("changed", local), gpu=gpu, target_model=target)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_mimo_v2_flash_is_priced_with_its_own_layer_kinds(tmp_path):
    """MiMo-V2-Flash marks its sliding layers 1, AISimulate's hybrid graph its
    global ones: the estimator reads a normalized copy (recorded), whose graph
    prices the model's 9 full (one dense, eight MoE) and 39 sliding MoE
    layers, while the geometry compared is the published config's."""
    from aisimulate_core.sdk import config as sdk_config
    from aisimulate_core.sdk.models import get_model

    target = "XiaomiMiMo/MiMo-V2-Flash"
    raw = {k: v for k, v in bundled_config(target).items() if k != "quantization_config"}
    local = real_engine_config(tmp_path, "mimo-bf16", raw, derived=True)
    p = provider(local, gpu="h200", target_model=target)
    a = p.assumptions()["aisimulate"]
    assert a["target_config_source"] == "bundled, normalized local copy"
    assert a["timing_input"]["transformation"].startswith("hybrid_layer_pattern inverted")
    assert a["compared_geometry"]["hybrid_layer_pattern"] == raw["hybrid_layer_pattern"]
    with open(os.path.join(p.config["model"], "config.json")) as f:
        read = json.load(f)
    assert read["hybrid_layer_pattern"] == [1 - v for v in raw["hybrid_layer_pattern"]]
    assert {k: v for k, v in read.items() if k != "hybrid_layer_pattern"} == {
        k: v for k, v in bundled_config(target).items() if k != "hybrid_layer_pattern"
    }
    model = get_model(
        p.config["model"],
        sdk_config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1),
        "vllm",
    )
    assert model._count_layer_types() == {
        "global_dense": 1, "global_moe": 8, "swa_moe": 39, "swa_dense": 0,
    }
    # Each phase's attention: (layers, window, KV heads) of each kind.
    window = raw["sliding_window_size"]
    for ops in (model.context_ops, model.generation_ops):
        attention = sorted(
            (op._scale_factor, op._window_size, op._n_kv)
            for op in ops
            if type(op).__name__.endswith("Attention")
        )
        assert attention == [
            (1.0, 0, raw["num_key_value_heads"]),
            (8.0, 0, raw["num_key_value_heads"]),
            (39.0, window, raw["swa_num_key_value_heads"]),
        ]


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize(
    "target, gpu, tp, kwargs, derived, modes",
    [
        # MiniMax-M2.x: a BF16 copy of its FP8 checkpoint, timed as published
        # (AISimulate has no TP8 for it).
        ("MiniMaxAI/MiniMax-M2.5", "h200", 4, {}, True, ("fp8_block", "fp8_block")),
        ("MiniMaxAI/MiniMax-M2.7", "h200", 4, {}, True, ("fp8_block", "fp8_block")),
        # Qwen3-VL, dense and MoE, served as its language model.
        ("Qwen/Qwen3-VL-8B-Instruct", "h100", 1, {"language_model_only": True}, False,
         ("bfloat16", "bfloat16")),
        ("Qwen/Qwen3-VL-30B-A3B-Instruct", "h100", 2, {"language_model_only": True}, False,
         ("bfloat16", "bfloat16")),
    ],
)
def test_full_attention_families_are_timed_as_published(
    tmp_path, target, gpu, tp, kwargs, derived, modes
):
    """MiniMax-M2.x and Qwen3-VL: full attention of their own KV heads,
    compared (a changed head count is not the target's); op-level estimates
    of the published precision over a BF16 cache."""
    raw = bundled_config(target)
    local_raw = (
        {k: v for k, v in raw.items() if k not in ("quantization_config", "auto_map")}
        if derived
        else raw
    )
    name = target.replace("/", "--")

    def config(suffix, local):
        return real_engine_config(
            tmp_path, f"{name}-{suffix}", local, derived=derived,
            tensor_parallel_size=tp, **kwargs,
        )

    p = provider(config("local", local_raw), gpu=gpu, target_model=target)
    a = p.assumptions()["aisimulate"]
    assert a["resolved_config"]["estimation_mode"] == "op_level"
    assert (a["config"]["gemm_quant_mode"], a["config"]["moe_quant_mode"]) == modes
    assert a["config"]["kvcache_quant_mode"] == "bfloat16"
    text = local_raw.get("text_config", local_raw)
    assert a["compared_geometry"]["num_key_value_heads"] == text["num_key_value_heads"]
    changed_text = {**text, "num_key_value_heads": text["num_key_value_heads"] // 2}
    changed = {**local_raw, "text_config": changed_text} if "text_config" in local_raw else changed_text
    with pytest.raises(ValueError, match="num_key_value_heads"):
        provider(config("changed", changed), gpu=gpu, target_model=target)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize("tp", [4, 8])
def test_kimi_k3_is_timed_with_its_mla_modules(tmp_path, tp):
    """Kimi-K3 (KDA and MLA layers, latent SiTU experts): a BF16 copy of its
    MXFP4-expert checkpoint timed as published (BF16 GEMMs, w4a16_mxfp4
    experts) over a BF16 cache, its MLA and linear attention geometry
    compared (a changed KDA head width is not the target's). AISimulate
    prices its MLA layers from MLA-module rows, falling back silently to a
    GQA stack where a module row is missing: the rows answer, both phases."""
    from aisimulate_core.sdk import config as sdk_config
    from aisimulate_core.sdk.engine import _evaluate_single_op
    from aisimulate_core.sdk.models import get_model
    from aisimulate_core.sdk.operations.overlap import FallbackOp
    from aisimulate_core.sdk.perf_database import get_database

    target = "moonshotai/Kimi-K3"
    raw = bundled_config(target)
    text = {k: v for k, v in raw["text_config"].items() if k != "quantization_config"}
    local = {**raw, "text_config": text}

    def config(suffix, local):
        return real_engine_config(
            tmp_path, f"kimi-k3-{suffix}", local, derived=True,
            language_model_only=True, tensor_parallel_size=tp,
        )

    p = provider(config("bf16", local), gpu="b200", target_model=target)
    a = p.assumptions()["aisimulate"]
    assert a["resolved_config"]["estimation_mode"] == "op_level"
    c = a["config"]
    assert (c["gemm_quant_mode"], c["moe_quant_mode"], c["kvcache_quant_mode"]) == (
        "bfloat16", "w4a16_mxfp4", "bfloat16",
    )
    assert a["compared_geometry"]["linear_attn_config"] == text["linear_attn_config"]
    assert a["compared_geometry"]["kv_lora_rank"] == text["kv_lora_rank"]

    model = get_model(
        p.config["model"],
        sdk_config.ModelConfig(tp_size=tp, moe_tp_size=1, moe_ep_size=tp),
        "vllm",
    )
    # The data the estimator resolved (backend_version None: its default).
    database = get_database("b200_sxm", "vllm", a["resolved_config"]["backend_version"])
    for context, ops, shape in (
        (True, model.context_ops, {"batch_size": 1, "s": 1024, "prefix": 0}),
        (False, model.generation_ops, {"batch_size": 4, "s": 1024, "prefix": 0}),
    ):
        (block,) = [op for op in ops if isinstance(op, FallbackOp)]
        result = _evaluate_single_op(
            database, block._primary, is_context=context, x=None,
            imbalance_correction_scale=1.0, **shape,
        )
        assert float(getattr(result, "latency", result)) > 0

    kda = {**text["linear_attn_config"], "head_dim": 64}
    changed = {**raw, "text_config": {**text, "linear_attn_config": kda}}
    with pytest.raises(ValueError, match="linear_attn_config"):
        provider(config("changed", changed), gpu="b200", target_model=target)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_each_caller_reads_its_own_bundled_snapshot(tmp_path):
    """A bundled config the package would download is copied, with its
    quantization sidecar or explicitly without one, into a snapshot private
    to its caller, intact for as long as the caller keeps it: ranks starting
    together never read another's half-written copy."""
    import json as json_module
    import os
    import threading

    from vllm.v1.worker.simulated.aisimulate_timing import _offline_model

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    config = {"architectures": ["MixtralForCausalLM"], "pad": "x" * 200_000}
    with_sidecar = bundle / "org--with_config.json"
    with_sidecar.write_text(json_module.dumps(config))
    (bundle / "org--with_hf_quant_config.json").write_text('{"quantization": {}}')
    without = bundle / "org--without_config.json"
    without.write_text(json_module.dumps(config))

    model, snapshot = _offline_model("org/with", str(with_sidecar))
    assert sorted(os.listdir(model)) == ["config.json", "hf_quant_config.json"]
    # No sidecar bundled: none in the snapshot, whatever another one held.
    other, other_snapshot = _offline_model("org/without", str(without))
    assert other != model and os.listdir(other) == ["config.json"]
    # A listed model is named, not copied.
    assert _offline_model("Qwen/Qwen3-30B-A3B", "unused") == ("Qwen/Qwen3-30B-A3B", None)

    errors, held = [], []

    def start():
        try:
            directory, keep = _offline_model("org/with", str(with_sidecar))
            held.append(keep)
            for _ in range(50):
                with open(os.path.join(directory, "config.json")) as f:
                    assert json_module.load(f) == config
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=start) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len({keep.name for keep in held}) == 8
    # The first snapshot is still whole while kept, gone once released.
    with open(os.path.join(model, "config.json")) as f:
        assert json_module.load(f) == config
    snapshot.cleanup()
    assert not os.path.exists(model)


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_kimi_k2_5_is_timed_with_its_mixed_precision(tmp_path):
    """Kimi-K2.5 (a vision-language wrapper served as its language model), a
    BF16 local copy timed as the published checkpoint: AISimulate's own
    reading of its compressed-tensors groups, BF16 GEMMs and int4 experts,
    and the nested text config's MLA geometry compared."""
    target = "moonshotai/Kimi-K2.5"
    raw = bundled_config(target)
    raw = {**raw, "text_config": {k: v for k, v in raw["text_config"].items()
                                  if k != "quantization_config"}}
    raw.pop("quantization_config", None)
    p = provider(
        real_engine_config(tmp_path, "moonshotai--Kimi-K2.5-bf16", raw, derived=True,
                           tensor_parallel_size=8, language_model_only=True,
                           trust_remote_code=True),
        gpu="b200", target_model=target,
    )
    a = p.assumptions()["aisimulate"]
    assert {k: a["config"][k] for k in ("gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode")} == {
        "gemm_quant_mode": "bfloat16", "moe_quant_mode": "int4_wo", "fmha_quant_mode": "bfloat16",
    }
    assert a["resolved_config"]["estimation_mode"] == "op_level"
    assert a["compared_geometry"]["kv_lora_rank"] == raw["text_config"]["kv_lora_rank"]


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize(
    "target, local",
    [
        ("deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2"),
        ("zai-org/GLM-5.2-FP8", "zai-org/GLM-5.2"),
    ],
)
def test_dsa_mtp_depths_match_the_pinned_estimator(tmp_path, target, local, k):
    """DeepSeek-V3.2's and GLM-5's one MTP layer is AISimulate's MTP depth k,
    its estimates the estimator's own (op-level, BF16 cache)."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics

    raw = {key: v for key, v in bundled_config(local).items() if key != "quantization_config"}
    p = provider(
        real_engine_config(
            tmp_path, f"{local.replace('/', '--')}-mtp{k}", raw, derived=True,
            tensor_parallel_size=8,
            speculative_config={"method": "mtp", "num_speculative_tokens": k},
        ),
        gpu="h200",
        target_model=target,
    )
    speculation = p.assumptions()["aisimulate"]["speculation"]
    assert (p.config["nextn"], speculation["draft_architectures"]) == (
        k, ["DeepseekV32MTPModel"],
    )
    direct = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=target, system="h200_sxm", backend="vllm", worker_type="aggregated",
            tp=8, moe_tp_size=8, moe_ep_size=1, nextn=k,
            gemm_quant_mode="fp8_block", moe_quant_mode="fp8_block",
            fmha_quant_mode="bfloat16", comm_quant_mode="half",
            kvcache_quant_mode="bfloat16", estimation_mode="op_level",
            fallback_policy="deny",
        )
    )
    drafted = dataclasses.replace(sim_req("d", 1 + k, 1024, 512), num_spec_tokens=k)
    step = sim_step(0, *[dataclasses.replace(drafted, req_id=f"d{i}") for i in range(8)])
    assert p.estimate(step).device_s * 1000 == pytest.approx(
        direct.estimate_forward_pass_time_ms(rank_metrics(list(step.requests))), rel=1e-9,
    )


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize("k", [1, 3])
def test_qwen3_5_mtp_depths_match_the_pinned_estimator(tmp_path, k):
    """Qwen3.5's one MTP layer (in its text config), served as its language
    model, is AISimulate's MTP depth k like DeepSeek's."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    from vllm.v1.worker.simulated.aisimulate_timing import rank_metrics

    target = "Qwen/Qwen3.5-35B-A3B"
    p = provider(
        real_engine_config(
            tmp_path,
            f"Qwen--Qwen3.5-35B-A3B-mtp{k}",
            bundled_config(target),
            derived=True,
            language_model_only=True,
            speculative_config={"method": "mtp", "num_speculative_tokens": k},
        ),
        gpu="h200",
        target_model=target,
    )
    speculation = p.assumptions()["aisimulate"]["speculation"]
    assert (p.config["nextn"], speculation["draft_architectures"]) == (
        k,
        ["Qwen3_5MoeMTP"],
    )
    # Its drafter priced as the hybrid target's layers: stated.
    assert any(
        "full-attention layer, is priced as the hybrid target's" in a
        for a in p.assumptions()["aisimulate"]["approximations"]
    )
    direct = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model=target,
            system="h200_sxm",
            backend="vllm",
            worker_type="aggregated",
            tp=1,
            moe_tp_size=1,
            moe_ep_size=1,
            nextn=k,
            kvcache_quant_mode="bfloat16",
            estimation_mode="op_level",
            fallback_policy="deny",
        )
    )
    drafted = dataclasses.replace(sim_req("d", 1 + k, 1024, 512), num_spec_tokens=k)
    step = sim_step(
        0, *[dataclasses.replace(drafted, req_id=f"d{i}") for i in range(8)]
    )
    assert p.estimate(step).device_s * 1000 == pytest.approx(
        direct.estimate_forward_pass_time_ms(rank_metrics(list(step.requests))),
        rel=1e-9,
    )


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_a_deepseek_v3_mtp_engine_runs_on_aisimulate_timing(tmp_path, monkeypatch):
    """Native single-module MTP through the engine: drafts scheduled and
    accepted (sampled), every step priced at depth 1."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    directory = model_dir(tmp_path, "deepseek-ai--DeepSeek-V3", bundled_config(V3))
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=directory,
        skip_tokenizer_init=True,
        max_model_len=4096,
        kv_cache_memory_bytes=1 << 30,
        enable_prefix_caching=False,
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        simulation_config={
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": {"gpu": "h200"},
            "spec_acceptance_rates": [0.7],
            "log_path": str(log),
        },
    )
    try:
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 512}],
            SamplingParams(max_tokens=16, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 16
        records = [json.loads(line) for line in log.read_text().splitlines()]
        (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
        assert timing["config"]["nextn"] == 1
        assert timing["speculation"]["verify_width"] == 2
        steps = [r for r in records if "step" in r and r["step"]["requests"]]
        drafted = [r for r in steps if r["cost"]["breakdown"].get("spec_draft_tokens")]
        assert drafted, "no step scheduled drafts"
        assert all(r["cost"]["breakdown"]["aisimulate_ms"] > 0 for r in steps)
        # Accepted drafts are recorded (sampled), not priced.
        assert any(r["cost"].get("num_accepted") for r in drafted)
        # The rank's own acceptance records: every drafted request, each step,
        # as the step's counts.
        logged = {
            r["spec_acceptance"]["step_index"]: r["spec_acceptance"]["requests"]
            for r in records
            if "spec_acceptance" in r
        }
        for r in drafted:
            expected = {
                q["req_id"]: [q["num_spec_tokens"], a]
                for q, a in zip(r["step"]["requests"], r["cost"]["num_accepted"])
                if q["num_spec_tokens"]
            }
            assert logged.pop(r["step"]["step_index"]) == expected
        assert not logged
    finally:
        del llm


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_deepseek_v3_mtp_draft_precision_through_native_configs(tmp_path):
    """Real `EngineArgs`/`SpeculativeConfig`: the draft inheriting the
    target's precision is accepted; its own KV cache dtype or LM-head
    quantization is not."""
    v3 = bundled_config(V3)
    spec = {"method": "mtp", "num_speculative_tokens": 1}
    config = real_engine_config(
        tmp_path, "deepseek-ai--DeepSeek-V3", v3, speculative_config=spec
    )
    assert config.speculative_config.draft_model_config.architectures == [
        "DeepSeekMTPModel"
    ]
    p = provider(config, gpu="h200")
    assert p.config["nextn"] == 1
    for name, extra, problem in (
        ("kv", {"kv_cache_dtype": "fp8"}, "draft KV cache dtype fp8"),
        (
            "head",
            {"draft_lm_head_quantization": "fp8"},
            "draft LM head quantization fp8",
        ),
    ):
        config = real_engine_config(
            tmp_path,
            f"deepseek-ai--DeepSeek-V3-{name}",
            v3,
            speculative_config={**spec, **extra},
        )
        with pytest.raises(ValueError, match=problem):
            provider(config, gpu="h200", target_model=V3)


V4_FLASH = "deepseek-ai/DeepSeek-V4-Flash"


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
def test_a_deepseek_v4_mtp_engine_builds_its_draft_layer_on_aisimulate_timing(
    tmp_path, monkeypatch
):
    """Native single-module MTP of DeepSeek-V4 through the engine: the CPU
    drafter is built for simulation, its one MTP layer (sliding-window,
    compress ratio 0) a KV cache layer of the target's groups, every step
    priced at depth 1 with V4's draft architecture."""
    from vllm import LLM, SamplingParams
    from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    v4 = bundled_config(V4_FLASH)
    directory = model_dir(tmp_path, "deepseek-ai--DeepSeek-V4-Flash", v4)
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=directory,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        kv_cache_memory_bytes=2 << 30,
        enable_prefix_caching=False,
        enforce_eager=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        simulation_config={
            "simulate_forward": True,
            "timing": "aisimulate",
            "timing_args": {"gpu": "b200"},
            "spec_acceptance_rates": [0.7],
            "log_path": str(log),
        },
    )
    try:
        mtp_layer = v4["num_hidden_layers"]
        groups = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
        drafted = [
            (group.kv_cache_spec, name)
            for group in groups.kv_cache_groups
            for name in group.layer_names
            if f".layers.{mtp_layer}." in name
        ]
        assert [name for _, name in drafted] == [f"model.layers.{mtp_layer}.attn.swa_cache"]
        ((spec, _),) = drafted
        assert isinstance(spec, SlidingWindowMLASpec)
        assert (spec.sliding_window, spec.block_size) == (128, 64)
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 512}],
            SamplingParams(max_tokens=16, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 16
        records = [json.loads(line) for line in log.read_text().splitlines()]
        (timing,) = [r["timing"]["aisimulate"] for r in records if "timing" in r]
        assert timing["config"]["nextn"] == 1
        assert timing["speculation"]["draft_architectures"] == ["DeepSeekV4MTPModel"]
        steps = [r for r in records if "step" in r and r["step"]["requests"]]
        assert any(r["cost"]["breakdown"].get("spec_draft_tokens") for r in steps)
        assert all(r["cost"]["breakdown"]["aisimulate_ms"] > 0 for r in steps)
    finally:
        del llm


def test_deepseek_v4_mtp_does_not_run_on_cpu_outside_simulation():
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v4.cpu.mtp import DeepSeekV4MTP

    with pytest.raises(NotImplementedError, match="not implemented on CPU"):
        DeepSeekV4MTP(vllm_config=VllmConfig())


@pytest.mark.parametrize(
    "target, quant, modes",
    [
        ("nvidia/Qwen3-30B-A3B-NVFP4", "nvfp4", ("nvfp4", "nvfp4")),
        ("nvidia/Qwen3-30B-A3B-FP8", "fp8", ("fp8_static", "fp8")),
        ("nvidia/Qwen3-30B-A3B-Mixed", "mixed_precision", ("fp8_block", "nvfp4")),
    ],
)
def test_a_sidecar_only_quantization_is_read_and_timed_as_the_checkpoints(
    fake_aisimulate, target, quant, modes
):
    """A ModelOpt checkpoint may declare its quantization in
    `hf_quant_config.json` only: it is read from the full config (never as
    unquantized, timed BF16) and its modes are AISimulate's own reading;
    attention stays BF16 and the KV mode the engine's cache dtype."""
    p = provider(engine_config(target), gpu="b200")
    config = fake_aisimulate.instances[0].config.kwargs
    assert (config["gemm_quant_mode"], config["moe_quant_mode"]) == modes
    assert config["fmha_quant_mode"] == "bfloat16"
    assert config["kvcache_quant_mode"] == "bfloat16"
    record = p.assumptions()["aisimulate"]
    assert record["target_quantization"] == quant
    assert f"({quant} weights)" in record["deployment"]


def test_a_quantization_without_modes_is_refused(fake_aisimulate):
    with pytest.raises(ValueError, match="does not map .*'w4a8_awq'"):
        provider(engine_config("nvidia/Qwen3-30B-A3B-Other"))


@pytest.mark.skipif(not REAL, reason="aisimulate is not installed")
@pytest.mark.parametrize(
    "target, gpu, tp, quant, modes",
    [
        ("nvidia/MiniMax-M2.5-NVFP4", "b200", 4, "nvfp4", ("nvfp4", "nvfp4")),
        ("nvidia/Qwen3-235B-A22B-NVFP4", "b200", 8, "nvfp4", ("nvfp4", "nvfp4")),
        # Quantization in the sidecar only: read as unquantized before.
        ("nvidia/Llama-3.1-70B-Instruct-FP8", "h200", 4, "fp8", ("fp8_static", "fp8")),
    ],
)
def test_modelopt_checkpoints_are_timed_with_their_own_precision(
    tmp_path, target, gpu, tp, quant, modes
):
    """ModelOpt NVFP4 and per-tensor FP8 checkpoints, timed as published:
    AISimulate's own reading of the full config, op-level estimates."""
    raw = bundled_config(target)
    p = provider(
        real_engine_config(
            tmp_path, target.replace("/", "--"), raw, tensor_parallel_size=tp
        ),
        gpu=gpu,
    )
    a = p.assumptions()["aisimulate"]
    assert a["target_quantization"] == quant
    assert (a["config"]["gemm_quant_mode"], a["config"]["moe_quant_mode"]) == modes
    assert a["config"]["fmha_quant_mode"] == "bfloat16"
    assert a["resolved_config"]["estimation_mode"] == "op_level"
    step = sim_step(0, sim_req("a", 1024, 0, 1024), sim_req("b", 1, 512, 512))
    assert p.estimate(step).device_s > 0
