# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roofline step timing of dense, MoE and MLA models on a GPU preset, and of
data-parallel attention with expert parallelism."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.simulated.utils import sim_req, sim_step
from vllm.v1.worker.simulated.timing import (
    EXPERT_PARALLEL_BACKENDS,
    LATENT_ATTENTION_BACKENDS,
    ExpertGeometry,
    LatentAttentionGeometry,
    LayerGeometry,
    ROUTING_MODEL,
    ModelGeometry,
    RooflineTimingProvider,
    UnsupportedModelLayout,
    merge_dp_descriptors,
    rank_reach_probability,
)

# Qwen2.5-7B in bf16, one rank.
QWEN_7B = ModelGeometry(
    num_layers=28,
    hidden_size=3584,
    num_heads=28,
    num_kv_heads=4,
    head_size=128,
    intermediate_size=18944,
    vocab_size=152064,
    weight_bytes=2,
    kv_bytes=2,
)
WEIGHT_BYTES = 14_140_571_648  # decoder layers and LM head
KV_BYTES_PER_TOKEN = 57_344  # 28 layers x K and V x 4 heads x 128 x 2 bytes


def h100(**kwargs):
    return RooflineTimingProvider(None, gpu="h100", geometry=QWEN_7B, **kwargs)


def test_geometry_of_a_dense_model():
    assert QWEN_7B.kv_bytes_per_token == KV_BYTES_PER_TOKEN
    assert (28 * QWEN_7B.layer_params + 3584 * 152064) * 2 == WEIGHT_BYTES


def qwen_7b_config(cache_dtype, dtype=torch.bfloat16):
    model = SimpleNamespace(
        dtype=dtype,
        use_mla=False,
        is_moe=False,
        is_hybrid=False,
        is_attention_free=False,
        hf_text_config=SimpleNamespace(intermediate_size=18944),
        get_num_layers=lambda parallel: 28,
        get_total_num_hidden_layers=lambda: 28,
        get_hidden_size=lambda: 3584,
        get_num_attention_heads=lambda parallel: 28,
        get_num_kv_heads=lambda parallel: 4,
        get_head_size=lambda: 128,
        get_vocab_size=lambda: 152064,
    )
    return SimpleNamespace(
        model_config=model,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, enable_expert_parallel=False
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


@pytest.mark.parametrize(
    "cache_dtype,kv_bytes",
    [("auto", 2), ("bfloat16", 2), ("float16", 2), ("float32", 4), ("fp8", 1)],
)
def test_kv_bytes_follow_the_cache_dtype(cache_dtype, kv_bytes):
    geometry = ModelGeometry.from_config(qwen_7b_config(cache_dtype))
    assert geometry.kv_bytes == kv_bytes
    assert geometry.kv_bytes_per_token == KV_BYTES_PER_TOKEN // 2 * kv_bytes


@pytest.mark.parametrize("cache_dtype", ["int4_per_token_head", "fp8_ds_mla"])
def test_packed_kv_dtypes_are_rejected(cache_dtype):
    with pytest.raises(ValueError, match=cache_dtype):
        ModelGeometry.from_config(qwen_7b_config(cache_dtype))


def test_decode_reads_the_weights_and_the_context():
    # One token for a request with 999 tokens of KV: memory-bound.
    cost = h100().estimate(sim_step(0, sim_req("r", 1, 999, 1000)))
    memory_us = (WEIGHT_BYTES + 1001 * KV_BYTES_PER_TOKEN) / (3.35e12 * 0.8) * 1e6
    assert cost.breakdown["memory_us"] == pytest.approx(memory_us)
    assert cost.breakdown["compute_us"] < memory_us
    assert cost.device_s == pytest.approx((20 + memory_us) * 1e-6)
    assert cost.device_s == pytest.approx(5.318e-3, rel=1e-3)


def test_prefill_is_compute_bound():
    # A 1000-token prompt: layers, one sampled token, causal attention.
    cost = h100().estimate(sim_step(0, sim_req("r", 1000, 0, 1000)))
    flops = (
        2 * 28 * QWEN_7B.layer_params * 1000
        + 2 * 3584 * 152064
        + 4 * 28 * 28 * 128 * (1000 * 1001 // 2)
    )
    compute_us = flops / (989e12 * 0.6) * 1e6
    assert cost.breakdown["compute_us"] == pytest.approx(compute_us)
    assert cost.breakdown["memory_us"] < compute_us
    assert cost.device_s == pytest.approx(22.353e-3, rel=1e-3)


def test_a_chunk_attends_to_the_computed_context():
    first = h100().estimate(sim_step(0, sim_req("r", 500, 0, 1000)))
    second = h100().estimate(sim_step(1, sim_req("r", 500, 500, 1000)))
    assert second.breakdown["compute_us"] > first.breakdown["compute_us"]


def test_presets_and_overrides():
    slow = h100(peak_tflops=98.9, hbm_tbps=0.335)
    step = sim_step(0, sim_req("r", 1, 999, 1000))
    assert slow.estimate(step).breakdown["memory_us"] == pytest.approx(
        10 * h100().estimate(step).breakdown["memory_us"]
    )
    with pytest.raises(ValueError, match="Unknown GPU"):
        RooflineTimingProvider(None, gpu="tpu", geometry=QWEN_7B)
    with pytest.raises(ValueError, match="needs gpu"):
        RooflineTimingProvider(None, geometry=QWEN_7B)


def test_assumptions_are_resolved():
    assumptions = h100(compute_efficiency=0.5, hbm_tbps=4.0).assumptions()
    assert assumptions["gpu"] == "h100"
    assert assumptions["peak_tflops"] == 989.0  # the preset's
    assert assumptions["hbm_tbps"] == 4.0  # overridden
    assert assumptions["compute_efficiency"] == 0.5
    assert assumptions["bandwidth_efficiency"] == 0.8
    assert assumptions["fixed_us"] == 20.0
    assert assumptions["geometry"]["kv_bytes"] == 2
    assert assumptions["geometry"]["num_kv_heads"] == 4


# MoE layers, from the built model. A 64-expert, top-4 MoE on Qwen-7B's
# attention: one dense layer, then 27 with experts and a shared expert.
EXPERT = ExpertGeometry(
    local_experts=64, global_experts=64, top_k=4, params_per_expert=3 * 3584 * 1024
)
ATTENTION = 2 * 3584 * 28 * 128 + 2 * 3584 * 4 * 128
SHARED = 3 * 3584 * 2048 + 3584 * 64 + 3584  # shared expert, router, its gate
MOE = dataclasses.replace(
    QWEN_7B,
    layers=(LayerGeometry(QWEN_7B.layer_params),)
    + (LayerGeometry(ATTENTION + SHARED, (EXPERT,)),) * 27,
)


def moe(**kwargs):
    return RooflineTimingProvider(None, gpu="h100", geometry=MOE, **kwargs)


@pytest.mark.parametrize(
    "step",
    [
        sim_step(0, sim_req("r", 1000, 0, 1000)),  # prefill
        sim_step(0, sim_req("r", 1, 999, 1000)),  # decode
        sim_step(0, sim_req("a", 1, 99, 100), sim_req("b", 300, 0, 300)),  # mixed
    ],
)
def test_dense_layers_of_the_built_model_time_as_the_config(step):
    built = dataclasses.replace(
        QWEN_7B, layers=(LayerGeometry(QWEN_7B.layer_params),) * 28
    )
    from_config = h100().estimate(step)
    from_model = RooflineTimingProvider(None, gpu="h100", geometry=built).estimate(step)
    assert from_model.device_s == pytest.approx(from_config.device_s, rel=1e-12)
    assert from_model.breakdown == pytest.approx(from_config.breakdown, rel=1e-12)


def test_routing_boundaries():
    # No tokens read no expert, one token its top-k, many tokens all of them.
    assert EXPERT.expected_read(0) == 0
    assert EXPERT.expected_read(1) == pytest.approx(4)
    assert EXPERT.expected_read(2) == pytest.approx(64 * (1 - (60 / 64) ** 2))
    assert EXPERT.expected_read(10_000) == pytest.approx(64)
    every = ExpertGeometry(local_experts=8, global_experts=8, top_k=8, params_per_expert=1)
    assert every.expected_read(0) == 0
    assert every.expected_read(1) == every.expected_read(1000) == 8


def test_a_token_goes_through_the_dense_weights_and_top_k_experts():
    dense = QWEN_7B.layer_params + 27 * (ATTENTION + SHARED)
    assert MOE.token_params == dense + 27 * 4 * EXPERT.params_per_expert
    assert MOE.weights_read(0) == dense
    assert MOE.weights_read(1) == pytest.approx(dense + 27 * 4 * EXPERT.params_per_expert)
    assert MOE.weights_read(10_000) == pytest.approx(
        dense + 27 * 64 * EXPERT.params_per_expert
    )


def test_moe_decode_reads_the_routed_experts_only():
    cost = moe().estimate(sim_step(0, sim_req("r", 1, 999, 1000)))
    weights = (MOE.weights_read(1) + 3584 * 152064) * 2
    memory_us = (weights + 1001 * KV_BYTES_PER_TOKEN) / (3.35e12 * 0.8) * 1e6
    assert cost.breakdown["memory_us"] == pytest.approx(memory_us)
    # Fewer bytes than a step reading every expert.
    assert MOE.weights_read(1) < MOE.weights_read(10_000)


def test_moe_prefill_computes_top_k_experts_per_token():
    cost = moe().estimate(sim_step(0, sim_req("r", 1000, 0, 1000)))
    flops = (
        2 * MOE.token_params * 1000
        + 2 * 3584 * 152064
        + 4 * 28 * 28 * 128 * (1000 * 1001 // 2)
    )
    assert cost.breakdown["compute_us"] == pytest.approx(flops / (989e12 * 0.6) * 1e6)
    weights = (MOE.weights_read(1000) + 3584 * 152064) * 2
    memory_us = (weights + 2000 * KV_BYTES_PER_TOKEN) / (3.35e12 * 0.8) * 1e6
    assert cost.breakdown["memory_us"] == pytest.approx(memory_us)


def test_layers_are_recorded_grouped():
    geometry = moe().assumptions()["geometry"]
    assert geometry["source"] == "built model"
    # The routing model the expert reads rest on.
    assert geometry["routing"] == ROUTING_MODEL
    assert geometry["routing"]["model"] == "independent_uniform_top_k"
    assert "routing" not in h100().assumptions()["geometry"]
    assert h100().assumptions()["geometry"]["source"] == "config"
    layers = geometry["layers"]
    assert layers == [
        {"first": 0, "last": 0, "dense_params": QWEN_7B.layer_params, "experts": []},
        {
            "first": 1,
            "last": 27,
            "dense_params": ATTENTION + SHARED,
            "experts": [dataclasses.asdict(EXPERT)],
        },
    ]


def test_an_moe_model_is_timed_from_its_built_layers():
    config = qwen_7b_config("auto")
    config.model_config.is_moe = True
    with pytest.raises(ValueError, match="from_model"):
        ModelGeometry.from_config(config)
    unbound = RooflineTimingProvider(config, gpu="h100")
    with pytest.raises(RuntimeError, match="bind_model"):
        unbound.estimate(sim_step(0, sim_req("r", 1, 999, 1000)))


def write_config(directory, config):
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps({"max_position_embeddings": 256, "vocab_size": 128, **config})
    )
    return str(directory)


def timing_record(
    tmp_path, model_dir, monkeypatch, tensor_parallel_size=1, steps=None, block_size=16
):
    """The timing record an in-process simulated engine writes, after one
    request's steps (the output rank's), which `steps` receives."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        block_size=block_size,
        kv_cache_memory_bytes=8 * 1024**2,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
        simulation_config={
            "simulate_forward": True,
            "timing": "roofline",
            "timing_args": {"gpu": "h100"},
            "log_path": str(log),
        },
    )
    try:
        output = llm.generate(
            [{"prompt_token_ids": [5] * 40}],
            SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output[0].outputs[0].token_ids) == 2
    finally:
        del llm
    records = [json.loads(line) for line in log.read_text().splitlines()]
    if steps is not None:
        steps.extend(r for r in records if "step" in r)
    timings = [r["timing"] for r in records if "timing" in r]
    assert len(timings) == 1
    # Recorded before any step.
    first_step = next(i for i, r in enumerate(records) if "step" in r)
    assert next(i for i, r in enumerate(records) if "timing" in r) < first_step
    return timings[0]


def test_a_built_dense_model_matches_its_config(tmp_path, monkeypatch):
    model_dir = write_config(
        tmp_path / "llama",
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
    )
    geometry = timing_record(tmp_path, model_dir, monkeypatch)["geometry"]
    layer_params = 2 * 256 * 2 * 128 + 2 * 256 * 2 * 128 + 3 * 256 * 512
    assert geometry["layers"] == [
        {"first": 0, "last": 1, "dense_params": layer_params, "experts": []}
    ]


QWEN2_MOE = {
    "architectures": ["Qwen2MoeForCausalLM"],
    "model_type": "qwen2_moe",
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [0],
    "norm_topk_prob": False,
    "tie_word_embeddings": False,
}


def test_a_built_moe_model_counts_each_weight_once(tmp_path, monkeypatch):
    # Layer 0 dense (mlp_only_layers), 1-3 MoE with a shared expert whose
    # gate the block also holds: counted once.
    model_dir = write_config(tmp_path / "qwen2-moe", QWEN2_MOE)
    geometry = timing_record(tmp_path, model_dir, monkeypatch)["geometry"]
    assert geometry["source"] == "built model"
    assert geometry["routing"]["model"] == "independent_uniform_top_k"
    attention = 64 * (64 + 2 * 32) + 64 * 64  # QKV, O
    dense_mlp = 64 * 2 * 96 + 96 * 64
    router, shared_gate, shared = 64 * 8, 64 * 1, 64 * 2 * 48 + 48 * 64
    assert geometry["layers"] == [
        {"first": 0, "last": 0, "dense_params": attention + dense_mlp, "experts": []},
        {
            "first": 1,
            "last": 3,
            "dense_params": attention + router + shared_gate + shared,
            "experts": [
                {
                    "local_experts": 8,
                    "global_experts": 8,
                    "top_k": 2,
                    "params_per_expert": 3 * 64 * 32,
                }
            ],
        },
    ]


class Block(torch.nn.Module):
    """A decoder layer of two matrix multiplications, 8x16 and 16x8."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16, bias=False, device="meta")
        self.down = torch.nn.Linear(16, 8, bias=False, device="meta")


def decoder(name, **others):
    """A model whose 28 decoder layers are the list `transformer.<name>`."""
    transformer = torch.nn.Module()
    transformer.add_module(name, torch.nn.ModuleList(Block() for _ in range(28)))
    for other, module in others.items():
        transformer.add_module(other, module)
    model = torch.nn.Module()
    model.transformer = transformer
    return model


def test_decoder_layers_under_another_name_are_found():
    # EXAONE's decoder is transformer.h.
    provider = RooflineTimingProvider(qwen_7b_config("auto"), gpu="h100")
    provider.bind_model(decoder("h"))
    assert provider.geometry.layers == (LayerGeometry(2 * 8 * 16),) * 28
    assert provider.assumptions()["geometry"]["source"] == "built model"


@pytest.mark.parametrize(
    "model",
    [
        torch.nn.Module(),  # no list of 28 layers
        decoder("h", g=torch.nn.ModuleList(Block() for _ in range(28))),  # two
    ],
)
def test_a_dense_model_whose_layers_are_not_found_keeps_its_config(model):
    config = qwen_7b_config("auto")
    provider = RooflineTimingProvider(config, gpu="h100")
    provider.bind_model(model)
    assert provider.geometry == ModelGeometry.from_config(config)
    assert provider.assumptions()["geometry"]["source"] == "config"
    step = sim_step(0, sim_req("r", 1000, 0, 1000))
    assert provider.estimate(step).device_s == h100().estimate(step).device_s


def test_an_moe_model_whose_layers_are_not_found_is_rejected():
    config = qwen_7b_config("auto")
    config.model_config.is_moe = True
    provider = RooflineTimingProvider(config, gpu="h100")
    with pytest.raises(UnsupportedModelLayout, match="decoder layers"):
        provider.bind_model(torch.nn.Module())


def test_one_rank_communicates_nothing():
    for step in (
        sim_step(0, sim_req("r", 1000, 0, 1000)),
        sim_step(0, sim_req("r", 1, 999, 1000)),
    ):
        cost = h100().estimate(step)
        assert cost.breakdown["communication_us"] == 0
        assert cost.device_s == pytest.approx(
            (20 + max(cost.breakdown["compute_us"], cost.breakdown["memory_us"])) * 1e-6
        )


@pytest.mark.parametrize("ranks", [2, 4])
def test_tensor_parallel_ranks_all_reduce_twice_per_layer(ranks):
    geometry = dataclasses.replace(QWEN_7B, tensor_parallel_size=ranks)
    provider = RooflineTimingProvider(None, gpu="h100", geometry=geometry)
    cost = provider.estimate(sim_step(0, sim_req("r", 1000, 0, 1000)))
    payload = 1000 * 3584 * 2  # tokens x hidden x bf16 activations
    per_collective = 10e-6 + 2 * (ranks - 1) / ranks * payload / (450e9 * 0.8)
    communication_us = 2 * 28 * per_collective * 1e6
    assert cost.breakdown["communication_us"] == pytest.approx(communication_us)
    assert cost.device_s == pytest.approx(
        (
            20
            + max(cost.breakdown["compute_us"], cost.breakdown["memory_us"])
            + communication_us
        )
        * 1e-6
    )
    # Overrides: the link's peak, efficiency and latency.
    slow = RooflineTimingProvider(
        None,
        gpu="h100",
        geometry=geometry,
        interconnect_gbs=45.0,
        interconnect_efficiency=1.0,
        collective_latency_us=0.0,
    ).estimate(sim_step(0, sim_req("r", 1000, 0, 1000)))
    assert slow.breakdown["communication_us"] == pytest.approx(
        2 * 28 * 2 * (ranks - 1) / ranks * payload / 45e9 * 1e6
    )


def test_all_reduces_carry_the_hidden_width_in_the_activation_dtype():
    # Qwen3-30B-A3B: 32 query heads of 128 (4096 wide), hidden width 2048.
    config = qwen_7b_config("fp8", dtype=torch.float32)
    geometry = ModelGeometry.from_config(config)
    assert (geometry.activation_bytes, geometry.kv_bytes, geometry.weight_bytes) == (4, 1, 4)
    qwen3 = dataclasses.replace(
        QWEN_7B, hidden_size=2048, num_heads=32, tensor_parallel_size=2
    )
    for activation_bytes in (2, 4):
        provider = RooflineTimingProvider(
            None,
            gpu="h100",
            geometry=dataclasses.replace(qwen3, activation_bytes=activation_bytes),
            collective_latency_us=0.0,
        )
        assert provider.communication_s(100) == pytest.approx(
            2 * 28 * (2 * 1 / 2) * 100 * 2048 * activation_bytes / (450e9 * 0.8)
        )


def test_tensor_parallelism_needs_an_interconnect():
    geometry = dataclasses.replace(QWEN_7B, tensor_parallel_size=2)
    with pytest.raises(ValueError, match="interconnect"):
        RooflineTimingProvider(None, peak_tflops=989.0, hbm_tbps=3.35, geometry=geometry)
    # One rank needs none.
    RooflineTimingProvider(None, peak_tflops=989.0, hbm_tbps=3.35, geometry=QWEN_7B)


def test_communication_assumptions_are_recorded():
    communication = h100().assumptions()["communication"]
    assert communication["model"] == "no_overlap_approximation"
    assert communication["interconnect"] == "NVLink 4"
    assert communication["interconnect_gbs"] == 450.0
    assert communication["interconnect_efficiency"] == 0.8
    assert communication["effective_gbs"] == pytest.approx(360.0)
    assert communication["collective_latency_us"] == 10.0
    assert "embedding and LM-head (logits) collectives" in communication["omitted"]


def test_tensor_parallel_moe_keeps_every_expert_with_its_matrices_split(
    tmp_path, monkeypatch
):
    # QWEN2_MOE on 2 ranks: heads, MLPs and expert matrices halved; the
    # router and shared-expert gate replicated; 8 experts and top-2 kept.
    model_dir = write_config(tmp_path / "qwen2-moe", QWEN2_MOE)
    record = timing_record(tmp_path, model_dir, monkeypatch, tensor_parallel_size=2)
    geometry = record["geometry"]
    assert geometry["tensor_parallel_size"] == 2
    attention = 64 * (32 + 2 * 16) + 32 * 64  # 2 query heads, 1 KV head of 16
    dense_mlp = 64 * 96 + 48 * 64
    router, shared_gate, shared = 64 * 8, 64 * 1, 64 * 48 + 24 * 64
    assert geometry["layers"] == [
        {"first": 0, "last": 0, "dense_params": attention + dense_mlp, "experts": []},
        {
            "first": 1,
            "last": 3,
            "dense_params": attention + router + shared_gate + shared,
            "experts": [
                {
                    "local_experts": 8,
                    "global_experts": 8,
                    "top_k": 2,
                    "params_per_expert": 3 * 64 * 16,
                }
            ],
        },
    ]
    assert record["communication"]["interconnect"] == "NVLink 4"


@pytest.mark.parametrize(
    "override",
    [
        {"interconnect_gbs": 0.0},
        {"interconnect_gbs": -450.0},
        {"interconnect_gbs": float("inf")},
        {"interconnect_gbs": float("nan")},
        {"interconnect_efficiency": 0.0},
        {"interconnect_efficiency": -0.8},
        {"interconnect_efficiency": float("nan")},
        {"collective_latency_us": -1.0},
        {"collective_latency_us": float("inf")},
    ],
)
def test_invalid_communication_overrides_are_rejected(override):
    (name,) = override
    with pytest.raises(ValueError, match=name):
        h100(**override)


def test_zero_collective_latency_is_allowed():
    geometry = dataclasses.replace(QWEN_7B, tensor_parallel_size=2)
    provider = RooflineTimingProvider(
        None, gpu="h100", geometry=geometry, collective_latency_us=0.0
    )
    assert provider.communication_s(0) == 0


def moe_parallel_config(**parallel):
    sizes = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "world_size_across_dp": 4,
        "enable_expert_parallel": False,
        "enable_eplb": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        **parallel,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization=None, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(**sizes),
    )


DP_ATTENTION_EP = {
    "tensor_parallel_size": 1,
    "data_parallel_size": 4,
    "world_size_across_dp": 4,
    "enable_expert_parallel": True,
}


def test_moe_timing_allows_tensor_or_dp_attention_with_expert_parallelism():
    ModelGeometry._check_moe(moe_parallel_config())
    ModelGeometry._check_moe(moe_parallel_config(**DP_ATTENTION_EP))
    for parallel in (
        {"decode_context_parallel_size": 2},  # reuses the TP ranks
        {"prefill_context_parallel_size": 2},
        {"enable_expert_parallel": True},  # EP across TP ranks
        {"world_size_across_dp": 8},  # PP or DP
        {**DP_ATTENTION_EP, "enable_expert_parallel": False},  # DP without EP
        {**DP_ATTENTION_EP, "tensor_parallel_size": 2, "world_size_across_dp": 8},
        {**DP_ATTENTION_EP, "pipeline_parallel_size": 2, "world_size_across_dp": 8},
    ):
        with pytest.raises(ValueError, match="data-parallel attention with expert"):
            ModelGeometry._check_moe(moe_parallel_config(**parallel))
    with pytest.raises(ValueError, match="EPLB"):
        ModelGeometry._check_moe(moe_parallel_config(**DP_ATTENTION_EP, enable_eplb=True))


# MLA, from the built model. DeepSeek-V2-Lite projects queries directly
# (q_lora_rank None), DeepSeek-V3 through a compressed query; both cache a
# latent of 512 plus a 64-wide rope key per token and layer.
DEEPSEEK_V2_LITE = dict(
    hidden=2048, heads=16, q_lora=None, layers=27, dense_layers=1,
    intermediate=10944, moe_intermediate=1408, experts=64, top_k=6, shared=2,
    vocab=102400,
)  # fmt: skip
DEEPSEEK_V3 = dict(
    hidden=7168, heads=128, q_lora=1536, layers=61, dense_layers=3,
    intermediate=18432, moe_intermediate=2048, experts=256, top_k=8, shared=1,
    vocab=129280,
)  # fmt: skip


def deepseek(spec, tp=1):
    """A DeepSeek model's geometry on one of `tp` ranks: heads, MLPs and
    experts split; the latent's down-projection and the router replicated."""
    hidden, heads = spec["hidden"], spec["heads"] // tp
    if spec["q_lora"]:
        query = hidden * spec["q_lora"] + spec["q_lora"] * heads * 192
    else:
        query = hidden * heads * 192  # nope 128 + rope 64 per head
    attention = query + hidden * (512 + 64) + heads * 128 * hidden
    latent = LatentAttentionGeometry(
        num_heads=heads,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_b_params=512 * heads * (128 + 128),
    )
    moe = spec["moe_intermediate"]
    expert = ExpertGeometry(
        spec["experts"], spec["experts"], spec["top_k"], 3 * hidden * moe // tp
    )
    dense = LayerGeometry(attention + 3 * hidden * spec["intermediate"] // tp, (), latent)
    sparse = LayerGeometry(
        attention + hidden * spec["experts"] + 3 * hidden * moe * spec["shared"] // tp,
        (expert,),
        latent,
    )
    return ModelGeometry(
        num_layers=spec["layers"],
        hidden_size=hidden,
        num_heads=heads,
        num_kv_heads=1,
        head_size=576,
        intermediate_size=spec["intermediate"] // tp,
        vocab_size=spec["vocab"] // tp,
        weight_bytes=2,
        kv_bytes=2,
        tensor_parallel_size=tp,
        layers=(dense,) * spec["dense_layers"]
        + (sparse,) * (spec["layers"] - spec["dense_layers"]),
    )


VARIANTS = pytest.mark.parametrize(
    "spec,tp",
    [(DEEPSEEK_V2_LITE, 1), (DEEPSEEK_V2_LITE, 2), (DEEPSEEK_V3, 1), (DEEPSEEK_V3, 2)],
    ids=["v2-lite-tp1", "v2-lite-tp2", "v3-tp1", "v3-tp2"],
)


def mla(geometry, **kwargs):
    return RooflineTimingProvider(None, gpu="h100", geometry=geometry, **kwargs)


def compute_flops(cost, gpu_tflops=989.0):
    return cost.breakdown["compute_us"] * 1e-6 * gpu_tflops * 1e12 * 0.6


def pairs(scheduled, computed):
    return scheduled * computed + scheduled * (scheduled + 1) // 2


@VARIANTS
def test_every_rank_caches_the_whole_latent(spec, tp):
    geometry = deepseek(spec, tp)
    assert geometry.kv_bytes_per_token == spec["layers"] * 576 * 2
    # A 64-token block of one layer, on each rank.
    assert geometry.kv_bytes_per_token // spec["layers"] * 64 == 73_728


@VARIANTS
def test_decode_absorbs_kv_b_proj_once_per_query_token(spec, tp):
    geometry = deepseek(spec, tp)
    heads, layers = spec["heads"] // tp, spec["layers"]
    kv_b = 512 * heads * 256
    cost = mla(geometry).estimate(sim_step(0, sim_req("r", 1, 1000, 1000)))
    flops = (
        2 * geometry.token_params
        + 2 * spec["hidden"] * spec["vocab"] // tp
        # MQA over the latent: keys 576 wide, values 512.
        + layers * 2 * heads * pairs(1, 1000) * (576 + 512)
        + layers * 2 * kv_b
    )
    assert compute_flops(cost) == pytest.approx(flops)
    assert cost.breakdown["mla_decode_path_reqs"] == 1
    # One kv_b_proj read; the whole latent context read and the token's written.
    weights = geometry.weights_read(1) + spec["hidden"] * spec["vocab"] // tp
    memory = weights * 2 + (1001 + 1) * geometry.kv_bytes_per_token
    assert cost.breakdown["memory_us"] == pytest.approx(memory / (3.35e12 * 0.8) * 1e6)
    assert geometry.weights_read(1) == geometry.weights_read(1, latent_paths=1)


@VARIANTS
def test_a_prefill_chunk_up_projects_its_cached_context(spec, tp):
    # FLASH_ATTN_MLA (h100) sends chunks over 512 tokens to the prefill path.
    geometry = deepseek(spec, tp)
    heads, layers = spec["heads"] // tp, spec["layers"]
    kv_b = 512 * heads * 256
    cost = mla(geometry).estimate(sim_step(1, sim_req("r", 1000, 1000, 3000)))
    flops = (
        2 * geometry.token_params * 1000
        # MHA: keys 192 wide (nope 128 + rope 64), values 128.
        + layers * 2 * heads * pairs(1000, 1000) * (192 + 128)
        # The chunk's tokens and the 1000 cached ones, up-projected.
        + layers * 2 * kv_b * 2000
    )
    assert compute_flops(cost) == pytest.approx(flops)
    assert cost.breakdown["mla_decode_path_reqs"] == 0


@VARIANTS
def test_a_mixed_batch_takes_both_paths(spec, tp):
    geometry = deepseek(spec, tp)
    heads, layers = spec["heads"] // tp, spec["layers"]
    kv_b = 512 * heads * 256
    step = sim_step(0, sim_req("d", 1, 1000, 1000), sim_req("p", 1000, 0, 1000))
    cost = mla(geometry).estimate(step)
    flops = (
        2 * geometry.token_params * 1001
        + 2 * 2 * spec["hidden"] * spec["vocab"] // tp
        + layers * (2 * heads * pairs(1, 1000) * (576 + 512) + 2 * kv_b)
        + layers * (2 * heads * pairs(1000, 0) * (192 + 128) + 2 * kv_b * 1000)
    )
    assert compute_flops(cost) == pytest.approx(flops)
    assert cost.breakdown["mla_decode_path_reqs"] == 1
    # kv_b_proj read twice: its absorbed copies and itself.
    weights = geometry.weights_read(1001, latent_paths=2) + spec["hidden"] * spec["vocab"] // tp
    assert geometry.weights_read(1001, latent_paths=2) - geometry.weights_read(1001) == (
        layers * kv_b
    )
    memory = weights * 2 + (2001 + 1001) * geometry.kv_bytes_per_token
    assert cost.breakdown["memory_us"] == pytest.approx(memory / (3.35e12 * 0.8) * 1e6)


def absorbed(backend, *requests):
    return LATENT_ATTENTION_BACKENDS[backend].absorbed(requests)


@pytest.mark.parametrize("backend", sorted(LATENT_ATTENTION_BACKENDS))
def test_the_token_recomputed_after_a_kv_handoff_takes_the_decode_path(backend):
    # Decode loads all of a prompt but its last token, which it computes.
    handoff = sim_req("new", 1, 999, 1000)
    assert absorbed(backend, handoff) == (True,)
    assert absorbed(backend, sim_req("old", 1, 1500, 1000), handoff) == (True, True)


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("TRITON_MLA", False),
        ("FLASHINFER_MLA", False),
        ("CUTLASS_MLA", False),
        ("FLASHMLA", True),  # at most 128 tokens
        ("FLASH_ATTN_MLA", True),  # at most 512 tokens
    ],
)
def test_short_prefills_take_the_decode_path_below_the_threshold(backend, expected):
    assert absorbed(backend, sim_req("p", 50, 0, 50)) == (expected,)
    assert absorbed(backend, sim_req("p", 600, 0, 600)) == (False,)


def test_uniform_decode_backends_split_after_the_first_query_length():
    batch = (sim_req("d", 1, 1500, 1000), sim_req("p", 50, 0, 50))
    # FlashMLA's decode queries must share the first one's length.
    assert absorbed("FLASHMLA", *batch) == (True, False)
    assert absorbed("FLASH_ATTN_MLA", *batch) == (True, True)
    # The short extend (a chunk of 100 continuing a prompt) follows decodes.
    batch = (sim_req("c", 100, 100, 1000), sim_req("d", 1, 1500, 1000))
    assert absorbed("FLASHMLA", *batch) == (False, True)
    assert absorbed("FLASHMLA", sim_req("a", 100, 100, 1000), sim_req("b", 100, 0, 100)) == (
        True,
        True,
    )


def test_the_batch_splits_after_its_first_long_query():
    # Ordered decode, long extend, first chunk: the 50-token prompt follows
    # the 600-token chunk onto the prefill path.
    batch = (
        sim_req("p", 50, 0, 50),
        sim_req("l", 600, 600, 2000),
        sim_req("d", 1, 1500, 1000),
    )
    assert absorbed("FLASH_ATTN_MLA", *batch) == (False, False, True)
    assert absorbed("FLASH_ATTN_MLA", batch[0], batch[2]) == (True, True)


@pytest.mark.parametrize(
    "gpu,backend,rejected",
    [
        ("a100", "TRITON_MLA", ["FLASH_ATTN_MLA", "FLASHMLA", "FLASHINFER_MLA"]),
        ("h100", "FLASH_ATTN_MLA", []),
        ("h200", "FLASH_ATTN_MLA", []),
        ("b200", "FLASHINFER_MLA", []),
    ],
)
def test_the_mla_backend_is_vllms_choice_on_the_gpu(gpu, backend, rejected):
    geometry = deepseek(DEEPSEEK_V2_LITE)
    provider = RooflineTimingProvider(None, gpu=gpu, geometry=geometry)
    recorded = provider.assumptions()["latent_attention"]
    assert recorded["backend"] == backend
    assert recorded["selected_by"].startswith(f"vLLM's CUDA priorities on {gpu}")
    assert list(recorded["rejected"]) == rejected
    assert recorded["decode_threshold"] == LATENT_ATTENTION_BACKENDS[backend].decode_threshold
    # The geometry records each layer's latent attention.
    assert provider.assumptions()["geometry"]["layers"][0]["latent"]["kv_b_params"] == (
        512 * 16 * 256
    )


def narrow(geometry, latent_size):
    """`geometry` with a latent of `latent_size` (rope 8, the rest compressed)."""
    layers = tuple(
        dataclasses.replace(
            layer,
            latent=dataclasses.replace(
                layer.latent, kv_lora_rank=latent_size - 8, qk_rope_head_dim=8
            ),
        )
        for layer in geometry.layers
    )
    return dataclasses.replace(geometry, layers=layers)


def test_a_latent_no_native_kernel_serves_falls_back_to_triton():
    # Native MLA kernels serve latents 320, 512 or 576 wide.
    recorded = mla(narrow(deepseek(DEEPSEEK_V2_LITE), 40)).assumptions()["latent_attention"]
    assert recorded["backend"] == "TRITON_MLA"
    assert recorded["rejected"]["FLASH_ATTN_MLA"] == ["latent width 40 not in (320, 512, 576)"]
    assert "latent width 40 not in (320, 512, 576)" in recorded["rejected"]["FLASHMLA"]
    with pytest.raises(ValueError, match="FLASH_ATTN_MLA here: latent width 40"):
        mla(narrow(deepseek(DEEPSEEK_V2_LITE), 40), mla_backend="FLASH_ATTN_MLA")


@pytest.mark.parametrize(
    "gpu,backend", [("h100", "FLASHMLA"), ("b200", "FLASHINFER_MLA")]
)
def test_an_fp8_kv_cache_selects_a_backend_serving_it(gpu, backend):
    fp8 = dataclasses.replace(deepseek(DEEPSEEK_V2_LITE), kv_bytes=1, kv_cache_dtype="fp8")
    provider = RooflineTimingProvider(None, gpu=gpu, geometry=fp8)
    assert provider.assumptions()["latent_attention"]["backend"] == backend
    # FlashAttention MLA serves 16-bit caches only.
    with pytest.raises(ValueError, match="KV cache dtype fp8 not in"):
        mla(fp8, mla_backend="FLASH_ATTN_MLA")
    # Triton's fp8 needs SM89: no backend on an A100.
    with pytest.raises(ValueError, match="No MLA backend vLLM would run on a100"):
        RooflineTimingProvider(None, gpu="a100", geometry=fp8)


def test_the_block_size_takes_part_in_the_choice():
    geometry = deepseek(DEEPSEEK_V2_LITE)
    blocks_of_16 = SimpleNamespace(cache_config=SimpleNamespace(block_size=16))

    def backend(**kwargs):
        provider = RooflineTimingProvider(blocks_of_16, geometry=geometry, **kwargs)
        return provider.assumptions()["latent_attention"]

    assert backend(gpu="h100")["backend"] == "FLASH_ATTN_MLA"
    # FlashInfer's 32 or 64 and CUTLASS's 128 do not divide 16.
    b200 = backend(gpu="b200")
    assert b200["backend"] == "TRITON_MLA"
    assert b200["checked"] == {
        "kv_cache_dtype": "auto",
        "block_size": 16,
        "compute_capability": (10, 0),
    }
    with pytest.raises(ValueError, match="block size 16 not a multiple of one of"):
        backend(gpu="h100", mla_backend="FLASHMLA")


def test_the_mla_backend_can_be_chosen():
    geometry = deepseek(DEEPSEEK_V2_LITE)
    provider = mla(geometry, mla_backend="FLASHMLA")
    assert provider.assumptions()["latent_attention"]["backend"] == "FLASHMLA"
    assert provider.assumptions()["latent_attention"]["selected_by"] == "mla_backend"
    with pytest.raises(ValueError, match="Unknown MLA backend"):
        mla(geometry, mla_backend="FLASHMLA_SPARSE")
    # FlashMLA's dense kernels run on Hopper only.
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\) not supported"):
        RooflineTimingProvider(None, gpu="b200", geometry=geometry, mla_backend="FLASHMLA")
    # Without a preset, the backend must be named; the GPU goes unchecked.
    with pytest.raises(ValueError, match="mla_backend"):
        RooflineTimingProvider(None, peak_tflops=989.0, hbm_tbps=3.35, geometry=geometry)
    custom = RooflineTimingProvider(
        None, peak_tflops=989.0, hbm_tbps=3.35, geometry=geometry, mla_backend="FLASHMLA"
    )
    assert custom.assumptions()["latent_attention"]["checked"]["compute_capability"] is None


def test_dense_and_moe_steps_record_no_mla():
    step = sim_step(0, sim_req("d", 1, 1000, 1000), sim_req("p", 1000, 0, 1000))
    for provider in (h100(), moe()):
        assert set(provider.estimate(step).breakdown) == {
            "compute_us",
            "memory_us",
            "communication_us",
        }
        assert "latent_attention" not in provider.assumptions()
        layers = provider.assumptions()["geometry"].get("layers") or []
        assert all("latent" not in group for group in layers)


def test_an_mla_model_is_timed_from_its_built_layers():
    config = qwen_7b_config("auto")
    config.model_config.use_mla = True
    with pytest.raises(ValueError, match="from_model"):
        ModelGeometry.from_config(config)
    provider = RooflineTimingProvider(config, gpu="h100")
    with pytest.raises(RuntimeError, match="bind_model"):
        provider.estimate(sim_step(0, sim_req("r", 1, 999, 1000)))
    # Not the config's geometry when its layers cannot be found.
    with pytest.raises(UnsupportedModelLayout):
        provider.bind_model(torch.nn.Module())


# Tiny DeepSeek models: V2 projects queries directly, V3 compresses them.
DEEPSEEK_V2 = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
    "hidden_size": 64,
    "intermediate_size": 96,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": None,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "greedy",
    "scoring_func": "softmax",
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "tie_word_embeddings": False,
}
DEEPSEEK_V3_TINY = {
    **DEEPSEEK_V2,
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "q_lora_rank": 24,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
}


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("config", [DEEPSEEK_V2, DEEPSEEK_V3_TINY], ids=["v2", "v3"])
def test_a_built_mla_model_keeps_kv_b_proj_apart(tmp_path, monkeypatch, config, tp):
    model_dir = write_config(tmp_path / config["model_type"], config)
    steps = []
    # 64-token blocks suit either CPU MLA kernel (AMX: multiples of 32; the
    # reference one, without AMX, is given 16 by the platform).
    record = timing_record(tmp_path, model_dir, monkeypatch, tp, steps, block_size=64)
    heads = 4 // tp
    if config["q_lora_rank"]:
        query = 64 * 24 + 24 * heads * 24  # compressed, then 16 + 8 per head
    else:
        query = 64 * heads * 24
    attention = query + 64 * (32 + 8) + heads * 16 * 64  # latent + rope key; O
    latent = {
        "num_heads": heads,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "kv_b_params": 32 * heads * (16 + 16),
    }
    router, shared = 64 * 8, 3 * 64 * 32 // tp
    assert record["geometry"]["layers"] == [
        {
            "first": 0,
            "last": 0,
            "dense_params": attention + 3 * 64 * 96 // tp,
            "experts": [],
            "latent": latent,
        },
        {
            "first": 1,
            "last": 2,
            "dense_params": attention + router + shared,
            "experts": [
                {
                    "local_experts": 8,
                    "global_experts": 8,
                    "top_k": 2,
                    "params_per_expert": 3 * 64 * 32 // tp,
                }
            ],
            "latent": latent,
        },
    ]
    # No native kernel serves a 40-wide latent: vLLM would run Triton, whose
    # decode path takes single tokens only.
    assert record["latent_attention"]["backend"] == "TRITON_MLA"
    assert record["latent_attention"]["checked"]["block_size"] == 64
    assert [s["cost"]["breakdown"]["mla_decode_path_reqs"] for s in steps] == [0, 1]


# Data-parallel attention with expert parallelism: one rank per DP rank, the
# routed experts split evenly across them.
def expert_parallel(geometry, ranks):
    layers = tuple(
        dataclasses.replace(
            layer,
            experts=tuple(
                dataclasses.replace(e, local_experts=e.global_experts // ranks)
                for e in layer.experts
            ),
        )
        for layer in geometry.layers
    )
    return dataclasses.replace(
        geometry, layers=layers, data_parallel_size=ranks, expert_parallel=True
    )


def on_rank(rank, request):
    return dataclasses.replace(request, dp_rank=rank)


# Two ranks of a model whose dense layer outweighs everything else, then an
# MoE layer of 4 experts (2 a rank), top-2; a vocabulary large enough that
# sampling many tokens takes long. Unit rates: 1 TFLOP/s, 1 TB/s, 1 GB/s.
SEGMENT_EXPERTS = ExpertGeometry(
    local_experts=2, global_experts=4, top_k=2, params_per_expert=1000
)
TWO_RANKS = ModelGeometry(
    num_layers=2,
    hidden_size=64,
    num_heads=2,
    num_kv_heads=1,
    head_size=32,
    intermediate_size=0,
    vocab_size=1_000_000,
    weight_bytes=2,
    kv_bytes=2,
    data_parallel_size=2,
    expert_parallel=True,
    layers=(LayerGeometry(10_000_000), LayerGeometry(1000, (SEGMENT_EXPERTS,))),
)


def unit_rates(geometry=TWO_RANKS, **kwargs):
    kwargs.setdefault("ep_moe_interface", "modular")
    return RooflineTimingProvider(
        None,
        peak_tflops=1.0,
        hbm_tbps=1.0,
        compute_efficiency=1.0,
        bandwidth_efficiency=1.0,
        fixed_us=0.0,
        interconnect_gbs=1.0,
        interconnect_efficiency=1.0,
        collective_latency_us=0.0,
        geometry=geometry,
        **kwargs,
    )


def test_routing_reaches_a_rank_with_the_probability_of_hitting_its_experts():
    # One expert of 256, 32 a rank: a token reaches 1 rank of 8; remote with
    # probability 7/8, never 0.
    p = rank_reach_probability(256, 32, 1)
    assert p == pytest.approx(1 / 8)
    assert (8 - 1) * p == pytest.approx(7 / 8)
    # top_k >= ranks does not reach every rank.
    assert rank_reach_probability(256, 32, 8) == pytest.approx(
        1 - (224 * 223 * 222 * 221 * 220 * 219 * 218 * 217)
        / (256 * 255 * 254 * 253 * 252 * 251 * 250 * 249)
    )
    assert rank_reach_probability(256, 32, 8) < 1
    # More experts than the other ranks hold: certainly this rank.
    assert rank_reach_probability(8, 4, 5) == 1
    assert rank_reach_probability(4, 2, 2) == pytest.approx(5 / 6)


def test_ranks_pace_each_segment_between_synchronizations():
    """Rank 0's prefill dominates before the MoE layer's dispatch, rank 1's
    50 sampled tokens after its combine: the step takes both, not the larger
    of the ranks' totals."""
    prefill = on_rank(0, sim_req("a", 1000, 0, 2000))
    decodes = [on_rank(1, sim_req(f"d{i}", 1, 10, 10)) for i in range(50)]
    cost = unit_rates().estimate(sim_step(0, prefill, *decodes))

    def seconds(flops, nbytes):
        return max(flops / 1e12, nbytes / 1e12)

    dense = 10_000_000 + 1000
    kv = 2 * 1 * 32 * 2  # K and V, one head of 32, bf16
    lm_head = 64 * 1_000_000
    before = [
        seconds(
            2 * dense * 1000 + 2 * 4 * 2 * 32 * (1000 * 1001 // 2),
            dense * 2 + 2 * (1000 + 1000) * kv,
        ),
        seconds(
            2 * dense * 50 + 2 * 4 * 2 * 32 * 50 * 11,
            dense * 2 + 2 * 50 * (11 + 1) * kv,
        ),
    ]
    after = [seconds(0, lm_head * 2), seconds(2 * lm_head * 50, lm_head * 2)]
    assert before[0] > before[1] and after[1] > after[0]
    assert cost.breakdown["local_us"] == pytest.approx((before[0] + after[1]) * 1e6)
    assert cost.breakdown["local_us"] > max(b + a for b, a in zip(before, after)) * 1e6
    # Every rank runs its share of the 1050 tokens' experts.
    experts = seconds(
        2 * 1050 * 2 * (2 / 4) * 1000,
        SEGMENT_EXPERTS.expected_read(1050) * 1000 * 2,
    )
    assert cost.breakdown["experts_us"] == pytest.approx(experts * 1e6)
    assert cost.device_s == pytest.approx(
        (cost.breakdown["local_us"] + cost.breakdown["experts_us"]
         + cost.breakdown["communication_us"]) * 1e-6
    )


@pytest.mark.parametrize("backend", sorted(EXPERT_PARALLEL_BACKENDS))
def test_the_all_to_all_moves_the_busiest_ranks_tokens(backend):
    # 1000 tokens on rank 0, 50 on rank 1.
    step = sim_step(
        0,
        on_rank(0, sim_req("a", 1000, 0, 2000)),
        *[on_rank(1, sim_req(f"d{i}", 1, 10, 10)) for i in range(50)],
    )
    cost = unit_rates(ep_all2all=backend).estimate(step)
    if backend == "allgather_reducescatter":
        # Each rank gathers the other's tokens: rank 1 receives 1000.
        copies, routing = 1000, 2 * (4 + 4)
    else:
        # Rank 0 sends 1000 p copies to rank 1 and receives 50 p.
        copies, routing = 1000 * 5 / 6, 2 * (8 + 4)
    seconds = copies * (64 * 2 + routing) / 1e9 + copies * 64 * 2 / 1e9
    assert cost.breakdown["communication_us"] == pytest.approx(seconds * 1e6)


def test_a_monolithic_moe_kernel_gathers_every_router_logit():
    """Through FlashInfer TRTLLM's monolithic kernel, allgather_reducescatter
    gathers all E router logits per token: 4 here, BF16."""
    geometry = dataclasses.replace(TWO_RANKS, router_logits_bytes=2)
    request = sim_req("r", 300, 0, 300)
    cost = unit_rates(geometry, ep_moe_interface="monolithic").estimate(
        sim_step(0, on_rank(1, request))
    )
    assert cost.breakdown["communication_us"] == pytest.approx(
        300 * (64 * 2 + 4 * 2 + 64 * 2) / 1e9 * 1e6
    )
    with pytest.raises(ValueError, match="router logits' dtype"):
        unit_rates(ep_moe_interface="monolithic")
    with pytest.raises(ValueError, match="modular MoE interface"):
        unit_rates(geometry, ep_all2all="deepep_high_throughput", ep_moe_interface="monolithic")
    with pytest.raises(ValueError, match="pass ep_moe_interface|a gpu preset, or ep_moe_interface"):
        unit_rates(geometry, ep_moe_interface=None)


def bf16_deepseek_v3_on_8_ranks():
    return dataclasses.replace(
        expert_parallel(deepseek(DEEPSEEK_V3), 8),
        routing_method="DeepSeekV3",
        router_logits_bytes=2,
    )


def per_layer_all2all_us(provider, metadata, tokens=100):
    hidden = 7168 * 2
    return (2 * 10e-6 + tokens * (2 * hidden + metadata) / provider.collective_bytes_per_s) * 1e6


@pytest.mark.parametrize("gpu", ["a100", "h100", "h200"])
def test_the_moe_interface_is_modular_where_vllm_has_no_monolithic_kernel(gpu):
    provider = RooflineTimingProvider(None, gpu=gpu, geometry=bf16_deepseek_v3_on_8_ranks())
    recorded = provider.assumptions()["expert_parallel"]
    assert recorded["moe_interface"] == "modular"
    assert recorded["moe_interface_selected_by"].startswith(
        f"vLLM's default unquantized MoE kernel on {gpu}"
    )
    # Top-k ids and weights: 64 bytes per token for BF16 DeepSeek-V3.
    step = sim_step(0, on_rank(0, sim_req("r", 100, 0, 100)))
    assert provider.estimate(step).breakdown["communication_us"] == pytest.approx(
        58 * per_layer_all2all_us(provider, 8 * (4 + 4))
    )


def test_b200_needs_the_moe_interface_named():
    """TRTLLM's monolithic kernel leads on Blackwell where its routing,
    activation and expert-count limits allow, which is not modeled."""
    geometry = bf16_deepseek_v3_on_8_ranks()
    with pytest.raises(ValueError, match="needs ep_moe_interface on b200"):
        RooflineTimingProvider(None, gpu="b200", geometry=geometry)
    step = sim_step(0, on_rank(0, sim_req("r", 100, 0, 100)))
    for interface, metadata in (("monolithic", 256 * 2), ("modular", 8 * (4 + 4))):
        provider = RooflineTimingProvider(
            None, gpu="b200", geometry=geometry, ep_moe_interface=interface
        )
        recorded = provider.assumptions()["expert_parallel"]
        assert (recorded["moe_interface"], recorded["moe_interface_selected_by"]) == (
            interface,
            "ep_moe_interface",
        )
        # 512 bytes of router logits per token, or 64 of top-k.
        assert provider.estimate(step).breakdown["communication_us"] == pytest.approx(
            58 * per_layer_all2all_us(provider, metadata)
        )
    # DeepEP dispatches top-k tokens whatever the GPU.
    deepep = RooflineTimingProvider(
        None, gpu="b200", geometry=geometry, ep_all2all="deepep_high_throughput"
    )
    assert deepep.assumptions()["expert_parallel"]["moe_interface"] == "modular"


def test_an_idle_rank_runs_its_share_of_the_experts_only():
    """DP rank 0 idle: no local work of its own (its dummy batch is not
    modeled), but its experts serve rank 1's tokens and it takes part in
    the all-to-all; the step is the same as with the roles swapped."""
    request = sim_req("r", 300, 0, 300)
    idle0 = unit_rates().estimate(sim_step(0, on_rank(1, request)))
    idle1 = unit_rates().estimate(sim_step(0, on_rank(0, request)))
    assert idle0.breakdown["idle_dp_ranks"] == 1
    assert idle0.device_s == pytest.approx(idle1.device_s)
    assert idle0.breakdown["experts_us"] > 0
    # The busy rank's whole share of the gather: all 300 tokens.
    assert idle0.breakdown["communication_us"] == pytest.approx(
        300 * (64 * 2 + 2 * 8 + 64 * 2) / 1e9 * 1e6
    )


def test_each_rank_splits_its_own_batch_between_the_mla_paths():
    """FLASH_ATTN_MLA (h100) takes a lone 50-token prompt on the decode path:
    another rank's 600-token prompt does not change that."""
    geometry = expert_parallel(deepseek(DEEPSEEK_V2_LITE), 2)
    long, short = sim_req("a", 600, 0, 600), sim_req("b", 50, 0, 50)
    cost = mla(geometry).estimate(sim_step(0, on_rank(0, long), on_rank(1, short)))
    assert cost.breakdown["mla_decode_path_reqs"] == 1
    # One merged batch would split after the long prompt.
    assert absorbed("FLASH_ATTN_MLA", long, short) == (False, False)


@pytest.mark.parametrize("backend", sorted(LATENT_ATTENTION_BACKENDS))
def test_an_empty_batch_takes_no_mla_path(backend):
    assert absorbed(backend) == ()


def test_expert_parallel_assumptions_are_recorded():
    geometry = expert_parallel(deepseek(DEEPSEEK_V3), 8)
    recorded = mla(geometry).assumptions()["expert_parallel"]
    assert recorded["ranks"] == 8
    assert recorded["all2all_modeled"] == "allgather_reducescatter"
    assert recorded["all2all_configured"] is None
    assert recorded["moe_interface"] == "modular"
    assert recorded["dispatch_dtype_bytes"] == 2
    assert "grouped (node-limited) routing, which changes which ranks a token reaches" in (
        recorded["not_modeled"]
    )
    assert "expert_parallel" not in moe().assumptions()
    # vLLM constructs one backend, the roofline models another.
    constructed = SimpleNamespace(
        parallel_config=SimpleNamespace(all2all_backend="deepep_low_latency"),
        cache_config=SimpleNamespace(block_size=64),
    )
    with pytest.raises(ValueError, match="pass ep_all2all"):
        RooflineTimingProvider(constructed, gpu="h100", geometry=geometry)
    provider = RooflineTimingProvider(
        constructed, gpu="h100", geometry=geometry, ep_all2all="deepep_high_throughput"
    )
    recorded = provider.assumptions()["expert_parallel"]
    assert (recorded["all2all_modeled"], recorded["all2all_configured"]) == (
        "deepep_high_throughput",
        "deepep_low_latency",
    )
    assert (recorded["moe_interface"], recorded["moe_interface_selected_by"]) == (
        "modular",
        "deepep_high_throughput",
    )


def test_unmodeled_expert_parallel_layouts_are_rejected():
    with pytest.raises(ValueError, match="not 'deepep_low_latency'"):
        mla(expert_parallel(deepseek(DEEPSEEK_V3), 8), ep_all2all="deepep_low_latency")
    with pytest.raises(ValueError, match="one NVLink domain"):
        mla(expert_parallel(deepseek(DEEPSEEK_V3), 16))
    with pytest.raises(ValueError, match="interconnect for expert parallelism"):
        RooflineTimingProvider(
            None,
            peak_tflops=989.0,
            hbm_tbps=3.35,
            geometry=expert_parallel(deepseek(DEEPSEEK_V3), 8),
            mla_backend="FLASHMLA",
        )


def test_dense_data_parallel_ranks_time_as_one_rank():
    # Independent engines: no joint steps, each rank times its own.
    geometry = dataclasses.replace(QWEN_7B, data_parallel_size=4)
    step = sim_step(0, sim_req("r", 1000, 0, 1000), sim_req("s", 1, 999, 1000))
    one = h100().estimate(step)
    four = RooflineTimingProvider(None, gpu="h100", geometry=geometry).estimate(step)
    assert four.device_s == one.device_s and four.breakdown == one.breakdown


def test_joint_steps_keep_each_requests_dp_rank():
    first = sim_step(3, on_rank(0, sim_req("b", 1, 9, 10)))
    second = sim_step(7, on_rank(1, sim_req("a", 5, 0, 5)))
    joint = merge_dp_descriptors([first, second])
    assert [(r.req_id, r.dp_rank) for r in joint.requests] == [("a", 1), ("b", 0)]
    assert joint.step_index == 3


class StubExperts(torch.nn.Module):
    """A rank's routed experts as a built model holds them: `local` of
    `total`, top-2, 3 * 8 * 4 weights each."""

    def __init__(self, local, total):
        super().__init__()
        self.moe_config = SimpleNamespace(
            hidden_dim=8,
            hidden_dim_unpadded=None,
            intermediate_size_per_partition=4,
            intermediate_size_per_partition_unpadded=None,
            is_act_and_mul=True,
            num_local_experts=local,
            num_experts=total,
            experts_per_token=2,
        )


@pytest.mark.parametrize("local,total,even", [(3, 9, True), (3, 8, False)])
def test_expert_parallel_experts_must_be_placed_evenly(monkeypatch, local, total, even):
    from vllm.model_executor.layers.fused_moe import routed_experts

    monkeypatch.setattr(routed_experts, "RoutedExperts", StubExperts)
    config = qwen_7b_config("auto")
    config.model_config.is_moe = True
    config.model_config.quantization = None
    config.parallel_config = moe_parallel_config(
        **{**DP_ATTENTION_EP, "data_parallel_size": 3, "world_size_across_dp": 3}
    ).parallel_config
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(StubExperts(local, total) for _ in range(28))
    if even:
        geometry = ModelGeometry.from_model(config, model)
        assert (geometry.ep_size, geometry.layers[0].experts[0].local_experts) == (3, 3)
    else:
        with pytest.raises(ValueError, match="placed evenly: 3 of 8 on one of 3"):
            ModelGeometry.from_model(config, model)


def test_dp_attention_ranks_build_their_experts_and_log_every_joint_step(tmp_path):
    """Served with DP=2 and EP, each rank builds 4 of the 8 experts. With
    the only request on DP rank 1, the idle DP rank 0 still logs each
    lockstep step, in order."""
    import requests

    from tests.utils import RemoteOpenAIServer

    model_dir = write_config(tmp_path / "qwen2-moe", QWEN2_MOE)
    log = tmp_path / "steps.jsonl"
    simulation = {"timing": "roofline", "timing_args": {"gpu": "h100"}, "log_path": str(log)}
    args = [
        "--skip-tokenizer-init",
        "--dtype", "bfloat16",
        "--max-model-len", "256",
        "--block-size", "16",
        "--kv-cache-memory-bytes", str(8 * 1024**2),
        "--enforce-eager",
        "--data-parallel-size", "2",
        "--enable-expert-parallel",
        "--simulate-forward",
        "--simulation-config", json.dumps(simulation),
    ]  # fmt: skip
    with RemoteOpenAIServer(model_dir, args, max_wait_seconds=600) as server:
        response = requests.post(
            server.url_for("v1/completions"),
            json={"model": model_dir, "prompt": [5] * 40, "max_tokens": 3, "ignore_eos": True},
            headers={"X-data-parallel-rank": "1"},
            timeout=120,
        )
        response.raise_for_status()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    (timing,) = [r["timing"] for r in records if "timing" in r]
    geometry = timing["geometry"]
    assert (geometry["data_parallel_size"], geometry["expert_parallel"]) == (2, True)
    experts = geometry["layers"][-1]["experts"][0]
    assert (experts["local_experts"], experts["global_experts"]) == (4, 8)
    assert timing["expert_parallel"]["all2all_modeled"] == "allgather_reducescatter"
    # Hopper: a modular MoE kernel, so top-k metadata is gathered.
    assert timing["expert_parallel"]["moe_interface"] == "modular"
    assert geometry["routing_method"] is not None
    assert geometry["router_logits_bytes"] in (2, 4)
    steps = [r for r in records if "step" in r]
    assert len(steps) >= 3  # prefill and decodes
    assert all(r["dp_rank"] == 1 for s in steps for r in s["step"]["requests"])
    assert all(s["cost"]["breakdown"]["idle_dp_ranks"] == 1 for s in steps)
    indexes = [s["step"]["step_index"] for s in steps]
    assert indexes == sorted(set(indexes))
