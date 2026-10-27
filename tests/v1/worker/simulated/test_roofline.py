# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roofline step timing of dense and MoE models on a GPU preset."""

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.simulated.utils import sim_req, sim_step
from vllm.v1.worker.simulated.timing import (
    ExpertGeometry,
    LayerGeometry,
    ROUTING_MODEL,
    ModelGeometry,
    RooflineTimingProvider,
    UnsupportedModelLayout,
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
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
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


def test_packed_kv_dtypes_are_rejected():
    with pytest.raises(ValueError, match="int4_per_token_head"):
        ModelGeometry.from_config(qwen_7b_config("int4_per_token_head"))


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


def timing_record(tmp_path, model_dir, monkeypatch, tensor_parallel_size=1):
    """The timing record an in-process simulated engine writes, after one
    request's steps (the output rank's)."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        block_size=16,
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
        "world_size_across_dp": 4,
        "enable_expert_parallel": False,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        **parallel,
    }
    return SimpleNamespace(
        model_config=SimpleNamespace(quantization=None, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(**sizes),
    )


def test_moe_timing_allows_pure_tensor_parallelism_only():
    ModelGeometry._check_moe(moe_parallel_config())
    for parallel in (
        {"decode_context_parallel_size": 2},  # reuses the TP ranks
        {"prefill_context_parallel_size": 2},
        {"enable_expert_parallel": True},
        {"world_size_across_dp": 8},  # PP or DP
    ):
        with pytest.raises(ValueError, match="tensor parallelism only"):
            ModelGeometry._check_moe(moe_parallel_config(**parallel))
