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
    RecurrentGeometry,
    RooflineTimingProvider,
    UnsupportedModelLayout,
    attention_pairs,
    kv_tokens,
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
        get_sliding_window=lambda: None,
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
    tmp_path,
    model_dir,
    monkeypatch,
    tensor_parallel_size=1,
    steps=None,
    block_size=16,
    **options,
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
        **options,
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
    # No layer records its own: two hidden-width all-reduces a layer.
    assert communication["event_source"] == "legacy_default"
    assert communication["events_per_step"] == [
        {"label": "hidden", "elements": 3584, "dtype_bytes": 2, "count": 2 * 28}
    ]


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


# Sliding windows: a layer whose tokens attend to at most `window` tokens,
# themselves included.
def test_a_window_bounds_the_pairs_and_the_kv_read():
    # Full attention: every cached token and the new ones before each.
    assert attention_pairs(500, 1000, None) == 1000 * 500 + 1000 * 1001 // 2
    assert kv_tokens(500, 1000, None) == 1000 + 500 + 1000
    # A first chunk of 1000 under a 128-token window: min(128, i) pairs.
    assert attention_pairs(0, 1000, 128) == sum(min(128, i) for i in range(1, 1001))
    # A window not yet full: 50 cached, 10 new.
    assert attention_pairs(50, 10, 128) == sum(range(51, 61))
    # Decode stays within the window.
    assert attention_pairs(5000, 1, 128) == 128
    assert kv_tokens(5000, 1, 128) == 1 + 127 + 1
    # A 1000-token chunk after an established window reads 1127 tokens of KV.
    assert kv_tokens(5000, 1000, 128) - 1000 == 1127
    assert attention_pairs(5000, 1000, 128) == 1000 * 128


# Qwen-7B's shapes, its layers alternating a 128-token window and full
# attention, as gpt-oss's do.
ALTERNATING = dataclasses.replace(
    QWEN_7B,
    layers=tuple(
        LayerGeometry(QWEN_7B.layer_params, sliding_window=128 if i % 2 == 0 else None)
        for i in range(28)
    ),
)


@pytest.mark.parametrize(
    "request_",
    [
        sim_req("prefill", 1000, 0, 1000),
        sim_req("chunk", 1000, 5000, 8000),
        sim_req("decode", 1, 5000, 5000),
    ],
    ids=["prefill", "chunk", "decode"],
)
def test_windowed_layers_attend_and_read_within_their_window(request_):
    cost = RooflineTimingProvider(None, gpu="h100", geometry=ALTERNATING).estimate(
        sim_step(0, request_)
    )
    c, t = request_.num_computed_tokens, request_.num_scheduled_tokens
    sampled = c + t >= request_.num_prompt_tokens
    flops = (
        2 * 28 * QWEN_7B.layer_params * t
        + 2 * 3584 * 152064 * sampled
        + 14 * 4 * 28 * 128 * attention_pairs(c, t, None)
        + 14 * 4 * 28 * 128 * attention_pairs(c, t, 128)
    )
    assert cost.breakdown["compute_us"] == pytest.approx(flops / (989e12 * 0.6) * 1e6)
    per_layer = KV_BYTES_PER_TOKEN // 28
    memory = (
        WEIGHT_BYTES
        + 14 * per_layer * kv_tokens(c, t, None)
        + 14 * per_layer * kv_tokens(c, t, 128)
    )
    assert cost.breakdown["memory_us"] == pytest.approx(memory / (3.35e12 * 0.8) * 1e6)
    # Full layers everywhere would read and compute as much or more (a first
    # chunk reads no cached KV either way).
    full = dataclasses.replace(
        ALTERNATING, layers=(LayerGeometry(QWEN_7B.layer_params),) * 28
    )
    full_cost = RooflineTimingProvider(None, gpu="h100", geometry=full).estimate(
        sim_step(0, request_)
    )
    assert full_cost.breakdown["compute_us"] >= cost.breakdown["compute_us"]
    assert full_cost.breakdown["memory_us"] >= cost.breakdown["memory_us"]


def test_windows_are_recorded_with_the_layers():
    layers = RooflineTimingProvider(None, gpu="h100", geometry=ALTERNATING).assumptions()[
        "geometry"
    ]["layers"]
    assert [group.get("sliding_window") for group in layers[:3]] == [128, None, 128]
    assert len(layers) == 28  # consecutive layers differ
    # The same total KV per token: a window bounds the reads, not the cache.
    assert ALTERNATING.kv_bytes_per_token == KV_BYTES_PER_TOKEN


def test_expert_parallel_ranks_attend_within_the_window():
    """The DP+EP path prices a windowed layer's attention and KV like the
    single-rank path: the two-rank model, its dense layer under a window."""
    windowed = dataclasses.replace(
        TWO_RANKS,
        layers=(
            dataclasses.replace(TWO_RANKS.layers[0], sliding_window=128),
            TWO_RANKS.layers[1],
        ),
    )
    step = sim_step(0, on_rank(0, sim_req("a", 1000, 5000, 8000)))
    full, window = (unit_rates(g).estimate(step) for g in (TWO_RANKS, windowed))
    kv_layer = 2 * 1 * 32 * 2
    flops_saved = 4 * 2 * 32 * (attention_pairs(5000, 1000, None) - attention_pairs(5000, 1000, 128))
    bytes_saved = kv_layer * (kv_tokens(5000, 1000, None) - kv_tokens(5000, 1000, 128))
    seconds = max(f / 1e12 for f in (flops_saved,))  # compute-bound prefill chunk
    assert full.breakdown["local_us"] - window.breakdown["local_us"] == pytest.approx(
        seconds * 1e6
    )
    assert bytes_saved > 0


def test_a_sliding_window_model_is_timed_from_its_built_layers():
    config = qwen_7b_config("auto")
    config.model_config.get_sliding_window = lambda: 4096
    with pytest.raises(ValueError, match="from_model"):
        ModelGeometry.from_config(config)
    provider = RooflineTimingProvider(config, gpu="h100")
    with pytest.raises(UnsupportedModelLayout):
        provider.bind_model(torch.nn.Module())


def attention_layer(kind=None, heads=28, kv_heads=4, head_size=128, head_size_v=None):
    """A built attention layer of `kind` (default plain `Attention`) with
    the fields its constructor sets, `sliding_window` the inherited None
    (chunked local and RSWA attention keep their windows apart), without the
    backend the constructor selects."""
    from vllm.model_executor.layers.attention.attention import Attention

    kind = kind or Attention
    layer = kind.__new__(kind)
    torch.nn.Module.__init__(layer)
    layer.num_heads, layer.num_kv_heads, layer.head_size = heads, kv_heads, head_size
    layer.head_size_v = head_size if head_size_v is None else head_size_v
    layer.sliding_window = None
    layer.attention_chunk_size = layer._rswa_window = 64
    return layer


def attention_decoder(layer):
    """A model of Qwen2.5-7B's 28 decoder layers, layer `i`'s attention
    `layer(i)`."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for index in range(28):
        block = torch.nn.Module()
        block.attn = layer(index)
        model.layers.append(block)
    return model


def test_plain_attention_of_the_models_heads_is_bound():
    provider = RooflineTimingProvider(qwen_7b_config("auto"), gpu="h100")
    provider.bind_model(attention_decoder(lambda _: attention_layer()))
    assert provider.assumptions()["geometry"]["source"] == "built model"


@pytest.mark.parametrize("kind", ["ChunkedLocalAttention", "RSWAAttention"])
def test_other_attention_kinds_are_rejected_whatever_their_window(kind):
    # Llama4's chunked local attention leaves `sliding_window` None: neither
    # priced as full attention nor timed from the dense config.
    from vllm.model_executor.layers.attention import (
        chunked_local_attention,
        rswa_attention,
    )

    module = {
        "ChunkedLocalAttention": chunked_local_attention,
        "RSWAAttention": rswa_attention,
    }[kind]
    provider = RooflineTimingProvider(qwen_7b_config("auto"), gpu="h100")
    model = attention_decoder(lambda _: attention_layer(getattr(module, kind)))
    with pytest.raises(ValueError, match=f"plain attention layers only, not \\['{kind}'\\]"):
        provider.bind_model(model)


@pytest.mark.parametrize(
    "dimensions",
    [
        {"heads": 32},
        {"kv_heads": 8},
        {"head_size": 64},
        {"head_size_v": 64},  # a narrower value
    ],
)
def test_attention_layers_of_other_heads_are_rejected(dimensions):
    # MiMoV2's sliding layers have their own heads and head sizes.
    provider = RooflineTimingProvider(qwen_7b_config("auto"), gpu="h100")
    model = attention_decoder(
        lambda index: attention_layer(**(dimensions if index % 2 else {}))
    )
    with pytest.raises(ValueError, match=r"\(28, 4, 128, 128\), not"):
        provider.bind_model(model)


GPT_OSS = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "hidden_size": 64,
    "intermediate_size": 32,
    "head_dim": 16,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "experts_per_token": 2,
    "sliding_window": 32,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
    "attention_bias": True,
    "hidden_act": "silu",
    "swiglu_limit": 7.0,
    "rope_theta": 150000,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
        "original_max_position_embeddings": 4096,
    },
    "tie_word_embeddings": False,
}
MISTRAL = {
    "architectures": ["MistralForCausalLM"],
    "model_type": "mistral",
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "sliding_window": 48,
    "tie_word_embeddings": False,
}


@pytest.mark.parametrize(
    "config,windows",
    [(GPT_OSS, [32, None, 32, None]), (MISTRAL, [48, 48])],
    ids=["gpt-oss", "mistral"],
)
def test_a_built_model_records_each_layers_window(tmp_path, monkeypatch, config, windows):
    model_dir = write_config(tmp_path / config["model_type"], config)
    record = timing_record(tmp_path, model_dir, monkeypatch)
    expanded = [
        group.get("sliding_window")
        for group in record["geometry"]["layers"]
        for _ in range(group["last"] - group["first"] + 1)
    ]
    assert expanded == windows


# Qwen3-Next-80B-A3B's gated delta-net on one rank: 16 key and 32 value
# heads of 128, a convolution of 8192 channels over 4 tokens, BF16 states.
NEXT_GDN = RecurrentGeometry(
    num_k_heads=16,
    num_v_heads=32,
    head_k_dim=128,
    head_v_dim=128,
    conv_dim=8192,
    conv_kernel=4,
    conv_state_bytes=3 * 8192 * 2,
    ssm_state_bytes=32 * 128 * 128 * 2,
    ssm_dtype_bytes=2,
    activation_bytes=2,
)
STATE = 32 * 128 * 128 * 2
FINAL = 32 * 128 * 128 * 4
CONV = 3 * 8192 * 2
# Per value head and 64-token chunk.
CHUNK_FLOPS = 6 * 64 * 128 * 128 + 6 * 64 * 64 * 128 + 4 * 64 * 64 * 128 + 2 * 64**3 / 3


def test_a_gdn_decode_reads_and_writes_its_states():
    assert NEXT_GDN.decode_flops() == 7 * 32 * 128 * 128 == 3_670_016
    assert NEXT_GDN.decode_bytes() == 2 * STATE + 2 * CONV == 2_195_456


def test_a_gdn_chunk_path_counts_whole_chunks_and_its_temporaries():
    # Full chunks: about 5.85 MFLOP and 163,840 bytes a token (106,496 of
    # temporaries, 57,344 of inputs and output) beside the fixed states.
    assert NEXT_GDN.chunk_flops(4096) / 4096 == pytest.approx(5.85455e6, rel=1e-5)
    fixed = 6 * STATE + 2 * FINAL + 2 * CONV
    assert fixed == 10_584_064
    assert NEXT_GDN.chunk_bytes(4096) == fixed + 4096 * 163_840
    # 100 tokens: two chunks' tiles and state snapshots, 100 tokens' buffers.
    assert NEXT_GDN.chunk_flops(100) == 32 * 2 * CHUNK_FLOPS
    per_token = 32 * (64 * 4 + 64 * 2 + 128 * 2 + 2 * 128 * 2)
    snapshots = 2 * 32 * 128 * 128 * 2
    inputs = 32 * (5 * 128 + 2 * 128) * 2
    assert NEXT_GDN.chunk_bytes(100) == fixed + 2 * (100 * per_token + snapshots) + 100 * inputs


@pytest.mark.parametrize(
    "ssm_bytes, fixed",
    [
        (2, 6 * STATE + 2 * FINAL),  # converted to BF16 through a temporary
        (4, 4 * (2 * STATE) + 2 * FINAL),  # an FP32 cache takes the state as is
    ],
)
def test_the_final_state_conversion_follows_the_cache_dtype(ssm_bytes, fixed):
    # A BF16 convolution state beside the SSM state's own dtype.
    gdn = dataclasses.replace(
        NEXT_GDN, ssm_state_bytes=STATE // 2 * ssm_bytes, ssm_dtype_bytes=ssm_bytes
    )
    assert gdn.chunk_bytes(64) - gdn.chunk_bytes(0) == NEXT_GDN.chunk_bytes(64) - (
        NEXT_GDN.chunk_bytes(0)
    )
    assert gdn.chunk_bytes(0) == fixed + 2 * CONV
    assert gdn.decode_bytes() == 2 * STATE // 2 * ssm_bytes + 2 * CONV


@pytest.mark.parametrize(
    "request_, decode",
    [
        (sim_req("decode", 1, 300, 300), True),
        # A resumed one-token chunk still marked as prefill: it has state.
        (sim_req("resumed", 1, 299, 300), True),
        # A first chunk of one token has none: the chunk path.
        (sim_req("first", 1, 0, 1), False),
        (sim_req("later", 200, 100, 300), False),
    ],
)
def test_gdn_dispatch_follows_scheduled_and_computed_tokens(request_, decode):
    assert NEXT_GDN.takes_decode(request_) is decode
    chunk = NEXT_GDN.chunk_flops(request_.num_scheduled_tokens)
    assert NEXT_GDN.step_flops(request_) == (NEXT_GDN.decode_flops() if decode else chunk)


def test_gdn_speculative_steps_are_rejected():
    request = dataclasses.replace(sim_req("r", 3, 10, 10), num_spec_tokens=2)
    with pytest.raises(ValueError, match="speculative"):
        NEXT_GDN.step_flops(request)


# Qwen3-Next's pattern on a small dense model: three GDN layers, then one
# of full attention.
HYBRID = dataclasses.replace(
    QWEN_7B,
    num_layers=4,
    layers=(LayerGeometry(1000, recurrent=NEXT_GDN),) * 3 + (LayerGeometry(1000),),
)


def unit(geometry, **kwargs):
    return RooflineTimingProvider(
        None,
        peak_tflops=1.0,
        hbm_tbps=1.0,
        compute_efficiency=1.0,
        bandwidth_efficiency=1.0,
        fixed_us=0.0,
        geometry=geometry,
        **kwargs,
    )


def test_gdn_layers_take_no_attention():
    assert HYBRID._attention_layers == (1, ())
    assert HYBRID.kv_bytes_per_token == 2 * 4 * 128 * 2
    request = sim_req("r", 100, 0, 100)
    assert HYBRID.attention_flops((request,), (False,)) == 4 * 28 * 128 * 5050
    assert HYBRID.layer_attention_flops(HYBRID.layers[0], request, False) == (
        NEXT_GDN.chunk_flops(100)
    )
    assert HYBRID.layer_kv_bytes(HYBRID.layers[0], request) == NEXT_GDN.chunk_bytes(100)


def test_a_mixed_batch_prices_each_requests_gdn_path():
    prefill, decode = sim_req("p", 300, 0, 300), sim_req("d", 1, 500, 400)
    cost = unit(HYBRID).estimate(sim_step(0, prefill, decode))
    lm_head = 3584 * 152064
    flops = (
        2 * 4000 * 301
        + 2 * lm_head * 2
        + 4 * 28 * 128 * (300 * 301 // 2 + 501)
        + 3 * (NEXT_GDN.chunk_flops(300) + NEXT_GDN.decode_flops())
    )
    memory = (
        (4000 + lm_head) * 2
        + (300 + 501 + 301) * 2 * 4 * 128 * 2
        + 3 * (NEXT_GDN.chunk_bytes(300) + NEXT_GDN.decode_bytes())
    )
    assert cost.breakdown["compute_us"] == pytest.approx(flops / 1e12 * 1e6)
    assert cost.breakdown["memory_us"] == pytest.approx(memory / 1e12 * 1e6)
    assert cost.breakdown["gdn_decode_reqs"] == 1


def test_gdn_dual_rank_ep_paces_each_ranks_own_requests():
    """DP=2 with EP: rank 0's prefill and rank 1's decodes each price their own
    GDN layers before the MoE layer; an idle rank pays none of it."""
    gdn = LayerGeometry(10_000, recurrent=NEXT_GDN)
    geometry = dataclasses.replace(TWO_RANKS, layers=(gdn, TWO_RANKS.layers[1]))
    prefill = on_rank(0, sim_req("a", 1000, 0, 2000))
    decodes = [on_rank(1, sim_req(f"d{i}", 1, 10, 10)) for i in range(50)]
    cost = unit_rates(geometry).estimate(sim_step(0, prefill, *decodes))

    def seconds(flops, nbytes):
        return max(flops / 1e12, nbytes / 1e12)

    # Before the MoE layer's dispatch: the GDN layer, then the MoE layer's
    # own attention (one head of 32 for K and V, bf16).
    dense = 10_000 + 1000
    kv = 2 * 1 * 32 * 2
    before = [
        seconds(
            2 * dense * 1000 + NEXT_GDN.chunk_flops(1000) + 4 * 2 * 32 * (1000 * 1001 // 2),
            dense * 2 + NEXT_GDN.chunk_bytes(1000) + (1000 + 1000) * kv,
        ),
        seconds(
            2 * dense * 50 + 50 * NEXT_GDN.decode_flops() + 4 * 2 * 32 * 50 * 11,
            dense * 2 + 50 * NEXT_GDN.decode_bytes() + 50 * (11 + 1) * kv,
        ),
    ]
    lm_head = 64 * 1_000_000
    after = [seconds(0, lm_head * 2), seconds(2 * lm_head * 50, lm_head * 2)]
    assert cost.breakdown["local_us"] == pytest.approx(
        (max(before) + max(after)) * 1e6
    )
    idle = unit_rates(geometry).estimate(sim_step(0, *decodes))
    assert idle.breakdown["idle_dp_ranks"] == 1
    assert idle.breakdown["local_us"] == pytest.approx((before[1] + after[1]) * 1e6)


class StubGDN:
    """A built `QwenGatedDeltaNetAttention`'s fields: global heads and
    convolution width, its rank's state shapes and dtypes."""

    def __init__(self, tp, conv_dtype=torch.bfloat16, ssm_dtype=torch.bfloat16, shape=None):
        self.tp_size = tp
        self.num_k_heads, self.num_v_heads = 16, 32
        self.head_k_dim = self.head_v_dim = 128
        self.conv_dim, self.conv_kernel_size = 8192, 4
        self.num_spec = 0
        self.gdn_prefill_backend = "triton"
        self._shape = shape or (32 // tp, 128, 128)
        self._dtypes = (conv_dtype, ssm_dtype)

    def get_state_shape(self):
        return (3, 8192 // self.tp_size), self._shape

    def get_state_dtype(self):
        return self._dtypes


def test_gdn_geometry_is_this_ranks_once():
    one, two = (RecurrentGeometry.from_module(StubGDN(tp), 2) for tp in (1, 2))
    assert one == dataclasses.replace(NEXT_GDN, constructed_prefill="triton")
    assert (two.num_k_heads, two.num_v_heads, two.conv_dim) == (8, 16, 4096)
    # The rank's state shapes are not divided again.
    assert (two.ssm_state_bytes, two.conv_state_bytes) == (STATE // 2, CONV // 2)
    assert two.decode_flops() * 2 == one.decode_flops()
    assert two.chunk_bytes(640) * 2 == pytest.approx(one.chunk_bytes(640))
    mixed = RecurrentGeometry.from_module(StubGDN(1, ssm_dtype=torch.float32), 2)
    assert (mixed.conv_state_bytes, mixed.ssm_state_bytes, mixed.ssm_dtype_bytes) == (
        CONV,
        2 * STATE,
        4,
    )
    with pytest.raises(ValueError, match="state on this rank of"):
        RecurrentGeometry.from_module(StubGDN(2, shape=(32, 128, 128)), 2)
    with pytest.raises(ValueError, match="divisible by the tensor-parallel size 3"):
        RecurrentGeometry.from_module(StubGDN(3), 2)


def test_gdn_assumptions_are_recorded():
    recorded = unit(HYBRID).assumptions()["recurrent"]
    assert recorded["modeled_prefill"] == "triton_fla"
    assert recorded["chunk_size"] == 64
    assert "FlashInfer" in recorded["not_modeled"][0]
    groups = unit(HYBRID).assumptions()["geometry"]["layers"]
    assert [g.get("recurrent", {}).get("num_v_heads") for g in groups] == [32, None]
    with pytest.raises(ValueError, match="GDN prefill"):
        unit(HYBRID, gdn_prefill_backend="flashinfer")


def gdn_layer(kind=None, **fields):
    """A built gated delta-net of `kind` (default Qwen's) with
    `StubGDN`'s fields, without the kernels its constructor selects."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    kind = kind or QwenGatedDeltaNetAttention
    layer = kind.__new__(kind)
    torch.nn.Module.__init__(layer)
    stub = StubGDN(1)
    for name in ("tp_size", "num_k_heads", "num_v_heads", "head_k_dim", "head_v_dim",
                 "conv_dim", "conv_kernel_size", "num_spec", "gdn_prefill_backend"):
        setattr(layer, name, getattr(stub, name))
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.cache_config = SimpleNamespace(mamba_cache_dtype="auto", mamba_ssm_cache_dtype="auto")
    for name, value in fields.items():
        setattr(layer, name, value)
    return layer


def hybrid_config(**cache):
    config = qwen_7b_config("auto")
    config.model_config.is_hybrid = True
    config.model_config.quantization = None
    config.cache_config.mamba_cache_mode = cache.get("mamba_cache_mode", "none")
    config.cache_config.enable_prefix_caching = cache.get("enable_prefix_caching", False)
    config.speculative_config = cache.get("speculative_config")
    config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
    )
    return config


def hybrid_model(layer):
    """28 decoder layers, layer `i`'s mixer `layer(i)`."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for index in range(28):
        block = torch.nn.Module()
        block.mixer = layer(index)
        model.layers.append(block)
    return model


def test_a_built_gdn_model_records_its_layers():
    model = hybrid_model(lambda i: gdn_layer() if i % 4 < 3 else attention_layer())
    geometry = ModelGeometry.from_model(hybrid_config(), model)
    assert [layer.recurrent is not None for layer in geometry.layers[:4]] == [
        True, True, True, False
    ]
    assert geometry.layers[0].recurrent == dataclasses.replace(
        NEXT_GDN, constructed_prefill="triton"
    )
    assert geometry._attention_layers == (7, ())
    with pytest.raises(ValueError, match="hybrid model's built layers"):
        ModelGeometry.from_config(hybrid_config())


def test_other_recurrent_layers_and_layouts_are_rejected():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    class OtherGDN(QwenGatedDeltaNetAttention):  # e.g. Kimi's or OLMo's
        pass

    def beside(index):
        block = torch.nn.Module()
        block.gdn, block.attn = gdn_layer(), attention_layer()
        return block

    for model, message in (
        (hybrid_model(lambda _: gdn_layer(OtherGDN)), r"only, not \['OtherGDN'\]"),
        (hybrid_model(beside), "without attention beside it"),
        (hybrid_model(lambda _: gdn_layer(num_spec=2)), "speculative state"),
    ):
        with pytest.raises(ValueError, match=message):
            ModelGeometry.from_model(hybrid_config(), model)
    model = hybrid_model(lambda _: gdn_layer())
    for cache in (
        {"mamba_cache_mode": "align"},
        {"enable_prefix_caching": True},
        {"speculative_config": object()},
    ):
        with pytest.raises(ValueError, match="mamba_cache_mode none"):
            ModelGeometry.from_model(hybrid_config(**cache), model)


QWEN3_NEXT = {
    "architectures": ["Qwen3NextForCausalLM"],
    "model_type": "qwen3_next",
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "full_attention_interval": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "norm_topk_prob": True,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000,
    "tie_word_embeddings": False,
}


def test_a_built_qwen3_next_is_timed_from_its_gdn_layers(tmp_path, monkeypatch):
    model_dir = write_config(tmp_path / "qwen3-next", QWEN3_NEXT)
    steps = []
    record = timing_record(
        tmp_path, model_dir, monkeypatch, steps=steps, block_size=None,
        enable_prefix_caching=False,
    )
    groups = record["geometry"]["layers"]
    recurrent = [g.get("recurrent") for g in groups for _ in range(g["last"] - g["first"] + 1)]
    assert [r is not None for r in recurrent] == [True, True, True, False]
    assert recurrent[0] == {
        "num_k_heads": 2,
        "num_v_heads": 4,
        "head_k_dim": 16,
        "head_v_dim": 16,
        "conv_dim": 2 * 2 * 16 + 4 * 16,
        "conv_kernel": 4,
        "conv_state_bytes": 3 * 128 * 2,
        "ssm_state_bytes": 4 * 16 * 16 * 2,
        "ssm_dtype_bytes": 2,
        "activation_bytes": 2,
        "constructed_prefill": "triton",
    }
    assert record["recurrent"]["constructed_prefill"] == ["triton"]
    # 40 prompt tokens, then a decode step.
    assert [s["cost"]["breakdown"]["gdn_decode_reqs"] for s in steps] == [0, 1]


from vllm.v1.worker.simulated.timing import Collective, Mamba2Geometry  # noqa: E402

# Nemotron-Nano-9B-v2's Mamba2 on one rank: 128 heads of 80, 8 groups,
# states of 128, chunks of 128; an FP32 SSM state, a BF16 convolution.
NEMOTRON = Mamba2Geometry(
    num_heads=128,
    head_dim=80,
    n_groups=8,
    state_size=128,
    conv_dim=12288,
    conv_kernel=4,
    chunk_size=128,
    conv_state_bytes=3 * 12288 * 2,
    ssm_state_bytes=128 * 80 * 128 * 4,
    ssm_dtype_bytes=4,
    activation_bytes=2,
)
M2_STATE, M2_CONV = 128 * 80 * 128 * 4, 3 * 12288 * 2
M2_CHUNK_FLOPS = 128 * (4 * 128 * 80 * 128 + 2 * 128 * 128 * 80 + 2 * 80 * 128) + (
    8 * 2 * 128 * 128 * 128
)


@pytest.mark.parametrize(
    "computed_scheduled",
    [
        [(0, 65), (0, 65)],  # fresh: a chunk each
        [(127, 2)],  # finishes a chunk, starts another
        [(128, 128)],  # aligned continuation
        [(100, 300), (0, 1), (256, 129)],
        [(5, 1000)],
    ],
)
def test_mamba2_chunks_follow_each_requests_own_positions(computed_scheduled):
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder

    computed = torch.tensor([c for c, _ in computed_scheduled])
    starts = [0]
    for _, t in computed_scheduled:
        starts.append(starts[-1] + t)
    _, seq_idx, _, _ = BaseMambaAttentionMetadataBuilder._compute_chunk_metadata(
        128, len(computed_scheduled), computed, torch.tensor(starts)
    )
    native = [seq_idx.count(i) for i in range(len(computed_scheduled))]
    ours = [NEMOTRON.chunks(sim_req("r", t, c, c + t)) for c, t in computed_scheduled]
    assert ours == native


def test_a_mamba2_update_reads_and_writes_its_states():
    decode = sim_req("d", 1, 500, 400)
    assert NEMOTRON.batch_flops((decode,)) == 5 * 128 * 80 * 128 == 6_553_600
    assert NEMOTRON.batch_bytes((decode,)) == 2 * M2_STATE + 2 * M2_CONV


def ssd_bytes(geometry, chunked, prepared):
    """The SSD batch's logical bytes, term by term."""
    h, p, n, g, c = 128, 80, 128, 8, 128
    a, state, conv = 2, geometry.ssm_state_bytes, geometry.conv_state_bytes
    b = len(chunked)
    chunks = sum(geometry.chunks(r) for r in chunked)
    tokens = sum(r.num_scheduled_tokens for r in chunked)
    terms = {
        "preparation": 6 * b * state if prepared else 0,
        "final": b * state,
        "conv": 2 * b * conv,
        "dt_dA": 2 * h * chunks * c * 4 + 4 * h * chunks * c * 4 + h * chunks * 4,
        "chunk_states": 2 * chunks * h * p * n * 4,
        "passed": (2 * chunks - b) * state,
        "cb": 4 * chunks * g * c * c + 4 * chunks * h * c * c,
        "inputs": 2 * tokens * h * p * a + 2 * tokens * (h + g) * n * a + tokens * h * a,
        "output": tokens * h * p * a,
    }
    # The passed states and the final writes: 2 n S.
    assert terms["passed"] + terms["final"] == 2 * chunks * state
    return sum(terms.values())


def test_a_fresh_ssd_prefill_prepares_no_state():
    fresh = sim_req("f", 300, 0, 300)  # 3 chunks
    assert NEMOTRON.batch_flops((fresh,)) == 3 * M2_CHUNK_FLOPS
    assert NEMOTRON.batch_bytes((fresh,)) == ssd_bytes(NEMOTRON, [fresh], prepared=False)


def test_a_continuing_prefill_prepares_every_prefill_of_its_batch():
    fresh, later = sim_req("f", 300, 0, 300), sim_req("l", 2, 127, 400)
    assert NEMOTRON.chunks(later) == 2
    batch = (fresh, later)
    assert NEMOTRON.batch_flops(batch) == 5 * M2_CHUNK_FLOPS
    assert NEMOTRON.batch_bytes(batch) == ssd_bytes(NEMOTRON, [fresh, later], prepared=True)
    # With a decode beside them: its in-place update on top.
    decode = sim_req("d", 1, 50, 40)
    assert NEMOTRON.batch_bytes(batch + (decode,)) == NEMOTRON.batch_bytes(batch) + (
        2 * M2_STATE + 2 * M2_CONV
    )


def test_mamba2_state_bytes_follow_their_dtype():
    bf16 = dataclasses.replace(NEMOTRON, ssm_state_bytes=M2_STATE // 2, ssm_dtype_bytes=2)
    decode = sim_req("d", 1, 500, 400)
    assert bf16.batch_bytes((decode,)) == M2_STATE + 2 * M2_CONV
    fresh = sim_req("f", 300, 0, 300)
    assert bf16.batch_bytes((fresh,)) == ssd_bytes(bf16, [fresh], prepared=False)


class StubMamba2:
    """A built `MambaMixer2`'s fields: global heads, groups (extended for
    tensor parallelism) and convolution width, its rank's state shapes."""

    def __init__(self, tp, heads=128, head_dim=80, groups=8, ssm=torch.float32):
        self.tp_size = tp
        self.num_heads, self.head_dim, self.ssm_state_size = heads, head_dim, 128
        # One group follows the heads under tensor parallelism: extended.
        self.n_groups = groups if groups % tp == 0 else groups + tp - groups
        self.conv_dim = heads * head_dim + 2 * self.n_groups * 128
        self.conv_kernel_size = 4
        self.num_spec = 0
        self.use_replayssm = False
        self._ssm = ssm

    def get_state_shape(self):
        tp = self.tp_size
        return (3, self.conv_dim // tp), (self.num_heads // tp, self.head_dim, 128)

    def get_state_dtype(self):
        return torch.bfloat16, self._ssm


def test_mamba2_geometry_is_this_ranks_once():
    one = Mamba2Geometry.from_module(StubMamba2(1), 2, 128)
    assert one == NEMOTRON
    two = Mamba2Geometry.from_module(StubMamba2(2), 2, 128)
    assert (two.num_heads, two.n_groups, two.conv_dim) == (64, 4, 6144)
    assert two.ssm_state_bytes == M2_STATE // 2
    # Granite: one group, replicated on each rank.
    granite = Mamba2Geometry.from_module(
        StubMamba2(2, head_dim=64, groups=1, ssm=torch.bfloat16), 2, 256
    )
    assert (granite.num_heads, granite.n_groups, granite.conv_dim) == (64, 1, 4352)
    assert granite.ssm_state_bytes + granite.conv_state_bytes == 1_074_688
    for name, value in (("num_spec", 1), ("use_replayssm", True)):
        stub = StubMamba2(1)
        setattr(stub, name, value)
        with pytest.raises(ValueError, match="speculative state or ReplaySSM"):
            Mamba2Geometry.from_module(stub, 2, 128)


def test_collectives_are_summed_by_size():
    """Two layers of one hidden reduction each, one with an FP32 norm sum,
    on two ranks of 1 GB/s and no latency."""
    hidden = Collective("hidden", 3584, 2)
    norm = Collective("gated_norm_sums", 1, 4)
    geometry = dataclasses.replace(
        QWEN_7B,
        tensor_parallel_size=2,
        num_layers=2,
        layers=(
            LayerGeometry(1000, collectives=(hidden,)),
            LayerGeometry(1000, collectives=(hidden, norm)),
        ),
    )
    provider = RooflineTimingProvider(
        None, gpu="h100", geometry=geometry, interconnect_gbs=1.0,
        interconnect_efficiency=1.0, collective_latency_us=10.0,
    )
    tokens = 100
    expected = 3 * 10e-6 + (2 * 100 * 3584 * 2 + 100 * 4) / 1e9
    assert provider.communication_s(tokens) == pytest.approx(expected)
    # Layers without their own: two hidden reductions each, as before.
    legacy = dataclasses.replace(geometry, layers=(LayerGeometry(1000),) * 2)
    plain = RooflineTimingProvider(
        None, gpu="h100", geometry=legacy, interconnect_gbs=1.0,
        interconnect_efficiency=1.0, collective_latency_us=10.0,
    )
    assert plain.communication_s(tokens) == pytest.approx(4 * 10e-6 + 4 * 100 * 3584 * 2 / 1e9)


def test_a_shared_mlp_after_the_moe_combine_runs_in_the_next_segment():
    """Granite under DP+EP: the shared MLP's weights, a part of the layer's
    dense ones, move after the experts. Rank 0 prefills, rank 1 decodes;
    the dense layer before is heavy, the shared MLP heavier still."""
    moe_layer = LayerGeometry(1000 + 50_000_000, (SEGMENT_EXPERTS,), post_moe_params=50_000_000)
    geometry = dataclasses.replace(TWO_RANKS, layers=(LayerGeometry(10_000_000), moe_layer))
    prefill = on_rank(0, sim_req("a", 1000, 0, 2000))
    decodes = [on_rank(1, sim_req(f"d{i}", 1, 10, 10)) for i in range(50)]
    after = unit_rates(geometry).estimate(sim_step(0, prefill, *decodes))
    before = unit_rates(
        dataclasses.replace(geometry, layers=(geometry.layers[0], dataclasses.replace(moe_layer, post_moe_params=0)))
    ).estimate(sim_step(0, prefill, *decodes))
    assert after.breakdown["local_us"] != pytest.approx(before.breakdown["local_us"])

    def seconds(flops, nbytes):
        return max(flops / 1e12, nbytes / 1e12)

    kv = 2 * 32 * 2
    dense, shared = 10_000_000 + 1000, 50_000_000
    lm_head = 64 * 1_000_000
    first = [
        seconds(2 * dense * 1000 + 2 * 4 * 2 * 32 * (1000 * 1001 // 2), dense * 2 + 2 * (1000 + 1000) * kv),
        seconds(2 * dense * 50 + 2 * 4 * 2 * 32 * 50 * 11, dense * 2 + 2 * 50 * 12 * kv),
    ]
    second = [
        seconds(2 * shared * 1000, (shared + lm_head) * 2),
        seconds(2 * shared * 50 + 2 * lm_head * 50, (shared + lm_head) * 2),
    ]
    assert after.breakdown["local_us"] == pytest.approx((max(first) + max(second)) * 1e6)
    # The weights read are conserved: only where they are read moved.
    assert moe_layer.dense_params == dense - 10_000_000 + shared


def test_dp_ranks_prepare_states_for_their_own_batches():
    """A fresh prefill on rank 0, a continuing one on rank 1: only rank 1's
    batch prepares its state."""
    layer = LayerGeometry(10_000, recurrent=NEMOTRON, attention=False)
    geometry = dataclasses.replace(TWO_RANKS, layers=(layer, TWO_RANKS.layers[1]))
    fresh, later = sim_req("f", 300, 0, 300), sim_req("l", 2, 127, 400)
    cost = unit_rates(geometry).estimate(sim_step(0, on_rank(0, fresh), on_rank(1, later)))

    def seconds(flops, nbytes):
        return max(flops / 1e12, nbytes / 1e12)

    kv = 2 * 32 * 2
    dense = 10_000 + 1000
    lm_head = 64 * 1_000_000
    first = [
        seconds(2 * dense * 300 + NEMOTRON.batch_flops((fresh,)) + 4 * 2 * 32 * (300 * 301 // 2),
                dense * 2 + NEMOTRON.batch_bytes((fresh,)) + 600 * kv),
        seconds(2 * dense * 2 + NEMOTRON.batch_flops((later,)) + 4 * 2 * 32 * (2 * 127 + 3),
                dense * 2 + NEMOTRON.batch_bytes((later,)) + (2 + 127 + 2) * kv),
    ]
    second = [seconds(2 * lm_head, lm_head * 2), seconds(0, lm_head * 2)]
    assert cost.breakdown["local_us"] == pytest.approx((max(first) + max(second)) * 1e6)
    # Pooled, the fresh prefill would have been prepared too.
    assert NEMOTRON.batch_bytes((fresh, later)) > (
        NEMOTRON.batch_bytes((fresh,)) + NEMOTRON.batch_bytes((later,))
    )


def mamba2_layer(kind=None, groups=8, **fields):
    """A built Mamba2 mixer of `kind` (default `MambaMixer2`) with
    `StubMamba2`'s fields and a gated norm of `groups` groups."""
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    kind = kind or MambaMixer2
    layer = kind.__new__(kind)
    torch.nn.Module.__init__(layer)
    stub = StubMamba2(1, groups=groups)
    for name in ("tp_size", "num_heads", "head_dim", "ssm_state_size", "n_groups",
                 "conv_dim", "conv_kernel_size", "num_spec", "use_replayssm"):
        setattr(layer, name, getattr(stub, name))
    layer.get_state_shape = stub.get_state_shape
    layer.get_state_dtype = stub.get_state_dtype
    layer.norm = SimpleNamespace(n_groups=groups, use_rms_norm=True)
    for name, value in fields.items():
        setattr(layer, name, value)
    return layer


def decoder_layer(name, **children):
    """A decoder layer of class `name` (the collectives follow it)."""
    layer = type(name, (torch.nn.Module,), {})()
    for child, module in children.items():
        setattr(layer, child, module)
    return layer


def mamba2_config(tp=1):
    config = hybrid_config()
    config.model_config.get_mamba_chunk_size = lambda: 128
    config.parallel_config.tensor_parallel_size = tp
    return config


def mamba2_model(layers):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    return model


def test_nemotron_layers_reduce_once_and_mlp_only_layers_take_no_attention():
    pattern = ["M", "-", "M", "-"] * 7  # Nemotron's `M-M-`, no attention
    layers = [
        decoder_layer("NemotronHMambaDecoderLayer", mixer=mamba2_layer())
        if kind == "M"
        else decoder_layer("NemotronHMLPDecoderLayer", mixer=torch.nn.Linear(8, 8, device="meta"))
        for kind in pattern
    ]
    geometry = ModelGeometry.from_model(mamba2_config(), mamba2_model(layers))
    hidden = Collective("hidden", 3584, 2)
    assert all(layer.collectives == (hidden,) for layer in geometry.layers)
    assert geometry._attention_layers == (0, ())
    assert geometry.kv_bytes_per_token == 0
    assert [layer.recurrent is not None for layer in geometry.layers[:2]] == [True, False]
    request = sim_req("r", 100, 0, 100)
    assert geometry.attention_flops((request,), (False,)) == 0
    assert geometry.recurrent_flops((request,)) == 14 * NEMOTRON.batch_flops((request,))
    # The saved record counts one all-reduce a layer, not two.
    record = RooflineTimingProvider(None, gpu="h100", geometry=geometry).assumptions()
    assert record["communication"]["event_source"] == "layers"
    assert record["communication"]["events_per_step"] == [
        {"label": "hidden", "elements": 3584, "dtype_bytes": 2, "count": 28}
    ]


def test_granite_layers_reduce_each_branch_they_have():
    moe, shared = torch.nn.Module(), torch.nn.Module()
    shared.down = torch.nn.Linear(16, 8, bias=False, device="meta")
    hidden = Collective("hidden", 3584, 2)
    norm = Collective("gated_norm_sums", 1, 4)
    for tp, children, events, post_moe in (
        (1, {"block_sparse_moe": moe, "shared_mlp": shared}, (hidden,) * 3, 128),
        # A missing branch reduces nothing.
        (1, {"block_sparse_moe": moe}, (hidden,) * 2, 0),
        # One group on two ranks: its norm's FP32 sums too.
        (2, {"block_sparse_moe": moe, "shared_mlp": shared}, (hidden,) * 3 + (norm,), 128),
    ):
        layer = decoder_layer(
            "GraniteMoeHybridMambaDecoderLayer", mamba=mamba2_layer(groups=1), **children
        )
        layer.mamba.tp_size = tp
        layer.mamba.get_state_shape = StubMamba2(tp, groups=1).get_state_shape
        layer.mamba.n_groups = StubMamba2(tp, groups=1).n_groups
        layer.mamba.conv_dim = StubMamba2(tp, groups=1).conv_dim
        geometry = ModelGeometry.from_model(mamba2_config(tp), mamba2_model([layer] * 28))
        assert geometry.layers[0].collectives == events, (tp, children)
        assert geometry.layers[0].post_moe_params == post_moe
    # The saved record of the last (two ranks, every branch): three
    # hidden-width all-reduces a layer and the norm's FP32 sums.
    record = RooflineTimingProvider(None, gpu="h100", geometry=geometry).assumptions()
    assert record["communication"]["event_source"] == "layers"
    assert record["communication"]["events_per_step"] == [
        {"label": "hidden", "elements": 3584, "dtype_bytes": 2, "count": 3 * 28},
        {"label": "gated_norm_sums", "elements": 1, "dtype_bytes": 4, "count": 28},
    ]


def test_other_mamba2_layouts_are_rejected():
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    class SubMixer(MambaMixer2):
        pass

    def model(layer):
        return mamba2_model([layer] * 28)

    cases = (
        (decoder_layer("NemotronHMambaDecoderLayer", mixer=mamba2_layer(SubMixer)),
         r"not \['SubMixer'\]", 1),
        (decoder_layer("NemotronHMambaDecoderLayer", mixer=mamba2_layer(use_replayssm=True)),
         "ReplaySSM", 1),
        (decoder_layer("NemotronHMoEDecoderLayer", mixer=torch.nn.Linear(8, 8, device="meta")),
         "NemotronHMoEDecoderLayer's collectives", 1),
        (decoder_layer("FalconH1DecoderLayer", mamba=mamba2_layer()),
         "FalconH1DecoderLayer's collectives", 1),
        (decoder_layer("FalconH1DecoderLayer", mamba=mamba2_layer(), attn=attention_layer()),
         "without attention beside it", 1),
        # Six groups on four ranks: the norm all-gathers.
        (decoder_layer("NemotronHMambaDecoderLayer", mixer=mamba2_layer(groups=6)), "all-gather", 4),
    )
    for layer, message, tp in cases:
        if tp > 1:
            layer.mixer.tp_size = tp
            layer.mixer.num_heads, layer.mixer.n_groups, layer.mixer.conv_dim = 128, 8, 8192
            layer.mixer.get_state_shape = lambda: ((3, 2048), (32, 80, 128))
        with pytest.raises(ValueError, match=message):
            ModelGeometry.from_model(mamba2_config(tp), model(layer))


NEMOTRON_H = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "hybrid_override_pattern": "M-*M",
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "mamba_num_heads": 8,
    "mamba_head_dim": 8,
    "n_groups": 2,
    "ssm_state_size": 16,
    "conv_kernel": 4,
    "chunk_size": 16,
    "mamba_hidden_act": "silu",
    "use_conv_bias": True,
    "mamba_proj_bias": False,
    "mlp_hidden_act": "relu2",
    "tie_word_embeddings": False,
}
GRANITE_4_H = {
    "architectures": ["GraniteMoeHybridForCausalLM"],
    "model_type": "granitemoehybrid",
    "hidden_size": 64,
    "intermediate_size": 32,
    "shared_intermediate_size": 64,
    "num_hidden_layers": 4,
    "layer_types": ["mamba", "mamba", "mamba", "attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "mamba_n_heads": 8,
    "mamba_d_head": 8,
    "mamba_n_groups": 1,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 1,
    "mamba_chunk_size": 16,
    "mamba_conv_bias": True,
    "mamba_proj_bias": False,
    "tie_word_embeddings": False,
}


@pytest.mark.parametrize(
    "config, kinds, events, post_moe",
    [
        # Mamba2, MLP-only, attention, Mamba2: one reduction each.
        (NEMOTRON_H, ["mamba2", None, "attention", "mamba2"], [1, 1, 1, 1], 0),
        # Mamba2 x 3, attention; each with routed experts and a shared MLP.
        (GRANITE_4_H, ["mamba2"] * 3 + ["attention"], [3, 3, 3, 3], 2 * 64 * 64 + 64 * 64),
    ],
    ids=["nemotron-h", "granite-4.0-h"],
)
def test_a_built_mamba2_hybrid_is_timed_from_its_layers(
    tmp_path, monkeypatch, config, kinds, events, post_moe
):
    model_dir = write_config(tmp_path / config["model_type"], config)
    steps = []
    record = timing_record(
        tmp_path, model_dir, monkeypatch, steps=steps, block_size=None,
        enable_prefix_caching=False,
    )
    expanded = [g for g in record["geometry"]["layers"] for _ in range(g["last"] - g["first"] + 1)]
    recorded = [
        "mamba2" if "mamba2" in g else ("attention" if g.get("attention", True) else None)
        for g in expanded
    ]
    assert recorded == kinds
    assert [len(g.get("collectives", [None, None])) for g in expanded] == events
    assert record["communication"]["event_source"] == "layers"
    assert record["communication"]["events_per_step"] == [
        {"label": "hidden", "elements": 64, "dtype_bytes": 2, "count": sum(events)}
    ]
    assert {g.get("post_moe_params", 0) for g in expanded} == {post_moe}
    mamba2 = next(g["mamba2"] for g in expanded if "mamba2" in g)
    assert (mamba2["num_heads"], mamba2["head_dim"], mamba2["state_size"], mamba2["chunk_size"]) == (
        8, 8, 16, 16
    )
    assert record["mamba2"]["modeled_prefill"].startswith("triton_ssd")
    assert [s["cost"]["breakdown"]["mamba2_decode_reqs"] for s in steps] == [0, 1]
