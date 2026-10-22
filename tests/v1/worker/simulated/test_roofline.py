# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Roofline step timing of a dense model on a GPU preset."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.worker.simulated.utils import sim_req, sim_step
from vllm.v1.worker.simulated.timing import ModelGeometry, RooflineTimingProvider

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


def qwen_7b_config(cache_dtype):
    model = SimpleNamespace(
        dtype=torch.bfloat16,
        use_mla=False,
        is_moe=False,
        is_hybrid=False,
        is_attention_free=False,
        hf_text_config=SimpleNamespace(intermediate_size=18944),
        get_num_layers=lambda parallel: 28,
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
