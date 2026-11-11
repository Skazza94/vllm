# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of real NIXL KV transfers under simulated compute."""

import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="Real KV transfers start with CPU buffers."
)

SIMULATED_NIXL = "vllm.v1.worker.simulated.kv_nixl"
SIMULATED_WORKER = "vllm.v1.worker.simulated.worker.SimulatedWorker"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A tiny dense Llama configuration: no weights or tokenizer are needed."""
    directory = tmp_path_factory.mktemp("tiny-llama")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "vocab_size": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 256,
                "torch_dtype": "float32",
            }
        )
    )
    return str(directory)


@pytest.fixture(scope="module")
def moe_model_dir(tmp_path_factory):
    """A tiny Qwen3-MoE configuration: dense attention, 8 experts, top-2."""
    directory = tmp_path_factory.mktemp("tiny-qwen3-moe")
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3MoeForCausalLM"],
                "model_type": "qwen3_moe",
                "vocab_size": 128,
                "hidden_size": 256,
                "intermediate_size": 512,
                "moe_intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "norm_topk_prob": True,
                "max_position_embeddings": 256,
                "torch_dtype": "float32",
            }
        )
    )
    return str(directory)


# A tiny DeepSeek-V3 (MLA and MoE) with DeepSeek's latent: kv_lora_rank 512
# plus a 64-wide rope key, 576 elements per token and layer.
DEEPSEEK_V3 = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "vocab_size": 128,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 256,
    "torch_dtype": "float32",
}
LATENT = 512 + 64


# DeepSeek-V4-Flash's published config: sparse MLA with sliding windows,
# compressed latents and compressor states (KV cache groups packed in one
# allocation), its cache the packed FP8 one.
DEEPSEEK_V4_FLASH = {'architectures': ['DeepseekV4ForCausalLM'],
 'attention_bias': False,
 'attention_dropout': 0.0,
 'bos_token_id': 0,
 'eos_token_id': 1,
 'expert_dtype': 'fp4',
 'hc_eps': 1e-06,
 'hc_mult': 4,
 'hc_sinkhorn_iters': 20,
 'head_dim': 512,
 'hidden_act': 'silu',
 'hidden_size': 4096,
 'index_head_dim': 128,
 'index_n_heads': 64,
 'index_topk': 512,
 'initializer_range': 0.02,
 'max_position_embeddings': 1048576,
 'model_type': 'deepseek_v4',
 'moe_intermediate_size': 2048,
 'n_routed_experts': 256,
 'n_shared_experts': 1,
 'norm_topk_prob': True,
 'num_attention_heads': 64,
 'num_experts_per_tok': 6,
 'num_hidden_layers': 43,
 'num_hash_layers': 3,
 'num_key_value_heads': 1,
 'num_nextn_predict_layers': 1,
 'o_groups': 8,
 'o_lora_rank': 1024,
 'q_lora_rank': 1024,
 'qk_rope_head_dim': 64,
 'quantization_config': {'activation_scheme': 'dynamic',
                         'fmt': 'e4m3',
                         'quant_method': 'fp8',
                         'scale_fmt': 'ue8m0',
                         'weight_block_size': [128, 128]},
 'rms_norm_eps': 1e-06,
 'rope_scaling': {'beta_fast': 32,
                  'beta_slow': 1,
                  'factor': 16,
                  'original_max_position_embeddings': 65536,
                  'type': 'yarn'},
 'rope_theta': 10000,
 'routed_scaling_factor': 1.5,
 'scoring_func': 'sqrtsoftplus',
 'sliding_window': 128,
 'swiglu_limit': 10.0,
 'tie_word_embeddings': False,
 'topk_method': 'noaux_tc',
 'torch_dtype': 'bfloat16',
 'transformers_version': '4.57.1',
 'use_cache': True,
 'vocab_size': 129280,
 'compress_rope_theta': 160000,
 'compress_ratios': [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
                     128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
                     128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0]}


@pytest.fixture(scope="module")
def v4_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("deepseek-v4-flash")
    (directory / "config.json").write_text(json.dumps(DEEPSEEK_V4_FLASH))
    return str(directory)


@pytest.fixture(scope="module")
def mla_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-deepseek-v3")
    (directory / "config.json").write_text(json.dumps(DEEPSEEK_V3))
    return str(directory)


# A tiny gpt-oss: sliding-window (32 tokens) and full attention alternating,
# two KV cache groups sharing memory.
GPT_OSS = {
    "architectures": ["GptOssForCausalLM"],
    "model_type": "gpt_oss",
    "vocab_size": 128,
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
    "max_position_embeddings": 256,
}


@pytest.fixture(scope="module")
def gpt_oss_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-gpt-oss")
    (directory / "config.json").write_text(json.dumps(GPT_OSS))
    return str(directory)


# A tiny Kimi-K3 language model (Kimi-Linear with its full-rank-gate KDA,
# KimiK3DeltaAttention): three KDA layers, then MLA; SiTU latent experts
# beside a shared one, attention residual blocks.
KIMI_K3 = {
    "architectures": ["KimiLinearForCausalLM"],
    "model_type": "kimi_linear",
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "routed_expert_hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 16,
    "v_head_dim": 32,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "linear_attn_config": {
        "kda_layers": [1, 2, 3],
        "full_attn_layers": [4],
        "num_heads": 4,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "use_full_rank_gate": True,
    },
    "num_experts": 4,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "num_expert_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "use_grouped_topk": True,
    "routed_scaling_factor": 1.0,
    "latent_moe_use_norm": True,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "attn_res_block_size": 2,
    "num_nextn_predict_layers": 0,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
}


# A tiny Qwen3.5-MoE, a vision-language wrapper served as its language model
# (language_model_only): three gated delta-net layers, then full attention.
QWEN3_5_MOE = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "image_token_id": 250,
    "video_token_id": 251,
    "vision_end_token_id": 252,
    "vision_start_token_id": 253,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 256,
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "attn_output_gate": True,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": 0,
        "max_position_embeddings": 256,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        "dtype": "float32",
    },
    "vision_config": {
        "model_type": "qwen3_5_moe",
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
        "num_position_embeddings": 64,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": [],
    },
}


@pytest.fixture(scope="module")
def qwen3_5_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-qwen3-5-moe")
    (directory / "config.json").write_text(json.dumps(QWEN3_5_MOE))
    return str(directory)


# A tiny Nemotron-H: Mamba2, MLP-only, attention, Mamba2 layers.
NEMOTRON_H = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "vocab_size": 128,
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
    "max_position_embeddings": 256,
}
# A tiny Nemotron-3: Mamba2, MoE (non-gated relu^2 experts and a shared
# one), attention layers.
NEMOTRON_3 = {
    **NEMOTRON_H,
    "num_hidden_layers": 5,
    "hybrid_override_pattern": "MEM*E",
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "n_shared_experts": 1,
    "moe_shared_expert_intermediate_size": 32,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
}
# A tiny Falcon-H1: each layer runs a Mamba2 mixer beside attention.
FALCON_H1 = {
    "architectures": ["FalconH1ForCausalLM"],
    "model_type": "falcon_h1",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "mamba_d_ssm": 64,
    "mamba_n_heads": 8,
    "mamba_d_head": 8,
    "mamba_n_groups": 1,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 1,
    "mamba_chunk_size": 16,
    "tie_word_embeddings": False,
    "max_position_embeddings": 256,
}
# A tiny Zamba2: Mamba2 layers, two of them hybrid (a shared transformer
# block, attention, beside the mixer).
ZAMBA2 = {
    "architectures": ["Zamba2ForCausalLM"],
    "model_type": "zamba2",
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "layers_block_type": ["mamba", "hybrid", "mamba", "hybrid"],
    "hybrid_layer_ids": [1, 3],
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "attention_head_dim": 32,
    "num_mem_blocks": 1,
    "use_shared_attention_adapter": False,
    "use_mem_rope": False,
    "mamba_d_state": 16,
    "mamba_d_conv": 4,
    "mamba_expand": 2,
    "mamba_ngroups": 1,
    "n_mamba_heads": 8,
    "mamba_headdim": 16,
    "chunk_size": 16,
    "tie_word_embeddings": False,
    "max_position_embeddings": 256,
}


def write_model_dir(tmp_path_factory, name, config):
    directory = tmp_path_factory.mktemp(name)
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


@pytest.fixture(scope="module")
def nemotron_h_model_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory, "tiny-nemotron-h", NEMOTRON_H)


@pytest.fixture(scope="module")
def nemotron_3_model_dir(tmp_path_factory):
    return write_model_dir(tmp_path_factory, "tiny-nemotron-3", NEMOTRON_3)


@pytest.fixture(params=["model_dir", "moe_model_dir", "mla_model_dir"])
def any_model_dir(request):
    """The dense model, the MoE one, then the MLA one: blocks of each layer's
    KV, K and V heads or one latent."""
    return request.getfixturevalue(request.param)


def engine_config(model_dir, simulation=None, kv_transfer=None, **kwargs):
    from vllm.engine.arg_utils import EngineArgs

    options = {
        "model": model_dir,
        "skip_tokenizer_init": True,
        "dtype": "float32",
        "max_model_len": 256,
        "block_size": 32,
        "kv_cache_memory_bytes": 8 * 1024**2,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "simulation_config": {
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            **(simulation or {}),
        },
        "kv_transfer_config": {
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu",
            **(kv_transfer or {}),
        },
    }
    options.update(kwargs)
    return EngineArgs(**options).create_engine_config()


def test_real_mode_keeps_the_native_connector(model_dir):
    config = engine_config(model_dir)
    assert config.simulation_config.kv_transfer_mode == "real"
    assert config.kv_transfer_config.kv_connector == "NixlConnector"
    # No module path: the connector factory resolves the native NIXL connector.
    assert config.kv_transfer_config.kv_connector_module_path is None
    assert config.parallel_config.worker_cls == SIMULATED_WORKER


def test_simulated_mode_is_the_default_and_unchanged(model_dir):
    config = engine_config(
        model_dir,
        simulation={"kv_transfer_mode": "simulated", "kv_transfer_timing": None},
    )
    assert config.kv_transfer_config.kv_connector_module_path == SIMULATED_NIXL
    from vllm.config.simulation import SimulationConfig

    assert SimulationConfig().kv_transfer_mode == "simulated"


@pytest.mark.parametrize(
    "simulation, kv_transfer, kwargs, error",
    [
        ({}, {"kv_connector": "NixlPushConnector"}, {}, "one of NixlConnector"),
        (
            {},
            {"kv_connector_module_path": SIMULATED_NIXL},
            {},
            "without kv_connector_module_path",
        ),
        ({}, {"kv_buffer_device": "cuda"}, {}, "CPU KV buffers"),
        ({"kv_transfer_timing": {"load_gbps": 1}}, {}, {}, "no kv_transfer_timing"),
        ({}, {}, {"kv_cache_memory_bytes": None}, "explicit kv_cache_memory_bytes"),
        ({"clock": "virtual"}, {}, {}, "the real clock"),
    ],
)
def test_real_mode_rejections(model_dir, simulation, kv_transfer, kwargs, error):
    with pytest.raises(ValueError, match=error):
        engine_config(model_dir, simulation, kv_transfer, **kwargs)


@pytest.mark.parametrize(
    "simulation, kv_transfer, error",
    [
        ({"kv_transfer_mode": "simulated", "kv_pattern_fill": True}, {}, "need kv_transfer_mode real"),
        ({"kv_pattern_fill": True}, {"kv_role": "kv_consumer"}, "kv_producer for kv_pattern_fill"),
        ({"kv_pattern_check": True, "log_path": "/tmp/steps"}, {"kv_role": "kv_producer"}, "kv_consumer for kv_pattern_check"),
        ({"kv_pattern_check": True}, {}, "log_path for kv_pattern_check"),
    ],
)
def test_pattern_options_need_real_mode_and_their_role(model_dir, simulation, kv_transfer, error):
    with pytest.raises(ValueError, match=error):
        engine_config(model_dir, simulation, kv_transfer)


def test_real_mode_allows_prefix_caching(model_dir, mla_model_dir):
    config = engine_config(model_dir, enable_prefix_caching=True)
    assert config.cache_config.enable_prefix_caching
    assert config.simulation_config.kv_transfer_mode == "real"
    # With the pattern checks too: a block's pattern is then its content's.
    check = {"kv_pattern_check": True, "log_path": "/tmp/steps"}
    for model in (model_dir, mla_model_dir):
        for connector in ("NixlConnector", "MooncakeConnector"):
            config = engine_config(
                model, check, {"kv_connector": connector}, enable_prefix_caching=True
            )
            assert config.simulation_config.kv_pattern_check


def test_pattern_checks_with_prefix_caching_take_dsa_and_refuse_mtp(tmp_path_factory):
    """A DSA layer's latent and indexer keys are its block's content like any
    page; speculative drafts are not validated with content patterns."""
    check = {"kv_pattern_check": True, "log_path": "/tmp/steps"}
    v32 = write_model_dir(tmp_path_factory, "tiny-deepseek-v3.2-pc", DEEPSEEK_V32)
    for prefix_caching in (True, False):
        config = engine_config(
            v32, check, dtype="bfloat16", block_size=64,
            enable_prefix_caching=prefix_caching,
        )
        assert config.cache_config.enable_prefix_caching is prefix_caching
    v3 = write_model_dir(tmp_path_factory, "tiny-deepseek-v3-pc", deepseek_v3(1))
    mtp = {"method": "mtp", "num_speculative_tokens": 1}
    with pytest.raises(ValueError, match="no speculative decoding with prefix caching"):
        engine_config(v3, check, speculative_config=mtp, enable_prefix_caching=True)
    engine_config(v3, check, speculative_config=mtp)


def test_simulated_blocks_need_a_multiple_of_16(model_dir):
    assert engine_config(model_dir, block_size=16).cache_config.block_size == 16
    with pytest.raises(ValueError, match="multiple of 16"):
        engine_config(model_dir, block_size=8)


def test_real_mode_accepts_moe_and_dense_mla_models(moe_model_dir, mla_model_dir, tmp_path):
    config = engine_config(moe_model_dir)
    assert config.model_config.is_moe
    assert config.simulation_config.kv_transfer_mode == "real"
    config = engine_config(mla_model_dir)
    assert config.model_config.use_mla
    # DeepSeek-V3.2: an indexer makes its MLA sparse.
    (tmp_path / "config.json").write_text(
        json.dumps({**DEEPSEEK_V3, "index_topk": 64, "index_n_heads": 2, "index_head_dim": 64})
    )
    with pytest.raises(ValueError, match="no sparse MLA"):
        engine_config(str(tmp_path))


def test_real_mode_takes_sliding_window_models_over_either_connector(
    gpt_oss_model_dir,
):
    for connector in ("MooncakeConnector", "NixlConnector", "NixlPullConnector"):
        config = engine_config(
            gpt_oss_model_dir, kv_transfer={"kv_connector": connector}
        )
        assert config.model_config.get_sliding_window() == 32


@pytest.fixture
def ds_conv_layout(monkeypatch):
    """The DS conv-state layout, its cached resolutions cleared around."""
    from vllm import envs
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    def clear():
        envs.disable_envs_cache()
        get_conv_state_layout.cache_clear()

    def use(layout):
        if layout is None:
            monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)
        else:
            monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
        clear()

    use("DS")
    yield use
    monkeypatch.delenv("VLLM_SSM_CONV_STATE_LAYOUT", raising=False)
    clear()


def hybrid_over_nixl_needs_ds(model_dir, ds_conv_layout, **kwargs):
    """Over NIXL, a hybrid model's state moves as its DS conv sub-projections
    and SSM state: refused in the SD layout (the default), taken in DS."""
    for layout in (None, "SD"):
        ds_conv_layout(layout)
        with pytest.raises(ValueError, match="DS conv-state layout"):
            engine_config(model_dir, **kwargs)
    ds_conv_layout("DS")
    for connector in ("NixlConnector", "NixlPullConnector"):
        config = engine_config(model_dir, kv_transfer={"kv_connector": connector}, **kwargs)
        assert config.cache_config.mamba_cache_mode == "none"
        cached = engine_config(
            model_dir,
            kv_transfer={"kv_connector": connector},
            enable_prefix_caching=True,
            **kwargs,
        )
        assert cached.cache_config.mamba_cache_mode == "align"


def test_real_mode_takes_gdn_hybrids_over_mooncake_or_nixl(
    qwen3_5_model_dir, ds_conv_layout
):
    model_dir, kwargs = qwen3_5_model_dir, {"language_model_only": True}
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(model_dir, kv_transfer=mooncake, **kwargs)
    assert config.model_config.is_hybrid
    assert config.cache_config.mamba_cache_mode == "none"
    # Prefix caching keeps a state a block (align mode), as vLLM sets it.
    cached = engine_config(
        model_dir, kv_transfer=mooncake, enable_prefix_caching=True, **kwargs
    )
    assert cached.cache_config.mamba_cache_mode == "align"
    hybrid_over_nixl_needs_ds(model_dir, ds_conv_layout, **kwargs)


def test_real_mode_takes_qwen3_5_native_mtp_over_nixl_only(
    tmp_path_factory, ds_conv_layout
):
    """Qwen3.5's one MTP layer (`mtp_num_hidden_layers`) moves with the
    target's KV over NIXL in the DS layout, its states' speculative slots
    left where they are: not over Mooncake, nor two MTP layers."""
    text = QWEN3_5_MOE["text_config"]
    one = write_model_dir(
        tmp_path_factory, "tiny-qwen3.5-mtp",
        {**QWEN3_5_MOE, "text_config": {**text, "mtp_num_hidden_layers": 1}},
    )
    two = write_model_dir(
        tmp_path_factory, "tiny-qwen3.5-mtp-two",
        {**QWEN3_5_MOE, "text_config": {**text, "mtp_num_hidden_layers": 2}},
    )

    def mtp(k=1):
        return {"method": "mtp", "num_speculative_tokens": k}

    for connector in ("NixlConnector", "NixlPullConnector"):
        for k in (1, 5):
            config = engine_config(
                one, kv_transfer={"kv_connector": connector},
                speculative_config=mtp(k), language_model_only=True,
            )
            assert config.speculative_config.draft_model_config.architectures == [
                "Qwen3_5MoeMTP"
            ]
    for model, kwargs, problem in (
        (one, {"kv_transfer": {"kv_connector": "MooncakeConnector"}}, "not MooncakeConnector"),
        (one, {"speculative_config": mtp(6)}, "6 speculative tokens"),
        (two, {}, "2 MTP layers"),
    ):
        kwargs.setdefault("speculative_config", mtp())
        with pytest.raises(ValueError, match=problem):
            engine_config(model, language_model_only=True, **kwargs)
    ds_conv_layout("SD")
    with pytest.raises(ValueError, match="DS conv-state layout"):
        engine_config(one, speculative_config=mtp(), language_model_only=True)


@pytest.mark.parametrize("model", ["nemotron_h_model_dir", "nemotron_3_model_dir"])
def test_real_mode_takes_mamba2_hybrids_over_mooncake_or_nixl(
    model, request, ds_conv_layout
):
    model_dir = request.getfixturevalue(model)
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(model_dir, kv_transfer=mooncake)
    assert config.model_config.is_hybrid
    assert config.cache_config.mamba_cache_mode == "none"
    cached = engine_config(model_dir, kv_transfer=mooncake, enable_prefix_caching=True)
    assert cached.cache_config.mamba_cache_mode == "align"
    hybrid_over_nixl_needs_ds(model_dir, ds_conv_layout)


def test_non_gated_cpu_experts_build_only_in_a_simulation(
    nemotron_3_model_dir, monkeypatch
):
    """The CPU kernels do not run non-gated (relu^2) experts: a CPU engine
    refuses Nemotron-3, a simulation builds them and never runs them."""
    from vllm import LLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(NotImplementedError, match="is_act_and_mul=False"):
        LLM(
            model=nemotron_3_model_dir,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=256,
            kv_cache_memory_bytes=8 * 1024**2,
            enable_prefix_caching=False,
            enforce_eager=True,
            load_format="dummy",
        )


def simulated_build(directory, **kwargs):
    """A simulated engine of the model in `directory`: what its layers were
    built as, and its KV cache specs, after generating two tokens."""
    from vllm import LLM, SamplingParams

    def describe(model):
        return sorted(
            {
                (
                    type(module.quant_method).__name__,
                    type(getattr(module.quant_method, "kernel", None)).__name__,
                    getattr(
                        getattr(module.quant_method, "nvfp4_backend", None),
                        "value",
                        None,
                    ),
                    *sorted(
                        {
                            (name, str(tensor.dtype), tensor.device.type)
                            for name, tensor in module.named_parameters(recurse=False)
                            if name in ("weight", "w13_weight")
                        }
                    ),
                )
                for module in model.modules()
                if getattr(module, "quant_method", None) is not None
            }
        )

    llm = LLM(
        model=str(directory), skip_tokenizer_init=True, dtype="bfloat16",
        max_model_len=256, kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64, enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={"simulate_forward": True, "timing": "linear",
                           "timing_args": {"fixed_us": 100}},
        **kwargs,
    )
    try:
        (out,) = llm.generate(
            [{"prompt_token_ids": [5] * 100}],
            SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(out.outputs[0].token_ids) == 2
        engine = llm.llm_engine.engine_core.engine_core
        runner = engine.model_executor.driver_worker.worker.model_runner
        specs = sorted(
            (name, str(spec.dtype), spec.page_size_bytes)
            for name, spec in runner.get_kv_cache_spec().items()
        )
        (built,) = llm.apply_model(describe)
        return built, specs, str(llm.llm_engine.vllm_config.model_config.dtype)
    finally:
        del llm


def test_nvfp4_checkpoints_build_on_cpu_in_a_simulation_only(
    model_dir, moe_model_dir, tmp_path, monkeypatch
):
    """NVFP4 has no CPU kernel. A simulation builds a ModelOpt NVFP4
    checkpoint as deployed and never runs it: its linear layers hold packed
    FP4 weights (the device-independent emulation kernel standing in), its
    experts those of the backend a GPU deployment tries first. Its KV cache
    is the unquantized model's: weight quantization does not change what a
    transfer moves. Outside a simulation the CPU platform refuses it."""
    from vllm import LLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # The MoE model first: an in-process engine keeps the first one's
    # parallel groups, and a dense model's have no expert-parallel group.
    for name, source in (("moe", moe_model_dir), ("dense", model_dir)):
        config = json.loads(Path(source, "config.json").read_text())
        quantized = tmp_path / f"{name}-nvfp4"
        quantized.mkdir()
        (quantized / "config.json").write_text(json.dumps({
            **config,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
        }))
        built, specs, dtype = simulated_build(quantized)
        _, plain_specs, plain_dtype = simulated_build(source)
        assert (specs, dtype) == (plain_specs, plain_dtype)
        linear = [b for b in built if b[0] == "ModelOptLinearMethod"]
        assert linear and {b[1] for b in linear} == {"EmulationNvFp4LinearKernel"}
        assert {b[3] for b in linear} == {("weight", "torch.uint8", "meta")}
        experts = [b for b in built if b[0] == "ModelOptNvFp4FusedMoE"]
        if name == "moe":
            assert [(b[2], b[3]) for b in experts] == [
                ("FLASHINFER_TRTLLM", ("w13_weight", "torch.uint8", "meta"))
            ]
        else:
            assert not experts
        with pytest.raises(
            (ValueError, NotImplementedError),
            match="NVFP4 linear layer|NvFp4 MoE backend",
        ):
            LLM(model=str(quantized), skip_tokenizer_init=True, dtype="bfloat16",
                max_model_len=256, kv_cache_memory_bytes=8 * 1024**2,
                enable_prefix_caching=False, enforce_eager=True,
                load_format="dummy")


@pytest.mark.parametrize("quantization", ["fp8", "fp8_per_tensor", "fp8_per_block"])
def test_online_quantization_builds_on_cpu_in_a_simulation(
    model_dir, moe_model_dir, monkeypatch, quantization
):
    """`--quantization fp8` quantizes the weights as they load. In a
    simulation they load on the meta device: the layers end as the deployment
    has them (FP8 weights), the CPU kernel's weight packing giving shapes
    without data, and the KV cache is the unquantized model's."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # The MoE model first (see the NVFP4 test).
    for source, methods in (
        (moe_model_dir, {"OnlineLinearMethod", "OnlineMoEMethod"}),
        (model_dir, {"OnlineLinearMethod"}),
    ):
        built, specs, dtype = simulated_build(source, quantization=quantization)
        _, plain_specs, plain_dtype = simulated_build(source)
        assert (specs, dtype) == (plain_specs, plain_dtype)
        online = [b for b in built if b[0].endswith(tuple(methods))]
        assert {m for m in methods if any(b[0].endswith(m) for b in online)} == methods
        weights = {weight for b in online for weight in b[3:]}
        kinds = {(kind, device) for _, kind, device in weights}
        assert ("torch.float8_e4m3fn", "meta") in kinds, built
        # The MoE model's router has 8 rows, fewer than a tile: the CPU
        # kernel keeps such a weight transposed in fp32.
        assert kinds <= {
            ("torch.float8_e4m3fn", "meta"),
            ("torch.float32", "meta"),
        }, built
        assert ("torch.float32", "meta") not in kinds or source == moe_model_dir


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((64, 128), torch.bfloat16),
        ((64, 128), torch.float16),
        ((32, 128), torch.float8_e4m3fn),
        ((32, 64), torch.int8),
        ((32, 64), torch.uint8),
        ((4, 32, 64), torch.bfloat16),
        ((4, 32, 64), torch.float8_e4m3fn),
        # Fewer rows than a tile: transposed to fp32, not packed.
        ((8, 64), torch.bfloat16),
        ((8, 64), torch.float8_e4m3fn),
    ],
)
def test_weight_packing_on_meta_gives_the_cpu_kernels_shape(shape, dtype):
    """Online quantization packs weights for the CPU kernel as they load; on
    the meta device the op answers with the shape and dtype the kernel would
    give, without data."""
    import vllm._custom_ops  # noqa: F401  (loads the op and its meta answer)

    if not hasattr(torch.ops._C, "convert_weight_packed"):
        pytest.skip("needs the CPU weight-packing kernel")
    real = torch.ops._C.convert_weight_packed(torch.zeros(shape, dtype=dtype))
    meta = torch.ops._C.convert_weight_packed(
        torch.empty(shape, dtype=dtype, device="meta")
    )
    assert meta.is_meta
    assert (meta.shape, meta.dtype) == (real.shape, real.dtype)


def test_real_mode_takes_deepseek_v4_over_nixl(v4_model_dir):
    """V4's groups move over NIXL, which keeps each group's own blocks;
    Mooncake cannot pair its block sizes. Tensor parallelism or data-parallel
    attention with expert parallelism (not both), with prefix caching or
    without, the packed FP8 cache."""
    v4 = {
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "block_size": None,
        "kv_cache_memory_bytes": 4 << 30,
    }
    dp_ep = {"data_parallel_size": 2, "enable_expert_parallel": True}
    for connector in ("NixlConnector", "NixlPullConnector"):
        for kwargs in (
            {},
            {"tensor_parallel_size": 2},
            {"tensor_parallel_size": 4},
            dp_ep,
            {"enable_prefix_caching": True},
        ):
            config = engine_config(
                v4_model_dir, kv_transfer={"kv_connector": connector}, **{**v4, **kwargs}
            )
            assert config.model_config.hf_config.model_type == "deepseek_v4"
    for kwargs, message in (
        ({"kv_transfer": {"kv_connector": "MooncakeConnector"}}, "NixlConnector for DeepSeek-V4"),
        ({**dp_ep, "tensor_parallel_size": 2}, "DeepSeek-V4 with tensor parallelism or"),
        ({"data_parallel_size": 2}, "MoE model's data-parallel attention"),
        ({"pipeline_parallel_size": 2}, "tensor parallelism and data-parallel attention at most"),
        ({"kv_cache_dtype": "bfloat16"}, "packed FP8 cache"),
    ):
        with pytest.raises(ValueError, match=message):
            engine_config(v4_model_dir, **{**v4, **kwargs})


# The tiny DeepSeek-V3 with its native MTP layer(s).
def deepseek_v3(mtp_layers):
    return {**DEEPSEEK_V3, "num_nextn_predict_layers": mtp_layers}


def test_real_mode_takes_native_single_module_mtp_over_nixl(
    tmp_path_factory, v4_model_dir, model_dir
):
    """DeepSeek-V3's and DeepSeek-V4's one MTP layer moves with the target's
    KV over NIXL, whose handshake checks the speculative configuration; no
    other speculative decoding, connector or draft."""
    v3 = write_model_dir(tmp_path_factory, "tiny-deepseek-v3", deepseek_v3(1))
    v3_two = write_model_dir(tmp_path_factory, "tiny-deepseek-v3-two", deepseek_v3(2))
    v4 = {
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "block_size": None,
        "kv_cache_memory_bytes": 4 << 30,
    }

    def mtp(k=1, **fields):
        return {"method": "mtp", "num_speculative_tokens": k, **fields}

    for connector in ("NixlConnector", "NixlPullConnector"):
        nixl = {"kv_connector": connector}
        for k in (1, 5):
            config = engine_config(v3, kv_transfer=nixl, speculative_config=mtp(k))
            assert config.speculative_config.draft_model_config.architectures == [
                "DeepSeekMTPModel"
            ]
        config = engine_config(
            v4_model_dir, kv_transfer=nixl, speculative_config=mtp(), **v4
        )
        assert config.speculative_config.draft_model_config.architectures == [
            "DeepSeekV4MTPModel"
        ]
    for model, kwargs, problem in (
        (v3, {"kv_transfer": {"kv_connector": "MooncakeConnector"}}, "not MooncakeConnector"),
        (v3, {"speculative_config": mtp(6)}, "6 speculative tokens"),
        # A checkpoint of two MTP layers, even at depth 1.
        (v3_two, {"speculative_config": mtp()}, "2 MTP layers"),
        (v3, {"speculative_config": mtp(kv_cache_dtype="fp8")}, "draft KV cache dtype fp8"),
        (model_dir, {"speculative_config": {"method": "ngram", "num_speculative_tokens": 2,
                                            "prompt_lookup_max": 2}}, "method ngram"),
    ):
        kwargs.setdefault("speculative_config", mtp())
        with pytest.raises(ValueError, match=problem):
            engine_config(model, **kwargs)


def test_packed_layers_resolve_their_own_specs_and_places():
    """A group of several kinds names each layer's spec; a packed block (each
    block all of a group's layers) places each layer's page at its offset,
    checked over its real bytes only, none overlapping."""
    from vllm.v1.kv_cache_interface import (
        KVCacheGroupSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.simulated.kv_transfer import (
        layer_specs,
        packed_members,
        packed_pages,
    )

    def mla(real, padded):
        return MLAAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=real,
            dtype=torch.uint8,
            page_size_padded=padded,
        )

    specs = {"a": mla(6, 8), "b": mla(10, 12)}
    group = KVCacheGroupSpec(
        ["a", "b"], UniformTypeKVCacheSpecs(block_size=1, kv_cache_specs=specs)
    )
    resolved = layer_specs([group])
    assert resolved == specs
    storage = torch.zeros(4 * 24, dtype=torch.uint8)

    def placed(offset, page):
        return torch.empty(0, dtype=torch.uint8).set_(
            storage.untyped_storage(), offset, (4, page), (24, 1)
        )

    caches = {"a": placed(0, 8), "b": placed(8, 12)}
    assert packed_pages(caches, resolved) == {"a": 6, "b": 10}
    members = packed_members(caches, resolved, ["a", "b"])
    assert [(m["layers"], m["offsets"], m["page_bytes"], m["real_page_bytes"]) for m in members] == [
        (1, [0], 8, 6),
        (1, [8], 12, 10),
    ]
    # Overlapping pages, or a page beyond the block, are refused.
    with pytest.raises(ValueError, match="overlap"):
        packed_members({"a": placed(0, 8), "b": placed(4, 12)}, resolved, ["a", "b"])
    with pytest.raises(ValueError, match="overlap"):
        packed_members({"a": placed(0, 8), "b": placed(16, 12)}, resolved, ["a", "b"])
    # A layout whose block stride is its page is not packed.
    plain = {"a": storage[: 4 * 8].view(4, 8)}
    assert packed_pages(plain, {"a": specs["a"]}) == {"a": 8}
    assert packed_members(plain, {"a": specs["a"]}, ["a"]) == []


def test_simulated_mla_keeps_the_gpu_deployments_settings(mla_model_dir):
    """The CPU platform forces MLA to 16-token blocks, without chunked prefill
    or prefix caching, for its own kernels; a simulation never runs them."""
    config = engine_config(
        mla_model_dir,
        block_size=64,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=128,
    )
    assert config.cache_config.block_size == 64
    assert config.cache_config.enable_prefix_caching
    assert config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.max_num_batched_tokens == 128
    with pytest.raises(ValueError, match="multiple of 16"):
        engine_config(mla_model_dir, block_size=8)


def test_real_mode_allows_tensor_parallelism_and_dp_attention(
    model_dir, moe_model_dir
):
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(model_dir, kv_transfer=mooncake, tensor_parallel_size=2)
    assert config.parallel_config.tensor_parallel_size == 2
    with pytest.raises(ValueError, match="data-parallel attention at most"):
        engine_config(model_dir, kv_transfer=mooncake, pipeline_parallel_size=2)
    # NIXL too: its handshake refuses a peer of another TP size.
    for connector in ("NixlConnector", "NixlPullConnector"):
        config = engine_config(
            model_dir, kv_transfer={"kv_connector": connector}, tensor_parallel_size=2
        )
        assert config.parallel_config.tensor_parallel_size == 2
    # An MoE model's DP ranks with EP: each holds its own requests' KV.
    config = engine_config(
        moe_model_dir, kv_transfer=mooncake, data_parallel_size=2, enable_expert_parallel=True
    )
    assert config.parallel_config.data_parallel_size == 2
    for model, kwargs in (
        (model_dir, {"enable_expert_parallel": True}),  # independent engines
        (moe_model_dir, {}),  # experts not split
    ):
        with pytest.raises(ValueError, match="only as an MoE model's data-parallel"):
            engine_config(model, kv_transfer=mooncake, data_parallel_size=2, **kwargs)
    # Over NIXL too: each DP engine has its own side channel and engine id.
    for connector in ("NixlConnector", "NixlPullConnector"):
        nixl = {"kv_connector": connector}
        config = engine_config(
            moe_model_dir, kv_transfer=nixl, data_parallel_size=2, enable_expert_parallel=True
        )
        assert config.parallel_config.data_parallel_size == 2
        for model, kwargs in ((model_dir, {"enable_expert_parallel": True}), (moe_model_dir, {})):
            with pytest.raises(ValueError, match="only as an MoE model's data-parallel"):
                engine_config(model, kv_transfer=nixl, data_parallel_size=2, **kwargs)
        with pytest.raises(ValueError, match="data-parallel attention at most"):
            engine_config(model_dir, kv_transfer=nixl, pipeline_parallel_size=2)
    with pytest.raises(ValueError, match="one of NixlConnector"):
        engine_config(
            moe_model_dir,
            kv_transfer={"kv_connector": "NixlPushConnector"},
            data_parallel_size=2,
            enable_expert_parallel=True,
        )


@pytest.mark.parametrize("local_tp, remote_tp", [(2, 1), (1, 2), (2, 2), (1, 4), (2, 3)])
def test_real_transfers_pair_tp_sizes_one_dividing_the_other(model_dir, local_tp, remote_tp):
    """What NIXL's handshake (and Mooncake's) checks first: real transfers of
    plain attention (the tiny Llama: 2 KV heads) pair TP sizes one dividing
    the other, heads re-sharded; NIXL's decode refuses a larger producer
    whose heads are replicated (TP 4 on 2 heads), as NIXL does. Ordinary
    execution keeps heterogeneous TP."""
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
    )

    simulated = engine_config(model_dir, tensor_parallel_size=local_tp)
    mismatch = simulated_tp_mismatch(simulated, local_tp, remote_tp)
    nixl = simulated_tp_mismatch(simulated, local_tp, remote_tp, nixl_consumer=True)
    if (local_tp, remote_tp) == (2, 3):
        assert f"one dividing the other, not local TP {local_tp} and remote TP {remote_tp}" in mismatch
    else:
        assert mismatch is None
        if (local_tp, remote_tp) == (1, 4):
            assert "replicated" in nixl
        else:
            assert nixl is None
    assert simulated_tp_mismatch(VllmConfig(), local_tp, remote_tp) is None


# The tiny models' 2 attention heads bound the local TP size at 2.
@pytest.mark.parametrize("local_tp, remote_tp", [(2, 1), (1, 2), (1, 4), (2, 4)])
def test_real_transfers_pair_plain_mla_at_unequal_tp(
    mla_model_dir, tmp_path_factory, local_tp, remote_tp
):
    """Plain MLA (the tiny DeepSeek-V3): one latent, the same on every rank,
    pairs unequal sizes either way over either connector (NIXL exempts MLA
    from its replicated-producer refusal). Sparse MLA (the tiny
    DeepSeek-V3.2: an indexer cache beside the latent) waits for several
    groups' checkpoint."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        plain_mla,
        simulated_tp_mismatch,
    )

    mla = engine_config(mla_model_dir, tensor_parallel_size=local_tp)
    assert plain_mla(mla.model_config)
    assert simulated_tp_mismatch(mla, local_tp, remote_tp) is None
    assert simulated_tp_mismatch(mla, local_tp, remote_tp, nixl_consumer=True) is None
    v32 = write_model_dir(tmp_path_factory, "tiny-deepseek-v3.2-tp", DEEPSEEK_V32)
    sparse = engine_config(v32, dtype="bfloat16", block_size=64, tensor_parallel_size=local_tp)
    assert sparse.model_config.use_mla and not plain_mla(sparse.model_config)
    assert "plain attention or plain MLA only" in simulated_tp_mismatch(sparse, local_tp, remote_tp)

# The tiny gpt-oss's 2 KV heads bound the local TP size at 2.
@pytest.mark.parametrize("local_tp, remote_tp", [(2, 1), (1, 2), (1, 4), (2, 4)])
def test_real_transfers_pair_sliding_window_models_at_unequal_tp(
    gpt_oss_model_dir, local_tp, remote_tp
):
    """The tiny gpt-oss (sliding-window and full groups, 2 KV heads each)
    pairs unequal sizes either way over either connector; each layer's own
    total heads (the model's) give its heads, so both groups are head-v1."""
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
    )

    config = engine_config(gpt_oss_model_dir, tensor_parallel_size=local_tp)
    assert config.model_config.get_sliding_window() == 32
    assert simulated_tp_mismatch(config, local_tp, remote_tp) is None
    # NIXL's decode refuses a larger producer whose 2 heads are replicated.
    refused = remote_tp > local_tp and remote_tp > 2
    assert (simulated_tp_mismatch(config, local_tp, remote_tp, nixl_consumer=True) is not None) == refused


def test_nixl_ranks_bind_their_own_address_on_the_mooncake_backend(
    monkeypatch, model_dir
):
    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import rank_kv
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import tent_config
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        _bind_rank_kv_address,
    )

    def config(backends, addresses=None):
        extra = {"backends": backends}
        if addresses is not None:
            extra["rank_addresses"] = addresses
        return engine_config(
            model_dir,
            kv_transfer={"kv_connector_extra_config": extra},
            tensor_parallel_size=2,
        )

    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(rank_kv, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setenv(tent_config.ENV, json.dumps({"rpc_server_hostname": "10.100.0.1"}))
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "10.100.0.1")
    addresses = ["10.100.0.1", "10.101.0.1"]

    _bind_rank_kv_address(config(["Mooncake"]), ["Mooncake"])  # the host's
    assert os.environ["NIXL_MOONCAKE_IP_ADDR"] == "10.100.0.1"
    _bind_rank_kv_address(config(["Mooncake"], addresses), ["Mooncake"])
    assert os.environ["NIXL_MOONCAKE_IP_ADDR"] == "10.101.0.1"
    conf = json.loads(os.environ[tent_config.ENV])
    assert tent_config.lookup(conf, "rpc_server_hostname") == (True, "10.101.0.1")
    assert tent_config.lookup(conf, "transports/hp_tcp/bind_address") == (
        True,
        "10.101.0.1",
    )
    for backends in (["Mooncake", "UCX"],):
        with pytest.raises(ValueError, match="Mooncake backend only"):
            _bind_rank_kv_address(config(backends, addresses), backends)

    # Simulated transfers (no real KV) and ordinary execution are left alone.
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "10.100.0.1")
    before = os.environ[tent_config.ENV]
    simulated = engine_config(
        model_dir,
        simulation={"kv_transfer_mode": "simulated"},
        kv_transfer={"kv_connector_extra_config": {"rank_addresses": addresses}},
        tensor_parallel_size=2,
    )
    ordinary = VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_connector="NixlConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={"rank_addresses": addresses},
        )
    )
    for vllm_config in (simulated, ordinary):
        _bind_rank_kv_address(vllm_config, ["UCX"])
        assert os.environ["NIXL_MOONCAKE_IP_ADDR"] == "10.100.0.1"
        assert os.environ[tent_config.ENV] == before


@pytest.mark.parametrize(
    "mode, dp_index, dp_size, tp_rank, name, steps",
    [
        ("real", 0, 1, 0, "steps.jsonl", True),
        ("real", 0, 1, 1, "steps.tp1.jsonl", True),
        ("simulated", 0, 1, 1, None, True),  # the output rank's log only
        # MoE DP: DP rank 0 logs the joint steps, the others their own KV.
        ("real", 0, 2, 0, "steps.jsonl", True),
        ("real", 1, 2, 0, "steps.dp1.jsonl", False),
        ("real", 1, 2, 1, "steps.dp1.tp1.jsonl", False),
        ("simulated", 1, 2, 0, None, False),
        # Dense DP: independent engines, each logging its own steps.
        ("simulated", 2, 1, 0, "steps.dp2.jsonl", True),
    ],
)
def test_each_rank_logs_its_own_transfers(
    monkeypatch, tmp_path, mode, dp_index, dp_size, tp_rank, name, steps
):
    from vllm.distributed import parallel_state
    from vllm.v1.worker.simulated.timing import logs_steps, step_log_path

    group = SimpleNamespace(rank_in_group=tp_rank, is_last_rank=True)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    config = SimpleNamespace(
        simulation_config=SimpleNamespace(
            log_path=str(tmp_path / "steps.jsonl"), kv_transfer_mode=mode
        ),
        # Dense DP engines run as DP=1, keeping their index.
        parallel_config=SimpleNamespace(
            data_parallel_index=dp_index,
            data_parallel_size=dp_size,
            data_parallel_rank=dp_index if dp_size > 1 else 0,
        ),
    )
    assert step_log_path(config) == (str(tmp_path / name) if name else None)
    assert logs_steps(config) is steps


def test_real_mode_needs_a_connector(model_dir):
    with pytest.raises(ValueError, match="needs a kv_transfer_config"):
        engine_config(model_dir, kv_transfer_config=None)


def test_real_mode_needs_simulated_execution(model_dir):
    with pytest.raises(ValueError, match="needs simulate_forward"):
        engine_config(model_dir, simulation={"simulate_forward": False})


def test_real_mode_rejects_an_explicit_offloading_connector(model_dir):
    """Regression: the simulated-only OffloadingConnector branch returned
    before the real-mode checks, accepting the virtual clock."""
    with pytest.raises(ValueError, match="one of NixlConnector.*the real clock"):
        engine_config(
            model_dir,
            simulation={"clock": "virtual"},
            kv_transfer={"kv_connector": "OffloadingConnector", "kv_role": "kv_both"},
        )


def test_real_mode_rejects_offloading_that_would_replace_nixl(model_dir):
    """Regression: kv_offloading_size passed with the NixlConnector, then
    VllmConfig replaced it with an offloading connector (kv_role kv_both)."""
    with pytest.raises(ValueError, match="no KV offloading"):
        engine_config(model_dir, kv_offloading_size=1)


def stand_in_metadata(loads=()):
    """Metadata of the stand-in connectors, sent to (spawned) workers: a
    module-level class, so it pickles."""
    return StandInMetadata(list(loads))


def _metadata_base():
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

    return KVConnectorMetadata


class StandInMetadata(_metadata_base()):  # type: ignore[misc]
    def __init__(self, loads):
        self.loads = loads


class RecordingConnector:
    """Stands in for the NIXL connector: keeps the KV tensors registered with
    it and, with several KV cache groups (`hma`), each finished request's
    blocks."""

    registered: list[dict] = []
    finished: list[tuple[str, tuple[list[int], ...]]] = []

    @classmethod
    def make(cls, hma=False):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            SupportsHMA,
        )

        class Connector(KVConnectorBase_V1, *((SupportsHMA,) if hma else ())):
            def request_finished_all_groups(self, request, block_ids):
                cls.finished.append((request.request_id, block_ids))
                return False, None

            def register_kv_caches(self, kv_caches):
                cls.registered.append(dict(kv_caches))

            def start_load_kv(self, forward_context, **kwargs):
                pass

            def wait_for_layer_load(self, layer_name):
                pass

            def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
                pass

            def wait_for_save(self):
                pass

            def get_num_new_matched_tokens(self, request, num_computed_tokens):
                return 0, False

            def update_state_after_alloc(self, request, blocks, num_external_tokens):
                pass

            def build_connector_meta(self, scheduler_output):
                return stand_in_metadata()

        return Connector


def expected_pages(llm):
    """The pattern a single-rank engine's layer page holds: each attention
    layer's KV heads, as the model runner maps them (head_map), else the
    rank's."""
    from vllm.v1.worker.simulated.kv_pattern import head_map, layer_pattern
    from vllm.v1.worker.simulated.kv_transfer import layer_specs

    engine = llm.llm_engine
    config = engine.engine_core.engine_core.scheduler.kv_cache_config
    heads = head_map(
        layer_specs(config.kv_cache_groups),
        engine.model_config.get_total_num_kv_heads(),
        0,
        1,
    )
    assert heads, "no layer keeps its heads' pattern"

    def page(req_id, layer, logical, size):
        return layer_pattern(req_id, layer, logical, size, 0, None, heads)

    return page


def test_producer_writes_the_pattern_into_its_kv_tensors(any_model_dir, monkeypatch):
    """With kv_pattern_fill, a prefill engine's real KV tensors hold each
    request's pattern in the blocks the request was given."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.worker.simulated.kv_pattern import block_views

    connector = RecordingConnector.make()
    RecordingConnector.registered.clear()
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=any_model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=32,
        kv_cache_memory_bytes=8 * 1024**2,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 70}],
            SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        caches = RecordingConnector.registered[0]
        num_blocks = next(iter(caches.values())).shape[0]
        views = block_views(caches, num_blocks)
        page = expected_pages(llm)
        # 70 tokens span logical blocks 0..2; each holds its pattern somewhere.
        for layer, view in views.items():
            for logical in range(3):
                want = page(output.request_id, layer, logical, view.shape[1])
                assert any(torch.equal(view[b], want) for b in range(num_blocks)), (
                    layer,
                    logical,
                )
    finally:
        del llm


def test_a_producer_fills_each_groups_blocks_through_a_chunked_window(
    gpt_oss_model_dir, monkeypatch, tmp_path
):
    """gpt-oss's sliding-window and full groups share memory. A prompt
    prefilled in chunks, in a pool of few blocks, leaves the window's head
    null and the request reuses blocks it freed: at the end each block it
    holds carries its group's pattern at its place, and the step log's
    allocations by group rebuild its blocks, reuse included."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.worker.simulated.kv_pattern import block_views

    connector = RecordingConnector.make(hma=True)
    RecordingConnector.registered.clear()
    RecordingConnector.finished.clear()
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    log = tmp_path / "steps.jsonl"
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=gpt_oss_model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        block_size=16,
        # 28 blocks of 2 x 2 KiB (a layer of each group).
        kv_cache_memory_bytes=28 * 4096,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        llm.generate(
            [{"prompt_token_ids": [5] * 200}],
            SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        config = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
        kinds = [type(g.kv_cache_spec).__name__ for g in config.kv_cache_groups]
        assert sorted(kinds) == ["FullAttentionSpec", "SlidingWindowSpec"]
        ((req_id, block_ids),) = RecordingConnector.finished
        caches = RecordingConnector.registered[0]
        views = block_views(caches, config.num_blocks)
        page = expected_pages(llm)
        for group, held in zip(config.kv_cache_groups, block_ids):
            # 200 tokens: 13 blocks, the window's first ones null.
            assert len(held) == 13
            nulls = held.count(0)
            if kinds[config.kv_cache_groups.index(group)] == "SlidingWindowSpec":
                assert nulls > 0 and set(held[:nulls]) == {0}
                window = config.kv_cache_groups.index(group)
            else:
                assert nulls == 0
            for logical, block in enumerate(held):
                if block == 0:
                    continue
                for layer in group.layer_names:
                    view = views[layer]
                    want = page(req_id, layer, logical, view.shape[1])
                    assert torch.equal(view[block], want), (layer, logical, block)
        # Each group's blocks by place, as the step log gave them.
        places: list[dict[int, int]] = [{} for _ in block_ids]
        for record in log_records(log):
            events = record.get("kv_events") or {}
            for entry in events.get("allocated_groups", {}).get(req_id, []):
                for offset, block in enumerate(entry["blocks"]):
                    places[entry["group"]][entry["first"] + offset] = block
        for group, held in enumerate(block_ids):
            for logical, block in enumerate(held):
                if block:
                    assert places[group][logical] == block, (group, logical)
        # The window's freed blocks were given to it again, at later places.
        given = [block for block in places[window].values() if block]
        assert len(given) > len(set(given))
    finally:
        del llm


@pytest.mark.parametrize(
    "model, mamba_type",
    [
        ("qwen3_5_model_dir", "GDN_ATTN"),
        ("nemotron_h_model_dir", "MAMBA2"),
        ("nemotron_3_model_dir", "MAMBA2"),
    ],
    ids=["qwen3.5", "nemotron-h", "nemotron-3"],
)
def test_a_hybrid_producer_fills_each_groups_pages_padding_included(
    model, mamba_type, request, monkeypatch, tmp_path
):
    """Tiny Qwen3.5 (three delta-net groups; a vision-language wrapper
    served as its language model), Nemotron-H
    (Mamba2 groups, an MLP-only layer without KV) and Nemotron-3 (Mamba2
    groups, MoE layers without KV, their non-gated experts built as such):
    recurrent groups and an attention group in one allocation, the
    platform's attention block covering a state's page. A chunked prompt:
    each attention block holds its place's pattern, each recurrent layer's
    single state block (place 0) its pattern over the whole page, padding
    included; the layout record describes the padding."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.worker.simulated.kv_pattern import block_views

    connector = RecordingConnector.make(hma=True)
    RecordingConnector.registered.clear()
    RecordingConnector.finished.clear()
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=request.getfixturevalue(model),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        language_model_only=model == "qwen3_5_model_dir",
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        llm.generate(
            [{"prompt_token_ids": [5] * 200}],
            SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        config = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
        groups = config.kv_cache_groups
        kinds = [type(g.kv_cache_spec).__name__ for g in groups]
        assert sorted(set(kinds)) == ["FullAttentionSpec", "MambaSpec"]
        pages = {
            name: g.kv_cache_spec.page_size_bytes for g in groups for name in g.layer_names
        }
        views = block_views(RecordingConnector.registered[0], config.num_blocks, pages)
        page = expected_pages(llm)
        ((req_id, block_ids),) = RecordingConnector.finished
        for group, held in zip(groups, block_ids):
            spec = group.kv_cache_spec
            if type(spec).__name__ == "MambaSpec":
                # The state, one block, padded to the attention page.
                assert len(held) == 1 and spec.real_page_size_bytes < spec.page_size_bytes
            for logical, block in enumerate(held):
                for layer in group.layer_names:
                    want = page(req_id, layer, logical, spec.page_size_bytes)
                    assert torch.equal(views[layer][block], want), (layer, logical)
        (layout,) = [r["kv_layout"] for r in log_records(log) if "kv_layout" in r]
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
            for t in RecordingConnector.registered[0].values()
        }
        assert layout["bytes"] == sum(storages.values())
        states = [g for g in layout["groups"] if g["kind"] == "MambaSpec"]
        assert {g["mamba_type"] for g in states} == {mamba_type}
        assert all(
            g["real_page_bytes"] < g["block_bytes"] == g["block_stride"] for g in states
        )
        if model == "qwen3_5_model_dir":
            # Text only: the vision tower is not built.
            (visual,) = llm.apply_model(lambda m: type(m.visual).__name__)
            assert visual == "StageMissingLayer"
        if model == "nemotron_3_model_dir":
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation

            (experts,) = llm.apply_model(
                lambda m: {
                    name: (e.activation, tuple(e.w13_weight.shape))
                    for name, e in m.named_modules()
                    if hasattr(e, "w13_weight")
                }
            )
            # Layers 1 and 4, their up projection alone (not gate and up).
            assert len(experts) == 2
            assert set(experts.values()) == {(MoEActivation.RELU2_NO_MUL, (4, 32, 64))}
    finally:
        del llm


@pytest.mark.parametrize(
    "family, kwargs",
    [
        ("NEMOTRON_3", {}),
        ("QWEN3_5_MOE", {"language_model_only": True}),
        ("KIMI_K3", {}),
    ],
    ids=["nemotron-3", "qwen3.5", "kimi-k3"],
)
def test_a_hybrids_cached_states_are_reused_as_written(
    family, kwargs, tmp_path, monkeypatch, ds_conv_layout
):
    """With prefix caching a hybrid keeps a state a block (align mode) and a
    later prompt reuses the state cached at a block's end. The native step
    copies states between blocks around the forward; this runner computes
    none, so it must leave a cached state as the pattern wrote it: a prompt
    sharing a prefix hits it, every state byte compared with its content's."""
    from vllm import LLM, SamplingParams
    from vllm.v1.kv_cache_interface import MambaSpec

    config = globals()[family]
    # Prompts of several blocks, whatever block the platform picks.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    (tmp_path / "config.json").write_text(json.dumps(config))
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=4096,
        kv_cache_memory_bytes=64 * 1024**2,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
        **kwargs,
    )
    try:
        engine = llm.llm_engine.engine_core.engine_core
        groups = engine.scheduler.kv_cache_config.kv_cache_groups
        states = [
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        assert states
        assert {groups[i].kv_cache_spec.mamba_cache_mode for i in states} == {"align"}
        (block,) = {g.kv_cache_spec.block_size for g in groups}
        params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False)
        shared = [5] * (2 * block)
        for tail in ([6] * 7, [7] * 5, [8] * (block + 3)):
            llm.generate([{"prompt_token_ids": shared + tail}], params, use_tqdm=False)
    finally:
        del llm
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    checks = [r["kv_check"] for r in records if "kv_check" in r]
    assert all(check["ok"] for check in checks), [c for c in checks if not c["ok"]][:1]
    hits = [check for check in checks if check.get("kind") == "hit"]
    # The second and third prompts each reuse a cached state of every group.
    assert len(hits) == 2
    for hit in hits:
        compared = {g["group"]: g for g in hit["groups"]}
        assert set(states) <= set(compared)
        assert all(compared[i]["bytes"] > 0 for i in states)


# A tiny Nemotron-3 whose state groups differ: 3 Mamba2 layers in groups of
# two slots (2 and 1), so one group holds no layer in the second slot.
NEMOTRON_3_UNEQUAL = {
    **NEMOTRON_3,
    "num_hidden_layers": 7,
    "hybrid_override_pattern": "M*EM*EM",
}


@pytest.mark.parametrize(
    "config, kwargs, transfer_only",
    [
        (NEMOTRON_3_UNEQUAL, {}, True),
        (NEMOTRON_3, {}, False),
        (QWEN3_5_MOE, {"language_model_only": True}, False),
        (KIMI_K3, {}, False),
    ],
    ids=["nemotron-3-unequal", "nemotron-3", "qwen3.5", "kimi-k3"],
)
def test_nixl_checks_cover_a_hybrids_native_selection(
    config, kwargs, transfer_only, tmp_path, monkeypatch, ds_conv_layout
):
    """Over NIXL a hybrid's blocks move as the native worker selects them:
    deduplicated regions (a slot of each group's layers), every region of a
    group's kind for each of its blocks (one holding none of its layers
    included: transfer-only), a state as its conv sub-projections then its
    SSM state, not its page's padding. The filler and the checker cover
    exactly that selection, in its order, and catch a changed conv or SSM
    byte, an unmoved transfer-only span, a written or copied padding byte."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.simulated.kv_pattern import (
        POISON,
        SENTINEL,
        NativeChecker,
        NativeFiller,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    (tmp_path / "config.json").write_text(json.dumps(config))
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
        **kwargs,
    )
    try:
        # The runtime path: a chunked prompt filled through the selection.
        llm.generate(
            [{"prompt_token_ids": [5] * 200}],
            SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        engine = llm.llm_engine.engine_core.engine_core
        runner = engine.model_executor.driver_worker.worker.model_runner
        groups = engine.scheduler.kv_cache_config.kv_cache_groups
        worker = get_kv_transfer_group().connector_worker
        filler = runner._kv_filler
        assert isinstance(filler, NativeFiller)
        s = filler.selection

        # One deduplicated region a slot, each state region prepared as
        # three conv sub-projections and the SSM state.
        slots = max(len(group.layer_names) for group in groups)
        bases = worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]
        assert worker.num_regions == len(bases) == slots
        state_regions = sorted(set(worker._ssm_region_indices))
        for region in state_regions:
            assert sum(row.state and row.region == region for row in s.rows) == 4

        # Each group's spans a block, from its own spec: attention its page in
        # every region; a state, in every state region, contiguous conv
        # spans then the SSM state, together its real page.
        for index, group in enumerate(groups):
            spec = group.kv_cache_spec
            spans = [
                (s.rows[row].region, s.rows[row].offset, s.rows[row].view.shape[1])
                for row, _, _ in s.select(index, [1])
            ]
            if isinstance(spec, MambaSpec):
                conv = spec.shapes[0]
                conv_bytes = math.prod(conv) * spec.dtypes[0].itemsize
                ssm_bytes = spec.real_page_size_bytes - conv_bytes
                want = []
                for region in state_regions:
                    mine = [span for span in spans if span[0] == region]
                    assert [span[1] for span in mine[:3]] == [
                        0, mine[0][2], mine[0][2] + mine[1][2]
                    ]
                    assert sum(span[2] for span in mine[:3]) == conv_bytes
                    want += mine[:3] + [(region, conv_bytes, ssm_bytes)]
                assert spans == want
            else:
                assert spans == [(r, 0, spec.page_size_bytes) for r in range(slots)]
            # A region holds the group's layers or is transfer-only.
            assert len(s.members[index]) == len(group.layer_names)
        has_transfer_only = any(
            {s.rows[row].region for row, _, _ in s.select(index, [1])}
            - s.members[index]
            for index in range(len(groups))
        )
        assert has_transfer_only == transfer_only

        # A request's ordered, paired descriptors: the native ids over the
        # native local descriptors, block by block, duplicates included.
        free = iter(range(2, 2 + 4 * len(groups)))
        source = [[next(free), next(free)] for _ in groups]
        target = [[next(free), next(free)] for _ in groups]
        rows = worker.src_blocks_data
        for index, blocks in enumerate(source):
            lists = [[] for _ in groups]
            lists[index] = blocks
            native = [
                (int(rows[r][0] + (d - s.starts[r]) * rows[r][3]), int(rows[r][1]))
                for d in worker._compute_desc_ids(lists, worker.num_blocks, None, 1)
                for r in [int(np.searchsorted(s.starts, d, side="right") - 1)]
            ]
            ours = [
                (s.rows[row].view[block].data_ptr(), s.rows[row].view.shape[1])
                for row, block, _ in s.select(index, blocks)
            ]
            assert ours == native
            assert len(ours) == len(blocks) * len(s.select(index, [1]))

        filler._extend("cmpl-a-0", source, replace=True)
        for index, blocks in enumerate(source):
            for region in s.state_regions(s.select(index, blocks)) & set(s.padding):
                assert bool((s.padding[region].view[blocks] == POISON).all())

        # The runner's own checker holds the filler's heads (an eligible hybrid's
        # attention groups are head patterns, its states state-v1).
        checker = NativeChecker(s, heads=filler.heads)
        posted_ids = {}
        if s.layout is not None:
            # An eligible hybrid is checked by the per-READ coverage rule
            # (tests of its own); this single-worker stand-in has no remote
            # engine, so it keeps asserting the posted descriptors are the
            # checked ones, never a state's scratch slot.
            def posted_only(expected, records):
                want = s.order({g.group: [b for b in g.block_ids if b != 0]
                                for g in expected.groups})
                return {"problems": [] if posted_ids.get("ids") == want else ["posted descriptors differ"],
                        "reads": []}
            checker._coverage = posted_only

        def post(blocks, extra=()):
            """The local descriptors NIXL posts for a read into `blocks`
            (`extra`: of other blocks besides, e.g. a state's scratch slot)."""
            checker.posted["cmpl-a-0"] = [
                int(d) for d in worker._compute_desc_ids(blocks, worker.num_blocks, None, 1)
            ] + list(extra)
            posted_ids["ids"] = list(checker.posted["cmpl-a-0"])

        def read(corrupt=None, whole_page=False, skip=None, extra=(), swap=False):
            """The read NIXL would post: each selected span from the source
            block to the target block (`swap`: a state's from the source's
            other slot), then a corruption; its check."""
            checker.expect("cmpl-b-0", "cmpl-a-0", [0] * len(groups), target)
            post(target, extra)
            for index in range(len(groups)):
                pairs = zip(
                    s.select(index, source[index]), s.select(index, target[index])
                )
                for (row, src, place), (dst_row, dst, dst_place) in pairs:
                    assert (row, place) == (dst_row, dst_place)
                    if skip is not None and skip(index, row):
                        continue
                    if swap and s.rows[row].state:
                        src = source[index][1 - place]
                    s.rows[row].view[dst] = s.rows[row].view[src]
                if whole_page:
                    for region in s.state_regions(s.select(index, source[index])):
                        if region in s.padding:
                            pad = s.padding[region].view
                            pad[target[index]] = pad[source[index]]
            if corrupt is not None:
                corrupt()
            return checker._compare("cmpl-b-0", checker.pending.pop("cmpl-b-0"))

        result = read()
        assert result["ok"], result
        for index, g in enumerate(result["groups"]):
            per_block = sum(
                s.rows[row].view.shape[1] for row, _, _ in s.select(index, [1])
            )
            assert g["bytes"] == 2 * per_block
            assert g["descriptors"] == 2 * len(s.select(index, [1]))
            assert g["layers"] == len(groups[index].layer_names)
            if isinstance(groups[index].kv_cache_spec, MambaSpec):
                assert g["padding_bytes"] > 0
        assert (sum(g["transfer_only_bytes"] for g in result["groups"]) > 0) == (
            transfer_only
        )

        state = next(
            i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)
        )
        conv_row = next(row for row, _, _ in s.select(state, [1]) if s.rows[row].state)
        ssm_row = [row for row, _, _ in s.select(state, [1]) if s.rows[row].state][3]
        padded = min(s.state_regions(s.select(state, [1])) & set(s.padding))

        def flip(row):
            def corrupt():
                s.rows[row].view[target[state][0], 0] ^= 1
            return corrupt

        def pad():
            s.padding[padded].view[target[state][1], 0] = 7

        assert not read(flip(conv_row))["ok"]
        assert not read(flip(ssm_row))["ok"]
        padding = read(pad)
        assert not padding["ok"] and sum(
            g["padding_mismatched_bytes"] for g in padding["groups"]
        ) == 1
        assert not read(whole_page=True)["ok"]  # the padding was copied
        # A state's other slot copied (a scratch slot where the running one
        # belongs), or a READ writing a block beside the checked ones.
        assert not read(swap=True)["ok"]
        spare = next(
            int(d) for d in worker._compute_desc_ids(
                [[worker.num_blocks - 1] if i == state else [] for i in range(len(groups))],
                worker.num_blocks, None, 1,
            )
        )
        stray = read(extra=[spare])
        assert stray["mismatched_bytes"] == 0 and not stray["posted_matches_checked"]
        assert not stray["ok"]
        if transfer_only:
            unmoved = read(
                skip=lambda index, row: s.rows[row].region not in s.members[index]
            )
            assert not unmoved["ok"]
        # Two reads pending into disjoint blocks of the same storages: the
        # second's initialization and a producer's fill of other blocks leave
        # the first's landed payload and its padding untouched.
        more = iter(range(2 + 4 * len(groups), worker.num_blocks))
        first, second, fresh = (
            [[next(more), next(more)] for _ in groups] for _ in range(3)
        )

        def land(blocks):
            for index in range(len(groups)):
                pairs = zip(s.select(index, source[index]), s.select(index, blocks[index]))
                for (row, src, _), (_, dst, _) in pairs:
                    s.rows[row].view[dst] = s.rows[row].view[src]

        def snapshot(blocks):
            spans = [
                s.rows[row].view[block].clone()
                for index in range(len(groups))
                for row, block, _ in s.select(index, blocks[index])
            ]
            return spans + [
                s.padding[region].view[blocks[index]].clone()
                for index in range(len(groups))
                for region in sorted(
                    s.state_regions(s.select(index, blocks[index])) & set(s.padding)
                )
            ]

        def complete(req_id, blocks):
            post(blocks)
            return checker._compare(req_id, checker.pending.pop(req_id))

        checker.expect("cmpl-d-0", "cmpl-a-0", [0] * len(groups), first)
        land(first)
        before = snapshot(first)
        checker.expect("cmpl-e-0", "cmpl-a-0", [0] * len(groups), second)
        filler._extend("cmpl-f-0", fresh, replace=True)
        assert {"cmpl-d-0", "cmpl-e-0"} <= set(checker.pending)
        assert all(torch.equal(a, b) for a, b in zip(before, snapshot(first)))
        land(second)
        assert complete("cmpl-d-0", first)["ok"]
        assert complete("cmpl-e-0", second)["ok"]
        # Completed and released, its blocks reused by another read take the
        # sentinel again, and that read is checked on its own.
        checker.expect("cmpl-g-0", "cmpl-a-0", [0] * len(groups), first)
        assert all(
            bool((s.rows[row].view[block] == SENTINEL).all())
            for index in range(len(groups))
            for row, block, _ in s.select(index, first[index])
        )
        assert not complete("cmpl-g-0", first)["ok"]  # nothing landed this time
        checker.expect("cmpl-g-0", "cmpl-a-0", [0] * len(groups), first)
        land(first)
        assert complete("cmpl-g-0", first)["ok"]
        assert not checker.pending
        assert read()["ok"]
        # A remote of another geometry is refused, not assumed equal.
        worker = s.worker
        for name, value in (
            ("tp_mappings", {"remote": SimpleNamespace(
                all_source_ranks=(0, 1), local_consumers=1)}),
            ("src_xfer_handles_by_block_size", {
                **worker.src_xfer_handles_by_block_size, worker.block_size + 1: object()}),
            ("src_xfer_handles_by_tp_ratio", {(-2, worker.block_size): []}),
        ):
            saved = getattr(worker, name)
            setattr(worker, name, value)
            try:
                if checker.heads is not None and name != "src_xfer_handles_by_block_size":
                    # Head and state patterns are global: another TP size's
                    # sources are checked (their ownership by the coverage
                    # rule), only another block size is refused.
                    assert read()["ok"]
                else:
                    with pytest.raises(ValueError, match="equal (TP and )?block size"):
                        read()
            finally:
                setattr(worker, name, saved)
        assert read()["ok"]
    finally:
        del llm


@pytest.mark.parametrize(
    "flags, role", [({}, "kv_producer"), ({"kv_pattern_check": True}, "kv_consumer")], ids=["checks-off", "checks-on"]
)
def test_a_hybrid_peer_of_another_block_size_is_refused_whether_or_not_bytes_are_checked(
    flags, role, tmp_path, monkeypatch, ds_conv_layout
):
    """The block-size equality the CP4 scope rests on is the peer validator's,
    not the pattern checker's (absent from performance runs): an eligible
    hybrid refuses a peer of a larger or a smaller block size at the handshake,
    before NIXL's own (not-mamba) assertion could, in both directions."""
    from vllm import LLM
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    (tmp_path / "config.json").write_text(json.dumps(NEMOTRON_3))
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            **({"log_path": str(tmp_path / "steps")} if flags else {}),
            **flags,
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": role,
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
    )
    try:
        worker = get_kv_transfer_group().connector_worker
        for remote in (worker.block_size * 2, worker.block_size // 2):
            peer = SimpleNamespace(
                ssm_sizes=tuple(worker._mamba_ssm_size), block_lens=list(worker.block_len_per_layer),
                block_size=remote,
            )
            with pytest.raises(ValueError, match="unequal block sizes"):
                worker.add_remote_agent(peer, 0, worker.world_size)
    finally:
        del llm


# One side of a NIXL pair, in a process of its own (the connector is a
# process-wide singleton): the producer serves one prompt and waits; the
# consumer reads it, recording the native descriptor lists and transfers.
NIXL_PEER = r"""
import json, os, sys, time
import numpy as np

def descriptors(rows, ids):
    rows = np.asarray(rows, dtype=np.int64)
    starts = np.cumsum(rows[:, 4]) - rows[:, 4]
    out = []
    for d in ids:
        r = int(np.searchsorted(starts, d, side="right") - 1)
        out.append([int(rows[r, 0] + (d - starts[r]) * rows[r, 3]), int(rows[r, 1])])
    return out

def main():
    role, model, kwargs, out, memory = sys.argv[1:6]
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    producer = role == "producer"
    simulation = {"simulate_forward": True, "kv_transfer_mode": "real",
                  "timing": "linear", "timing_args": {"fixed_us": 100}}
    kwargs = json.loads(kwargs)
    check = kwargs.pop("check", True)
    simulation.update({"kv_pattern_fill": True} if producer
                      else {"kv_pattern_check": True, "log_path": out + ".steps"}
                      if check else {})
    speculative = kwargs.get("speculative_config")
    if speculative:
        simulation["spec_acceptance_rates"] = [0.7] * speculative["num_speculative_tokens"]
    llm = LLM(model=model, skip_tokenizer_init=True, dtype="bfloat16",
              max_model_len=256, kv_cache_memory_bytes=int(memory),
              max_num_batched_tokens=64, enable_prefix_caching=False,
              enforce_eager=True, simulation_config=simulation,
              kv_transfer_config={
                  "kv_connector": "NixlConnector",
                  "kv_role": "kv_producer" if producer else "kv_consumer",
                  "kv_buffer_device": "cpu",
                  "kv_connector_extra_config": {"backends": ["Mooncake"]}},
              **kwargs)
    engine = llm.llm_engine.engine_core.engine_core
    runner = engine.model_executor.driver_worker.worker.model_runner
    worker = get_kv_transfer_group().connector_worker
    mine = {"bases": [int(b) for b in worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank]],
            "strides": [int(b) for b in worker.block_stride_per_layer],
            "lens": [int(b) for b in worker.block_len_per_layer],
            "num_blocks": int(worker.num_blocks)}
    prompt = [{"prompt_token_ids": [5] * 200}]
    if producer:
        params = SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False,
                                extra_args={"kv_transfer_params": {"do_remote_decode": True}})
        (output,) = llm.generate(prompt, params, use_tqdm=False)
        with open(out + ".tmp", "w") as f:
            json.dump({**mine, "params": output.kv_transfer_params}, f)
        os.rename(out + ".tmp", out)
        while not os.path.exists(out + ".done"):
            time.sleep(0.2)
        return
    with open(sys.argv[6]) as f:
        remote = json.load(f)
    wrapper, checker = worker.nixl_wrapper, runner._kv_checker
    dlists, transfers, reads = {}, [], []
    prep, make = wrapper.prep_xfer_dlist, wrapper.make_prepped_xfer
    before = checker.before_step if checker is not None else None

    def prep_xfer_dlist(agent, blocks, *args, **kw):
        handle = prep(agent, blocks, *args, **kw)
        dlists[handle] = np.asarray(blocks).tolist()
        return handle

    def make_prepped_xfer(op, local, local_ids, remote_handle, remote_ids, *args, **kw):
        transfers.append([op, local, [int(i) for i in local_ids], remote_handle,
                          [int(i) for i in remote_ids]])
        return make(op, local, local_ids, remote_handle, remote_ids, *args, **kw)

    def record(scheduler_output):
        from vllm.v1.worker.simulated.kv_transfer import requested_reads
        for read in requested_reads(scheduler_output.kv_connector_metadata):
            if read.blocks:
                reads.append([[int(b) for b in ids] for ids in read.block_ids])
        return before(scheduler_output)

    wrapper.prep_xfer_dlist = prep_xfer_dlist
    wrapper.make_prepped_xfer = make_prepped_xfer
    if checker is not None:
        checker.before_step = record
    params = SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False,
                            extra_args={"kv_transfer_params": remote["params"]})
    llm.generate(prompt, params, use_tqdm=False)
    if not transfers:  # the handshake refused the producer: no READ posted
        with open(out, "w") as f:
            json.dump({**mine, "transfers": 0, "reads": len(reads)}, f)
        return
    local_handle = worker.src_xfer_handles_by_block_size[worker.block_size]
    native = []
    for op, local, local_ids, remote_handle, remote_ids in transfers:
        assert (op, local) == ("READ", local_handle), (op, local)
        native += [a + b for a, b in zip(descriptors(worker.src_blocks_data, local_ids),
                                         descriptors(dlists[remote_handle], remote_ids))]
    # Ours: the selection of the consumer's blocks, each span paired with the
    # same region and offset in the producer's block at the same place.
    (block_ids,) = reads
    ours = []
    s = getattr(checker, "selection", None)
    pairs = [(ids, remote_ids[len(remote_ids) - len(ids):])
             for ids, remote_ids in zip(block_ids, remote["params"]["remote_block_ids"])]
    if s is None:
        # Not a hybrid: each region (a layer slot, which every group's
        # blocks index) a whole page of each block, region by region and,
        # within one, group by group.
        for region, (base, stride, length) in enumerate(
                zip(mine["bases"], mine["strides"], mine["lens"])):
            for ids, remote_ids in pairs:
                for place, block in enumerate(ids):
                    ours.append([base + block * stride, length,
                                 remote["bases"][region] + remote_ids[place] * remote["strides"][region],
                                 remote["lens"][region]])
    # A selection's: each group's spans paired by place, in the native order
    # of every group's blocks together (region by region, then group).
    spans = {}
    for group, (ids, remote_ids) in enumerate(pairs):
        if s is None:
            break
        for row, block, place in s.select(group, ids):
            r = s.rows[row]
            spans[int(s.starts[row]) + block] = [
                r.view[block].data_ptr(), r.view.shape[1],
                remote["bases"][r.region] + r.offset
                + remote_ids[place] * remote["strides"][r.region],
                r.view.shape[1]]
    if s is not None:
        ours = [spans[d] for d in s.order(dict(enumerate(block_ids)))]
    with open(out + ".steps") as f:
        checks = [json.loads(line)["kv_check"] for line in f if '"kv_check"' in line]
    with open(out, "w") as f:
        json.dump({**mine, "native": native, "ours": ours, "checks": checks,
                   "transfers": len(transfers)}, f)

if __name__ == "__main__":
    main()
"""


# A tiny DeepSeek-V3.2 (DSA): each layer a BF16 latent and an indexer key
# cache, of two pages, in one group.
DEEPSEEK_V32 = {
    **DEEPSEEK_V3,
    "architectures": ["DeepseekV32ForCausalLM"],
    "model_type": "deepseek_v32",
    "q_lora_rank": 64,
    "index_topk": 16,
    "index_head_dim": 128,
    "index_n_heads": 4,
}


# A tiny GLM-5 (DSA): its published config scaled down, an indexer on
# alternate layers ("full"), the others sharing the previous top-k.
GLM_5 = {
    "architectures": ["GlmMoeDsaForCausalLM"],
    "model_type": "glm_moe_dsa",
    "vocab_size": 128,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "kv_lora_rank": 512,
    "q_lora_rank": 64,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 64,
    "v_head_dim": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "moe_router_dtype": "float32",
    "index_topk": 16,
    "index_head_dim": 128,
    "index_n_heads": 4,
    "index_topk_freq": 2,
    "index_skip_topk_offset": 1,
    "index_topk_pattern": None,
    "indexer_rope_interleave": True,
    "indexer_types": ["full", "shared", "full", "shared"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "num_nextn_predict_layers": 0,
    "max_position_embeddings": 256,
    "rope_theta": 1000000,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def test_dsa_sparse_mla_builds_on_cpu_in_a_simulation_only(tmp_path, monkeypatch):
    """DeepSeek-V3.2's and GLM-5's sparse MLA: in a simulation the CPU
    platform keeps the GPU deployment's sparse backend for its caches (no
    kernel runs); each attention layer a BF16 latent, each layer with an
    indexer its FP8 keys and scales (132 bytes a token), GLM-5's on the
    layers its config marks full. Outside a simulation, refused."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for name, config, indexers in (("v32", DEEPSEEK_V32, 2), ("glm", GLM_5, 2)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        llm = LLM(
            model=str(directory), skip_tokenizer_init=True, dtype="bfloat16",
            max_model_len=256, kv_cache_memory_bytes=8 * 1024**2,
            max_num_batched_tokens=64, enable_prefix_caching=False,
            enforce_eager=True, block_size=64,
            simulation_config={"simulate_forward": True, "timing": "linear",
                               "timing_args": {"fixed_us": 100}},
        )
        try:
            (out,) = llm.generate(
                [{"prompt_token_ids": [5] * 100}],
                SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
                use_tqdm=False,
            )
            assert len(out.outputs[0].token_ids) == 2
            engine = llm.llm_engine.engine_core.engine_core
            runner = engine.model_executor.driver_worker.worker.model_runner
            specs = runner.get_kv_cache_spec()
            latents = {n: s for n, s in specs.items() if not n.endswith("indexer.k_cache")}
            keys = {n: s for n, s in specs.items() if n.endswith("indexer.k_cache")}
            assert len(latents) == config["num_hidden_layers"] and len(keys) == indexers
            assert {(s.head_size, s.dtype, s.page_size_bytes) for s in latents.values()} == {
                (576, torch.bfloat16, 64 * 576 * 2)
            }
            assert {(s.head_size, s.dtype, s.page_size_bytes) for s in keys.values()} == {
                (132, torch.uint8, 64 * 132)
            }
            (backends,) = llm.apply_model(
                lambda m: sorted({
                    a.get_attn_backend().get_name()
                    for a in m.modules()
                    if hasattr(a, "get_attn_backend")
                })
            )
            assert "CPU_SIMULATED_MLA_SPARSE" in backends, backends
        finally:
            del llm
    with pytest.raises(Exception, match="Sparse Attention is not supported on CPU"):
        LLM(model=str(tmp_path / "v32"), skip_tokenizer_init=True, dtype="bfloat16",
            max_model_len=256, enforce_eager=True, block_size=64)


def test_mla_keeps_a_quantized_cache_in_a_simulation(tmp_path, monkeypatch):
    """A checkpoint that declares an FP8 KV cache starts as on its GPU: a
    sparse MLA latent in FlashMLA's packed format (656 bytes a token), a
    dense MLA latent a byte an element; the CPU kernels that take neither
    never run."""
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    for name, config, cache_dtype, token_bytes in (
        ("v32", DEEPSEEK_V32, "fp8_ds_mla", 656),
        ("glm", GLM_5, "fp8_ds_mla", 656),
        ("v3", DEEPSEEK_V3, "fp8", 576),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(config))
        llm = LLM(
            model=str(directory), skip_tokenizer_init=True, dtype="bfloat16",
            max_model_len=256, kv_cache_memory_bytes=8 * 1024**2,
            max_num_batched_tokens=64, enable_prefix_caching=False,
            enforce_eager=True, block_size=64, kv_cache_dtype="fp8",
            simulation_config={"simulate_forward": True, "timing": "linear",
                               "timing_args": {"fixed_us": 100}},
        )
        try:
            (out,) = llm.generate(
                [{"prompt_token_ids": [5] * 100}],
                SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
                use_tqdm=False,
            )
            assert len(out.outputs[0].token_ids) == 2
            engine = llm.llm_engine.engine_core.engine_core
            runner = engine.model_executor.driver_worker.worker.model_runner
            latents = [s for n, s in runner.get_kv_cache_spec().items()
                       if not n.endswith("indexer.k_cache")]
            assert {(s.cache_dtype_str, s.dtype, s.page_size_bytes) for s in latents} == {
                (cache_dtype, torch.uint8, 64 * token_bytes)
            }, name
        finally:
            del llm


def test_real_mode_takes_dsa_native_mtp_over_nixl_only(tmp_path_factory):
    """DeepSeek-V3.2's and GLM-5's one MTP layer (its own latent and indexer
    keys) moves with the target's over NIXL: not over Mooncake, nor two MTP
    layers."""
    kwargs = {"dtype": "bfloat16", "block_size": 64}

    def mtp(k=1):
        return {"method": "mtp", "num_speculative_tokens": k}

    for name, config in (("v32", DEEPSEEK_V32), ("glm", GLM_5)):
        one = write_model_dir(tmp_path_factory, f"tiny-{name}-mtp",
                              {**config, "num_nextn_predict_layers": 1})
        two = write_model_dir(tmp_path_factory, f"tiny-{name}-mtp-two",
                              {**config, "num_nextn_predict_layers": 2})
        for connector in ("NixlConnector", "NixlPullConnector"):
            for k in (1, 5):
                built = engine_config(one, kv_transfer={"kv_connector": connector},
                                      speculative_config=mtp(k), **kwargs)
                assert built.speculative_config.draft_model_config.architectures == [
                    "DeepseekV32MTPModel"
                ]
        for model, changes, problem in (
            (one, {"kv_transfer": {"kv_connector": "MooncakeConnector"}}, "not MooncakeConnector"),
            (one, {"speculative_config": mtp(6)}, "6 speculative tokens"),
            (two, {}, "2 MTP layers"),
        ):
            changes.setdefault("speculative_config", mtp())
            with pytest.raises(ValueError, match=problem):
                engine_config(model, **changes, **kwargs)


def test_real_mode_takes_dsa_sparse_mla_with_a_bf16_latent(tmp_path_factory):
    """DeepSeek-V3.2's and GLM-5's sparse MLA over either connector, its
    latent BF16 as the deployment keeps it (`auto`) or FlashMLA's packed FP8
    (any FP8 cache dtype); other sparse MLA is refused."""
    v32 = write_model_dir(tmp_path_factory, "tiny-deepseek-v3.2", DEEPSEEK_V32)
    glm = write_model_dir(tmp_path_factory, "tiny-glm-5", GLM_5)
    kwargs = {"dtype": "bfloat16", "block_size": 64}
    for model in (v32, glm):
        for connector in ("NixlConnector", "MooncakeConnector"):
            for dtype in ("auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_ds_mla"):
                engine_config(model, kv_transfer={"kv_connector": connector},
                              kv_cache_dtype=dtype, **kwargs)
        # `auto` takes the model's dtype: a float16 model's latent is not BF16.
        with pytest.raises(ValueError, match="BF16 sparse-MLA latent cache"):
            engine_config(model, kv_cache_dtype="auto", dtype="float16", block_size=64)
    other = write_model_dir(
        tmp_path_factory, "tiny-sparse-v3",
        {**DEEPSEEK_V32, "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"},
    )
    with pytest.raises(ValueError, match="no sparse MLA"):
        engine_config(other, **kwargs)


# A tiny MiMo-V2-Flash: full layers of 1 KV head, sliding ones (window 32,
# sink biases) of 2, K 48 / V 32 wide: equal pages, blocks of two sizes.
MIMO_V2_FLASH = {
    "architectures": ["MiMoV2FlashForCausalLM"],
    "model_type": "mimo_v2_flash",
    "vocab_size": 256,
    "hidden_size": 128,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 48,
    "v_head_dim": 32,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 48,
    "swa_v_head_dim": 32,
    "hybrid_layer_pattern": [0, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1],
    "sliding_window": 32,
    "sliding_window_size": 32,
    "attention_chunk_size": 32,
    "add_swa_attention_sink_bias": True,
    "add_full_attention_sink_bias": False,
    "attention_value_scale": 0.707,
    "partial_rotary_factor": 0.334,
    "rope_theta": 5000000,
    "swa_rope_theta": 10000,
    "n_routed_experts": 4,
    "n_shared_experts": None,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 1,
    "topk_group": 1,
    "hidden_act": "silu",
    "layernorm_epsilon": 1e-05,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


# A tiny MiMo-V2-Flash whose attention groups differ (2 full, 3 sliding
# layers in groups of two slots: 2, 2 and 1), so the last group holds no
# layer in the second slot, which NIXL moves with it.
MIMO_V2_FLASH_UNEQUAL = {
    **MIMO_V2_FLASH,
    "num_hidden_layers": 5,
    "hybrid_layer_pattern": [0, 1, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1, 1],
}


def test_nixl_checks_cover_unequal_attention_groups(tmp_path, monkeypatch):
    """Attention groups of unequal layers over NIXL (MiMo-V2-Flash's 9 full
    and 39 sliding layers in groups of 9, 8 and 7, say): each block of a
    group moves in every layer slot, a slot holding none of a short group's
    layers included (transfer-only). The filler fills and the checker
    requires every selected span, a window's null head holding its places;
    the transfer-only bytes are counted apart, and an unmoved transfer-only
    span or a block of another place fails the check."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker, NativeFiller, pattern

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    (tmp_path / "config.json").write_text(json.dumps(MIMO_V2_FLASH_UNEQUAL))
    llm = LLM(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
    )
    # The runtime path: a chunked prompt beyond the window.
    llm.generate(
        [{"prompt_token_ids": [5] * 200}],
        SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    engine = llm.llm_engine.engine_core.engine_core
    runner = engine.model_executor.driver_worker.worker.model_runner
    groups = engine.scheduler.kv_cache_config.kv_cache_groups
    worker = get_kv_transfer_group().connector_worker
    filler = runner._kv_filler
    assert isinstance(filler, NativeFiller)
    s = filler.selection

    # A region a slot; each group's block in every one, a whole page.
    slots = max(len(group.layer_names) for group in groups)
    assert sorted(len(group.layer_names) for group in groups) == [1, 2, 2]
    assert worker.num_regions == slots
    page = groups[0].kv_cache_spec.page_size_bytes
    for index, group in enumerate(groups):
        spans = [
            (s.rows[row].region, s.rows[row].offset, s.rows[row].view.shape[1])
            for row, _, _ in s.select(index, [1])
        ]
        assert spans == [(region, 0, page) for region in range(slots)]
        assert len(s.members[index]) == len(group.layer_names)
    short = next(i for i, group in enumerate(groups) if len(group.layer_names) < slots)
    (spare,) = set(range(slots)) - s.members[short]

    # A producer's blocks after a null head: each span's pattern is of its
    # place among the request's blocks, nulls counted.
    free = iter(range(2, worker.num_blocks))
    source = [[0, 0, next(free), next(free)] for _ in groups]
    target = [[0, 0, next(free), next(free)] for _ in groups]
    filler._extend("cmpl-a-0", source, replace=True)
    for index, blocks in enumerate(source):
        for row, block, place in s.select(index, blocks[2:]):
            view = s.rows[row].view
            assert torch.equal(
                view[block], pattern("cmpl-a-0", s.name(index, row), 2 + place, view.shape[1], 0)
            )

    checker = NativeChecker(s)

    def read(skip=None, shift=False):
        """The read NIXL would post into `target` (the null head unread),
        `skip`ping spans or `shift`ing a block to the other place; its check."""
        checker.expect("cmpl-b-0", "cmpl-a-0", [0] * len(groups), target)
        checker.posted["cmpl-a-0"] = [
            int(d) for d in worker._compute_desc_ids(
                [blocks[2:] for blocks in target], worker.num_blocks, None, 1
            )
        ]
        for index in range(len(groups)):
            src = source[index][2:][::-1] if shift else source[index][2:]
            pairs = zip(s.select(index, src), s.select(index, target[index][2:]))
            for (row, a, _), (_, b, _) in pairs:
                if skip is None or not skip(index, row):
                    s.rows[row].view[b] = s.rows[row].view[a]
        return checker._compare("cmpl-b-0", checker.pending.pop("cmpl-b-0"))

    result = read()
    assert result["ok"], result
    for index, g in enumerate(result["groups"]):
        assert (g["blocks"], g["descriptors"], g["bytes"]) == (2, 2 * slots, 2 * slots * page)
        assert g["layers"] == len(groups[index].layer_names)
        # The slot of none of its layers: moved and checked, counted apart.
        assert g["transfer_only_bytes"] == (2 * page if index == short else 0)
    unmoved = read(skip=lambda index, row: index == short and s.rows[row].region == spare)
    assert not unmoved["ok"]
    assert not read(shift=True)["ok"]

# Qwen3.5 with its one MTP layer, whose attention KV joins the target's.
QWEN3_5_MOE_MTP = {
    **QWEN3_5_MOE,
    "text_config": {**QWEN3_5_MOE["text_config"], "mtp_num_hidden_layers": 1},
}


def mtp(k):
    return {"language_model_only": True,
            "speculative_config": {"method": "mtp", "num_speculative_tokens": k}}


@pytest.mark.parametrize(
    "config, kwargs",
    [
        (NEMOTRON_3_UNEQUAL, {}),
        (QWEN3_5_MOE, {"language_model_only": True}),
        (QWEN3_5_MOE_MTP, mtp(1)),
        (QWEN3_5_MOE_MTP, mtp(3)),
        (DEEPSEEK_V32, {"block_size": 64}),
        ({**DEEPSEEK_V32, "num_nextn_predict_layers": 1},
         {"block_size": 64, "speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}),
        ({**DEEPSEEK_V32, "num_nextn_predict_layers": 1},
         {"block_size": 64, "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}}),
        (MIMO_V2_FLASH, {}),
        (MIMO_V2_FLASH_UNEQUAL, {}),
        (KIMI_K3, {}),
    ],
    ids=["nemotron-3-unequal", "qwen3.5", "qwen3.5-mtp-k1", "qwen3.5-mtp-k3", "deepseek-v3.2",
         "deepseek-v3.2-mtp-k1", "deepseek-v3.2-mtp-k3", "mimo-v2-flash", "mimo-v2-flash-unequal",
         "kimi-k3"],
)
def test_nixl_peers_pair_a_hybrids_native_descriptors(config, kwargs, tmp_path):
    """A producer and a consumer of different pools (their own processes,
    backing addresses and block counts), joined by NIXL's own handshake: the
    consumer's native READs pair, in order and with duplicates, each span of
    our selection of its blocks with the same region and offset of the
    producer's block at the same place, and its byte check passes."""
    p, c = run_nixl_peers(config, kwargs, tmp_path)
    # Distinct pools and backing storage.
    assert p["num_blocks"] != c["num_blocks"]
    assert not set(p["bases"]) & set(c["bases"])
    assert c["transfers"] >= 1
    # The native pairs, in order and with their multiplicity, are ours.
    assert c["native"] and c["native"] == c["ours"]
    assert [d[1] for d in c["native"]] == [d[3] for d in c["native"]]
    (check,) = c["checks"]
    assert check["ok"], check
    # A hybrid's check counts NIXL's selection; another model's, its layers'
    # pages, a descriptor each.
    if "descriptors" in check["groups"][0]:
        assert sum(g["descriptors"] for g in check["groups"]) == len(c["native"])
    assert check["bytes"] == sum(d[1] for d in c["native"])
    # Unequal attention groups: the slot holding none of the short group's
    # layers is moved, checked and counted apart.
    if config is MIMO_V2_FLASH_UNEQUAL:
        assert sum(g["transfer_only_bytes"] for g in check["groups"]) > 0


@pytest.mark.parametrize("check", [True, False], ids=["checked", "unchecked"])
def test_nixl_peers_of_other_mtp_depths_are_refused_before_a_read(tmp_path, check):
    """Qwen3.5 at MTP depths 2 and 3: the same attention blocks, conv states
    of other widths. The consumer's handshake refuses the producer, so no
    READ is posted, whether the consumer checks bytes or not."""
    p, c = run_nixl_peers(QWEN3_5_MOE_MTP, mtp(2), tmp_path,
                          consumer_kwargs={**mtp(3), "check": check}, logs=True)
    assert c["transfers"] == 0 and c["reads"] == int(check)
    assert "holds its states otherwise" in c["log"]


# Prefix caching between NIXL peers: request A, then B sharing A's prompt
# and A's generated tokens (the fallback token, 0) on decode.
NIXL_PREFIX_PEER = r"""
import json, os, sys, time

def main():
    role, model, kwargs, out, memory = sys.argv[1:6]
    from vllm import LLM, SamplingParams
    producer = role == "producer"
    simulation = {"simulate_forward": True, "kv_transfer_mode": "real",
                  "timing": "linear", "timing_args": {"fixed_us": 100},
                  "log_path": out + ".steps",
                  ("kv_pattern_fill" if producer else "kv_pattern_check"): True}
    llm = LLM(model=model, skip_tokenizer_init=True, dtype="bfloat16",
              max_model_len=256, kv_cache_memory_bytes=int(memory), block_size=16,
              max_num_batched_tokens=64, enable_prefix_caching=True,
              enforce_eager=True, simulation_config=simulation,
              kv_transfer_config={
                  "kv_connector": "NixlConnector",
                  "kv_role": "kv_producer" if producer else "kv_consumer",
                  "kv_buffer_device": "cpu",
                  "kv_connector_extra_config": {"backends": ["Mooncake"]}},
              **json.loads(kwargs))
    prompts = json.loads(sys.argv[6])
    if producer:
        params = []
        for prompt in prompts:
            (output,) = llm.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False,
                               extra_args={"kv_transfer_params": {"do_remote_decode": True}}),
                use_tqdm=False)
            params.append(output.kv_transfer_params)
        with open(out + ".tmp", "w") as f:
            json.dump({"params": params}, f)
        os.rename(out + ".tmp", out)
        while not os.path.exists(out + ".done"):
            time.sleep(0.2)
    else:
        with open(sys.argv[7]) as f:
            remote = json.load(f)
        outputs = []
        for prompt, params, max_tokens in zip(prompts, remote["params"], json.loads(sys.argv[8])):
            (output,) = llm.generate(
                [{"prompt_token_ids": prompt}],
                SamplingParams(max_tokens=max_tokens, ignore_eos=True, detokenize=False,
                               extra_args={"kv_transfer_params": params}),
                use_tqdm=False)
            outputs.append(list(output.outputs[0].token_ids))
        with open(out, "w") as f:
            json.dump({"outputs": outputs}, f)
    with open(out + ".steps") as f:
        checks = [json.loads(line)["kv_check"] for line in f if '"kv_check"' in line]
    with open(out + ".checks", "w") as f:
        json.dump(checks, f)

if __name__ == "__main__":
    main()
"""


def test_nixl_peers_check_prefix_hits_and_pages_completed_by_generation(model_dir, tmp_path):
    """Prefix caching on both instances. A (40 tokens, its last page partial)
    is pulled, then decode generates 26 tokens, completing that page and a
    wholly generated one. B's prompt is A's, 24 of its generated tokens and
    10 more: prefill hits A's two full prompt pages, decode hits four pages
    (two completed by generation) and pulls the rest. Every check passes."""
    import socket
    import subprocess
    import sys

    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    a = [(7 * i) % 120 + 1 for i in range(40)]
    b = a + [0] * 24 + [3] * 10
    prompts, max_tokens = json.dumps([a, b]), json.dumps([26, 2])
    script = tmp_path / "peer.py"
    script.write_text(NIXL_PREFIX_PEER)
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_HOST_IP": "127.0.0.1",
        "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
    }
    produced, consumed = tmp_path / "producer.json", tmp_path / "consumer.json"
    producer = subprocess.Popen(
        [sys.executable, str(script), "producer", model_dir, "{}", str(produced),
         str(8 * 1024**2), prompts],
        env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
    )
    try:
        deadline = time.monotonic() + 600
        while not produced.exists():
            assert producer.poll() is None, "the producer exited"
            assert time.monotonic() < deadline, "no producer output"
            time.sleep(0.5)
        consumer = subprocess.run(
            [sys.executable, str(script), "consumer", model_dir, "{}", str(consumed),
             str(12 * 1024**2), prompts, str(produced), max_tokens],
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
            timeout=600, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert consumer.returncode == 0, consumer.stdout[-4000:]
    finally:
        (tmp_path / "producer.json.done").touch()
        try:
            producer.wait(timeout=60)
        finally:
            producer.kill()
    assert json.loads(consumed.read_text())["outputs"][0] == [0] * 26
    p_checks = json.loads((tmp_path / "producer.json.checks").read_text())
    c_checks = json.loads((tmp_path / "consumer.json.checks").read_text())
    assert all(c["ok"] for c in p_checks + c_checks), p_checks + c_checks
    # Prefill: B hits A's two full prompt pages.
    (p_hit,) = p_checks
    assert p_hit["kind"] == "hit" and p_hit["groups"][0]["logical_blocks"] == [0, 1]
    # Decode: A's pull (3 pages), B's hit (4 pages, two completed by
    # generation) and B's pull after it (its last page, logical 4).
    pulls = [c for c in c_checks if "kind" not in c]
    hits = [c for c in c_checks if c.get("kind") == "hit"]
    assert [c["blocks"] for c in pulls] == [3, 1]
    (hit,) = hits
    assert hit["groups"][0]["logical_blocks"] == [0, 1, 2, 3]


def run_nixl_peers(config, kwargs, tmp_path, consumer_kwargs=None, logs=False):
    """Run NIXL_PEER's producer and consumer; their JSON records (the
    consumer's with its output when `logs`)."""
    import socket
    import subprocess
    import sys

    def free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text(json.dumps(config))
    script = tmp_path / "peer.py"
    script.write_text(NIXL_PEER)
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_HOST_IP": "127.0.0.1",
        "NIXL_MOONCAKE_IP_ADDR": "127.0.0.1",
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "127.0.0.1",
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    }
    produced, consumed = tmp_path / "producer.json", tmp_path / "consumer.json"

    def run(role, out, memory, *extra):
        own = consumer_kwargs if role == "consumer" and consumer_kwargs else kwargs
        return [sys.executable, str(script), role, str(tmp_path / "model"),
                json.dumps(own), str(out), str(memory), *extra]

    producer = subprocess.Popen(
        run("producer", produced, 8 * 1024**2),
        env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
    )
    try:
        deadline = time.monotonic() + 600
        while not produced.exists():
            assert producer.poll() is None, "the producer exited"
            assert time.monotonic() < deadline, "no producer output"
            time.sleep(0.5)
        consumer = subprocess.run(
            run("consumer", consumed, 12 * 1024**2, str(produced)),
            env={**env, "VLLM_NIXL_SIDE_CHANNEL_PORT": str(free_port())},
            timeout=600,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert consumer.returncode == 0, consumer.stdout[-4000:]
        consumer_log = consumer.stdout
    finally:
        (tmp_path / "producer.json.done").touch()
        try:
            producer.wait(timeout=60)
        finally:
            producer.kill()
    p = json.loads(produced.read_text())
    c = json.loads(consumed.read_text())
    if logs:
        c["log"] = consumer_log
    return p, c


@pytest.mark.parametrize(
    "config, layer",
    [(FALCON_H1, "0 (FalconH1ParallelHybrid)"), (ZAMBA2, "1 (Zamba2HybridLayer)")],
    ids=["falcon-h1", "zamba2"],
)
def test_real_mode_refuses_a_mixer_beside_attention_whatever_the_timing(
    config, layer, monkeypatch, tmp_path
):
    """Falcon-H1 runs attention and a Mamba2 mixer in each layer, Zamba2's
    hybrid layers a shared transformer block beside theirs: refused as built,
    under linear timing too: the refusal does not depend on the timing."""
    from vllm import LLM
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    connector = RecordingConnector.make(hma=True)
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    model_dir = tmp_path / config["model_type"]
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    message = rf"decoder layer {re.escape(layer)} has 1 mixers and attention"
    with pytest.raises(Exception, match=message):
        LLM(
            model=str(model_dir),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=256,
            kv_cache_memory_bytes=8 * 1024**2,
            enable_prefix_caching=False,
            enforce_eager=True,
            simulation_config={
                "simulate_forward": True,
                "kv_transfer_mode": "real",
                "timing": "linear",
                "timing_args": {"fixed_us": 100},
            },
            kv_transfer_config={
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "kv_buffer_device": "cpu",
            },
        )


@pytest.mark.parametrize("block_size", [32, 16])
def test_real_mode_allocates_and_registers_cpu_kv_tensors(
    any_model_dir, monkeypatch, block_size
):
    """An in-process engine in real mode allocates real CPU KV tensors within
    the KV budget and registers them with the connector; a request then runs
    its simulated steps over them. Blocks may be as small as a GPU's (16),
    below what the CPU attention kernels need: simulated steps never run them."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    connector = RecordingConnector.make()
    RecordingConnector.registered.clear()
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    budget = 8 * 1024**2
    llm = LLM(
        model=any_model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=block_size,
        kv_cache_memory_bytes=budget,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        assert len(RecordingConnector.registered) == 1
        caches = RecordingConnector.registered[0]
        # One cache per attention layer of the two-layer model.
        assert len(caches) == 2
        storages = {}
        for name, tensor in caches.items():
            assert tensor.device.type == "cpu" and not tensor.is_meta, name
            # [blocks, KV heads, block size, K and V], or MLA's [blocks, 1,
            # block size, latent]: the GPU's block geometry.
            assert tensor.shape[2] == block_size, (name, tuple(tensor.shape))
            assert tensor.numel() > 0 and tensor.data_ptr() != 0, name
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        allocated = sum(storages.values())
        assert 0 < allocated <= budget
        # Real, writable memory: a pattern written through one view reads back.
        tensor = next(iter(caches.values()))
        tensor.fill_(3)
        assert float(tensor.sum()) == 3 * tensor.numel()
        tensor.zero_()

        output = llm.generate(
            [{"prompt_token_ids": [5] * 40}],
            SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output[0].outputs[0].token_ids) == 2
    finally:
        del llm


@pytest.mark.parametrize("simulated_first", [False, True])
def test_mla_backend_choice_is_cached_per_mode(monkeypatch, simulated_first):
    """On an AMX host ordinary runs take AMX_MLA and simulations CPU_MLA,
    whichever selects first: the selector's cache keys on the mode."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.simulation import SimulationConfig
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: True)
    configs = [
        (VllmConfig(), "AMX_MLA"),
        (VllmConfig(simulation_config=SimulationConfig(simulate_forward=True)), "CPU_MLA"),
    ]
    if simulated_first:
        configs.reverse()
    _cached_get_attn_backend.cache_clear()
    try:
        for config, name in configs * 2:
            with set_current_vllm_config(config):
                backend = get_attn_backend(LATENT, torch.bfloat16, "auto", use_mla=True)
            assert backend.get_name() == name, config.simulation_config.simulate_forward
    finally:
        _cached_get_attn_backend.cache_clear()


def test_only_a_simulation_keeps_a_quantized_mla_cache_on_cpu():
    """The CPU MLA kernels read no quantized cache: an ordinary run still
    refuses one; a simulation, which runs no kernel, keeps the backend."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.simulation import SimulationConfig
    from vllm.v1.attention.selector import _cached_get_attn_backend, get_attn_backend

    _cached_get_attn_backend.cache_clear()
    try:
        for dtype in ("fp8", "fp8_e4m3"):
            with (
                set_current_vllm_config(VllmConfig()),
                pytest.raises(ValueError, match="not supported by MLA attention on CPU"),
            ):
                get_attn_backend(LATENT, torch.bfloat16, dtype, use_mla=True)
            simulated = VllmConfig(simulation_config=SimulationConfig(simulate_forward=True))
            with set_current_vllm_config(simulated):
                backend = get_attn_backend(LATENT, torch.bfloat16, dtype, use_mla=True)
            assert backend.get_name() == "CPU_MLA"
    finally:
        _cached_get_attn_backend.cache_clear()


@pytest.mark.parametrize("amx", [True, False])
def test_mla_blocks_stay_whole_on_any_host(mla_model_dir, monkeypatch, tmp_path, amx):
    """Without AMX the CPU MLA kernel takes 16-token blocks only, with AMX
    multiples of 32: a simulation registers the latent in the blocks it asks
    for (64 tokens here) on either host, and logs their bytes."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    connector = RecordingConnector.make()
    RecordingConnector.registered.clear()
    monkeypatch.setattr(
        KVConnectorFactory, "get_connector_class", classmethod(lambda cls, c: connector)
    )
    monkeypatch.setattr(torch.cpu, "_is_amx_tile_supported", lambda: amx)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=mla_model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=64,
        kv_cache_memory_bytes=8 * 1024**2,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        caches = RecordingConnector.registered[0]
        assert len(caches) == 2
        num_blocks = next(iter(caches.values())).shape[0]
        for name, tensor in caches.items():
            # [blocks, one latent "head", block size, latent].
            assert tuple(tensor.shape) == (num_blocks, 1, 64, LATENT), name
        llm.generate(
            [{"prompt_token_ids": [5] * 100}],
            SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
    finally:
        del llm
    (layout,) = [r["kv_layout"] for r in log_records(log) if "kv_layout" in r]
    assert layout["shape"] == [num_blocks, 1, 64, LATENT]
    assert layout["block_bytes"] == 64 * LATENT * 4  # float32
    assert layout["bytes"] == 2 * num_blocks * 64 * LATENT * 4


@pytest.mark.parametrize("real", [True, False])
def test_runner_uses_native_kv_tensors_only_in_real_mode(monkeypatch, real):
    """Real mode allocates and registers KV tensors, zeroes new blocks and
    applies copy-on-write through the native runner; simulated mode does none."""
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.simulated.model_runner import SimulatedModelRunner

    calls = []
    for name in (
        "initialize_kv_cache_tensors",
        "_init_kv_zero_meta",
        "_apply_kv_cache_memory_updates",
    ):
        monkeypatch.setattr(
            GPUModelRunner,
            name,
            lambda self, *args, _name=name, **kwargs: calls.append(_name),
        )
    from vllm.config import VllmConfig

    runner = object.__new__(SimulatedModelRunner)
    runner._real_kv_transfer = real
    runner.vllm_config = VllmConfig()
    runner.kv_cache_config = type("Config", (), {"kv_cache_groups": [], "num_blocks": 0})
    runner.kv_caches = []
    runner.device = "cpu"
    monkeypatch.setattr(
        "vllm.v1.worker.simulated.model_runner.has_kv_transfer_group", lambda: False
    )
    runner.initialize_kv_cache_tensors(
        is_profiling=False, kv_cache_allocation_context=None
    )
    runner._init_kv_zero_meta()
    runner._apply_kv_cache_memory_updates(None)
    if real:
        assert calls == [
            "initialize_kv_cache_tensors",
            "_init_kv_zero_meta",
            "_apply_kv_cache_memory_updates",
        ]
    else:
        assert calls == [] and runner.kv_caches == []


# The CPU platform spawns its workers: a test's monkeypatch never reaches them.
# This sitecustomize, on their PYTHONPATH, makes the connector factory build
# the stand-in that VLLM_SIM_TEST_CONNECTOR names (a function and its
# arguments), imported once a worker first asks for its connector.
SITECUSTOMIZE = """
import json, os

_spec = os.environ.get("VLLM_SIM_TEST_CONNECTOR")
if _spec:
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    _made = {}

    def _get_connector_class(cls, config):
        if "class" not in _made:
            import importlib

            target, *args = json.loads(_spec)
            module, _, name = target.rpartition(".")
            _made["class"] = getattr(importlib.import_module(module), name)(*args)
        return _made["class"]

    KVConnectorFactory.get_connector_class = classmethod(_get_connector_class)
"""


@pytest.fixture
def stand_in_connector(monkeypatch, tmp_path):
    """Make this process and the spawned workers build a stand-in connector:
    a module-level function of this file and its arguments."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    def install(function, *args):
        site = tmp_path / "site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
        # The repository root too: a served engine's processes do not start
        # in it, yet import the stand-in from this module.
        root = str(Path(__file__).resolve().parents[4])
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
            [str(site), root, *filter(None, [os.environ.get("PYTHONPATH")])]
        ))
        monkeypatch.setenv(
            "VLLM_SIM_TEST_CONNECTOR",
            json.dumps([f"{__name__}.{function.__name__}", *args]),
        )
        connector = function(*args)
        monkeypatch.setattr(
            KVConnectorFactory,
            "get_connector_class",
            classmethod(lambda cls, c: connector),
        )

    return install


def recording_connector():
    return RecordingConnector.make()


def slow_rank_connector(directory):
    return SlowRankConnector.make(directory)


def log_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("model", ["model_dir", "mla_model_dir"])
def test_tensor_parallel_ranks_register_and_log_their_own_kv(
    model, request, monkeypatch, tmp_path, stand_in_connector
):
    """On 2 ranks each worker registers its own KV heads (1 of the model's 2),
    or the whole MLA latent, which every rank holds; logs them in its own
    step log and serves TENT's metrics on its own port."""
    from vllm import LLM, SamplingParams

    model_dir = request.getfixturevalue(model)

    stand_in_connector(recording_connector)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("TENT_METRICS_HTTP_PORT", "9100")
    monkeypatch.delenv("VLLM_SIM_TENT_METRICS_BASE_PORT", raising=False)
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=32,
        kv_cache_memory_bytes=8 * 1024**2,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=2,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        llm.generate(
            [{"prompt_token_ids": [5] * 70}],
            SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
    finally:
        del llm
    for rank, path in ((0, log), (1, tmp_path / "steps.tp1.jsonl")):
        records = log_records(path)
        (layout,) = [r["kv_layout"] for r in records if "kv_layout" in r]
        assert (layout["tp_rank"], layout["tp_size"]) == (rank, 2)
        assert layout["tent_metrics_port"] == str(9100 + rank)
        if model == "mla_model_dir":
            # [blocks, 1, block size, latent]: not split across ranks.
            assert layout["shape"][1:] == [1, 32, LATENT], layout
            assert layout["block_bytes"] == 32 * LATENT * 4
        else:
            # [blocks, KV heads, block size, K and V]: 1 of the model's 2 heads.
            assert layout["shape"][1] == 1, layout
        assert any("step" in r for r in records)


def test_dp_attention_ranks_log_their_own_kv_apart_from_the_joint_steps(
    moe_model_dir, tmp_path, stand_in_connector
):
    """Served with DP=2 and EP in real mode, each DP rank logs its own KV
    records under its global DP rank (DP rank 1 into `steps.dp1.jsonl`);
    only DP rank 0's log holds the timing record and the joint steps, which
    keep the request's DP rank."""
    import requests

    from tests.utils import RemoteOpenAIServer

    stand_in_connector(recording_connector)
    log = tmp_path / "steps.jsonl"
    simulation = {
        "kv_transfer_mode": "real",
        "kv_pattern_fill": True,
        "timing": "linear",
        "timing_args": {"fixed_us": 100},
        "log_path": str(log),
    }
    kv_transfer = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer",
        "kv_buffer_device": "cpu",
    }
    args = [
        "--skip-tokenizer-init",
        "--dtype", "float32",
        "--max-model-len", "256",
        "--block-size", "32",
        "--kv-cache-memory-bytes", str(8 * 1024**2),
        "--no-enable-prefix-caching",
        "--enforce-eager",
        "--data-parallel-size", "2",
        "--enable-expert-parallel",
        "--simulate-forward",
        "--simulation-config", json.dumps(simulation),
        "--kv-transfer-config", json.dumps(kv_transfer),
    ]  # fmt: skip
    with RemoteOpenAIServer(moe_model_dir, args, max_wait_seconds=600) as server:
        response = requests.post(
            server.url_for("v1/completions"),
            json={
                "model": moe_model_dir,
                "prompt": [5] * 70,
                "max_tokens": 1,
                "ignore_eos": True,
            },
            headers={"X-data-parallel-rank": "1"},
            timeout=120,
        )
        response.raise_for_status()
    joint, own = log_records(log), log_records(tmp_path / "steps.dp1.jsonl")
    for records, rank in ((joint, 0), (own, 1)):
        (layout,) = [r["kv_layout"] for r in records if "kv_layout" in r]
        assert (layout["dp_rank"], layout["dp_local_rank"]) == (rank, rank)
    # The request ran on DP rank 1: its KV events are in its own log.
    assert any("kv_events" in r for r in own)
    assert not any("kv_events" in r for r in joint)
    assert not any("step" in r or "timing" in r for r in own)
    assert sum("timing" in r for r in joint) == 1
    steps = [r for r in joint if "step" in r]
    assert steps
    assert all(q["dp_rank"] == 1 for s in steps for q in s["step"]["requests"])


class SlowRankConnector:
    """Stands in for a consumer's connector: every request's KV but its last
    token loads asynchronously, and TP rank r reports a load finished at its
    (1 + 5 r)-th poll, writing when into `<directory>/rank<r>`."""

    @classmethod
    def make(cls, directory):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

        class Connector(KVConnectorBase_V1):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.to_load, self.matched, self.polls = [], set(), {}

            def get_num_new_matched_tokens(self, request, num_computed_tokens):
                if request.request_id in self.matched:
                    return 0, False
                self.matched.add(request.request_id)
                return request.num_prompt_tokens - 1 - num_computed_tokens, True

            def update_state_after_alloc(self, request, blocks, num_external_tokens):
                if num_external_tokens:
                    self.to_load.append(request.request_id)

            def build_connector_meta(self, scheduler_output):
                loads, self.to_load = self.to_load, []
                return stand_in_metadata(loads)

            def register_kv_caches(self, kv_caches):
                pass

            def start_load_kv(self, forward_context, **kwargs):
                for req_id in self._get_connector_metadata().loads:
                    self.polls.setdefault(req_id, 0)

            def get_finished(self, finished_req_ids):
                from vllm.distributed.parallel_state import get_tp_group

                rank = get_tp_group().rank_in_group
                done = set()
                for req_id in list(self.polls):
                    self.polls[req_id] += 1
                    if self.polls[req_id] >= 1 + 5 * rank:
                        del self.polls[req_id]
                        done.add(req_id)
                        with open(f"{directory}/rank{rank}", "a") as f:
                            f.write(json.dumps({"req_id": req_id, "t": time.monotonic()}) + "\n")
                return None, done or None

            def wait_for_layer_load(self, layer_name):
                pass

            def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
                pass

            def wait_for_save(self):
                pass

        return Connector


def test_a_slower_rank_holds_the_request_until_its_load_finishes(
    model_dir, monkeypatch, tmp_path, stand_in_connector
):
    """Decode resumes a request only once every rank's load finished: the
    executor aggregates the ranks' reports (rank 1 reports 5 polls later)."""
    from vllm import LLM, SamplingParams

    stand_in_connector(slow_rank_connector, str(tmp_path))
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    log = tmp_path / "steps.jsonl"
    llm = LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=32,
        kv_cache_memory_bytes=8 * 1024**2,
        enable_prefix_caching=False,
        enforce_eager=True,
        tensor_parallel_size=2,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "timing": "linear",
            "timing_args": {"fixed_us": 2000},
            "log_path": str(log),
        },
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            "kv_buffer_device": "cpu",
        },
    )
    try:
        (output,) = llm.generate(
            [{"prompt_token_ids": [5] * 64}],
            SamplingParams(max_tokens=2, ignore_eos=True, detokenize=False),
            use_tqdm=False,
        )
        assert len(output.outputs[0].token_ids) == 2
    finally:
        del llm
    (rank0,) = log_records(tmp_path / "rank0")
    (rank1,) = log_records(tmp_path / "rank1")
    assert rank0["t"] < rank1["t"]
    # The request's first step on decode: only its last prompt token.
    first = next(
        r
        for r in log_records(log)
        if "step" in r
        and any(q["num_computed_tokens"] == 63 for q in r["step"]["requests"])
    )
    assert first["t_start"] >= rank1["t"]


@pytest.mark.parametrize("dp_local_size", [2, 1])
@pytest.mark.parametrize(
    "dp_local_rank, tp_rank, port", [(0, 0, 9100), (0, 1, 9101), (1, 0, 9102), (1, 1, 9103)]
)
def test_dp_ranks_on_a_host_serve_tent_metrics_on_their_own_ports(
    monkeypatch, dp_local_size, dp_local_rank, tp_rank, port
):
    """Two DP engines of TP=2 on one host: four transfer engines, four ports,
    whether served together or launched one by one (local size 1, keeping
    their local ranks)."""
    from vllm.distributed import parallel_state
    from vllm.v1.worker.simulated import worker

    for name in ("TENT_METRICS_HTTP_PORT", "VLLM_SIM_TENT_METRICS_BASE_PORT", "MC_TENT_CONF"):
        monkeypatch.delenv(name, raising=False)
    group = SimpleNamespace(rank_in_group=tp_rank, world_size=2)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=dp_local_rank, data_parallel_size_local=dp_local_size
        )
    )
    worker._rank_tent_metrics_port(config)
    assert os.environ["TENT_METRICS_HTTP_PORT"] == str(port)


@pytest.mark.parametrize(
    "supervised_rank, tp_rank, port", [(0, 0, 9100), (0, 1, 9101), (1, 0, 9102), (1, 1, 9103)]
)
def test_supervised_dp_servers_on_a_host_serve_tent_metrics_on_their_own_ports(
    monkeypatch, supervised_rank, tp_rank, port
):
    """The same two DP engines of TP=2, a server each under the multi-port
    external-LB supervisor: every server's own local rank is 0, its place on
    the host is the supervisor's."""
    from vllm.distributed import parallel_state
    from vllm.v1.worker.simulated import worker

    for name in ("TENT_METRICS_HTTP_PORT", "VLLM_SIM_TENT_METRICS_BASE_PORT", "MC_TENT_CONF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_DP_SUPERVISED_RANK_LOCAL", str(supervised_rank))
    monkeypatch.setenv("VLLM_DP_SUPERVISED_SIZE_LOCAL", "2")
    group = SimpleNamespace(rank_in_group=tp_rank, world_size=2)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            data_parallel_rank_local=0,
            data_parallel_size_local=1,
            data_parallel_external_lb=True,
        )
    )
    worker._rank_tent_metrics_port(config)
    assert os.environ["TENT_METRICS_HTTP_PORT"] == str(port)


@pytest.mark.parametrize(
    "env, conf, rank, port, conf_port",
    [
        ({}, None, 1, 9101, None),  # TENT's default port
        ({"TENT_METRICS_HTTP_PORT": "9200"}, None, 1, 9201, None),
        ({}, "inline", 1, 9301, 9301),  # MC_TENT_CONF's metrics/http_port wins
        ({"TENT_METRICS_HTTP_PORT": "9200"}, "file", 1, 9301, 9301),
        ({}, "flat", 1, 9401, 9401),  # TENT's flat-key fallback
        ({}, "both", 1, 9301, 9301),  # the nested path before the flat key
        ({}, None, 0, 9100, None),
    ],
)
def test_each_rank_serves_tent_metrics_on_its_own_port(
    monkeypatch, tmp_path, env, conf, rank, port, conf_port
):
    from vllm.distributed import parallel_state
    from vllm.v1.worker.simulated import worker

    for name in ("TENT_METRICS_HTTP_PORT", "VLLM_SIM_TENT_METRICS_BASE_PORT", "MC_TENT_CONF"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    tent = {"transports": {"hp_tcp": {"enable": True}}, "metrics": {"http_port": 9300}}
    flat = {"transports": tent["transports"], "metrics/http_port": 9400}
    if conf == "inline":
        monkeypatch.setenv("MC_TENT_CONF", json.dumps(tent))
    elif conf == "flat":
        monkeypatch.setenv("MC_TENT_CONF", json.dumps(flat))
    elif conf == "both":
        monkeypatch.setenv("MC_TENT_CONF", json.dumps({**tent, **flat}))
    elif conf == "file":
        (tmp_path / "tent.json").write_text(json.dumps(tent))
        monkeypatch.setenv("MC_TENT_CONF", str(tmp_path / "tent.json"))
    group = SimpleNamespace(rank_in_group=rank, world_size=2)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank_local=None, data_parallel_size_local=1)
    )

    # A second initialization (a restarted connector) keeps the port.
    for _ in range(2):
        worker._rank_tent_metrics_port(config)
        assert os.environ["TENT_METRICS_HTTP_PORT"] == str(port)
        if conf_port is not None:
            resolved = json.loads(os.environ["MC_TENT_CONF"])
            if conf == "flat":
                assert resolved["metrics/http_port"] == conf_port
            else:
                assert resolved["metrics"]["http_port"] == conf_port
            if conf == "both":
                assert resolved["metrics/http_port"] == 9400  # TENT ignores it
            assert resolved["transports"] == tent["transports"]


def test_nixl_two_attention_groups_are_seeded_by_their_global_heads(
    gpt_oss_model_dir, monkeypatch
):
    """The runner's own selection for the tiny gpt-oss's two attention groups
    over NIXL (sliding and full layers of 2 KV heads): its filler is
    head-aware (`heads` the rank's global heads), so each selected span is
    its heads' slabs, the same bytes on any TP size, not the rank's pattern;
    a checker over that selection accepts a read of them and fails a flipped
    byte."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.worker.simulated.kv_pattern import (
        NativeChecker,
        NativeFiller,
        _stream,
        get_base_request_id,
        pattern,
    )

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    llm = LLM(
        model=gpt_oss_model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            "kv_pattern_fill": True,
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
    )
    llm.generate(
        [{"prompt_token_ids": [5] * 100}],
        SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
        use_tqdm=False,
    )
    engine = llm.llm_engine.engine_core.engine_core
    runner = engine.model_executor.driver_worker.worker.model_runner
    groups = engine.scheduler.kv_cache_config.kv_cache_groups
    worker = get_kv_transfer_group().connector_worker
    filler = runner._kv_filler
    assert isinstance(filler, NativeFiller)
    assert filler.heads == ((0, 1),) * len(groups)
    s = filler.selection
    free = iter(range(2, worker.num_blocks))
    source = [[next(free), next(free)] for _ in groups]
    target = [[next(free), next(free)] for _ in groups]
    filler._extend("cmpl-a-0", source, replace=True)
    base = get_base_request_id("cmpl-a-0")
    for index, blocks in enumerate(source):
        for row, block, place in s.select(index, blocks):
            view = s.rows[row].view
            size = view.shape[1]
            want = torch.cat(
                [_stream(f"{base}|h{head}|{s.name(index, row)}|{place}", size // 2) for head in (0, 1)]
            )
            assert torch.equal(view[block], want)
            assert not torch.equal(
                view[block], pattern("cmpl-a-0", s.name(index, row), place, size, 0)
            )

    # A read of them (what NIXL posts: every group's target blocks) checks.
    checker = NativeChecker(s, heads=filler.heads)

    def read(corrupt=False):
        checker.expect("cmpl-b-0", "cmpl-a-0", [0] * len(groups), target)
        ids = [int(d) for d in worker._compute_desc_ids(target, worker.num_blocks, None, 1)]
        checker.posted["cmpl-a-0"] = ids
        checker.reads["cmpl-a-0"] = [ids]
        for index in range(len(groups)):
            for (row, src, _), (_, dst, _) in zip(
                s.select(index, source[index]), s.select(index, target[index])
            ):
                s.rows[row].view[dst] = s.rows[row].view[src]
        if corrupt:
            s.rows[0].view[target[0][0], 0] ^= 1
        return checker._compare("cmpl-b-0", checker.pending.pop("cmpl-b-0"))

    result = read()
    assert result["ok"]
    # What the harness requires of every transfer check's groups.
    assert all(
        g["pattern"] == "head-v1" and g["global_heads"] == [0, 1] for g in result["groups"]
    ) and len(result["groups"]) == len(groups)
    assert not read(corrupt=True)["ok"]


@pytest.mark.parametrize("fill", [True, False], ids=["patterned", "unpatterned"])
def test_nixl_attention_groups_take_a_peer_of_another_tp_size(
    gpt_oss_model_dir, monkeypatch, fill
):
    """The hybrid-state guard (a peer's conv/SSM sizes and block lengths must
    equal ours) is not installed on attention-only groups, with byte patterns
    on or off: a peer whose blocks are twice as long (another TP size) is
    not refused by it. The hybrid's refusal is kept in both modes
    (test_nixl_peers_of_other_mtp_depths_are_refused_before_a_read)."""
    from types import SimpleNamespace

    from vllm import LLM
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    llm = LLM(  # kept: its engine owns the transfer group
        model=gpt_oss_model_dir,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=256,
        kv_cache_memory_bytes=8 * 1024**2,
        max_num_batched_tokens=64,
        enable_prefix_caching=False,
        enforce_eager=True,
        simulation_config={
            "simulate_forward": True,
            "kv_transfer_mode": "real",
            **({"kv_pattern_fill": True} if fill else {}),
            "timing": "linear",
            "timing_args": {"fixed_us": 100},
        },
        kv_transfer_config={
            "kv_connector": "NixlConnector",
            "kv_role": "kv_producer",
            "kv_buffer_device": "cpu",
            "kv_connector_extra_config": {"backends": ["Mooncake"]},
        },
    )
    worker = get_kv_transfer_group().connector_worker
    peer = SimpleNamespace(
        ssm_sizes=tuple(worker._mamba_ssm_size),
        block_lens=[2 * n for n in worker.block_len_per_layer],
    )
    try:
        worker.add_remote_agent(peer)
    except Exception as e:  # the stand-in peer fails NIXL's own checks, not ours
        assert "holds its states otherwise" not in str(e)
    assert llm is not None
