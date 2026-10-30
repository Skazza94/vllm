# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of real NIXL KV transfers under simulated compute."""

import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

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


# A tiny DeepSeek-V2 (MLA and MoE) with DeepSeek's latent: kv_lora_rank 512
# plus a 64-wide rope key, 576 elements per token and layer.
DEEPSEEK_V2 = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "model_type": "deepseek_v2",
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
    directory = tmp_path_factory.mktemp("tiny-deepseek-v2")
    (directory / "config.json").write_text(json.dumps(DEEPSEEK_V2))
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


# A tiny Qwen3-Next: three gated delta-net layers, then full attention.
QWEN3_NEXT = {
    "architectures": ["Qwen3NextForCausalLM"],
    "model_type": "qwen3_next",
    "vocab_size": 128,
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
    "max_position_embeddings": 256,
}


@pytest.fixture(scope="module")
def qwen3_next_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-qwen3-next")
    (directory / "config.json").write_text(json.dumps(QWEN3_NEXT))
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
        (
            {"kv_pattern_check": True, "log_path": "/tmp/steps"},
            {"kv_role": "kv_consumer"},
            {"enable_prefix_caching": True},
            "prefix caching disabled for kv_pattern_fill",
        ),
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


def test_real_mode_allows_prefix_caching(model_dir):
    config = engine_config(model_dir, enable_prefix_caching=True)
    assert config.cache_config.enable_prefix_caching
    assert config.simulation_config.kv_transfer_mode == "real"


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
        json.dumps({**DEEPSEEK_V2, "index_topk": 64, "index_n_heads": 2, "index_head_dim": 64})
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


def test_real_mode_takes_gdn_hybrids_over_mooncake_without_prefix_caching(
    qwen3_next_model_dir,
):
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(qwen3_next_model_dir, kv_transfer=mooncake)
    assert config.model_config.is_hybrid
    assert config.cache_config.mamba_cache_mode == "none"
    # A state per block (align mode) is not modeled, byte checks or not.
    with pytest.raises(ValueError, match="hybrid model without prefix caching"):
        engine_config(qwen3_next_model_dir, kv_transfer=mooncake, enable_prefix_caching=True)
    with pytest.raises(ValueError, match="MooncakeConnector for a hybrid model"):
        engine_config(qwen3_next_model_dir)


def test_real_mode_takes_mamba2_hybrids_over_mooncake_without_prefix_caching(
    nemotron_h_model_dir,
):
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(nemotron_h_model_dir, kv_transfer=mooncake)
    assert config.model_config.is_hybrid
    assert config.cache_config.mamba_cache_mode == "none"
    with pytest.raises(ValueError, match="hybrid model without prefix caching"):
        engine_config(
            nemotron_h_model_dir, kv_transfer=mooncake, enable_prefix_caching=True
        )
    with pytest.raises(ValueError, match="MooncakeConnector for a hybrid model"):
        engine_config(nemotron_h_model_dir)


def test_real_mode_takes_deepseek_v4_over_nixl(v4_model_dir):
    """V4's groups move over NIXL, which keeps each group's own blocks;
    Mooncake cannot pair its block sizes. Tensor parallelism or data-parallel
    attention with expert parallelism (not both), no prefix caching, the
    packed FP8 cache."""
    v4 = {
        "dtype": "bfloat16",
        "max_model_len": 4096,
        "block_size": None,
        "kv_cache_memory_bytes": 4 << 30,
    }
    dp_ep = {"data_parallel_size": 2, "enable_expert_parallel": True}
    for connector in ("NixlConnector", "NixlPullConnector"):
        for kwargs in ({}, {"tensor_parallel_size": 2}, {"tensor_parallel_size": 4}, dp_ep):
            config = engine_config(
                v4_model_dir, kv_transfer={"kv_connector": connector}, **{**v4, **kwargs}
            )
            assert config.model_config.hf_config.model_type == "deepseek_v4"
    for kwargs, message in (
        ({"kv_transfer": {"kv_connector": "MooncakeConnector"}}, "NixlConnector for DeepSeek-V4"),
        ({**dp_ep, "tensor_parallel_size": 2}, "DeepSeek-V4 with tensor parallelism or"),
        ({"data_parallel_size": 2}, "MoE model's data-parallel attention"),
        ({"pipeline_parallel_size": 2}, "tensor parallelism and data-parallel attention at most"),
        ({"enable_prefix_caching": True}, "DeepSeek-V4 without prefix caching"),
        ({"kv_cache_dtype": "bfloat16"}, "packed FP8 cache"),
    ):
        with pytest.raises(ValueError, match=message):
            engine_config(v4_model_dir, **{**v4, **kwargs})


# A tiny DeepSeek-V3 with its native MTP layer(s).
def deepseek_v3(mtp_layers):
    return {
        **DEEPSEEK_V2,
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "num_nextn_predict_layers": mtp_layers,
    }


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


@pytest.mark.parametrize("local_tp, remote_tp", [(2, 1), (1, 2), (2, 2)])
def test_real_transfers_pair_equal_tp_sizes_only(model_dir, local_tp, remote_tp):
    """What NIXL's handshake (and Mooncake's) checks first: real transfers
    pair equal TP sizes (each rank's pattern is its own); ordinary execution
    keeps heterogeneous TP."""
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
        simulated_tp_mismatch,
    )

    simulated = engine_config(model_dir, tensor_parallel_size=local_tp)
    mismatch = simulated_tp_mismatch(simulated, local_tp, remote_tp)
    if local_tp == remote_tp:
        assert mismatch is None
    else:
        assert f"local TP {local_tp} and remote TP {remote_tp}" in mismatch
    assert simulated_tp_mismatch(VllmConfig(), local_tp, remote_tp) is None


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
    for backends in (["UCX"], ["Mooncake", "UCX"]):
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


def test_producer_writes_the_pattern_into_its_kv_tensors(any_model_dir, monkeypatch):
    """With kv_pattern_fill, a prefill engine's real KV tensors hold each
    request's pattern in the blocks the request was given."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.worker.simulated.kv_pattern import block_views, pattern

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
        # 70 tokens span logical blocks 0..2; each holds its pattern somewhere.
        for layer, view in views.items():
            for logical in range(3):
                want = pattern(output.request_id, layer, logical, view.shape[1])
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
    from vllm.v1.worker.simulated.kv_pattern import block_views, pattern

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
                    want = pattern(req_id, layer, logical, view.shape[1])
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
    [("qwen3_next_model_dir", "GDN_ATTN"), ("nemotron_h_model_dir", "MAMBA2")],
    ids=["qwen3-next", "nemotron-h"],
)
def test_a_hybrid_producer_fills_each_groups_pages_padding_included(
    model, mamba_type, request, monkeypatch, tmp_path
):
    """Tiny Qwen3-Next (three delta-net groups) and Nemotron-H (Mamba2
    groups, an MLP-only layer without KV): recurrent groups and an attention
    group in one allocation, the platform's attention block covering a
    state's page. A chunked prompt: each attention block holds its place's
    pattern, each recurrent layer's single state block (place 0) its
    pattern over the whole page, padding included; the layout record
    describes the padding."""
    from vllm import LLM, SamplingParams
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.v1.worker.simulated.kv_pattern import block_views, pattern

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
        ((req_id, block_ids),) = RecordingConnector.finished
        for group, held in zip(groups, block_ids):
            spec = group.kv_cache_spec
            if type(spec).__name__ == "MambaSpec":
                # The state, one block, padded to the attention page.
                assert len(held) == 1 and spec.real_page_size_bytes < spec.page_size_bytes
            for logical, block in enumerate(held):
                for layer in group.layer_names:
                    want = pattern(req_id, layer, logical, spec.page_size_bytes)
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
    finally:
        del llm


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
