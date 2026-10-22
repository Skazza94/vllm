# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of real NIXL KV transfers under simulated compute."""

import json

import pytest

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
        ({}, {"kv_connector": "NixlPushConnector"}, {}, "pull-mode NixlConnector"),
        (
            {},
            {"kv_connector_module_path": SIMULATED_NIXL},
            {},
            "without kv_connector_module_path",
        ),
        ({}, {"kv_buffer_device": "cuda"}, {}, "CPU KV buffers"),
        ({"kv_transfer_timing": {"load_gbps": 1}}, {}, {}, "no kv_transfer_timing"),
        ({}, {}, {"kv_cache_memory_bytes": None}, "explicit kv_cache_memory_bytes"),
        ({}, {}, {"enable_prefix_caching": True}, "prefix caching disabled"),
        ({"clock": "virtual"}, {}, {}, "the real clock"),
    ],
)
def test_real_mode_rejections(model_dir, simulation, kv_transfer, kwargs, error):
    with pytest.raises(ValueError, match=error):
        engine_config(model_dir, simulation, kv_transfer, **kwargs)


def test_real_mode_needs_a_connector(model_dir):
    with pytest.raises(ValueError, match="needs a kv_transfer_config"):
        engine_config(model_dir, kv_transfer_config=None)


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
    runner = object.__new__(SimulatedModelRunner)
    runner._real_kv_transfer = real
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
