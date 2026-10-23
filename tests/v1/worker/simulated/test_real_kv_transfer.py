# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of real NIXL KV transfers under simulated compute."""

import json

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


class RecordingConnector:
    """Stands in for the NIXL connector: keeps the KV tensors registered with it."""

    registered: list[dict] = []

    @classmethod
    def make(cls):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
            KVConnectorMetadata,
        )

        class Metadata(KVConnectorMetadata):
            pass

        class Connector(KVConnectorBase_V1):
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
                return Metadata()

        return Connector


def test_producer_writes_the_pattern_into_its_kv_tensors(model_dir, monkeypatch):
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
        model=model_dir,
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


def test_real_mode_allocates_and_registers_cpu_kv_tensors(model_dir, monkeypatch):
    """An in-process engine in real mode allocates real CPU KV tensors within
    the KV budget and registers them with the connector; a request then runs
    its simulated steps over them."""
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
        model=model_dir,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        block_size=32,
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
