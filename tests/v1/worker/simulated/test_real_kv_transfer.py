# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration of real NIXL KV transfers under simulated compute."""

import json
import os
import time
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


@pytest.fixture(scope="module")
def mla_model_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-deepseek-v2")
    (directory / "config.json").write_text(json.dumps(DEEPSEEK_V2))
    return str(directory)


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


def test_real_mode_allows_tensor_parallelism_only_over_mooncake(model_dir):
    mooncake = {"kv_connector": "MooncakeConnector"}
    config = engine_config(model_dir, kv_transfer=mooncake, tensor_parallel_size=2)
    assert config.parallel_config.tensor_parallel_size == 2
    with pytest.raises(ValueError, match="tensor parallelism at most"):
        engine_config(model_dir, kv_transfer=mooncake, pipeline_parallel_size=2)
    # NIXL does not check the peer's TP size.
    with pytest.raises(ValueError, match="MooncakeConnector for tensor parallelism"):
        engine_config(model_dir, tensor_parallel_size=2)


@pytest.mark.parametrize(
    "mode, rank, name",
    [
        ("real", 0, "steps.jsonl"),
        ("real", 1, "steps.tp1.jsonl"),
        ("simulated", 1, None),  # the output rank's log only
    ],
)
def test_each_tensor_parallel_rank_logs_its_own_transfers(
    monkeypatch, tmp_path, mode, rank, name
):
    from vllm.distributed import parallel_state
    from vllm.v1.worker.simulated.timing import step_log_path

    group = SimpleNamespace(rank_in_group=rank, is_last_rank=True)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)
    config = SimpleNamespace(
        simulation_config=SimpleNamespace(
            log_path=str(tmp_path / "steps.jsonl"), kv_transfer_mode=mode
        ),
        parallel_config=SimpleNamespace(
            data_parallel_index=0, data_parallel_size=1, data_parallel_rank=0
        ),
    )
    path = step_log_path(config)
    assert path == (str(tmp_path / name) if name else None)


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
    """Stands in for the NIXL connector: keeps the KV tensors registered with it."""

    registered: list[dict] = []

    @classmethod
    def make(cls):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )

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
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
            [str(site), *filter(None, [os.environ.get("PYTHONPATH")])]
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
    group = SimpleNamespace(rank_in_group=rank)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)

    # A second initialization (a restarted connector) keeps the port.
    for _ in range(2):
        worker._rank_tent_metrics_port()
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
