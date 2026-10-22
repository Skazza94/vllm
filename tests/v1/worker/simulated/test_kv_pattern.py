# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte checks of real KV transfers: pattern on prefill, sentinel and
comparison on decode."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
    RemoteMeta,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.simulated.kv_pattern import (
    SENTINEL,
    PatternChecker,
    PatternFiller,
    block_views,
    pattern,
)

LAYERS = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")
PREFILL_ID = "cmpl-abc-0-81bb7a28"
DECODE_ID = "cmpl-abc-0-902c6ab2"


def caches(num_blocks=8):
    # Block-major like the NIXL connector's CPU layout: [blocks, heads, tokens, dim].
    return {name: torch.zeros(num_blocks, 2, 4, 8) for name in LAYERS}


def new_request_step(req_id, block_ids):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id=req_id, block_ids=(block_ids,))],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        finished_req_ids=set(),
    )


def recv_step(local, remote, computed=()):
    metadata = NixlConnectorMetadata()
    metadata.reqs_to_recv[DECODE_ID] = ReqMeta(
        local_block_ids=(local,),
        local_physical_block_ids=(local,),
        tp_size=1,
        local_num_computed_blocks=computed,
        remote=RemoteMeta(
            block_ids=(remote,),
            host="10.0.0.1",
            port=5600,
            engine_id="prefill",
            request_id=PREFILL_ID,
        ),
    )
    return SimpleNamespace(kv_connector_metadata=metadata)


def transfer(prefill, decode, pairs):
    for layer in LAYERS:
        for d_block, p_block in pairs:
            decode[layer][d_block].copy_(prefill[layer][p_block])


@pytest.fixture
def sides():
    prefill_caches, decode_caches = caches(), caches()
    prefill = block_views(prefill_caches, 8)
    decode = block_views(decode_caches, 8)
    PatternFiller(prefill).after_update(new_request_step(PREFILL_ID, [3, 5, 6]))
    return prefill, decode


def test_pattern_is_nonzero_and_distinct():
    a = pattern(PREFILL_ID, LAYERS[0], 0, 256)
    assert int((a == SENTINEL).sum()) == 0
    # The per-request random suffix does not change the pattern; the rest does.
    assert torch.equal(a, pattern(DECODE_ID, LAYERS[0], 0, 256))
    assert not torch.equal(a, pattern(PREFILL_ID, LAYERS[1], 0, 256))
    assert not torch.equal(a, pattern(PREFILL_ID, LAYERS[0], 1, 256))
    assert not torch.equal(a, pattern("cmpl-other-0-81bb7a28", LAYERS[0], 0, 256))


def test_block_views_need_block_major_tensors():
    with pytest.raises(ValueError, match="block-major"):
        block_views({"layer": torch.zeros(2, 8, 4)}, 8)


def test_received_blocks_match_by_logical_block(sides):
    prefill, decode = sides
    checker = PatternChecker(decode)
    # Different physical ids: decode blocks 1, 2 hold prefill's logical 0, 1.
    checker.before_step(recv_step([1, 2], [3, 5]))
    assert all(int(view[[1, 2]].sum()) == SENTINEL for view in decode.values())
    transfer(prefill, decode, [(1, 3), (2, 5)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result
    assert result["blocks"] == 2 and result["layers"] == 2
    assert result["bytes"] == 2 * 2 * decode[LAYERS[0]].shape[1]


@pytest.mark.parametrize(
    "pairs, corrupt",
    [
        ([(1, 3), (2, 5)], True),  # one corrupted byte
        ([(1, 3)], False),  # a block the read never reached: still sentinel
        ([(1, 5), (2, 6)], False),  # shifted by one logical block
    ],
)
def test_bad_transfers_fail(sides, pairs, corrupt):
    prefill, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(recv_step([1, 2], [3, 5]))
    transfer(prefill, decode, pairs)
    if corrupt:
        decode[LAYERS[1]][2][7] ^= 0xFF
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert result["mismatched_bytes"] > 0 and result["mismatched_blocks"]


def test_prefix_hit_offsets_the_logical_blocks(sides):
    prefill, decode = sides
    checker = PatternChecker(decode)
    # Decode already holds logical block 0; it reads logical 1 and 2.
    checker.before_step(recv_step([4, 7], [5, 6], computed=(1,)))
    transfer(prefill, decode, [(4, 5), (7, 6)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result


def test_failed_receive_is_reported(sides):
    _, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(recv_step([1, 2], [3, 5]))
    (result,) = checker.after_step(KVConnectorOutput(failed_recving={DECODE_ID}))
    assert result == {"req_id": DECODE_ID, "ok": False, "error": "receive failed"}
    assert checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID})) == []
