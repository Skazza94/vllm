# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte checks of real KV transfers: pattern on prefill, sentinel and
comparison on decode."""

import json
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
from vllm.v1.worker.simulated.kv_transfer import (
    ReadsInFlight,
    kv_layout,
    scheduled_kv_events,
)
from vllm.v1.worker.simulated.timing import StepLogWriter, load_step_log

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


def test_patterns_of_different_blocks_never_coincide():
    """Regression: an arithmetic pattern had only 255 distinct blocks, so a
    block of another request could match. Distinct identities must differ in
    almost every byte, and a pattern must vary along the block."""
    size = 64 * 1024
    identities = [
        (f"cmpl-{request}-0-81bb7a28", layer, logical)
        for request in range(40)
        for layer in LAYERS
        for logical in range(4)
    ]
    blocks = torch.stack([pattern(*identity, size) for identity in identities])
    assert int((blocks == SENTINEL).sum()) == 0
    first = blocks[:, :16].numpy().tobytes()
    assert len({first[i * 16 : (i + 1) * 16] for i in range(len(identities))}) == len(
        identities
    )
    for other in range(1, len(identities)):
        # Unrelated streams agree on about 1/255 of their bytes.
        assert int((blocks[0] == blocks[other]).sum()) < size // 64
    # Not a short period: shifting a block does not reproduce it.
    for shift in (1, 64, 255, 256, 4096):
        assert int((blocks[0][shift:] == blocks[0][:-shift]).sum()) < size // 64


def test_chunked_prefill_blocks_continue_the_request_pattern():
    # Chunked prefill: the scheduler allocates a request's blocks over
    # several steps; later ones carry the next logical blocks' pattern.
    views = block_views(caches(), 8)
    filler = PatternFiller(views)
    filler.after_update(new_request_step(PREFILL_ID, [3, 5]))
    chunk = new_request_step("unused", [])
    chunk.scheduled_new_reqs = []
    chunk.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[PREFILL_ID], new_block_ids=[([1, 6],)], resumed_req_ids=set()
    )
    filler.after_update(chunk)
    for layer, view in views.items():
        for logical, block in enumerate([3, 5, 1, 6]):
            assert torch.equal(view[block], pattern(PREFILL_ID, layer, logical, view.shape[1]))
    # A step that allocates the request nothing leaves its blocks as they are.
    chunk.scheduled_cached_reqs.new_block_ids = [None]
    filler.after_update(chunk)
    assert filler.blocks[PREFILL_ID] == [[3, 5, 1, 6]]


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
    assert result["block_ids"] == [1, 2]
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


def test_a_block_of_another_request_fails(sides):
    """Regression for the reviewer's case: a block of the 63-token request
    placed where the 65-token request's block belongs."""
    prefill, decode = sides
    other = "cmpl-2639b8b9-5010-4d39-a15b-3a54d956c503-0-b88c99a0"
    PatternFiller(prefill).after_update(new_request_step(other, [0, 1]))
    checker = PatternChecker(decode)
    checker.before_step(recv_step([1, 2], [3, 5]))
    transfer(prefill, decode, [(1, 3), (2, 1)])  # logical 1 comes from `other`
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert [block[1:3] for block in result["mismatched_blocks"]] == [[1, 2], [1, 2]]


def test_a_shifted_region_fails(sides):
    """A block whose bytes landed a few bytes off (e.g. a wrong offset)."""
    prefill, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(recv_step([1, 2], [3, 5]))
    transfer(prefill, decode, [(1, 3), (2, 5)])
    for view in decode.values():
        view[2] = torch.roll(view[2].clone(), 4)
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert {block[1] for block in result["mismatched_blocks"]} == {1}


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


def test_notification_only_reads_are_not_checked(sides):
    # A request aborted before decode allocated blocks only notifies prefill.
    _, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(recv_step([], [3, 5]))
    assert checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID})) == []


def test_kv_events_follow_a_request_lifecycle():
    step = recv_step([1, 2], [3, 5])
    step.kv_connector_metadata.reqs_to_recv["cmpl-notify-0-1"] = ReqMeta(
        local_block_ids=([],),
        local_physical_block_ids=([],),
        tp_size=1,
        remote=step.kv_connector_metadata.reqs_to_recv[DECODE_ID].remote,
    )
    step.scheduled_new_reqs = [SimpleNamespace(req_id=DECODE_ID, block_ids=([1, 2],))]
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["cmpl-old-0-1", "cmpl-pre-0-1"],
        new_block_ids=[([7],), ([4, 6],)],
        resumed_req_ids={"cmpl-pre-0-1"},
    )
    step.finished_req_ids = {"cmpl-done-0-1"}
    assert scheduled_kv_events(step) == {
        "allocated": {DECODE_ID: [1, 2], "cmpl-old-0-1": [7], "cmpl-pre-0-1": [4, 6]},
        "resumed": ["cmpl-pre-0-1"],
        "reads_posted": [DECODE_ID],
        # The aborted request only frees prefill's blocks, named by prefill's id.
        "prefill_notified": [PREFILL_ID],
        "finished": ["cmpl-done-0-1"],
    }
    # Decode's blocks for a read are logged when the read is posted.
    step.scheduled_new_reqs = []
    assert scheduled_kv_events(step)["allocated"][DECODE_ID] == [1, 2]
    assert ReadsInFlight().events(
        KVConnectorOutput(finished_recving={DECODE_ID}, finished_sending={"cmpl-p-0-1"})
    ) == {"reads_done": [DECODE_ID], "reads_failed": [], "sends_done": ["cmpl-p-0-1"]}
    assert ReadsInFlight().events(None) == {}


def test_kv_events_records_leave_out_empty_events(tmp_path):
    log = StepLogWriter(str(tmp_path / "steps.jsonl"))
    log.write_kv_events({"allocated": {}, "finished": [], "reads_done": []}, 1.0, 0.5)
    log.write_kv_events({"allocated": {}, "reads_failed": [DECODE_ID]}, 2.0, 1.5)
    log.close()
    lines = (tmp_path / "steps.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"kv_events": {"reads_failed": [DECODE_ID]}, "t_scheduled": 1.5, "t": 2.0}
    ]


def test_kv_layout_describes_the_registered_blocks():
    assert kv_layout(caches(), 8, 4, "LBHNC") == {
        "layers": 2,
        "num_blocks": 8,
        "block_size": 4,
        "layout": "LBHNC",
        "dtype": "float32",
        "shape": [8, 2, 4, 8],
        "block_bytes": 2 * 4 * 8 * 4,
        "bytes": 2 * 8 * 2 * 4 * 8 * 4,
    }


def test_step_records_leave_out_other_records(tmp_path):
    log = StepLogWriter(str(tmp_path / "steps.jsonl"))
    log.write_kv_layout({"layers": 2}, 0.5)
    log.write_kv_events({"finished": [DECODE_ID]}, 1.0, 0.9)
    log.write_kv_checks([{"req_id": DECODE_ID, "ok": True}], 1.0)
    log.write_threads({1234: "Thread-1 (process_output_sockets)"}, 1.0)
    log.close()
    with (tmp_path / "steps.jsonl").open("a") as f:
        f.write(json.dumps({"device_s": 0.01, "host_s": 0.0}) + "\n")
    assert load_step_log(str(tmp_path / "steps.jsonl")) == [{"device_s": 0.01, "host_s": 0.0}]


def test_nixl_reads_are_logged_one_by_one():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
        NixlKVConnectorStats,
    )
    from vllm.v1.worker.simulated.kv_transfer import completed_transfers

    stats = NixlKVConnectorStats()
    for bytes_, us, post_us, descs in ((393216, 2000, 150, 2), (262144, 1500, 100, 2)):
        stats.record_transfer(
            SimpleNamespace(
                xferDuration=us, postDuration=post_us, totalBytes=bytes_, descCount=descs
            )
        )
    output = KVConnectorOutput(kv_connector_stats=stats)
    assert completed_transfers(output) == {
        "load": {
            "count": 2,
            "bytes": 655360,
            "seconds": 0.0035,
            "each": [[393216, 0.002, 0.00015, 2, None], [262144, 0.0015, 0.0001, 2, None]],
        }
    }


def mooncake_step(req_id, local, transfer_id="xfer-abc"):
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnectorMetadata,
    )

    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        req_id,
        [local],
        {
            "transfer_id": transfer_id,
            "remote_engine_id": "prefill",
            "remote_bootstrap_addr": "http://10.0.0.1:8998",
        },
    )
    return SimpleNamespace(kv_connector_metadata=metadata)


def test_mooncake_pulls_are_checked_by_the_shared_base_request_id(sides):
    # Prefill wrote PREFILL_ID's pattern; decode's id shares its base.
    prefill, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(mooncake_step(DECODE_ID, [1, 2, 4]))
    assert all(int(view[[1, 2, 4]].sum()) == 0 for view in decode.values())
    transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result
    assert result["remote_request_id"] == DECODE_ID


def test_mooncake_requests_are_lifecycle_events():
    step = mooncake_step(DECODE_ID, [1, 2])
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[], new_block_ids=[], resumed_req_ids=set()
    )
    step.finished_req_ids = set()
    events = scheduled_kv_events(step)
    assert (events["reads_posted"], events["allocated"]) == ([DECODE_ID], {DECODE_ID: [1, 2]})
    # A request that only frees prefill's blocks, under an id of its own:
    # named by the transfer id it shares with the real one.
    notify = mooncake_step("a5fc471726e14fb7", [], transfer_id="xfer-abc")
    notify.scheduled_new_reqs = []
    notify.scheduled_cached_reqs = step.scheduled_cached_reqs
    notify.finished_req_ids = set()
    assert scheduled_kv_events(notify)["prefill_notified"] == ["xfer-abc"]


def test_mooncake_sends_and_failed_receives_are_logged():
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
        MooncakeKVConnectorStats,
    )
    from vllm.v1.worker.simulated.kv_transfer import completed_transfers

    sender = MooncakeKVConnectorStats()
    sender.record_transfer(0.002, 393216, 3, wait_s=0.0005, end=12.5)
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=sender)) == {
        "send": {
            "count": 1,
            "bytes": 393216,
            "seconds": 0.002,
            "each": [[393216, 0.002, 0.0005, 3, 12.5]],
        }
    }
    # Stats from before the wait and end were recorded.
    old = MooncakeKVConnectorStats(
        data={"transfer_duration": [0.002], "bytes_transferred": [1], "num_descriptors": [1],
              "num_failed_transfers": [], "num_failed_recvs": [], "num_kv_expired_reqs": []}
    )
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=old))["send"]["each"] == [
        [1, 0.002, None, 1, None]
    ]
    # A failed write: its bytes and descriptors, to compare with the
    # transport's limits.
    failing = MooncakeKVConnectorStats()
    failing.record_failed_transfer(total_bytes=176160768, num_descs=2688, start=7.5)
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=failing)) == {
        "send": {
            "count": 0,
            "bytes": 0,
            "seconds": 0,
            "each": [],
            "failed": 1,
            "failed_each": [[176160768, 2688, 7.5]],
        }
    }
    receiver = MooncakeKVConnectorStats()
    receiver.data["num_failed_recvs"] = [1]
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=receiver)) == {
        "load": {"count": 0, "bytes": 0, "seconds": 0.0, "failed": 1}
    }


def test_a_load_reported_finished_with_invalid_blocks_failed(sides):
    # Mooncake reports a failed load as finished, its blocks as invalid.
    _, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(mooncake_step(DECODE_ID, [1, 2]))
    failed = KVConnectorOutput(finished_recving={DECODE_ID}, invalid_block_ids={2})
    (result,) = checker.after_step(failed)
    assert result == {"req_id": DECODE_ID, "ok": False, "error": "receive failed"}
    reads = ReadsInFlight({DECODE_ID: [1, 2]})
    assert reads.events(failed) == {
        "reads_done": [],
        "reads_failed": [DECODE_ID],
        "sends_done": [],
    }
    assert not reads
    # Another request's invalid blocks do not touch this one.
    other = KVConnectorOutput(finished_recving={DECODE_ID}, invalid_block_ids={7})
    assert ReadsInFlight({DECODE_ID: [1, 2]}).events(other)["reads_done"] == [DECODE_ID]


def test_invalid_blocks_reported_a_poll_before_the_load_finished(sides):
    # The failure is kept until the connector reports the load finished, and
    # recorded then.
    _, decode = sides
    checker = PatternChecker(decode)
    checker.before_step(mooncake_step(DECODE_ID, [1, 2]))
    reads = ReadsInFlight({DECODE_ID: [1, 2]})
    invalid = KVConnectorOutput(invalid_block_ids={2})
    assert checker.after_step(invalid) == []
    assert reads.events(invalid) == {"reads_done": [], "reads_failed": [], "sends_done": []}
    finished = KVConnectorOutput(finished_recving={DECODE_ID})
    (result,) = checker.after_step(finished)
    assert result == {"req_id": DECODE_ID, "ok": False, "error": "receive failed"}
    assert reads.events(finished)["reads_failed"] == [DECODE_ID]
    assert not checker.lost and not reads.lost
    # The next read into the same blocks is checked afresh.
    reads.blocks[DECODE_ID] = [1, 2]
    assert reads.events(finished)["reads_done"] == [DECODE_ID]


def test_a_block_of_another_tensor_parallel_rank_fails():
    """Each rank holds its own KV heads: decode's rank r must receive prefill's
    rank r blocks, not another rank's."""
    assert not torch.equal(
        pattern(PREFILL_ID, LAYERS[0], 0, 64, tp_rank=0),
        pattern(PREFILL_ID, LAYERS[0], 0, 64, tp_rank=1),
    )
    prefill = {rank: block_views(caches(), 8) for rank in (0, 1)}
    for rank, views in prefill.items():
        PatternFiller(views, rank).after_update(new_request_step(PREFILL_ID, [3, 5, 6]))
    for decode_rank, sender_rank, ok in ((1, 1, True), (1, 0, False), (0, 1, False)):
        decode = block_views(caches(), 8)
        checker = PatternChecker(decode, decode_rank)
        checker.before_step(recv_step([1, 2], [3, 5]))
        transfer(prefill[sender_rank], decode, [(1, 3), (2, 5)])
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"] is ok, (decode_rank, sender_rank)


# gpt-oss's KV cache: a sliding-window group (layers 0 and 2) and a full one
# (1 and 3), layer i of each viewing the same memory; a block is one group's
# at a time.
SLIDING = ("model.layers.0.attn.attn", "model.layers.2.attn.attn")
FULL = ("model.layers.1.attn.attn", "model.layers.3.attn.attn")
GROUPS = [list(SLIDING), list(FULL)]


def shared_caches(num_blocks=16):
    memory = [torch.zeros(num_blocks, 2, 4, 8) for _ in SLIDING]
    return {
        SLIDING[0]: memory[0],
        FULL[0]: memory[0],
        SLIDING[1]: memory[1],
        FULL[1]: memory[1],
    }


def groups_step(req_id, *block_ids, cached=False):
    """A step allocating a request blocks of each group: a new request's, or
    with `cached`, a running one's next blocks."""
    step = new_request_step(req_id, [])
    if cached:
        step.scheduled_new_reqs = []
        step.scheduled_cached_reqs = SimpleNamespace(
            req_ids=[req_id], new_block_ids=[tuple(block_ids)], resumed_req_ids=set()
        )
    else:
        step.scheduled_new_reqs[0].block_ids = tuple(block_ids)
    return step


def mooncake_pull(req_id, block_ids, first):
    """A Mooncake pull of each group's blocks, `first` saying where each
    group's start among the request's blocks."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
        MooncakeConnectorMetadata,
    )

    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        req_id,
        [list(group) for group in block_ids],
        {
            "transfer_id": "xfer-abc",
            "remote_engine_id": "prefill",
            "remote_bootstrap_addr": "http://10.0.0.1:8998",
        },
        first_block_indices=first,
    )
    # Pulled before the request is scheduled (it waits for its KV).
    return SimpleNamespace(
        kv_connector_metadata=metadata,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        finished_req_ids=set(),
    )


def copy_blocks(prefill, decode, layers, pairs):
    for layer in layers:
        for d_block, p_block in pairs:
            decode[layer][d_block].copy_(prefill[layer][p_block])


def test_groups_sharing_memory_hold_their_own_layers_pattern():
    prefill = block_views(shared_caches(), 16)
    PatternFiller(prefill, groups=GROUPS).after_update(
        groups_step(PREFILL_ID, [3, 5], [1, 2])
    )
    for layers, blocks in ((SLIDING, [3, 5]), (FULL, [1, 2])):
        for layer in layers:
            for logical, block in enumerate(blocks):
                want = pattern(PREFILL_ID, layer, logical, 256)
                assert torch.equal(prefill[layer][block], want), (layer, block)
    # Filling every layer instead would leave the last writer's pattern.
    unaware = block_views(shared_caches(), 16)
    PatternFiller(unaware).after_update(new_request_step(PREFILL_ID, [3, 5]))
    want = pattern(PREFILL_ID, SLIDING[0], 0, 256)
    assert not torch.equal(unaware[SLIDING[0]][3], want)


def test_a_pull_of_groups_sharing_memory_is_checked_group_by_group():
    prefill = block_views(shared_caches(), 16)
    PatternFiller(prefill, groups=GROUPS).after_update(
        groups_step(PREFILL_ID, [3, 5], [1, 2])
    )
    decode = block_views(shared_caches(), 16)
    checker = PatternChecker(decode, groups=GROUPS, transfer_group_ids=[0, 1])
    checker.before_step(mooncake_pull(DECODE_ID, [[7, 8], [9, 10]], [0, 0]))
    copy_blocks(prefill, decode, SLIDING, [(7, 3), (8, 5)])
    copy_blocks(prefill, decode, FULL, [(9, 1), (10, 2)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result
    # Each group's blocks over its own layers: each (layer, block) once.
    group = {"first_logical_block": 0, "blocks": 2, "layers": 2, "block_bytes": 512,
             "bytes": 1024, "mismatched_bytes": 0}
    assert result["groups"] == [{"group": 0, **group}, {"group": 1, **group}]
    assert (result["blocks"], result["layers"], result["bytes"]) == (4, 4, 2048)
    assert result["block_ids"] == [7, 8, 9, 10]
    # Bytes of the other group's layer in the same memory fail.
    checker.before_step(mooncake_pull(DECODE_ID, [[7, 8], [9, 10]], [0, 0]))
    copy_blocks(prefill, decode, SLIDING, [(7, 3), (8, 5)])
    copy_blocks(prefill, decode, FULL, [(9, 3), (10, 2)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert [g["mismatched_bytes"] > 0 for g in result["groups"]] == [False, True]


def test_a_window_is_checked_at_its_place_past_null_and_reused_blocks():
    """A long prompt prefilled in chunks: the window's head turns null and its
    blocks are freed and reused, by the request itself or another. Mooncake
    pulls the window's last blocks into decode before the request is
    scheduled, saying where they start."""
    prefill = block_views(caches(), 8)
    filler = PatternFiller(prefill)
    filler.after_update(new_request_step(PREFILL_ID, [1, 2, 3]))
    # Positions 0 and 1 left the window (null); block 1 returns to the request
    # at position 4, block 2 goes to another request.
    filler.after_update(groups_step(PREFILL_ID, [4, 1], cached=True))
    other = "cmpl-other-0-81bb7a28"
    filler.after_update(new_request_step(other, [2]))
    assert filler.blocks[PREFILL_ID] == [[1, 2, 3, 4, 1]]
    for logical, block in ((2, 3), (3, 4), (4, 1)):
        for layer in LAYERS:
            want = pattern(PREFILL_ID, layer, logical, 256)
            assert torch.equal(prefill[layer][block], want)

    decode = block_views(caches(), 8)
    for first, ok in ((2, True), (0, False)):
        checker = PatternChecker(decode)
        checker.before_step(mooncake_pull(DECODE_ID, [[5, 6, 7]], [first]))
        transfer(prefill, decode, [(5, 3), (6, 4), (7, 1)])
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"] is ok, first
        assert result["groups"][0]["first_logical_block"] == first


def test_null_blocks_keep_their_places_and_are_never_checked():
    prefill = block_views(caches(), 8)
    PatternFiller(prefill).after_update(new_request_step(PREFILL_ID, [0, 0, 3, 5]))
    assert int(prefill[LAYERS[0]][0].sum()) == 0
    decode = block_views(caches(), 8)
    for view in decode.values():
        view[0] = 7
    checker = PatternChecker(decode)
    checker.before_step(mooncake_pull(DECODE_ID, [[0, 0, 1, 2]], [0]))
    # The null block is not the read's: left alone, not lost for another's.
    assert all(int((view[0] == 7).sum()) == view.shape[1] for view in decode.values())
    assert checker.after_step(KVConnectorOutput(invalid_block_ids={0})) == []
    transfer(prefill, decode, [(1, 3), (2, 5)])
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result
    assert (result["blocks"], result["block_ids"]) == (2, [1, 2])


def test_kv_events_leave_out_the_null_block():
    pull = mooncake_pull(DECODE_ID, [[7, 8], [0, 9]], [0, 1])
    events = scheduled_kv_events(pull)
    assert (events["reads_posted"], events["allocated"]) == (
        [DECODE_ID],
        {DECODE_ID: [7, 8, 9]},
    )
    scheduled = groups_step(DECODE_ID, [7, 8], [0, 9])
    scheduled.kv_connector_metadata = None
    assert scheduled_kv_events(scheduled)["allocated"] == {DECODE_ID: [7, 8, 9]}


def test_allocations_by_group_tell_groups_apart():
    from vllm.v1.worker.simulated.kv_transfer import GroupAllocations

    def events(*block_ids):
        step = groups_step(DECODE_ID, *block_ids)
        step.kv_connector_metadata = None
        return scheduled_kv_events(step, GroupAllocations([0, 1]))

    one, other = events([5], [8, 9]), events([5, 8], [9])
    assert one["allocated"] == other["allocated"] == {DECODE_ID: [5, 8, 9]}
    assert one["allocated_groups"] == {
        DECODE_ID: [
            {"group": 0, "first": 0, "blocks": [5]},
            {"group": 1, "first": 0, "blocks": [8, 9]},
        ]
    }
    assert other["allocated_groups"] == {
        DECODE_ID: [
            {"group": 0, "first": 0, "blocks": [5, 8]},
            {"group": 1, "first": 0, "blocks": [9]},
        ]
    }


def test_allocations_by_group_keep_places_through_chunks_resumption_and_pulls():
    from vllm.v1.worker.simulated.kv_transfer import GroupAllocations

    # KV cache group 1 transfers no KV: a pull lists groups 0 and 2.
    allocations = GroupAllocations([0, 2])

    def events(step):
        step.kv_connector_metadata = getattr(step, "kv_connector_metadata", None)
        return scheduled_kv_events(step, allocations)["allocated_groups"]

    # A window's head null, then a chunk: the window gets block 1 again at
    # its third place.
    assert events(groups_step(PREFILL_ID, [0, 1], [3], [4])) == {
        PREFILL_ID: [
            {"group": 0, "first": 0, "blocks": [0, 1]},
            {"group": 1, "first": 0, "blocks": [3]},
            {"group": 2, "first": 0, "blocks": [4]},
        ]
    }
    assert events(groups_step(PREFILL_ID, [1], [], [6], cached=True)) == {
        PREFILL_ID: [
            {"group": 0, "first": 2, "blocks": [1]},
            {"group": 2, "first": 1, "blocks": [6]},
        ]
    }
    # Resumed after a preemption: its blocks replaced, from the start.
    resumed = groups_step(PREFILL_ID, [2], [5], [7], cached=True)
    resumed.scheduled_cached_reqs.resumed_req_ids = {PREFILL_ID}
    assert [e["first"] for e in events(resumed)[PREFILL_ID]] == [0, 0, 0]
    # A pull clipped to the window, before the request is scheduled: by its
    # KV cache groups, where the connector's blocks start.
    pull = mooncake_pull(DECODE_ID, [[8, 9], [10, 11, 12]], [5, 0])
    pull.finished_req_ids = {PREFILL_ID}
    assert events(pull) == {
        DECODE_ID: [
            {"group": 0, "first": 5, "blocks": [8, 9]},
            {"group": 2, "first": 0, "blocks": [10, 11, 12]},
        ]
    }
    assert allocations.lengths == {}


def test_allocations_by_group_are_kept_for_several_groups_or_a_window():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.simulated.kv_transfer import group_allocations

    spec = {"block_size": 16, "num_kv_heads": 1, "head_size": 8, "dtype": torch.float32}
    full = FullAttentionSpec(**spec)
    window = SlidingWindowSpec(sliding_window=32, **spec)

    def config(*specs):
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"l{i}"], kind) for i, kind in enumerate(specs)
            ],
        )

    assert group_allocations(config(full)) is None
    assert group_allocations(config(window)) is not None
    assert group_allocations(config(full, full)).transfer_group_ids == (0, 1)


def test_kv_layout_counts_memory_shared_by_groups_once():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        SlidingWindowSpec,
    )

    # A block of a layer: K and V of 4 tokens, one head of 8, FP32.
    spec = {"block_size": 4, "num_kv_heads": 1, "head_size": 8, "dtype": torch.float32}
    groups = [
        KVCacheGroupSpec(list(SLIDING), SlidingWindowSpec(sliding_window=8, **spec)),
        KVCacheGroupSpec(list(FULL), FullAttentionSpec(**spec)),
    ]
    layout = kv_layout(shared_caches(), 16, 4, "LBHNC", groups)
    assert (layout["layers"], layout["block_bytes"], layout["bytes"]) == (
        4,
        2 * 4 * 8 * 4,
        2 * 16 * 2 * 4 * 8 * 4,
    )
    page = {"layers": 2, "block_bytes": 256, "real_page_bytes": 256, "block_stride": 256}
    assert layout["groups"] == [
        {"kind": "SlidingWindowSpec", "sliding_window": 8, "block_size": 4, **page},
        {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 4, **page},
    ]


def test_real_transfers_take_full_and_sliding_window_groups_only():
    from vllm.v1.kv_cache_interface import (
        ChunkedLocalAttentionSpec,
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_groups

    spec = {"block_size": 16, "num_kv_heads": 1, "head_size": 8, "dtype": torch.float32}

    def config(*specs):
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec([f"l{i}"], kind) for i, kind in enumerate(specs)
            ],
        )

    check_real_kv_groups(
        config(FullAttentionSpec(**spec), SlidingWindowSpec(sliding_window=32, **spec))
    )
    check_real_kv_groups(config(MLAAttentionSpec(**spec)))
    with pytest.raises(ValueError, match="not ChunkedLocalAttentionSpec"):
        check_real_kv_groups(
            config(
                FullAttentionSpec(**spec),
                ChunkedLocalAttentionSpec(attention_chunk_size=32, **spec),
            )
        )


# A gated delta-net hybrid's KV cache on a small scale: an attention group
# and a gated delta-net group, two layers each, layer i of both in slot i of
# one allocation (slot 1 at a nonzero offset). A block of an attention layer is
# 256 bytes; a delta-net's state is 240, padded to the same 256-byte page.
ATTENTION = ("model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn")
GDN = ("model.layers.0.linear_attn", "model.layers.4.linear_attn")
HYBRID_GROUPS = [list(GDN), list(ATTENTION)]
PAGE, STATE_BYTES, BLOCKS = 256, 240, 8


def hybrid_caches(storage_bytes=None):
    memory = torch.zeros(storage_bytes or 2 * BLOCKS * PAGE, dtype=torch.uint8)
    caches = {}
    for slot, (gdn, attention) in enumerate(zip(GDN, ATTENTION)):
        offset = slot * BLOCKS * PAGE
        if offset + BLOCKS * PAGE <= len(memory):
            caches[attention] = (
                memory[offset : offset + BLOCKS * PAGE]
                .view(torch.float32)
                .view(BLOCKS, 2, 4, 8)
            )
        # The state's bytes, then the page's padding, as vLLM views them.
        caches[gdn] = memory.view(torch.int8).as_strided(
            (BLOCKS, 1, 1, STATE_BYTES), (PAGE, STATE_BYTES, STATE_BYTES, 1), offset
        )
    return caches


PAGES = {name: PAGE for name in GDN + ATTENTION}


def test_a_padded_state_is_viewed_as_the_page_moved():
    caches = hybrid_caches()
    assert not caches[GDN[0]].is_contiguous()
    views = block_views(caches, BLOCKS, PAGES)
    assert {tuple(v.shape) for v in views.values()} == {(BLOCKS, PAGE)}
    # Slot 1's layers start past slot 0's blocks.
    assert views[GDN[1]].data_ptr() - views[GDN[0]].data_ptr() == BLOCKS * PAGE
    # Layer i of both groups is the same memory; the padding is the page's.
    views[GDN[0]][2] = 7
    assert int((views[ATTENTION[0]][2] == 7).sum()) == PAGE
    # Never another layer's: slot 1 untouched.
    assert int(views[GDN[1]].sum()) == int(views[ATTENTION[1]].sum()) == 0


def test_a_page_beyond_its_stride_or_storage_is_refused():
    with pytest.raises(ValueError, match="300-byte page within its layer's 256-byte"):
        block_views(hybrid_caches(), BLOCKS, {**PAGES, GDN[0]: 300})
    # Slot 1's last state fits its storage, its padded page does not.
    short = hybrid_caches(2 * BLOCKS * PAGE - (PAGE - STATE_BYTES))
    with pytest.raises(ValueError, match="starts at byte 2048"):
        block_views(short, BLOCKS, PAGES)


def test_a_delta_net_state_is_filled_and_checked_whole_padding_included():
    prefill = block_views(hybrid_caches(), BLOCKS, PAGES)
    # The state's one block (place 0), the attention group's two.
    PatternFiller(prefill, groups=HYBRID_GROUPS).after_update(
        groups_step(PREFILL_ID, [5], [3, 4])
    )
    for layer in GDN:
        assert torch.equal(prefill[layer][5], pattern(PREFILL_ID, layer, 0, PAGE))
    decode = block_views(hybrid_caches(), BLOCKS, PAGES)
    checker = PatternChecker(decode, groups=HYBRID_GROUPS, transfer_group_ids=[0, 1])

    def pull():
        checker.before_step(mooncake_pull(DECODE_ID, [[7], [1, 2]], [0, 0]))
        copy_blocks(prefill, decode, GDN, [(7, 5)])
        copy_blocks(prefill, decode, ATTENTION, [(1, 3), (2, 4)])

    pull()
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"], result
    assert [(g["group"], g["blocks"], g["layers"], g["bytes"]) for g in result["groups"]] == [
        (0, 1, 2, 2 * PAGE),
        (1, 2, 2, 4 * PAGE),
    ]
    # A byte of the state's padding not delivered fails.
    pull()
    decode[GDN[1]][7][PAGE - 1] ^= 0xFF
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert result["mismatched_blocks"] == [[GDN[1], 0, 7, 1]]


def test_kv_layout_describes_a_padded_state_and_counts_storage_once():
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheGroupSpec,
        MambaSpec,
    )

    groups = [
        KVCacheGroupSpec(
            list(GDN),
            MambaSpec(
                block_size=4096,
                shapes=((STATE_BYTES // 4,),),
                dtypes=(torch.float32,),
                page_size_padded=PAGE,
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            ),
        ),
        KVCacheGroupSpec(
            list(ATTENTION),
            FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32),
        ),
    ]
    layout = kv_layout(hybrid_caches(), BLOCKS, 4, "LBHNC", groups)
    # One allocation, as the connector registers it.
    assert layout["bytes"] == 2 * BLOCKS * PAGE
    assert layout["groups"][0] == {
        "kind": "MambaSpec", "sliding_window": None, "block_size": 4096, "layers": 2,
        "block_bytes": PAGE, "real_page_bytes": STATE_BYTES, "block_stride": PAGE,
        "mamba_type": "GDN_ATTN",
        "speculative_blocks": 0,
    }


def test_real_transfers_take_a_recurrent_layers_single_state_only():
    """A gated delta-net's or a Mamba2 mixer's state, one block a request
    (mamba_cache_mode none), without speculative blocks but a native MTP's."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_groups

    def config(**mamba):
        spec = {"block_size": 16, "shapes": ((4,),), "dtypes": (torch.float32,),
                "mamba_type": MambaAttentionBackendEnum.GDN_ATTN, **mamba}
        return KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["a"],
                    FullAttentionSpec(
                        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                    ),
                ),
                KVCacheGroupSpec(["m"], MambaSpec(**spec)),
            ],
        )

    check_real_kv_groups(config())
    check_real_kv_groups(config(mamba_type=MambaAttentionBackendEnum.MAMBA2))
    mamba2 = {"mamba_type": MambaAttentionBackendEnum.MAMBA2}
    for mamba, message in (
        ({"mamba_type": MambaAttentionBackendEnum.MAMBA1}, "MAMBA1"),
        ({"mamba_type": MambaAttentionBackendEnum.SHORT_CONV}, "SHORT_CONV"),
        ({"mamba_cache_mode": "align"}, "GDN_ATTN, mamba_cache_mode align"),
        ({**mamba2, "mamba_cache_mode": "align"}, "MAMBA2, mamba_cache_mode align"),
        ({"num_speculative_blocks": 1}, "1 speculative blocks"),
        ({**mamba2, "num_speculative_blocks": 1}, "1 speculative blocks"),
    ):
        with pytest.raises(ValueError, match=message):
            check_real_kv_groups(config(**mamba))
    # A native MTP of depth k: exactly k scratch slots a state (NIXL drops
    # them), neither none nor another depth's.
    for k in (1, 3):
        check_real_kv_groups(config(num_speculative_blocks=k), speculative_blocks=k)
        check_real_kv_groups(config(**mamba2, num_speculative_blocks=k), speculative_blocks=k)
        for blocks in (0, k + 1):
            with pytest.raises(ValueError, match=f"{blocks} speculative blocks"):
                check_real_kv_groups(
                    config(num_speculative_blocks=blocks), speculative_blocks=k
                )


def test_real_transfers_take_qwens_gated_delta_net_layers_only():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_layers

    class KimiDeltaAttention(QwenGatedDeltaNetAttention):  # shares GDN_ATTN
        pass

    def model(kind):
        layer = kind.__new__(kind)
        torch.nn.Module.__init__(layer)
        container = torch.nn.Module()
        container.layers = torch.nn.ModuleList([layer])
        return container

    check_real_kv_layers(model(QwenGatedDeltaNetAttention))
    with pytest.raises(ValueError, match="not KimiDeltaAttention"):
        check_real_kv_layers(model(KimiDeltaAttention))


def mamba2_mixer(kind=None, use_replayssm=False, num_spec=0):
    """A built Mamba2 mixer (`kind`, default `MambaMixer2`), its fields the
    guard reads."""
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2

    kind = kind or MambaMixer2
    mixer = kind.__new__(kind)
    torch.nn.Module.__init__(mixer)
    mixer.use_replayssm, mixer.num_spec = use_replayssm, num_spec
    return mixer


def attention_layer():
    from vllm.model_executor.layers.attention import Attention

    attention = Attention.__new__(Attention)
    torch.nn.Module.__init__(attention)
    return attention


def decoder(**children):
    layer = torch.nn.Module()
    for name, module in children.items():
        setattr(layer, name, module)
    return layer


def recurrent_model(layers, **extra):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(layers)
    for name, module in extra.items():
        setattr(model, name, module)
    return model


def test_real_transfers_take_one_mamba2_mixer_a_decoder_layer_without_attention():
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_layers

    # Nemotron-H's sublayers: a mixer, an MLP or attention a layer.
    mlp = decoder(mixer=torch.nn.Linear(4, 4, device="meta"))
    check_real_kv_layers(
        recurrent_model([
            decoder(mixer=mamba2_mixer()), mlp, decoder(mixer=attention_layer()),
            decoder(mixer=mamba2_mixer()),
        ])
    )
    # Nemotron-3's: an MoE a layer of its own, its experts no KV.
    moe = decoder(experts=torch.nn.ModuleList([torch.nn.Linear(4, 4, device="meta")] * 2))
    check_real_kv_layers(
        recurrent_model([
            decoder(mixer=mamba2_mixer()), decoder(mixer=moe),
            decoder(mixer=attention_layer()), decoder(mixer=moe),
        ])
    )

    class SubMixer(MambaMixer2):
        pass

    shared = decoder(self_attn=attention_layer())
    for model, message in (
        (recurrent_model([decoder(mixer=mamba2_mixer(SubMixer))]), "not SubMixer's"),
        (
            recurrent_model([decoder(mixer=mamba2_mixer(use_replayssm=True))]),
            "ReplaySSM",
        ),
        (recurrent_model([decoder(mixer=mamba2_mixer(num_spec=1))]), "speculative"),
        # Falcon-H1: attention and a mixer in each layer.
        (
            recurrent_model([
                decoder(
                    mamba=decoder(mamba=mamba2_mixer()), self_attn=attention_layer()
                )
            ]),
            r"decoder layer 0 \(Module\) has 1 mixers and attention",
        ),
        # Zamba2: one transformer block shared by the hybrid layers, each
        # beside its own mixer; the second layer's is no less attention.
        (
            recurrent_model([
                decoder(mamba_decoder=decoder(mamba=mamba2_mixer())),
                *(
                    decoder(
                        shared_transformer=shared,
                        mamba_decoder=decoder(mamba=mamba2_mixer()),
                    )
                    for _ in range(2)
                ),
            ]),
            "decoder layer 1 .* has 1 mixers and attention",
        ),
        (
            recurrent_model([decoder(a=mamba2_mixer(), b=mamba2_mixer())]),
            "has 2 mixers",
        ),
        # A mixer outside the decoder layers, or mixers in two lists: no
        # unambiguous decoder layers.
        (
            recurrent_model([decoder(mixer=mamba2_mixer())], head=mamba2_mixer()),
            "one list of decoder layers",
        ),
        (
            recurrent_model(
                [decoder(mixer=mamba2_mixer())],
                blocks=torch.nn.ModuleList([decoder(mixer=mamba2_mixer())]),
            ),
            "one list of decoder layers",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            check_real_kv_layers(model)


def test_packed_blocks_check_each_layers_real_page_at_its_offset():
    """DeepSeek-V4's packed blocks: each 24-byte block holds both layers'
    pages (8 bytes at 0, 12 at 8), of which 6 and 10 bytes are real. The
    pattern lands on the real bytes at their raw offsets, padding and other
    blocks untouched; a connector moving whole blocks passes the check, a
    corrupted real byte fails it, a corrupted padding byte does not."""
    stride, untouched = 24, 0xAB
    places = {LAYERS[0]: (0, 6), LAYERS[1]: (8, 10)}

    def side():
        storage = torch.full((4 * stride,), untouched, dtype=torch.uint8)
        caches = {
            name: torch.empty(0, dtype=torch.uint8).set_(
                storage.untyped_storage(), offset, (4, page), (stride, 1)
            )
            for name, (offset, page) in places.items()
        }
        pages = {name: page for name, (_, page) in places.items()}
        return storage, block_views(caches, 4, pages)

    prefill_raw, prefill = side()
    PatternFiller(prefill).after_update(new_request_step(PREFILL_ID, [1, 3]))
    for logical, block in enumerate([1, 3]):
        row = prefill_raw[block * stride : (block + 1) * stride]
        for name, (offset, page) in places.items():
            assert torch.equal(
                row[offset : offset + page], pattern(PREFILL_ID, name, logical, page)
            )
        assert bool((row[6:8] == untouched).all()) and bool((row[18:] == untouched).all())
    for block in (0, 2):
        assert bool((prefill_raw[block * stride : (block + 1) * stride] == untouched).all())

    def received(corrupt=None):
        decode_raw, decode = side()
        checker = PatternChecker(decode)
        checker.before_step(recv_step([2, 3], [1, 3]))
        # The connector moves whole blocks, padding included.
        for d_block, p_block in ((2, 1), (3, 3)):
            decode_raw[d_block * stride : (d_block + 1) * stride] = prefill_raw[
                p_block * stride : (p_block + 1) * stride
            ]
        if corrupt is not None:
            decode_raw[corrupt] ^= 0xFF
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        return result

    result = received()
    assert result["ok"], result
    assert result["bytes"] == 2 * (6 + 10)  # the real pages only
    assert not received(corrupt=2 * stride + 8 + 3)["ok"]  # layer 1's real byte
    assert received(corrupt=2 * stride + 7)["ok"]  # padding is not checked


def test_native_checks_release_an_ended_receives_posted_reads():
    """The READs NIXL posted for a request are released when its receive
    ends, failed (a setup failure included: recorded, then raised), or
    finished after its blocks were reported invalid; another pending
    request's are kept."""
    from types import SimpleNamespace

    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.worker.simulated.kv_pattern import (
        NativeChecker,
        _Expected,
        _ExpectedGroup,
    )

    def failing(op, local, local_ids, remote, remote_ids, notif_msg=b""):
        raise RuntimeError("transfer setup failed")

    worker = SimpleNamespace(nixl_wrapper=SimpleNamespace(make_prepped_xfer=failing))
    checker = NativeChecker(SimpleNamespace(worker=worker))
    checker.watch_reads()
    for req, remote, blocks in (("a", "ra", [1]), ("b", "rb", [2]), ("c", "rc", [3])):
        checker.pending[req] = _Expected(remote, [_ExpectedGroup(0, 0, blocks)])
        checker.posted[remote] = [blocks[0]]
    # A setup failure: the descriptors are recorded before the call raises.
    with pytest.raises(RuntimeError):
        worker.nixl_wrapper.make_prepped_xfer("READ", 0, [7], 1, [7], notif_msg=b"ra:1")
    assert checker.posted["ra"] == [1, 7]

    (failed,) = checker.after_step(KVConnectorOutput(failed_recving={"a"}))
    assert not failed["ok"] and set(checker.posted) == {"rb", "rc"}
    assert checker.after_step(KVConnectorOutput(invalid_block_ids={3})) == []
    (lost,) = checker.after_step(KVConnectorOutput(finished_recving={"c"}))
    assert lost["error"] == "receive failed"
    assert set(checker.posted) == {"rb"} and set(checker.pending) == {"b"}


def test_real_transfers_take_a_dsa_group_of_its_latent_and_indexer_keys():
    """A DSA model's group of two kinds: BF16 latents and uint8 indexer keys
    and scales (132 bytes a token), MLA specs of one block size; another
    role, dtype, width or block size is refused, as is the mixed group
    outside DSA."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        MLAAttentionSpec,
        SparseCacheRole,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.simulated.kv_transfer import check_real_kv_groups

    def latent(**changed):
        return MLAAttentionSpec(**{"block_size": 64, "num_kv_heads": 1, "head_size": 576,
                                   "dtype": torch.bfloat16, **changed})

    def keys(**changed):
        return MLAAttentionSpec(**{"block_size": 64, "num_kv_heads": 1, "head_size": 132,
                                   "dtype": torch.uint8, "cache_role": SparseCacheRole.INDEXER,
                                   **changed})

    def config(*members):
        specs = {f"layer{i}": spec for i, spec in enumerate(members)}
        return KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(
                list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs))],
        )

    check_real_kv_groups(config(latent(), keys(), latent()), dsa=True)
    with pytest.raises(ValueError, match="UniformTypeKVCacheSpecs"):
        check_real_kv_groups(config(latent(), keys()))
    for members, message in (
        ((latent(dtype=torch.float16), keys()), "not BF16"),
        ((latent(), keys(dtype=torch.bfloat16)), "not uint8 x 132"),
        ((latent(), keys(head_size=128)), "not uint8 x 132"),
        ((keys(), keys()), "no latent"),
        ((latent(), keys(block_size=32)), "block sizes"),
    ):
        with pytest.raises(ValueError, match=message):
            check_real_kv_groups(config(*members), dsa=True)

