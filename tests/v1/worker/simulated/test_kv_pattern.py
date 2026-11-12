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
    TokenChains,
    block_views,
    pattern,
    row_pattern,
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
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id, block_ids=(block_ids,), num_computed_tokens=0)
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()
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


def test_a_nixl_read_names_its_producer_and_its_blocks():
    """The checker learns the producer engine (its block size) and how many
    blocks it holds per group from NIXL's metadata, before the READ."""
    from vllm.v1.worker.simulated.kv_transfer import requested_reads

    (read,) = requested_reads(recv_step([3, 4], [7, 8, 9]).kv_connector_metadata)
    assert (read.req_id, read.remote_request_id) == (DECODE_ID, PREFILL_ID)
    assert (read.remote_engine_id, read.remote_block_counts) == ("prefill", [3])


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
    step.scheduled_new_reqs = [
        SimpleNamespace(req_id=DECODE_ID, block_ids=([1, 2],), num_computed_tokens=7)
    ]
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=["cmpl-old-0-1", "cmpl-pre-0-1"],
        new_block_ids=[([7],), ([4, 6],)],
        num_computed_tokens=[30, 4],
        resumed_req_ids={"cmpl-pre-0-1"},
    )
    step.finished_req_ids = {"cmpl-done-0-1"}
    assert scheduled_kv_events(step) == {
        "allocated": {DECODE_ID: [1, 2], "cmpl-old-0-1": [7], "cmpl-pre-0-1": [4, 6]},
        # Computed at admission: a new request's, a resumed one's (its hit).
        "admitted": {DECODE_ID: 7, "cmpl-pre-0-1": 4},
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
    log.write_kv_writes([{"req_id": DECODE_ID, "phase": "completion", "group": 0,
                          "logical_block": 1, "block_id": 7, "end": 9}], 12, 1.0)
    log.write_threads({1234: "Thread-1 (process_output_sockets)"}, 1.0)
    log.close()
    records = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    # A materialized page names its step as spec_acceptance does.
    (write,) = [r["kv_write"] for r in records if "kv_write" in r]
    assert (write["step_index"], write["phase"], write["block_id"]) == (12, "completion", 7)
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
            # Bytes, seconds, posting, descriptors, end, peer, plan (the
            # last three unknown here).
            "each": [
                [393216, 0.002, 0.00015, 2, None, None, None],
                [262144, 0.0015, 0.0001, 2, None, None, None],
            ],
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


@pytest.mark.parametrize(
    "draft",
    [
        ("model.layers.2.self_attn.attn",),
        ("model.layers.2.self_attn.attn", "model.layers.2.self_attn.indexer.k_cache"),
    ],
    ids=["mla", "sparse-mla"],
)
@pytest.mark.parametrize("fault", ["omitted", "corrupted"])
def test_a_mooncake_pull_checks_the_mtp_draft_layer_with_the_target(draft, fault):
    """A native MTP's draft KV (its latent, and a sparse draft's indexer
    keys) is a layer of the target's groups: a Mooncake pull missing or
    corrupting only it fails, naming only its layers."""
    layers = (*LAYERS, *draft)

    def all_caches():
        return {name: torch.zeros(8, 2, 4, 8) for name in layers}

    prefill, decode = block_views(all_caches(), 8), block_views(all_caches(), 8)
    PatternFiller(prefill).after_update(new_request_step(PREFILL_ID, [3, 5, 6]))
    pairs = [(1, 3), (2, 5), (4, 6)]

    def pull(skip=(), flip=None):
        checker = PatternChecker(decode)
        checker.before_step(mooncake_step(DECODE_ID, [1, 2, 4]))
        for layer in layers:
            if layer not in skip:
                for d_block, p_block in pairs:
                    decode[layer][d_block].copy_(prefill[layer][p_block])
        if flip is not None:
            decode[flip][2][7] ^= 0xFF
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        return result

    whole = pull()
    assert whole["ok"] and whole["layers"] == len(layers), whole
    for layer in draft:
        bad = pull(skip=(layer,)) if fault == "omitted" else pull(flip=layer)
        assert not bad["ok"] and bad["mismatched_bytes"] > 0
        assert {block[0] for block in bad["mismatched_blocks"]} == {layer}


def test_mooncake_requests_are_lifecycle_events():
    step = mooncake_step(DECODE_ID, [1, 2])
    step.scheduled_new_reqs = []
    step.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()
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
            "each": [[393216, 0.002, 0.0005, 3, 12.5, None, None]],
        }
    }
    # Stats from before the wait and end were recorded.
    old = MooncakeKVConnectorStats(
        data={"transfer_duration": [0.002], "bytes_transferred": [1], "num_descriptors": [1],
              "num_failed_transfers": [], "num_failed_recvs": [], "num_kv_expired_reqs": []}
    )
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=old))["send"]["each"] == [
        [1, 0.002, None, 1, None, None, None]
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
            # Bytes, descriptors, start, the consumer rank (unknown here).
            "failed_each": [[176160768, 2688, 7.5, None]],
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
            req_ids=[req_id],
            new_block_ids=[tuple(block_ids)],
            num_computed_tokens=[0],
            resumed_req_ids=set(),
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
            req_ids=[], new_block_ids=[], num_computed_tokens=[], resumed_req_ids=set()
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
    # Without a head map (here), each page its rank's pattern.
    group = {"first_logical_block": 0, "blocks": 2, "layers": 2, "block_bytes": 512,
             "bytes": 1024, "mismatched_bytes": 0, "pattern": "rank-v0", "global_heads": None}
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
    # With prefix caching a pull starts past a hit: logged by group too.
    assert group_allocations(config(full), prefix_caching=True) is not None
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


def test_real_transfers_take_a_recurrent_layers_state_only():
    """A gated delta-net's or a Mamba2 mixer's state, one block a request
    or (prefix caching: mamba_cache_mode align) a block a state, without
    speculative blocks but a native MTP's."""
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
    check_real_kv_groups(config(mamba_cache_mode="align"))
    check_real_kv_groups(config(**mamba2, mamba_cache_mode="align"))
    for mamba, message in (
        ({"mamba_type": MambaAttentionBackendEnum.MAMBA1}, "MAMBA1"),
        ({"mamba_type": MambaAttentionBackendEnum.SHORT_CONV}, "SHORT_CONV"),
        (
            {"mamba_type": MambaAttentionBackendEnum.MAMBA1, "mamba_cache_mode": "align"},
            "MAMBA1, mamba_cache_mode align",
        ),
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
    packed = {"dtype": torch.uint8, "cache_dtype_str": "fp8_ds_mla"}
    check_real_kv_groups(config(latent(**packed), keys(), latent(**packed)), dsa=True)
    with pytest.raises(ValueError, match="UniformTypeKVCacheSpecs"):
        check_real_kv_groups(config(latent(), keys()))
    for members, message in (
        ((latent(dtype=torch.float16), keys()), "not BF16"),
        ((latent(dtype=torch.uint8, cache_dtype_str="fp8"), keys()), "not BF16"),
        ((latent(), keys(dtype=torch.bfloat16)), "not uint8 x 132"),
        ((latent(), keys(head_size=128)), "not uint8 x 132"),
        ((keys(), keys()), "no latent"),
        ((latent(), keys(block_size=32)), "block sizes"),
    ):
        with pytest.raises(ValueError, match=message):
            check_real_kv_groups(config(*members), dsa=True)



# Prefix caching: a block's pattern is its content's (TokenChains).

BLOCK = 4  # tokens a block, as caches() holds them


def step(new=(), cached=(), finished=(), metadata=None):
    """A scheduler output: new requests (req_id, block ids, prompt, computed,
    scheduled), running ones (req_id, new block ids, computed, scheduled,
    resumed)."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=r, block_ids=(ids,), prompt_token_ids=prompt, num_computed_tokens=c
            )
            for r, ids, prompt, c, _ in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, *_ in cached],
            new_block_ids=[None if ids is None else (ids,) for _, ids, *_ in cached],
            num_computed_tokens=[c for _, _, c, _, _ in cached],
            resumed_req_ids={r for r, *_, resumed in cached if resumed},
        ),
        num_scheduled_tokens={r: n for r, *_, n in new}
        | {r: n for r, _, _, n, _ in cached},
        finished_req_ids=set(finished),
        kv_connector_metadata=metadata,
    )


def pull(local, remote, computed=()):
    """A step posting a pull (a full scheduler output)."""
    return step(metadata=recv_step(local, remote, computed).kv_connector_metadata)


def test_content_identity_follows_the_tokens():
    chains = TokenChains([BLOCK, 2 * BLOCK])
    chains.admit("a", list(range(16)))
    chains.admit("b", list(range(12)) + [99, 98])
    chains.admit("c", [7] + list(range(1, 16)))
    # A shared prefix: the same full blocks, whichever request.
    assert chains.identity("a", 0, 2, BLOCK) == chains.identity("b", 0, 2, BLOCK)
    assert chains.identity("a", 1, 0, 8) == chains.identity("b", 1, 0, 8)
    # One token earlier in the chain differs: every later block differs.
    assert chains.identity("a", 0, 0, BLOCK) != chains.identity("c", 0, 0, BLOCK)
    assert chains.identity("a", 0, 3, BLOCK) != chains.identity("c", 0, 3, BLOCK)
    # Another place, fill or block size differs.
    assert chains.identity("a", 0, 3, BLOCK) != chains.identity("b", 0, 3, 2)
    assert chains.identity("a", 0, 3, 2) != chains.identity("a", 0, 3, 3)
    assert chains.identity("a", 0, 1, BLOCK) != chains.identity("a", 1, 0, BLOCK)
    # A prompt's partial last block: its tokens so far.
    assert chains.prompt_content("b", 0, 3) == chains.identity("b", 0, 3, 2)
    # Tokens not known yet are refused, not guessed.
    with pytest.raises(ValueError, match="tokens up to 20 are unknown"):
        chains.identity("a", 0, 4, BLOCK)
    chains.admit("e", None)
    with pytest.raises(ValueError, match="no prompt token ids"):
        chains.prompt_content("e", 0, 0)
    # The same identity seeds the same bytes on any layer it is given with.
    a = pattern("x", LAYERS[0], 2, 64, identity=chains.identity("a", 0, 2, BLOCK))
    b = pattern("y", LAYERS[0], 2, 64, identity=chains.identity("b", 0, 2, BLOCK))
    assert torch.equal(a, b)
    assert not torch.equal(a, pattern("x", LAYERS[1], 2, 64, identity=chains.identity("a", 0, 2, BLOCK)))
    assert not torch.equal(a, pattern("x", LAYERS[0], 2, 64, 1, chains.identity("a", 0, 2, BLOCK)))


def test_without_prefix_caching_the_pattern_is_still_the_requests():
    import hashlib

    import numpy as np

    seed = hashlib.blake2b(f"cmpl-abc-0|tp1|{LAYERS[0]}|3".encode(), digest_size=16).digest()
    want = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "little"))).integers(
        1, 256, 128, dtype=np.uint8
    )
    assert pattern(PREFILL_ID, LAYERS[0], 3, 128, 1).numpy().tolist() == want.tolist()


PROMPT_A = list(range(100, 110))  # 10 tokens: two full blocks and a partial one


def content_sides():
    chains_p, chains_d = TokenChains([BLOCK]), TokenChains([BLOCK])
    prefill, decode = block_views(caches(12), 12), block_views(caches(12), 12)
    filler = PatternFiller(prefill, chains=chains_p)
    checker = PatternChecker(decode, chains=chains_d)
    return prefill, decode, filler, checker


def holds(views, block, chains, req_id, logical, n):
    return all(
        torch.equal(
            view[block],
            pattern(req_id, layer, logical, view.shape[1], identity=chains.identity(req_id, 0, logical, n)),
        )
        for layer, view in views.items()
    )


def test_prefill_prefix_hits_are_compared_and_never_rewritten():
    prefill, _, filler, _ = content_sides()
    filler.after_update(step(new=[("a", [1, 2, 3], PROMPT_A, 0, 10)]))
    assert holds(prefill, 3, filler.chains, "a", 2, 2)
    assert filler.take_results() == []
    # B shares A's two full blocks (a prefix hit of 8 tokens).
    prompt_b = PROMPT_A[:8] + [7, 7, 7]
    before = {layer: view[[1, 2]].clone() for layer, view in prefill.items()}
    filler.after_update(step(new=[("b", [1, 2, 5], prompt_b, 8, 3)]))
    (hit,) = filler.take_results()
    assert hit["kind"] == "hit" and hit["ok"], hit
    assert hit["blocks"] == 2 and hit["groups"][0]["logical_blocks"] == [0, 1]
    assert all(torch.equal(view[[1, 2]], before[layer]) for layer, view in prefill.items())
    assert holds(prefill, 5, filler.chains, "b", 2, 3)
    # A corrupted cached page fails the next request's hit, unrepaired.
    prefill[LAYERS[1]][2][5] ^= 0xFF
    filler.after_update(step(new=[("c", [1, 2, 6], PROMPT_A[:8] + [1], 8, 1)]))
    (hit,) = filler.take_results()
    assert not hit["ok"] and hit["mismatched_bytes"] == 1
    assert hit["mismatched_blocks"] == [[LAYERS[1], 1, 2, 1]]


def test_a_pull_is_compared_when_its_request_runs():
    prefill, decode, filler, checker = content_sides()
    filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A, 0, 10)]))
    checker.before_step(pull([1, 2, 4], [3, 5, 6]))
    transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
    # Finished, but the prompt arrives with the request's first step.
    assert checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID})) == []
    checker.before_step(step(new=[(DECODE_ID, [1, 2, 4], PROMPT_A, 9, 1)]))
    (result,) = checker.after_step(None)
    assert result["ok"], result
    assert result["blocks"] == 3 and "kind" not in result


def test_a_pull_stops_where_the_producer_backed_off():
    """A producer of recurrent states (or a bounded window replay) computes
    the prompt but its last token: its last block names one token fewer than
    the consumer's prompt, and a checker told of that backoff expects it."""
    assert TokenChains([BLOCK]).backoff == 0
    for backoff, ok in ((1, True), (0, False)):
        prefill, decode, filler, _ = content_sides()
        chains = TokenChains([BLOCK], backoff=backoff)
        checker = PatternChecker(decode, chains=chains)
        filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A[:-1], 0, 9)]))
        assert holds(prefill, 6, filler.chains, PREFILL_ID, 2, 1)
        checker.before_step(pull([1, 2, 4], [3, 5, 6]))
        transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
        checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        checker.before_step(step(new=[(DECODE_ID, [1, 2, 4], PROMPT_A, 9, 1)]))
        (result,) = checker.after_step(None)
        assert result["ok"] is ok, result
        if not ok:
            # Only the last block names the token the producer left out.
            assert {block[1] for block in result["mismatched_blocks"]} == {2}
    # The content a pull holds: the prompt's, short of the backoff.
    chains = TokenChains([BLOCK], backoff=1)
    chains.admit("a", PROMPT_A)
    assert chains.pulled_content("a", 0, 2) == chains.identity("a", 0, 2, 1)
    assert chains.pulled_content("a", 0, 1) == chains.identity("a", 0, 1, BLOCK)
    assert chains.prompt_content("a", 0, 2) == chains.identity("a", 0, 2, 2)


def test_a_pull_of_another_prompt_fails():
    prefill, decode, filler, checker = content_sides()
    filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A, 0, 10)]))
    checker.before_step(pull([1, 2, 4], [3, 5, 6]))
    transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
    checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    checker.before_step(step(new=[(DECODE_ID, [1, 2, 4], PROMPT_A[:9] + [0], 9, 1)]))
    (result,) = checker.after_step(None)
    assert not result["ok"]
    # Only the last block holds the differing token.
    assert {block[1] for block in result["mismatched_blocks"]} == {2}


def test_a_page_completed_by_generation_is_reused_by_a_later_request():
    """Response 126's case: A's last pulled page is partial; decode's own
    steps complete it with a generated token, and B, whose prompt is A's
    prompt and that token and more, hits it. A page wholly generated is
    reused the same way."""
    prefill, decode, filler, checker = content_sides()
    filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A, 0, 10)]))
    checker.before_step(pull([1, 2, 4], [3, 5, 6]))
    transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
    checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    # Decode recomputes the last prompt token (N - 1), then generates.
    first = step(new=[(DECODE_ID, [1, 2, 4], PROMPT_A, 9, 1)])
    checker.before_step(first)
    checker.after_update(first)
    (pulled,) = checker.after_step(None)
    assert pulled["ok"]
    assert holds(decode, 4, checker.chains, DECODE_ID, 2, 2)  # not complete yet
    generated = [50, 51, 52, 53, 54, 55]
    computed = 10
    for token in generated:
        checker.sampled([DECODE_ID], [[token]])
        new_ids = [8] if computed == 12 else None
        s = step(cached=[(DECODE_ID, new_ids, computed, 1, False)])
        checker.before_step(s)
        checker.after_update(s)
        assert checker.after_step(None) == []
        computed += 1
    chain = PROMPT_A + generated
    # Page 2 (tokens 8-11, two generated) and page 3 (12-15, all generated).
    assert holds(decode, 4, checker.chains, DECODE_ID, 2, BLOCK)
    assert holds(decode, 8, checker.chains, DECODE_ID, 3, BLOCK)
    # B hits all four pages, generated ones included, without a pull.
    b = "cmpl-b-0-11111111"
    hit_step = step(new=[(b, [1, 2, 4, 8, 9], chain[:16] + [1, 2], 16, 2)])
    checker.before_step(hit_step)
    checker.after_update(hit_step)
    (hit,) = checker.after_step(None)
    assert hit["kind"] == "hit" and hit["ok"] and hit["blocks"] == 4, hit
    # The same page corrupted fails C's hit.
    decode[LAYERS[0]][8][0] ^= 0xFF
    c = "cmpl-c-0-22222222"
    c_step = step(new=[(c, [1, 2, 4, 8, 10], chain[:16] + [3], 16, 1)])
    checker.before_step(c_step)
    checker.after_update(c_step)
    (hit,) = checker.after_step(None)
    assert not hit["ok"] and hit["mismatched_blocks"][0][2] == 8


# Speculative decoding (MTP) with prefix caching: a step schedules its bonus
# token and k drafts; the drafts' tokens are known only once accepted.
PROMPT_S = [10, 11, 12, 13, 14, 15]  # six tokens: one full page and two more


def run_step(checker, s):
    checker.before_step(s)
    checker.after_update(s)


def spec_start(checker):
    """Decode computes A's six prompt tokens itself (no pull) and samples its
    bonus token at position 6; page 0 is written, page 1 not yet."""
    run_step(checker, step(new=[("a", [1, 7], PROMPT_S, 0, 6)]))
    checker.sampled(["a"], [[20]])
    checker.completed({"a": 6})
    assert [(e["phase"], e["logical_block"]) for e in checker.take_write_events()] == [("step", 0)]
    assert checker.after_step(None) == []


def zero(views, block):
    return all(int(view[block].abs().sum()) == 0 for view in views.values())


def test_accepted_drafts_complete_a_page_written_before_the_request_ends():
    """238's counterexample: A's step computes its bonus and two drafts
    (positions 6-8), both accepted, and A ends with that step. The page they
    complete (tokens 4-7) is written at acceptance, never from a draft
    before it, and a later request's prefix hit on it compares equal; the
    same page corrupted fails."""
    _, decode, _, checker = content_sides()
    spec_start(checker)
    run_step(checker, step(cached=[("a", [8], 6, 3, False)]))
    assert zero(decode, 7)  # drafts at 7-8: not written before acceptance
    checker.sampled(["a"], [[21, 22, 23]])  # two accepted drafts and a bonus
    checker.completed({"a": 9})  # computed 6 + 3, nothing rejected
    events = checker.take_write_events()
    assert [(e["phase"], e["logical_block"], e["block_id"]) for e in events] == [
        ("completion", 1, 7)
    ]
    chain = PROMPT_S + [20, 21, 22, 23]
    assert holds(decode, 7, checker.chains, "a", 1, BLOCK)
    run_step(checker, step(finished=["a"]))
    assert checker.after_step(None) == []
    hit_step = step(new=[("b", [1, 7, 9], chain[:8] + [5, 5], 8, 2)])
    run_step(checker, hit_step)
    (hit,) = checker.after_step(None)
    assert hit["kind"] == "hit" and hit["ok"] and hit["blocks"] == 2, hit
    decode[LAYERS[0]][7][0] ^= 0xFF
    run_step(checker, step(new=[("c", [1, 7, 10], chain[:8] + [6], 8, 1)]))
    (hit,) = checker.after_step(None)
    assert not hit["ok"] and hit["mismatched_blocks"][0][2] == 7


def test_rejected_drafts_write_nothing_and_their_page_waits_for_accepted_tokens():
    _, decode, _, checker = content_sides()
    spec_start(checker)
    run_step(checker, step(cached=[("a", [8], 6, 3, False)]))
    checker.sampled(["a"], [[21]])  # both drafts rejected, a new bonus
    checker.completed({"a": 7})  # computed 6 + 3 - 2 rejected
    assert checker.take_write_events() == [] and zero(decode, 7)
    # The next step computes the bonus at 7, completing page 1 with known tokens.
    run_step(checker, step(cached=[("a", None, 7, 3, False)]))
    events = checker.take_write_events()
    assert [(e["phase"], e["logical_block"]) for e in events] == [("step", 1)]
    assert holds(decode, 7, checker.chains, "a", 1, BLOCK)
    assert checker.after_step(None) == []


def test_a_page_ending_at_the_new_bonus_token_waits_for_its_kv():
    """A seven-token prompt: page 1 ends at position 7, the sampled bonus
    token, whose KV a later step computes."""
    _, decode, _, checker = content_sides()
    run_step(checker, step(new=[("a", [1, 7], PROMPT_S + [16], 0, 7)]))
    checker.sampled(["a"], [[20]])
    checker.completed({"a": 7})
    assert [e["logical_block"] for e in checker.take_write_events()] == [0]
    assert zero(decode, 7)
    run_step(checker, step(cached=[("a", None, 7, 1, False)]))
    assert [e["logical_block"] for e in checker.take_write_events()] == [1]
    assert holds(decode, 7, checker.chains, "a", 1, BLOCK)


def test_an_incomplete_prefill_writes_no_page_its_kv_does_not_hold():
    """239's case: a 20-token prompt of which a step computes 4 and samples
    nothing; its later pages stay untouched and its watermark at page 1."""
    _, decode, _, checker = content_sides()
    prompt = list(range(30, 50))
    run_step(checker, step(new=[("a", [1, 2, 3, 4, 5], prompt, 0, 4)]))
    checker.sampled(["a"], [[]])
    checker.completed({"a": 4})
    assert [e["logical_block"] for e in checker.take_write_events()] == [0]
    assert all(zero(decode, b) for b in (2, 3, 4, 5))
    assert checker.written["a"] == [1]
    assert checker.after_step(None) == []


def test_an_optimistic_counter_writes_nothing_past_the_known_tokens():
    """Async scheduling counts the previous step's drafts as computed before
    their rejection rolls it back: a scheduled range past the known tokens
    writes only what they complete."""
    _, decode, _, checker = content_sides()
    spec_start(checker)
    run_step(checker, step(cached=[("a", [8], 6, 3, False)]))
    checker.sampled(["a"], [[21]])
    checker.completed({"a": 7})
    # The scheduler still counts 9 computed; the chain knows 8 tokens.
    run_step(checker, step(cached=[("a", None, 9, 3, False)]))
    assert [e["logical_block"] for e in checker.take_write_events()] == [1]
    assert zero(decode, 8) and checker.written["a"] == [2]


def test_a_completion_at_the_workers_endpoint_leaves_the_bonus_page():
    """240's case at the checker: the scheduler counts 8 computed (two
    rejected drafts unresolved) against an actual 6; the step computes 6-8
    and rejects both drafts. The worker's endpoint, 7, leaves the page
    ending at the new bonus token (position 7) untouched; the mirror's
    arithmetic, 9, would write it. The runner's endpoint read is tested on a
    real runner (`test_real_kv_transfer.py`)."""
    _, decode, _, checker = content_sides()
    spec_start(checker)  # 6 computed, the bonus at 6 known
    run_step(checker, step(cached=[("a", [8], 8, 3, False)]))
    assert checker.take_write_events() == []
    checker.sampled(["a"], [[21]])  # both rejected; the new bonus at 7
    checker.completed({"a": 7})
    assert checker.take_write_events() == [] and zero(decode, 7)
    checker.completed({"a": 9})
    assert [e["logical_block"] for e in checker.take_write_events()] == [1]


def test_a_resumed_request_restarts_its_watermark_at_its_new_table():
    """Preempted and resumed with a new table and four computed tokens
    (recomputing from there): known tokens beyond the recomputed frontier
    write nothing until the worker computes them."""
    _, decode, _, checker = content_sides()
    spec_start(checker)
    run_step(checker, step(cached=[("a", [8], 6, 3, False)]))
    checker.sampled(["a"], [[21, 22, 23]])
    checker.completed({"a": 9})
    checker.take_write_events()
    # Resumed: blocks 1 (its hit page 0), 10, 11; two tokens recomputed.
    run_step(checker, step(cached=[("a", [1, 10, 11], 4, 2, True)]))
    (hit,) = checker.after_step(None)
    assert hit["kind"] == "hit" and hit["ok"]
    assert checker.written["a"] == [1] and checker.take_write_events() == []
    checker.sampled(["a"], [[]])
    checker.completed({"a": 6})
    assert checker.take_write_events() == [] and zero(decode, 10)


def test_a_decode_hit_before_a_pull_is_checked_at_its_place():
    prefill, decode, filler, checker = content_sides()
    filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A, 0, 10)]))
    # Decode holds logical block 0 from an earlier request of that prefix.
    checker.chains.admit("earlier", PROMPT_A[:4] + [0])
    checker._write_block("earlier", 0, 0, 7, checker.chains.identity("earlier", 0, 0, BLOCK))
    checker.before_step(pull([2, 4], [5, 6], computed=(1,)))
    transfer(prefill, decode, [(2, 5), (4, 6)])
    checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    first = step(new=[(DECODE_ID, [7, 2, 4], PROMPT_A, 9, 1)])
    checker.before_step(first)
    checker.after_update(first)
    pulled, hit = checker.after_step(None)
    assert hit["kind"] == "hit" and hit["ok"] and hit["groups"][0]["block_ids"] == [7]
    assert pulled["ok"] and pulled["blocks"] == 2


def test_a_prefill_hit_on_a_page_filled_in_the_same_step_is_compared_after_it():
    """Native prefix caching shares a block as soon as it schedules the
    tokens filling it: B, listed first, hits A's pages of this very step."""
    prefill, _, filler, _ = content_sides()
    prompt_b = PROMPT_A[:8] + [7, 7]
    filler.after_update(step(new=[("b", [1, 2, 5], prompt_b, 8, 2), ("a", [1, 2, 3], PROMPT_A, 0, 10)]))
    (hit,) = filler.take_results()
    assert hit["kind"] == "hit" and hit["ok"], hit
    assert hit["groups"][0]["written_this_step"] == [[1, "a"], [2, "a"]]
    assert holds(prefill, 5, filler.chains, "b", 2, 2)


def test_a_decode_page_completed_and_adopted_in_the_same_step():
    """A's step completes its page 2 (the received partial page and a
    generated token) while B, scheduled in that step, adopts it."""
    prefill, decode, filler, checker = content_sides()
    filler.after_update(step(new=[(PREFILL_ID, [3, 5, 6], PROMPT_A, 0, 10)]))
    checker.before_step(pull([1, 2, 4], [3, 5, 6]))
    transfer(prefill, decode, [(1, 3), (2, 5), (4, 6)])
    checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    first = step(new=[(DECODE_ID, [1, 2, 4], PROMPT_A, 9, 1)])
    checker.before_step(first)
    checker.after_update(first)
    assert checker.after_step(None)[0]["ok"]
    checker.sampled([DECODE_ID], [[50]])
    checker.sampled([DECODE_ID], [[51]])  # page 2's last token, sampled, not computed
    s = step(cached=[(DECODE_ID, None, 10, 1, False)])
    checker.before_step(s)
    checker.after_update(s)
    # Token 11 (sampled 51) completes page 2 as B adopts it.
    b = "cmpl-b-0-11111111"
    together = step(cached=[(DECODE_ID, [8], 11, 1, False)],
                    new=[(b, [1, 2, 4, 9], PROMPT_A + [50, 51, 9, 9], 12, 2)])
    checker.before_step(together)
    checker.after_update(together)
    (hit,) = checker.after_step(None)
    assert hit["kind"] == "hit" and hit["ok"], hit
    assert hit["groups"][0]["logical_blocks"] == [0, 1, 2]
    assert hit["groups"][0]["written_this_step"] == [[4, DECODE_ID]]


def test_a_corrupted_ready_hit_fails_though_the_step_writes_others():
    """Comparing after the step's writes cannot mask a page that was ready
    before it: only its writer of the step writes a page, never a reader."""
    prefill, _, filler, _ = content_sides()
    filler.after_update(step(new=[("a", [1, 2, 3], PROMPT_A, 0, 10)]))
    prefill[LAYERS[0]][1][0] ^= 0xFF
    filler.after_update(step(new=[("b", [1, 2, 5], PROMPT_A[:8] + [7], 8, 1),
                                  ("c", [6, 7], [1, 2, 3, 4, 5], 0, 5)]))
    (hit,) = filler.take_results()
    assert not hit["ok"] and hit["mismatched_blocks"][0][2] == 1
    assert hit["groups"][0]["written_this_step"] == []


def test_a_partial_prefix_hit_fails_closed():
    """A hit ending inside a block (copy on write) needs an aligned Mamba
    group, which prefix caching here excludes: refused, not checked."""
    _, _, filler, _ = content_sides()
    filler.after_update(step(new=[("a", [1, 2, 3], PROMPT_A, 0, 10)]))
    filler.after_update(step(new=[("b", [1, 4, 5], PROMPT_A[:6] + [0] * 4, 6, 4)]))
    failure, hit = filler.take_results()
    assert not failure["ok"] and "ends inside a block" in failure["error"]
    assert hit["ok"] and hit["blocks"] == 1


def test_a_page_is_never_written_while_its_pull_waits():
    _, _, _, checker = content_sides()
    checker.before_step(pull([1, 2], [3, 5]))
    checker.chains.admit("other", PROMPT_A)
    checker.tables["other"] = [[2, 6, 7]]
    with pytest.raises(RuntimeError, match="not compared yet"):
        checker.after_update(step(cached=[("other", None, 3, 2, False)]))


def test_requests_outside_the_lifecycle_fail_closed():
    _, _, filler, _ = content_sides()
    filler.after_update(step(new=[("embeds", [1, 2], None, 0, 8)]))
    (result,) = filler.take_results()
    assert not result["ok"] and result["kind"] == "fill"
    assert "outside the prefix-caching pattern lifecycle" in result["error"]


@pytest.mark.parametrize(
    "request_kwargs, refused",
    [
        ({"lora_request": object()}, "a LoRA adapter"),
        ({"cache_salt": "s"}, "a cache_salt"),
        ({"prompt_embeds": torch.zeros(1)}, "prompt embeddings"),
        ({"mm_features": [object()]}, "multimodal inputs"),
    ],
)
def test_pattern_checks_with_prefix_caching_take_token_prompts_only(request_kwargs, refused):
    from vllm.config.simulation import SimulationConfig

    sim = SimulationConfig(kv_transfer_mode="real", kv_pattern_check=True)
    plain = {"lora_request": None, "cache_salt": None, "prompt_embeds": None, "mm_features": None}
    sim.check_pattern_request(True, **plain)
    sim.check_pattern_request(False, **(plain | request_kwargs))
    with pytest.raises(ValueError, match=refused):
        sim.check_pattern_request(True, **(plain | request_kwargs))


# The same lifecycle fed by a native scheduler with prefix caching.


def native_requests(prompts, max_tokens=8):
    from tests.v1.core.utils import create_requests
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher
    from vllm.v1.request import Request

    # Initializes the block hashes' seed; its sampling parameters are reused.
    (template,) = create_requests(1, block_size=BLOCK, max_tokens=max_tokens)
    hasher = get_request_block_hasher(BLOCK, sha256)
    return [
        Request(
            request_id=req_id,
            prompt_token_ids=prompt,
            sampling_params=template.sampling_params,
            pooling_params=None,
            block_hasher=hasher,
        )
        for req_id, prompt in prompts.items()
    ]


def native_output(scheduler_output, tokens):
    """A runner's output sampling `tokens[req_id]` for each scheduled request."""
    from vllm.v1.outputs import ModelRunnerOutput

    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[tokens[req_id]] for req_id in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_a_native_same_step_prefill_dedup_is_compared_after_its_writer():
    """The native scheduler shares A's pages with B in the step that computes
    them (`test_prefix_caching_for_prefill_dedup`): B's hit is compared after
    A's writes, read-only, A named as their writer."""
    from tests.v1.core.utils import create_scheduler

    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK, num_blocks=32)
    a, b = native_requests({"a": PROMPT_A, "b": PROMPT_A[:8] + [7, 7]})
    scheduler.add_request(a)
    scheduler.add_request(b)
    out = scheduler.schedule()
    new = {r.req_id: r for r in out.scheduled_new_reqs}
    assert out.num_scheduled_tokens == {"a": 10, "b": 2}
    assert new["b"].num_computed_tokens == 8
    shared = new["a"].block_ids[0][:2]
    assert new["b"].block_ids[0][:2] == shared
    prefill = block_views(caches(32), 32)
    filler = PatternFiller(prefill, chains=TokenChains([BLOCK]))
    filler.after_update(out)
    (hit,) = filler.take_results()
    assert hit["req_id"] == "b" and hit["kind"] == "hit" and hit["ok"], hit
    assert hit["groups"][0]["block_ids"] == shared
    assert hit["groups"][0]["written_this_step"] == [[shared[0], "a"], [shared[1], "a"]]
    # B's own page holds its content, A's pages A's.
    assert holds(prefill, new["b"].block_ids[0][2], filler.chains, "b", 2, 2)
    assert holds(prefill, shared[1], filler.chains, "a", 1, BLOCK)


def test_a_native_same_step_decode_adoption_is_compared_after_its_writer():
    """A's decode step computes the last token of its page 2 (two of its
    tokens generated); the native scheduler caches that page as it schedules
    the token, and B, admitted in the same step, hits it. The checker writes
    the page, then compares B's hit."""
    from tests.v1.core.utils import create_scheduler

    scheduler = create_scheduler(enable_prefix_caching=True, block_size=BLOCK, num_blocks=32)
    (a,) = native_requests({"a": PROMPT_A})
    scheduler.add_request(a)
    decode = block_views(caches(32), 32)
    checker = PatternChecker(decode, chains=TokenChains([BLOCK]))

    def run(out, tokens):
        checker.before_step(out)
        checker.after_update(out)
        results = checker.after_step(None)
        req_ids = list(out.num_scheduled_tokens)
        checker.sampled(req_ids, [[tokens[r]] for r in req_ids])
        scheduler.update_from_output(out, native_output(out, tokens))
        return results

    first = scheduler.schedule()
    a_blocks = first.scheduled_new_reqs[0].block_ids[0]
    assert run(first, {"a": 50}) == []
    assert run(scheduler.schedule(), {"a": 51}) == []
    (b,) = native_requests({"b": PROMPT_A + [50, 51, 1, 2]})
    scheduler.add_request(b)
    third = scheduler.schedule()
    (new_b,) = third.scheduled_new_reqs
    # A computes token 11, completing page 2; B hits it in that step.
    assert third.scheduled_cached_reqs.num_computed_tokens == [11]
    assert new_b.num_computed_tokens == 12 and new_b.block_ids[0][:3] == a_blocks[:3]
    (hit,) = run(third, {"a": 52, "b": 60})
    assert hit["req_id"] == "b" and hit["kind"] == "hit" and hit["ok"], hit
    assert hit["groups"][0]["written_this_step"] == [[a_blocks[2], "a"]]
    assert holds(decode, a_blocks[2], checker.chains, "a", 2, BLOCK)


# Unequal TP: each attention layer's page is its global KV heads' patterns.


def attention_specs(kv_heads_per_rank, layers=LAYERS, padded=None):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    # 4 tokens a block, head size 4 in uint8: a head's slab is 4 x 8 = 32 bytes.
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=kv_heads_per_rank,
        head_size=4,
        dtype=torch.uint8,
        page_size_padded=padded,
    )
    return dict.fromkeys(layers, spec)


SLAB = 32


def test_global_heads_follow_vllm_sharding_layer_by_layer():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.simulated.kv_pattern import head_map

    # K = 4 KV heads: TP2 ranks hold heads 0-1 and 2-3, TP4 one each, TP8
    # one each, replicated on ranks 2h and 2h + 1.
    assert head_map(attention_specs(2), 4, 1, 2)[LAYERS[0]] == ((2, 3), SLAB)
    assert head_map(attention_specs(1), 4, 3, 4)[LAYERS[0]] == ((3,), SLAB)
    assert [head_map(attention_specs(1), 4, r, 8)[LAYERS[0]][0] for r in range(8)] == [
        (h,) for h in (0, 0, 1, 1, 2, 2, 3, 3)
    ]
    # A layer whose local heads do not match that sharding keeps its rank's.
    assert head_map(attention_specs(1), 4, 0, 2) == {}
    # MLA: one latent, the same on every rank.
    mla = {"mla": MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=16, dtype=torch.uint8)}
    assert head_map(mla, 1, 3, 4) == {"mla": ((0,), 64)}
    # A packed latent (FlashMLA's FP8 one) keeps its rank's pattern, but a
    # sparse-MLA model's: whole on every rank, as the dense one.
    packed = {
        "mla": MLAAttentionSpec(
            block_size=4, num_kv_heads=1, head_size=16, dtype=torch.uint8,
            state_content_bytes=41,
        )
    }
    assert head_map(packed, 1, 3, 4) == {}
    assert head_map(packed, 1, 3, 4, packed_latent=True) == {"mla": ((0,), 4 * 41)}
    # A recurrent state keeps its rank's pattern.
    from vllm.v1.kv_cache_interface import MambaSpec

    state = {"ssm": MambaSpec(block_size=4, shapes=((4,),), dtypes=(torch.uint8,))}
    assert head_map(state, 4, 0, 2) == {}


def test_a_heads_bytes_do_not_depend_on_tp_size_page_or_rank():
    from vllm.v1.worker.simulated.kv_pattern import head_map, head_pattern

    def page(tp, rank, padded=None):
        specs = attention_specs(max(1, 4 // tp), padded=padded)
        heads = head_map(specs, 4, rank, tp)[LAYERS[0]]
        size = padded or len(heads[0]) * SLAB
        return heads, head_pattern(PREFILL_ID, LAYERS[0], 3, size, heads, rank)

    (_, tp2_rank0), (_, tp2_rank1) = page(2, 0), page(2, 1)
    # TP4 rank 1 holds head 1: the second slab of TP2 rank 0's page.
    assert torch.equal(page(4, 1)[1], tp2_rank0[SLAB : 2 * SLAB])
    assert torch.equal(page(4, 2)[1], tp2_rank1[:SLAB])
    # TP8 ranks 2 and 3 replicate head 1: the same bytes.
    assert torch.equal(page(8, 2)[1], page(8, 3)[1])
    assert torch.equal(page(8, 2)[1], tp2_rank0[SLAB : 2 * SLAB])
    # Another request, layer or block differs.
    heads = page(4, 1)[0]
    assert not torch.equal(head_pattern("cmpl-other-0-1", LAYERS[0], 3, SLAB, heads), page(4, 1)[1])
    assert not torch.equal(head_pattern(PREFILL_ID, LAYERS[1], 3, SLAB, heads), page(4, 1)[1])
    assert not torch.equal(head_pattern(PREFILL_ID, LAYERS[0], 4, SLAB, heads), page(4, 1)[1])
    # A padded page: its heads' slabs, then its rank's padding.
    padded_heads, padded = page(2, 0, padded=80)
    assert torch.equal(padded[: 2 * SLAB], tp2_rank0)
    assert not torch.equal(padded[2 * SLAB :], page(2, 1, padded=80)[1][2 * SLAB :])
    with pytest.raises(ValueError, match="exceed"):
        head_pattern(PREFILL_ID, LAYERS[0], 3, SLAB, padded_heads)


def tp_sides(prefill_tp, decode_tp, kv_heads=4, num_blocks=8):
    """Each prefill rank's filled pages (blocks 3, 5 of a 2-block request)
    and each decode rank's empty pages, under head patterns."""
    from vllm.v1.worker.simulated.kv_pattern import head_map

    def side(tp):
        ranks = []
        for rank in range(tp):
            heads = head_map(attention_specs(max(1, kv_heads // tp)), kv_heads, rank, tp)
            page = len(next(iter(heads.values()))[0]) * SLAB
            views = {layer: torch.zeros(num_blocks, page, dtype=torch.uint8) for layer in LAYERS}
            ranks.append((views, heads))
        return ranks

    prefill = side(prefill_tp)
    for rank, (views, heads) in enumerate(prefill):
        PatternFiller(views, rank, heads=heads).after_update(
            new_request_step(PREFILL_ID, [3, 5])
        )
    return prefill, side(decode_tp)


def mla_sides(prefill_tp, decode_tp, num_blocks=8):
    """`tp_sides` for MLA: one latent of SLAB bytes a block, the same on
    every rank (global head 0), filled on every prefill rank."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.simulated.kv_pattern import head_map

    spec = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=8, dtype=torch.uint8)
    specs = dict.fromkeys(LAYERS, spec)

    def side(tp):
        return [
            (
                {layer: torch.zeros(num_blocks, SLAB, dtype=torch.uint8) for layer in LAYERS},
                head_map(specs, 1, rank, tp),
            )
            for rank in range(tp)
        ]

    prefill = side(prefill_tp)
    for rank, (views, heads) in enumerate(prefill):
        PatternFiller(views, rank, heads=heads).after_update(new_request_step(PREFILL_ID, [3, 5]))
    return prefill, side(decode_tp)


def test_an_mla_latent_is_the_same_page_at_any_tp_size():
    """MLA: every rank of any TP size holds the same latent (global head 0),
    so a decode rank of TP8 checks a TP4 producer's page byte for byte, and
    any producer's would do."""
    prefill, decode = mla_sides(4, 8)
    assert all(torch.equal(prefill[0][0][l], prefill[r][0][l]) for r in range(4) for l in LAYERS)
    for rank in (0, 5):
        checker = checker_for(decode, rank)
        checker.before_step(recv_step([1, 2], [3, 5]))
        copy_slice(prefill[rank * 4 // 8][0], decode[rank][0], 0, 0, SLAB)
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"], result
        assert result["groups"][0]["global_heads"] == [0]
        assert result["groups"][0]["pattern"] == "head-v1"


def checker_for(decode, rank):
    views, heads = decode[rank]
    return PatternChecker(views, rank, heads=heads)


def copy_slice(prefill_views, decode_views, src_offset, dst_offset, length, pairs=((1, 3), (2, 5))):
    for layer in LAYERS:
        for d_block, p_block in pairs:
            decode_views[layer][d_block][dst_offset : dst_offset + length] = prefill_views[layer][
                p_block
            ][src_offset : src_offset + length]


def test_decode_ranks_check_head_slices_of_a_larger_producer_page():
    """P TP2 -> D TP4: decode rank d holds head d, bytes (d % 2) x slab of
    prefill rank d // 2's page; the wrong slice fails."""
    prefill, decode = tp_sides(2, 4)
    for rank in range(4):
        checker = checker_for(decode, rank)
        checker.before_step(recv_step([1, 2], [3, 5]))
        copy_slice(prefill[rank // 2][0], decode[rank][0], (rank % 2) * SLAB, 0, SLAB)
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"], result
        assert result["groups"][0]["global_heads"] == [rank]
        assert result["groups"][0]["pattern"] == "head-v1"
    checker = checker_for(decode, 1)
    checker.before_step(recv_step([1, 2], [3, 5]))
    copy_slice(prefill[0][0], decode[1][0], 0, 0, SLAB)  # head 0, not 1
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]


def test_a_decode_page_assembled_from_producers_is_compared_when_all_arrived():
    """P TP4 -> D TP2: decode rank d's page is prefill ranks 2d and 2d + 1's
    pages side by side. Compared only once the connector reports the request
    finished (after its last source); reported finished early, the missing
    part (still the sentinel) fails it."""
    prefill, decode = tp_sides(4, 2)
    checker = checker_for(decode, 1)
    checker.before_step(recv_step([1, 2], [3, 5]))
    copy_slice(prefill[2][0], decode[1][0], 0, 0, SLAB)
    # The last source is delayed: nothing is compared.
    assert checker.after_step(KVConnectorOutput()) == []
    copy_slice(prefill[3][0], decode[1][0], 0, SLAB, SLAB)
    (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert result["ok"] and result["groups"][0]["global_heads"] == [2, 3], result
    early = checker_for(decode, 0)
    early.before_step(recv_step([1, 2], [3, 5]))
    copy_slice(prefill[0][0], decode[0][0], 0, 0, SLAB)
    (result,) = early.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
    assert not result["ok"]
    assert result["mismatched_bytes"] == 2 * len(LAYERS) * SLAB


def test_replicated_heads_carry_the_same_bytes_on_either_side():
    """K = 4. P TP2 -> D TP8: decode ranks 2 and 3 both hold head 1, the
    second slab of prefill rank 0's page. P TP8 -> D TP2: decode rank 0's
    heads 0 and 1 come from producer ranks 0 and 2 (their replicas 1 and 3
    write nothing), or equally from 1 and 3."""
    prefill, decode = tp_sides(2, 8)
    for rank in (2, 3):
        checker = checker_for(decode, rank)
        checker.before_step(recv_step([1, 2], [3, 5]))
        copy_slice(prefill[0][0], decode[rank][0], SLAB, 0, SLAB)
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"] and result["groups"][0]["global_heads"] == [1], result
    prefill, decode = tp_sides(8, 2)
    for sources in ((0, 2), (1, 3)):
        checker = checker_for(decode, 0)
        checker.before_step(recv_step([1, 2], [3, 5]))
        for slot, source in enumerate(sources):
            copy_slice(prefill[source][0], decode[0][0], 0, slot * SLAB, SLAB)
        (result,) = checker.after_step(KVConnectorOutput(finished_recving={DECODE_ID}))
        assert result["ok"], (sources, result)


def test_a_delayed_decode_worker_holds_the_request():
    """vLLM's aggregation across decode workers: a request finishes receiving
    only when every worker reported it."""
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.v1.outputs import ModelRunnerOutput

    def output(finished):
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            kv_connector_output=KVConnectorOutput(finished_recving=finished),
        )

    aggregator = KVOutputAggregator(expected_finished_count=2)
    first = aggregator.aggregate([output({DECODE_ID}), output(set())])
    assert not first.kv_connector_output.finished_recving
    second = aggregator.aggregate([output(set()), output({DECODE_ID})])
    assert second.kv_connector_output.finished_recving == {DECODE_ID}


def test_nixl_reads_record_their_producer_rank_and_slot():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
        NixlKVConnectorStats,
    )
    from vllm.v1.worker.simulated.kv_transfer import NixlReadPeers, completed_transfers

    telemetry = {
        101: SimpleNamespace(xferDuration=1000, postDuration=10, totalBytes=64, descCount=2),
        102: SimpleNamespace(xferDuration=1000, postDuration=10, totalBytes=64, descCount=2),
    }
    handles = iter((101, 102))
    wrapper = SimpleNamespace(
        make_prepped_xfer=lambda *args, **kw: next(handles),
        get_xfer_telemetry=lambda handle: telemetry[handle],
    )
    worker = SimpleNamespace(
        nixl_wrapper=wrapper,
        xfer_stats=NixlKVConnectorStats(),
        dst_xfer_side_handles={"prefill": {2: 7002, 3: 7003}},
        src_xfer_handles_by_tp_ratio={(-2, 16): [9000, 9001]},
    )
    NixlReadPeers(worker)
    # Decode rank 1 of TP2 reads producer ranks 2 and 3 into slots 0 and 1.
    first = wrapper.make_prepped_xfer("READ", 9000, [0, 1], 7002, [4, 5], b"req:1")
    second = wrapper.make_prepped_xfer("READ", 9001, [0, 1], 7003, [4, 5], b"req:1")
    # Completed in the other order: each keeps its own.
    for handle in (second, first):
        worker.xfer_stats.record_transfer(wrapper.get_xfer_telemetry(handle))
    each = completed_transfers(KVConnectorOutput(kv_connector_stats=worker.xfer_stats))["load"][
        "each"
    ]
    # No prepared runs known for these stand-in lists: no prediction.
    assert [[entry[5], entry[6][:3]] for entry in each] == [[3, [1, 2, None]], [2, [0, 2, None]]]
    assert [entry[6][3:] for entry in each] == [[None, None], [None, None]]


def test_a_layer_with_its_own_total_heads_is_sharded_by_them():
    """Gemma4: sliding layers of 8 KV heads beside global layers of 2 (the
    model's `per_layer_config`). Each layer's global heads follow its own
    total: at TP 2, rank 1 holds sliding heads 4-7 and global head 1; at TP 4
    the global layers replicate (ranks 2 and 3 both hold head 1). Under the
    model's total of 8 a global layer's heads would not match its sharding
    and the layer would keep the rank's pattern."""
    from types import SimpleNamespace

    import torch

    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.simulated.kv_pattern import head_map, layer_total_heads

    def spec(local, head_size):
        return FullAttentionSpec(block_size=4, num_kv_heads=local, head_size=head_size, dtype=torch.uint8)

    per_layer = [SimpleNamespace(num_key_value_heads=8), SimpleNamespace(num_key_value_heads=2)]
    model = SimpleNamespace(hf_text_config=SimpleNamespace(per_layer_config=per_layer))
    names = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    totals = layer_total_heads(model, names)
    assert totals == {names[0]: 8, names[1]: 2}
    assert layer_total_heads(SimpleNamespace(hf_text_config=SimpleNamespace()), names) == {}
    for tp, rank, local, want in (
        (2, 1, (4, 1), ((4, 5, 6, 7), (1,))),
        (1, 0, (8, 2), (tuple(range(8)), (0, 1))),
        (4, 2, (2, 1), ((4, 5), (1,))),
        (4, 3, (2, 1), ((6, 7), (1,))),
    ):
        specs = {names[0]: spec(local[0], 8), names[1]: spec(local[1], 16)}
        heads = head_map(specs, 8, rank, tp, totals)
        assert (heads[names[0]][0], heads[names[1]][0]) == want, (tp, rank)
        # The model's total alone leaves the global layer's pattern the rank's.
        assert names[1] not in head_map(specs, 8, rank, tp) or tp == 1


def _stub_selection(rows, groups=1, block_ids=(1,)):
    """A native selection's surface over given spans: `rows` are
    [num_blocks, size] byte tensors, one a region, each group moving block b
    of every row."""
    from types import SimpleNamespace

    def select(group, blocks):
        return [(r, b, i) for i, b in enumerate(blocks) for r in range(len(rows))]

    return SimpleNamespace(
        rows=[SimpleNamespace(view=v, region=i, offset=0, state=False) for i, v in enumerate(rows)],
        select=select,
        members=[set(range(len(rows))) for _ in range(groups)],
        padding={},
        groups=[["l"] for _ in range(groups)],
        name=lambda group, row: f"g{group}.r{row}.o0",
        state_regions=lambda selected: set(),
        regions=lambda selected: {r for r, _, _ in selected},
        order=lambda blocks: [b for ids in blocks.values() for b in ids],
        worker=None,
    )


def test_native_head_patterns_check_reads_from_any_tp_size():
    """A head-aware NIXL selection (several attention groups) is checked from
    unequal TP sizes: a decode page of 4 KV heads' ranks reads its heads'
    slabs from whichever producers hold them. K = 4 heads of 8 bytes, 2
    regions. P TP2 -> D TP4: decode rank 1 (head 1) takes it from producer 0
    (heads 0, 1); P TP4 -> D TP2: decode rank 1 (heads 2, 3) gathers
    producers 2 and 3. A wrong head's slice, or a missing one, fails."""
    from types import SimpleNamespace

    from vllm.v1.worker.simulated.kv_pattern import (
        NativeChecker,
        NativeFiller,
        row_pattern,
    )

    slab, regions, blocks = 8, 2, 4

    def side(heads):
        return _stub_selection([torch.zeros(blocks, slab * len(heads), dtype=torch.uint8) for _ in range(regions)])

    def producer(heads):
        s = side(heads)
        NativeFiller(s, 0, heads=heads)._write_block("cmpl-a-0", 0, 3, 2, None)
        return s

    def consumer(heads, copy, reads=1):
        s = side(heads)
        for r in range(regions):
            s.rows[r].view[2] = copy(r)
        checker = NativeChecker(s, tp_rank=1, heads=heads)
        checker._check_geometry = lambda: None
        # One READ a source posts the checked blocks.
        checker.posted["cmpl-a-0"] = [2] * reads
        checker.reads["cmpl-a-0"] = [[2] for _ in range(reads)]
        expected = SimpleNamespace(
            remote_request_id="cmpl-a-0",
            block_ids=[[2]],
            groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=3)],
        )
        return checker._compare("cmpl-b-0", expected)

    def harness_accepts(result, mine):
        """What the e2e harness reads of every transfer check (pd.py's
        `_hetero_tp`): each group's pattern is head-v1 and its global heads
        the decode rank's own."""
        assert result["groups"]
        for group in result["groups"]:
            assert group["pattern"] == "head-v1" and group["global_heads"] == mine, group

    # Split: decode head 1 from producer 0's heads (0, 1), at slab 1.
    p = producer((0, 1))
    ok = consumer((1,), lambda r: p.rows[r].view[2][slab:2 * slab])
    assert ok["ok"], ok
    harness_accepts(ok, [1])
    wrong = consumer((1,), lambda r: p.rows[r].view[2][:slab])  # head 0's slice
    assert not wrong["ok"] and wrong["mismatched_bytes"] > 0
    missing = consumer((1,), lambda r: torch.zeros(slab, dtype=torch.uint8))
    assert not missing["ok"]
    # Gather: decode heads (2, 3) from producers holding head 2 and head 3.
    p2, p3 = producer((2,)), producer((3,))
    ok = consumer((2, 3), lambda r: torch.cat([p2.rows[r].view[2], p3.rows[r].view[2]]), reads=2)
    assert ok["ok"] and ok["posted_matches_checked"], ok
    harness_accepts(ok, [2, 3])
    swapped = consumer((2, 3), lambda r: torch.cat([p3.rows[r].view[2], p2.rows[r].view[2]]), reads=2)
    assert not swapped["ok"]
    # Truthful fallbacks: rank patterns report rank-v0 and no heads; a group
    # holding a state's span under head patterns is mixed.
    for heads, state, want in ((None, False, ("rank-v0", None)), ((1,), True, ("mixed", None))):
        s = side((1,))
        s.rows[0].state = state
        NativeFiller(s, 1, heads=heads)._write_block("cmpl-a-0", 0, 3, 2, None)
        checker = NativeChecker(s, tp_rank=1, heads=heads)
        checker._check_geometry = lambda: None
        checker.posted["cmpl-a-0"] = [2]
        checker.reads["cmpl-a-0"] = [[2]]
        expected = SimpleNamespace(
            remote_request_id="cmpl-a-0", block_ids=[[2]],
            groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=3)],
        )
        (group,) = checker._compare("cmpl-b-0", expected)["groups"]
        assert (group["pattern"], group["global_heads"]) == want
    # Only the head patterns are TP-independent: the rank's are not.
    assert not torch.equal(
        row_pattern("cmpl-a-0", "g0.r0.o0", 3, slab, 0, None, None),
        row_pattern("cmpl-a-0", "g0.r0.o0", 3, slab, 1, None, None),
    )


def test_a_block_gathered_from_smaller_producer_blocks_is_their_heads_slabs():
    """NIXL's block-size ratio R: a consumer block of R producer blocks
    holds, head by head, the producer blocks' slabs in token order, and
    zeros where the producer had no block (NIXL zeroes the uncovered tail)."""
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    heads, slab = (0, 1), 8
    producer = [row_pattern("cmpl-a-0", "L", b, 2 * slab, 0, None, heads) for b in (4, 5)]
    gathered = row_pattern("cmpl-a-0", "L", 2, 4 * slab, 0, None, heads, (2, 2))
    for h in range(2):
        assert torch.equal(
            gathered[h * 2 * slab:(h + 1) * 2 * slab],
            torch.cat([producer[0][h * slab:(h + 1) * slab], producer[1][h * slab:(h + 1) * slab]]),
        )
    tail = row_pattern("cmpl-a-0", "L", 2, 4 * slab, 0, None, heads, (2, 1))
    assert torch.equal(tail[:slab], producer[0][:slab]) and not tail[slab:2 * slab].any()
    assert torch.equal(row_pattern("cmpl-a-0", "L", 2, 4 * slab, 0, None, heads, (1, 1)),
                       row_pattern("cmpl-a-0", "L", 2, 4 * slab, 0, None, heads))
    with pytest.raises(ValueError, match="needs head patterns"):
        row_pattern("cmpl-a-0", "L", 2, 4 * slab, 0, None, None, (2, 2))
    # The sub-blocks NIXL puts in a group's i-th block: b*R + k, clipped to
    # the producer's blocks.
    group = SimpleNamespace(remote_blocks=3)
    assert [NativeChecker._sub_blocks(2, group, i) for i in range(3)] == [(2, 2), (2, 1), (2, 0)]
    assert NativeChecker._sub_blocks(1, group, 5) == (1, 1)


def test_native_geometry_check_admits_other_tp_only_with_head_patterns():
    """`NativeChecker._check_geometry`: a split or gathered read (another
    source rank, several consumers, split local handles) is refused for rank
    patterns, as NIXL's selection is the local one, and admitted for head
    patterns; a remote of another block size is refused for rank patterns,
    and for head patterns unless it divides this instance's (NIXL's
    block-size ratio) without prefix caching."""
    from types import SimpleNamespace

    import pytest

    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    def worker(block_sizes=(16,), split=None):
        plan = SimpleNamespace(all_source_ranks=(0, 1), local_consumers=2)
        return SimpleNamespace(
            tp_rank=1, engine_id="d", block_size=16,
            tp_mappings={"d": SimpleNamespace(all_source_ranks=(1,), local_consumers=1), "p": plan},
            src_xfer_handles_by_block_size=dict.fromkeys(block_sizes, 1),
            src_xfer_handles_by_tp_ratio=split or {},
        )

    def checker(w, heads, chains=None):
        return NativeChecker(SimpleNamespace(worker=w), heads=heads, chains=chains)

    with pytest.raises(ValueError, match="equal TP and block size"):
        checker(worker(split={(-2, 16): [1, 2]}), None)._check_geometry()
    checker(worker(split={(-2, 16): [1, 2]}), (2, 3))._check_geometry()
    for heads in (None, (2, 3)):
        with pytest.raises(ValueError, match="block size"):
            checker(worker(block_sizes=(16, 32)), heads)._check_geometry()
    with pytest.raises(ValueError, match="block size"):
        checker(worker(block_sizes=(16, 8)), None)._check_geometry()
    checker(worker(block_sizes=(16, 8)), (2, 3))._check_geometry()
    with pytest.raises(ValueError, match="without prefix caching"):
        checker(worker(block_sizes=(16, 8)), (2, 3), chains=object())._check_geometry()


def test_native_group_heads_decide_the_selections_eligibility():
    """A native NIXL selection is head-aware only when the model holds no
    state and every transfer group's layers hold one head set: from the
    model's heads at this TP size, never the pattern flags. A group lacking
    heads (a layer without a head pattern, or layers of different heads), or
    a hybrid, keeps the rank's patterns."""
    from vllm.v1.worker.simulated.kv_pattern import native_group_heads

    heads = {"a0": ((0, 1, 2, 3), 8), "a1": ((0, 1, 2, 3), 8), "b0": ((0,), 32), "b1": ((0,), 32)}
    groups = [["a0", "a1"], ["b0", "b1"]]
    assert native_group_heads(heads, groups, False) == ((0, 1, 2, 3), (0,))
    assert native_group_heads(heads, groups, True) is None
    assert native_group_heads(None, groups, False) is None
    assert native_group_heads({}, groups, False) is None
    assert native_group_heads(heads, [], False) is None
    # A layer without a head pattern, or two head sets in one group.
    assert native_group_heads({k: v for k, v in heads.items() if k != "b1"}, groups, False) is None
    assert native_group_heads({**heads, "a1": ((4, 5, 6, 7), 8)}, groups, False) is None
    # The same heads in every group (gpt-oss).
    same = {k: ((0, 1), 8) for k in ("x", "y")}
    assert native_group_heads(same, [["x"], ["y"]], False) == ((0, 1), (0, 1))


def test_native_per_group_heads_check_a_gemma_like_split_and_gather():
    """Two transfer groups of different total heads (Gemma4's sliding 8 and
    global 2), pages of equal bytes (32) on a TP1 rank: group A 8 heads of
    4 bytes, group B 2 of 16. At TP2 a rank holds 4 of A's heads and 1 of
    B's, a page of 16 bytes either way. P TP1 -> D TP2 splits both groups;
    P TP2 -> D TP1 gathers them from two producers. A group's slice in the
    other group's place, a swapped producer order or a missing head fails;
    each group reports its own pattern and heads."""
    from types import SimpleNamespace

    from vllm.v1.worker.simulated.kv_pattern import NativeChecker, NativeFiller

    blocks = 4

    def stub(size):
        """Group g's one row of `size` bytes (a block each)."""
        rows = [SimpleNamespace(view=torch.zeros(blocks, size, dtype=torch.uint8), region=g, offset=0, state=False)
                for g in range(2)]
        return SimpleNamespace(
            rows=rows,
            select=lambda group, held: [(group, b, i) for i, b in enumerate(held)],
            members=[{0}, {1}], padding={}, groups=[["a"], ["b"]],
            name=lambda group, row: f"g{group}.r{row}.o0",
            state_regions=lambda selected: set(),
            regions=lambda selected: {r for r, _, _ in selected},
            order=lambda held: [b for ids in held.values() for b in ids],
            worker=None,
        )

    def producer(size, heads):
        s = stub(size)
        filler = NativeFiller(s, 0, heads=heads)
        for group in range(2):
            filler._write_block("cmpl-a-0", group, 3, 2, None)
        return s

    def consumer(size, heads, fill):
        s = stub(size)
        for group in range(2):
            s.rows[group].view[2] = fill(group)
        checker = NativeChecker(s, tp_rank=1, heads=heads)
        checker._check_geometry = lambda: None
        checker.posted["cmpl-a-0"] = [2, 2]
        checker.reads["cmpl-a-0"] = [[2, 2]]
        expected = SimpleNamespace(
            remote_request_id="cmpl-a-0", block_ids=[[2], [2]],
            groups=[SimpleNamespace(group=g, block_ids=[2], first_logical_block=3) for g in range(2)],
        )
        return checker._compare("cmpl-b-0", expected)

    def identity(result, want):
        got = [(g["pattern"], g["global_heads"]) for g in result["groups"]]
        assert got == [("head-v1", w) for w in want], got

    a_slab, b_slab = 4, 16
    tp1 = producer(32, (tuple(range(8)), (0, 1)))
    # Split: decode rank 1 holds A heads 4..7 and B head 1.
    mine = (tuple(range(4, 8)), (1,))

    def split(group):
        row = tp1.rows[group].view[2]
        return row[4 * a_slab:8 * a_slab] if group == 0 else row[b_slab:2 * b_slab]

    ok = consumer(16, mine, split)
    assert ok["ok"], ok
    identity(ok, [[4, 5, 6, 7], [1]])
    # A's slice where B's goes, and B's where A's goes (equal bytes, wrong group).
    crossed = consumer(16, mine, lambda g: split(1 - g))
    assert not crossed["ok"] and crossed["mismatched_bytes"] > 0
    # A missing head: B's slice from head 0 instead of 1.
    wrong_head = consumer(16, mine, lambda g: split(g) if g == 0 else tp1.rows[1].view[2][:b_slab])
    assert not wrong_head["ok"]
    # Gather: decode TP1 holds A 0..7 and B 0, 1, from producers 0 and 1.
    p0 = producer(16, ((0, 1, 2, 3), (0,)))
    p1 = producer(16, ((4, 5, 6, 7), (1,)))

    def gather(group, order=(0, 1)):
        rows = [(p0, p1)[i].rows[group].view[2] for i in order]
        return torch.cat(rows)

    ok = consumer(32, (tuple(range(8)), (0, 1)), gather)
    assert ok["ok"], ok
    identity(ok, [list(range(8)), [0, 1]])
    swapped = consumer(32, (tuple(range(8)), (0, 1)), lambda g: gather(g, (1, 0)))
    assert not swapped["ok"]
    # A rank-pattern check of the same bytes (no heads) does not accept them.
    s = stub(16)
    for group in range(2):
        s.rows[group].view[2] = split(group)
    assert NativeChecker(s, tp_rank=1, heads=None) is not None
