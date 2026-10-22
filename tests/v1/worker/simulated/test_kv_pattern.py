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
            "each": [[393216, 0.002, 0.00015, 2], [262144, 0.0015, 0.0001, 2]],
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
    sender.data["transfer_duration"] = [0.002]
    sender.data["bytes_transferred"] = [393216]
    sender.data["num_descriptors"] = [3]
    assert completed_transfers(KVConnectorOutput(kv_connector_stats=sender)) == {
        "send": {
            "count": 1,
            "bytes": 393216,
            "seconds": 0.002,
            "each": [[393216, 0.002, None, 3]],
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
