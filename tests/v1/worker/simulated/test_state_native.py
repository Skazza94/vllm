# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP4 step A: NIXL's native state descriptors against the model's layout,
the eligibility of a hybrid's selection, the peer's aggregate check and the
exact selected-byte telemetry."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.simulated import state_layout as sl
from vllm.v1.worker.simulated.kv_pattern import NativeSelection, native_state_heads
from vllm.v1.worker.simulated.kv_transfer import selected_lengths, validate_state_peer

TOY = sl.StateLayout("mamba2", (("x", 16), ("B", 4), ("C", 4)), 3, 2, 4, 8, (4, 4))
PAGE = 256


def _worker(tp, rank, lengths=None, gap=0, gaps=None):
    """A worker of state regions (page PAGE, 4 blocks each) whose descriptors
    are the layout's spans at (rank, tp), or `lengths` instead; `gap` shifts
    every span of every region (a common error), `gaps` one region's each."""
    gaps = gaps or [gap]
    spans = TOY.spans(rank, tp)
    storage = torch.zeros(PAGE * 4 * len(gaps) + (1 << 12), dtype=torch.uint8)
    shift = storage.data_ptr()
    rows, bases = [], []
    for r, g in enumerate(gaps):
        base = shift + r * PAGE * 4
        bases.append(base)
        rows += [
            (base + s.offset + g, s.length if lengths is None else lengths[i], 0, PAGE, 4)
            for i, s in enumerate(spans)
        ]
    worker = SimpleNamespace(
        _physical_blocks_per_logical_kv_block=1, engine_id="e", tp_rank=rank,
        kv_caches_base_addr={"e": {rank: bases}}, block_stride_per_layer=[PAGE] * len(gaps),
        num_blocks=4, src_blocks_data=np.array(rows, dtype=np.int64), region_num_blocks=[0] * len(gaps),
        _ssm_region_indices=list(range(len(gaps))), block_len_per_layer=[PAGE] * len(gaps),
        transfer_topo=SimpleNamespace(tp_size=tp),
    )
    return worker, {"L": storage[:PAGE * 4]}


def test_native_state_rows_are_the_layouts_spans_or_refused():
    for tp, rank in ((1, 0), (2, 1), (4, 3)):
        worker, caches = _worker(tp, rank)
        selection = NativeSelection(worker, caches, [["L"], ["other"]], [0, 1], TOY)
        spans = TOY.spans(rank, tp)
        assert [r.span for r in selection.rows] == spans
        assert all(r.state for r in selection.rows)
        # Group 0 holds layer L in region 0: the model's name; group 1 holds none there.
        assert selection.state_identity(0, 0) == "L"
        assert selection.state_identity(1, 0) == "xfer.g1.r0"
        # The page beyond the state is padding, not moved.
        assert selection.padding[0].view.shape[1] == PAGE - sum(s.length for s in spans)
    # Every span shifted together (lengths and spacing intact) is refused:
    # the state starts at the page's origin, not wherever the descriptors do.
    for tp, rank in ((1, 0), (2, 0), (2, 1)):
        for gap in (6, 1, 64):
            worker, caches = _worker(tp, rank, gap=gap)
            with pytest.raises(ValueError, match="not the mamba2 layout's spans"):
                NativeSelection(worker, caches, [["L"]], [0], TOY)
    # A transfer-only region (one the group holds no layer in) follows the same
    # rule: a correct two-region worker passes, the second shifted alone fails.
    worker, caches = _worker(2, 1, gaps=[0, 0])
    NativeSelection(worker, caches, [["L"], ["other"]], [0, 1], TOY)
    worker, caches = _worker(2, 1, gaps=[0, 6])
    with pytest.raises(ValueError, match="region 1"):
        NativeSelection(worker, caches, [["L"], ["other"]], [0, 1], TOY)
    # A native decomposition of other lengths (components cut elsewhere) is refused.
    worker, caches = _worker(2, 0, lengths=[24, 6, 6, 8])
    with pytest.raises(ValueError, match="not the mamba2 layout's spans"):
        NativeSelection(worker, caches, [["L"]], [0], TOY)


def test_a_state_group_of_the_wrong_layout_is_not_eligible():
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

    text = SimpleNamespace(mamba_num_heads=64, mamba_head_dim=64, n_groups=8, ssm_state_size=128, conv_kernel=4,
                           mamba_ssm_cache_dtype="float32")
    layout = sl.layout_from_config(text, 2, 4)

    def config(tp=2, spec=None):
        return SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=True, hf_text_config=text, dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_ssm_cache_dtype="auto"),
            speculative_config=None,
            parallel_config=SimpleNamespace(tensor_parallel_size=tp),
        )

    def mamba(tp, **over):
        conv = sum(w // tp for _, w in layout.components)
        values = dict(
            block_size=16, shapes=((conv, 3), (64 // tp, 64, 128)), dtypes=(torch.bfloat16, torch.float32),
            mamba_type=MambaAttentionBackendEnum.MAMBA2,
        )
        values.update(over)
        return MambaSpec(**values)

    attn = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16)
    heads = {"a0": ((0,), 4096)}
    groups = lambda m: [KVCacheGroupSpec(["a0"], attn), KVCacheGroupSpec(["m0", "m1"], m)]
    held, got = native_state_heads(config(), groups(mamba(2)), heads, [["a0"], ["m0", "m1"]])
    assert held == ((0,), ()) and got == layout
    # The eligibility does not read the pattern flags or the peer: a TP4 rank is as eligible.
    assert native_state_heads(config(4), groups(mamba(4)), heads, [["a0"], ["m0", "m1"]])[0] == ((0,), ())
    # A state group whose spec differs from the layout (a wider conv, another dtype)
    # keeps the guard.
    for bad in (
        mamba(2, shapes=((4096 // 2 + 512 + 513, 3), (32, 64, 128))),
        mamba(2, dtypes=(torch.float32, torch.float32)),
        mamba(2, mamba_type=MambaAttentionBackendEnum.MAMBA1),
    ):
        assert native_state_heads(config(), groups(bad), heads, [["a0"], ["m0", "m1"]]) == (None, None)
    # Equal bytes, other SSM semantics: not eligible either.
    for bad in (
        mamba(2, shapes=((4096 // 2 + 512 + 512, 3), (64, 32, 128))),
        mamba(2, shapes=((4096 // 2 + 512 + 512, 3), (32, 8192))),
        mamba(2, shapes=((4096 // 2 + 512 + 512, 3), (32, 64, 256)), dtypes=(torch.bfloat16, torch.bfloat16)),
    ):
        assert native_state_heads(config(), groups(bad), heads, [["a0"], ["m0", "m1"]]) == (None, None)
    # A group mixing attention and state layers, or a hybrid without heads for its attention.
    mixed = [KVCacheGroupSpec(["a0", "m0"], attn)]
    assert native_state_heads(config(), groups(mamba(2)), {}, [["a0"], ["m0", "m1"]]) == (None, None)
    del mixed
    # Not a hybrid: the attention rule, no layout.
    plain = config()
    plain.model_config.is_hybrid = False
    assert native_state_heads(plain, [KVCacheGroupSpec(["a0"], attn)], heads, [["a0"]]) == (((0,),), None)


def test_the_peer_check_validates_the_aggregates_it_is_given():
    seen = []
    worker = SimpleNamespace(
        add_remote_agent=lambda *a, **k: seen.append(a) or "agent", _ssm_region_indices=[1],
        block_len_per_layer=[512, 512], world_size=1,
    )
    validate_state_peer(worker, TOY)
    worker.block_size = 64
    ok = SimpleNamespace(ssm_sizes=TOY.local_sizes(2), block_lens=[512, 512], block_size=64)
    assert worker.add_remote_agent(ok, 0, 2) == "agent" and seen
    # A producer block size dividing this instance's pairs (NIXL's
    # block-size ratio), its regions' lengths at that size; any other is
    # refused (the validator is installed whatever the pattern flags; NIXL's
    # own assertion for P TP > D TP is skipped for a mamba model).
    smaller = SimpleNamespace(ssm_sizes=TOY.local_sizes(2), block_lens=[256, 256], block_size=32)
    assert worker.add_remote_agent(smaller, 0, 2) == "agent"
    half = SimpleNamespace(ssm_sizes=TOY.local_sizes(1), block_lens=[256, 256], block_size=32)
    assert worker.add_remote_agent(half, 0, 1) == "agent"
    with pytest.raises(ValueError, match="of equal TP holds its regions otherwise"):
        worker.add_remote_agent(SimpleNamespace(ssm_sizes=TOY.local_sizes(1), block_lens=[512, 512], block_size=32), 0, 1)
    for remote in (48, 128):
        other = SimpleNamespace(ssm_sizes=TOY.local_sizes(2), block_lens=[512, 512], block_size=remote)
        with pytest.raises(ValueError, match="or a divisor of it"):
            worker.add_remote_agent(other, 0, 2)
        with pytest.raises(ValueError, match="or a divisor of it"):
            worker.add_remote_agent(other, 0, 1)
    for bad, match in (
        (SimpleNamespace(ssm_sizes=TOY.local_sizes(1), block_lens=[512, 512], block_size=64), "holds its states otherwise"),
        (SimpleNamespace(ssm_sizes=TOY.local_sizes(2), block_lens=[512, 4], block_size=64), "shorter than its state"),
        # A peer listing fewer regions is refused, not skipped region by region.
        (SimpleNamespace(ssm_sizes=TOY.local_sizes(2), block_lens=[512], block_size=64), "registers 1 regions"),
    ):
        with pytest.raises(ValueError, match=match):
            worker.add_remote_agent(bad, 0, 2)
    # At equal TP every region's block length must match, as the equal-geometry guard required.
    same = SimpleNamespace(ssm_sizes=TOY.local_sizes(1), block_lens=[512, 512], block_size=64)
    assert worker.add_remote_agent(same, 0, 1) == "agent"
    with pytest.raises(ValueError, match="of equal TP holds its regions otherwise"):
        worker.add_remote_agent(SimpleNamespace(ssm_sizes=TOY.local_sizes(1), block_lens=[1024, 512], block_size=64), 0, 1)
    with pytest.raises(ValueError, match="does not split"):
        worker.add_remote_agent(ok, 0, 8)


def test_selected_bytes_and_lengths_are_exact_whatever_nixl_merges():
    # Two strided runs: 3 descriptors of 100 bytes, then 4 of 40 and 1 of 8 ... (length, stride, count).
    runs = np.array([[100, 100, 3], [40, 40, 4], [8, 8, 2]], dtype=np.int64)
    total, table = selected_lengths(runs, [0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert total == 3 * 100 + 4 * 40 + 2 * 8 and table == {8: 2, 40: 4, 100: 3}
    # Adjacent descriptors may merge into one NIXL descriptor; the bytes do not change.
    assert selected_lengths(runs, [0, 1]) == (200, {100: 2})
    assert selected_lengths(runs, [3, 8]) == (48, {8: 1, 40: 1})
    assert selected_lengths(None, [0]) == (None, None)


def test_watched_reads_resolve_pieces_through_their_handles_and_completion():
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    worker, caches = _worker(2, 1)
    selection = NativeSelection(worker, caches, [["L"]], [0], TOY)
    worker.block_size = 64
    worker.src_xfer_handles_by_block_size = {64: "local-h"}
    worker.dst_xfer_side_handles = {"remote-engine": {3: "remote-h"}}
    posted = []
    wrapper = SimpleNamespace(
        make_prepped_xfer=lambda op, local, lids, remote, rids, *a, **k: posted.append((local, remote)) or "xfer-1",
        prep_xfer_dlist=lambda agent, lst, *a, **k: "remote-h",
        check_xfer_state=lambda handle: "DONE",
    )
    worker.nixl_wrapper = wrapper
    checker = NativeChecker(selection, tp_rank=1, heads=((),))
    checker.watch_reads()
    # Descriptor ids of the registered local list: row r, block b is id 4*r + b.
    wrapper.make_prepped_xfer("READ", "local-h", [1, 5, 9, 13], "remote-h", [0, 1, 2, 3], notif_msg=b"req-a:7")
    (record,) = checker.records["req-a"]
    spans = TOY.spans(1, 2)
    assert record.rank == 3 and record.engine == "remote-engine" and not record.done
    assert record.pieces == [(0, 1, s.offset, s.length) for s in spans]
    assert wrapper.check_xfer_state("xfer-1") == "DONE" and record.done
    # A handle whose prepared list was never seen resolves to None (reported, never assumed).
    wrapper.make_prepped_xfer("READ", "unknown-h", [0], "remote-h", [0], notif_msg=b"req-b:1")
    assert checker.records["req-b"][0].pieces is None


def test_a_requests_coverage_reports_bytes_by_group_component_and_class():
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    worker, caches = _worker(2, 1)
    worker.tp_mappings = {"e": object(), "r": object()}
    worker.transfer_topo = SimpleNamespace(
        tp_size=2, total_num_kv_heads=4, get_engine_info=lambda engine: SimpleNamespace(remote_tp_size=1)
    )
    rows = len(worker.src_blocks_data)
    # Every group's blocks in every state region, row by row (as NIXL selects them).
    worker._compute_desc_ids = lambda lists, num_blocks, *a: np.array(
        [4 * r + b for r in range(rows) for blocks in lists for b in blocks], dtype=np.int64
    )
    selection = NativeSelection(worker, caches, [["L"], ["other"]], [0, 1], TOY)
    checker = NativeChecker(selection, tp_rank=1, heads=((), ()))
    expected = SimpleNamespace(
        groups=[SimpleNamespace(group=0, block_ids=[2]), SimpleNamespace(group=1, block_ids=[3])]
    )
    # TP2 rank 1 <- TP1: every span's second half comes from producer 0: group 0's
    # blocks logically hold layer L, group 1's block a transfer-only region.
    spans = TOY.spans(1, 2)
    produced = [
        (0, block, s.offset, s.length) for block in (2, 3) for s in spans
    ]
    ok = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", produced, True)])
    assert ok["problems"] == [], ok
    (read,) = ok["reads"]
    assert read["completed"] and read["rank"] == 0 and read["descriptors"] == 8
    assert read["bytes"] == 2 * sum(s.length for s in spans)
    assert set(read["classes"]) == {
        "g0.conv.x.logical", "g0.conv.B.logical", "g0.conv.C.logical", "g0.ssm.ssm.logical",
        "g1.conv.x.transfer-only", "g1.conv.B.transfer-only", "g1.conv.C.transfer-only", "g1.ssm.ssm.transfer-only",
    }
    assert read["classes"]["g0.conv.x.logical"] == [1, spans[0].length]
    # The same bytes missing for the transfer-only group, or the READ unfinished.
    short = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", produced[:4], True)])
    assert short["problems"]
    unfinished = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", produced, False)])
    assert any("had not completed" in p for p in unfinished["problems"])
    # An unresolved READ is reported, not assumed.
    unresolved = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", None, True)])
    assert any("could not be resolved" in p for p in unresolved["problems"])


def test_coverage_follows_the_requests_producer_among_several_engines():
    """A DP producer is one engine a rank: the consumer meets several, while
    each request reads from one of them."""
    from vllm.v1.worker.simulated.kv_pattern import NativeChecker

    worker, caches = _worker(2, 1)
    worker.tp_mappings = {"e": object(), "p-dp0": object(), "p-dp1": object()}
    worker.transfer_topo = SimpleNamespace(
        tp_size=2, total_num_kv_heads=4, get_engine_info=lambda engine: SimpleNamespace(remote_tp_size=1)
    )
    rows = len(worker.src_blocks_data)
    worker._compute_desc_ids = lambda lists, num_blocks, *a: np.array(
        [4 * r + b for r in range(rows) for blocks in lists for b in blocks], dtype=np.int64
    )
    selection = NativeSelection(worker, caches, [["L"], ["other"]], [0, 1], TOY)
    checker = NativeChecker(selection, tp_rank=1, heads=((), ()))
    expected = SimpleNamespace(
        groups=[SimpleNamespace(group=0, block_ids=[2]), SimpleNamespace(group=1, block_ids=[3])]
    )
    produced = [(0, block, s.offset, s.length) for block in (2, 3) for s in TOY.spans(1, 2)]
    one = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", produced, True, engine="p-dp1")])
    assert one["problems"] == [], one
    split = checker._coverage(
        expected,
        [sl.ReadRecord("r", 0, "h", produced[:4], True, engine="p-dp0"),
         sl.ReadRecord("r", 0, "h2", produced[4:], True, engine="p-dp1")],
    )
    assert split["problems"] == ["the request's READs name 2 producer engines, one expected"]
    # Unknown engines with several remotes: reported, never guessed.
    unknown = checker._coverage(expected, [sl.ReadRecord("r", 0, "h", produced, True)])
    assert unknown["problems"] == ["the request's READs name 0 producer engines, one expected"]
