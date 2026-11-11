# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CP4 step A: a hybrid's recurrent state as the model defines it (state-v1),
ownership across unequal TP sizes, the posted-read coverage rule, and the
mutations the independent reference must fail."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.simulated import state_layout as sl
from vllm.v1.worker.simulated.kv_pattern import (
    NativeChecker,
    NativeFiller,
    native_group_heads,
)

NEMOTRON = SimpleNamespace(
    mamba_num_heads=64, mamba_head_dim=64, n_groups=8, ssm_state_size=128, conv_kernel=4
)
QWEN = SimpleNamespace(
    linear_num_key_heads=16, linear_key_head_dim=128, linear_num_value_heads=48,
    linear_value_head_dim=128, linear_conv_kernel_dim=4,
)


def test_the_layouts_come_from_the_model_config_alone():
    n = sl.layout_from_config(NEMOTRON, 2, 4)
    assert n.family == "mamba2" and n.components == (("x", 4096), ("B", 1024), ("C", 1024))
    assert n.conv_rows == 3 and n.ssm_heads == 64 and n.ssm_head_bytes == 64 * 128 * 4
    # The saved Nemotron TP2 state spans: 12,288 / 3,072 / 3,072 / 1,048,576 bytes.
    assert [s.length for s in n.spans(0, 2)] == [12288, 3072, 3072, 1048576]
    q = sl.layout_from_config(QWEN, 2, 4)
    assert q.family == "gdn" and q.components == (("Q", 2048), ("K", 2048), ("V", 6144))
    assert q.ssm_heads == 48 and q.ssm_head_bytes == 128 * 128 * 4
    assert q.local_sizes(1) == (10240 * 3 * 2, 48 * 128 * 128 * 4)
    assert sl.layout_from_config(SimpleNamespace(hidden_size=1), 2, 4) is None
    # Offsets are per rank, components in order, then the heads.
    assert [s.offset for s in q.spans(1, 2)] == [0, 1024 * 6, 2048 * 6, (1024 * 2 + 3072) * 6]
    assert [(s.first, s.count) for s in q.spans(1, 2)] == [(1024, 1024), (1024, 1024), (3072, 3072), (24, 24)]
    assert not q.divides(5) and q.divides(8)
    with pytest.raises(ValueError):
        q.spans(0, 5)


def test_element_sizes_are_resolved_as_the_native_allocation_resolves_them():
    def config(mamba="auto", ssm="auto", text=NEMOTRON):
        return SimpleNamespace(
            speculative_config=None,
            model_config=SimpleNamespace(hf_text_config=text, dtype=torch.bfloat16),
            cache_config=SimpleNamespace(mamba_cache_dtype=mamba, mamba_ssm_cache_dtype=ssm),
        )

    sizes = lambda c: (sl.layout_for(c).conv_itemsize, sl.layout_for(c).ssm_itemsize)
    # Both "auto": the model's dtype, the SSM following the conv state.
    assert sizes(config()) == (2, 2)
    # --mamba-cache-dtype sets the conv state, and an "auto" SSM state with it.
    assert sizes(config(mamba="float32")) == (4, 4)
    assert sizes(config(mamba="float32", ssm="bfloat16")) == (4, 2)
    # An "auto" SSM dtype takes the config's own, as the hybrid configs' verification does.
    own = SimpleNamespace(**vars(NEMOTRON), mamba_ssm_cache_dtype="float32")
    assert sizes(config(text=own)) == (2, 4)


def _spec(layout, tp, **over):
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    conv = sum(w // tp for _, w in layout.components)
    values = dict(
        mamba_type=MambaAttentionBackendEnum.MAMBA2 if layout.family == "mamba2" else MambaAttentionBackendEnum.GDN_ATTN,
        tp_replicated=False,
        shapes=((conv, layout.conv_rows), (layout.ssm_heads // tp, *layout.ssm_shape)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    values.update(over)
    return SimpleNamespace(**values)


def test_a_registered_spec_must_be_the_layout_field_by_field():
    n = sl.layout_from_config(NEMOTRON, 2, 4)
    assert sl.check_spec(n, _spec(n, 2), 2) == []
    assert sl.check_spec(n, _spec(n, 1), 1) == []
    bad = sl.check_spec(n, _spec(n, 2, dtypes=(torch.float32, torch.float32)), 2)
    assert any("conv element" in p for p in bad)
    bad = sl.check_spec(n, _spec(n, 2, dtypes=(torch.bfloat16, torch.bfloat16)), 2)
    assert any("SSM element" in p for p in bad)
    # Equal bytes, other semantics: the head axis elsewhere, flattened heads, a
    # wider state in a narrower element: each fails on its own field.
    for shape, dtype, field in (
        ((32, 64, 128), torch.float32, None),
        ((64, 32, 128), torch.float32, "SSM shape"),
        ((32, 8192), torch.float32, "SSM shape"),
        ((32, 64, 256), torch.bfloat16, "SSM"),
    ):
        spec = _spec(n, 2, dtypes=(torch.bfloat16, dtype))
        spec.shapes = (spec.shapes[0], shape)
        found = sl.check_spec(n, spec, 2)
        assert (found == []) if field is None else any(field in p for p in found), (shape, dtype, found)
    assert sl.check_spec(n, _spec(n, 2, tp_replicated=True), 2)
    spec = _spec(n, 2)
    spec.shapes = ((4096 // 2 + 513 + 512, 3), spec.shapes[1])  # a conv wider by a column
    assert any("conv shape" in p for p in sl.check_spec(n, spec, 2))
    q = sl.layout_from_config(QWEN, 2, 4)
    assert any("mamba_type" in p for p in sl.check_spec(q, _spec(n, 2), 2))


# ----- the pattern is the same bytes of the same global coordinates -----

TOY = sl.StateLayout("mamba2", (("x", 16), ("B", 4), ("C", 4)), 3, 2, 4, 8)


def _page(layout, rank, tp, block, layer="L", base="req"):
    return {
        (s.kind, s.component): sl.state_span_pattern(base, layer, block, layout, s)
        for s in layout.spans(rank, tp)
    }


def test_state_bytes_are_the_same_at_any_tp_size():
    whole = _page(TOY, 0, 1, 3)
    for tp in (2, 4):
        for key in whole:
            joined = torch.cat([_page(TOY, r, tp, 3)[key] for r in range(tp)])
            assert torch.equal(joined, whole[key]), (tp, key)
    # Another layer, block, request or component is another stream.
    assert not torch.equal(_page(TOY, 0, 1, 3, layer="M")[("ssm", "ssm")], whole[("ssm", "ssm")])
    assert not torch.equal(_page(TOY, 0, 1, 4)[("conv", "x")], whole[("conv", "x")])
    assert not torch.equal(_page(TOY, 0, 1, 3, base="other")[("conv", "B")], whole[("conv", "B")])
    assert not torch.equal(whole[("conv", "B")], whole[("conv", "C")])


def test_two_state_groups_never_share_a_stream_for_the_same_ordinal():
    """Identity is the model's layer name (or a transfer-only region's own
    identity), never an ordinal within a group: layer 0 of two groups differ."""
    a = _page(TOY, 0, 1, 1, layer="model.layers.0.mixer")
    b = _page(TOY, 0, 1, 1, layer="model.layers.7.mixer")
    x = _page(TOY, 0, 1, 1, layer="xfer.g1.r4")
    for key in a:
        assert not torch.equal(a[key], b[key]) and not torch.equal(a[key], x[key])


# ----- ownership -----


def test_ownership_covers_every_local_head_and_state_range():
    # Qwen: 4 KV heads; decode TP1 owns heads 0..3 and needs producers 0 and 1 of TP2.
    assert sl.attention_sources(0, 1, 2, 4) == {0: (0, 1), 1: (2, 3)}
    # TP2 <- TP1: both ranks take their heads from the one producer.
    assert sl.attention_sources(1, 2, 1, 4) == {0: (2, 3)}
    # Nemotron K=2: TP2 <- TP4 attention from producer 0 / 2, state from 0,1 / 2,3.
    assert sl.attention_sources(0, 2, 4, 2) == {0: (0,)}
    assert sl.attention_sources(1, 2, 4, 2) == {2: (1,)}
    assert sl.state_sources(0, 2, 4, 4096) == {0: (0, 1024), 1: (1024, 2048)}
    assert sl.state_sources(1, 2, 4, 4096) == {2: (2048, 3072), 3: (3072, 4096)}
    # TP4 <- TP2 (replicated attention on the decode side).
    assert [sl.attention_sources(r, 4, 2, 2) for r in range(4)] == [{0: (0,)}, {0: (0,)}, {1: (1,)}, {1: (1,)}]
    assert sl.state_sources(1, 4, 2, 4096) == {0: (1024, 2048)}
    assert sl.state_sources(0, 2, 2, 4096) == {0: (0, 2048)}


@pytest.mark.parametrize("kv", [2, 4])
def test_ownership_matches_nixl_compute_tp_mapping(kv):
    """The independent rule against the native mapping, every pairing of TP
    1, 2, 4, 8 (the larger a multiple of the smaller)."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import compute_tp_mapping
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    width = 4096
    for local in (1, 2, 4, 8):
        for remote in (1, 2, 4, 8):
            if max(local, remote) % min(local, remote):
                continue
            for rank in range(local):
                topo = SimpleNamespace(
                    tp_rank=rank, tp_size=local, is_mla=False, total_num_kv_heads=kv,
                    dcp_source_ranks=None, dcp_consumer_count=lambda *a: 1,
                )
                plan = compute_tp_mapping(topo, remote, (FullAttentionSpec, MambaSpec))
                assert tuple(sorted(sl.attention_sources(rank, local, remote, kv))) == plan.source_ranks_per_group[0], (local, remote, rank)
                assert tuple(sorted(sl.state_sources(rank, local, remote, width))) == plan.source_ranks_per_group[1], (local, remote, rank)


@pytest.mark.parametrize("config", [NEMOTRON, QWEN], ids=["mamba2", "gdn"])
@pytest.mark.parametrize("local,remote", [(1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (4, 8)])
def test_native_split_reads_land_where_ownership_says(monkeypatch, config, local, remote):
    """P TP > D TP: the decode rank's local state descriptors as native NIXL
    builds them (`_build_mamba_local`, the conv split from `MambaSpec`), cut
    per producer by native `_build_local_splits_from_plan` under the native
    mapping, resolved to (region, block, offset, length) pieces, equal the
    independent rule's (`expected_pieces`) for every producer."""
    import types

    from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker as Native,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import compute_tp_mapping
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    monkeypatch.setattr(ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True)
    layout = sl.layout_from_config(config, 2, 4)
    spec = _spec(layout, local)
    decomp = ssm_conv_transfer_utils.derive_mamba_conv_split(spec, local)
    page, regions, num_blocks, held = sum(decomp.ssm_sizes) + 512, 3, 6, [1, 4]
    bases = [(r + 1) << 32 for r in range(regions)]
    for rank in range(local):
        topo = SimpleNamespace(
            tp_rank=rank, tp_size=local, is_mla=False, total_num_kv_heads=8,
            dcp_source_ranks=None, dcp_consumer_count=lambda *a: 1,
        )
        plan = compute_tp_mapping(topo, remote, (FullAttentionSpec, MambaSpec))
        worker = SimpleNamespace(
            _conv_decomp=decomp, _mamba_ssm_size=decomp.ssm_sizes, _logical_num_blocks=num_blocks,
            _physical_blocks_per_logical_kv_block=1, device_id=0, _ssm_region_indices=list(range(regions)),
            block_stride_per_layer=[page] * regions, _ple_region_index=None,
            _group_spec_types=[FullAttentionSpec, MambaSpec], transfer_topo=topo,
            block_len_per_layer=[], num_regions=0, _pack_strided_descs=Native._pack_strided_descs,
            _strided_desc_starts=Native._strided_desc_starts,
        )
        worker._fa_desc_replicated = types.MethodType(Native._fa_desc_replicated, worker)
        local_descs = Native._build_mamba_local(worker, bases)
        handles = list(Native._build_local_splits_from_plan(worker, plan, local_descs, num_fa_descs=0))
        assert len(handles) == len(plan.all_source_ranks)
        native = {}
        for p_idx, handle in enumerate(handles):
            for addr, length, _, stride, count in handle.tolist():
                region = next(r for r, base in enumerate(bases) if base <= addr < base + page)
                assert stride == page and count == num_blocks
                native.setdefault(plan.all_source_ranks[p_idx], []).extend(
                    (region, b, addr - bases[region], length) for b in held
                )
        rows = [
            {"region": r, "offset": span.offset, "length": span.length, "state": True, "span": span}
            for r in range(regions)
            for span in layout.spans(rank, local)
        ]
        want = sl.expected_pieces(rows, held, rank, local, remote, layout, 8)
        assert {p: sorted(v) for p, v in native.items()} == {p: sorted(v) for p, v in want.items()}, (
            local, remote, rank,
        )


# ----- the posted-read coverage rule -----

ROWS = [
    {"region": 0, "offset": 0, "length": 64, "state": False, "heads": (0, 1, 2, 3)},
    {"region": 1, "offset": 0, "length": 8, "state": True, "span": TOY.spans(0, 1)[0]},
]


def _expected(local=1, remote=2, rank=0, kv=4, blocks=(5,)):
    rows = [dict(ROWS[0]), dict(ROWS[1])]
    rows[1]["span"] = TOY.spans(rank, local)[0]
    rows[1]["length"] = rows[1]["span"].length
    return sl.expected_pieces(rows, list(blocks), rank, local, remote, TOY, kv)


def _read(rank, pieces, done=True):
    return sl.ReadRecord("req", rank, object(), pieces, done)


def _from(expected, rank, only=None):
    pieces = [p for p in expected[rank] if only is None or only(p)]
    return pieces


def test_a_valid_attention_plus_state_read_then_a_state_only_read_is_covered():
    want = _expected()
    assert set(want) == {0, 1}
    # Producer 0: attention heads 0,1 and the first half of x; producer 1: heads 2,3 and the second.
    reads = [_read(0, want[0]), _read(1, want[1])]
    assert sl.coverage_problems(want, reads) == []
    # Producer 1's attention READ first, its state in a later state-only READ.
    attn = lambda p: p[0] == 0
    reads = [_read(0, want[0]), _read(1, _from(want, 1, attn)), _read(1, _from(want, 1, lambda p: p[0] == 1))]
    assert sl.coverage_problems(want, reads) == []
    # NIXL cutting the same bytes into more descriptors is the same coverage.
    split = [(r, b, o, 1) for r, b, o, n in want[0] if r == 1 for o in range(o, o + n)] + [p for p in want[0] if p[0] == 0]
    assert sl.coverage_problems(want, [_read(0, split), _read(1, want[1])]) == []


def test_heads_a_page_does_not_hold_are_an_ownership_error_not_a_crash():
    rows = [dict(ROWS[0], heads=(0, 1)), dict(ROWS[1], span=TOY.spans(0, 1)[0])]
    with pytest.raises(ValueError, match=r"supplies heads \[2, 3\]"):
        sl.expected_pieces(rows, [5], 0, 1, 2, TOY, 4)


def test_coverage_rejects_what_ownership_does_not_say():
    want = _expected()
    ok0, ok1 = want[0], want[1]
    state = lambda p: p[0] == 1
    cases = {
        "missing shard": [_read(0, ok0), _read(1, [p for p in ok1 if p[0] == 0])],
        "duplicated shard": [_read(0, ok0), _read(1, ok1), _read(1, _from(want, 1, state))],
        "swapped sources": [_read(1, ok0), _read(0, ok1)],
        "foreign region": [_read(0, ok0 + [(9, 5, 0, 8)]), _read(1, ok1)],
        "wrong block": [_read(0, [(r, 6, o, n) for r, b, o, n in ok0]), _read(1, ok1)],
        "unknown producer": [_read(0, ok0), _read(1, ok1), _read(7, [(1, 5, 0, 8)])],
        "state from the attention producer only": [_read(0, ok0 + _from(want, 1, state)), _read(1, [p for p in ok1 if p[0] == 0])],
    }
    for name, reads in cases.items():
        assert sl.coverage_problems(want, reads), name
    # Both READs posted, one still outstanding when the request is reported complete.
    pending = sl.coverage_problems(want, [_read(0, ok0), _read(1, ok1, done=False)])
    assert pending and "had not completed" in pending[0]


# ----- mutations: the native side moves, the semantic pattern stays -----


def _stub(layout, rank, tp, blocks=4):
    rows = [
        SimpleNamespace(
            view=torch.zeros(blocks, s.length, dtype=torch.uint8), region=7, offset=s.offset,
            state=True, span=s,
        )
        for s in layout.spans(rank, tp)
    ]
    return SimpleNamespace(
        rows=rows, layout=layout, groups=[["L"]], members=[{7}], padding={},
        select=lambda group, held: [(i, b, k) for k, b in enumerate(held) for i in range(len(rows))],
        state_identity=lambda group, region: "L",
        name=lambda group, row: f"g{group}.r7.o{rows[row].offset}",
        state_regions=lambda selected: {7},
        regions=lambda selected: {7},
        order=lambda held: [b for ids in held.values() for b in ids],
        worker=None,
    )


def _produce(tp):
    stubs = [_stub(TOY, r, tp) for r in range(tp)]
    for r, s in enumerate(stubs):
        NativeFiller(s, r, heads=((),))._write_block("cmpl-a-0", 0, 0, 2, None)
    return stubs


def _check(stub, rank):
    checker = NativeChecker(stub, tp_rank=rank, heads=((),))
    checker._check_geometry = lambda: None
    checker._coverage = lambda e, r: {"problems": [], "reads": []}  # tested apart
    checker.posted["cmpl-a-0"] = [2]
    checker.reads["cmpl-a-0"] = [[2]]
    expected = SimpleNamespace(
        remote_request_id="cmpl-a-0", block_ids=[[2]],
        groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=3)],
    )
    return checker._compare("cmpl-b-0", expected)


def test_a_cached_state_is_checked_at_its_blocks_place():
    """Without prefix caching a request's state is its one block (place 0,
    whatever index the read's metadata gives). With it (align mode) a state
    is its block's: written and compared at the block's place among the
    request's, under its content's identity."""
    identity = "content|B4|n4|00"
    producer = _stub(TOY, 0, 1)
    NativeFiller(producer, 0, heads=((),))._write_block("cmpl-a-0", 0, 3, 2, identity)

    def compare(place, identity_of):
        consumer = _stub(TOY, 0, 1)
        for row, source in zip(consumer.rows, producer.rows):
            row.view[2] = source.view[2]
        checker = NativeChecker(consumer, tp_rank=0, heads=((),))
        checker._check_geometry = lambda: None
        checker._coverage = lambda e, r: {"problems": [], "reads": []}
        checker.posted["cmpl-a-0"] = [2]
        checker.reads["cmpl-a-0"] = [[2]]
        expected = SimpleNamespace(
            remote_request_id="cmpl-a-0", block_ids=[[2]],
            groups=[SimpleNamespace(group=0, block_ids=[2], first_logical_block=place)],
        )
        return checker._compare("cmpl-b-0", expected, identity_of)

    ok = compare(3, lambda group, logical: identity)
    assert ok["ok"] and ok["groups"][0]["pattern_first_block"] == 3, ok
    assert ok["groups"][0]["pattern"] == sl.STATE_PATTERN
    # Another place, or another content, is another state.
    assert not compare(2, lambda group, logical: identity)["ok"]
    assert not compare(3, lambda group, logical: "content|B4|n4|01")["ok"]
    # Without prefix caching the place stays the request's one state's.
    assert not compare(3, None)["ok"]


def _receive(rank, local_tp, remote_tp, mutate=None):
    """A consumer rank's state as a native split / gather would deliver it
    (components in order, each from its producers), optionally mutated."""
    producers = _produce(remote_tp)
    consumer = _stub(TOY, rank, local_tp)
    pieces = {}
    for span in TOY.spans(rank, local_tp):
        width = dict(TOY.components)[span.component] if span.kind == "conv" else TOY.ssm_heads
        parts = []
        for p, (first, last) in sorted(sl.state_sources(rank, local_tp, remote_tp, width).items()):
            source = next(
                s for s in TOY.spans(p, remote_tp) if (s.kind, s.component) == (span.kind, span.component)
            )
            row = producers[p].rows[TOY.spans(p, remote_tp).index(source)].view[2]
            parts.append(row[(first - source.first) * source.unit_bytes : (last - source.first) * source.unit_bytes])
        pieces[(span.kind, span.component)] = torch.cat(parts)
    if mutate:
        mutate(pieces, TOY.spans(rank, local_tp))
    for row, span in zip(consumer.rows, TOY.spans(rank, local_tp)):
        row.view[2] = pieces[(span.kind, span.component)]
    return _check(consumer, rank)


@pytest.mark.parametrize("local,remote", [(1, 1), (2, 2), (2, 1), (1, 2), (1, 4), (4, 1), (2, 4), (4, 2)])
def test_a_correct_split_or_gather_passes(local, remote):
    for rank in range(local):
        result = _receive(rank, local, remote)
        assert result["ok"], (local, remote, rank, result)
        assert result["groups"][0]["pattern"] == "rank-v0" or result["groups"][0]["pattern"] == "state-v1"


def test_native_offsets_component_order_and_head_slices_are_failures():
    def swap_x_b(pieces, spans):  # components in the wrong order (same bytes moved, B where x goes)
        x, b = pieces[("conv", "x")], pieces[("conv", "B")]
        pieces[("conv", "x")], pieces[("conv", "B")] = b.repeat(len(x) // len(b)), x[: len(b)]

    def shift_column(pieces, spans):  # one column late
        x = pieces[("conv", "x")]
        pieces[("conv", "x")] = torch.cat([x[6:], x[:6]])

    def shift_head(pieces, spans):  # an SSM head off by one
        h = pieces[("ssm", "ssm")]
        pieces[("ssm", "ssm")] = torch.cat([h[8:], h[:8]])

    def swap_b_c(pieces, spans):
        pieces[("conv", "B")], pieces[("conv", "C")] = pieces[("conv", "C")], pieces[("conv", "B")]

    for local, remote in ((1, 1), (2, 1), (1, 2), (2, 4)):
        assert _receive(0, local, remote)["ok"]
        for mutation in (swap_x_b, shift_column, shift_head, swap_b_c):
            assert not _receive(0, local, remote, mutation)["ok"], (local, remote, mutation.__name__)


def test_a_state_of_another_layer_with_the_same_shape_fails():
    """Two state groups of identical shape: layer 0's bytes in layer 1's place
    (a cross-group swap) must fail, ordinals notwithstanding."""
    producers = _produce(1)
    consumer = _stub(TOY, 0, 1)
    consumer.state_identity = lambda group, region: "M"  # the checker expects layer M, delivery is L's
    for row, src in zip(consumer.rows, producers[0].rows):
        row.view[2] = src.view[2]
    assert not _check(consumer, 0)["ok"]


def test_native_group_heads_admit_a_hybrid_only_with_its_state_groups():
    heads = {"a0": ((0, 1), 16), "a1": ((0, 1), 16)}
    groups = [["a0", "a1"], ["m0"], ["m1"]]
    assert native_group_heads(heads, groups, True) is None
    held = native_group_heads(heads, groups, True, frozenset({1, 2}))
    assert held == ((0, 1), (), ())
    # A state group left out, or an attention group without heads, keeps the guard.
    assert native_group_heads(heads, groups, True, frozenset({1})) is None
    assert native_group_heads({"a0": ((0, 1), 16)}, groups, True, frozenset({1, 2})) is None
    # Not a hybrid: unchanged.
    assert native_group_heads(heads, [["a0", "a1"]], False) == ((0, 1),)
