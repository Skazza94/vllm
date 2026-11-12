# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A hybrid model's recurrent state as its model defines it, independent of
how NIXL's descriptors cut it, and the expected pattern and ownership that
follow (state-v1).

The reference (`StateLayout`) comes from the model's configuration only
(Mamba2: conv `x|B|C` and SSM heads; GDN: conv `Q|K|V` and value heads), never
from `derive_mamba_conv_split` or the descriptor builder: the native
decomposition is compared against it (`check_spec`, `NativeSelection`), so a
shifted offset, a swapped component or a wrong head slice is a failure against
a fixed expected pattern.

A rank holds, of each conv component of global width w, the columns
`[rank*w/tp, (rank+1)*w/tp)`, each column's `conv_rows` values contiguous (DS
layout), the components in order; then its `ssm_heads/tp` SSM heads, each head
`ssm_head_bytes` contiguous. A state's bytes are keyed by global coordinates
(layer, component, column / head), so any TP size writes and reads the same
bytes of the same coordinates.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

STATE_PATTERN = "state-v1"
"""A hybrid's recurrent state as its global conv columns and SSM heads."""


@dataclass(frozen=True)
class Span:
    """One contiguous part of a rank's state page."""

    kind: str  # "conv" or "ssm"
    component: str  # conv: x/B/C or Q/K/V; ssm: "ssm"
    offset: int  # within the state's bytes in the page
    length: int
    first: int  # first global column (conv) or head (ssm)
    count: int  # columns or heads
    unit_bytes: int  # one column's or head's bytes


@dataclass(frozen=True)
class StateLayout:
    family: str  # "mamba2", "gdn" or "kda"
    components: tuple[tuple[str, int], ...]  # conv (name, global columns)
    conv_rows: int
    conv_itemsize: int
    ssm_heads: int
    ssm_head_bytes: int
    ssm_shape: tuple[int, ...] = ()
    """One SSM head's own shape (Mamba2 `(head_dim, state)`, GDN
    `(value_dim, key_dim)`): a rank's SSM state is `(heads/tp, *ssm_shape)`,
    heads outermost and contiguous."""
    ssm_itemsize: int = 0
    units: tuple[int, ...] = ()
    """Counts that must divide a TP size too (Mamba2's groups, GDN's key
    heads): a size above them replicates, which this layout does not model."""

    @property
    def column_bytes(self) -> int:
        return self.conv_rows * self.conv_itemsize

    def divides(self, tp: int) -> bool:
        return (
            tp >= 1
            and all(w % tp == 0 for _, w in self.components)
            and self.ssm_heads % tp == 0
            and all(n % tp == 0 for n in self.units)
        )

    def local_sizes(self, tp: int) -> tuple[int, int]:
        """A rank's (conv bytes, SSM bytes) at `tp`."""
        self._need(tp)
        conv = sum(w // tp for _, w in self.components) * self.column_bytes
        return conv, self.ssm_heads // tp * self.ssm_head_bytes

    def spans(self, rank: int, tp: int) -> list[Span]:
        """The rank's conv components in order, then its SSM heads."""
        self._need(tp)
        out, offset = [], 0
        for name, width in self.components:
            n = width // tp
            out.append(
                Span("conv", name, offset, n * self.column_bytes, rank * n, n, self.column_bytes)
            )
            offset += n * self.column_bytes
        n = self.ssm_heads // tp
        out.append(Span("ssm", "ssm", offset, n * self.ssm_head_bytes, rank * n, n, self.ssm_head_bytes))
        return out

    def _need(self, tp: int) -> None:
        if not self.divides(tp):
            raise ValueError(
                f"A {self.family} state of components {dict(self.components)} and "
                f"{self.ssm_heads} SSM heads does not split over TP {tp}."
            )


def _itemsize(dtype) -> int:
    return torch.empty(0, dtype=dtype).element_size()


def layout_from_config(
    text_config, conv_itemsize: int, ssm_itemsize: int
) -> StateLayout | None:
    """The model's state from its config, None for a model that is none of
    them (Nemotron-H / Mamba2: `mamba_num_heads`, `mamba_head_dim`, `n_groups`,
    `ssm_state_size`, `conv_kernel`; GDN: `linear_*`; Kimi-K3's KDA: its
    nested `linear_attn_config`, its conv `Q|K|V` and recurrent heads as
    `MambaStateShapeCalculator.kda_state_shape` builds them, key heads the
    value heads')."""
    get = lambda name: getattr(text_config, name, None)
    kda = get("linear_attn_config")
    if get("model_type") == "kimi_linear" and isinstance(kda, dict):
        heads, dim = kda.get("num_heads"), kda.get("head_dim")
        kernel = kda.get("short_conv_kernel_size")
        if not (heads and dim and kernel):
            return None
        width = heads * dim
        return StateLayout(
            "kda",
            (("Q", width), ("K", width), ("V", width)),
            kernel - 1,
            conv_itemsize,
            heads,
            dim * dim * ssm_itemsize,
            (dim, dim),
            ssm_itemsize,
            (heads,),
        )
    if get("linear_num_value_heads") is not None and get("linear_key_head_dim") is not None:
        key = get("linear_num_key_heads") * get("linear_key_head_dim")
        value = get("linear_num_value_heads") * get("linear_value_head_dim")
        return StateLayout(
            "gdn",
            (("Q", key), ("K", key), ("V", value)),
            get("linear_conv_kernel_dim") - 1,
            conv_itemsize,
            get("linear_num_value_heads"),
            get("linear_value_head_dim") * get("linear_key_head_dim") * ssm_itemsize,
            (get("linear_value_head_dim"), get("linear_key_head_dim")),
            ssm_itemsize,
            (get("linear_num_key_heads"), get("linear_num_value_heads")),
        )
    if get("mamba_num_heads") is not None and get("mamba_head_dim") is not None:
        n_groups = get("n_groups") or get("mamba_n_groups")
        state = get("ssm_state_size") or get("mamba_d_state")
        kernel = get("conv_kernel") or get("mamba_d_conv")
        if not (n_groups and state and kernel):
            return None
        groups = n_groups * state
        return StateLayout(
            "mamba2",
            (("x", get("mamba_num_heads") * get("mamba_head_dim")), ("B", groups), ("C", groups)),
            kernel - 1,
            conv_itemsize,
            get("mamba_num_heads"),
            get("mamba_head_dim") * state * ssm_itemsize,
            (get("mamba_head_dim"), state),
            ssm_itemsize,
            (n_groups, get("mamba_num_heads")),
        )
    return None


def layout_for(vllm_config) -> StateLayout | None:
    """The model's layout under this configuration, its element sizes as the
    native allocation resolves them (`MambaStateDtypeCalculator`): conv of the
    cache's `mamba_cache_dtype`, SSM of its `mamba_ssm_cache_dtype` (an "auto"
    one the config's own, as the hybrid configs' verification sets it, else
    the conv dtype); None for a model without one or one run with
    speculative decoding (a widened conv state)."""
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    if vllm_config.speculative_config is not None:
        return None
    model = vllm_config.model_config
    text = model.hf_text_config
    cache = vllm_config.cache_config
    if getattr(text, "model_type", None) == "kimi_linear":
        # KDA's own rule (`kda_state_dtype`): conv the cache's dtype, the
        # recurrent state FP32 unless asked for float32 or bfloat16.
        conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(
            model.dtype,
            getattr(cache, "mamba_cache_dtype", "auto"),
            cache.mamba_ssm_cache_dtype,
        )
        return layout_from_config(text, _itemsize(conv), _itemsize(ssm))
    name = cache.mamba_ssm_cache_dtype
    if name == "auto":
        name = (
            getattr(text, "mamba_ssm_cache_dtype", None)
            or getattr(text, "mamba_ssm_dtype", None)
            or "auto"
        )
    conv, ssm = MambaStateDtypeCalculator._mamba_state_dtype(
        model.dtype, getattr(cache, "mamba_cache_dtype", "auto"), name
    )
    return layout_from_config(text, _itemsize(conv), _itemsize(ssm))


def pairing_problem(vllm_config, local_tp: int, remote_tp: int) -> str | None:
    """Why a hybrid of this model cannot pair these TP sizes over NIXL, None
    when it can: a Mamba2 or GDN layout that splits over both sizes, no
    speculative decoding, no MLA (but Kimi-K3's, at decode TP4 from prefill
    TP8 without prefix caching), and attention KV heads
    that split or replicate evenly on each side as NIXL maps them (the
    replicated-producer refusal of pure attention does not apply: a hybrid's
    mapping separates attention sources from state sources)."""
    model = vllm_config.model_config
    if model.use_mla:
        # Kimi-K3 (KDA beside an MLA latent the same on every rank) at decode
        # TP4 from prefill TP8 only: its attention blocks are then twice the
        # producer's (each read as two), no GCD split of the producer's.
        # Kimi-K3's text model is Kimi-Linear's (`kimi_linear`); real
        # transfers admit its KDA class only (`check_real_kv_layers`).
        if getattr(model.hf_text_config, "model_type", None) != "kimi_linear":
            return "an MLA hybrid other than Kimi-K3"
        if (local_tp, remote_tp) != (4, 8):
            return "Kimi-K3 at TP sizes other than prefill TP8 to decode TP4"
        if vllm_config.cache_config.enable_prefix_caching:
            return "Kimi-K3 at unequal TP sizes with prefix caching"
    if vllm_config.speculative_config is not None:
        return "speculative decoding"
    layout = layout_for(vllm_config)
    if layout is None:
        return "a recurrent state that is neither Mamba2 nor GDN (or has no known layout)"
    for tp in (local_tp, remote_tp):
        if not layout.divides(tp):
            return (
                f"a {layout.family} state of components {dict(layout.components)} and "
                f"{layout.ssm_heads} SSM heads that does not split over TP {tp}"
            )
    kv = model.get_total_num_kv_heads()
    for tp in (local_tp, remote_tp):
        if (tp <= kv and kv % tp) or (tp > kv and tp % kv):
            return f"attention KV heads ({kv}) that neither split nor replicate evenly over TP {tp}"
    return None


def state_layout_record(layout: StateLayout, rank: int, tp: int) -> dict:
    """The layout a state group's bytes follow, as the layout record logs it:
    the model's components and this rank's spans (what the harness
    reconciles against)."""
    return {
        "family": layout.family,
        "components": [list(c) for c in layout.components],
        "conv_rows": layout.conv_rows,
        "conv_itemsize": layout.conv_itemsize,
        "ssm_heads": layout.ssm_heads,
        "ssm_head_bytes": layout.ssm_head_bytes,
        "spans": [
            [s.kind, s.component, s.offset, s.length, s.first, s.count] for s in layout.spans(rank, tp)
        ],
    }


def check_spec(layout: StateLayout, spec, tp: int) -> list[str]:
    """How a registered `MambaSpec` differs from the layout at `tp` (empty:
    it is the layout's), field by field."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    problems = []
    want_type = {
        "mamba2": MambaAttentionBackendEnum.MAMBA2,
        "gdn": MambaAttentionBackendEnum.GDN_ATTN,
        "kda": MambaAttentionBackendEnum.GDN_ATTN,  # KDA inherits GDN's tag
    }[layout.family]
    if spec.mamba_type != want_type:
        problems.append(f"mamba_type {spec.mamba_type!r}, not {want_type!r}")
    if spec.tp_replicated:
        problems.append("a TP-replicated state")
    if not layout.divides(tp):
        return problems + [f"the state does not split over TP {tp}"]
    conv_local = sum(w // tp for _, w in layout.components)
    if tuple(spec.shapes[0]) != (conv_local, layout.conv_rows):
        problems.append(
            f"conv shape {tuple(spec.shapes[0])}, not {(conv_local, layout.conv_rows)} (DS)"
        )
    if _itemsize(spec.dtypes[0]) != layout.conv_itemsize:
        problems.append(f"conv element {_itemsize(spec.dtypes[0])} bytes, not {layout.conv_itemsize}")
    if not layout.ssm_shape or not layout.ssm_itemsize:
        problems.append("the layout has no SSM shape or element size")
        return problems
    want_shape = (layout.ssm_heads // tp, *layout.ssm_shape)
    if tuple(spec.shapes[1]) != want_shape:
        problems.append(f"SSM shape {tuple(spec.shapes[1])}, not {want_shape} (heads outermost)")
    if _itemsize(spec.dtypes[1]) != layout.ssm_itemsize:
        problems.append(f"SSM element {_itemsize(spec.dtypes[1])} bytes, not {layout.ssm_itemsize}")
    return problems


def state_span_pattern(
    base: str, layer: str, logical_block: int, layout: StateLayout, span: Span
) -> torch.Tensor:
    """A state span's expected bytes: its global columns / heads in order,
    keyed by (request, layer, component, coordinate, logical block). `layer`
    is the model's own layer name (or a transfer-only region's own identity),
    never a byte offset of this TP size."""
    from vllm.v1.worker.simulated.kv_pattern import _stream

    if span.kind == "conv":
        # One stream a component (global columns), sliced: cheap and the same
        # bytes of a column on any TP size.
        width = dict(layout.components)[span.component]
        whole = _stream(
            f"{base}|conv|{layer}|{span.component}|{logical_block}", width * span.unit_bytes
        )
        return whole[span.first * span.unit_bytes : (span.first + span.count) * span.unit_bytes]
    return torch.cat(
        [
            _stream(f"{base}|ssm|{layer}|h{head}|{logical_block}", span.unit_bytes)
            for head in range(span.first, span.first + span.count)
        ]
    )


# ----- ownership: who supplies a rank's part of a request's state -----


def head_global(tp: int, rank: int, total: int) -> tuple[int, ...]:
    """The global KV heads rank `rank` holds at `tp` (replicated beyond
    `total`)."""
    if tp <= total:
        first = rank * total // tp
        return tuple(range(first, first + total // tp))
    return (rank * total // tp,)


def candidate_producers(rank: int, local_tp: int, remote_tp: int) -> list[int]:
    """The producers of the pairing a decode rank may read: one when the
    producer has no more ranks (floor(r*R/L)), else the R/L contiguous ones
    assigned to it."""
    if remote_tp <= local_tp:
        return [rank * remote_tp // local_tp]
    k = remote_tp // local_tp
    return list(range(rank * k, (rank + 1) * k))


def attention_sources(
    rank: int, local_tp: int, remote_tp: int, total: int
) -> dict[int, tuple[int, ...]]:
    """producer -> the global heads it supplies a decode rank: for each local
    head the first candidate holding it (deduplicated within the candidates,
    equal TP included), the union over heads."""
    cands = candidate_producers(rank, local_tp, remote_tp)
    out: dict[int, list[int]] = {}
    for head in head_global(local_tp, rank, total):
        for p in cands:
            if head in head_global(remote_tp, p, total):
                out.setdefault(p, []).append(head)
                break
        else:
            raise ValueError(f"No producer of {cands} holds head {head}.")
    return {p: tuple(h) for p, h in out.items()}


def state_sources(
    rank: int, local_tp: int, remote_tp: int, width: int
) -> dict[int, tuple[int, int]]:
    """producer -> the global range [first, last) of a component (width
    columns or heads) it supplies: every candidate whose range overlaps the
    rank's."""
    lo, hi = rank * width // local_tp, (rank + 1) * width // local_tp
    out = {}
    for p in candidate_producers(rank, local_tp, remote_tp):
        a, b = p * width // remote_tp, (p + 1) * width // remote_tp
        first, last = max(lo, a), min(hi, b)
        if first < last:
            out[p] = (first, last)
    return out


def merge_ranges(ranges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """(offset, length) ranges, sorted, adjacent ones joined."""
    merged: list[tuple[int, int]] = []
    for offset, length in sorted(ranges):
        if merged and merged[-1][0] + merged[-1][1] == offset:
            merged[-1] = (merged[-1][0], merged[-1][1] + length)
        else:
            merged.append((offset, length))
    return merged


Piece = tuple[int, int, int, int]
"""(region, block, offset in the region's page, length): where bytes land."""


@dataclass
class ReadRecord:
    """One posted NIXL READ, normalized: the producer rank, the destination
    pieces of the selected descriptors (resolved through that handle's own
    prepared list, not raw descriptor ids), and whether it completed."""

    request: str
    rank: int | None
    handle: object
    pieces: list[Piece]
    done: bool = False
    engine: str | None = None
    """The producer engine the READ's remote handle belongs to (None: not
    known), so a request's coverage is reconciled with its own producer when
    the consumer has met several (a DP producer is one engine a rank)."""


def expected_pieces(
    rows: Sequence[dict],
    blocks: Sequence[int],
    rank: int,
    local_tp: int,
    remote_tp: int,
    layout: StateLayout,
    kv_heads: int,
) -> dict[int, list[Piece]]:
    """producer -> the destination pieces it must supply for `blocks` of the
    selected `rows`, from ownership alone. A row is {"region", "offset",
    "length", "state": bool, "heads": tuple|None, "span": Span|None} (an
    attention row: its heads' slabs of one page, `length` the page's; a state
    row: its span)."""
    out: dict[int, list[Piece]] = {}
    for row in rows:
        if row["state"]:
            span = row["span"]
            width = dict(layout.components)[span.component] if span.kind == "conv" else layout.ssm_heads
            for p, (first, last) in state_sources(rank, local_tp, remote_tp, width).items():
                off = row["offset"] + (first - span.first) * span.unit_bytes
                size = (last - first) * span.unit_bytes
                out.setdefault(p, []).extend((row["region"], b, off, size) for b in blocks)
        else:
            heads = row["heads"]
            slab = row["length"] // len(heads)
            for p, got in attention_sources(rank, local_tp, remote_tp, kv_heads).items():
                missing = [head for head in got if head not in heads]
                if missing:
                    raise ValueError(
                        f"producer {p} supplies heads {missing} that region "
                        f"{row['region']}'s page (heads {heads}) does not hold"
                    )
                for head in got:
                    off = row["offset"] + heads.index(head) * slab
                    out.setdefault(p, []).extend((row["region"], b, off, slab) for b in blocks)
    return out


def coverage_problems(
    expected: Mapping[int, Sequence[Piece]], reads: Sequence[ReadRecord]
) -> list[str]:
    """What a request's READs got wrong against ownership: per producer, per
    (region, block), the delivered ranges (adjacent ones joined) must be the
    expected ones exactly once; every posted READ must have completed, and
    none may name a producer or a piece nobody expects. Raw descriptor ids are
    never compared: only the resolved destination pieces."""
    problems = []
    got: dict[tuple, list[tuple[int, int]]] = {}
    for read in reads:
        if not read.done:
            problems.append(
                f"a READ from producer {read.rank} had not completed when the request was reported complete"
            )
        for region, block, offset, length in read.pieces:
            got.setdefault((read.rank, region, block), []).append((offset, length))
    want: dict[tuple, list[tuple[int, int]]] = {}
    for p, pieces in expected.items():
        for region, block, offset, length in pieces:
            want.setdefault((p, region, block), []).append((offset, length))
    for key in sorted(set(got) | set(want), key=repr):
        have, need = sorted(got.get(key, [])), sorted(want.get(key, []))
        if have == need:
            continue
        if merge_ranges(have) == merge_ranges(need) and sum(l for _, l in have) == sum(l for _, l in need):
            continue  # NIXL's descriptors cut the same bytes differently
        rank, region, block = key
        if not need:
            problems.append(f"unexpected delivery from producer {rank} to region {region} block {block}: {have}")
        elif not have:
            problems.append(f"nothing from producer {rank} for region {region} block {block}, expected {need}")
        elif sum(l for _, l in have) > sum(l for _, l in need):
            problems.append(
                f"duplicated or surplus bytes from producer {rank} in region {region} block {block}: {have} for {need}"
            )
        else:
            problems.append(f"missing bytes from producer {rank} in region {region} block {block}: {have} for {need}")
    return problems
