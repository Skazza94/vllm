# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-level checks of real KV transfers under simulated compute.

Simulated steps do not write the KV cache, so these checks give it known
content. A prefill instance (`PatternFiller`) writes a deterministic pattern
into each request's blocks as they are allocated. A decode instance
(`PatternChecker`) fills the blocks a remote read will land in with a
sentinel before the read is posted, then compares them with the pattern once
the connector reports the read finished, before any later step touches them.

The pattern of a block is a pseudo-random byte stream seeded by a 128-bit
hash of the prefill request's id, the layer and the block's position in the
request (its logical block), never of physical block ids, which differ
between the instances. Every byte depends on that whole identity, so a block
of another request, layer or position, or a shifted range, does not match
(by chance with negligible probability). Blocks are compared whole, so K
and V (packed in each block) and every layer are covered. Pattern bytes are
never zero and the sentinel is zero, so a block the read did not reach fails.

A model of several KV cache groups (full and sliding-window attention, say)
gives each group its own blocks; groups may share storage, a block being one
group's at a time. A block's bytes are its group's layers', its logical
block its place among the request's blocks of that group, where null blocks
(a sliding window's head, no longer attended) hold their places. The null
block itself holds no request's KV and is neither filled nor checked.

With prefix caching on, a block may hold KV computed for another request, so
its pattern is seeded by its content instead (`TokenChains`): the request's
tokens up to the block's end, which is what native prefix caching keys a full
block by. Pages are written as their tokens land: the producer's prompt
pages as they are allocated, the consumer's pages as its own steps complete
them (generation included), so a page reused later holds what a request with
that prefix expects. Prefix hits are compared read-only on both instances
(`kind` hit), apart from the transferred blocks, after the step's own writes:
the scheduler shares a block as soon as it schedules the tokens that fill it,
so a hit may be a page another request writes in the same step (each write
is recorded with its writer). A received page is compared before any later
write.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.simulated.kv_transfer import requested_reads

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)

SENTINEL = 0


def block_views(
    kv_caches: dict[str, torch.Tensor],
    num_blocks: int,
    page_bytes: Mapping[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Each layer's KV cache as `[num_blocks, page]` raw bytes.

    A block's page is what a connector moves of it: `page_bytes[layer]`
    (a gated delta-net's state padded to the attention page), at the
    tensor's block stride and storage offset, never beyond the page (the
    stride may cover another layer's). Without `page_bytes`, a block-major
    contiguous tensor's rows."""
    views = {}
    for name, tensor in kv_caches.items():
        if tensor.shape[0] != num_blocks or (
            page_bytes is None and not tensor.is_contiguous()
        ):
            raise ValueError(
                f"KV pattern checks need block-major contiguous KV tensors; "
                f"layer {name} has shape {tuple(tensor.shape)} for {num_blocks} "
                "blocks."
            )
        size = tensor.element_size()
        stride = tensor.stride(0) * size
        page = (
            page_bytes[name] if page_bytes is not None else tensor[0].numel() * size
        )
        storage = tensor.untyped_storage()
        offset = tensor.storage_offset() * size
        end = offset + (num_blocks - 1) * stride + page
        if stride < page or end > storage.nbytes():
            raise ValueError(
                f"KV pattern checks need each block's {page}-byte page within "
                f"its layer's {stride}-byte block stride and its storage; layer "
                f"{name} starts at byte {offset} of {storage.nbytes()}."
            )
        views[name] = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            storage, offset, (num_blocks, page), (stride, 1)
        )
    return views


def pattern(
    request_id: str,
    layer: str,
    logical_block: int,
    size: int,
    tp_rank: int = 0,
    identity: str | None = None,
) -> torch.Tensor:
    """The expected bytes of a request's logical block in one layer of a
    tensor-parallel rank (its KV heads): never zero, and another rank's
    differ, so a block delivered to the wrong rank fails. `identity` (a
    block's content, with prefix caching) stands for the request.

    PCG64 (numpy's portable generator) seeded by the identity's 128-bit
    BLAKE2b hash; both instances run the same numpy, so they agree."""
    base = identity if identity is not None else get_base_request_id(request_id)
    return _stream(f"{base}|tp{tp_rank}|{layer}|{logical_block}", size)


def _stream(name: str, size: int) -> torch.Tensor:
    seed = int.from_bytes(
        hashlib.blake2b(name.encode(), digest_size=16).digest(), "little"
    )
    stream = np.random.Generator(np.random.PCG64(seed)).integers(
        1, 256, size, dtype=np.uint8
    )
    return torch.from_numpy(stream)


HEAD_PATTERN = "head-v1"
"""A layer's page as its KV heads' slabs, each its global head's pattern."""
RANK_PATTERN = "rank-v0"
"""A layer's page as one pattern of its tensor-parallel rank (`pattern`)."""

HeadMap = Mapping[str, tuple[tuple[int, ...], int]]
"""Each layer's global KV heads, in their order in its page, and one head's
bytes (its slab: every token's K and V in the block); a layer absent keeps
its rank's pattern."""


def head_pattern(
    request_id: str,
    layer: str,
    logical_block: int,
    size: int,
    heads: tuple[tuple[int, ...], int],
    tp_rank: int = 0,
    identity: str | None = None,
) -> torch.Tensor:
    """A layer's page of `size` bytes holding global KV heads `heads[0]`,
    `heads[1]` bytes each: each slab its head's pattern, whichever rank holds
    it and however the connector re-shards it (a head held by several ranks,
    the same bytes on each); bytes past them (a page's padding), the rank's."""
    global_heads, slab = heads
    base = identity if identity is not None else get_base_request_id(request_id)
    parts = [_stream(f"{base}|h{head}|{layer}|{logical_block}", slab) for head in global_heads]
    rest = size - slab * len(global_heads)
    if rest < 0:
        raise ValueError(
            f"{layer}: {len(global_heads)} heads of {slab} bytes exceed its {size}-byte page"
        )
    if rest:
        parts.append(_stream(f"{base}|tp{tp_rank}|pad|{layer}|{logical_block}", rest))
    return torch.cat(parts)


def layer_pattern(
    request_id: str,
    layer: str,
    logical_block: int,
    size: int,
    tp_rank: int,
    identity: str | None,
    heads: "HeadMap | None",
) -> torch.Tensor:
    """A layer's expected page: its heads' (`heads`), else its rank's."""
    if heads is not None and layer in heads:
        return head_pattern(
            request_id, layer, logical_block, size, heads[layer], tp_rank, identity
        )
    return pattern(request_id, layer, logical_block, size, tp_rank, identity)


def group_pattern(views: Mapping[str, object], heads: "HeadMap | None") -> dict:
    """A group's pattern scheme and the global KV heads its layers hold: one
    scheme for all its layers, or "mixed"."""
    schemes = {HEAD_PATTERN if heads and layer in heads else RANK_PATTERN for layer in views}
    covered = {heads[layer][0] for layer in views if heads and layer in heads}
    return {
        "pattern": schemes.pop() if len(schemes) == 1 else "mixed",
        "global_heads": list(covered.pop()) if len(covered) == 1 else None,
    }


def head_map(
    specs: Mapping[str, "KVCacheSpec"],
    total_num_kv_heads: int,
    tp_rank: int,
    tp_size: int,
) -> dict[str, tuple[tuple[int, ...], int]]:
    """Each attention layer's global KV heads and slab, from its own spec:
    one slot per KV head of plain states (`num_kv_heads`, `head_size`), the
    heads vLLM gives this rank (rank r of tp <= K heads: K/tp from r*K/tp;
    of tp > K: head r*K/tp, replicated); an MLA layer one latent, global
    head 0, on every rank. A layer whose page packs otherwise (head slots,
    packed content, compressed states), a state or a layer whose heads do
    not match that sharding keeps its rank's pattern (absent)."""
    from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

    heads = {}
    for layer, spec in specs.items():
        if (
            not isinstance(spec, AttentionSpec)
            or spec.num_head_slots is not None
            or spec.state_content_bytes is not None
            or spec.tokens_per_state != 1
        ):
            continue
        if isinstance(spec, MLAAttentionSpec):
            if spec.num_kv_heads == 1:
                heads[layer] = ((0,), spec.unpadded_page_size_bytes)
            continue
        local = spec.num_kv_heads
        if tp_size <= total_num_kv_heads:
            first = tp_rank * total_num_kv_heads // tp_size
            expected = total_num_kv_heads // tp_size
            global_heads = tuple(range(first, first + local))
        else:
            expected = 1
            global_heads = (tp_rank * total_num_kv_heads // tp_size,)
        if local != expected:
            continue
        heads[layer] = (global_heads, spec.unpadded_page_size_bytes // local)
    return heads


class OutsideLifecycle(ValueError):
    """A block whose content the token chains cannot name."""


def _digest(previous: bytes, tokens: Sequence[int]) -> bytes:
    data = previous + np.asarray(tokens, dtype=np.int64).tobytes()
    return hashlib.blake2b(data, digest_size=16).digest()


class TokenChains:
    """With prefix caching: each request's token chain (its prompt, then the
    tokens this worker sampled for it) and its blocks' content identities.
    A full block's is that of its tokens and every earlier one, as native
    prefix caching keys it, so requests sharing a prefix expect the same bytes
    in its blocks, whichever request computed them; a partial block's names
    its tokens so far."""

    def __init__(self, block_sizes: Sequence[int]) -> None:
        self.block_sizes = list(block_sizes)
        """Each KV cache group's."""
        self.tokens: dict[str, list[int] | None] = {}
        self.prompt_lens: dict[str, int] = {}
        # Each request's full blocks' chained digests, by block size.
        self._full: dict[tuple[str, int], list[bytes]] = {}

    def admit(self, req_id: str, prompt_token_ids: Sequence[int] | None) -> None:
        """A request's first scheduling (a resumed one keeps its chain)."""
        if req_id in self.tokens:
            return
        self.tokens[req_id] = (
            None if prompt_token_ids is None else list(prompt_token_ids)
        )
        self.prompt_lens[req_id] = len(prompt_token_ids or ())

    def extend(self, req_id: str, tokens: Sequence[int]) -> None:
        if (chain := self.tokens.get(req_id)) is not None:
            chain.extend(tokens)

    def drop(self, req_id: str) -> None:
        self.tokens.pop(req_id, None)
        self.prompt_lens.pop(req_id, None)
        for size in set(self.block_sizes):
            self._full.pop((req_id, size), None)

    def identity(self, req_id: str, group: int, logical: int, n: int) -> str:
        """The content of a group's logical block holding its first `n`
        tokens."""
        tokens = self.tokens.get(req_id)
        size = self.block_sizes[group]
        end = logical * size + n
        if tokens is None or end > len(tokens):
            known = "no prompt token ids" if tokens is None else f"{len(tokens)} known"
            raise OutsideLifecycle(
                f"request {req_id}'s tokens up to {end} are unknown ({known})"
            )
        if n == 0:
            return f"content|B{size}|n0|i{logical}"
        full = self._full.setdefault((req_id, size), [])
        while len(full) < logical + (n == size):
            k = len(full)
            full.append(_digest(full[-1] if full else b"", tokens[k * size : (k + 1) * size]))
        if n == size:
            digest = full[logical]
        else:
            digest = _digest(full[logical - 1] if logical else b"", tokens[logical * size : end])
        return f"content|B{size}|n{n}|{digest.hex()}"

    def prompt_content(self, req_id: str, group: int, logical: int) -> str:
        """A block as the producer computes it: its prompt tokens."""
        size = self.block_sizes[group]
        n = min(size, max(0, self.prompt_lens.get(req_id, 0) - logical * size))
        return self.identity(req_id, group, logical, n)


def _lifecycle_failure(req_id: str, kind: str, error: Exception) -> dict:
    return {
        "req_id": req_id,
        "kind": kind,
        "ok": False,
        "error": f"outside the prefix-caching pattern lifecycle: {error}",
    }


class _ContentPages:
    """The lifecycle a filler and a checker share with prefix caching on
    (`chains` set): each request's block table, its prefix hits compared
    read-only, and (consumer) the pages its own steps complete written. A
    subclass writes and compares a block (`_write_block`, `_compare_block`)."""

    chains: "TokenChains | None"
    tables: dict[str, list[list[int]]]
    results: list[dict]
    # This step's writes (block: writing request) and the prefix hits to
    # compare once they are done.
    step_writes: dict[int, str]
    hits_due: list[tuple[str, list[int]]]

    def _write_block(
        self, req_id: str, group: int, logical: int, block: int, identity: str
    ) -> None:
        raise NotImplementedError

    def _compare_block(
        self, req_id: str, group: int, logical: int, block: int, identity: str
    ) -> tuple[int, int, list]:
        """(bytes compared, bytes wrong, mismatched entries)."""
        raise NotImplementedError

    def take_results(self) -> list[dict]:
        results, self.results = self.results, []
        for result in results:
            if not result["ok"]:
                logger.error("KV pattern check failed: %s", result)
        return results

    def _track(
        self, req_id: str, block_ids: Sequence[Sequence[int]], replace: bool
    ) -> None:
        known = None if replace else self.tables.get(req_id)
        if known is None:
            known = [[] for _ in block_ids]
        for held, new_ids in zip(known, block_ids):
            held.extend(new_ids)
        self.tables[req_id] = known

    def _prefix_hits(self, req_id: str, num_computed_tokens: int) -> list[int]:
        """The full blocks of a request's prefix hit, by group; queued for
        comparison after the step's writes. A hit ending inside a block (a
        partial hit, copied on write) is outside this lifecycle: native
        caching shares one only for an aligned Mamba group, which prefix
        caching here excludes."""
        assert self.chains is not None
        sizes = self.chains.block_sizes
        if any(num_computed_tokens % size for size in sizes):
            self.results.append(
                _lifecycle_failure(
                    req_id,
                    "hit",
                    OutsideLifecycle(
                        f"a prefix hit of {num_computed_tokens} tokens ends inside a "
                        f"block of {sizes} tokens (a partial hit)"
                    ),
                )
            )
        hits = [num_computed_tokens // size for size in sizes]
        self.hits_due.append((req_id, hits))
        return hits

    def _record_writes(self, req_id: str, blocks: Sequence[int]) -> None:
        if self.chains is not None:
            for block in blocks:
                self.step_writes[block] = req_id

    def _verify_due(self) -> None:
        """Compare the prefix hits found in this step, its writes done."""
        due, self.hits_due = self.hits_due, []
        for req_id, hits in due:
            if req_id in self.tables:
                self._verify_hits(req_id, hits)
        self.step_writes = {}

    def _verify_hits(self, req_id: str, hits: Sequence[int]) -> None:
        """Compare a request's leading `hits[g]` blocks of each group (prefix
        hits, never written by it): full blocks of its prompt's content,
        noting those another request wrote in this step."""
        assert self.chains is not None
        table = self.tables[req_id]
        groups, mismatched = [], []
        for group, count in enumerate(hits):
            ids, logicals, compared, wrong, written = [], [], 0, 0, []
            for logical in range(min(count, len(table[group]))):
                block = table[group][logical]
                if block == NULL_BLOCK_ID:
                    continue
                size = self.chains.block_sizes[group]
                try:
                    identity = self.chains.identity(req_id, group, logical, size)
                except OutsideLifecycle as e:
                    self.results.append(_lifecycle_failure(req_id, "hit", e))
                    return
                b, w, entries = self._compare_block(req_id, group, logical, block, identity)
                if (writer := self.step_writes.get(block)) is not None:
                    written.append([block, writer])
                ids.append(block)
                logicals.append(logical)
                compared += b
                wrong += w
                mismatched.extend(entries)
            if ids:
                groups.append(
                    {
                        "group": group,
                        "logical_blocks": logicals,
                        "block_ids": ids,
                        "blocks": len(ids),
                        "bytes": compared,
                        "mismatched_bytes": wrong,
                        # Hits on pages written in this step, and by whom.
                        "written_this_step": written,
                    }
                )
        if groups:
            wrong = sum(g["mismatched_bytes"] for g in groups)
            self.results.append(
                {
                    "req_id": req_id,
                    "kind": "hit",
                    "blocks": sum(g["blocks"] for g in groups),
                    "bytes": sum(g["bytes"] for g in groups),
                    "groups": groups,
                    "mismatched_bytes": wrong,
                    "mismatched_blocks": mismatched[:16],
                    "ok": wrong == 0,
                }
            )



def group_views(
    views: dict[str, torch.Tensor], groups: Sequence[Sequence[str]] | None
) -> list[dict[str, torch.Tensor]]:
    """Each KV cache group's layer views, `groups` naming each group's
    layers; None: one group of every layer."""
    if groups is None:
        return [views]
    return [{name: views[name] for name in names if name in views} for names in groups]


def _after_update(filler, scheduler_output: "SchedulerOutput") -> None:
    """A filler's step: each new or resumed request's blocks, a running
    one's new blocks; with prefix caching, a request's prefix hits are not
    written but compared, after every write of the step (a hit may be a page
    another request of the step fills)."""
    chains = filler.chains
    for new in scheduler_output.scheduled_new_reqs:
        hits = None
        if chains is not None:
            chains.admit(new.req_id, new.prompt_token_ids)
            filler._track(new.req_id, new.block_ids, replace=True)
            hits = filler._prefix_hits(new.req_id, new.num_computed_tokens)
        filler._extend(new.req_id, new.block_ids, replace=True, hits=hits)
    cached = scheduler_output.scheduled_cached_reqs
    for index, (req_id, new_ids) in enumerate(
        zip(cached.req_ids, cached.new_block_ids)
    ):
        if new_ids is not None:
            resumed = req_id in cached.resumed_req_ids
            hits = None
            if chains is not None:
                filler._track(req_id, new_ids, replace=resumed)
                if resumed:
                    hits = filler._prefix_hits(req_id, cached.num_computed_tokens[index])
            filler._extend(req_id, new_ids, replace=resumed, hits=hits)
    if chains is not None:
        filler._verify_due()
    for req_id in scheduler_output.finished_req_ids:
        filler.blocks.pop(req_id, None)
        filler.tables.pop(req_id, None)
        if chains is not None:
            chains.drop(req_id)


def _fill_identity(filler, req_id: str, group: int, logical: int) -> str | None:
    """A new block's identity: the request's (prefix caching off) or its
    prompt content's; raises OutsideLifecycle."""
    if filler.chains is None:
        return None
    return filler.chains.prompt_content(req_id, group, logical)


@dataclass
class PatternFiller(_ContentPages):
    """Prefill side: fills each request's blocks with its pattern as the
    scheduler allocates them (after the runner zeroed new blocks)."""

    views: dict[str, torch.Tensor]
    tp_rank: int = 0
    groups: Sequence[Sequence[str]] | None = None
    """Each KV cache group's layers; None: one group of every layer."""
    chains: TokenChains | None = None
    """With prefix caching: blocks seeded by their content."""
    heads: HeadMap | None = None
    """Each attention layer's global KV heads (`head_map`): its page their
    pattern; other layers their rank's."""
    blocks: dict[str, list[list[int]]] = field(default_factory=dict)
    """Each request's blocks by group, null ones included."""
    tables: dict[str, list[list[int]]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    step_writes: dict[int, str] = field(default_factory=dict)
    hits_due: list[tuple[str, list[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._groups = group_views(self.views, self.groups)

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        _after_update(self, scheduler_output)

    def _write_block(self, req_id, group, logical, block, identity) -> None:
        for layer, view in self._groups[group].items():
            view[block].copy_(
                layer_pattern(
                    req_id, layer, logical, view.shape[1], self.tp_rank, identity, self.heads
                )
            )

    def _compare_block(self, req_id, group, logical, block, identity):
        return _compare_views(
            self._groups[group], req_id, logical, block, self.tp_rank, identity, self.heads
        )

    def _extend(
        self,
        req_id: str,
        block_ids: Sequence[Sequence[int]],
        replace: bool,
        hits: Sequence[int] | None = None,
    ) -> None:
        known = None if replace else self.blocks.get(req_id)
        if known is None:
            known = [[] for _ in self._groups]
        for group, (held, new_ids) in enumerate(zip(known, block_ids)):
            start = len(held)
            held.extend(new_ids)
            for offset, block_id in enumerate(new_ids):
                logical = start + offset
                if block_id == NULL_BLOCK_ID or (hits and logical < hits[group]):
                    continue
                try:
                    identity = _fill_identity(self, req_id, group, logical)
                except OutsideLifecycle as e:
                    self.results.append(_lifecycle_failure(req_id, "fill", e))
                    break
                self._write_block(req_id, group, logical, block_id, identity)
                self._record_writes(req_id, [block_id])
        self.blocks[req_id] = known


def _compare_views(
    views: dict[str, torch.Tensor],
    req_id: str,
    logical: int,
    block: int,
    tp_rank: int,
    identity: str | None,
    heads: HeadMap | None = None,
) -> tuple[int, int, list]:
    compared = wrong = 0
    entries = []
    for layer, view in views.items():
        want = layer_pattern(req_id, layer, logical, view.shape[1], tp_rank, identity, heads)
        bad = int((view[block] != want).sum())
        compared += view.shape[1]
        if bad:
            wrong += bad
            entries.append([layer, logical, block, bad])
    return compared, wrong, entries


@dataclass
class _ExpectedGroup:
    group: int
    """Its KV cache group."""
    first_logical_block: int
    block_ids: list[int]


@dataclass
class _Expected:
    remote_request_id: str
    groups: list[_ExpectedGroup]

    @property
    def block_ids(self) -> list[int]:
        """Its blocks that hold KV, every group's."""
        return [
            block_id
            for group in self.groups
            for block_id in group.block_ids
            if block_id != NULL_BLOCK_ID
        ]


class _ConsumerPages(_ContentPages):
    """A checker's part of the lifecycle with prefix caching: a pull's
    blocks are compared once the request runs (its prompt tokens arrive with
    its first scheduling, after the pull), before any write of that step;
    the pages its steps complete are written, never a block a check still
    waits for; then its prefix hits are compared."""

    pending: dict[str, "_Expected"]
    awaiting: dict[str, "_Expected"]
    pull_starts: dict[str, dict[int, int]]

    def _record_pull_start(self, req_id: str, groups: Sequence["_ExpectedGroup"]) -> None:
        if self.chains is not None:
            # Where each group's pulled blocks start among the request's:
            # the blocks before them are its local prefix hit (or a window's
            # clipped head, null).
            self.pull_starts[req_id] = {g.group: g.first_logical_block for g in groups}

    def _admit_scheduled(self, scheduler_output: "SchedulerOutput") -> None:
        chains = self.chains
        if chains is None:
            return
        for new in scheduler_output.scheduled_new_reqs:
            chains.admit(new.req_id, new.prompt_token_ids)
            self._track(new.req_id, new.block_ids, replace=True)
            starts = self.pull_starts.pop(new.req_id, None)
            if starts is None:
                self._prefix_hits(new.req_id, new.num_computed_tokens)
            else:
                # Before the pull: decode's own hit (or a window's null head).
                self.hits_due.append(
                    (new.req_id, [starts.get(g, 0) for g in range(len(chains.block_sizes))])
                )
            if (expected := self.awaiting.pop(new.req_id, None)) is not None:
                self.results.append(self._compare_pull(new.req_id, expected))
        cached = scheduler_output.scheduled_cached_reqs
        for index, (req_id, new_ids) in enumerate(
            zip(cached.req_ids, cached.new_block_ids)
        ):
            if new_ids is None:
                continue
            resumed = req_id in cached.resumed_req_ids
            self._track(req_id, new_ids, replace=resumed)
            if resumed:
                self._prefix_hits(req_id, cached.num_computed_tokens[index])
        for req_id in scheduler_output.finished_req_ids:
            if (expected := self.awaiting.pop(req_id, None)) is not None:
                self.results.append(
                    {
                        "req_id": req_id,
                        "remote_request_id": expected.remote_request_id,
                        "ok": False,
                        "error": "finished before its pulled blocks were compared",
                    }
                )
            self.tables.pop(req_id, None)
            self.pull_starts.pop(req_id, None)
            chains.drop(req_id)

    def _compare_pull(self, req_id: str, expected: "_Expected") -> dict:
        """A finished pull, against its blocks' prompt content."""
        chains = self.chains
        assert chains is not None
        try:
            for group in expected.groups:
                for offset, block in enumerate(group.block_ids):
                    if block != NULL_BLOCK_ID:
                        chains.prompt_content(
                            req_id, group.group, group.first_logical_block + offset
                        )
        except OutsideLifecycle as e:
            return {
                **_lifecycle_failure(req_id, "transfer", e),
                "remote_request_id": expected.remote_request_id,
            }
        result = self._compare(
            req_id,
            expected,
            lambda group, logical: chains.prompt_content(req_id, group, logical),
        )
        return result

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        """Write the pages this step's computed tokens complete (after the
        runner zeroed the step's new blocks): each group's blocks whose last
        token the step computes, full blocks of the request's content (its
        computed tokens: a sampled token is computed by a later step). Then
        compare the step's prefix hits."""
        chains = self.chains
        if chains is None:
            return
        computed = {new.req_id: new.num_computed_tokens for new in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        waiting = {
            block
            for expected in (*self.pending.values(), *self.awaiting.values())
            for block in expected.block_ids
        }
        for req_id, start in computed.items():
            end = start + scheduler_output.num_scheduled_tokens.get(req_id, 0)
            table = self.tables.get(req_id)
            if table is None:
                continue
            for group, size in enumerate(chains.block_sizes):
                for logical in range(start // size, min(end // size, len(table[group]))):
                    block = table[group][logical]
                    if block == NULL_BLOCK_ID:
                        continue
                    if block in waiting:
                        raise RuntimeError(
                            f"Request {req_id} would write block {block}, whose "
                            "pulled bytes are not compared yet."
                        )
                    try:
                        identity = chains.identity(req_id, group, logical, size)
                    except OutsideLifecycle as e:
                        self.results.append(_lifecycle_failure(req_id, "write", e))
                        break
                    self._write_block(req_id, group, logical, block, identity)
                    self._record_writes(req_id, [block])
        self._verify_due()

    def sampled(self, req_ids: Sequence[str], tokens: Sequence[Sequence[int]]) -> None:
        """The tokens this step sampled, each request's chain's next."""
        if self.chains is not None:
            for req_id, new in zip(req_ids, tokens):
                self.chains.extend(req_id, new)

    def _finish(self, req_id: str, expected: "_Expected") -> dict | None:
        """A finished pull: compared now, or (prefix caching) once its
        request runs."""
        if self.chains is None:
            return self._compare(req_id, expected)
        self.awaiting[req_id] = expected
        return None


@dataclass
class PatternChecker(_ConsumerPages):
    """Decode side: sentinel before each remote read, comparison after it."""

    views: dict[str, torch.Tensor]
    tp_rank: int = 0
    groups: Sequence[Sequence[str]] | None = None
    """Each KV cache group's layers; None: one group of every layer."""
    transfer_group_ids: Sequence[int] | None = None
    """The KV cache groups a read's blocks are listed by (those that
    transfer KV); None: every group, in order."""
    chains: TokenChains | None = None
    """With prefix caching: blocks seeded by their content."""
    heads: HeadMap | None = None
    """Each attention layer's global KV heads (`head_map`): its page their
    pattern, whichever producer ranks wrote it; other layers their rank's."""
    pending: dict[str, _Expected] = field(default_factory=dict)
    # Pending reads whose blocks the connector reported invalid.
    lost: set[str] = field(default_factory=set)
    awaiting: dict[str, _Expected] = field(default_factory=dict)
    pull_starts: dict[str, dict[int, int]] = field(default_factory=dict)
    tables: dict[str, list[list[int]]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    step_writes: dict[int, str] = field(default_factory=dict)
    hits_due: list[tuple[str, list[int]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._groups = group_views(self.views, self.groups)

    def _write_block(self, req_id, group, logical, block, identity) -> None:
        for layer, view in self._groups[group].items():
            view[block].copy_(
                layer_pattern(
                    req_id, layer, logical, view.shape[1], self.tp_rank, identity, self.heads
                )
            )

    def _compare_block(self, req_id, group, logical, block, identity):
        return _compare_views(
            self._groups[group], req_id, logical, block, self.tp_rank, identity, self.heads
        )

    def before_step(self, scheduler_output: "SchedulerOutput") -> None:
        """Fill the blocks of the reads this step will post (call before the
        connector sees the step's metadata); with prefix caching, compare the
        requests that run for the first time (prefix hits, finished pulls)."""
        self._admit_scheduled(scheduler_output)
        for read in requested_reads(scheduler_output.kv_connector_metadata):
            if not read.blocks:
                continue
            groups = []
            for index, (first, block_ids) in enumerate(
                zip(read.first_logical_blocks, read.block_ids)
            ):
                group = (
                    self.transfer_group_ids[index]
                    if self.transfer_group_ids is not None
                    else index
                )
                held = [b for b in block_ids if b != NULL_BLOCK_ID]
                for view in self._groups[group].values():
                    view[held] = SENTINEL
                groups.append(_ExpectedGroup(group, first, list(block_ids)))
            self.pending[read.req_id] = _Expected(read.remote_request_id, groups)
            self._record_pull_start(read.req_id, groups)

    def after_step(self, kv_output: "KVConnectorOutput | None") -> list[dict]:
        """Compare the reads the connector reported finished or failed. A read
        reported finished whose blocks were reported invalid, in this poll or
        an earlier one, failed (Mooncake reports failed loads that way)."""
        ready = self.take_results()
        if kv_output is None:
            return ready
        if invalid := kv_output.invalid_block_ids:
            self.lost.update(
                req_id
                for req_id, expected in self.pending.items()
                if invalid & set(expected.block_ids)
            )
        results = []
        for req_id in kv_output.failed_recving or set():
            expected = self.pending.pop(req_id, None)
            self.lost.discard(req_id)
            if expected is not None:
                results.append(
                    {"req_id": req_id, "ok": False, "error": "receive failed"}
                )
        for req_id in kv_output.finished_recving or set():
            expected = self.pending.pop(req_id, None)
            if expected is None:
                continue
            if req_id in self.lost:
                self.lost.discard(req_id)
                results.append({"req_id": req_id, "ok": False, "error": "receive failed"})
            elif (result := self._finish(req_id, expected)) is not None:
                results.append(result)
        for result in results:
            if not result["ok"]:
                logger.error("KV pattern check failed: %s", result)
        return ready + results

    def _compare(self, req_id: str, expected: _Expected, identity_of=None) -> dict:
        """One request's result, its groups' nested: each group's blocks are
        compared over the group's layers, each (layer, block) once.
        `identity_of(group, logical)`: a block's content (prefix caching)."""
        groups = []
        mismatched_blocks = []
        for group in expected.groups:
            views = self._groups[group.group]
            group_mismatched = 0
            blocks = 0
            for offset, block_id in enumerate(group.block_ids):
                if block_id == NULL_BLOCK_ID:
                    continue
                blocks += 1
                logical = group.first_logical_block + offset
                identity = identity_of(group.group, logical) if identity_of else None
                for layer, view in views.items():
                    want = layer_pattern(
                        expected.remote_request_id,
                        layer,
                        logical,
                        view.shape[1],
                        self.tp_rank,
                        identity,
                        self.heads,
                    )
                    wrong = int((view[block_id] != want).sum())
                    if wrong:
                        group_mismatched += wrong
                        mismatched_blocks.append([layer, logical, block_id, wrong])
            groups.append(
                {
                    "group": group.group,
                    "first_logical_block": group.first_logical_block,
                    "blocks": blocks,
                    "layers": len(views),
                    "block_bytes": sum(view.shape[1] for view in views.values()),
                    "bytes": blocks * sum(view.shape[1] for view in views.values()),
                    "mismatched_bytes": group_mismatched,
                    **group_pattern(views, self.heads),
                }
            )
        mismatched_bytes = sum(group["mismatched_bytes"] for group in groups)
        return {
            "req_id": req_id,
            "remote_request_id": expected.remote_request_id,
            "blocks": sum(group["blocks"] for group in groups),
            # Decode's physical blocks: shows block reuse across requests.
            "block_ids": expected.block_ids,
            "layers": sum(group["layers"] for group in groups),
            "bytes": sum(group["bytes"] for group in groups),
            "groups": groups,
            "mismatched_bytes": mismatched_bytes,
            "mismatched_blocks": mismatched_blocks[:16],
            "ok": mismatched_bytes == 0,
        }


# A model of several KV cache groups over NIXL (a hybrid's, or attention groups,
# whose layers may be unequal): what the native worker moves of a request's
# blocks, not each layer's page. NIXL registers deduplicated regions (a
# backing slot shared by one layer of each group), moves a state as its conv
# sub-projections and SSM state (not the page's padding), and moves each of a
# group's blocks in every region of its kind, a region holding no layer of
# that group included (a transfer-only region). The filler and the checker
# follow the native descriptor selection itself.

POISON = 0xA5
"""What a producer writes into a state page's padding, which no transfer
moves: a receiver's padding keeps its sentinel, or the padding was copied."""


@dataclass(frozen=True)
class _Row:
    """One native local strided descriptor: a span of each block of a
    region."""

    view: torch.Tensor
    """`[num_blocks, length]` bytes at the native block stride."""
    region: int
    offset: int
    """Within the region's block."""
    state: bool
    """A recurrent state's (conv or SSM), past the attention descriptors."""


@dataclass(frozen=True)
class _Padding:
    view: torch.Tensor
    region: int


def _byte_view(
    storages: Sequence[torch.UntypedStorage],
    address: int,
    rows: int,
    length: int,
    stride: int,
) -> torch.Tensor:
    """`[rows, length]` bytes at `address`, `stride` apart, within one of the
    registered storages."""
    for storage in storages:
        start = storage.data_ptr()
        end = address + (rows - 1) * stride + length
        if start <= address and end <= start + storage.nbytes():
            return torch.empty(0, dtype=torch.uint8).set_(
                storage, address - start, (rows, length), (stride, 1)
            )
    raise ValueError(
        f"A native NIXL descriptor ({rows} blocks of {length} bytes from "
        f"{address:#x}, {stride} apart) lies outside the registered KV storage."
    )


class NativeSelection:
    """A native NIXL worker's selection of a request's blocks: its local
    strided descriptors (`src_blocks_data`) and the descriptor ids
    `_compute_desc_ids` picks for a transfer group's blocks, in its order.
    Read once the worker registered the KV caches; P and D have the same
    geometry (equal TP, block sizes and one physical block a logical one),
    which a remote of another geometry would break and which is refused."""

    def __init__(
        self,
        worker,
        kv_caches: dict[str, torch.Tensor],
        groups: Sequence[Sequence[str]],
        transfer_group_ids: Sequence[int] | None = None,
    ) -> None:
        if transfer_group_ids is not None and list(transfer_group_ids) != list(
            range(len(groups))
        ):
            raise ValueError(
                "Pattern checks of several KV cache groups over NIXL need every KV cache "
                f"group transferred, not groups {list(transfer_group_ids)} of "
                f"{len(groups)}."
            )
        if worker._physical_blocks_per_logical_kv_block != 1:
            raise ValueError(
                "Pattern checks of several KV cache groups over NIXL need one physical "
                "block a logical one, not "
                f"{worker._physical_blocks_per_logical_kv_block}."
            )
        self.worker = worker
        self.groups = [list(names) for names in groups]
        storages = list(
            {
                t.untyped_storage().data_ptr(): t.untyped_storage()
                for t in kv_caches.values()
            }.values()
        )
        bases = list(worker.kv_caches_base_addr[worker.engine_id][worker.tp_rank])
        strides = list(worker.block_stride_per_layer)
        num_blocks = worker.num_blocks

        def region_of(address: int) -> int:
            for index, (base, stride) in enumerate(zip(bases, strides)):
                if base <= address < base + stride:
                    return index
            raise ValueError(f"No native region holds {address:#x}.")

        data = np.asarray(worker.src_blocks_data, dtype=np.int64)
        self.starts = np.cumsum(data[:, 4]) - data[:, 4]
        # Native order: every region's attention descriptors, then the states'.
        attention_descs = sum(worker.region_num_blocks)
        self.rows = []
        for (address, length, _, stride, count), start in zip(
            data.tolist(), self.starts.tolist()
        ):
            region = region_of(address)
            self.rows.append(
                _Row(
                    _byte_view(storages, address, count, length, stride),
                    region,
                    address - bases[region],
                    start >= attention_descs,
                )
            )
        # A region holds a group's layer when one of its layers' tensors
        # starts within the region's first block.
        self.members = [
            {
                region_of(kv_caches[name].data_ptr())
                for name in names
                if name in kv_caches
            }
            for names in self.groups
        ]
        # A state region's page beyond its conv and SSM spans: padding.
        self.padding: dict[int, _Padding] = {}
        for region in sorted(set(worker._ssm_region_indices)):
            end = max(
                row.offset + row.view.shape[1]
                for row in self.rows
                if row.region == region and row.state
            )
            page = worker.block_len_per_layer[region]
            if page > end:
                self.padding[region] = _Padding(
                    _byte_view(
                        storages,
                        bases[region] + end,
                        num_blocks,
                        page - end,
                        strides[region],
                    ),
                    region,
                )

    def select(
        self, group: int, block_ids: Sequence[int]
    ) -> list[tuple[int, int, int]]:
        """The native selection of a transfer group's blocks: (descriptor
        row, block id, the block's place among the request's), in the native
        order, duplicates included."""
        if NULL_BLOCK_ID in block_ids:
            raise ValueError("A selection of NIXL blocks holds no null block.")
        lists: list[list[int]] = [[] for _ in self.groups]
        lists[group] = list(block_ids)
        ids = self.worker._compute_desc_ids(lists, self.worker.num_blocks, None, 1)
        rows = np.searchsorted(self.starts, ids, side="right") - 1
        selected = []
        for index, (desc, row) in enumerate(zip(ids.tolist(), rows.tolist())):
            block = desc - int(self.starts[row])
            place = index % len(block_ids)
            if block != block_ids[place]:
                raise ValueError(
                    f"The native selection of group {group} put block {block} "
                    f"where block {block_ids[place]} was expected."
                )
            selected.append((row, block, place))
        return selected

    def order(self, block_ids: Mapping[int, Sequence[int]]) -> list[int]:
        """The native descriptor ids of several groups' blocks moved
        together, in the native order."""
        lists = [list(block_ids.get(group, ())) for group in range(len(self.groups))]
        if not any(lists):
            return []
        return [
            int(d)
            for d in self.worker._compute_desc_ids(lists, self.worker.num_blocks, None, 1)
        ]

    def name(self, group: int, row: int) -> str:
        """A span's identity: its group and native region (and the
        descriptor's place in it)."""
        r = self.rows[row]
        return f"g{group}.r{r.region}.o{r.offset}"

    def state_regions(self, selected) -> set[int]:
        """The regions whose state a selection moves: their padding is not
        moved."""
        rows = self.rows
        return {rows[row].region for row, _, _ in selected if rows[row].state}

    def regions(self, selected) -> set[int]:
        return {self.rows[row].region for row, _, _ in selected}

    def describe(self) -> list[dict]:
        """Each transfer group's selected spans a block, for the layout
        record: logical (a region holding one of its layers) or transfer-only."""
        groups = []
        for group in range(len(self.groups)):
            spans = []
            for row, _, _ in self.select(group, [NULL_BLOCK_ID + 1]):
                r = self.rows[row]
                spans.append(
                    {
                        "row": row,
                        "region": r.region,
                        "offset": r.offset,
                        "bytes": r.view.shape[1],
                        "logical": r.region in self.members[group],
                    }
                )
            padded = sorted(
                {s["region"] for s in spans if self.rows[s["row"]].state}
                & set(self.padding)
            )
            groups.append(
                {
                    "regions": len({s["region"] for s in spans}),
                    "transfer_only_regions": len(
                        {s["region"] for s in spans if not s["logical"]}
                    ),
                    "descriptors_per_block": len(spans),
                    "bytes_per_block": sum(s["bytes"] for s in spans),
                    "transfer_only_bytes_per_block": sum(
                        s["bytes"] for s in spans if not s["logical"]
                    ),
                    "padding_bytes_per_block": sum(
                        self.padding[r].view.shape[1] for r in padded
                    ),
                    "spans": spans,
                }
            )
        return groups


def _write_selected(selection, req_id, group, logical, block, tp_rank, identity):
    s = selection
    for row, b, _ in s.select(group, [block]):
        view = s.rows[row].view
        view[b].copy_(
            pattern(req_id, s.name(group, row), logical, view.shape[1], tp_rank, identity)
        )


def _compare_selected(selection, req_id, group, logical, block, tp_rank, identity):
    s = selection
    compared = wrong = 0
    entries = []
    for row, b, _ in s.select(group, [block]):
        view = s.rows[row].view
        want = pattern(req_id, s.name(group, row), logical, view.shape[1], tp_rank, identity)
        bad = int((view[b] != want).sum())
        compared += view.shape[1]
        if bad:
            wrong += bad
            entries.append([s.name(group, row), logical, b, bad])
    return compared, wrong, entries


@dataclass
class NativeFiller(_ContentPages):
    """Prefill side over NIXL: fills each selected span of a request's new
    blocks with its pattern and its state pages' padding with POISON."""

    selection: NativeSelection
    tp_rank: int = 0
    chains: TokenChains | None = None
    """With prefix caching: blocks seeded by their content."""
    blocks: dict[str, list[list[int]]] = field(default_factory=dict)
    tables: dict[str, list[list[int]]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    step_writes: dict[int, str] = field(default_factory=dict)
    hits_due: list[tuple[str, list[int]]] = field(default_factory=list)

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        _after_update(self, scheduler_output)

    def _write_block(self, req_id, group, logical, block, identity) -> None:
        _write_selected(self.selection, req_id, group, logical, block, self.tp_rank, identity)

    def _compare_block(self, req_id, group, logical, block, identity):
        return _compare_selected(
            self.selection, req_id, group, logical, block, self.tp_rank, identity
        )

    def _extend(self, req_id, block_ids, replace: bool, hits=None) -> None:
        s = self.selection
        known = None if replace else self.blocks.get(req_id)
        if known is None:
            known = [[] for _ in s.groups]
        for group, (held, new_ids) in enumerate(zip(known, block_ids)):
            start = len(held)
            held.extend(new_ids)
            # Each block's place among the request's, null blocks (a window's
            # head) and prefix hits holding theirs.
            places = [
                i
                for i, b in enumerate(new_ids)
                if b != NULL_BLOCK_ID and not (hits and start + i < hits[group])
            ]
            new_ids = [new_ids[i] for i in places]
            if not new_ids:
                continue
            selected = s.select(group, new_ids)
            try:
                identities = [
                    _fill_identity(self, req_id, group, start + i) for i in places
                ]
            except OutsideLifecycle as e:
                self.results.append(_lifecycle_failure(req_id, "fill", e))
                continue
            for row, block, place in selected:
                view = s.rows[row].view
                view[block].copy_(
                    pattern(
                        req_id,
                        s.name(group, row),
                        start + places[place],
                        view.shape[1],
                        self.tp_rank,
                        identities[place],
                    )
                )
            for region in s.state_regions(selected) & set(s.padding):
                s.padding[region].view[new_ids] = POISON
            self._record_writes(req_id, new_ids)
        self.blocks[req_id] = known


@dataclass
class NativeChecker(_ConsumerPages):
    """Decode side over NIXL: sentinel over each selected span and padding
    of a read's blocks before it is posted; after it, the spans must hold
    the pattern and the padding its sentinel."""

    selection: NativeSelection
    tp_rank: int = 0
    transfer_group_ids: Sequence[int] | None = None
    chains: TokenChains | None = None
    """With prefix caching: blocks seeded by their content."""
    pending: dict[str, _Expected] = field(default_factory=dict)
    lost: set[str] = field(default_factory=set)
    # The local descriptors of the READs NIXL posted, by producer request.
    posted: dict[str, list[int]] = field(default_factory=dict)
    awaiting: dict[str, _Expected] = field(default_factory=dict)
    pull_starts: dict[str, dict[int, int]] = field(default_factory=dict)
    tables: dict[str, list[list[int]]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    step_writes: dict[int, str] = field(default_factory=dict)
    hits_due: list[tuple[str, list[int]]] = field(default_factory=list)

    def _write_block(self, req_id, group, logical, block, identity) -> None:
        _write_selected(self.selection, req_id, group, logical, block, self.tp_rank, identity)

    def _compare_block(self, req_id, group, logical, block, identity):
        return _compare_selected(
            self.selection, req_id, group, logical, block, self.tp_rank, identity
        )

    def watch_reads(self) -> None:
        """Record each READ's local descriptors as NIXL posts it (its
        notification names the producer's request): the blocks it writes,
        which must be the checked ones, never a state's scratch slot."""
        wrapper = self.selection.worker.nixl_wrapper
        make = wrapper.make_prepped_xfer

        def make_prepped_xfer(op, local, local_ids, remote, remote_ids, *args, **kw):
            notif = kw.get("notif_msg", args[0] if args else b"")
            if isinstance(notif, bytes):
                notif = notif.decode()
            request = notif.rsplit(":", 1)[0]
            self.posted.setdefault(request, []).extend(int(i) for i in local_ids)
            return make(op, local, local_ids, remote, remote_ids, *args, **kw)

        wrapper.make_prepped_xfer = make_prepped_xfer

    def before_step(self, scheduler_output: "SchedulerOutput") -> None:
        self._admit_scheduled(scheduler_output)
        for read in requested_reads(scheduler_output.kv_connector_metadata):
            if read.blocks:
                self.expect(
                    read.req_id,
                    read.remote_request_id,
                    read.first_logical_blocks,
                    read.block_ids,
                )

    def expect(
        self,
        req_id: str,
        remote_request_id: str,
        first_logical_blocks: Sequence[int],
        block_ids: Sequence[Sequence[int]],
    ) -> None:
        """A read about to be posted into a request's own new blocks: their
        selected spans and padding take the sentinel."""
        s = self.selection
        groups = []
        for index, (first, ids) in enumerate(zip(first_logical_blocks, block_ids)):
            group = (
                self.transfer_group_ids[index]
                if self.transfer_group_ids is not None
                else index
            )
            held = [b for b in ids if b != NULL_BLOCK_ID]
            if held:
                selected = s.select(group, held)
                for row, block, _ in selected:
                    s.rows[row].view[block] = SENTINEL
                for region in s.state_regions(selected) & set(s.padding):
                    s.padding[region].view[held] = SENTINEL
            groups.append(_ExpectedGroup(group, first, list(ids)))
        self.pending[req_id] = _Expected(remote_request_id, groups)
        self._record_pull_start(req_id, groups)

    def after_step(self, kv_output: "KVConnectorOutput | None") -> list[dict]:
        """PatternChecker's, then each ended receive's posted descriptors
        released (a failed or lost read's too, never compared), another
        pending read's kept, and a read awaiting its comparison's
        (prefix caching) kept until it is compared."""
        ended = {}
        if kv_output is not None:
            for req_id in (kv_output.failed_recving or set()) | (
                kv_output.finished_recving or set()
            ):
                if (expected := self.pending.get(req_id)) is not None:
                    ended[req_id] = expected.remote_request_id
        results = PatternChecker.after_step(self, kv_output)
        for req_id, remote_request_id in ended.items():
            if req_id not in self.awaiting:
                self.posted.pop(remote_request_id, None)
        return results

    def _check_geometry(self) -> None:
        """The selection is the local one: a remote read whole by the same
        rank, at the same block size. NIXL's other geometries (split reads,
        another block size) are refused, not assumed."""
        w = self.selection.worker
        other = {
            engine: plan
            for engine, plan in w.tp_mappings.items()
            if engine != w.engine_id
            and (plan.all_source_ranks != (w.tp_rank,) or plan.local_consumers != 1)
        }
        sizes = set(w.src_xfer_handles_by_block_size) - {w.block_size}
        if other or sizes or w.src_xfer_handles_by_tp_ratio:
            raise ValueError(
                "Pattern checks of several KV cache groups over NIXL need a remote of "
                "equal TP and block size, not "
                f"source ranks {[p.all_source_ranks for p in other.values()]}, "
                f"block sizes {sorted(sizes)}, "
                f"split reads {sorted(w.src_xfer_handles_by_tp_ratio)}."
            )

    def _compare(self, req_id: str, expected: _Expected, identity_of=None) -> dict:
        s = self.selection
        self._check_geometry()
        groups = []
        mismatched_blocks = []
        for group in expected.groups:
            places = [i for i, b in enumerate(group.block_ids) if b != NULL_BLOCK_ID]
            held = [group.block_ids[i] for i in places]
            selected = s.select(group.group, held) if held else []
            # A state group moves its running slot only: NIXL drops a native
            # MTP's trailing scratch slots and keeps the one before them
            # (`blocks[:-k][-1:]`, the first). Its place is that slot's, not
            # the index the consumer's clipped-read metadata gives (k).
            first = (
                0
                if selected and all(s.rows[row].state for row, _, _ in selected)
                else group.first_logical_block
            )
            payload = transfer_only = wrong_total = 0
            for row, block, place in selected:
                r = s.rows[row]
                view = r.view
                logical = first + places[place]
                want = pattern(
                    expected.remote_request_id,
                    s.name(group.group, row),
                    logical,
                    view.shape[1],
                    self.tp_rank,
                    identity_of(group.group, logical) if identity_of else None,
                )
                wrong = int((view[block] != want).sum())
                payload += view.shape[1]
                if r.region not in s.members[group.group]:
                    transfer_only += view.shape[1]
                if wrong:
                    wrong_total += wrong
                    mismatched_blocks.append(
                        [
                            s.name(group.group, row),
                            first + places[place],
                            block,
                            wrong,
                        ]
                    )
            padding = padding_wrong = 0
            for region in s.state_regions(selected) & set(s.padding):
                view = s.padding[region].view
                padding += view.shape[1] * len(held)
                bad = int((view[held] != SENTINEL).sum())
                if bad:
                    padding_wrong += bad
                    mismatched_blocks.append([f"padding.r{region}", None, held, bad])
            groups.append(
                {
                    "group": group.group,
                    # Where the pull's metadata placed the blocks among the
                    # request's (a state's clipped running slot: index k, its
                    # slot physically before the scratch ones), and the
                    # producer's place their pattern is of.
                    "first_logical_block": group.first_logical_block,
                    "pattern_first_block": first,
                    "block_ids": held,
                    "blocks": len(held),
                    "layers": len(s.groups[group.group]),
                    "regions": len(s.regions(selected)),
                    "descriptors": len(selected),
                    "bytes": payload,
                    "transfer_only_bytes": transfer_only,
                    "padding_bytes": padding,
                    "mismatched_bytes": wrong_total + padding_wrong,
                    "padding_mismatched_bytes": padding_wrong,
                }
            )
        mismatched_bytes = sum(g["mismatched_bytes"] for g in groups)
        # The READs wrote these descriptors' blocks and no others (a
        # state's scratch slots included), in the native order of every
        # group's blocks together (region by region, then group by group).
        checked = s.order(
            {
                group.group: [b for b in group.block_ids if b != NULL_BLOCK_ID]
                for group in expected.groups
            }
        )
        posted = self.posted.pop(expected.remote_request_id, None)
        posted_ok = posted == checked
        return {
            "req_id": req_id,
            "remote_request_id": expected.remote_request_id,
            "blocks": sum(g["blocks"] for g in groups),
            "block_ids": expected.block_ids,
            "layers": sum(g["layers"] for g in groups),
            "bytes": sum(g["bytes"] for g in groups),
            "groups": groups,
            "posted_descriptors": None if posted is None else len(posted),
            "posted_matches_checked": posted_ok,
            "mismatched_bytes": mismatched_bytes,
            "mismatched_blocks": mismatched_blocks[:16],
            "ok": mismatched_bytes == 0 and posted_ok,
        }
