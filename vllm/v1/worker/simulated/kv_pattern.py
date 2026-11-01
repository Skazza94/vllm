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
    request_id: str, layer: str, logical_block: int, size: int, tp_rank: int = 0
) -> torch.Tensor:
    """The expected bytes of a request's logical block in one layer of a
    tensor-parallel rank (its KV heads): never zero, and another rank's
    differ, so a block delivered to the wrong rank fails.

    PCG64 (numpy's portable generator) seeded by the identity's 128-bit
    BLAKE2b hash; both instances run the same numpy, so they agree."""
    base = get_base_request_id(request_id)
    identity = f"{base}|tp{tp_rank}|{layer}|{logical_block}".encode()
    seed = int.from_bytes(hashlib.blake2b(identity, digest_size=16).digest(), "little")
    stream = np.random.Generator(np.random.PCG64(seed)).integers(
        1, 256, size, dtype=np.uint8
    )
    return torch.from_numpy(stream)


def group_views(
    views: dict[str, torch.Tensor], groups: Sequence[Sequence[str]] | None
) -> list[dict[str, torch.Tensor]]:
    """Each KV cache group's layer views, `groups` naming each group's
    layers; None: one group of every layer."""
    if groups is None:
        return [views]
    return [{name: views[name] for name in names if name in views} for names in groups]


@dataclass
class PatternFiller:
    """Prefill side: fills each request's blocks with its pattern as the
    scheduler allocates them (after the runner zeroed new blocks)."""

    views: dict[str, torch.Tensor]
    tp_rank: int = 0
    groups: Sequence[Sequence[str]] | None = None
    """Each KV cache group's layers; None: one group of every layer."""
    blocks: dict[str, list[list[int]]] = field(default_factory=dict)
    """Each request's blocks by group, null ones included."""

    def __post_init__(self) -> None:
        self._groups = group_views(self.views, self.groups)

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        for new in scheduler_output.scheduled_new_reqs:
            self._extend(new.req_id, new.block_ids, replace=True)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is not None:
                resumed = req_id in cached.resumed_req_ids
                self._extend(req_id, new_ids, replace=resumed)
        for req_id in scheduler_output.finished_req_ids:
            self.blocks.pop(req_id, None)

    def _extend(
        self, req_id: str, block_ids: Sequence[Sequence[int]], replace: bool
    ) -> None:
        known = None if replace else self.blocks.get(req_id)
        if known is None:
            known = [[] for _ in self._groups]
        for views, held, new_ids in zip(self._groups, known, block_ids):
            start = len(held)
            held.extend(new_ids)
            for offset, block_id in enumerate(new_ids):
                if block_id == NULL_BLOCK_ID:
                    continue
                for layer, view in views.items():
                    view[block_id].copy_(
                        pattern(
                            req_id, layer, start + offset, view.shape[1], self.tp_rank
                        )
                    )
        self.blocks[req_id] = known


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


@dataclass
class PatternChecker:
    """Decode side: sentinel before each remote read, comparison after it."""

    views: dict[str, torch.Tensor]
    tp_rank: int = 0
    groups: Sequence[Sequence[str]] | None = None
    """Each KV cache group's layers; None: one group of every layer."""
    transfer_group_ids: Sequence[int] | None = None
    """The KV cache groups a read's blocks are listed by (those that
    transfer KV); None: every group, in order."""
    pending: dict[str, _Expected] = field(default_factory=dict)
    # Pending reads whose blocks the connector reported invalid.
    lost: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._groups = group_views(self.views, self.groups)

    def before_step(self, scheduler_output: "SchedulerOutput") -> None:
        """Fill the blocks of the reads this step will post (call before the
        connector sees the step's metadata)."""
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

    def after_step(self, kv_output: "KVConnectorOutput | None") -> list[dict]:
        """Compare the reads the connector reported finished or failed. A read
        reported finished whose blocks were reported invalid, in this poll or
        an earlier one, failed (Mooncake reports failed loads that way)."""
        if kv_output is None:
            return []
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
            else:
                results.append(self._compare(req_id, expected))
        for result in results:
            if not result["ok"]:
                logger.error("KV pattern check failed: %s", result)
        return results

    def _compare(self, req_id: str, expected: _Expected) -> dict:
        """One request's result, its groups' nested: each group's blocks are
        compared over the group's layers, each (layer, block) once."""
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
                for layer, view in views.items():
                    want = pattern(
                        expected.remote_request_id,
                        layer,
                        logical,
                        view.shape[1],
                        self.tp_rank,
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


# A hybrid model's KV over NIXL: what the native worker moves of a request's
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
                "Pattern checks of a hybrid model over NIXL need every KV cache "
                f"group transferred, not groups {list(transfer_group_ids)} of "
                f"{len(groups)}."
            )
        if worker._physical_blocks_per_logical_kv_block != 1:
            raise ValueError(
                "Pattern checks of a hybrid model over NIXL need one physical "
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
            raise ValueError("A hybrid model's NIXL blocks hold no null block.")
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


@dataclass
class NativeFiller:
    """Prefill side over NIXL: fills each selected span of a request's new
    blocks with its pattern and its state pages' padding with POISON."""

    selection: NativeSelection
    tp_rank: int = 0
    blocks: dict[str, list[list[int]]] = field(default_factory=dict)

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        for new in scheduler_output.scheduled_new_reqs:
            self._extend(new.req_id, new.block_ids, replace=True)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is not None:
                resumed = req_id in cached.resumed_req_ids
                self._extend(req_id, new_ids, replace=resumed)
        for req_id in scheduler_output.finished_req_ids:
            self.blocks.pop(req_id, None)

    def _extend(self, req_id, block_ids, replace: bool) -> None:
        s = self.selection
        known = None if replace else self.blocks.get(req_id)
        if known is None:
            known = [[] for _ in s.groups]
        for group, (held, new_ids) in enumerate(zip(known, block_ids)):
            start = len(held)
            held.extend(new_ids)
            new_ids = [b for b in new_ids if b != NULL_BLOCK_ID]
            if not new_ids:
                continue
            selected = s.select(group, new_ids)
            for row, block, place in selected:
                view = s.rows[row].view
                view[block].copy_(
                    pattern(
                        req_id,
                        s.name(group, row),
                        start + place,
                        view.shape[1],
                        self.tp_rank,
                    )
                )
            for region in s.state_regions(selected) & set(s.padding):
                s.padding[region].view[new_ids] = POISON
        self.blocks[req_id] = known


@dataclass
class NativeChecker:
    """Decode side over NIXL: sentinel over each selected span and padding
    of a read's blocks before it is posted; after it, the spans must hold
    the pattern and the padding its sentinel."""

    selection: NativeSelection
    tp_rank: int = 0
    transfer_group_ids: Sequence[int] | None = None
    pending: dict[str, _Expected] = field(default_factory=dict)
    lost: set[str] = field(default_factory=set)
    # The local descriptors of the READs NIXL posted, by producer request.
    posted: dict[str, list[int]] = field(default_factory=dict)

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

    def after_step(self, kv_output: "KVConnectorOutput | None") -> list[dict]:
        """PatternChecker's, then each ended receive's posted descriptors
        released (a failed or lost read's too, never compared), another
        pending read's kept."""
        ended = {}
        if kv_output is not None:
            for req_id in (kv_output.failed_recving or set()) | (
                kv_output.finished_recving or set()
            ):
                if (expected := self.pending.get(req_id)) is not None:
                    ended[req_id] = expected.remote_request_id
        results = PatternChecker.after_step(self, kv_output)
        for remote_request_id in ended.values():
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
                "Pattern checks of a hybrid model over NIXL need a remote of "
                "equal TP and block size, not "
                f"source ranks {[p.all_source_ranks for p in other.values()]}, "
                f"block sizes {sorted(sizes)}, "
                f"split reads {sorted(w.src_xfer_handles_by_tp_ratio)}."
            )

    def _compare(self, req_id: str, expected: _Expected) -> dict:
        s = self.selection
        self._check_geometry()
        groups = []
        mismatched_blocks = []
        for group in expected.groups:
            held = [b for b in group.block_ids if b != NULL_BLOCK_ID]
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
                want = pattern(
                    expected.remote_request_id,
                    s.name(group.group, row),
                    first + place,
                    view.shape[1],
                    self.tp_rank,
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
                            first + place,
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
        # state's scratch slots included), in the selection's order.
        checked = [
            int(s.starts[row]) + block
            for group in expected.groups
            for row, block, _ in s.select(
                group.group, [b for b in group.block_ids if b != NULL_BLOCK_ID]
            )
            if any(b != NULL_BLOCK_ID for b in group.block_ids)
        ]
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
