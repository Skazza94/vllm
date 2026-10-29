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
