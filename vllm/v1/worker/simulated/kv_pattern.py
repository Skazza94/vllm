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
"""

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput

logger = init_logger(__name__)

SENTINEL = 0


def block_views(
    kv_caches: dict[str, torch.Tensor], num_blocks: int
) -> dict[str, torch.Tensor]:
    """Each layer's KV cache as `[num_blocks, block_bytes]` raw bytes.

    Needs block-major, contiguous tensors (the NIXL connector's layout on CPU),
    which is what makes a block one contiguous range of bytes."""
    views = {}
    for name, tensor in kv_caches.items():
        if tensor.shape[0] != num_blocks or not tensor.is_contiguous():
            raise ValueError(
                f"KV pattern checks need block-major contiguous KV tensors; "
                f"layer {name} has shape {tuple(tensor.shape)} for {num_blocks} "
                "blocks."
            )
        views[name] = tensor.view(torch.uint8).reshape(num_blocks, -1)
    return views


def pattern(request_id: str, layer: str, logical_block: int, size: int) -> torch.Tensor:
    """The expected bytes of a request's logical block in one layer: never zero.

    PCG64 (numpy's portable generator) seeded by the identity's 128-bit
    BLAKE2b hash; both instances run the same numpy, so they agree."""
    identity = f"{get_base_request_id(request_id)}|{layer}|{logical_block}".encode()
    seed = int.from_bytes(hashlib.blake2b(identity, digest_size=16).digest(), "little")
    stream = np.random.Generator(np.random.PCG64(seed)).integers(
        1, 256, size, dtype=np.uint8
    )
    return torch.from_numpy(stream)


@dataclass
class PatternFiller:
    """Prefill side: fills each request's blocks with its pattern as the
    scheduler allocates them (after the runner zeroed new blocks)."""

    views: dict[str, torch.Tensor]
    blocks: dict[str, list[int]] = field(default_factory=dict)

    def after_update(self, scheduler_output: "SchedulerOutput") -> None:
        for new in scheduler_output.scheduled_new_reqs:
            self._extend(new.req_id, list(new.block_ids[0]), replace=True)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is not None:
                resumed = req_id in cached.resumed_req_ids
                self._extend(req_id, list(new_ids[0]), replace=resumed)
        for req_id in scheduler_output.finished_req_ids:
            self.blocks.pop(req_id, None)

    def _extend(self, req_id: str, block_ids: list[int], replace: bool) -> None:
        known = [] if replace else self.blocks.get(req_id, [])
        start = len(known)
        self.blocks[req_id] = known + block_ids
        for offset, block_id in enumerate(block_ids):
            for layer, view in self.views.items():
                view[block_id].copy_(
                    pattern(req_id, layer, start + offset, view.shape[1])
                )


@dataclass
class _Expected:
    remote_request_id: str
    first_logical_block: int
    block_ids: list[int]


@dataclass
class PatternChecker:
    """Decode side: sentinel before each remote read, comparison after it."""

    views: dict[str, torch.Tensor]
    pending: dict[str, _Expected] = field(default_factory=dict)

    def before_step(self, scheduler_output: "SchedulerOutput") -> None:
        """Fill the blocks of the reads this step will post (call before the
        connector sees the step's metadata)."""
        metadata = scheduler_output.kv_connector_metadata
        for req_id, meta in getattr(metadata, "reqs_to_recv", {}).items():
            if meta.remote is None or not any(meta.local_block_ids):
                continue
            block_ids = list(meta.local_block_ids[0])
            computed = meta.local_num_computed_blocks
            for view in self.views.values():
                view[block_ids] = SENTINEL
            self.pending[req_id] = _Expected(
                remote_request_id=meta.remote.request_id,
                first_logical_block=computed[0] if computed else 0,
                block_ids=block_ids,
            )

    def after_step(self, kv_output: "KVConnectorOutput | None") -> list[dict]:
        """Compare the reads the connector reported finished or failed."""
        if kv_output is None:
            return []
        results = []
        for req_id in kv_output.failed_recving or set():
            expected = self.pending.pop(req_id, None)
            if expected is not None:
                results.append(
                    {"req_id": req_id, "ok": False, "error": "receive failed"}
                )
        for req_id in kv_output.finished_recving or set():
            expected = self.pending.pop(req_id, None)
            if expected is not None:
                results.append(self._compare(req_id, expected))
        for result in results:
            if not result["ok"]:
                logger.error("KV pattern check failed: %s", result)
        return results

    def _compare(self, req_id: str, expected: _Expected) -> dict:
        mismatched_bytes = 0
        mismatched_blocks = []
        total = 0
        for layer, view in self.views.items():
            for offset, block_id in enumerate(expected.block_ids):
                logical = expected.first_logical_block + offset
                want = pattern(expected.remote_request_id, layer, logical, view.shape[1])
                wrong = int((view[block_id] != want).sum())
                total += view.shape[1]
                if wrong:
                    mismatched_bytes += wrong
                    mismatched_blocks.append([layer, logical, block_id, wrong])
        return {
            "req_id": req_id,
            "remote_request_id": expected.remote_request_id,
            "blocks": len(expected.block_ids),
            # Decode's physical blocks: shows block reuse across requests.
            "block_ids": expected.block_ids,
            "layers": len(self.views),
            "bytes": total,
            "mismatched_bytes": mismatched_bytes,
            "mismatched_blocks": mismatched_blocks[:16],
            "ok": mismatched_bytes == 0,
        }
