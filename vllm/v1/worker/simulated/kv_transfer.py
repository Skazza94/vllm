# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV connector transfers under simulated execution.

A connector's scheduler side runs unmodified; on the worker, a simulation that
moves no data completes each transfer after its modeled duration:
`SimulatedOffloadingWorker` for CPU offloading, and `kv_nixl` for NIXL.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.simulation import KVTransferTiming
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    get_offloading_group_ids,
    group_kv_bytes_per_block,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.rank_kv import (
    dcp_pair_refusal,
    dcp_scope_problems,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.simulated.timing import Clock, load_step_log

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Fitted records this many times slower than predicted (a connection's setup,
# or a NIXL agent's first postings to new memory) are outliers.
_OUTLIER_FACTOR = 20
_MAX_REFITS = 3
# Rows whose units per count vary within this ratio have a single shape.
_SAME_SHAPE_RATIO = 1.05


@dataclass(frozen=True)
class PostCost:
    """Host time a NIXL decode rank spends posting a step's reads, which holds
    its engine thread: `per_read_s` per read plus `per_run_s` per run of
    blocks that are consecutive on both instances. NIXL merges such a run
    into one descriptor per KV cache region, so the second term grows as the
    blocks a decode instance hands out scatter (once its cache has filled)."""

    per_read_s: float = 0.0
    per_run_s: float = 0.0

    def seconds(self, num_reads: int, num_runs: int) -> float:
        return num_reads * self.per_read_s + num_runs * self.per_run_s


def read_runs(local_block_ids: BlockIds, remote_block_ids: BlockIds) -> int:
    """Runs of blocks consecutive on both sides in one read, over its KV cache
    groups. Remote blocks the decode instance already holds (a local prefix
    hit) are not read: the last ones are paired with the local blocks."""
    runs = 0
    for local, remote in zip(local_block_ids, remote_block_ids):
        if not local:
            continue
        remote = remote[len(remote) - len(local) :]
        runs += 1 + sum(
            1
            for i in range(1, len(local))
            if local[i] != local[i - 1] + 1 or remote[i] != remote[i - 1] + 1
        )
    return runs


def nixl_reads(metadata: NixlConnectorMetadata) -> tuple[int, int]:
    """(reads, runs) a NIXL decode rank posts for a step's connector metadata."""
    num_reads = num_runs = 0
    for meta in metadata.reqs_to_recv.values():
        if meta.remote is None or not any(meta.local_block_ids):
            continue
        num_reads += 1
        num_runs += read_runs(meta.local_block_ids, meta.remote.block_ids)
    return num_reads, num_runs


@dataclass(frozen=True)
class TransferRate:
    """Duration of one transfer: `latency_s + num_bytes / bytes_per_s`."""

    latency_s: float
    bytes_per_s: float

    def seconds(self, num_bytes: int) -> float:
        return self.latency_s + num_bytes / self.bytes_per_s


def kv_transfer_rates(
    timing: KVTransferTiming | str,
) -> tuple[TransferRate, TransferRate | None]:
    """(load, store) rates of `SimulationConfig.kv_transfer_timing`; store is
    None if not given."""
    if isinstance(timing, str):
        return transfer_rates_from_log(timing)
    assert timing.load_gbps is not None
    load = TransferRate(timing.load_latency_us * 1e-6, timing.load_gbps * 1e9)
    store = (
        TransferRate(timing.store_latency_us * 1e-6, timing.store_gbps * 1e9)
        if timing.store_gbps is not None
        else None
    )
    return load, store


def kv_post_cost(timing: KVTransferTiming | str) -> PostCost:
    """NIXL read posting cost of `SimulationConfig.kv_transfer_timing`."""
    if isinstance(timing, str):
        return post_cost_from_log(timing)
    return PostCost(timing.post_us_per_read * 1e-6, timing.post_us_per_run * 1e-6)


def post_cost_from_log(path: str) -> PostCost:
    """Least-squares `PostCost` of the read postings recorded in a step log,
    without outliers."""
    rows = [
        (rec["kv_post"]["reads"], rec["kv_post"]["runs"], rec["kv_post"]["seconds"])
        for rec in load_step_log(path, kind="kv_post")
    ]
    if not rows:
        return PostCost()
    a = np.asarray(rows, dtype=np.float64)
    # Both terms grow with reads + runs: screen out the slowest postings
    # before the first fit, which they would otherwise dominate.
    unit_s = a[:, 2] / (a[:, 0] + a[:, 1])
    a = a[unit_s <= _OUTLIER_FACTOR * np.median(unit_s)]
    a, per_read_s, per_run_s = _fit(a)
    if _single_shape(a):
        logger.warning(
            "The reads recorded in %s are all alike, so the posting cost of "
            "reads into scattered blocks is extrapolated. Record until the "
            "decode instance's KV cache has filled and its blocks are reused.",
            path,
        )
    cost = PostCost(per_read_s, per_run_s)
    logger.info(
        "NIXL read posting from %s: %.1f us per read + %.2f us per run (%d records)",
        path,
        per_read_s * 1e6,
        per_run_s * 1e6,
        len(rows),
    )
    return cost


def _single_shape(a: np.ndarray) -> bool:
    units_per_count = a[:, 1] / a[:, 0]
    return units_per_count.max() <= _SAME_SHAPE_RATIO * units_per_count.min()


def _lstsq(a: np.ndarray) -> tuple[float, float]:
    """Least-squares seconds per count and per unit of (count, units,
    seconds) rows; all to the units if the two are not separable."""
    (per_count, per_unit), *_ = np.linalg.lstsq(a[:, :2], a[:, 2], rcond=None)
    if per_count < 0 or per_unit <= 0 or _single_shape(a):
        per_count, per_unit = 0.0, a[:, 2].sum() / a[:, 1].sum()
    return float(per_count), float(per_unit)


def _fit(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """`_lstsq`, refitted without outliers; also returns the rows kept."""
    per_count, per_unit = _lstsq(a)
    for _ in range(_MAX_REFITS):
        typical = a[:, 2] <= _OUTLIER_FACTOR * (
            a[:, 0] * per_count + a[:, 1] * per_unit
        )
        if typical.all():
            break
        a = a[typical]
        per_count, per_unit = _lstsq(a)
    return a, per_count, per_unit


def transfer_rates_from_log(path: str) -> tuple[TransferRate, TransferRate | None]:
    """(load, store) rates fitted by least squares, without outliers, to the
    transfers of a step log recorded with a KV connector; store is None if
    none were recorded."""
    rows: dict[str, list[tuple[float, float, float]]] = {"load": [], "store": []}
    for rec in load_step_log(path, kind="kv_transfers"):
        for direction, t in rec["kv_transfers"].items():
            if direction in rows:  # not a connector's `send`
                rows[direction].append((t["count"], t["bytes"], t["seconds"]))
    rates: list[TransferRate | None] = []
    for direction, direction_rows in rows.items():
        a = np.asarray(direction_rows, dtype=np.float64).reshape(-1, 3)
        if not a[:, 1].any():
            if direction == "store":
                rates.append(None)
                continue
            raise ValueError(f"{path} has no recorded KV {direction}s")
        # Bytes in GB keep the two columns on a similar scale.
        a = a[a[:, 1] > 0] * [1.0, 1e-9, 1.0]
        kept, latency_s, s_per_gb = _fit(a)
        rate = TransferRate(latency_s, 1e9 / s_per_gb)
        rates.append(rate)
        logger.info(
            "KV %ss from %s: %.1f us + %.2f GB/s (%d of %d steps)",
            direction,
            path,
            rate.latency_s * 1e6,
            rate.bytes_per_s * 1e-9,
            len(kept),
            len(direction_rows),
        )
    assert rates[0] is not None
    return rates[0], rates[1]


def nixl_runs(xfer_list) -> np.ndarray | None:
    """The prepared runs of a descriptor list in NIXL's strided form (Nx5:
    address, length, device, stride, count; one run a row, kept as given):
    each run's (length, stride, count). None for any other form (a flat
    list, whose runs NIXL builds itself)."""
    if not isinstance(xfer_list, np.ndarray) or xfer_list.ndim != 2 or xfer_list.shape[1] != 5:
        return None
    return xfer_list[:, [1, 3, 4]].astype(np.int64)


def nixl_merged_descriptors(
    local_runs: np.ndarray,
    local_ids: Sequence[int],
    remote_runs: np.ndarray,
    remote_ids: Sequence[int],
    skip_desc_merge: bool = False,
) -> int | None:
    """How many descriptors NIXL's `makeXferReq` makes of the selected
    indices (pinned NIXL 1683cf3b, src/core/nixl_agent.cpp:831-901): a
    selection i merges with the following ones while both sides' indices
    advance by one, within the prepared run holding each side's index (it
    stops at either run's end, even where the next run's addresses
    continue), and only when the local run is dense (stride = length) and
    both runs share that stride. `*_runs` are `nixl_runs`. None where NIXL
    would refuse the request (an index out of range, lengths that differ)."""
    starts = [np.cumsum(runs[:, 2]) - runs[:, 2] for runs in (local_runs, remote_runs)]
    sizes = [int(runs[:, 2].sum()) for runs in (local_runs, remote_runs)]
    n = len(local_ids)
    if n != len(remote_ids):
        return None

    def run_of(side: int, index: int) -> tuple[int, int, int, int]:
        runs = (local_runs, remote_runs)[side]
        r = int(np.searchsorted(starts[side], index, side="right")) - 1
        return int(runs[r, 0]), int(runs[r, 1]), int(starts[side][r]), int(runs[r, 2])

    merged, i = 0, 0
    while i < n:
        local, remote = int(local_ids[i]), int(remote_ids[i])
        if not (0 <= local < sizes[0] and 0 <= remote < sizes[1]):
            return None
        l_len, l_stride, l_start, l_count = run_of(0, local)
        r_len, r_stride, r_start, r_count = run_of(1, remote)
        if l_len != r_len:
            return None
        seq = 1
        if not skip_desc_merge and l_stride == l_len and l_stride == r_stride:
            cap = min(l_start + l_count - local, r_start + r_count - remote, n - i)
            while (
                seq < cap
                and int(local_ids[i + seq]) == local + seq
                and int(remote_ids[i + seq]) == remote + seq
            ):
                seq += 1
        merged += 1
        i += seq
    return merged


def selected_lengths(runs: np.ndarray | None, ids: Sequence[int]) -> tuple:
    """(selected bytes, {length: count}) of the descriptors `ids` pick from a
    prepared list's runs: exact whatever NIXL later merges; (None, None) when
    the runs are unknown. Both sides' lists carry the same lengths per
    descriptor, so the local list's are the READ's."""
    if runs is None:
        return None, None
    starts = np.cumsum(runs[:, 2]) - runs[:, 2]
    rows = np.searchsorted(starts, np.asarray(ids, dtype=np.int64), side="right") - 1
    lengths = runs[rows, 0]
    values, counts = np.unique(lengths, return_counts=True)
    return int(lengths.sum()), {int(v): int(c) for v, c in zip(values, counts)}


class NixlReadPeers:
    """Each NIXL READ's producer TP rank, local slot and descriptors, beside
    its telemetry in the worker's stats (`peer`, `plan`): the hooks wrap the
    wrapper's `make_prepped_xfer` (the handle's remote side names the rank,
    its local side the slot a heterogeneous-TP split writes) and
    `get_xfer_telemetry` (read just before the stats record that READ).

    A plan is [slot, selected, predicted, selected bytes, {length: count}]: the indices selected (vLLM's
    descriptors, before NIXL merges them) and the descriptors NIXL makes of
    them, predicted from the prepared runs of both lists
    (`nixl_merged_descriptors`), independently of the telemetry's measured
    count; None when a list's runs are unknown. `prep_xfer_dlist` is wrapped
    to keep each prepared list's runs; the local list prepared at
    registration, before this hook, is the worker's `src_blocks_data`."""

    def __init__(self, worker) -> None:
        self.worker = worker
        self.handles: dict[object, tuple[int | None, list[int | None]]] = {}
        # id(prepared list handle) -> its runs (handles are kept by the worker).
        self.runs: dict[int, np.ndarray | None] = {}
        wrapper = worker.nixl_wrapper
        make, telemetry = wrapper.make_prepped_xfer, wrapper.get_xfer_telemetry
        prep = getattr(wrapper, "prep_xfer_dlist", None)
        registered = getattr(worker, "src_xfer_handles_by_block_size", {}).get(
            getattr(worker, "block_size", None)
        )
        if registered is not None and not getattr(worker, "_mixed_mem_types", False):
            self.runs[id(registered)] = nixl_runs(getattr(worker, "src_blocks_data", None))

        def prep_xfer_dlist(agent_name, xfer_list, *args, **kw):
            handle = prep(agent_name, xfer_list, *args, **kw)
            self.runs[id(handle)] = nixl_runs(xfer_list)
            return handle

        def make_prepped_xfer(op, local, local_ids, remote, remote_ids, *args, **kw):
            handle = make(op, local, local_ids, remote, remote_ids, *args, **kw)
            skip = kw.get("skip_desc_merge", args[2] if len(args) > 2 else False)
            local_runs, remote_runs = self.runs.get(id(local)), self.runs.get(id(remote))
            predicted = (
                nixl_merged_descriptors(local_runs, local_ids, remote_runs, remote_ids, bool(skip))
                if local_runs is not None and remote_runs is not None
                else None
            )
            self.handles[handle] = (
                self._rank(remote),
                [self._slot(local), len(local_ids), predicted, *selected_lengths(local_runs, local_ids)],
            )
            return handle

        def get_xfer_telemetry(handle):
            result = telemetry(handle)
            peer, plan = self.handles.pop(handle, (None, None))
            if result is not None:
                data = self.worker.xfer_stats.data
                data.setdefault("peer", []).append(peer)
                data.setdefault("plan", []).append(plan)
            return result

        wrapper.make_prepped_xfer = make_prepped_xfer
        wrapper.get_xfer_telemetry = get_xfer_telemetry
        if prep is not None:
            wrapper.prep_xfer_dlist = prep_xfer_dlist

    def _rank(self, remote) -> int | None:
        for ranks in self.worker.dst_xfer_side_handles.values():
            for rank, handle in ranks.items():
                if handle == remote:
                    return int(rank)
        return None

    def _slot(self, local) -> int:
        for handles in getattr(self.worker, "src_xfer_handles_by_tp_ratio", {}).values():
            if local in handles:
                return handles.index(local)
        return 0


def completed_transfers(
    kv_output: KVConnectorOutput | None,
) -> dict[str, dict[str, float]] | None:
    """The KV transfers that completed in a step, per direction, as a step log's
    `kv_transfers` record holds them: CPU offloading's loads and stores, a
    NIXL decode instance's reads (`load`) or a Mooncake prefill instance's
    writes (`send`), the latter two with each transfer's bytes, seconds,
    seconds before it (NIXL: posting it; Mooncake: from its requests' blocks
    being ready to the write starting), descriptors and end (Mooncake, on
    time.monotonic()), and the transfers that failed (Mooncake: each one's
    bytes, descriptors and start, `failed_each`); a NIXL producer's requests
    released by their lease's expiry and notifications after it (`lease`)."""
    if kv_output is None:
        return None
    meta = kv_output.kv_connector_worker_meta
    stats = kv_output.kv_connector_stats
    if isinstance(meta, OffloadingWorkerMetadata):
        transfer_stats = meta.transfer_stats
        transfers = {
            direction: {"count": len(s.sizes), "bytes": s.bytes, "seconds": s.time}
            for direction, s in (
                ("load", transfer_stats.load),
                ("store", transfer_stats.store),
            )
            if s.sizes
        }
    elif isinstance(stats, (NixlKVConnectorStats, MooncakeKVConnectorStats)):
        data = stats.data
        nixl = isinstance(stats, NixlKVConnectorStats)
        transfers = {}
        failed = len(data["num_failed_transfers"])
        if stats.num_successful_transfers or failed:
            unknown = [None] * stats.num_successful_transfers
            before = data["post_duration"] if nixl else data.get("wait_duration", unknown)
            ends = unknown if nixl else data.get("transfer_end", unknown)
            # The peer rank and its plan (Mooncake: the consumer written,
            # each region's plan; NIXL: the producer read and its READ, as
            # `NixlReadPeers` saw them).
            peers = data.get("peer") or unknown
            plans = data.get("plan") or unknown
            transfers["load" if nixl else "send"] = {
                "count": stats.num_successful_transfers,
                "bytes": sum(data["bytes_transferred"]),
                "seconds": sum(data["transfer_duration"]),
                "each": [
                    list(transfer)
                    for transfer in zip(
                        data["bytes_transferred"],
                        data["transfer_duration"],
                        before,
                        data["num_descriptors"],
                        ends,
                        peers,
                        plans,
                    )
                ],
                **({"failed": failed} if failed else {}),
            }
            failed_each = [
                list(entry) + [peer]
                for entry, peer in zip(
                    zip(
                        data.get("failed_bytes", []),
                        data.get("failed_descriptors", []),
                        data.get("failed_start", []),
                    ),
                    data.get("failed_peer") or [None] * failed,
                )
            ]
            if not nixl and failed_each:
                transfers["send"]["failed_each"] = [list(f) for f in failed_each]
        if nixl and (
            expired := (
                len(data["num_kv_expired_reqs"]),
                len(data["num_notifications_after_expiry"]),
            )
        ) != (0, 0):
            # A producer's blocks released by its lease, not its readers'
            # notifications, and notifications that came after it.
            transfers["lease"] = {
                "expired": expired[0],
                "late_notifications": expired[1],
            }
        if not nixl and data.get("zero_payload"):
            # Consumer ranks a replicated producer answered without writing.
            transfers.setdefault(
                "send", {"count": 0, "bytes": 0, "seconds": 0.0, "each": []}
            )["zero_payload"] = [list(z) for z in data["zero_payload"]]
        # Mooncake's consumer counts the transfers it waited for in vain.
        failed_recvs = 0 if nixl else len(data["num_failed_recvs"])
        if failed_recvs:
            transfers["load"] = {"count": 0, "bytes": 0, "seconds": 0.0, "failed": failed_recvs}
    else:
        return None
    return transfers or None


# The KV cache groups real transfers move and check: full attention (MLA's
# included) and sliding-window attention, whose blocks are the tokens' in
# order, a window's head left null; and a gated delta-net's or a Mamba2
# mixer's state, one block a request (mamba_cache_mode none).
REAL_KV_SPECS = (FullAttentionSpec, MLAAttentionSpec, SlidingWindowSpec, MambaSpec)
# DeepSeek-V4's beside those: its sliding windows and compressor states
# (SlidingWindowMLASpec, a SlidingWindowSpec) and its compressed latents and
# indexer keys (MLAAttentionSpec of several tokens a state), several kinds
# in one group (UniformTypeKVCacheSpecs).
V4_REAL_KV_SPECS = (SlidingWindowMLASpec, UniformTypeKVCacheSpecs)


def layer_specs(groups: Sequence[KVCacheGroupSpec]) -> dict[str, KVCacheSpec]:
    """Each layer's own KV cache spec: a group of several kinds
    (UniformTypeKVCacheSpecs) names each layer's."""
    specs = {}
    for group in groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            specs[name] = (
                spec.kv_cache_specs[name]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
    return specs


def packed_pages(
    kv_caches: dict[str, torch.Tensor], specs: dict[str, KVCacheSpec]
) -> dict[str, int]:
    """Each layer's bytes a block holds, as pattern checks cover them: its
    page (a gated delta-net's state padded to the attention page), or, in
    a packed layout (each block all of a group's layers, DeepSeek-V4's), its
    real page only, its padding unchecked."""
    pages = {}
    for name, tensor in kv_caches.items():
        spec = specs[name]
        stride = tensor.stride(0) * tensor.element_size()
        pages[name] = (
            spec.real_page_size_bytes
            if stride > spec.page_size_bytes
            else spec.page_size_bytes
        )
    return pages


def refuse_foreign_state_geometry(worker) -> None:
    """A hybrid's peer over NIXL must hold its states as this instance does:
    the same conv and SSM sizes (a native MTP's depth widens the conv state)
    and region block lengths. NIXL checks neither between equal TP sizes
    (its per-region check skips hybrids), and would read a state of another
    geometry at this one's offsets; the handshake refuses it before any
    READ, whether bytes are checked or not."""
    add = worker.add_remote_agent

    def add_remote_agent(metadata, *args, **kwargs):
        local = (tuple(worker._mamba_ssm_size), list(worker.block_len_per_layer))
        remote = (tuple(metadata.ssm_sizes), list(metadata.block_lens))
        if remote != local:
            raise ValueError(
                "A hybrid model's NIXL peer holds its states otherwise: conv and "
                f"SSM sizes {remote[0]} and block lengths {remote[1]}, not "
                f"{local[0]} and {local[1]} (e.g. another MTP depth)."
            )
        return add(metadata, *args, **kwargs)

    worker.add_remote_agent = add_remote_agent


def dcp_peer_refusal(
    vllm_config: VllmConfig,
    local_dcp_size: int,
    local_page: tuple[int, int],
    remote_dcp_size: int,
    metadata: NixlAgentMetadata,
) -> str | None:
    """Why a NIXL peer (its handshake `metadata`) cannot pair with this
    instance (`local_page`: its block size and kernel blocks a block), if it
    cannot. With decode context parallelism on either side both instances
    must be in its scope (`dcp_scope_problems`: this one's from its config,
    the peer's as its metadata states it, `simulated_scope`) and the peer
    must hold pages of this instance's tokens, each block one page (no
    kernel sub-blocks on either side): a rank's blocks are whole global
    pages, and the checks pair them by that."""
    remote_dcp = max(remote_dcp_size, metadata.dcp_size)
    if local_dcp_size == 1 and remote_dcp == 1:
        return None
    problems = dcp_scope_problems(vllm_config, transfers=True)
    scope = metadata.simulated_scope
    if scope is None:
        problems.append("a peer that states no simulated scope")
    problems.extend(f"the peer's {problem}" for problem in scope or ())
    if problems:
        return dcp_pair_refusal(local_dcp_size, remote_dcp, problems)
    remote = (metadata.block_size, metadata.physical_blocks_per_logical_kv_block)
    if local_page != (remote[0], 1) or remote[1] != 1:
        return (
            "A NIXL peer with decode context parallelism (local DCP "
            f"{local_dcp_size}, remote DCP {remote_dcp}) must hold pages of this "
            "instance's tokens, one kernel block each: block size and kernel "
            f"blocks a block {remote}, not {local_page}."
        )
    return None


def validate_dcp_peer(worker) -> None:
    """Refuse at the handshake, before any READ and whether bytes are checked
    or not, a NIXL peer `dcp_peer_refusal` refuses."""
    add = worker.add_remote_agent

    def add_remote_agent(
        metadata,
        remote_tp_rank=0,
        remote_tp_size=1,
        remote_dcp_size=1,
        *args,
        **kwargs,
    ):
        local_page = (worker.block_size, worker._physical_blocks_per_logical_kv_block)
        if refusal := dcp_peer_refusal(
            worker.vllm_config, worker.dcp_size, local_page, remote_dcp_size, metadata
        ):
            raise ValueError(refusal)
        return add(
            metadata, remote_tp_rank, remote_tp_size, remote_dcp_size, *args, **kwargs
        )

    worker.add_remote_agent = add_remote_agent


class NixlReadDestinations:
    """The local blocks each READ NIXL posts writes, by request, as NIXL
    selects them (`_read_blocks`'s spec): under decode context parallelism
    the blocks a check expects moved, apart from those it must find
    untouched; a notify-only READ names none. A request whose READs name
    regions is recorded None."""

    def __init__(self, worker) -> None:
        self.blocks: dict[str, list[set[int]] | None] = {}
        read = worker._read_blocks

        def _read_blocks(*args, **kwargs):
            spec = kwargs.get("read_spec", args[0] if args else None)
            request = kwargs.get("request_id")
            if spec.block_ids_by_region:
                self.blocks[request] = None
            elif (groups := self.blocks.setdefault(request, [])) is not None:
                groups.extend(set() for _ in spec.local_block_ids[len(groups) :])
                for group, ids in zip(groups, spec.local_block_ids):
                    group.update(int(i) for i in ids)
            return read(*args, **kwargs)

        worker._read_blocks = _read_blocks

    def take(self, req_id: str) -> list[set[int]] | None:
        """A request's destinations by group, forgotten; None if NIXL posted
        no READ for it or named regions."""
        return self.blocks.pop(req_id, None)


def validate_state_peer(worker, layout) -> None:
    """A hybrid's NIXL peer of another TP size must hold states of the same
    model at its own size: the peer's `ssm_sizes` (conv and SSM bytes, the
    metadata's aggregates) equal the layout's at its TP size, its state
    regions at least that long, as many regions as this instance (of equal
    block lengths at equal TP), and that size splits the layout. Conv
    components and the two dtypes are not in the peer's metadata: they are
    validated locally on each endpoint (`check_spec`, under the identical
    model/config assumption), not observed remotely. Replaces the equal-geometry
    guard (`refuse_foreign_state_geometry`) for eligible states; block sizes
    are enforced here (equal, or the peer's a divisor of this instance's)
    whether or not bytes are checked."""
    add = worker.add_remote_agent

    def add_remote_agent(metadata, remote_tp_rank=0, remote_tp_size=1, *args, **kwargs):
        # Equal block sizes, or the producer's a divisor of this instance's
        # (NIXL's block-size ratio: each attention block read as several, the
        # states 1:1), whether or not bytes are checked: NIXL's own assertion
        # for P TP > D TP is in its not-mamba branch, and the pattern checker
        # (absent from performance runs) is not the guard.
        if worker.block_size % metadata.block_size:
            raise ValueError(
                f"A hybrid model's NIXL peer has block size {metadata.block_size}, not this "
                f"instance's {worker.block_size} or a divisor of it: pairs of other block "
                "sizes are not supported (tokens per block would not correspond)."
            )
        if not layout.divides(remote_tp_size):
            raise ValueError(
                f"A {layout.family} peer of TP {remote_tp_size} does not split the model's state."
            )
        want = layout.local_sizes(remote_tp_size)
        if tuple(metadata.ssm_sizes) != want:
            raise ValueError(
                "A hybrid model's NIXL peer holds its states otherwise: conv and "
                f"SSM sizes {tuple(metadata.ssm_sizes)}, not {want} (the model's at TP "
                f"{remote_tp_size}; e.g. another MTP depth or dtype)."
            )
        local_lens = list(worker.block_len_per_layer)
        if len(metadata.block_lens) != len(local_lens):
            raise ValueError(
                f"A hybrid model's NIXL peer registers {len(metadata.block_lens)} regions, "
                f"not this instance's {len(local_lens)}."
            )
        # Its regions' block lengths, at this instance's block size.
        ratio = worker.block_size // metadata.block_size
        if remote_tp_size == worker.world_size and [
            length * ratio for length in metadata.block_lens
        ] != local_lens:
            raise ValueError(
                "A hybrid model's NIXL peer of equal TP holds its regions otherwise: "
                f"block lengths {list(metadata.block_lens)} at block size "
                f"{metadata.block_size}, not {local_lens} at {worker.block_size}."
            )
        need = sum(want)
        regions = [i for i in worker._ssm_region_indices if metadata.block_lens[i] < need]
        if regions:
            raise ValueError(
                f"A peer's state regions {regions} are shorter than its state ({need} bytes)."
            )
        return add(metadata, remote_tp_rank, remote_tp_size, *args, **kwargs)

    worker.add_remote_agent = add_remote_agent


# DSA's indexer keys: 128 FP8 bytes and a 4-byte scale a token.
DSA_INDEXER_HEAD_SIZE = 132


def _dsa_group_problems(kv_cache_config: KVCacheConfig) -> set[str]:
    """A DSA model's groups as built: each layer's latent, BF16 or packed
    FP8 (the sparse role) and, with an indexer, its uint8 keys and scales, MLA specs of one
    block size; a group of several kinds holds these two roles only."""
    from vllm.v1.kv_cache_interface import SparseCacheRole

    problems = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        members = (
            spec.kv_cache_specs
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else dict.fromkeys(group.layer_names, spec)
        )
        roles = {}
        for name, member in members.items():
            role = getattr(member, "cache_role", None)
            roles.setdefault(role, []).append(name)
            if type(member) is not MLAAttentionSpec:
                problems.add(f"DSA layer {name} of {type(member).__name__}")
            elif role == SparseCacheRole.SPARSE and not (
                member.dtype == torch.bfloat16
                or (member.dtype, member.cache_dtype_str) == (torch.uint8, "fp8_ds_mla")
            ):
                problems.add(
                    f"DSA latent {name} of {member.dtype} ({member.cache_dtype_str}), "
                    "not BF16 or packed FP8 (fp8_ds_mla)"
                )
            elif role == SparseCacheRole.INDEXER and (
                member.dtype != torch.uint8
                or member.head_size != DSA_INDEXER_HEAD_SIZE
            ):
                problems.add(
                    f"DSA indexer {name} of {member.dtype} x {member.head_size}, "
                    f"not uint8 x {DSA_INDEXER_HEAD_SIZE}"
                )
            elif role not in (SparseCacheRole.SPARSE, SparseCacheRole.INDEXER):
                problems.add(f"DSA layer {name} of role {role}")
        if SparseCacheRole.SPARSE not in roles:
            problems.add(f"a DSA group of no latent (roles {sorted(map(str, roles))})")
        if len({m.block_size for m in members.values()}) != 1:
            problems.add(
                "a DSA group of block sizes "
                f"{sorted({m.block_size for m in members.values()})}"
            )
    return problems


def check_real_kv_groups(
    kv_cache_config: KVCacheConfig,
    v4: bool = False,
    speculative_blocks: int = 0,
    dsa: bool = False,
) -> None:
    """Refuse real KV transfers of other KV cache groups (chunked local or
    RSWA attention, attention sinks), and of Mamba states other than a
    gated delta-net's or a Mamba2 mixer's (one block a request, or with
    prefix caching a block a state: mamba_cache_mode align) without
    speculative blocks other than a native MTP's (`speculative_blocks`, its
    depth: scratch slots NIXL does not move), whose blocks follow other
    rules; DeepSeek-V4's (`v4`) groups too, of its own kinds."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    states = (MambaAttentionBackendEnum.GDN_ATTN, MambaAttentionBackendEnum.MAMBA2)
    kinds = (
        REAL_KV_SPECS
        + (V4_REAL_KV_SPECS if v4 else ())
        # A DSA layer's latent and indexer keys: MLA specs of two pages.
        + ((UniformTypeKVCacheSpecs,) if dsa else ())
    )
    other = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        members = (
            list(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
        if type(spec) not in kinds or any(type(m) not in kinds for m in members):
            other.add(type(spec).__name__)
        elif isinstance(spec, MambaSpec) and (
            spec.mamba_type not in states
            or spec.num_speculative_blocks != speculative_blocks
        ):
            other.add(
                f"MambaSpec ({spec.mamba_type.name}, mamba_cache_mode "
                f"{spec.mamba_cache_mode}, {spec.num_speculative_blocks} "
                "speculative blocks)"
            )
    if dsa:
        other.update(_dsa_group_problems(kv_cache_config))
    if other:
        raise ValueError(
            "kv_transfer_mode real moves full and sliding-window attention KV, "
            "gated delta-net and Mamba2 states only, not "
            f"{'; '.join(sorted(other))}."
        )


def check_real_kv_layers(model: torch.nn.Module) -> None:
    """Refuse real KV transfers of a model whose recurrent layers are not
    Qwen's gated delta-net (`QwenGatedDeltaNetAttention` itself), Kimi-K3's
    delta attention (`KimiK3DeltaAttention`, packed like Qwen's: its q, k
    and v convolutions in one conv state, without RecoverSSM or speculative
    state) or Mamba2 (`MambaMixer2` itself, without ReplaySSM or speculative
    state): Kimi-Linear's and OLMo's delta-nets share their KV cache groups
    but not their states' rules. Each recurrent mixer must be the only one of
    its decoder layer (an element of the one module list holding them all),
    without attention beside it (Falcon-H1's parallel hybrid, Zamba2's shared
    transformer)."""
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
    from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention

    mixers = [m for m in model.modules() if isinstance(m, MambaBase)]
    if not mixers:
        return
    other = sorted(
        {
            type(m).__name__
            for m in mixers
            if type(m)
            not in (QwenGatedDeltaNetAttention, KimiK3DeltaAttention, MambaMixer2)
        }
    )
    if other:
        raise ValueError(
            "kv_transfer_mode real moves Qwen's gated delta-net "
            "(QwenGatedDeltaNetAttention), Kimi-K3's delta attention "
            "(KimiK3DeltaAttention) and Mamba2 (MambaMixer2) states only, "
            f"not {', '.join(other)}'s."
        )
    if any(
        type(m) is KimiK3DeltaAttention and (m.use_recoverssm or m.num_spec)
        for m in mixers
    ):
        raise ValueError(
            "kv_transfer_mode real moves Kimi-K3's delta attention states "
            "without RecoverSSM or speculative state only."
        )
    if any(
        type(m) is MambaMixer2 and (m.use_replayssm or m.num_spec) for m in mixers
    ):
        raise ValueError(
            "kv_transfer_mode real moves Mamba2 states without ReplaySSM or "
            "speculative state only."
        )
    # The decoder layers: the elements of the one module list whose
    # elements hold every recurrent mixer (a layer's modules taken whole,
    # shared ones included).
    lists = [
        layers
        for layers in model.modules()
        if isinstance(layers, torch.nn.ModuleList)
        and any(
            isinstance(m, MambaBase) for layer in layers for m in layer.modules()
        )
    ]
    held = {id(m) for layers in lists[:1] for layer in layers for m in layer.modules()}
    if len(lists) != 1 or any(id(m) not in held for m in mixers):
        raise ValueError(
            "kv_transfer_mode real needs a recurrent model's mixers in one list "
            "of decoder layers, not this layout."
        )
    for index, layer in enumerate(lists[0]):
        modules = list(layer.modules())
        recurrent = sum(isinstance(m, MambaBase) for m in modules)
        attends = any(
            isinstance(m, AttentionLayerBase) and not isinstance(m, MambaBase)
            for m in modules
        )
        if recurrent > 1 or (recurrent and attends):
            raise ValueError(
                "kv_transfer_mode real moves one recurrent mixer's state a "
                "decoder layer, without attention beside it; decoder layer "
                f"{index} ({type(layer).__name__}) has {recurrent} mixers"
                f"{' and attention' if attends else ''}."
            )


def kv_layout(
    kv_caches: dict[str, torch.Tensor],
    num_blocks: int,
    block_size: int,
    layout: str,
    groups: Sequence[KVCacheGroupSpec] | None = None,
    transfer_block_bytes: int | None = None,
) -> dict[str, object]:
    """The KV cache a connector registers, for a step log's `kv_layout`
    record: what each block of each layer is in bytes and tokens, and, with
    the KV cache groups, each group's: the page a block moves of each layer
    (`block_bytes`; a gated delta-net's state padded to it,
    `real_page_bytes`) and the distance between blocks. Layers of different
    groups may share memory (a block is one group's at a time): the bytes
    count each storage once, as the connector registers them.

    In a packed layout (DeepSeek-V4's: each block all of a group's layers,
    at offsets within it) a group also names its `members` (each kind of
    layer: count, offsets, page and real page, dtype), the bytes pattern
    checks cover of a block (`checked_bytes_per_block`, the real pages) and
    those the connector moves of it (`transfer_bytes_per_block`, its
    registered region's block; None if unknown). A model whose decoder
    loads N - 1 of a prompt's N tokens and computes the last (a recurrent
    state, or a window spec with `bounded_replay`) has `prefill_backoff` 1."""
    first = next(iter(kv_caches.values()))
    layer_bytes = first.numel() * first.element_size()
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in kv_caches.values()
    }
    record: dict[str, object] = {
        "layers": len(kv_caches),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "layout": layout,
        "dtype": str(first.dtype).removeprefix("torch."),
        "shape": list(first.shape),
        "block_bytes": layer_bytes // num_blocks,
        "bytes": sum(storages.values()),
    }
    if groups is not None:
        specs = layer_specs(groups)
        described = []
        for group in groups:
            spec = group.kv_cache_spec
            layers = [name for name in group.layer_names if name in kv_caches]
            tensor = kv_caches[layers[0]] if layers else None
            own = specs[layers[0]] if layers else spec
            entry = {
                "kind": type(spec).__name__,
                "sliding_window": getattr(own, "sliding_window", None),
                "block_size": spec.block_size,
                "layers": len(layers),
                # Each layer's.
                "block_bytes": own.page_size_bytes,
                "real_page_bytes": own.real_page_size_bytes,
                "block_stride": (
                    tensor.stride(0) * tensor.element_size()
                    if tensor is not None
                    else 0
                ),
            }
            if isinstance(spec, MambaSpec):
                entry["mamba_type"] = spec.mamba_type.name
                # A native MTP's scratch slots: NIXL moves the running slot
                # (before them), whose clipped-read metadata index is k.
                entry["speculative_blocks"] = spec.num_speculative_blocks
            def kind_of(name: str) -> tuple:
                spec_n, tensor_n = specs[name], kv_caches[name]
                return (
                    str(getattr(spec_n, "cache_role", None) or type(spec_n).__name__),
                    str(tensor_n.dtype).removeprefix("torch."),
                    spec_n.page_size_bytes,
                    spec_n.real_page_size_bytes,
                    tensor_n.stride(0) * tensor_n.element_size(),
                )

            kinds_of = {name: kind_of(name) for name in layers}
            if len(set(kinds_of.values())) > 1 and not packed_members(
                kv_caches, specs, layers
            ):
                # Layers of several pages, each in its own region (a DSA
                # layer's latent and indexer keys): each kind's own geometry.
                kinds: dict[tuple, dict[str, object]] = {}
                for name, key in kinds_of.items():
                    kind = kinds.setdefault(key, {
                        "role": key[0].rsplit(".", 1)[-1].lower(),
                        "dtype": key[1],
                        "layers": 0,
                        "page_bytes": key[2],
                        "real_page_bytes": key[3],
                        "block_stride": key[4],
                    })
                    kind["layers"] += 1
                entry["kinds"] = list(kinds.values())
                entry["checked_bytes_per_block"] = sum(
                    k["page_bytes"] * k["layers"] for k in kinds.values()
                )
            if packed := packed_members(kv_caches, specs, layers):
                entry["members"] = packed
                entry["checked_bytes_per_block"] = sum(
                    member["real_page_bytes"] * member["layers"] for member in packed
                )
                entry["transfer_bytes_per_block"] = transfer_block_bytes
            described.append(entry)
        record["groups"] = described
        if backoff := prefill_backoff(specs):
            record["prefill_backoff"] = backoff
    return record


def prefill_backoff(specs: dict[str, KVCacheSpec]) -> int:
    """The trailing prompt tokens a producer of these layers does not
    compute, the consumer computing them: a recurrent state's last (decode
    derives the state after it), a bounded window replay's."""
    return int(
        any(
            spec.prefix_replay_tokens > 0 or isinstance(spec, MambaSpec)
            for spec in specs.values()
        )
    )


def registered_block_bytes() -> int | None:
    """The bytes a block of the connector's registered KV region moves, when
    it registered regions of one block length (NIXL's packed DeepSeek-V4
    cache: one region); None otherwise."""
    worker = getattr(get_kv_transfer_group(), "connector_worker", None)
    lengths = set(getattr(worker, "block_len_per_layer", None) or ())
    return lengths.pop() if len(lengths) == 1 else None


def packed_members(
    kv_caches: dict[str, torch.Tensor],
    specs: dict[str, KVCacheSpec],
    layers: Sequence[str],
) -> list[dict[str, object]]:
    """A packed group's kinds of layer, from the allocated tensors: each
    layer's page at its offset within the block, none overlapping another
    of the group or beyond the block. Empty when the group is not packed
    (each layer's block stride its page)."""
    placed = []
    for name in layers:
        tensor, spec = kv_caches[name], specs[name]
        size = tensor.element_size()
        stride = tensor.stride(0) * size
        if stride <= spec.page_size_bytes:
            continue
        placed.append((tensor.storage_offset() * size, name, spec, stride))
    if not placed:
        return []
    if len(placed) != len(layers) or len({stride for *_, stride in placed}) != 1:
        raise ValueError(
            "KV pattern checks need a group's layers packed alike: "
            f"{len(placed)} of {len(layers)} packed, block strides "
            f"{sorted({stride for *_, stride in placed})}."
        )
    stride = placed[0][3]
    end = 0
    members: dict[tuple, dict[str, object]] = {}
    for offset, name, spec, _ in sorted(placed, key=lambda p: p[0]):
        if offset < end or offset + spec.page_size_bytes > stride:
            raise ValueError(
                f"KV pattern checks need a packed group's pages apart within "
                f"its {stride}-byte block; layer {name}'s "
                f"{spec.page_size_bytes} bytes at {offset} overlap."
            )
        end = offset + spec.page_size_bytes
        dtype = str(kv_caches[name].dtype).removeprefix("torch.")
        key = (type(spec).__name__, spec.page_size_bytes, spec.real_page_size_bytes, dtype)
        member = members.setdefault(key, {
            "kind": key[0],
            "layers": 0,
            "offsets": [],
            "page_bytes": key[1],
            "real_page_bytes": key[2],
            "dtype": dtype,
            # Tokens a state (DeepSeek-V4's compressed latents: several).
            "tokens_per_state": float(getattr(spec, "tokens_per_state", 1)),
        })
        member["layers"] += 1
        member["offsets"].append(offset)
    return list(members.values())


@dataclass(frozen=True)
class RequestedRead:
    """A transfer of a producer's KV into a consumer request's blocks, as the
    consumer's connector metadata asks for it this step."""

    req_id: str
    block_ids: list[list[int]]
    """Local blocks it fills, by KV cache group that transfers KV; none when
    it only frees the producer's."""
    remote_request_id: str
    """The producer's request id. Mooncake's metadata does not carry it: the
    consumer's own id stands in (both share the client's base request id), or,
    for a request with no blocks (one that only frees the producer's, under
    an id of its own), the transfer id the two share."""
    first_logical_blocks: list[int]
    """Where each group's blocks start among the request's blocks (after a
    prefix hit, or a sliding window's head the connector does not read)."""
    remote_engine_id: str | None = None
    """The producer engine (NIXL)."""
    remote_block_counts: list[int] | None = None
    """The producer's blocks by group (NIXL): with a smaller producer block
    size, the sub-blocks of this instance's blocks it covers."""
    remote_dcp_size: int = 1
    """The producer's decode context parallel size (NIXL)."""
    remote_num_tokens: int | None = None
    """The tokens whose KV the producer hands off (NIXL): with decode context
    parallelism, which global pages hold KV and which are padding."""

    @property
    def blocks(self) -> list[int]:
        """Its blocks that hold KV, every group's (not the null block)."""
        return kv_blocks(self.block_ids)


def kv_blocks(block_ids: Sequence[Sequence[int]]) -> list[int]:
    """Every group's blocks but the null block (a sliding window's head)."""
    return [b for group in block_ids for b in group if b != NULL_BLOCK_ID]


def requested_reads(metadata: object) -> list[RequestedRead]:
    """The transfers a consumer requests in a step: NIXL's READs
    (``reqs_to_recv`` by request), or Mooncake's pulls, which the producer
    serves by writing (``reqs_to_recv`` by producer engine, then request)."""
    reads = []
    for key, value in getattr(metadata, "reqs_to_recv", {}).items():
        if isinstance(value, dict):  # Mooncake
            for req_id, meta in value.items():
                groups = [list(group) for group in meta.local_block_ids]
                remote = req_id if any(groups) else meta.transfer_id
                first = meta.first_block_indices or [0] * len(groups)
                reads.append(RequestedRead(req_id, groups, remote, list(first)))
        elif value.remote is not None:  # NIXL
            groups = [list(group) for group in value.local_block_ids]
            # Where the scheduler placed the pull (a window's clipped head
            # included), else its prefix hit.
            first = value.local_first_block_indices or value.local_num_computed_blocks
            reads.append(
                RequestedRead(
                    key,
                    groups,
                    value.remote.request_id,
                    list(first) if first else [0] * len(groups),
                    value.remote.engine_id,
                    [len(group) for group in value.remote.block_ids],
                    value.dcp_size,
                    value.remote.num_tokens,
                )
            )
    return reads


@dataclass
class GroupAllocations:
    """The blocks given to each request by KV cache group, for a step log's
    `kv_events` record (`allocated_groups`), where the flat `allocated` loses
    which group a block is and where it sits: models of several groups or of
    sliding windows. Each entry names the group, the place of its first block
    among the request's blocks of that group, and the blocks, in order, the
    null block (0, a window's head) holding its places. A new or resumed
    request's blocks start at 0, a running one's continue its blocks, and a
    pull lists the blocks the connector fills, where they start."""

    transfer_group_ids: Sequence[int]
    """The KV cache groups a pull lists its blocks by."""
    lengths: dict[str, list[int]] = field(default_factory=dict)
    """Each request's blocks so far, by group."""

    def events(
        self, scheduler_output: "SchedulerOutput", reads: list[RequestedRead]
    ) -> dict[str, list[dict[str, object]]]:
        allocated: dict[str, list[dict[str, object]]] = {}

        def record(req_id: str, group: int, first: int, blocks: Sequence[int]):
            if blocks:
                allocated.setdefault(req_id, []).append(
                    {"group": group, "first": first, "blocks": list(blocks)}
                )

        def extend(req_id: str, block_ids: Sequence[Sequence[int]], replace: bool):
            known = None if replace else self.lengths.get(req_id)
            lengths = known if known is not None else [0] * len(block_ids)
            for group, blocks in enumerate(block_ids):
                record(req_id, group, lengths[group], blocks)
                lengths[group] += len(blocks)
            self.lengths[req_id] = lengths

        for new in scheduler_output.scheduled_new_reqs:
            extend(new.req_id, new.block_ids, replace=True)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
            if new_ids is not None:
                extend(req_id, new_ids, replace=req_id in cached.resumed_req_ids)
        for read in reads:
            if read.blocks:
                for index, (first, blocks) in enumerate(
                    zip(read.first_logical_blocks, read.block_ids)
                ):
                    record(read.req_id, self.transfer_group_ids[index], first, blocks)
        for req_id in scheduler_output.finished_req_ids:
            self.lengths.pop(req_id, None)
        return allocated


def group_allocations(
    kv_cache_config: KVCacheConfig, prefix_caching: bool = False
) -> GroupAllocations | None:
    """A `GroupAllocations` for a model of several KV cache groups or of
    sliding windows, or with prefix caching (a pull starts past a prefix
    hit, the blocks it fills logged where they are); None for one group of
    full attention."""
    groups = kv_cache_config.kv_cache_groups
    if (
        len(groups) == 1
        and not isinstance(groups[0].kv_cache_spec, SlidingWindowSpec)
        and not prefix_caching
    ):
        return None
    return GroupAllocations(kv_cache_config.transfer_group_ids)


def scheduled_kv_events(
    scheduler_output: "SchedulerOutput",
    allocations: GroupAllocations | None = None,
) -> dict[str, object]:
    """A step's request lifecycle as the scheduler sent it, for a step log's
    `kv_events` record: blocks newly allocated to each request (every KV
    cache group's, but the null block; by group with `allocations`), the
    requests
    resumed after a preemption (their blocks replaced), the tokens each new
    or resumed request was admitted with as computed (its prefix hit, or
    with a pull its hit and the pulled tokens), the remote reads this
    step posts, the producer requests told to free their blocks without a
    read (by their remote id), and the requests finished (completed or
    aborted) since the previous step."""
    allocated: dict[str, list[int]] = {
        new.req_id: blocks
        for new in scheduler_output.scheduled_new_reqs
        if (blocks := kv_blocks(new.block_ids))
    }
    cached = scheduler_output.scheduled_cached_reqs
    for req_id, new_ids in zip(cached.req_ids, cached.new_block_ids):
        if new_ids is not None and (blocks := kv_blocks(new_ids)):
            allocated.setdefault(req_id, []).extend(blocks)
    requested = requested_reads(scheduler_output.kv_connector_metadata)
    reads, notified = [], []
    for read in requested:
        if read.blocks:
            reads.append(read.req_id)
            # Allocated when the request arrived; its first scheduled step,
            # after the transfer, lists them again.
            allocated.setdefault(read.req_id, read.blocks)
        else:
            # Aborted before decode allocated blocks: only frees prefill's.
            notified.append(read.remote_request_id)
    admitted = {new.req_id: new.num_computed_tokens for new in scheduler_output.scheduled_new_reqs}
    admitted.update(
        (req_id, computed)
        for req_id, computed in zip(cached.req_ids, cached.num_computed_tokens)
        if req_id in cached.resumed_req_ids
    )
    events: dict[str, object] = {"allocated": allocated, "admitted": admitted}
    if allocations is not None:
        events["allocated_groups"] = allocations.events(scheduler_output, requested)
    return {
        **events,
        "resumed": sorted(cached.resumed_req_ids),
        "reads_posted": sorted(reads),
        "prefill_notified": sorted(notified),
        "finished": sorted(scheduler_output.finished_req_ids),
    }


@dataclass
class ReadsInFlight:
    """Decode's posted reads, their blocks by request, until the connector
    reports them done or failed. Mooncake reports a failed load as finished
    and its blocks as invalid, in the same poll or an earlier one: a read any
    of whose blocks were reported invalid while it was in flight failed, when
    the connector reports it finished."""

    blocks: dict[str, list[int]] = field(default_factory=dict)
    lost: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.blocks)

    def events(self, kv_output: KVConnectorOutput | None) -> dict[str, object]:
        """The transfers the connector reported done in a step: reads finished
        or failed (decode) and sends whose blocks the producer may free
        (prefill: read by every consumer, or lease expired)."""
        if kv_output is None:
            return {}
        if invalid := kv_output.invalid_block_ids:
            self.lost.update(
                req_id for req_id, blocks in self.blocks.items() if invalid & set(blocks)
            )
        done = set(kv_output.finished_recving or ())
        failed = set(kv_output.failed_recving or ()) | (done & self.lost)
        for req_id in done | failed:
            self.blocks.pop(req_id, None)
            self.lost.discard(req_id)
        return {
            "reads_done": sorted(done - failed),
            "reads_failed": sorted(failed),
            "sends_done": sorted(kv_output.finished_sending or ()),
        }


_LOAD, _STORE = 0, 1


@dataclass
class _Transfer:
    job_id: int
    num_bytes: int
    start: float
    end: float


class SimulatedOffloadingWorker(OffloadingWorker):
    """Times KV transfers between device and host memory without moving data.

    As with the CPU offloading worker's copy streams, each direction is a
    serial queue, and a transfer starts only once the device has finished the
    work enqueued before it was submitted (which writes the blocks a store
    reads, or last reads the blocks a load overwrites)."""

    def __init__(
        self,
        clock: Clock,
        load: TransferRate,
        store: TransferRate,
        bytes_per_block: Sequence[int],
        device_free_at: Callable[[], float],
    ):
        self.clock = clock
        self.rates = (load, store)
        # Per offloaded KV cache group, in the order of GPULoadStoreSpec groups.
        self.bytes_per_block = bytes_per_block
        self.device_free_at = device_free_at
        # Indexed by _LOAD and _STORE.
        self._queues: tuple[deque[_Transfer], deque[_Transfer]] = (deque(), deque())
        self._free_at = [0.0, 0.0]

    def _submit(self, direction: int, job_id: int, gpu_spec: GPULoadStoreSpec) -> bool:
        num_bytes = sum(
            n * b for n, b in zip(gpu_spec.group_sizes, self.bytes_per_block)
        )
        start = max(self.clock.now(), self.device_free_at(), self._free_at[direction])
        end = start + self.rates[direction].seconds(num_bytes)
        self._free_at[direction] = end
        self._queues[direction].append(_Transfer(job_id, num_bytes, start, end))
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(_STORE, job_id, src_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(_LOAD, job_id, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        now = self.clock.now()
        results = []
        for queue in self._queues:
            while queue and queue[0].end <= now:
                t = queue.popleft()
                results.append(
                    TransferResult(
                        job_id=t.job_id,
                        success=True,
                        transfer_size=t.num_bytes,
                        transfer_time=t.end - t.start,
                    )
                )
        return results

    def wait(self, job_ids: set[int]) -> None:
        ends = [t.end for queue in self._queues for t in queue if t.job_id in job_ids]
        if ends:
            self.clock.sleep_until(max(ends))

    def next_completion(self) -> float | None:
        ends = [queue[0].end for queue in self._queues if queue]
        return min(ends, default=None)


class SimulatedTransfers(Protocol):
    def next_completion(self) -> float | None:
        """When the next in-flight transfer completes, if any."""
        ...


class SimulatedKVConnector(ActiveKVConnector):
    """The model runner's KV connector hooks, with the connector's
    worker-side transfers simulated instead of registering KV caches."""

    def __init__(self, vllm_config: VllmConfig, transfers: SimulatedTransfers):
        self.transfers = transfers
        super().__init__(vllm_config, {})

    def register_kv_caches(self, kv_caches_dict: dict[str, torch.Tensor]) -> None:
        connector = self.kv_connector
        if isinstance(connector, OffloadingConnector):
            assert connector.connector_worker is not None
            assert isinstance(self.transfers, SimulatedOffloadingWorker)
            connector.connector_worker.worker = self.transfers


def simulated_kv_connector(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    clock: Clock,
    device_free_at: Callable[[], float],
) -> SimulatedKVConnector:
    from vllm.v1.worker.simulated.kv_nixl import (
        NixlConnector,
        NixlPushConnector,
        SimulatedNixlWorker,
    )

    timing = vllm_config.simulation_config.kv_transfer_timing
    if timing is None:
        raise ValueError(
            "Simulated KV transfer needs simulation_config.kv_transfer_timing: "
            "transfer bandwidths, or the path of a step log recorded with the "
            "connector."
        )
    load, store = kv_transfer_rates(timing)
    connector = get_kv_transfer_group()
    transfers: SimulatedTransfers
    if isinstance(connector, OffloadingConnector):
        if store is None:
            raise ValueError("KV offloading needs kv_transfer_timing store_gbps.")
        groups = kv_cache_config.kv_cache_groups
        transfers = SimulatedOffloadingWorker(
            clock,
            load,
            store,
            bytes_per_block=[
                group_kv_bytes_per_block(groups[group_id])
                for group_id in get_offloading_group_ids(kv_cache_config)
            ],
            device_free_at=device_free_at,
        )
    elif isinstance(connector, NixlConnector | NixlPushConnector):
        assert isinstance(connector.connector_worker, SimulatedNixlWorker)
        transfers = connector.connector_worker
        transfers.configure(clock, load, kv_post_cost(timing))
    else:
        raise ValueError(
            "Simulated execution supports the OffloadingConnector and the "
            f"NixlConnector, not {type(connector).__name__}."
        )
    return SimulatedKVConnector(vllm_config, transfers)
