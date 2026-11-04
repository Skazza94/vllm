# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""How many descriptors NIXL makes of a READ's selected indices
(`nixl_merged_descriptors`, pinned NIXL 1683cf3b), predicted from the
prepared runs of both lists, and the READ hook's [slot, selected, predicted]
plan: against hand-made runs and, where NIXL with its Mooncake backend is
installed, against NIXL's own `make_prepped_xfer` and telemetry."""

import importlib.util
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.worker.simulated.kv_transfer import (
    NixlReadPeers,
    nixl_merged_descriptors,
    nixl_runs,
)


def strided(rows, base=0, device=0):
    """NIXL's Nx5 strided list (address, length, device, stride, count) of
    runs given as (offset, length, stride, count)."""
    return np.array(
        [(base + offset, length, device, stride, count) for offset, length, stride, count in rows],
        dtype=np.uint64,
    )


def regions(layers, blocks, length, stride, base=0):
    """vLLM's per-region layout (`_build_fa_local`): one run a layer region
    of `blocks` blocks, `stride` bytes apart, `length` bytes each (the whole
    page for MLA, a head slice for split attention); a block's descriptor is
    region * blocks + block."""
    return strided(
        [(layer * blocks * stride, length, stride, blocks) for layer in range(layers)], base
    )


def ids(layers, blocks, block_ids):
    return [layer * blocks + b for layer in range(layers) for b in block_ids]


def predict(local, local_ids, remote, remote_ids, skip=False):
    return nixl_merged_descriptors(nixl_runs(local), local_ids, nixl_runs(remote), remote_ids, skip)


# Two explicitly prepared dense runs at addresses 0 and 8 (length and stride
# 4, two blocks each): selections 1 and 2 are adjacent in memory (4 and 8)
# but cross the prepared-run boundary, so NIXL makes two descriptors.
BOUNDARY = strided([(0, 4, 4, 2), (8, 4, 4, 2)])

LAYERS, BLOCKS, PAGE = 3, 8, 64
MLA = regions(LAYERS, BLOCKS, PAGE, PAGE)
SLICED = regions(LAYERS, BLOCKS, PAGE // 2, PAGE)

CASES = {
    # name: (local runs, local ids, remote runs, remote ids, descriptors)
    "boundary": (BOUNDARY, [1, 2], BOUNDARY, [1, 2], 2),
    "within one run": (BOUNDARY, [0, 1], BOUNDARY, [2, 3], 1),
    # MLA, equal TP: the same consecutive blocks on both sides, one a layer.
    "mla consecutive": (MLA, ids(LAYERS, BLOCKS, [2, 3, 4, 5]), MLA, ids(LAYERS, BLOCKS, [2, 3, 4, 5]), 3),
    # MLA, other block ids on the producer, still consecutive: one a layer.
    "mla shifted": (MLA, ids(LAYERS, BLOCKS, [0, 1, 2]), MLA, ids(LAYERS, BLOCKS, [5, 6, 7]), 3),
    # Fragmented: two runs of blocks a layer.
    "mla fragmented": (MLA, ids(LAYERS, BLOCKS, [0, 1, 4, 5]), MLA, ids(LAYERS, BLOCKS, [0, 1, 4, 5]), 6),
    # Contiguous on one side only: nothing merges.
    "local contiguous only": (MLA, ids(LAYERS, BLOCKS, [0, 1, 2]), MLA, ids(LAYERS, BLOCKS, [0, 2, 4]), 9),
    "remote contiguous only": (MLA, ids(LAYERS, BLOCKS, [1, 3, 5]), MLA, ids(LAYERS, BLOCKS, [1, 2, 3]), 9),
    # CP1's head slices (length below stride): never merged.
    "head-sliced": (SLICED, ids(LAYERS, BLOCKS, [2, 3, 4, 5]), SLICED, ids(LAYERS, BLOCKS, [2, 3, 4, 5]), 12),
    # A selection reaching the last block of one layer and the first of the
    # next: the regions abut in memory, but each is its own run.
    "across regions": (MLA, [BLOCKS - 2, BLOCKS - 1, BLOCKS], MLA, [BLOCKS - 2, BLOCKS - 1, BLOCKS], 2),
}


@pytest.mark.parametrize("name", list(CASES))
def test_merged_descriptors_follow_nixls_prepared_runs(name):
    local, local_ids, remote, remote_ids, want = CASES[name]
    assert predict(local, local_ids, remote, remote_ids) == want
    # NIXL's skip_desc_merge: one descriptor a selected index.
    assert predict(local, local_ids, remote, remote_ids, skip=True) == len(local_ids)


def test_dense_runs_of_different_strides_do_not_merge():
    wide = regions(LAYERS, BLOCKS, PAGE, 2 * PAGE)
    selected = ids(LAYERS, BLOCKS, [0, 1, 2])
    assert predict(MLA, selected, wide, selected) == len(selected)
    assert predict(wide, selected, MLA, selected) == len(selected)


def test_a_request_nixl_would_refuse_has_no_prediction():
    assert predict(BOUNDARY, [4], BOUNDARY, [0]) is None  # out of range
    assert predict(SLICED, [0], MLA, [0]) is None  # lengths differ
    assert predict(BOUNDARY, [0, 1], BOUNDARY, [0]) is None
    assert nixl_runs([(0, 4, 0)]) is None  # a flat list: NIXL builds its runs


def test_the_hook_logs_selected_and_predicted_beside_the_measured_count():
    """The READ hook keeps each prepared list's runs (`prep_xfer_dlist`, and
    the local list registered before it, `src_blocks_data`) and logs
    [slot, selected, predicted]; a list whose runs it never saw gives no
    prediction."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
        NixlKVConnectorStats,
    )

    made: list[object] = []
    telemetry = {}

    class Handle:
        pass

    def make(op, local, local_ids, remote, remote_ids, *args, **kw):
        handle = Handle()
        made.append(handle)
        # A stand-in "measured" count: what NIXL would report.
        telemetry[handle] = SimpleNamespace(
            xferDuration=1000, postDuration=10, totalBytes=64,
            descCount=predict(MLA, local_ids, MLA, remote_ids) or len(local_ids),
        )
        return handle

    registered, remote, unknown = Handle(), None, Handle()
    wrapper = SimpleNamespace(
        make_prepped_xfer=make,
        get_xfer_telemetry=lambda handle: telemetry[handle],
        prep_xfer_dlist=lambda agent, xfer_list, **kw: Handle(),
    )
    worker = SimpleNamespace(
        nixl_wrapper=wrapper,
        xfer_stats=NixlKVConnectorStats(),
        block_size=16,
        src_xfer_handles_by_block_size={16: registered},
        src_blocks_data=MLA,
        dst_xfer_side_handles={},
    )
    NixlReadPeers(worker)
    remote = wrapper.prep_xfer_dlist("prefill", MLA, mem_type="DRAM")
    worker.dst_xfer_side_handles = {"prefill": {0: remote}}
    selected = ids(LAYERS, BLOCKS, [1, 2, 3])
    for local in (registered, unknown):
        handle = wrapper.make_prepped_xfer("READ", local, selected, remote, selected, b"req:1")
        worker.xfer_stats.record_transfer(wrapper.get_xfer_telemetry(handle))
    plans = worker.xfer_stats.data["plan"]
    # [slot, selected, predicted] then the selected bytes and length table,
    # exact whatever NIXL merges (None where the runs are unknown).
    assert [plan[:3] for plan in plans] == [[0, 9, 3], [0, 9, None]]
    assert plans[0][3:] == [576, {64: 9}]
    assert worker.xfer_stats.data["peer"] == [0, 0]


NATIVE = importlib.util.find_spec("nixl") is not None and os.path.exists(
    os.path.join(os.environ.get("NIXL_PLUGIN_DIR", "/opt/nixl/lib/plugins"), "libplugin_Mooncake.so")
)


@pytest.fixture(scope="module")
def agents():
    """Two NIXL agents over the Mooncake backend (TENT's hp_tcp on the
    loopback): A reads B's registered buffer."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield from _agents(monkeypatch)


def _agents(monkeypatch):
    monkeypatch.setenv("MC_USE_TENT", "1")
    monkeypatch.setenv("NIXL_MOONCAKE_IP_ADDR", "127.0.0.1")
    monkeypatch.setenv("NIXL_TELEMETRY_ENABLE", "y")
    disabled = {"enable": False}
    monkeypatch.setenv("MC_TENT_CONF", __import__("json").dumps({
        "rpc_server_hostname": "127.0.0.1",
        "transports": {"tcp": disabled, "rdma": disabled, "shm": disabled, "gds": disabled,
                       "ascend_direct": disabled,
                       "hp_tcp": {"enable": True, "bind_address": "127.0.0.1",
                                  "advertise_address": "127.0.0.1", "rail_addresses": ["127.0.0.1"]}},
    }))
    from nixl._api import nixl_agent, nixl_agent_config

    a, b = (nixl_agent(name, nixl_agent_config(backends=["Mooncake"])) for name in ("A", "B"))
    size = LAYERS * BLOCKS * 2 * PAGE
    src = np.zeros(size, dtype=np.uint8)
    dst = (np.arange(size) % 251).astype(np.uint8)
    a.register_memory([(src.ctypes.data, src.nbytes, 0, "")], "DRAM")
    b.register_memory([(dst.ctypes.data, dst.nbytes, 0, "")], "DRAM")
    a.add_remote_agent(b.get_agent_metadata())
    yield a, src, dst


def read(agent, local, local_ids, remote, remote_ids):
    handle = agent.make_prepped_xfer("READ", local, local_ids, remote, remote_ids)
    agent.transfer(handle)
    start = time.time()
    while agent.check_xfer_state(handle) == "PROC":
        assert time.time() - start < 30
        time.sleep(0.005)
    assert agent.check_xfer_state(handle) == "DONE"
    return handle


@pytest.mark.skipif(not NATIVE, reason="needs NIXL with its Mooncake backend")
def test_nixl_measures_the_predicted_descriptors(agents):
    """NIXL's own `make_prepped_xfer` and telemetry: each case's measured
    descriptors are the prediction, every selected byte landing."""
    a, src, dst = agents
    for name, (local_rows, local_ids, remote_rows, remote_ids, want) in CASES.items():
        local = a.prep_xfer_dlist("NIXL_INIT_AGENT", local_rows + np.array(
            [src.ctypes.data, 0, 0, 0, 0], dtype=np.uint64), mem_type="DRAM")
        remote = a.prep_xfer_dlist("B", remote_rows + np.array(
            [dst.ctypes.data, 0, 0, 0, 0], dtype=np.uint64), mem_type="DRAM")
        src[:] = 0
        telemetry = a.get_xfer_telemetry(read(a, local, local_ids, remote, remote_ids))
        assert telemetry.descCount == want, name
        for li, ri in zip(local_ids, remote_ids):
            ls, rs = _start(local_rows, li), _start(remote_rows, ri)
            length = int(_run(local_rows, li)[0][1])
            assert (src[ls:ls + length] == dst[rs:rs + length]).all(), name


@pytest.mark.skipif(not NATIVE, reason="needs NIXL with its Mooncake backend")
def test_the_hook_predicts_what_nixl_measures(agents):
    """The hook on a real NIXL agent: MLA pages read at unequal block ids
    (merged a layer) and CP1's head slices (never merged) log [slot,
    selected, predicted], the prediction equal to NIXL's measured count."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
        NixlKVConnectorStats,
    )

    a, src, dst = agents
    worker = SimpleNamespace(nixl_wrapper=a, xfer_stats=NixlKVConnectorStats(),
                             dst_xfer_side_handles={})
    NixlReadPeers(worker)
    measured = []
    for rows, local_blocks, remote_blocks in ((MLA, [0, 1, 2, 3], [4, 5, 6, 7]),
                                              (SLICED, [0, 1, 2, 3], [0, 1, 2, 3])):
        local = a.prep_xfer_dlist("NIXL_INIT_AGENT", rows + np.array(
            [src.ctypes.data, 0, 0, 0, 0], dtype=np.uint64), mem_type="DRAM")
        remote = a.prep_xfer_dlist("B", rows + np.array(
            [dst.ctypes.data, 0, 0, 0, 0], dtype=np.uint64), mem_type="DRAM")
        worker.dst_xfer_side_handles = {"prefill": {0: remote}}
        handle = read(a, local, ids(LAYERS, BLOCKS, local_blocks), remote,
                      ids(LAYERS, BLOCKS, remote_blocks))
        telemetry = a.get_xfer_telemetry(handle)
        measured.append(telemetry.descCount)
        worker.xfer_stats.record_transfer(telemetry)
    assert measured == [LAYERS, 4 * LAYERS]
    plans = worker.xfer_stats.data["plan"]
    assert [plan[:3] for plan in plans] == [[0, 4 * LAYERS, LAYERS], [0, 4 * LAYERS, 4 * LAYERS]]
    assert all(plan[3] == sum(n * c for n, c in plan[4].items()) and sum(plan[4].values()) == 4 * LAYERS
               for plan in plans)


def _run(rows, index):
    start = 0
    for row in rows:
        if index < start + int(row[4]):
            return row, index - start
        start += int(row[4])
    raise IndexError(index)


def _start(rows, index):
    row, k = _run(rows, index)
    return int(row[0]) + k * int(row[3])
