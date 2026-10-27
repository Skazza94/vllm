"""Prefill/decode runs: the checks alone, and a whole run on docker with
E2E_VLLM=1 (several minutes, most of the machine's cores)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

from harness import pd, spec, topology


E2E = Path(__file__).resolve().parent.parent
RUN = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"),
             decode=pd.Side("d", "data0"), client="c", control_network="control",
             workload=pd.Workload(num_prompts=2))


def line(**record) -> str:
    return json.dumps(record) + "\n"


ONE_RANK = line(kv_layout={"tp_rank": 0, "tp_size": 1, "block_bytes": 4, "bytes": 1600})
GOOD = {
    "bench": {"completed": 2, "errors": ["", ""], "mean_ttft_ms": 50.0},
    "prefill": ONE_RANK + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 0}}),
    "decode": ONE_RANK + (line(kv_check={"ok": True, "bytes": 1000, "mismatched_bytes": 0})
               + line(kv_check={"ok": True, "bytes": 1000, "mismatched_bytes": 0})),
    "routers": {"control": 50, "data0": 2100},
}


class CheckTests(unittest.TestCase):
    def verdict(self, **changes) -> dict:
        given = {**GOOD, **changes}
        return pd.check(given["bench"], [given["prefill"]], [given["decode"]], given["routers"], RUN)

    def test_a_clean_run_passes(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_bytes_checked"], 2000)
        self.assertEqual(verdict["transfers"], {"count": 2, "bytes": 2000, "failed": 0})

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "completed": {"bench": {"completed": 1, "errors": ["", "HTTP 500"]}},
            "byte-checked 1": {"decode": ONE_RANK + line(kv_check={"ok": True, "bytes": 1000, "mismatched_bytes": 0})},
            "unlike prefill": {"decode": GOOD["decode"].replace('"ok": true, "bytes": 1000, "mismatched_bytes": 0}',
                                                                '"ok": false, "bytes": 1000, "mismatched_bytes": 9}', 1)},
            "failed 1 transfer": {
                "prefill": ONE_RANK + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 1}})},
            "fewer than the 2000 decode checked": {
                "prefill": ONE_RANK + line(kv_transfers={"send": {"count": 1, "bytes": 1000}}),
                "routers": {"control": 50, "data0": 5000}},
            "KV network": {"routers": {"control": 50, "data0": 100}},
            "control network carried": {"routers": {"control": 1000, "data0": 2100}},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**changes)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_tent_stays_on_the_kv_address(self) -> None:
        tent = pd.tent_config("10.100.0.1", {"max_outstanding_tasks": 8192})
        self.assertEqual(tent["rpc_server_hostname"], "10.100.0.1")
        hp_tcp = tent["transports"]["hp_tcp"]
        self.assertEqual((hp_tcp["bind_address"], hp_tcp["advertise_address"], hp_tcp["rail_addresses"]),
                         ("10.100.0.1", "10.100.0.1", ["10.100.0.1"]))
        self.assertEqual(hp_tcp["max_outstanding_tasks"], 8192)
        self.assertTrue(all(not t["enable"] for name, t in tent["transports"].items() if name != "hp_tcp"))

    def test_the_example_run_loads(self) -> None:
        run = pd.load(E2E / "runs" / "pd-smoke.yaml")
        self.assertEqual((run.prefill.node, run.decode.kv_network), ("prefill0", "data0"))
        self.assertTrue((run.model / "config.json").is_file(), run.model)
        self.assertTrue(run.proxy.is_file(), run.proxy)


TP2 = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", ["data0", "data1"]),
             decode=pd.Side("d", ["data0", "data1"]), client="c", control_network="control",
             server=pd.Server(tensor_parallel_size=2), workload=pd.Workload(num_prompts=2))


def rank_logs(rank: int, moved: int = 2000, ok: bool = True) -> tuple[str, str]:
    """A rank's prefill and decode step logs: its layout (4 KB a block and
    layer), two byte-checked requests of 5 blocks of 50 layers."""

    layout = line(kv_layout={"tp_rank": rank, "tp_size": 2, "shape": [8, 1, 16, 128],
                             "block_bytes": 4, "bytes": 1600})
    check = {"ok": ok, "bytes": 1000, "mismatched_bytes": 0 if ok else 9, "blocks": 5, "layers": 50}
    return (layout + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}),
            layout + line(kv_check=check) + line(kv_check=check))


def tent(written: float = 2000, registered: float = 1600) -> dict:
    return {"tent_write_bytes_total": written, "tent_write_requests_total": 2,
            "tent_registered_buffer_bytes": registered}


class TensorParallelCheckTests(unittest.TestCase):
    def verdict(self, logs=None, routers=None, tents=None, run=TP2) -> dict:
        logs = logs or [rank_logs(0), rank_logs(1)]
        tents = tents or {"prefill": [tent(), tent()], "decode": [tent(0), tent(0)]}
        return pd.check({"completed": 2, "errors": []}, [p for p, _ in logs], [d for _, d in logs],
                        routers or {"control": 50, "data0": 2100, "data1": 2100}, run, tents)

    def test_each_rank_is_checked_on_its_own_rail(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["requests"], 2)  # once, not per rank
        self.assertEqual(verdict["kv_bytes_checked"], 4000)
        self.assertEqual(verdict["transfers"]["bytes"], 4000)
        self.assertEqual([r["kv_layout"]["decode"]["tp_rank"] for r in verdict["ranks"]], [0, 1])

    def test_a_shared_rail_carries_every_rank(self) -> None:
        shared = pd.Run(**{**TP2.__dict__, "prefill": pd.Side("p", "data0"), "decode": pd.Side("d", "data0")})
        self.assertEqual(self.verdict(run=shared, routers={"control": 50, "data0": 4100, "data1": 0})["problems"], [])
        problems = self.verdict(run=shared, routers={"control": 50, "data0": 2100, "data1": 2000})["problems"]
        self.assertTrue(any("fewer than the 4000" in p for p in problems), problems)

    def test_each_rank_failure_is_reported(self) -> None:
        cases = {
            "rank 1: 2 request(s) received KV bytes unlike": {"logs": [rank_logs(0), rank_logs(1, ok=False)]},
            "rank 1: the connector moved 1000 bytes": {"logs": [rank_logs(0), rank_logs(1, moved=1000)],
                                                       "tents": {"prefill": [tent(), tent(1000)],
                                                                 "decode": [tent(0), tent(0)]}},
            # Beyond headers and acknowledgements (5% and 1 MiB).
            "another rank's KV": {"routers": {"control": 50, "data0": 2100 + (8 << 20), "data1": 2100}},
            "rank 0: TENT wrote 1500": {"tents": {"prefill": [tent(1500), tent()], "decode": [tent(0), tent(0)]}},
            "rank 1: decode's TENT metrics did not answer": {
                "tents": {"prefill": [tent(), tent()], "decode": [tent(0), None]}},
            "registered 100 bytes, less than its 1600": {
                "tents": {"prefill": [tent(), tent(registered=100)], "decode": [tent(0), tent(0)]}},
            # Rank 0's logs supplied for rank 1 too.
            "rank 1: prefill's log is TP rank 0 of 2, not 1 of 2": {"logs": [rank_logs(0), rank_logs(0)]},
            "rank 0: decode logged no KV layout": {
                "logs": [(rank_logs(0)[0], rank_logs(0)[1].split("\n", 1)[1]), rank_logs(1)]},
            # Counters that exist but lack what the checks need.
            "rank 1: prefill's TENT metrics lack tent_registered_buffer_bytes, tent_write_bytes_total": {
                "tents": {"prefill": [tent(), {}], "decode": [tent(0), tent(0)]}},
            "rank 0: decode's TENT metrics lack tent_registered_buffer_bytes": {
                "tents": {"prefill": [tent(), tent()], "decode": [{}, tent(0)]}},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_cleanup_finds_renamed_workers(self) -> None:
        for cmdline, comm, ours in (
            (b"/opt/venv/bin/python3\0-c\0...", b"python3", True),
            (b"VLLM::Worker_TP1\0", b"VLLM::Worker_TP", True),
            (b"VLLM::EngineCore\0", b"VLLM::EngineCor", True),
            (b"", b"VLLM::Worker_TP", True),  # an emptied command line
            (b"sleep\0infinity\0", b"sleep", False),
        ):
            with self.subTest(cmdline=cmdline):
                self.assertIs(pd.is_service(cmdline, comm), ours)
        self.assertIn("def is_service", pd.STOP)

    def test_byte_checks_must_match_the_rank_layout(self) -> None:
        prefill, decode = rank_logs(1)
        problems = self.verdict(logs=[rank_logs(0), (prefill, decode.replace('"blocks": 5', '"blocks": 4'))])[
            "problems"]
        self.assertTrue(any("disagree with the rank's KV layout" in p for p in problems), problems)

    def test_rank_networks_must_pair(self) -> None:
        self.assertEqual(TP2.prefill.rank_networks(2), ["data0", "data1"])
        self.assertEqual(pd.Side("p", "data0").rank_networks(2), ["data0", "data0"])
        own = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.101.0.1"]))
        self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1"]', own)
        shared = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.100.0.1"]))
        self.assertNotIn("rank_addresses", shared)
        self.assertIn("--tensor-parallel-size 2", shared)


@unittest.skipUnless(os.environ.get("E2E_VLLM") == "1", "set E2E_VLLM=1 to run vLLM in docker")
class PrefillDecodeRunTests(unittest.TestCase):
    def test_a_byte_checked_run_keeps_its_kv_on_the_data_rail(self) -> None:
        parsed = spec.load(E2E / "topologies" / "pd-two-rails.yaml")
        topology.down(parsed.name)
        topology.up(parsed)
        try:
            verdict = pd.execute(pd.load(E2E / "runs" / "pd-smoke.yaml"))
        finally:
            topology.down(parsed.name)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_checks"], 40)
        # The other rail stayed idle.
        self.assertLess(verdict["router_bytes"]["data1"], 1 << 20)


if __name__ == "__main__":
    unittest.main()
