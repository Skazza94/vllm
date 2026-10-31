"""Prefill/decode runs: the checks alone, and a whole run on docker with
E2E_VLLM=1 (several minutes, most of the machine's cores)."""

from __future__ import annotations

import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from harness import pd, spec, topology


E2E = Path(__file__).resolve().parent.parent
RUN = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", "data0"),
             decode=pd.Side("d", "data0"), client="c", control_network="control",
             workload=pd.Workload(num_prompts=2))


def line(**record) -> str:
    return json.dumps(record) + "\n"


ONE_RANK = line(kv_layout={"tp_rank": 0, "tp_size": 1, "block_bytes": 4, "bytes": 1600})


def allocated(*requests: str) -> str:
    """Prefill's allocations of requests, under its own request ids."""
    return line(kv_events={"allocated": {f"cmpl-{r}-0-p": [1] for r in requests}})


def checked(request: str, ok: bool = True, **more) -> dict:
    """Decode's byte check of a request, under its own request id: 5 blocks
    of 50 layers, 4 bytes a block and layer."""
    return {"req_id": f"cmpl-{request}-0-d", "ok": ok, "blocks": 5, "layers": 50, "bytes": 1000,
            "mismatched_bytes": 0 if ok else 9, **more}


GOOD = {
    "bench": {"completed": 2, "errors": ["", ""], "mean_ttft_ms": 50.0},
    "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 0}}),
    "decode": ONE_RANK + line(kv_check=checked("a")) + line(kv_check=checked("b")),
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
            "byte-checked 1": {"decode": ONE_RANK + line(kv_check=checked("a"))},
            "unlike prefill": {"decode": ONE_RANK + line(kv_check=checked("a", ok=False))
                               + line(kv_check=checked("b"))},
            "failed 1 transfer": {
                "prefill": ONE_RANK + allocated("a", "b")
                + line(kv_transfers={"send": {"count": 2, "bytes": 2000, "failed": 1}})},
            "moved 1000 bytes, not the 2000 decode checked": {
                "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 1, "bytes": 1000}}),
                "routers": {"control": 50, "data0": 5000}},
            # Decode checks every block pulled: more is not the plan either.
            "moved 3000 bytes, not the 2000 decode checked": {
                "prefill": ONE_RANK + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": 3000}}),
                "routers": {"control": 50, "data0": 3100}},
            "not both produced and consumed": {"prefill": ONE_RANK + allocated("a", "b", "c")
                                               + line(kv_transfers={"send": {"count": 2, "bytes": 2000}})},
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

    def test_the_kv_cache_dtype_reaches_both_servers(self) -> None:
        def flag(run: pd.Run, role: str, name: str) -> str:
            command = pd.server_command(run, role, ["10.100.0.1"])
            return command[command.index(name) + 1]

        self.assertEqual(flag(RUN, "prefill", "--kv-cache-dtype"), "auto")  # the model's
        fp8 = pd.Run(**{**RUN.__dict__, "server": pd.Server(kv_cache_dtype="fp8")})
        for role in ("prefill", "decode"):
            self.assertEqual(flag(fp8, role, "--kv-cache-dtype"), "fp8")
            self.assertEqual(flag(fp8, role, "--dtype"), "auto")  # the weights' unchanged

    def test_a_multimodal_model_serves_its_language_model_only_when_asked(self) -> None:
        self.assertNotIn("--language-model-only", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        text = pd.Run(**{**RUN.__dict__, "server": pd.Server(language_model_only=True)})
        for role in ("prefill", "decode"):
            self.assertIn("--language-model-only", pd.server_command(text, role, ["10.100.0.1"]))

    def test_nixl_moves_the_kv_over_its_mooncake_backend_on_the_kv_address(self) -> None:
        nixl = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl")})
        # A hybrid's state as NIXL locates it; a spec's env may override.
        self.assertIn("VLLM_SSM_CONV_STATE_LAYOUT=DS", pd.server_command(nixl, "prefill", ["10.100.0.1"]))
        self.assertNotIn("VLLM_SSM_CONV_STATE_LAYOUT=DS", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        sd = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl", env={"VLLM_SSM_CONV_STATE_LAYOUT": "SD"})})
        self.assertIn("VLLM_SSM_CONV_STATE_LAYOUT=SD", pd.server_command(sd, "prefill", ["10.100.0.1"]))
        for role in ("prefill", "decode"):
            command = pd.server_command(nixl, role, ["10.100.0.1"])
            config = json.loads(command[command.index("--kv-transfer-config") + 1])
            self.assertEqual((config["kv_connector"], config["kv_connector_extra_config"]),
                             ("NixlConnector", {"backends": ["Mooncake"]}))
            env = dict(item.split("=", 1) for item in command[1:command.index(pd.VLLM)])
            self.assertEqual((env["VLLM_NIXL_SIDE_CHANNEL_HOST"], env["NIXL_MOONCAKE_IP_ADDR"]),
                             ("10.100.0.1", "10.100.0.1"))
            self.assertEqual(json.loads(env["MC_TENT_CONF"])["rpc_server_hostname"], "10.100.0.1")
            self.assertNotIn("VLLM_MOONCAKE_BOOTSTRAP_PORT", env)
        mooncake = pd.server_command(RUN, "prefill", ["10.100.0.1"])
        self.assertEqual(json.loads(mooncake[mooncake.index("--kv-transfer-config") + 1])["kv_connector"],
                         "MooncakeConnector")

    def test_nixl_ranks_take_their_own_addresses(self) -> None:
        nixl = pd.Run(**{**TP2.__dict__, "server": pd.Server(connector="nixl", tensor_parallel_size=2)})
        command = pd.server_command(nixl, "decode", ["10.100.0.1", "10.101.0.1"])
        config = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(config["kv_connector_extra_config"],
                         {"backends": ["Mooncake"], "rank_addresses": ["10.100.0.1", "10.101.0.1"]})
        env = dict(item.split("=", 1) for item in command[1:command.index(pd.VLLM)])
        # Rank 0's address until each rank binds its own; one side channel.
        self.assertEqual((env["VLLM_NIXL_SIDE_CHANNEL_HOST"], env["NIXL_MOONCAKE_IP_ADDR"]),
                         ("10.100.0.1", "10.100.0.1"))

    def test_the_connector_is_mooncake_or_nixl(self) -> None:
        base = {"name": "t", "topology": "t", "model": ".", "client": "c", "control_network": "control",
                "prefill": {"node": "p", "kv_network": "data0"}, "decode": {"node": "d", "kv_network": "data0"}}

        def load(server: dict, **sides) -> pd.Run:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run.yaml"
                path.write_text(json.dumps({**base, **sides, "server": {"timing": "zero", **server}}))
                return pd.load(path)

        self.assertEqual((load({}).server.connector, load({}).proxy), ("mooncake", pd.DEFAULT_PROXY))
        self.assertEqual(load({"connector": "nixl"}).proxy, pd.NIXL_PROXY)
        self.assertEqual(load({"connector": "nixl", "tensor_parallel_size": 2}).server.tensor_parallel_size, 2)
        with self.assertRaisesRegex(ValueError, "server.connector is one of mooncake, nixl"):
            load({"connector": "ucx"})
        dp = load({"connector": "nixl"}, decode={"node": "d", "kv_network": "data0", "data_parallel_size": 2})
        self.assertEqual(dp.decode.data_parallel_size, 2)

    def test_the_platform_may_choose_the_block_size(self) -> None:
        default = pd.server_command(RUN, "prefill", ["10.100.0.1"])
        self.assertEqual(default[default.index("--block-size") + 1], "16")
        unset = pd.Run(**{**RUN.__dict__, "server": pd.Server(block_size=None)})
        self.assertNotIn("--block-size", pd.server_command(unset, "prefill", ["10.100.0.1"]))

    def test_prefix_caching_is_off_for_byte_checks_and_settable(self) -> None:
        def flags(kv_check: bool, enabled: bool | None) -> list[str]:
            run = pd.Run(**{**RUN.__dict__, "kv_check": kv_check,
                            "server": pd.Server(enable_prefix_caching=enabled)})
            return [flag for flag in pd.server_command(run, "prefill", ["10.100.0.1"]) if "prefix-caching" in flag]

        self.assertEqual(flags(True, None), ["--no-enable-prefix-caching"])
        self.assertEqual(flags(False, None), [])  # vLLM's default
        # A hybrid model's run without byte checks keeps it off.
        self.assertEqual(flags(False, False), ["--no-enable-prefix-caching"])
        self.assertEqual(flags(False, True), ["--enable-prefix-caching"])
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            spec.write_text((E2E / "runs" / "pd-smoke.yaml").read_text().replace(
                "server:  # both servers", "server:  # both servers\n  enable_prefix_caching: true"))
            with self.assertRaises(ValueError) as raised:
                pd.load(spec)
            self.assertIn("kv_check needs prefix caching off", str(raised.exception))

    def test_a_new_run_names_its_timing_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            spec.write_text((E2E / "runs" / "pd-smoke.yaml").read_text().replace("  timing: linear\n", ""))
            with self.assertRaises(ValueError) as raised:
                pd.load(spec)
            self.assertIn("needs server.timing", str(raised.exception))

    def test_the_example_run_loads(self) -> None:
        run = pd.load(E2E / "runs" / "pd-smoke.yaml")
        self.assertEqual((run.prefill.node, run.decode.kv_network), ("prefill0", "data0"))
        self.assertTrue((run.model / "config.json").is_file(), run.model)
        self.assertTrue(run.proxy.is_file(), run.proxy)


# gpt-oss-like: a 32-token window group and a full one, 2 layers each, 4
# bytes a block and layer; 1024-token prompts (64 blocks).
GROUPS_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 4, "bytes": 1600,
    "groups": [{"kind": "SlidingWindowSpec", "sliding_window": 32, "block_size": 16, "layers": 2, "block_bytes": 4},
               {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}]})


def group_check(group: int, first: int, blocks: int, **more) -> dict:
    return {"group": group, "first_logical_block": first, "blocks": blocks, "layers": 2, "block_bytes": 8,
            "bytes": blocks * 8, "mismatched_bytes": 0, **more}


def grouped(request: str, window=(61, 3), full=(0, 64)) -> dict:
    """Decode's check of a request: the window's last blocks, every full one."""
    groups = [group_check(0, *window), group_check(1, *full)]
    return checked(request, blocks=window[1] + full[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def pulled(request: str, window=(61, 3), full=(0, 64), offset: int = 100) -> str:
    """Decode's pull of a request as posted: its blocks by group."""
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": window[0], "blocks": list(range(offset, offset + window[1]))},
        {"group": 1, "first": full[0], "blocks": list(range(offset + 10, offset + 10 + full[1]))}]}})


PULLS = pulled("a") + pulled("b", offset=300)
# Decode's first steps of the two 1024-token requests.
FIRST_STEPS = "".join(
    line(step={"requests": [{"req_id": f"cmpl-{r}-0-d", "num_computed_tokens": 1023,
                             "num_scheduled_tokens": 1, "num_prompt_tokens": 1024}]})
    for r in ("a", "b"))


def allocated_groups(request: str, *entries: tuple[int, int, list[int]], side: str = "p") -> str:
    return line(kv_events={"allocated_groups": {f"cmpl-{request}-0-{side}": [
        {"group": group, "first": first, "blocks": blocks} for group, first, blocks in entries]}})


class GroupCheckTests(unittest.TestCase):
    MOVED = 2 * (3 + 64) * 8

    def verdict(self, checks=None, moved=MOVED, prefill_events="", decode_events="", pulls=PULLS) -> dict:
        checks = checks or [grouped("a"), grouped("b")]
        prefill = (GROUPS_LAYOUT + allocated("a", "b") + prefill_events
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = GROUPS_LAYOUT + pulls + FIRST_STEPS + decode_events + "".join(line(kv_check=c) for c in checks)
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                        {"control": 50, "data0": moved + 100}, RUN)

    def test_groups_are_checked_and_their_evidence_kept(self) -> None:
        # Prefill gave the window block 1 again at its fourth place; decode's
        # table pads the window's head with the null block.
        verdict = self.verdict(
            prefill_events=allocated_groups("a", (0, 0, [1, 2, 3]), (1, 0, [4, 5, 6]))
            + allocated_groups("a", (0, 3, [1])),
            decode_events=allocated_groups("a", (0, 61, [7, 8, 9]), side="d")
            + allocated_groups("a", (0, 0, [0] * 61 + [7, 8, 9]), side="d"))
        self.assertEqual(verdict["problems"], [])
        window, full = verdict["ranks"][0]["groups"]["decode"][0]
        self.assertEqual((window["kind"], window["checked_blocks"], window["clipped_pulls"],
                          window["first_logical_block_max"], window["null_places"]),
                         ("SlidingWindowSpec", 6, 2, 61, 61))
        self.assertEqual((full["checked_blocks"], full["checked_bytes"], full["clipped_pulls"]), (128, 1024, 0))
        window = verdict["ranks"][0]["groups"]["prefill"][0][0]
        self.assertEqual((window["reused_blocks"], window["reallocated_blocks"]), (1, 1))
        # Listed twice at one place (a pull, then the scheduled table) is not
        # a reallocation; a block given to another request is.
        verdict = self.verdict(
            prefill_events=allocated_groups("a", (0, 0, [1, 2])) + allocated_groups("b", (0, 0, [2])),
            decode_events=allocated_groups("a", (0, 61, [7, 8, 9]), side="d")
            + allocated_groups("a", (0, 0, [0] * 61 + [7, 8, 9]), side="d"))
        decode_window = verdict["ranks"][0]["groups"]["decode"][0][0]
        self.assertEqual((decode_window["reused_blocks"], decode_window["reallocated_blocks"]), (0, 0))
        prefill_window = verdict["ranks"][0]["groups"]["prefill"][0][0]
        self.assertEqual((prefill_window["reused_blocks"], prefill_window["reallocated_blocks"]), (0, 1))

    def test_group_failures_are_reported(self) -> None:
        wrong_bytes = grouped("a")
        wrong_bytes["groups"][1] = group_check(1, 0, 64, bytes=1000)
        wrong_bytes["bytes"] = 24 + 1000
        unsummed = grouped("a")
        unsummed["bytes"] += 8
        omitted = grouped("a")
        omitted["groups"] = omitted["groups"][:1]
        omitted["bytes"] = 24
        duplicated = grouped("a")
        duplicated["groups"] = [duplicated["groups"][0], *duplicated["groups"]]
        duplicated["bytes"] += 24
        ungrouped = grouped("a")
        del ungrouped["groups"]
        cases = {
            "covers groups [0], not each of the layout's 2 once": [omitted, grouped("b")],
            "covers groups [0, 0, 1], not each of the layout's 2 once": [duplicated, grouped("b")],
            "cmpl-a-0-d's byte check lacks groups": [ungrouped, grouped("b")],
            # Within every bound, but not what was pulled.
            "group 1 checked 32 block(s) from 0, not the pull's 64 from 0": [grouped("a", full=(0, 32)),
                                                                             grouped("b")],
            "group 0 checked 3 block(s) from 1, not the pull's 3 from 61": [grouped("a", window=(1, 3)),
                                                                            grouped("b")],
            "group 1 check disagrees with the layout": [wrong_bytes, grouped("b")],
            "is not its groups' sum": [unsummed, grouped("b")],
            "pulled 4 blocks, more than its window's 3": [grouped("a", window=(60, 4)), grouped("b")],
            "group 1 (FullAttentionSpec) was pulled from block 2": [grouped("a", full=(2, 62)), grouped("b")],
            "no pull of window group 0 was clipped": [grouped("a", window=(0, 3)), grouped("b", window=(0, 3))],
        }
        for message, checks in cases.items():
            with self.subTest(message=message):
                moved = sum(c["bytes"] for c in checks)
                problems = self.verdict(checks, moved=moved)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        problems = self.verdict(pulls=pulled("a"))["problems"]
        self.assertTrue(any("cmpl-b-0-d's pull plan was not logged" in p for p in problems), problems)

    def test_checks_missing_fields_are_not_skipped(self) -> None:
        # A truncated full group whose aggregate blocks were dropped, or
        # both aggregates and groups: validated, not let through.
        truncated = grouped("a", full=(0, 32))
        del truncated["blocks"]
        bare = grouped("a")
        del bare["blocks"], bare["groups"]
        for message, check in (("lacks blocks", truncated), ("lacks blocks, groups", bare)):
            with self.subTest(message=message):
                problems = self.verdict([check, grouped("b")], moved=check["bytes"] + grouped("b")["bytes"])[
                    "problems"]
                self.assertTrue(any(f"cmpl-a-0-d's byte check {message}" in p for p in problems), problems)

    def test_every_group_of_a_pull_holds_blocks(self) -> None:
        # Full attention left out of the posted pull and checked empty.
        empty = grouped("a", full=(0, 0))
        pulls = line(kv_events={"reads_posted": ["cmpl-a-0-d"], "allocated_groups": {"cmpl-a-0-d": [
            {"group": 0, "first": 61, "blocks": [100, 101, 102]}]}}) + pulled("b", offset=300)
        problems = self.verdict([empty, grouped("b")], moved=empty["bytes"] + grouped("b")["bytes"],
                                pulls=pulls)["problems"]
        self.assertTrue(any("cmpl-a-0-d's group 1 checked no blocks" in p for p in problems), problems)
        self.assertTrue(any("not the pull's 0 from None" in p for p in problems), problems)

    def test_a_one_token_window_pulls_none_of_a_block_aligned_prompt(self) -> None:
        # The next token's window skips the 1024-token prompt whole: the pull
        # posts no window entry and checks none of its blocks.
        def verdict(window_blocks: int) -> dict:
            layout = GROUPS_LAYOUT.replace('"sliding_window": 32', '"sliding_window": 1')
            checks = [grouped(r, window=(64, window_blocks)) for r in ("a", "b")]
            entries = [{"group": 1, "first": 0, "blocks": list(range(10, 74))}]
            if window_blocks:
                entries.insert(0, {"group": 0, "first": 64, "blocks": list(range(window_blocks))})
            pulls = "".join(line(kv_events={"reads_posted": [f"cmpl-{r}-0-d"],
                                            "allocated_groups": {f"cmpl-{r}-0-d": entries}}) for r in ("a", "b"))
            moved = sum(c["bytes"] for c in checks)
            prefill = (layout + allocated("a", "b")
                       + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
            decode = layout + pulls + FIRST_STEPS + "".join(line(kv_check=c) for c in checks)
            return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                            {"control": 50, "data0": moved + 100}, RUN)

        self.assertEqual(verdict(0)["problems"], [])
        problems = verdict(1)["problems"]
        self.assertTrue(any("1-token window group 0 checked 1 blocks from 64, not 0 from 64" in p
                            for p in problems), problems)

    def test_a_skipped_window_starts_at_the_prompts_end(self) -> None:
        # The empty group's first block is the prompt's end, cdiv'd: tokens
        # over block size, N - 1 tokens for a hybrid; its pull posts no
        # entry or an empty one.
        window = {"kind": "SlidingWindowSpec", "sliding_window": 1, "block_size": 16, "layers": 2,
                  "block_bytes": 4}
        full = {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}
        state = {"kind": "MambaSpec", "sliding_window": None, "block_size": 4096, "layers": 2, "block_bytes": 4}

        def problems(groups: list[dict], prompt: int, first: int, posted: dict) -> list[str]:
            nested = [group_check(0, first, 0)] + [
                group_check(1, 0, 1 if groups[1] is state else -(-prompt // 16))]
            check = checked("a", blocks=sum(g["blocks"] for g in nested), layers=4,
                            bytes=sum(g["bytes"] for g in nested), groups=nested)
            plan = {"cmpl-a-0-d": {1: (0, nested[1]["blocks"]), **posted}}
            return pd._byte_check_problems({"groups": groups}, [check], prompt, plan, {"cmpl-a-0-d": prompt})

        self.assertEqual(problems([window, full], 1024, 64, {}), [])
        self.assertEqual(problems([window, full], 1024, 64, {0: (64, 0)}), [])
        bogus = problems([window, full], 1024, 999, {})
        self.assertTrue(any("checked 0 blocks from 999, not 0 from 64" in p for p in bogus), bogus)
        posted = problems([window, full], 1024, 999, {0: (999, 0)})
        self.assertTrue(any("not 0 from 64" in p for p in posted), posted)
        # A hybrid's 33-token prompt hands 32 tokens over: aligned, first 2.
        self.assertEqual(problems([window, state], 33, 2, {}), [])
        hybrid = problems([window, state], 33, 3, {})
        self.assertTrue(any("not 0 from 2" in p for p in hybrid), hybrid)


# A gated delta-net hybrid's: its state group (one 8-byte padded page a
# layer, 6 real) and a full-attention group of 16-token blocks, 2 layers
# each; 33-token prompts, of which decode pulls 32 tokens' pages.
HYBRID_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 8, "bytes": 1600,
    "groups": [{"kind": "MambaSpec", "sliding_window": None, "block_size": 4096, "layers": 2, "block_bytes": 8,
                "real_page_bytes": 6, "block_stride": 8, "mamba_type": "GDN_ATTN"},
               {"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 8, "real_page_bytes": 8, "block_stride": 8}]})


def hybrid_check(request: str, state=(0, 1), attention=(0, 2)) -> dict:
    groups = [{"group": 0, "first_logical_block": state[0], "blocks": state[1], "layers": 2, "block_bytes": 16,
               "bytes": state[1] * 16, "mismatched_bytes": 0},
              {"group": 1, "first_logical_block": attention[0], "blocks": attention[1], "layers": 2,
               "block_bytes": 16, "bytes": attention[1] * 16, "mismatched_bytes": 0}]
    return checked(request, blocks=state[1] + attention[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def hybrid_pull(request: str, state=(0, 1), attention=(0, 2), offset: int = 100) -> str:
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": state[0], "blocks": list(range(offset, offset + state[1]))},
        {"group": 1, "first": attention[0], "blocks": list(range(offset + 10, offset + 10 + attention[1]))}]}})


def stepped(request: str, side: str, computed: int, scheduled: int, prompt: int) -> str:
    return line(step={"requests": [{"req_id": f"cmpl-{request}-0-{side}", "num_computed_tokens": computed,
                                    "num_scheduled_tokens": scheduled, "num_prompt_tokens": prompt}]})


class HybridCheckTests(unittest.TestCase):
    def verdict(self, checks=None, prefill_steps=None, decode_steps=None, pulls=None,
                layout=HYBRID_LAYOUT) -> dict:
        checks = checks or [hybrid_check("a"), hybrid_check("b")]
        moved = sum(c["bytes"] for c in checks)
        prefill_steps = prefill_steps if prefill_steps is not None else "".join(
            stepped(r, "p", 0, 32, 32) for r in ("a", "b"))
        decode_steps = decode_steps if decode_steps is not None else "".join(
            stepped(r, "d", 32, 1, 33) for r in ("a", "b"))
        pulls = pulls if pulls is not None else hybrid_pull("a") + hybrid_pull("b", offset=300)
        prefill = (layout + allocated("a", "b") + prefill_steps
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = layout + pulls + decode_steps + "".join(line(kv_check=c) for c in checks)
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode],
                        {"control": 1, "data0": moved + 100}, RUN)

    def test_a_hybrid_handoff_passes_with_its_evidence(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["ranks"][0]["handoff"]["requests"], 2)
        self.assertEqual(verdict["ranks"][0]["handoff"]["prompt_tokens"], [33, 33])

    def test_hybrid_failures_are_reported(self) -> None:
        cases = {
            "state group 0 checked 2 blocks, not 1": {
                "checks": [hybrid_check("a", state=(0, 2)), hybrid_check("b")],
                "pulls": hybrid_pull("a", state=(0, 2)) + hybrid_pull("b", offset=300)},
            # The whole prompt's pages: not truncated.
            "group 1 checked 3 blocks, not the cdiv(32, 16) of its prompt": {
                "checks": [hybrid_check("a", attention=(0, 3)), hybrid_check("b")],
                "pulls": hybrid_pull("a", attention=(0, 3)) + hybrid_pull("b", offset=300)},
            "resumed at 33 tokens with 1 scheduled, not at 32 with 1": {
                "decode_steps": stepped("a", "d", 33, 1, 33) + stepped("b", "d", 32, 1, 33)},
            "prefill took a 33-token prompt to 33 tokens, not its 33 tokens but the last": {
                "prefill_steps": stepped("a", "p", 0, 33, 33) + stepped("b", "p", 0, 32, 32)},
            "cmpl-b-0-d has no prefill steps": {"prefill_steps": stepped("a", "p", 0, 32, 32)},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**changes)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)


    def test_every_consumed_request_needs_its_decode_steps(self) -> None:
        only_b = stepped("b", "d", 32, 1, 33)
        cases = {
            "complete absence": ({"decode_steps": ""}, ["cmpl-a-0-d has no decode steps",
                                                        "cmpl-b-0-d has no decode steps"]),
            "partial absence": ({"decode_steps": only_b}, ["cmpl-a-0-d has no decode steps"]),
            # One page where two are needed, behind the missing steps.
            "wrong pages": ({"decode_steps": only_b,
                             "checks": [hybrid_check("a", attention=(0, 1)), hybrid_check("b")],
                             "pulls": hybrid_pull("a", attention=(0, 1)) + hybrid_pull("b", offset=300)},
                            ["cmpl-a-0-d has no decode steps",
                             "cmpl-a-0-d's group 1 pages cannot be checked"]),
        }
        for case, (changes, messages) in cases.items():
            with self.subTest(case=case):
                verdict = self.verdict(**changes)
                for message in messages:
                    self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])
                self.assertLess(verdict["ranks"][0]["handoff"]["requests"], 2)

    def test_over_nixl_the_checks_follow_its_native_selection(self) -> None:
        """NIXL moves a state as its real spans in every state region, one
        holding none of the group's layers included, not the padding: the
        checks are compared with the layout's selection, not the pages."""
        raw = json.loads(HYBRID_LAYOUT)
        state, attention = raw["kv_layout"]["groups"]
        # Two state regions of 6 real bytes (one transfer-only), 2 padding
        # bytes each; the attention group's 8 bytes in each region.
        state["transfer"] = {"bytes_per_block": 12, "transfer_only_bytes_per_block": 6,
                             "padding_bytes_per_block": 4, "descriptors_per_block": 8}
        attention["transfer"] = {"bytes_per_block": 16, "transfer_only_bytes_per_block": 0,
                                 "padding_bytes_per_block": 0, "descriptors_per_block": 2}
        layout = line(**raw)

        def check(request: str, transfer_only: int = 6) -> dict:
            nixl = hybrid_check(request)
            state_check, attention_check = nixl["groups"]
            state_check.update(bytes=12, transfer_only_bytes=transfer_only, padding_bytes=4, descriptors=8)
            attention_check.update(bytes=32, transfer_only_bytes=0, padding_bytes=0, descriptors=4)
            nixl["bytes"] = 44
            return nixl

        right = self.verdict(checks=[check("a"), check("b")], layout=layout)
        self.assertEqual(right["problems"], [])
        self.assertEqual(right["kv_bytes_transfer_only"], 12)
        wrong = self.verdict(checks=[check("a", transfer_only=0), check("b")], layout=layout)["problems"]
        self.assertTrue(any("cmpl-a-0-d's group 0 check disagrees with NIXL's selection" in p
                            for p in wrong), wrong)

    def test_a_one_token_prompt_pulls_its_token(self) -> None:
        # Neither side truncates it: its one token's page is pulled, the token
        # computed again.
        checks = [hybrid_check(r, attention=(0, 1)) for r in ("a", "b")]
        verdict = self.verdict(
            checks=checks,
            prefill_steps="".join(stepped(r, "p", 0, 1, 1) for r in ("a", "b")),
            decode_steps="".join(stepped(r, "d", 0, 1, 1) for r in ("a", "b")),
            pulls=hybrid_pull("a", attention=(0, 1)) + hybrid_pull("b", attention=(0, 1), offset=300))
        self.assertEqual(verdict["problems"], [])

    def test_dp_ranks_take_prompts_from_the_joint_steps(self) -> None:
        """Decode DP=2: DP rank 0 logs both requests' joint steps, DP rank 1
        only its own KV (request a); a's pages are still checked."""

        def layout(dp: int) -> str:
            raw = json.loads(HYBRID_LAYOUT)
            raw["kv_layout"]["dp_rank"] = dp
            return line(**raw)

        def verdict(a_pages: int) -> dict:
            a = hybrid_check("a", attention=(0, a_pages))
            moved = a["bytes"] + hybrid_check("b")["bytes"]
            prefill = (layout(0) + allocated("a", "b") + stepped("a", "p", 0, 32, 32)
                       + stepped("b", "p", 0, 32, 32) + sent(moved))
            decode = [layout(0) + stepped("a", "d", 32, 1, 33) + stepped("b", "d", 32, 1, 33)
                      + hybrid_pull("b", offset=300) + line(kv_check=hybrid_check("b")),
                      layout(1) + hybrid_pull("a", attention=(0, a_pages)) + line(kv_check=a)]
            return pd.check({"completed": 2, "errors": []}, [prefill], decode,
                            {"control": 1, "data0": moved + 100}, dp_run(1, 2))

        right = verdict(2)
        self.assertEqual(right["problems"], [])
        self.assertEqual(right["ranks"][0]["handoff"]["requests"], 2)
        problems = verdict(3)["problems"]
        self.assertTrue(any("DP rank 1: cmpl-a-0-d's group 1 checked 3 blocks, not the cdiv(32, 16)" in p
                            for p in problems), problems)


    def test_without_byte_checks_decodes_allocations_require_their_steps(self) -> None:
        """Byte checks off (prefix caching off): the requests decode
        allocated, not those its steps name, need their handoff steps; on
        one DP rank or across two, the joint steps in DP rank 0's log."""

        def decode_allocated(*requests: str) -> str:
            return line(kv_events={"allocated": {f"cmpl-{r}-0-d": [1] for r in requests}})

        def layout(dp: int) -> str:
            raw = json.loads(HYBRID_LAYOUT)
            raw["kv_layout"]["dp_rank"] = dp
            return line(**raw)

        prefill = (HYBRID_LAYOUT + allocated("a", "b") + stepped("a", "p", 0, 32, 32)
                   + stepped("b", "p", 0, 32, 32) + sent(2000))
        steps = {r: stepped(r, "d", 32, 1, 33) for r in ("a", "b")}
        one = pd.Run(**{**RUN.__dict__, "kv_check": False})
        two = pd.Run(**{**dp_run(1, 2).__dict__, "kv_check": False})
        cases = {
            # (decode logs, run): the problems, the requests the handoff checked.
            "complete": ([HYBRID_LAYOUT + decode_allocated("a", "b") + steps["a"] + steps["b"]], one, [], 2),
            "partial": ([HYBRID_LAYOUT + decode_allocated("a", "b") + steps["a"]], one,
                        ["cmpl-b-0-d has no decode steps"], 1),
            "absent": ([HYBRID_LAYOUT + decode_allocated("a", "b")], one,
                       ["cmpl-a-0-d has no decode steps", "cmpl-b-0-d has no decode steps"], 0),
            "DP complete": ([layout(0) + decode_allocated("b") + steps["a"] + steps["b"],
                             layout(1) + decode_allocated("a")], two, [], 2),
            "DP partial": ([layout(0) + decode_allocated("b") + steps["b"],
                            layout(1) + decode_allocated("a")], two, ["cmpl-a-0-d has no decode steps"], 1),
            "DP absent": ([layout(0) + decode_allocated("b"), layout(1) + decode_allocated("a")], two,
                          ["cmpl-a-0-d has no decode steps", "cmpl-b-0-d has no decode steps"], 0),
        }
        for case, (decode, run, messages, checked) in cases.items():
            with self.subTest(case=case):
                prefills = [prefill if not run.decode.data_parallel_size > 1 else
                            prefill.replace(HYBRID_LAYOUT, layout(0))]
                verdict = pd.check({"completed": 2, "errors": []}, prefills, decode,
                                   {"control": 1, "data0": 2100}, run)
                self.assertEqual(verdict["kv_checks"], 0)
                if not messages:
                    self.assertEqual(verdict["problems"], [])
                for message in messages:
                    self.assertTrue(any(message in p for p in verdict["problems"]), verdict["problems"])
                self.assertEqual(verdict["ranks"][0]["handoff"]["requests"], checked)


class RecheckTests(unittest.TestCase):
    """`python -m harness recheck` on a saved results directory."""

    def save(self, directory: Path, checks: list[dict], timing: bool = True) -> None:
        moved = sum(c["bytes"] for c in checks)
        saved = {**asdict(RUN), "model": str(RUN.model), "proxy": str(RUN.proxy), "addresses": {}}
        if not timing:  # saved before server.timing existed (the roofline's runs)
            saved["server"] = {k: v for k, v in saved["server"].items() if k not in ("timing", "timing_args")}
        (directory / "run.json").write_text(json.dumps(saved, default=str))
        (directory / "bench.json").write_text(json.dumps({"completed": 2, "errors": []}))
        (directory / "routers.json").write_text(json.dumps({"control": 50, "data0": moved + 100}))
        (directory / "tent.json").write_text("{}")
        (directory / "prefill-steps.jsonl").write_text(
            GROUPS_LAYOUT + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": moved}}))
        (directory / "decode-steps.jsonl").write_text(
            GROUPS_LAYOUT + PULLS + FIRST_STEPS + "".join(line(kv_check=c) for c in checks))

    def test_recheck_exits_with_the_verdict(self) -> None:
        from harness.__main__ import main

        for checks, status in (([grouped("a"), grouped("b")], 0), ([grouped("a", full=(0, 32)), grouped("b")], 1)):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                self.save(Path(directory), checks)
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    self.assertEqual(main(["recheck", directory]), status)
                verdict = json.loads((Path(directory) / "verdict-recheck.json").read_text())
                self.assertEqual(json.loads(printed.getvalue())["problems"], verdict["problems"])
                self.assertEqual(bool(verdict["problems"]), bool(status))

    def test_a_run_saved_without_a_timing_provider_still_rechecks(self) -> None:
        from harness.__main__ import main

        with tempfile.TemporaryDirectory() as directory:
            self.save(Path(directory), [grouped("a"), grouped("b")], timing=False)
            self.assertNotIn("timing", json.loads((Path(directory) / "run.json").read_text())["server"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["recheck", directory]), 0)


TP2 = pd.Run(name="t", topology="t", model=Path("/m"), prefill=pd.Side("p", ["data0", "data1"]),
             decode=pd.Side("d", ["data0", "data1"]), client="c", control_network="control",
             server=pd.Server(tensor_parallel_size=2), workload=pd.Workload(num_prompts=2))


def rank_logs(rank: int, moved: int = 2000, ok: bool = True) -> tuple[str, str]:
    """A rank's prefill and decode step logs: its layout (4 KB a block and
    layer), two byte-checked requests of 5 blocks of 50 layers."""

    layout = line(kv_layout={"tp_rank": rank, "tp_size": 2, "shape": [8, 1, 16, 128],
                             "block_bytes": 4, "bytes": 1600})
    return (layout + allocated("a", "b") + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}),
            layout + "".join(line(kv_check=checked(r, ok, blocks=5, layers=50)) for r in ("a", "b")))


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
        self.assertTrue(any("disagrees with the rank's KV layout" in p for p in problems), problems)
        # An older layout's check without its blocks is not let through.
        problems = self.verdict(logs=[rank_logs(0), (prefill, decode.replace('"blocks": 5, ', ''))])["problems"]
        self.assertTrue(any("disagrees with the rank's KV layout" in p for p in problems), problems)

    def test_rank_networks_must_pair(self) -> None:
        self.assertEqual(TP2.prefill.rank_networks(2), ["data0", "data1"])
        self.assertEqual(pd.Side("p", "data0").rank_networks(2), ["data0", "data0"])
        own = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.101.0.1"]))
        self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1"]', own)
        shared = " ".join(pd.server_command(TP2, "prefill", ["10.100.0.1", "10.100.0.1"]))
        self.assertNotIn("rank_addresses", shared)
        self.assertIn("--tensor-parallel-size 2", shared)


def dp_layout(dp: int, tp: int = 0, tp_size: int = 1) -> str:
    return line(kv_layout={"dp_rank": dp, "tp_rank": tp, "tp_size": tp_size, "block_bytes": 4, "bytes": 1600})


def sent(nbytes: int) -> str:
    return line(kv_transfers={"send": {"count": nbytes // 1000, "bytes": nbytes, "failed": 0}})


def dp_run(prefill_dp: int, decode_dp: int, tp: int = 1) -> pd.Run:
    networks = "data0" if tp == 1 else ["data0", "data1"]
    return pd.Run(name="t", topology="t", model=Path("/m"),
                  prefill=pd.Side("p", networks, data_parallel_size=prefill_dp),
                  decode=pd.Side("d", networks, data_parallel_size=decode_dp), client="c",
                  control_network="control", server=pd.Server(tensor_parallel_size=tp),
                  workload=pd.Workload(num_prompts=2))


class DataParallelCheckTests(unittest.TestCase):
    """Requests a and b; prefill DP=2, decode DP=1 unless a case says."""

    def verdict(self, prefill=None, decode=None, run=None, routers=None) -> dict:
        prefill = prefill or [dp_layout(0) + allocated("a") + sent(1000), dp_layout(1) + allocated("b") + sent(1000)]
        decode = decode or [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b"))]
        run = run or dp_run(len(prefill), len(decode))
        return pd.check({"completed": 2, "errors": []}, prefill, decode,
                        routers or {"control": 50, "data0": 2100}, run)

    def test_each_request_has_one_producer_and_one_consumer(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["ownership"], {"producers": {0: 1, 1: 1}, "consumers": {0: 2}})
        # Bytes counted once: each request's KV crossed data0 once.
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"], verdict["transfers"]["bytes"]),
                         (2, 2000, 2000))
        self.assertEqual([r["requests"] for r in verdict["dp_ranks"]["prefill"]], [1, 1])

    def test_ownership_failures_are_reported(self) -> None:
        cases = {
            # Both decode DP ranks checked a.
            "1 request(s) more than one of the decode DP ranks consumed": {"decode": [
                dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b")),
                dp_layout(1) + line(kv_check=checked("a"))]},
            "decode logged 3 byte check(s) for 2 distinct request(s)": {"decode": [
                dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("b")),
                dp_layout(1) + line(kv_check=checked("a"))]},
            "1 request(s) more than one of the prefill DP ranks produced": {"prefill": [
                dp_layout(0) + allocated("a", "b") + sent(1000), dp_layout(1) + allocated("b") + sent(1000)]},
            "1 request(s) not both produced and consumed": {"decode": [dp_layout(0) + line(kv_check=checked("a"))]},
            "DP rank 1: prefill's log is DP rank 0, not 1": {"prefill": [
                dp_layout(0) + allocated("a") + sent(1000), dp_layout(0) + allocated("b") + sent(1000)]},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_every_request_is_byte_checked_once(self) -> None:
        """Two requests completed, but the logs hold only a, checked twice:
        as many checks as requests, yet b went unchecked."""
        prefill = [dp_layout(0) + allocated("a") + sent(1000), dp_layout(1) + sent(1000)]
        decode = [dp_layout(0) + line(kv_check=checked("a")) + line(kv_check=checked("a"))]
        problems = self.verdict(prefill, decode)["problems"]
        for message in ("decode byte-checked 1 distinct request(s), not 2",
                        "decode logged 2 byte check(s) for 1 distinct request(s)"):
            self.assertTrue(any(message in p for p in problems), problems)

    def test_without_byte_checks_ownership_comes_from_allocations(self) -> None:
        """kv_check off: decode's allocations name the consumers, and no
        byte check is reported (regression: they were counted as checks)."""
        run = dp_run(2, 1)
        run.kv_check = False
        decode = [dp_layout(0) + allocated("a", "b")]
        verdict = self.verdict(decode=decode, run=run)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"]), (0, 0))
        self.assertEqual(verdict["ranks"][0]["consumed"], 2)
        self.assertEqual(verdict["ownership"], {"producers": {0: 1, 1: 1}, "consumers": {0: 2}})
        # A request decode never took is still caught.
        problems = self.verdict(decode=[dp_layout(0) + allocated("a")], run=run)["problems"]
        self.assertTrue(any("1 request(s) not both produced and consumed" in p for p in problems), problems)

    def test_every_tp_rank_of_a_request_has_the_same_owner(self) -> None:
        # TP=2: rank 1 places a and b on the other prefill DP ranks.
        prefill = [dp_layout(0, 0, 2) + allocated("a") + sent(1000), dp_layout(0, 1, 2) + allocated("b") + sent(1000),
                   dp_layout(1, 0, 2) + allocated("b") + sent(1000), dp_layout(1, 1, 2) + allocated("a") + sent(1000)]
        decode = [dp_layout(0, tp, 2) + line(kv_check=checked("a")) + line(kv_check=checked("b")) for tp in (0, 1)]
        problems = self.verdict(prefill, decode, dp_run(2, 1, tp=2),
                                {"control": 50, "data0": 2100, "data1": 2100})["problems"]
        self.assertTrue(any("rank 1: 2 request(s) with other prefill DP ranks than rank 0's" in p
                            for p in problems), problems)

    def test_ranks_follow_the_hosts_order(self) -> None:
        run = dp_run(2, 2, tp=2)
        self.assertEqual(run.prefill.ranks(2), [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertEqual(pd._step_log_name("decode", 1, 1), "decode-steps.dp1.tp1.jsonl")
        self.assertEqual(pd._step_log_name("decode", 0, 0), "decode-steps.jsonl")
        command = " ".join(pd.server_command(run, "prefill", ["10.100.0.1", "10.101.0.1"]))
        # Local DP rank by TP rank, each TP rank on its own rail.
        self.assertIn('"rank_addresses": ["10.100.0.1", "10.101.0.1", "10.100.0.1", "10.101.0.1"]', command)
        self.assertIn("--data-parallel-size 2 --enable-expert-parallel", command)
        self.assertNotIn("--data-parallel-size", " ".join(pd.server_command(TP2, "prefill", ["a", "b"])))


# DeepSeek-V4-like packed blocks: every 20-byte block is all of one group's
# 2 layers, 8-byte pages of 6 real bytes; a 16-token window and compressed
# latents (16 tokens a block), decode recomputing the prompt's last token.
def packed_group(kind: str, window: int | None, **changes) -> dict:
    return {"kind": kind, "sliding_window": window, "block_size": 16, "layers": 2, "block_bytes": 8,
            "real_page_bytes": 6, "block_stride": 20,
            "members": [{"kind": kind, "layers": 2, "offsets": [0, 8], "page_bytes": 8, "real_page_bytes": 6,
                         "dtype": "uint8", "tokens_per_state": 1.0}],
            "checked_bytes_per_block": 12, "transfer_bytes_per_block": 20, **changes}


def packed_layout(window=None, full=None, **changes) -> str:
    return line(kv_layout={"tp_rank": 0, "tp_size": 1, "block_bytes": 20, "bytes": 2000, "prefill_backoff": 1,
                           "groups": [window or packed_group("SlidingWindowMLASpec", 16),
                                      full or packed_group("MLAAttentionSpec", None)], **changes})


def packed_check(request: str, window=(1, 1), full=(0, 2)) -> dict:
    groups = [{"group": 0, "first_logical_block": window[0], "blocks": window[1], "layers": 2, "block_bytes": 12,
               "bytes": window[1] * 12, "mismatched_bytes": 0},
              {"group": 1, "first_logical_block": full[0], "blocks": full[1], "layers": 2, "block_bytes": 12,
               "bytes": full[1] * 12, "mismatched_bytes": 0}]
    return checked(request, blocks=window[1] + full[1], layers=4, bytes=sum(g["bytes"] for g in groups),
                   groups=groups)


def packed_pull(request: str, window=(1, 1), full=(0, 2), offset: int = 100) -> str:
    return line(kv_events={"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {f"cmpl-{request}-0-d": [
        {"group": 0, "first": window[0], "blocks": list(range(offset, offset + window[1]))},
        {"group": 1, "first": full[0], "blocks": list(range(offset + 10, offset + 10 + full[1]))}]}})


class PackedCheckTests(unittest.TestCase):
    """33-token prompts: decode loads 32 tokens (2 blocks of the latents,
    the window's last block from place 1) and computes the last."""

    def verdict(self, layout=None, checks=None, loaded=None, prompt=33, pulls=None) -> dict:
        layout = layout or packed_layout()
        checks = checks or [packed_check("a"), packed_check("b")]
        loaded = loaded if loaded is not None else 2 * 3 * 20
        decode = (layout + (pulls or packed_pull("a") + packed_pull("b", offset=300))
                  + "".join(stepped(r, "d", prompt - 1, 1, prompt) for r in ("a", "b"))
                  + line(kv_transfers={"load": {"count": 2, "bytes": loaded, "failed": 0}})
                  + "".join(line(kv_check=c) for c in checks))
        prefill = layout + allocated("a", "b") + "".join(stepped(r, "p", 0, prompt - 1, prompt - 1) for r in ("a", "b"))
        return pd.check({"completed": 2, "errors": []}, [prefill], [decode], {"control": 1, "data0": loaded + 100},
                        NIXL)

    def test_checked_pages_and_moved_blocks_reconcile(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_bytes_checked"], 2 * 3 * 12)  # real pages only
        self.assertEqual(verdict["ranks"][0]["kv_bytes_spanned"], 2 * 3 * 20)  # whole blocks
        self.assertIsNotNone(verdict["ranks"][0]["handoff"])  # N - 1 handed over

    def test_each_packed_failure_is_reported(self) -> None:
        malformed = packed_group("MLAAttentionSpec", None, transfer_bytes_per_block=None)
        cases = {
            # Moving checked bytes only is not the connector's whole blocks.
            "moved 72 bytes, not the 120 decode's checked blocks span (72 checked)": {"loaded": 72},
            "check disagrees with the layout (2 layers, 12 bytes checked a block)": {
                "checks": [packed_check("a"), {**packed_check("b"), "bytes": 30, "groups": [
                    {**packed_check("b")["groups"][0], "bytes": 6}, packed_check("b")["groups"][1]]}]},
            "group 1 has a malformed packed layout": {"layout": packed_layout(full=malformed)},
            "group 0 checks 10 bytes a block, not its members' 12 real bytes": {
                "layout": packed_layout(window=packed_group("SlidingWindowMLASpec", 16, checked_bytes_per_block=10))},
            "group 1 packs 1 layers, not its 2": {"layout": packed_layout(full=packed_group(
                "MLAAttentionSpec", None,
                members=[{**packed_group("x", None)["members"][0], "layers": 1, "offsets": [0]}]))},
            "prefill_backoff is 2, not 1": {"layout": packed_layout(prefill_backoff=2)},
            # The latents hold the 32 tokens handed over, not all 33.
            "checked 3 blocks, not the cdiv(32, 16)": {
                "checks": [packed_check("a", full=(0, 3)), packed_check("b")],
                "pulls": packed_pull("a", full=(0, 3)) + packed_pull("b", offset=300), "loaded": 140},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_malformed_member_placements_are_reported(self) -> None:
        member = packed_group("x", None)["members"][0]

        def with_members(*members, **changes) -> str:
            return packed_layout(full=packed_group("MLAAttentionSpec", None, members=list(members), **changes))

        stripped = {key: value for key, value in member.items()
                    if key not in ("offsets", "page_bytes", "kind", "dtype", "tokens_per_state")}
        cases = {
            "group 1 has a malformed packed member 0 (needs": with_members(stripped),
            "group 1 has a malformed packed member 0 (": with_members({**member, "offsets": [0]}),
            "member 0's page at 0 overlaps member 0's": with_members({**member, "offsets": [0, 0]}),
            "member 0's page at 1000 ends beyond the 20-byte block": with_members({**member, "offsets": [0, 1000]}),
            "member 0 has 9 real bytes in a 8-byte page": with_members(
                {**member, "real_page_bytes": 9}, checked_bytes_per_block=18),
            "group 1 has a malformed packed layout": with_members(member, checked_bytes_per_block=None),
            "moves 16 bytes a block, less than its 20-byte block": with_members(
                member, transfer_bytes_per_block=16),
        }
        for message, layout in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(layout=layout)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_a_single_token_prompt_is_handed_over_whole(self) -> None:
        # One token: nothing is backed off; its latents' block is pulled.
        checks = [packed_check(r, window=(0, 1), full=(0, 1)) for r in ("a", "b")]
        pulls = packed_pull("a", (0, 1), (0, 1)) + packed_pull("b", (0, 1), (0, 1), offset=300)
        decode = (packed_layout() + pulls + "".join(stepped(r, "d", 0, 1, 1) for r in ("a", "b"))
                  + line(kv_transfers={"load": {"count": 2, "bytes": 80, "failed": 0}})
                  + "".join(line(kv_check=c) for c in checks))
        prefill = packed_layout() + allocated("a", "b") + "".join(stepped(r, "p", 0, 1, 1) for r in ("a", "b"))
        short = pd.Run(**{**NIXL.__dict__, "workload": pd.Workload(num_prompts=2, input_len=1)})
        verdict = pd.check({"completed": 2, "errors": []}, [prefill], [decode], {"control": 1, "data0": 180},
                           short)
        self.assertEqual(verdict["problems"], [])


NIXL = pd.Run(**{**RUN.__dict__, "server": pd.Server(connector="nixl")})


class NixlCheckTests(unittest.TestCase):
    """NIXL's decode reads the KV: its connector's loads and TENT's reads
    count the bytes, prefill moves none."""

    def verdict(self, loads=None, decode_tent=None, routers=None, run=NIXL) -> dict:
        loads = loads or {"count": 2, "bytes": 2000, "failed": 0}
        decode = ONE_RANK + line(kv_transfers={"load": loads}) + line(kv_check=checked("a")) + line(
            kv_check=checked("b"))
        tents = {"prefill": [{"tent_registered_buffer_bytes": 1600}],
                 "decode": [decode_tent if decode_tent is not None else {
                     "tent_read_bytes_total": 2000, "tent_registered_buffer_bytes": 1600}]}
        return pd.check({"completed": 2, "errors": []}, [ONE_RANK + allocated("a", "b")], [decode],
                        routers or {"control": 50, "data0": 2100}, run, tents)

    def test_decode_reads_the_bytes_it_checks(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["transfers"], {"count": 2, "bytes": 2000, "failed": 0})

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "decode's connector failed 1 transfer": {"loads": {"count": 2, "bytes": 2000, "failed": 1}},
            "moved 1000 bytes, not the 2000 decode checked": {
                "loads": {"count": 1, "bytes": 1000, "failed": 0}, "routers": {"control": 50, "data0": 5000}},
            "TENT read 1500 bytes, not the 2000": {
                "decode_tent": {"tent_read_bytes_total": 1500, "tent_registered_buffer_bytes": 1600}},
            # Unavailable telemetry is a problem, not zero.
            "decode's TENT metrics lack tent_read_bytes_total": {
                "decode_tent": {"tent_registered_buffer_bytes": 1600}},
            "fewer than the 2000 its ranks moved": {"routers": {"control": 50, "data0": 100}},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # Mooncake's decode needs no reads; its prefill needs its writes.
        problems = self.verdict(run=RUN)["problems"]
        self.assertTrue(any("prefill's TENT metrics lack tent_write_bytes_total" in p for p in problems), problems)


class SpeculativeCheckTests(unittest.TestCase):
    """Native MTP over NIXL: both instances take it, decode schedules drafts,
    every request returns exactly its output length."""

    SPEC = {"method": "mtp", "num_speculative_tokens": 3, "acceptance_rates": [1.0], "seed": 7}

    def run_with(self, **spec) -> pd.Run:
        return pd.Run(**{**NIXL.__dict__, "server": pd.Server(connector="nixl", speculative={**self.SPEC, **spec})})

    def test_both_instances_take_the_speculative_config(self) -> None:
        for role in ("prefill", "decode"):
            command = pd.server_command(self.run_with(), role, ["10.100.0.1"])
            self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]),
                             {"method": "mtp", "num_speculative_tokens": 3})
            self.assertEqual(command[command.index("--seed") + 1], "7")
            simulation = json.loads(command[command.index("--simulation-config") + 1])
            self.assertEqual(simulation["spec_acceptance_rates"], [1.0])
        plain = pd.server_command(NIXL, "decode", ["10.100.0.1"])
        self.assertNotIn("--speculative-config", plain)
        self.assertNotIn("spec_acceptance_rates", json.loads(plain[plain.index("--simulation-config") + 1]))

    def test_only_native_mtp_over_nixl_with_valid_rates(self) -> None:
        for spec, connector, message in (
            ({"method": "ngram"}, "nixl", "native MTP"),
            ({"num_speculative_tokens": 6}, "nixl", "native MTP"),
            ({"acceptance_rates": []}, "nixl", "1 to 3 probabilities"),
            ({"acceptance_rates": [1.5]}, "nixl", "1 to 3 probabilities"),
            ({"acceptance_rates": [0.5] * 4}, "nixl", "1 to 3 probabilities"),
            ({"seed": "x"}, "nixl", "seed is an integer"),
            ({"draft": "x"}, "nixl", "server.speculative takes"),
            ({}, "mooncake", "needs the nixl connector"),
        ):
            with self.subTest(spec=spec, connector=connector):
                with self.assertRaisesRegex(ValueError, message):
                    pd._check_speculative(pd.Server(connector=connector, speculative={**self.SPEC, **spec}))

    def verdict(self, steps, lengths=(16, 16), extra=(), owner=0, **spec) -> dict:
        """Decode's steps: (drafts, the step's own counts, the rank's logged
        (drafts, accepted) pairs; "same": the counts' pairs)."""
        loads = {"count": 2, "bytes": 2000, "failed": 0}
        decode = ONE_RANK + line(kv_transfers={"load": loads}) + line(kv_check=checked("a")) + line(
            kv_check=checked("b"))
        for i, (drafts, counts, logged) in enumerate(steps):
            requests = [f"cmpl-{r}-0-d" for r in "ab"[:len(drafts)]]
            decode += line(step={"step_index": i, "requests": [
                {"req_id": r, "num_computed_tokens": 100 + i, "num_scheduled_tokens": 1 + d,
                 "num_prompt_tokens": 100, "num_spec_tokens": d, "dp_rank": owner}
                for r, d in zip(requests, drafts)]},
                cost={"num_accepted": counts})
            if logged == "same":
                logged = [[d, a] for d, a in zip(drafts, counts)]
            pairs = {r: pair for r, pair, d in zip(requests, logged or [], drafts) if d}
            if pairs:
                decode += line(spec_acceptance={"step_index": i, "requests": pairs})
        decode += "".join(line(spec_acceptance=record) for record in extra)
        tents = {"prefill": [{"tent_registered_buffer_bytes": 1600}],
                 "decode": [{"tent_read_bytes_total": 2000, "tent_registered_buffer_bytes": 1600}]}
        return pd.check({"completed": 2, "errors": [], "output_lens": list(lengths)},
                        [ONE_RANK + allocated("a", "b")], [decode], {"control": 50, "data0": 2100},
                        self.run_with(**spec), tents)

    def test_drafts_are_scheduled_and_accepted_as_the_rates_say(self) -> None:
        verdict = self.verdict([([0, 0], [0, 0], None), ([3, 3], [3, 3], "same"), ([3, 1], [3, 1], "same")])
        self.assertEqual(verdict["problems"], [])
        self.assertEqual({k: verdict["speculation"][k] for k in ("steps_with_drafts", "drafted", "accepted")},
                         {"steps_with_drafts": 2, "drafted": 10, "accepted": 10})
        self.assertEqual(verdict["speculation"]["seed"], 7)
        # Rates all 0: none accepted. A joint MoE-DP step has no counts of its
        # own; its rank's acceptance records cover it.
        verdict = self.verdict([([3, 3], [0, 0], "same"), ([3], None, [[3, 0]])], acceptance_rates=[0.0])
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["speculation"]["drafts_logged"], 9)

    def test_each_failure_is_reported(self) -> None:
        cases = {
            "decode scheduled no drafts": {"steps": [([0, 0], [0, 0], None)]},
            "5 of 6 drafts accepted at acceptance rates 1": {"steps": [([3, 3], [3, 2], "same")]},
            "2 drafts accepted at acceptance rates 0": {"steps": [([3, 3], [2, 0], "same")],
                                                       "acceptance_rates": [0.0]},
            "accepted [2, 1] of drafts [1, 3]": {"steps": [([1, 3], [2, 1], [[1, 1], [3, 1]])],
                                                 "acceptance_rates": [0.5]},
            "output lengths [15, 16] of 2 request(s), not 16 each": {"steps": [([3, 3], [3, 3], "same")],
                                                                    "lengths": (16, 15)},
            # Unobserved acceptance does not pass, even at rates all 1.
            "2 drafted request step(s) lack their owner's acceptance": {"steps": [([3, 3], None, None)]},
            "1 drafted request step(s) lack their owner's acceptance": {"steps": [([3, 3], None, [[3, 3]])]},
            # Each record matches its own step: the step's drafts and counts.
            "request cmpl-a-0-d logged [3, 3], the step drafted 3 and accepted 0": {
                "steps": [([3, 3], [0, 0], [[3, 3], [3, 3]])]},
            "step 0: request cmpl-a-0-d logged [3, 3], the step drafted 1": {
                "steps": [([1, 1], None, [[3, 3], [3, 3]]), ([3, 3], None, [[1, 1], [1, 1]])]},
            "request cmpl-a-0-d logged twice": {
                "steps": [([3, 3], None, [[3, 3], [3, 3]])],
                "extra": [{"step_index": 0, "requests": {"cmpl-a-0-d": [3, 3]}}]},
            "1 acceptance record(s) for no drafted request step of their DP rank, e.g. DP rank 0 step 7": {
                "steps": [([3, 3], None, [[3, 3], [3, 3]])],
                "extra": [{"step_index": 7, "requests": {"cmpl-a-0-d": [3, 3]}}]},
            # A step's counts: one per request, nonnegative.
            "accepted [] of drafts [3, 3]": {"steps": [([3, 3], [], [[3, 1], [3, 1]])], "acceptance_rates": [0.5]},
            "accepted [-1, 1] of drafts [3, 3]": {"steps": [([3, 3], [-1, 1], [[3, 0], [3, 0]])],
                                                  "acceptance_rates": [0.0]},
            "request cmpl-a-0-d accepted [3, -1]": {"steps": [([3, 3], None, [[3, -1], [3, 1]])],
                                                    "acceptance_rates": [0.0]},
            "request cmpl-b-0-d accepted [3, 4]": {"steps": [([3, 3], None, [[3, 3], [3, 4]])]},
        }
        for message, given in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(**given)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # A record in another DP rank's log than the request's owner.
        problems = self.verdict([([3, 3], None, [[3, 3], [3, 3]])], owner=1)["problems"]
        for message in ("2 drafted request step(s) lack their owner's acceptance (spec_acceptance), e.g. DP rank 1",
                        "2 acceptance record(s) for no drafted request step of their DP rank, e.g. DP rank 0"):
            self.assertTrue(any(message in p for p in problems), problems)
        # A rates-1 run with a request's acceptance missing does not pass on
        # the drafts that were logged.
        problems = self.verdict([([3, 3], None, [[3, 3]])])["problems"]
        self.assertTrue(any("3 of 6 drafts accepted at acceptance rates 1" in p for p in problems), problems)

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
