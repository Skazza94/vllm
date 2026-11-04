"""Prefill/decode runs: the checks alone, and a whole run on docker with
E2E_VLLM=1 (several minutes, most of the machine's cores)."""

from __future__ import annotations

import contextlib
from dataclasses import asdict
import io
import json
import os
import re
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

    def test_a_repositorys_own_code_runs_only_when_asked(self) -> None:
        self.assertNotIn("--trust-remote-code", pd.server_command(RUN, "prefill", ["10.100.0.1"]))
        self.assertNotIn("--trust-remote-code", pd.bench_command(RUN))
        trusted = pd.Run(**{**RUN.__dict__, "server": pd.Server(trust_remote_code=True)})
        for role in ("prefill", "decode"):
            self.assertIn("--trust-remote-code", pd.server_command(trusted, role, ["10.100.0.1"]))
        self.assertIn("--trust-remote-code", pd.bench_command(trusted))
        # Only a boolean: the string "false" would turn the flags on.
        base = Path(__file__).parents[1] / "runs" / "pd-smoke.yaml"
        for value, ok in (("true", True), ("false", True), ('"false"', False), ("1", False), ("yes", True)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "x.yaml"
                text = re.sub(r"^(server:.*)$", rf"\1\n  trust_remote_code: {value}",
                              base.read_text(), count=1, flags=re.M)
                spec.write_text(text.replace("model: ", f"model: {base.parent}/", 1))
                if ok:
                    pd.load(spec)
                else:
                    with self.assertRaisesRegex(ValueError, "trust_remote_code is true or false"):
                        pd.load(spec)

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
        # With the byte checks too: a block's pattern is then its content's.
        self.assertEqual(flags(True, True), ["--enable-prefix-caching"])
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "run.yaml"
            spec.write_text((E2E / "runs" / "pd-smoke.yaml").read_text().replace(
                "server:  # both servers", "server:  # both servers\n  enable_prefix_caching: true").replace(
                "model: ", f"model: {E2E / 'runs'}/", 1))
            run = pd.load(spec)
            self.assertTrue(run.kv_check and run.server.enable_prefix_caching)

    def test_a_workload_may_share_prefixes(self) -> None:
        repetition = {"prefix_len": 768, "suffix_len": 256, "num_prefixes": 4}
        run = pd.Run(**{**RUN.__dict__, "workload": pd.Workload(input_len=1024, prefix_repetition=repetition)})
        command = pd.bench_command(run)
        self.assertEqual(command[command.index("--dataset-name") + 1], "prefix_repetition")
        self.assertEqual([command[command.index(f"--prefix-repetition-{name}") + 1]
                          for name in ("prefix-len", "suffix-len", "num-prefixes", "output-len")],
                         ["768", "256", "4", "16"])
        self.assertNotIn("--random-input-len", command)
        self.assertIn("--random-input-len", pd.bench_command(RUN))
        base = (E2E / "runs" / "pd-smoke.yaml").read_text().replace("model: ", f"model: {E2E / 'runs'}/", 1)
        for workload, extra, error in (
                ("prefix_repetition: {prefix_len: 700, suffix_len: 256, num_prefixes: 4}", "", "input_len is"),
                ("prefix_repetition: {prefix_len: 768, suffix_len: 256}", "", "takes positive"),
                ("input_len: 1024", "expect_fragmentation: 1\n", "true or false"),
                ("input_len: 1024", "expect_fragmentation: true\nkv_check: false\n", "needs kv_check")):
            with self.subTest(workload=workload, extra=extra), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "run.yaml"
                text = re.sub(r"^workload:.*$", f"workload:\n  {workload}", base, count=1, flags=re.M)
                spec.write_text(re.sub(r"^kv_check:.*\n", "", text, flags=re.M) + extra)
                with self.assertRaisesRegex(ValueError, error):
                    pd.load(spec)

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


def stepped(request: str, side: str, computed: int, scheduled: int, prompt: int, drafts: int = 0) -> str:
    return line(step={"requests": [{"req_id": f"cmpl-{request}-0-{side}", "num_computed_tokens": computed,
                                    "num_scheduled_tokens": scheduled, "num_prompt_tokens": prompt,
                                    "num_spec_tokens": drafts}]})


class HybridCheckTests(unittest.TestCase):
    def test_handoff_uses_every_native_producer_rank(self) -> None:
        for p, d in ((2, 2), (1, 2), (2, 1), (2, 4), (4, 2)):
            joint = {("prefill", r): {"cmpl-a-0-p": [(0, 32, 32, 0)]} for r in range(p)}
            joint.update({("decode", r): {"cmpl-a-0-d": [(32, 1, 33, 0)]} for r in range(d)})
            for rank in range(d):
                sources = ([rank * p // d] if p <= d else list(range(rank * p // d, (rank + 1) * p // d)))
                with self.subTest(prefill_tp=p, decode_tp=d, decode_rank=rank):
                    result = pd._handoff_from_sources(joint, rank, sources, {"cmpl-a-0-d"})
                    self.assertEqual(result["problems"], [])
                    self.assertEqual(result["evidence"]["requests"], 1)
                    self.assertEqual(result["evidence"]["prompt_tokens"], [33, 33])

    def test_handoff_rejects_missing_or_wrong_steps_on_each_gather_source(self) -> None:
        for source in (0, 1):
            for bad_steps in ({}, {"cmpl-a-0-p": [(0, 31, 31, 0)]}):
                joint = {("prefill", r): {"cmpl-a-0-p": [(0, 32, 32, 0)]} for r in (0, 1)}
                joint[("decode", 0)] = {"cmpl-a-0-d": [(32, 1, 33, 0)]}
                joint[("prefill", source)] = bad_steps
                with self.subTest(source=source, bad_steps=bad_steps):
                    result = pd._handoff_from_sources(joint, 0, [0, 1], {"cmpl-a-0-d"})
                    self.assertEqual(len(result["problems"]), 1)
                    self.assertIn(f"prefill rank {source}:", result["problems"][0])
                    self.assertIn("no prefill steps" if not bad_steps else "prefill took", result["problems"][0])

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

    def test_a_speculative_resume_verifies_its_drafts(self) -> None:
        # Native MTP at depth 2: the last prompt token and two drafts.
        steps = "".join(stepped(r, "d", 32, 3, 33, drafts=2) for r in ("a", "b"))
        self.assertEqual(self.verdict(decode_steps=steps)["problems"], [])

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
            # With speculative decoding the resume verifies its drafts too;
            # scheduled tokens beside them are not the last prompt token's.
            "resumed at 32 tokens with 3 scheduled, not at 32 with 2": {
                "decode_steps": stepped("a", "d", 32, 3, 33, drafts=1) + stepped("b", "d", 32, 1, 33)},
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



class NativeMtpStateTests(unittest.TestCase):
    """Real decode records of Qwen3.5's native MTP over NIXL (two processes,
    MTP depths 1 and 3): the clipped read's metadata gives a state's running
    slot (physically before its trailing scratch slots) index k, the depth;
    the checks agree with the layout there."""

    DATA = Path(__file__).parent / "data"

    def records(self, k: int) -> list[dict]:
        path = self.DATA / f"nixl-qwen3.5-mtp-k{k}-decode-steps.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def problems(self, records: list[dict]) -> list[str]:
        layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
        checks = [r["kv_check"] for r in records if "kv_check" in r]
        return pd._byte_check_problems(layout, checks, 200, pd._pull_plans(records),
                                       {c["req_id"]: 199 for c in checks})

    def test_states_start_at_the_mtp_depth(self) -> None:
        for k in (1, 3):
            with self.subTest(k=k):
                records = self.records(k)
                layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
                self.assertEqual([g.get("speculative_blocks") for g in layout["groups"]], [k, k, None])
                self.assertEqual(self.problems(records), [])

    def test_other_starts_are_reported(self) -> None:
        for k in (1, 3):
            records = self.records(k)
            check = next(r["kv_check"] for r in records if "kv_check" in r)
            # A state's check placed at index 0, where the clipped read's
            # metadata gives k.
            check["groups"][0]["first_logical_block"] = 0
            # Attention still starts at 0: not moved by the depth.
            check["groups"][2]["first_logical_block"] = k
            problems = self.problems(records)
            with self.subTest(k=k):
                self.assertTrue(any(f"group 0 (MambaSpec) was pulled from block 0, not {k}" in p
                                    for p in problems), problems)
                self.assertTrue(any(f"group 2 (FullAttentionSpec) was pulled from block {k}, not 0" in p
                                    for p in problems), problems)



class SparseMlaLayoutTests(unittest.TestCase):
    """A real decode record of a tiny DeepSeek-V3.2 over NIXL (two
    processes): each layer's BF16 latent and indexer keys, pages of two
    sizes in one group, each layer its own region; a check's bytes are its
    blocks of every layer's page."""

    PATH = Path(__file__).parent / "data" / "nixl-deepseek-v3.2-decode-steps.jsonl"

    def problems(self, change=None) -> list[str]:
        records = [json.loads(line) for line in self.PATH.read_text().splitlines()]
        layout = next(r["kv_layout"] for r in records if "kv_layout" in r)
        checks = [r["kv_check"] for r in records if "kv_check" in r]
        if change:
            change(layout["groups"][0], checks[0])
        return pd._byte_check_problems(layout, checks, 200, pd._pull_plans(records),
                                       {c["req_id"]: 200 for c in checks})

    def test_a_sparse_mla_check_counts_each_layers_page(self) -> None:
        self.assertEqual(self.problems(), [])

    def test_its_failures_are_reported(self) -> None:
        def kinds_short(group, check):
            group["kinds"][0]["layers"] = 1

        def checked_wrong(group, check):
            group["checked_bytes_per_block"] += 1

        def latent_pages_only(group, check):
            # Every layer counted at the latent's page.
            check["groups"][0]["bytes"] = check["groups"][0]["blocks"] * 4 * 73728

        cases = {
            "has kinds of 3 layers, not its 4": kinds_short,
            "checks 164353 bytes a block, not its kinds' pages": checked_wrong,
            "check disagrees with the layout": latent_pages_only,
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                problems = self.problems(change)
                self.assertTrue(any(message in p for p in problems), problems)


# ----- a hybrid's unequal TP: attention head-v1 beside a Mamba2 state group (state-v1) -----

STATE = {"family": "mamba2", "components": [["x", 16], ["B", 4], ["C", 4]], "conv_rows": 3, "conv_itemsize": 2,
         "ssm_heads": 4, "ssm_head_bytes": 8}
ATTN_HEAD = 10  # bytes of one KV head over a request's blocks of the attention group


def _hybrid_expectations(prefill_tp: int, decode_tp: int, kv_heads: int, only_regions: int = 1, logical_regions: int = 1):
    """Per decode rank and producer: attention heads, the state's classes
    (component -> bytes of a request's one block), written here from the
    ownership rule (a producer's range overlap) for the fixture alone."""
    out = {}
    for d in range(decode_tp):
        cands = ([d * prefill_tp // decode_tp] if prefill_tp <= decode_tp
                 else list(range(d * (prefill_tp // decode_tp), (d + 1) * (prefill_tp // decode_tp))))
        for p in cands:
            heads = [h for h in pd.global_heads(d, decode_tp, kv_heads)
                     if h in pd.global_heads(p, prefill_tp, kv_heads)
                     and p == next(q for q in cands if h in pd.global_heads(q, prefill_tp, kv_heads))]
            classes = {}
            for kind, name, width, unit in ([("conv", n, w, 6) for n, w in STATE["components"]]
                                            + [("ssm", "ssm", STATE["ssm_heads"], STATE["ssm_head_bytes"])]):
                lo, hi = d * width // decode_tp, (d + 1) * width // decode_tp
                a, b = p * width // prefill_tp, (p + 1) * width // prefill_tp
                if max(lo, a) < min(hi, b):
                    n = (min(hi, b) - max(lo, a)) * unit
                    for tag, regs in (("logical", logical_regions), ("transfer-only", only_regions)):
                        if regs:
                            classes[f"g1.{kind}.{name}.{tag}"] = n * regs
            if heads or classes:
                out[(d, p)] = (heads, classes)
    return out


def hybrid_logs(prefill_tp, decode_tp, kv_heads=4, requests=("a", "b"), only_regions=1):
    """Logs of a hybrid's unequal-TP run over NIXL consistent with ownership:
    each READ's bytes, its class table, descriptors and length table."""
    expect = _hybrid_expectations(prefill_tp, decode_tp, kv_heads, only_regions)
    tps = (("prefill", prefill_tp), ("decode", decode_tp))

    def layout(rank, tp):
        spans = pd._state_spans(STATE, rank, tp)
        heads = pd.global_heads(rank, tp, kv_heads)
        return line(kv_layout={"tp_rank": rank, "tp_size": tp, "block_bytes": 20, "bytes": 1600, "groups": [
            {"kind": "FullAttentionSpec", "block_size": 16, "layers": 2, "block_bytes": ATTN_HEAD * len(heads),
             "real_page_bytes": ATTN_HEAD * len(heads), "pattern": "head-v1", "global_heads": heads},
            {"kind": "MambaSpec", "block_size": 16, "layers": 2, "pattern": "state-v1", "global_heads": None,
             "state": {**STATE, "spans": spans}}]})

    loads = {d: [] for d in range(decode_tp)}
    checks_of = {d: [] for d in range(decode_tp)}
    sent = {p: 0 for p in range(prefill_tp)}
    received = {d: 0 for d in range(decode_tp)}
    for d in range(decode_tp):
        spans = pd._state_spans(STATE, d, decode_tp)
        span_bytes = sum(sp[3] for sp in spans)
        mine = pd.global_heads(d, decode_tp, kv_heads)
        for r in requests:
            reads = []
            for (dd, p), (heads, classes) in sorted(expect.items()):
                if dd != d:
                    continue
                total = ATTN_HEAD * len(heads) + sum(classes.values())
                a = total // 2
                table = {a: 1, total - a: 1} if a != total - a else {a: 2}
                classes_log = {k: [1, v] for k, v in classes.items()}
                if heads:
                    classes_log["g0.attention.logical"] = [1, ATTN_HEAD * len(heads)]
                reads.append({"rank": p, "completed": True, "descriptors": 2, "bytes": total, "classes": classes_log})
                loads[d].append([total, 0.01, 0.0, 2, 10.0, p, [0, 2, 2, total, table]])
                sent[p] += total
                received[d] += total
            state_bytes = 1 * (1 + only_regions) * span_bytes
            checks_of[d].append(checked(
                r, blocks=6, layers=4, bytes=ATTN_HEAD * len(mine) + state_bytes, posted_matches_checked=True,
                coverage={"problems": [], "reads": reads},
                groups=[{"group": 0, "blocks": 5, "bytes": ATTN_HEAD * len(mine), "mismatched_bytes": 0,
                         "pattern": "head-v1", "global_heads": mine},
                        {"group": 1, "blocks": 1, "regions": 1 + only_regions, "bytes": state_bytes,
                         "transfer_only_bytes": only_regions * span_bytes, "mismatched_bytes": 0,
                         "pattern": "state-v1", "global_heads": None}]))
    prefill = [layout(p, prefill_tp) + allocated(*requests) for p in range(prefill_tp)]
    decode = [layout(d, decode_tp) + "".join(stepped(r, "d", 80, 1, 80) for r in requests)
              + line(kv_transfers={"load": {"count": len(loads[d]), "bytes": sum(e[0] for e in loads[d]),
                                            "each": loads[d]}})
              + "".join(line(kv_check=c) for c in checks_of[d]) for d in range(decode_tp)]
    ports = {"kv": {**{f"p{p}": {"node": "p", "slot": p, "rx": sent[p] + 100, "tx": 40} for p in range(prefill_tp)},
                    **{f"p{prefill_tp + d}": {"node": "d", "slot": d, "rx": 40, "tx": received[d] + 100}
                       for d in range(decode_tp)}}}
    return prefill, decode, ports


class HybridHeterogeneousTpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)
        (self.model / "config.json").write_text(json.dumps({"num_key_value_heads": 4}))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def reconcile(self, prefill_tp, decode_tp, change=None, nixl=True, kv_heads=4, **kw) -> dict:
        prefill, decode, ports = hybrid_logs(prefill_tp, decode_tp, kv_heads, **kw)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        run = pd.Run(**{**RUN.__dict__, "model": self.model,
                        "prefill": pd.Side("p", "kv", tensor_parallel_size=prefill_tp),
                        "decode": pd.Side("d", "kv", tensor_parallel_size=decode_tp),
                        "server": pd.Server(connector="nixl" if nixl else "mooncake")})
        logs = {"prefill": {(0, r): pd._records(text) for r, text in enumerate(prefill)},
                "decode": {(0, r): pd._records(text) for r, text in enumerate(decode)}}
        moved = lambda records: sum(((r.get("kv_transfers") or {}).get("load") or {}).get("bytes", 0) for r in records)
        ranks = {role: {key: {"transfers": {"bytes": moved(records)}, "requests": ["a", "b"]}
                        for key, records in logs[role].items()} for role in logs}
        return pd._hetero_tp(run, {"prefill": prefill_tp, "decode": decode_tp}, logs, ranks, ports)

    def test_state_and_attention_pair_every_direction(self) -> None:
        for sizes in ((1, 2), (2, 1), (2, 4), (4, 2), (1, 4), (4, 1)):
            with self.subTest(sizes=sizes):
                hetero = self.reconcile(*sizes)
                self.assertEqual(hetero["problems"], [], sizes)
                self.assertTrue(hetero["hybrid"])
        # TP2 <- TP4 (K=4): decode 0 takes heads 0,1 and half its state from 0 and 1.
        self.assertEqual(self.reconcile(4, 2)["sources"], {0: [0, 1], 1: [2, 3]})
        self.assertEqual(self.reconcile(2, 4)["sources"], {0: [0], 1: [0], 2: [1], 3: [1]})
        classes = self.reconcile(2, 1)["byte_classes"]
        self.assertGreater(classes["state_logical"], 0)
        self.assertEqual(classes["state_transfer_only"], classes["state_logical"])

    def test_transfer_only_regions_are_apart_from_logical_state(self) -> None:
        none = self.reconcile(1, 2, only_regions=0)
        self.assertEqual(none["problems"], [])
        self.assertEqual(none["byte_classes"]["state_transfer_only"], 0)
        two = self.reconcile(1, 2, only_regions=2)
        self.assertEqual(two["problems"], [])
        self.assertEqual(two["byte_classes"]["state_transfer_only"], 2 * two["byte_classes"]["state_logical"])

    def test_mooncake_does_not_pair_a_hybrid(self) -> None:
        problems = self.reconcile(1, 2, nixl=False)["problems"]
        self.assertTrue(any("NIXL only" in p for p in problems), problems)

    def mutate(self, text, find, replace, count=1):
        self.assertIn(find, text)
        return text.replace(find, replace, count)

    def test_state_spans_must_be_the_models_layout(self) -> None:
        # A rank's span offsets or lengths other than the layout's.
        def spans(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"spans": [["conv", "x", 0, 48', '"spans": [["conv", "x", 0, 54'),
                             *decode[1:]], ports
        problems = self.reconcile(1, 2, change=spans)["problems"]
        self.assertTrue(any("state spans" in p for p in problems), problems)
        # Another layout on one side.
        def other(prefill, decode, ports):
            return [self.mutate(prefill[0], '"ssm_heads": 4', '"ssm_heads": 8'), *prefill[1:]], decode, ports
        self.assertTrue(any("differs from" in p or "state spans" in p for p in self.reconcile(1, 2, change=other)["problems"]))

    def test_the_posted_read_record_is_required_and_must_be_clean(self) -> None:
        def strip(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"coverage": {"problems": [], "reads"', '"nocoverage": {"reads"'),
                             *decode[1:]], ports
        self.assertTrue(any("no READ coverage" in p for p in self.reconcile(1, 2, change=strip)["problems"]))
        def flagged(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"coverage": {"problems": []',
                                         '"coverage": {"problems": ["a shard twice"]'), *decode[1:]], ports
        self.assertTrue(any("a shard twice" in p for p in self.reconcile(1, 2, change=flagged)["problems"]))
        def unfinished(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"completed": true', '"completed": false'), *decode[1:]], ports
        self.assertTrue(any("not complete" in p for p in self.reconcile(1, 2, change=unfinished)["problems"]))
        def unmatched(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"posted_matches_checked": true', '"posted_matches_checked": false'),
                             *decode[1:]], ports
        self.assertTrue(any("did not match its posted READs" in p for p in self.reconcile(1, 2, change=unmatched)["problems"]))

    def test_a_reads_bytes_lengths_and_counts_are_exact(self) -> None:
        def bytes_off(prefill, decode, ports):  # a READ's bytes differ from its selected descriptors' sum
            return prefill, [re.sub(r'\[(\d+), 0.01, 0.0, 2, 10.0, 0, \[0, 2, 2, (\d+)', lambda m: f"[{int(m.group(1)) + 8}, 0.01, 0.0, 2, 10.0, 0, [0, 2, 2, {m.group(2)}",
                                    decode[0], count=1), *decode[1:]], ports
        self.assertTrue(any("selected descriptors sum" in p for p in self.reconcile(1, 2, change=bytes_off)["problems"]))
        def merged(prefill, decode, ports):  # predicted != measured
            return prefill, [self.mutate(decode[0], ", 2, 10.0, 0, [0, 2, 2,", ", 2, 10.0, 0, [0, 2, 1,"), *decode[1:]], ports
        self.assertTrue(any("predicted" in p for p in self.reconcile(1, 2, change=merged)["problems"]))
        # A merged READ (predicted = measured < selected) is fine.
        def success(prefill, decode, ports):
            text = self.mutate(decode[0], ", 2, 10.0, 0, [0, 2, 2,", ", 1, 10.0, 0, [0, 2, 1,")
            return prefill, [text, *decode[1:]], ports
        self.assertEqual(self.reconcile(1, 2, change=success)["problems"], [])

    def test_a_swapped_source_or_wrong_class_bytes_are_rejected(self) -> None:
        def swapped(prefill, decode, ports):
            text = decode[0].replace('"rank": 0', '"rank": 9', 1)
            return prefill, [text, *decode[1:]], ports
        self.assertTrue(self.reconcile(1, 2, change=swapped)["problems"])
        def wrong_class(prefill, decode, ports):
            return prefill, [self.mutate(decode[0], '"g1.conv.x.logical": [1, 48]', '"g1.conv.x.logical": [1, 42]'),
                             *decode[1:]], ports
        self.assertTrue(self.reconcile(1, 2, change=wrong_class)["problems"])


if __name__ == "__main__":
    unittest.main()


# Prefix caching: hits checked on both instances, pulls past decode's hit.
PREFIX = pd.Run(**{**RUN.__dict__, "server": pd.Server(enable_prefix_caching=True)})
FULL_LAYOUT = line(kv_layout={
    "tp_rank": 0, "tp_size": 1, "block_bytes": 8, "bytes": 1600,
    "groups": [{"kind": "FullAttentionSpec", "sliding_window": None, "block_size": 16, "layers": 2,
                "block_bytes": 4}]})


def admit(request: str, side: str, computed: int, blocks: list[int]) -> str:
    """A request admitted (new or resumed) with `computed` tokens and its
    blocks then."""
    rid = f"cmpl-{request}-0-{side}"
    return line(kv_events={"allocated": {rid: [b for b in blocks if b]}, "admitted": {rid: computed},
                           "allocated_groups": {rid: [{"group": 0, "first": 0, "blocks": blocks}]}})


def post(request: str, first: int, blocks: list[int]) -> str:
    """Decode's pull of a request's blocks from place `first`, posted."""
    rid = f"cmpl-{request}-0-d"
    return line(kv_events={"allocated": {rid: blocks}, "reads_posted": [rid],
                           "allocated_groups": {rid: [{"group": 0, "first": first, "blocks": blocks}]}})


def hit(request: str, side: str, blocks: int, ok: bool = True, first_block: int | None = None) -> dict:
    """A hit check of a request's first `blocks` places: prefill's blocks
    from 1, decode's from 101, unless `first_block`."""
    first_block = (1 if side == "p" else 101) if first_block is None else first_block
    return {"req_id": f"cmpl-{request}-0-{side}", "kind": "hit", "blocks": blocks, "bytes": blocks * 8,
            "groups": [{"group": 0, "logical_blocks": list(range(blocks)),
                        "block_ids": list(range(first_block, first_block + blocks)), "blocks": blocks,
                        "bytes": blocks * 8, "mismatched_bytes": 0 if ok else 3, "written_this_step": []}],
            "mismatched_bytes": 0 if ok else 3, "ok": ok}


def blocks(first: int, count: int) -> list[int]:
    return list(range(first, first + count))


# b's 1024-token prompt shares a's first 48 blocks (of 64): prefill admits b
# with them cached; decode too, and pulls b's last 16.
PREFILL_ADMISSIONS = admit("a", "p", 0, blocks(1, 64)) + admit("b", "p", 768, blocks(1, 48) + blocks(201, 16))
DECODE_ADMISSIONS = (post("a", 0, blocks(101, 64)) + post("b", 48, blocks(301, 16))
                     + admit("a", "d", 1023, blocks(101, 64)) + admit("b", "d", 1023, blocks(101, 48) + blocks(301, 16)))
A_CHECK = checked("a", blocks=64, layers=2, bytes=64 * 8, groups=[group_check(0, 0, 64)])
B_CHECK = checked("b", blocks=16, layers=2, bytes=16 * 8, groups=[group_check(0, 48, 16)])


class PrefixCachingCheckTests(unittest.TestCase):
    def verdict(self, decode_checks=None, prefill_checks=None, prefill_events="", decode_events="",
                run=PREFIX, moved=(64 + 16) * 8, prefill_admissions=PREFILL_ADMISSIONS,
                decode_admissions=DECODE_ADMISSIONS, decode_steps=None) -> dict:
        decode_checks = [A_CHECK, hit("b", "d", 48), B_CHECK] if decode_checks is None else decode_checks
        prefill_checks = [hit("b", "p", 48)] if prefill_checks is None else prefill_checks
        decode_steps = decode_steps or stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 1023, 1, 1024)
        prefill = (FULL_LAYOUT + prefill_admissions + prefill_events
                   + "".join(line(kv_check=c) for c in prefill_checks)
                   + line(kv_transfers={"send": {"count": 2, "bytes": moved, "failed": 0}}))
        decode = (FULL_LAYOUT + decode_steps + decode_admissions + decode_events
                  + "".join(line(kv_check=c) for c in decode_checks))
        return pd.check({"completed": run.workload.num_prompts, "errors": []}, [prefill], [decode],
                        {"control": 50, "data0": moved + 100}, run)

    def assertProblem(self, verdict: dict, *messages: str) -> None:
        for message in messages:
            self.assertTrue(any(message in problem for problem in verdict["problems"]), verdict["problems"])

    def test_hits_are_checked_apart_from_transfers(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual((verdict["kv_checks"], verdict["kv_bytes_checked"]), (2, 80 * 8))
        self.assertEqual(verdict["prefix_hits"], {
            "prefill": {"kv_hit_checks": 1, "kv_hit_blocks": 48, "kv_bytes_hit_checked": 48 * 8},
            "decode": {"kv_hit_checks": 1, "kv_hit_blocks": 48, "kv_bytes_hit_checked": 48 * 8}})
        self.assertEqual(verdict["fragmentation"]["contiguous"], 2)
        # Without prefix caching a pull starts at the prompt's first block.
        off = self.verdict(run=pd.Run(**{**RUN.__dict__, "server": pd.Server(block_size=16)}))
        self.assertProblem(off, "pulled from block 48, not 0")
        self.assertIsNone(off["prefix_hits"])

    def test_prefix_caching_failures_are_reported(self) -> None:
        outside = {"req_id": "cmpl-b-0-p", "kind": "fill", "ok": False, "error": "outside the lifecycle"}
        cases = {
            "prefill: 1 prefix hit(s) held bytes unlike": {"prefill_checks": [hit("b", "p", 48, ok=False)]},
            "decode: 1 prefix hit(s) held bytes unlike": {"decode_checks": [A_CHECK, hit("b", "d", 48, False),
                                                                            B_CHECK]},
            "outside the pattern lifecycle": {"prefill_checks": [hit("b", "p", 48), outside]},
            "prefill showed no checked prefix hit": {"prefill_checks": []},
            # The pull past the hit takes the rest of the prompt, not less.
            "checked 15 blocks, not the cdiv(1024, 16) of its prompt past its 48-block prefix hit": {
                "decode_checks": [A_CHECK, hit("b", "d", 48),
                                  checked("b", blocks=15, layers=2, bytes=120, groups=[group_check(0, 48, 15)])],
                "moved": 79 * 8},
        }
        for message, changes in cases.items():
            with self.subTest(message=message):
                self.assertProblem(self.verdict(**changes), message)

    def test_hits_must_cover_each_admitted_prefix(self) -> None:
        """Response 128: a hit check must be the request's own, of exactly
        the places its admission left to the cache, its blocks there."""
        cases = {
            # b's 48-block hit missing, a's unrelated one-block hit instead.
            "unrelated": ({"decode_checks": [A_CHECK, hit("a", "d", 1), B_CHECK]},
                          ["decode: cmpl-a-0-d logged 1 prefix hit check(s), not the 0",
                           "decode: cmpl-b-0-d logged 0 prefix hit check(s), not the 1",
                           "decode showed no checked prefix hit"]),
            "missing": ({"decode_checks": [A_CHECK, B_CHECK]},
                        ["cmpl-b-0-d logged 0 prefix hit check(s), not the 1 its admissions imply ([48] block(s))"]),
            "shortened": ({"decode_checks": [A_CHECK, hit("b", "d", 47), B_CHECK]},
                          ["cmpl-b-0-d's hit check covers group 0: 47 block(s) 0:101..46:147, not its admitted "
                           "hit group 0: 48 block(s) 0:101..47:148"]),
            "other blocks": ({"prefill_checks": [hit("b", "p", 48, first_block=2)]},
                             ["prefill: cmpl-b-0-p's hit check covers group 0: 48 block(s) 0:2..47:49"]),
            "zero blocks and bytes": ({"decode_checks": [A_CHECK, hit("b", "d", 0), B_CHECK],
                                       "prefill_checks": [hit("b", "p", 0)]},
                                      ["prefill: cmpl-b-0-p's hit check covers no bytes",
                                       "decode: cmpl-b-0-d's hit check covers no bytes",
                                       "prefill showed no checked prefix hit", "decode showed no checked prefix hit"]),
            "bytes unlike the layout": ({"prefill_checks": [{**hit("b", "p", 48), "bytes": 48 * 4, "groups": [
                {**hit("b", "p", 48)["groups"][0], "bytes": 48 * 4}]}]},
                ["cmpl-b-0-p's hit check of group 0 (48 block(s), 192 bytes) disagrees with the layout (8 bytes"]),
            # The pull's own plan: its blocks are the request's at their
            # places, and it starts within the tokens it was admitted with.
            "plan unlike the table": ({"decode_admissions": DECODE_ADMISSIONS.replace(
                json.dumps(blocks(301, 16)), json.dumps(blocks(302, 16)), 2)},
                ["cmpl-b-0-d's pull of group 0 (16 block(s) from 48) is not its blocks there when admitted"]),
            "plan past the admission": ({"decode_admissions": DECODE_ADMISSIONS.replace(
                '"cmpl-b-0-d": 1023', '"cmpl-b-0-d": 700')},
                ["cmpl-b-0-d's pull of group 0 starts at block 48, past the 700 tokens"]),
        }
        for name, (changes, messages) in cases.items():
            with self.subTest(name):
                self.assertProblem(self.verdict(**changes), *messages)
        # Prefill admitted b with 48 blocks cached: a 47-block admission
        # makes its 48-block check another's.
        shorter = PREFILL_ADMISSIONS.replace('"cmpl-b-0-p": 768', '"cmpl-b-0-p": 752')
        self.assertProblem(self.verdict(prefill_admissions=shorter), "not its admitted hit group 0: 47 block(s)")

    def test_a_request_served_from_hits_alone_is_consumed_once(self) -> None:
        """a pulled; b's prompt (1025 tokens) is a's and one more: decode
        hits all 64 of its full blocks and pulls nothing; c hits 48 and
        pulls 16."""
        run = pd.Run(**{**PREFIX.__dict__, "workload": pd.Workload(num_prompts=3)})
        prefill = (admit("a", "p", 0, blocks(1, 64)) + admit("b", "p", 1024, blocks(1, 64) + [217])
                   + admit("c", "p", 768, blocks(1, 48) + blocks(201, 16)))
        decode = (post("a", 0, blocks(101, 64)) + admit("a", "d", 1023, blocks(101, 64))
                  + admit("b", "d", 1024, blocks(101, 64) + [400])
                  + post("c", 48, blocks(301, 16)) + admit("c", "d", 1023, blocks(101, 48) + blocks(301, 16)))
        steps = (stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 1024, 1, 1025)
                 + stepped("c", "d", 1023, 1, 1024))
        c_check = {**B_CHECK, "req_id": "cmpl-c-0-d"}
        good = dict(run=run, prefill_admissions=prefill, decode_admissions=decode, decode_steps=steps,
                    decode_checks=[A_CHECK, hit("b", "d", 64), hit("c", "d", 48), c_check],
                    prefill_checks=[hit("b", "p", 64), hit("c", "p", 48)])
        verdict = self.verdict(**good)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["kv_checks"], 2)
        self.assertEqual(verdict["prefix_hits"]["decode"], {"kv_hit_checks": 2, "kv_hit_blocks": 112,
                                                            "kv_bytes_hit_checked": 112 * 8})
        self.assertEqual(verdict["ownership"], {"producers": {0: 3}, "consumers": {0: 3}})
        # b's hit not checked: nothing shows decode consumed it.
        self.assertProblem(self.verdict(**{**good, "decode_checks": [A_CHECK, hit("c", "d", 48), c_check]}),
                           "decode admitted cmpl-b-0-d with neither a pull nor a checked prefix hit")
        # c pulled: its byte check is still required.
        self.assertProblem(self.verdict(**{**good, "decode_checks": [A_CHECK, hit("b", "d", 64),
                                                                     hit("c", "d", 48)]}),
                           "1 pulled request(s) without a successful byte check: cmpl-c-0-d",
                           "decode logged 1 byte check(s) for 2 distinct request(s) that pulled")
        # Response 129: b reuses 768 of its 1024 tokens and prefills the
        # other 256 itself, with no pull: a missing transfer, not a full hit.
        partial = dict(run=PREFIX, decode_checks=[A_CHECK, hit("b", "d", 48)], moved=512,
                       decode_admissions=post("a", 0, blocks(101, 64)) + admit("a", "d", 1023, blocks(101, 64))
                       + admit("b", "d", 768, blocks(101, 48) + blocks(301, 16)),
                       decode_steps=stepped("a", "d", 1023, 1, 1024) + stepped("b", "d", 768, 256, 1024))
        self.assertProblem(self.verdict(**partial),
                           "decode admitted cmpl-b-0-d without a pull, 768 of its 1024 prompt tokens cached, its "
                           "first step computing from 768: it prefilled 256 tokens itself, not only the last")
        # Its admission alone claiming the whole prompt does not hide the step.
        whole = hit("b", "d", 63)
        whole["groups"][0]["block_ids"] = blocks(101, 48) + blocks(301, 15)
        claimed = {**partial, "decode_checks": [A_CHECK, whole],
                   "decode_admissions": partial["decode_admissions"].replace('"cmpl-b-0-d": 768', '"cmpl-b-0-d": 1023')}
        self.assertEqual([p for p in self.verdict(**claimed)["problems"] if "hit" in p and "prefilled" not in p], [])
        self.assertProblem(self.verdict(**claimed), "it prefilled 256 tokens itself")
        # A request decode never admitted is not consumed.
        self.assertProblem(self.verdict(**{**good, "decode_admissions": decode.replace(
            admit("b", "d", 1024, blocks(101, 64) + [400]), "")}), "decode admitted 2 distinct request(s), not 3")

    def test_fragmentation_counts_checked_matched_pulls_only(self) -> None:
        def pull(request: str, places: list[int], first: int = 2) -> dict:
            return {"kv_events": {"reads_posted": [f"cmpl-{request}-0-d"], "allocated_groups": {
                f"cmpl-{request}-0-d": [{"group": 0, "first": first, "blocks": places}]}}}

        # Prefill's tables (a chunk's allocations continue at their places).
        prefill = [pd._records(allocated_groups("a", (0, 0, [1, 2, 3, 4])) + allocated_groups("b", (0, 0, [5, 6]))
                               + allocated_groups("b", (0, 2, [9, 7]))
                               + allocated_groups("c", (0, 0, [10, 11, 12, 13]))
                               + allocated_groups("d", (0, 0, [14, 15, 17, 19])))]
        decode = [pull("a", [20, 21]), pull("b", [30, 31]), pull("c", [40, 42]),
                  pull("x", [50]),  # no producer allocation logged: unmatched
                  pull("d", [60, 62]),  # fragmented on both, but not checked
                  {"kv_events": {"reads_posted": ["cmpl-y-0-d"], "allocated": {"cmpl-y-0-d": [70, 72]}}}]
        checked_ids = {f"cmpl-{r}-0-d" for r in "abcxy"}
        self.assertEqual(pd._fragmentation(prefill, [decode], [checked_ids]), {
            "pulls": 6, "contiguous": 1, "fragmented_prefill_only": 1, "fragmented_decode_only": 1,
            "fragmented_both": 0, "unlogged": 1, "unchecked": 1, "unmatched": 1,
            "jointly_contiguous_runs": 1 + 2 + 2, "unmatched_groups": 1})
        # Checked, d is fragmented on both.
        self.assertEqual(pd._fragmentation(prefill, [decode], [checked_ids | {"cmpl-d-0-d"}])["fragmented_both"], 1)
        # A checked pull without the producer's allocation is not contiguous.
        alone = pd._fragmentation([], [[pull("missing", [50])]], [{"cmpl-missing-0-d"}])
        self.assertEqual((alone["contiguous"], alone["unmatched"], alone["unmatched_groups"]), (0, 1, 1))
        # In a run: neither an unchecked pull nor an unmatched one meets
        # expect_fragmentation (response 128's counterexamples).
        expecting = pd.Run(**{**PREFIX.__dict__, "expect_fragmentation": True})
        for prefill_events, decode_events in (
                (allocated_groups("unverified", (0, 0, [14, 15, 17, 19])), line(**pull("unverified", [60, 62]))),
                ("", line(**pull("missing", [50])))):
            verdict = self.verdict(prefill_events=prefill_events, decode_events=decode_events, run=expecting)
            self.assertProblem(verdict, "no byte-checked transfer was fragmented on both instances",
                               "1 pull(s) posted for requests decode never admitted")
            self.assertEqual((verdict["fragmentation"]["contiguous"], verdict["fragmentation"]["unchecked"]), (2, 1))


# hp_tcp's task limit: connector writes reported, TENT's failures checked.
def sends(*writes, failed=(), t=20.0) -> str:
    """A Mooncake prefill step's writes: (bytes, descriptors, start, end)
    each, and failed ones (bytes, descriptors, start), reported at `t`."""
    send = {"count": len(writes), "bytes": sum(w[0] for w in writes),
            "each": [[b, end - start, 0.0, n, end] for b, n, start, end in writes]}
    if failed:
        send |= {"failed": len(failed), "failed_each": [list(f) for f in failed]}
    return line(kv_transfers={"send": send}, t=t)


class HpTcpEvidenceTests(unittest.TestCase):
    def verdict(self, writes: str, prefill_tent=None, run=RUN) -> dict:
        prefill = ONE_RANK + allocated("a", "b") + writes
        tents = {"prefill": [{**tent(), **(prefill_tent or {})}], "decode": [tent(0)]}
        return pd.check({"completed": 2, "errors": []}, [prefill], [GOOD["decode"]],
                        {"control": 50, "data0": 2100}, run, tents)

    def hp_tcp(self, verdict: dict) -> dict:
        return verdict["dp_ranks"]["prefill"][0]["hp_tcp"]

    def test_a_mergeable_write_above_the_limit_is_reported_not_rejected(self) -> None:
        # 5,000 descriptors TENT merged into 3,000 tasks, admitted.
        verdict = self.verdict(sends((2000, 5000, 10.0, 10.2)), {"tent_transport_attempts_total": 3000})
        self.assertEqual(verdict["problems"], [])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["max_connector_descriptors_per_write"], evidence["transport_attempts"],
                          evidence["task_failures"], evidence["task_limit"]), (5000, 3000, 0, 4096))
        self.assertIn("exceeds the task limit 4096 (TENT may merge them", evidence["notes"][0])
        self.assertEqual(verdict["hp_tcp_task_limit"], 4096)

    def test_a_recovered_attempt_still_fails_the_clean_transfer_check(self) -> None:
        # Refused once, then admitted: the connector saw success.
        verdict = self.verdict(sends((1000, 2288, 10.0, 10.5), (1000, 2548, 9.95, 10.1)),
                               {"tent_task_failures_total": 2288, "tent_transport_attempt_failures_total": 2288})
        self.assertEqual(verdict["problems"], [
            "prefill's TENT recorded 2288 failed task(s) and 2288 failed transport attempt(s): not a clean "
            "transfer, whether or not a retry recovered it (0 connector transfer(s) failed)"])
        self.assertEqual(self.hp_tcp(verdict)["failed_connector_writes"], 0)

    def test_an_overlap_above_the_limit_without_failures_passes(self) -> None:
        verdict = self.verdict(sends((1000, 3000, 10.0, 10.1), (1000, 2500, 10.05, 10.12)))
        self.assertEqual(verdict["problems"], [])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["overlapping_call_descriptor_bound"], evidence["bound_complete"]), (5500, True))
        self.assertIn("writes of 5500 connector descriptors overlapped, above the task limit 4096 (an upper bound",
                      evidence["notes"][0])

    def test_failed_writes_count_and_unknown_spans_leave_the_bound_incomplete(self) -> None:
        # A failed write ends no later than the record reporting it.
        evidence = pd._hp_tcp_writes(pd._records(sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, 10.05)])), 4096)
        self.assertEqual((evidence["overlapping_call_descriptor_bound"], evidence["bound_complete"],
                          evidence["failed_connector_writes"], evidence["max_connector_descriptors_per_write"]),
                         (3100, True, 1, 3000))
        for name, records in (
                ("no start", sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, None)])),
                ("reported before its start", sends((1000, 100, 10.0, 10.1), failed=[(500, 3000, 30.0)])),
                ("counted, not listed", line(kv_transfers={"send": {"count": 0, "bytes": 0, "failed": 1}})),
                ("a write without its end", line(kv_transfers={"send": {"count": 1, "bytes": 1000,
                                                                        "each": [[1000, 0.1, 0.0, 100, None]]}}))):
            with self.subTest(name):
                evidence = pd._hp_tcp_writes(pd._records(records), 4096)
                self.assertFalse(evidence["bound_complete"])

    def test_the_c6_shape(self) -> None:
        # 2,288 refused while 2,548 were in flight, then admitted; a write of
        # two requests' 4,264 refused twice, failed.
        writes = sends((1000, 2548, 11.0, 11.2), (1000, 2288, 11.02, 11.52), failed=[(1000, 4264, 12.0)], t=14.1)
        verdict = self.verdict(writes, {"tent_task_failures_total": 10816,
                                        "tent_transport_attempt_failures_total": 10816})
        self.assertIn("prefill's connector failed 1 transfer(s)", verdict["problems"])
        self.assertTrue(any("TENT recorded 10816 failed task(s) and 10816 failed transport attempt(s)" in p
                            and "(1 connector transfer(s) failed)" in p for p in verdict["problems"]),
                        verdict["problems"])
        evidence = self.hp_tcp(verdict)
        self.assertEqual((evidence["max_connector_descriptors_per_write"], evidence["overlapping_call_descriptor_bound"],
                          evidence["failed_connector_writes"], evidence["bound_complete"]), (4264, 4836, 1, True))

    def test_the_effective_limit(self) -> None:
        conf = lambda hp_tcp: {"MC_TENT_CONF": json.dumps({"transports": {"hp_tcp": hp_tcp}})}
        for server, limit in (
                (pd.Server(), 4096),
                (pd.Server(hp_tcp_args={"max_outstanding_tasks": 16384}), 16384),
                # A spec's own TENT configuration replaces the harness's.
                (pd.Server(hp_tcp_args={"max_outstanding_tasks": 16384},
                           env=conf({"enable": True, "max_outstanding_tasks": 8192})), 8192),
                (pd.Server(env=conf({"enable": True})), 4096),
                (pd.Server(env={"MC_TENT_CONF": "{not json"}), None),
                (pd.Server(env={"MC_TENT_CONF": json.dumps({"transports": {}})}), None),
                (pd.Server(env={"MC_USE_TENT": 0}), None)):
            with self.subTest(server=server):
                self.assertEqual(pd.hp_tcp_task_limit(server), limit)
        unknown = pd.Run(**{**RUN.__dict__, "server": pd.Server(env={"MC_TENT_CONF": "{not json"})})
        verdict = self.verdict(sends((2000, 100, 10.0, 10.1)), run=unknown)
        self.assertIn("task limit is unknown", self.hp_tcp(verdict)["notes"][0])
        # NIXL: TENT's counters, no connector writes.
        self.assertIsNone(self.verdict(sends((2000, 100, 10.0, 10.1)))["dp_ranks"]["decode"][0]["hp_tcp"].get(
            "max_connector_descriptors_per_write"))


# Unequal TP: heads re-sharded, reconciled with the native pairing.
SLAB = 10  # a head's bytes in a block of a layer
PER_HEAD = 100  # a request's bytes of one head (5 blocks x 2 layers x SLAB)


def hetero_logs(prefill_tp: int, decode_tp: int, kv_heads: int, nixl: bool, requests=("a", "b"), groups=None):
    """Consistent step logs, transfers, ports and TENT scrapes of a run whose
    instances have unequal TP sizes: each request's heads moved as the
    native pairing gives them, a READ's descriptors one per block and layer,
    a replica's answers naming their requests, and each rank's TENT series
    as a scrape serves them (one never incremented absent). `groups`: KV
    cache groups of (kind, total heads, slab) each, in place of one group of
    `kv_heads` heads of SLAB bytes (Gemma4: sliding 8 heads of 10 bytes, full
    2 of 40: pages of equal bytes at TP1)."""
    groups = groups or [("FullAttentionSpec", kv_heads, SLAB)]
    tps = (("prefill", prefill_tp), ("decode", decode_tp))
    gheads = {role: [[pd.global_heads(r, n, g[1]) for g in groups] for r in range(n)] for role, n in tps}
    heads = {role: [gh[0] for gh in gheads[role]] for role in gheads}
    ratio = max(prefill_tp, decode_tp) // min(prefill_tp, decode_tp)
    paired = {d: [d * prefill_tp // decode_tp] if decode_tp >= prefill_tp else
              [d * ratio + i for i in range(ratio)] for d in range(decode_tp)}
    gsender = [{(h, d): min(p for p in paired[d] if h in gheads["prefill"][p][g])
                for d in range(decode_tp) for h in gheads["decode"][d][g]} for g in range(len(groups))]
    ph = [10 * g[2] for g in groups]  # a request's bytes of one head of each group

    def layout(rank, tp, mine):
        return line(kv_layout={"tp_rank": rank, "tp_size": tp, "block_bytes": 20, "bytes": 1600, "groups": [
            {"kind": kind, "sliding_window": None, "block_size": 16, "layers": 2,
             "block_bytes": slab * len(mine[g]), "real_page_bytes": slab * len(mine[g]), "pattern": "head-v1",
             "global_heads": mine[g]} for g, (kind, _, slab) in enumerate(groups)]})

    def group(mine):
        return [{"group": g, "first_logical_block": 0, "blocks": 5, "layers": 2, "block_bytes": groups[g][2] * len(mine[g]),
                 "bytes": ph[g] * len(mine[g]), "mismatched_bytes": 0, "pattern": "head-v1", "global_heads": mine[g]}
                for g in range(len(groups))]

    sends = {p: [] for p in range(prefill_tp)}
    loads = {d: [] for d in range(decode_tp)}
    zero = {p: [] for p in range(prefill_tp)}
    for d in range(decode_tp):
        sources = sorted({gsender[g][(h, d)] for g in range(len(groups)) for h in gheads["decode"][d][g]})
        for p in paired[d]:
            carried = [[h for h in gheads["decode"][d][g] if gsender[g][(h, d)] == p] for g in range(len(groups))]
            if not any(carried):
                zero[p] += [[d, [f"cmpl-{r}-0-d"], [[False, 0, 0, SLAB]]] for r in requests]
                continue
            plan = []
            for g, c in enumerate(carried):
                region = [True, gheads["prefill"][p][g].index(c[0]) * groups[g][2],
                          gheads["decode"][d][g].index(c[0]) * groups[g][2], groups[g][2] * len(c)]
                if c and region not in plan:
                    plan.append(region)
            for _ in requests:
                moved_bytes = sum(ph[g] * len(c) for g, c in enumerate(carried))
                entry = [moved_bytes, 0.01, 0.0, 10 * len(groups), 10.0]
                if nixl:
                    slot = sources.index(p) if len(sources) > 1 else 0
                    loads[d].append(entry + [p, [slot, 10 * len(groups), 10 * len(groups)]])
                else:
                    sends[p].append(entry + [d, plan])

    def transfers(direction, each, zero_payload=()):
        record = {"count": len(each), "bytes": sum(e[0] for e in each), "each": each}
        if zero_payload:
            record["zero_payload"] = list(zero_payload)
        return line(kv_transfers={direction: record})

    prefill = [layout(p, prefill_tp, gheads["prefill"][p]) + allocated(*requests)
               + transfers("send", sends[p], zero[p]) for p in range(prefill_tp)]
    decode = [layout(d, decode_tp, gheads["decode"][d])
              + "".join(stepped(r, "d", 80, 1, 80) for r in requests)
              + transfers("load", loads[d])
              + "".join(line(kv_check=checked(r, blocks=5, layers=2,
                                              bytes=sum(ph[g] * len(gheads["decode"][d][g]) for g in range(len(groups))),
                                              groups=group(gheads["decode"][d]))) for r in requests)
              for d in range(decode_tp)]
    sent = {p: sum(e[0] for e in sends[p]) + sum(e[0] for d in loads for e in loads[d] if e[5] == p)
            for p in range(prefill_tp)}
    received = {d: sum(e[0] for e in loads[d]) + sum(e[0] for p in sends for e in sends[p] if e[5] == d)
                for d in range(decode_tp)}
    ports = {"kv": {**{f"p{p}": {"node": "p", "slot": p, "rx": sent[p] + 100, "tx": 40} for p in range(prefill_tp)},
                    **{f"p{prefill_tp + d}": {"node": "d", "slot": d, "rx": 40, "tx": received[d] + 100}
                       for d in range(decode_tp)}}}

    def scrape(moved=None, kind="write", transfers=0):
        """A rank's TENT series before and after the workload: an engine
        that never moved a byte serves no series for it."""
        def served(moved_bytes, count):
            lines = ["tent_registered_buffer_bytes 1600"]
            if moved is not None:
                lines += [f'tent_{kind}_bytes_total{{transport="tcp"}} {moved_bytes}',
                          f"tent_{kind}_requests_total {count}", f"tent_transport_attempts_total {count}"]
            return "\n".join(lines)

        return pd._tent_delta(served(0, 0), served(moved, transfers))

    if nixl:
        tents = {"prefill": [scrape() for _ in range(prefill_tp)],
                 "decode": [scrape(received[d], "read", len(loads[d])) for d in range(decode_tp)]}
    else:
        tents = {"prefill": [scrape(sent[p], "write", len(sends[p])) if sends[p] else scrape()
                             for p in range(prefill_tp)],
                 "decode": [scrape() for _ in range(decode_tp)]}
    return prefill, decode, ports, sum(sent.values()), tents


# Gemma4's groups: sliding layers of 8 KV heads (10-byte slabs), full ones of
# 2 (40-byte slabs): pages of equal bytes (80) at TP1, a common carried span of
# 40 bytes at TP1 <-> 2.
GEMMA = [("SlidingWindowSpec", 8, 10), ("FullAttentionSpec", 2, 40)]


class HeterogeneousTpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.model = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def run_for(self, prefill_tp, decode_tp, kv_heads, nixl, global_heads=None):
        # kv_heads 1: an MLA model's latent (DeepSeek-V3's config: 128
        # attention KV heads, one latent cached).
        config = {"num_key_value_heads": 128, "kv_lora_rank": 512} if kv_heads == 1 else {
            "num_key_value_heads": kv_heads}
        if global_heads is not None:
            config["num_global_key_value_heads"] = global_heads
        (self.model / "config.json").write_text(json.dumps(config))
        return pd.Run(**{**RUN.__dict__, "model": self.model,
                         "prefill": pd.Side("p", "kv", tensor_parallel_size=prefill_tp),
                         "decode": pd.Side("d", "kv", tensor_parallel_size=decode_tp),
                         "server": pd.Server(connector="nixl" if nixl else "mooncake")})

    def verdict(self, prefill_tp, decode_tp, kv_heads, nixl, change=None, tent_change=None, groups=None,
                global_heads=None) -> dict:
        prefill, decode, ports, moved, tents = hetero_logs(prefill_tp, decode_tp, kv_heads, nixl, groups=groups)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        if tent_change:
            tent_change(tents)
        return pd.check({"completed": 2, "errors": []}, prefill, decode, {"control": 50, "kv": moved + 500},
                        self.run_for(prefill_tp, decode_tp, kv_heads, nixl, global_heads), tents, ports)

    def test_heads_resharded_either_way_over_both_connectors(self) -> None:
        for sizes, kv_heads in (((2, 4), 8), ((4, 2), 8), ((2, 8), 4)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    verdict = self.verdict(*sizes, kv_heads, nixl)
                    self.assertEqual(verdict["problems"], [])
                    hetero = verdict["hetero_tp"]
                    self.assertEqual(hetero["sizes"], {"prefill": sizes[0], "decode": sizes[1]})
                    # Decode replicating heads (TP 8 > K 4) carries each twice.
                    self.assertEqual(hetero["replication_factor"], 2.0 if sizes == (2, 8) else 1.0)
        # Each decode rank reads half a producer's heads (2 -> 4), or
        # gathers two producers' (4 -> 2).
        self.assertEqual(self.verdict(2, 4, 8, True)["hetero_tp"]["sources"], {0: [0], 1: [0], 2: [1], 3: [1]})
        self.assertEqual(self.verdict(4, 2, 8, True)["hetero_tp"]["sources"], {0: [0, 1], 1: [2, 3]})

    def reconcile(self, prefill_tp, decode_tp, kv_heads, nixl, change=None, groups=None, global_heads=None) -> dict:
        """`_hetero_tp` alone over the fixture's logs (several groups' byte
        checks and pull events are `check`'s other rules): its evidence,
        with the problems."""
        prefill, decode, ports, _, tents = hetero_logs(prefill_tp, decode_tp, kv_heads, nixl, groups=groups)
        if change:
            prefill, decode, ports = change(prefill, decode, ports)
        run = self.run_for(prefill_tp, decode_tp, kv_heads, nixl, global_heads)
        logs = {"prefill": {(0, r): pd._records(text) for r, text in enumerate(prefill)},
                "decode": {(0, r): pd._records(text) for r, text in enumerate(decode)}}
        def moved(records):
            return sum(((r.get("kv_transfers") or {}).get(d) or {}).get("bytes", 0)
                       for r in records for d in ("send", "load"))

        ranks = {role: {key: {"transfers": {"bytes": moved(records)}, "requests": ["a", "b"]} for key, records in logs[role].items()}
                 for role in logs}
        return pd._hetero_tp(run, {"prefill": prefill_tp, "decode": decode_tp}, logs, ranks, ports)

    def test_groups_of_differing_total_heads_pair_per_group(self) -> None:
        """Gemma4-like groups (sliding 8 heads, full 2) at TP1 <-> 2 over both
        connectors: each group's heads, slab and bytes are its own, the
        carried span (40 B) is common, and the reconciliation passes."""
        for sizes in ((1, 2), (2, 1)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    hetero = self.reconcile(*sizes, 8, nixl, groups=GEMMA, global_heads=2)
                    self.assertEqual(hetero["problems"], [])
                    self.assertEqual(hetero["group_kv_heads"], [8, 2])
                    self.assertEqual(hetero["slab_bytes"], [10, 40])
                    self.assertEqual(hetero["replication_factor"], 1.0)

    def test_a_wrong_group_total_is_rejected_despite_matching_slabs_and_bytes(self) -> None:
        """The slab counterexample: the full group's total taken as 8 (the
        config's global count wrong) gives the same 10-byte slab on both
        sides (80/8 at TP1, 40/4 at TP2) and the same bytes, yet its actual
        TP2 head set is [0], not four heads: rejected by the logged heads."""
        for nixl in (False, True):
            with self.subTest(nixl=nixl):
                problems = self.reconcile(1, 2, 8, nixl, groups=GEMMA, global_heads=8)["problems"]
                self.assertTrue(any("group 1 holds head-v1 heads [0], not head-v1 heads [0, 1, 2, 3]" in problem
                                    for problem in problems), problems)
                self.assertTrue(any("decode rank 0 checked" in problem and "heads [0]" in problem
                                    for problem in problems), problems)

    def test_group_totals_come_from_the_config_by_kind_or_fail(self) -> None:
        def totals(config, kinds):
            (self.model / "config.json").write_text(json.dumps(config))
            return pd.group_kv_heads(self.model, kinds)

        plain = {"num_key_value_heads": 8}
        self.assertEqual(totals(plain, ["FullAttentionSpec", "SlidingWindowSpec"]), ([8, 8], None))
        gemma = {"num_key_value_heads": 8, "num_global_key_value_heads": 2}
        self.assertEqual(totals(gemma, ["SlidingWindowSpec", "FullAttentionSpec"]), ([8, 2], None))
        # Nested text config, as a multimodal wrapper writes it.
        nested = {"text_config": {"num_key_value_heads": 8, "num_global_key_value_heads": 2}}
        self.assertEqual(totals(nested, ["FullAttentionSpec"]), ([2], None))
        # No default: an unknown kind, or an invalid count, is an error.
        self.assertIsNone(totals(gemma, ["MLAAttentionSpec"])[0])
        self.assertIn("MLAAttentionSpec", totals(gemma, ["MLAAttentionSpec"])[1])
        for bad in (0, -2, True, 2.0, "2"):
            result = totals({**gemma, "num_global_key_value_heads": bad}, ["FullAttentionSpec"])
            self.assertIsNone(result[0], bad)
        self.assertIsNone(totals({"num_global_key_value_heads": 2}, ["FullAttentionSpec"])[0])

    def test_groups_must_agree_across_ranks_and_carry_one_span(self) -> None:
        # A rank whose groups are of other kinds than the rest's.
        def other_kind(prefill, decode, ports):
            decode = list(decode)
            decode[1] = decode[1].replace('"kind": "SlidingWindowSpec"', '"kind": "FullAttentionSpec"', 1)
            return prefill, decode, ports

        found = self.reconcile(1, 2, 8, False, other_kind, groups=GEMMA, global_heads=2)["problems"]
        self.assertTrue(any("groups are" in problem for problem in found), found)
        # A READ whose bytes are not its selected descriptors' carried span.
        def off_by_one(prefill, decode, ports):
            return prefill, [log.replace("[800, 0.01", "[801, 0.01") for log in decode], ports

        found = self.reconcile(1, 2, 8, True, off_by_one, groups=GEMMA, global_heads=2)["problems"]
        self.assertTrue(any("carried span" in problem for problem in found), found)

    def test_a_replicated_producer_answers_without_a_payload(self) -> None:
        """Mooncake 8 -> 2 with 4 KV heads: ranks 0, 2, 4, 6 send; 1, 3, 5, 7
        answer with no payload, which must be recorded."""
        verdict = self.verdict(8, 2, 4, False)
        self.assertEqual(verdict["problems"], [])
        pairs = {(pair["prefill"], pair["decode"]): pair for pair in verdict["hetero_tp"]["pairs"]}
        self.assertEqual(pairs[(1, 0)], {"prefill": 1, "decode": 0, "bytes": 0, "expected": 0,
                                         "zero_payload_answers": 2})
        self.assertEqual(pairs[(2, 0)]["bytes"], 2 * PER_HEAD)

        def silent(prefill, decode, ports):
            records = [json.loads(text) for text in prefill[1].splitlines()]
            for record in records:
                record.get("kv_transfers", {}).get("send", {}).pop("zero_payload", None)
            return [prefill[0], "".join(json.dumps(r) + "\n" for r in records), *prefill[2:]], decode, ports

        problems = self.verdict(8, 2, 4, False, silent)["problems"]
        self.assertIn("prefill rank 1 never answered decode rank 0 (no payload and no zero-payload answer) for 2 "
                      "request(s), cmpl-a-0", problems)

    def test_failures_are_reported(self) -> None:
        def edit(role, rank, old, new):
            def change(prefill, decode, ports):
                logs = {"prefill": list(prefill), "decode": list(decode)}
                self.assertIn(old, logs[role][rank])
                logs[role][rank] = logs[role][rank].replace(old, new)
                return logs["prefill"], logs["decode"], ports
            return change

        def no_ports(prefill, decode, ports):
            return prefill, decode, None

        def short_port(prefill, decode, ports):
            ports["kv"]["p0"]["rx"] = 10
            return prefill, decode, ports

        cases = {
            # Decode rank 1 holding head 0's bytes, not its own.
            "decode rank 1 checked cmpl-a-0-d's heads [0]": (2, 4, 8, False, edit(
                "decode", 1, '"global_heads": [2, 3]}]}', '"global_heads": [0]}]}')),
            "decode rank 1's group 0 holds head-v1 heads [0, 1], not head-v1 heads [2, 3]": (2, 4, 8, False, edit(
                "decode", 1, '"global_heads": [2, 3]}]}}', '"global_heads": [0, 1]}]}}')),
            "prefill rank 0 -> decode rank 1 moved 400 bytes, not the 400": None,
            "plan [True, 0, 0, 20], not [True, 20, 0, 20]": (2, 4, 8, False, edit(
                "prefill", 0, "[[true, 20, 0, 20]]", "[[true, 0, 0, 20]]")),
            "a READ into slot 0, not 1": (4, 2, 8, True, edit(
                "decode", 0, "10.0, 1, [1, 10, 10]], [200", "10.0, 1, [0, 10, 10]], [200")),
            "transfer(s) logged without their peer rank": (2, 4, 8, True, edit(
                "decode", 0, "10.0, 0, [0, 10, 10]]", "10.0]")),
            "a READ of 2000 selected descriptors, more than the 1024 guard": (2, 4, 8, True, edit(
                "decode", 0, "0.0, 10, 10.0, 0, [0, 10, 10]", "0.0, 2000, 10.0, 0, [0, 2000, 2000]")),
            "prefill rank 0's router port RX 10 bytes": (2, 4, 8, False, short_port),
            "no router port counters": (2, 4, 8, False, no_ports),
        }
        cases.pop("prefill rank 0 -> decode rank 1 moved 400 bytes, not the 400")
        for message, (p, d, k, nixl, change) in cases.items():
            with self.subTest(message=message):
                problems = self.verdict(p, d, k, nixl, change)["problems"]
                self.assertTrue(any(message in problem for problem in problems), problems)
        # A pair moving other bytes than the heads it holds.
        problems = self.verdict(2, 4, 8, False, edit("prefill", 0, "[200, 0.01", "[150, 0.01"))["problems"]
        self.assertTrue(any("prefill rank 0 -> decode rank 0 moved" in p and "not the 400" in p
                            for p in problems), problems)

    @staticmethod
    def rewrite(role, rank, change):
        """A change of one rank's step log, record by record."""
        def apply(prefill, decode, ports):
            logs = {"prefill": list(prefill), "decode": list(decode)}
            records = [json.loads(text) for text in logs[role][rank].splitlines()]
            logs[role][rank] = "".join(json.dumps(record) + "\n" for record in map(change, records))
            return logs["prefill"], logs["decode"], ports
        return apply

    @staticmethod
    def each_transfer(direction, change):
        """`change(entry)` of each transfer a record logs (its new entry)."""
        def apply(record):
            transfer = (record.get("kv_transfers") or {}).get(direction)
            if transfer:
                transfer["each"] = [change(list(entry)) for entry in transfer["each"]]
            return record
        return apply

    @staticmethod
    def answers(change):
        """`change(answers)` of a prefill rank's zero-payload answers."""
        def apply(record):
            send = (record.get("kv_transfers") or {}).get("send")
            if send and "zero_payload" in send:
                send["zero_payload"] = change(send["zero_payload"])
            return record
        return apply

    def test_an_mla_latent_reaches_every_decode_rank_whole(self) -> None:
        """MLA: one latent, the same on every rank. Each decode rank receives
        it whole from one producer (decode rank d's first paired one), so the
        payload is decode TP times the unique KV: 8 at 4 -> 8, 4 at 8 -> 4.
        Mooncake's other paired producers (8 -> 4: 1, 3, 5, 7) answer without
        a payload; NIXL reads only from the first."""
        for sizes, factor in (((4, 8), 8.0), ((8, 4), 4.0)):
            for nixl in (False, True):
                with self.subTest(sizes=sizes, nixl=nixl):
                    verdict = self.verdict(*sizes, 1, nixl)
                    self.assertEqual(verdict["problems"], [])
                    hetero = verdict["hetero_tp"]
                    self.assertEqual(hetero["kv_heads"], 1)
                    self.assertEqual(hetero["replication_factor"], factor)
                    self.assertEqual(hetero["payload_bytes"], factor * hetero["unique_kv_bytes"])
                    p, d = sizes
                    if nixl or p < d:
                        self.assertEqual(hetero["sources"], {r: [r * p // d] for r in range(d)})
                    else:
                        self.assertEqual(hetero["sources"], {r: [2 * r, 2 * r + 1] for r in range(d)})
                    self.assertEqual(hetero["idle_replicas"], [1, 3, 5, 7] if (p, d) == (8, 4) and not nixl
                                     else [])
                    self.assertTrue(all(pair["bytes"] == pair["expected"] for pair in hetero["pairs"]))
        # A Mooncake replica's missing answer, as with attention heads.
        problems = self.verdict(8, 4, 1, False, self.rewrite("prefill", 3, self.answers(lambda answers: [])))[
            "problems"]
        self.assertTrue(any("prefill rank 3 never answered decode rank 1" in problem for problem in problems),
                        problems)

    def test_idle_replicas_are_accepted_only_on_their_answers(self) -> None:
        """Mooncake 8 -> 2, 4 KV heads: ranks 1, 3, 5, 7 never write, so their
        TENT scrapes serve no write series (as `_tent_delta` reads them). That
        is accepted only for a replica answering each of its decode ranks'
        requests once, with a healthy endpoint and registration."""
        verdict = self.verdict(8, 2, 4, False)
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["hetero_tp"]["idle_replicas"], [1, 3, 5, 7])
        self.assertEqual(verdict["hetero_tp"]["unwritten_accepted"], [1, 3, 5, 7])
        for rank in (1, 3, 5, 7):
            self.assertNotIn("tent_write_bytes_total", verdict["dp_ranks"]["prefill"][rank]["tent"])
            self.assertIn("tent_write_bytes_total", verdict["dp_ranks"]["prefill"][rank - 1]["tent"])
        lack = "prefill rank 1's TENT metrics lack tent_write_bytes_total, and it is not a replica"

        def problems(change=None, tent_change=None):
            return self.verdict(8, 2, 4, False, change, tent_change)["problems"]

        cases = {
            # Both answers replaced by one counting none (the old record).
            "answered decode rank 0 without a payload for unnamed requests (0)": self.rewrite(
                "prefill", 1, self.answers(lambda answers: [[0, 0, answers[0][2]]])),
            "never answered decode rank 0 (no payload and no zero-payload answer) for 1 request(s), cmpl-b-0":
                self.rewrite("prefill", 1, self.answers(lambda answers: answers[:1])),
            "answered decode rank 0 without a payload twice for 1 request(s), cmpl-a-0": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + answers[:1])),
            "prefill rank 1 answered decode rank 99, not one it pairs with": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + [[99, ["cmpl-a-0-d"], answers[0][2]]])),
            # Rank 1 pairs with decode rank 0 only.
            "prefill rank 1 answered decode rank 1, not one it pairs with": self.rewrite(
                "prefill", 1, self.answers(lambda answers: answers + [[1, ["cmpl-a-0-d"], answers[0][2]]])),
            "answered decode rank 0 without a payload for requests it never pulled: 1 request(s), cmpl-z-0":
                self.rewrite("prefill", 1, self.answers(
                    lambda answers: answers + [[0, ["cmpl-z-0-d"], answers[0][2]]])),
            "answered decode rank 0 without a payload under a plan that sends": self.rewrite(
                "prefill", 1, self.answers(lambda answers: [[0, a[1], [[True, 0, 0, 10]]] for a in answers])),
            # A sending rank that also answers without a payload.
            "prefill rank 0 answered decode rank 0 without a payload for 1 request(s), though it holds heads":
                self.rewrite("prefill", 0, lambda record: {**record, "kv_transfers": {"send": {
                    **record["kv_transfers"]["send"], "zero_payload": [[0, ["cmpl-a-0-d"], [[False, 0, 0, 10]]]]}}}
                    if "kv_transfers" in record else record),
        }
        for message, change in cases.items():
            with self.subTest(message=message):
                found = problems(change)
                self.assertTrue(any(message in problem for problem in found), found)
                # Rank 1's absent write series is no longer accepted.
                if "prefill rank 0" not in message:
                    self.assertTrue(any(lack in problem for problem in found), found)

        def drop(rank, *names):
            def change(tents):
                for name in names:
                    tents["prefill"][rank].pop(name)
            return change

        def silent(rank):
            def change(tents):
                tents["prefill"][rank] = None
            return change

        for message, tent_change in {
                # An active producer without its write series.
                "prefill rank 0's TENT metrics lack tent_write_bytes_total": drop(0, "tent_write_bytes_total"),
                "rank 1: prefill's TENT metrics did not answer": silent(1),
                "rank 1: prefill's TENT metrics lack tent_registered_buffer_bytes": drop(
                    1, "tent_registered_buffer_bytes")}.items():
            with self.subTest(message=message):
                found = problems(tent_change=tent_change)
                self.assertTrue(any(message in problem for problem in found), found)
        # A missing answer fails the rank's absent write series too; with the
        # series served (as zero), only the answer.
        def zero_write(tents):
            tents["prefill"][1]["tent_write_bytes_total"] = 0.0

        found = problems(self.rewrite("prefill", 1, self.answers(lambda answers: answers[:1])), zero_write)
        self.assertFalse(any(lack in problem for problem in found), found)
        self.assertTrue(any("never answered decode rank 0" in problem for problem in found), found)

    def test_plans_and_descriptors_are_measured_against_the_layouts(self) -> None:
        """A head's bytes in a block of a layer (10 here) come from both sides'
        layouts: each Mooncake region's plan must be its heads' places and
        length in them. Each NIXL READ carries three counts: the descriptors
        vLLM selected (each one block of a layer of its heads, within the
        guard), the hook's prediction of what NIXL merges them into, and what
        NIXL measured; measured = predicted <= selected, and a READ without a
        prediction fails."""
        def problems(sizes, nixl, change):
            return self.verdict(*sizes, 8, nixl, change)["problems"]

        def plans(change):
            return self.rewrite("prefill", 0, self.each_transfer(
                "send", lambda entry: entry[:6] + [change(entry[6])]))

        def reads(change):
            return self.rewrite("decode", 0, self.each_transfer("load", change))

        cases = {
            "a transfer of 200 bytes without its plan ([])": ((2, 4), False, plans(lambda plan: [])),
            "a transfer of 200 bytes without its plan (None)": ((2, 4), False, plans(lambda plan: None)),
            # A zero length while bytes moved.
            "plan [True, 0, 0, 0], not [True, 0, 0, 20]": ((2, 4), False, plans(lambda plan: [[True, 0, 0, 0]])),
            # Every length and offset scaled alike (a 64-byte head).
            "plan [True, 0, 0, 128], not [True, 0, 0, 20]": ((2, 4), False, plans(
                lambda plan: [[s, o // 10 * 64, d // 10 * 64, n // 10 * 64] for s, o, d, n in plan])),
            # A second region of another geometry.
            "plan [True, 0, 0, 10], not [True, 0, 0, 20]": ((2, 4), False, plans(
                lambda plan: plan + [[True, 0, 0, 10]])),
            "a READ without NIXL's measured descriptors (None; 10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:3] + [None] + entry[4:])),
            # The hook's count alone, as before the prediction: not accepted.
            "a READ without the hook's predicted merged descriptors (None; 10 selected, NIXL measured 10)": (
                (2, 4), True, reads(lambda entry: entry[:6] + [entry[6][:2]])),
            "a READ without its selected descriptors (None)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [entry[6][:1]])),
            "a READ without the hook's predicted merged descriptors (0;": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 0]])),
            # A wrong prediction, and a merge predicted that NIXL did not make.
            "a READ NIXL measured at 10 descriptors, predicted at 9 (10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 9]])),
            "a READ NIXL measured at 10 descriptors, predicted at 2 (10 selected)": ((2, 4), True, reads(
                lambda entry: entry[:6] + [[entry[6][0], 10, 2]])),
            # More descriptors than vLLM selected, measured and predicted alike.
            "a READ NIXL measured at 12 descriptors, more than its 10 selected": ((2, 4), True, reads(
                lambda entry: entry[:3] + [12] + entry[4:6] + [[entry[6][0], 10, 12]])),
            "a READ predicted at 12 merged descriptors, more than its 10 selected": ((2, 4), True, reads(
                lambda entry: entry[:3] + [12] + entry[4:6] + [[entry[6][0], 10, 12]])),
            "prefill rank 0 -> decode rank 0: a transfer of 200 bytes without its plan (None)": (
                (4, 2), True, reads(lambda entry: entry[:6] + [None])),
            # Bytes that are not its descriptors' blocks of its heads.
            "a READ of 200 bytes in 5 selected descriptors, not 2 head(s) of 10 bytes each": ((2, 4), True, reads(
                lambda entry: entry[:3] + [5] + entry[4:6] + [[entry[6][0], 5, 5]])),
            # A layout whose heads' bytes disagree with the other side's.
            "the ranks' layouts give no single head size": ((2, 4), False, self.rewrite(
                "decode", 1, lambda record: {"kv_layout": {**record["kv_layout"], "groups": [
                    {**record["kv_layout"]["groups"][0], "real_page_bytes": 64}]}}
                if "kv_layout" in record else record)),
        }
        for message, (sizes, nixl, change) in cases.items():
            with self.subTest(message=message):
                found = problems(sizes, nixl, change)
                self.assertTrue(any(message in problem for problem in found), found)
        # Split, gather and replicated heads, both connectors, pass as logged.
        for sizes, kv_heads, nixl in (((2, 4), 8, False), ((2, 4), 8, True), ((4, 2), 8, False),
                                      ((4, 2), 8, True), ((2, 8), 4, False), ((2, 8), 4, True),
                                      ((8, 2), 4, False)):
            with self.subTest(sizes=sizes, nixl=nixl):
                verdict = self.verdict(*sizes, kv_heads, nixl)
                self.assertEqual(verdict["problems"], [])
                self.assertEqual(verdict["hetero_tp"]["slab_bytes"], SLAB)
                if nixl:
                    self.assertEqual((verdict["hetero_tp"]["max_descriptors"],
                                      verdict["hetero_tp"]["max_merged_descriptors"]), (10, 10))
        # Decode rank 0's whole-page READs, whose 10 selected descriptors NIXL
        # merged into 2, as predicted: their bytes are still the selected ones'.
        for sizes in ((2, 4), (4, 2)):
            with self.subTest(sizes=sizes, merged=True):
                verdict = self.verdict(*sizes, 8, True, reads(
                    lambda entry: entry[:3] + [2] + entry[4:6] + [[entry[6][0], 10, 2]]))
                self.assertEqual(verdict["problems"], [])
                self.assertEqual(verdict["hetero_tp"]["max_descriptors"], 10)

    def test_unequal_sizes_need_one_switched_network(self) -> None:
        base = (E2E / "runs" / "pd-smoke.yaml").read_text().replace("model: ", f"model: {E2E / 'runs'}/", 1)
        for prefill, decode, error in (
                ("{node: prefill0, kv_network: data0, tensor_parallel_size: 2}",
                 "{node: decode0, kv_network: data0, tensor_parallel_size: 3}", "divide one another"),
                ("{node: prefill0, kv_network: [data0, data1], tensor_parallel_size: 2}",
                 "{node: decode0, kv_network: data0, tensor_parallel_size: 4}", "one switched network"),
                ("{node: prefill0, kv_network: data0, tensor_parallel_size: 2, data_parallel_size: 2}",
                 "{node: decode0, kv_network: data0, tensor_parallel_size: 4}", "without data parallelism"),
                ("{node: prefill0, kv_network: data0, tensor_parallel_size: 0}",
                 "{node: decode0, kv_network: data0}", "positive integer")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                spec = Path(directory) / "run.yaml"
                text = re.sub(r"^prefill:.*$", f"prefill: {prefill}", base, count=1, flags=re.M)
                spec.write_text(re.sub(r"^decode:.*$", f"decode: {decode}", text, count=1, flags=re.M))
                with self.assertRaisesRegex(ValueError, error):
                    pd.load(spec)

    def test_rank_r_takes_link_r_of_a_switched_network(self) -> None:
        interfaces = {"control": {"address": "10.250.0.1", "network": "control", "slot": 0},
                      **{f"kvr{k}": {"address": f"10.102.0.{2 * k + 1}", "network": "kv", "slot": k}
                         for k in range(4)}}
        self.assertEqual(pd._kv_addresses(interfaces, pd.Side("p", "kv"), 4),
                         ["10.102.0.1", "10.102.0.3", "10.102.0.5", "10.102.0.7"])
        with self.assertRaisesRegex(ValueError, "4 links on kv, not one per rank"):
            pd._kv_addresses(interfaces, pd.Side("p", "kv"), 8)
        # A single link (or an older topology's state) serves every rank.
        self.assertEqual(pd._kv_addresses({"data0": {"address": "10.100.0.1"}}, pd.Side("p", "data0"), 2),
                         ["10.100.0.1", "10.100.0.1"])
        self.assertEqual(pd.global_heads(3, 8, 4), [1])
        self.assertEqual(pd.global_heads(1, 2, 8), [4, 5, 6, 7])
