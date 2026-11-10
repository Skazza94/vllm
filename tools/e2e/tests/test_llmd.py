"""llm-d's router: a run's router configs, the router evidence check, and
(E2E_DOCKER=1) the llm-d containers' cleanup."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import yaml

from harness import llmd, pd, shell, spec, topology


E2E = Path(__file__).resolve().parent.parent


def run_config(**router) -> dict:
    return {"name": "llmd-test", "topology": "llmd-1p1d", "model": "model",
            "prefill": {"node": "prefill0", "kv_network": "data0"},
            "decode": {"node": "decode0", "kv_network": "data0"},
            "client": "client", "control_network": "control",
            "router": {"gateway": "gateway", **router},
            "server": {"connector": "nixl", "timing": "zero"}}


def load(raw: dict) -> types.SimpleNamespace:
    """The run (``.run``) and its router (``.router``)."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        run = pd.load(path)
        return types.SimpleNamespace(run=run, router=run.router)


def placed(run: pd.Run, cpus: dict[str, str] | None = None) -> dict:
    """A run's components where they belong: each in its node's network
    namespace and on its CPU set."""
    cpus = cpus or {}
    return {name: {"cpuset": cpus.get(node, "15"), "network_mode": "container:abc", "node": node,
                   "node_cpuset": cpus.get(node, "15")}
            for name, node in llmd.expected_components(run).items()}


class PlacementTests(unittest.TestCase):
    """A component runs on its node's CPU set, not merely in its network
    namespace."""

    def setUp(self) -> None:
        self.run = load(run_config()).run

    def test_a_component_is_created_on_its_nodes_cpus(self) -> None:
        for cpus, expected in (("7-13", ["--cpuset-cpus", "7-13"]), ("", [])):
            calls: list[list[str]] = []

            def run(command, **_):
                calls.append(command)
                return cpus + "\n" if command[:2] == ["docker", "inspect"] else ""

            with self.subTest(cpus=cpus), mock.patch.object(shell, "run", side_effect=run):
                llmd.start_component("t", "t-decode0", "t-decode0-sidecar", "image", ["--port=1"])
                create = next(c for c in calls if c[:2] == ["docker", "create"])
                self.assertEqual(calls[0], ["docker", "inspect", "-f", "{{.HostConfig.CpusetCpus}}", "t-decode0"])
                self.assertIn("container:t-decode0", create)
                image = create.index("image")
                self.assertEqual([a for a in create[:image] if a in ("--cpuset-cpus", cpus) and cpus], expected)
                self.assertEqual(create[image:], ["image", "--port=1"])

    def test_the_runs_components_are_its_gateways_and_its_decode_nodes(self) -> None:
        self.assertEqual(llmd.expected_components(self.run), {
            "llmd-1p1d-gateway-epp": "llmd-1p1d-gateway", "llmd-1p1d-gateway-envoy": "llmd-1p1d-gateway",
            "llmd-1p1d-decode0-sidecar": "llmd-1p1d-decode0"})

    def test_placement_is_read_from_docker(self) -> None:
        inspected = {"c-envoy": {"HostConfig": {"CpusetCpus": "15", "NetworkMode": "container:ID"}},
                     "ID": {"Name": "/t-gateway", "HostConfig": {"CpusetCpus": "15"}}}
        with mock.patch.object(shell, "run", side_effect=lambda command, **_: json.dumps([inspected[command[2]]])):
            self.assertEqual(llmd.placement(["c-envoy"]), {"c-envoy": {
                "cpuset": "15", "network_mode": "container:ID", "node": "t-gateway", "node_cpuset": "15"}})

    def test_saved_placement_is_checked(self) -> None:
        good = placed(self.run, {"llmd-1p1d-gateway": "15", "llmd-1p1d-decode0": "7-13"})
        self.assertEqual(llmd.placement_problems(good, self.run), [])
        sidecar, envoy = "llmd-1p1d-decode0-sidecar", "llmd-1p1d-gateway-envoy"

        def changed(name, **change):
            return {**good, name: {**good[name], **change}}

        cases = {
            "unrestricted while its node is pinned": (changed(envoy, cpuset=""), "ran on CPUs ''"),
            "other cores than its node": (changed(sidecar, cpuset="0-6"), "ran on CPUs '0-6', its node"),
            "another node's namespace": (changed(sidecar, node="llmd-1p1d-prefill0"), "not its node"),
            "no node at all": (changed(envoy, node=None, node_cpuset=None), "not its node"),
            "a missing component": ({k: v for k, v in good.items() if k != sidecar}, "no placement evidence"),
            "an unknown component": ({**good, "x": good[envoy]}, "does not have"),
            "a cpuset that is no string": (changed(envoy, cpuset=None, node_cpuset=None), "ran on CPUs None"),
            "no document": (None, "placement evidence is missing"),
            "a list": ([], "placement evidence is missing"),
        }
        for name, (saved, message) in cases.items():
            with self.subTest(name):
                self.assertRegex("; ".join(llmd.placement_problems(saved, self.run)), message)
        # Unrestricted nodes: their components are unrestricted too.
        self.assertEqual(llmd.placement_problems(placed(self.run, {"llmd-1p1d-gateway": "", "llmd-1p1d-decode0": ""}),
                                                 self.run), [])


class RankRoutingTests(unittest.TestCase):
    """``router.dp_routing: rank``: each DP rank of an instance is an
    endpoint of its own, the EPP picking it (llm-d's per-rank routing)."""

    ADDRESS = {"instances": {"prefill": {"prefill0": {"control": "10.0.0.1"}},
                             "decode": {"decode0": {"control": "10.0.0.3"}}}}

    def run_config(self, prefill_dp=4, decode_dp=2, **router) -> pd.Run:
        raw = run_config(dp_routing="rank", **router)
        raw["prefill"]["data_parallel_size"] = prefill_dp
        raw["decode"]["data_parallel_size"] = decode_dp
        return load(raw).run

    def test_rank_routing_needs_ranks_and_is_one_of_two(self) -> None:
        self.assertEqual(self.run_config().router.dp_routing, "rank")
        self.assertEqual(load(run_config()).run.router.dp_routing, "instance")
        with self.assertRaisesRegex(ValueError, "needs a side with data_parallel_size > 1"):
            self.run_config(1, 1)
        with self.assertRaisesRegex(ValueError, "one of instance, rank"):
            load(run_config(dp_routing="pod"))
        raw = run_config(dp_routing="rank")
        raw["decode"]["data_parallel_size"] = 2
        raw["server"]["connector"] = "mooncake"
        with self.assertRaisesRegex(ValueError, "needs server.connector: nixl"):
            load(raw)

    def test_a_ranks_server_and_sidecar_listener_are_its_endpoint(self) -> None:
        run = self.run_config()
        found = llmd.endpoints(run, self.ADDRESS)["endpoints"]
        self.assertEqual([(e["name"], e["address"], e["port"], e["rankIndex"], e["labels"][llmd.ROLE_LABEL])
                          for e in found],
                         [(f"prefill0-dp{r}", "10.0.0.1", str(8000 + r), r, "prefill") for r in range(4)]
                         + [(f"decode0-dp{r}", "10.0.0.3", str(8000 + r), r, "decode") for r in range(2)])
        # A side of one rank stays the instance's one endpoint.
        mixed = llmd.endpoints(self.run_config(1, 2), self.ADDRESS)["endpoints"]
        self.assertEqual([(e["name"], e["port"], "rankIndex" in e) for e in mixed],
                         [("prefill0", "8000", False), ("decode0-dp0", "8000", True), ("decode0-dp1", "8001", True)])
        # Instance routing: one endpoint an instance, whatever its ranks.
        raw = run_config()
        raw["prefill"]["data_parallel_size"] = 4
        self.assertEqual([e["name"] for e in llmd.endpoints(load(raw).run, self.ADDRESS)["endpoints"]],
                         ["prefill0", "decode0"])

    def test_servers_run_an_api_server_a_rank(self) -> None:
        run = self.run_config()
        for role, port, ranks in (("prefill", 8000, 4), ("decode", 8200, 2)):
            command = pd.server_command(run, role, ["10.1.0.1"], port, ("10.0.0.1", 8000))
            at = command.index("--data-parallel-multi-port-external-lb")
            self.assertEqual(command[at + 1:at + 5], ["--data-parallel-size-local", str(ranks),
                                                      "--data-parallel-supervisor-port", str(pd.DP_SUPERVISOR_PORT)])
            self.assertEqual(command[command.index("--data-parallel-size") + 1], str(ranks))
            self.assertEqual(command[command.index("--port") + 1], str(port))
        # A side of one rank, and instance routing, keep the one server.
        self.assertNotIn("--data-parallel-multi-port-external-lb",
                         pd.server_command(self.run_config(1, 2), "prefill", ["10.1.0.1"]))
        raw = run_config()
        raw["decode"]["data_parallel_size"] = 2
        self.assertNotIn("--data-parallel-multi-port-external-lb",
                         pd.server_command(load(raw).run, "decode", ["10.1.0.1"]))

    def test_the_sidecar_listens_for_each_decode_rank(self) -> None:
        self.assertIn("--data-parallel-size=2", llmd.sidecar_args(self.run_config()))
        self.assertFalse(any("data-parallel" in a for a in llmd.sidecar_args(self.run_config(4, 1))))
        # It names its listeners by its pod address.
        calls: list[list[str]] = []
        with mock.patch.object(shell, "run", side_effect=lambda command, **_: calls.append(command) or ""):
            llmd.start_component("t", "t-decode0", "t-decode0-sidecar", "image", ["--port=8000"],
                                 env={"POD_IP": "10.0.0.3"})
        create = next(c for c in calls if c[:2] == ["docker", "create"])
        at = create.index("--env")
        self.assertEqual((create[at + 1], create.index("image") > at), ("POD_IP=10.0.0.3", True))

    def test_a_rank_routed_instance_is_up_when_all_its_ranks_are(self) -> None:
        front = llmd.Front(self.run_config(4, 1))
        self.assertEqual(front.health_url("prefill", "10.0.0.1"), f"http://10.0.0.1:{pd.DP_SUPERVISOR_PORT}/health")
        self.assertEqual(front.health_url("decode", "10.0.0.3"), f"http://10.0.0.3:{llmd.DECODE_ENGINE_PORT}/health")

    def test_each_ranks_step_logs_are_its_endpoints(self) -> None:
        run = self.run_config(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for role, side in (("prefill", run.prefill), ("decode", run.decode)):
                for dp, tp in side.ranks(side.tp(run.server)):
                    (output / side.step_log(role, dp, tp)).write_text(f"{role}{dp}")
            found = llmd.instance_steps(output, run)
        self.assertEqual(found, {"prefill0-dp0": ("prefill", ["prefill0"]), "prefill0-dp1": ("prefill", ["prefill1"]),
                                 "decode0-dp0": ("decode", ["decode0"]), "decode0-dp1": ("decode", ["decode1"])})


class RankRoutingEvidenceTests(unittest.TestCase):
    """The routing evidence per rank: the rank the EPP picked ran the
    request."""

    def setUp(self) -> None:
        raw = run_config(dp_routing="rank")
        raw["prefill"]["data_parallel_size"] = 2
        raw["decode"]["data_parallel_size"] = 2
        self.run_ = load(raw).run
        self.output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shell.run(["rm", "-rf", str(self.output)]))

    def write(self, ran: dict[str, tuple[int, int]] | None = None) -> None:
        """Requests a (prefill rank 0 -> decode rank 1) and b (1 -> 0), as
        picked; ``ran``: the ranks they ran on (default: as picked)."""
        run = self.run_
        picked = {"a": (0, 1), "b": (1, 0)}
        ran = ran or picked
        directory = self.output / "llmd"
        directory.mkdir(exist_ok=True)
        address = {"instances": {"prefill": {"prefill0": {"control": "10.0.0.1"}},
                                 "decode": {"decode0": {"control": "10.0.0.3"}}}}
        (directory / "endpoints.yaml").write_text(yaml.safe_dump(llmd.endpoints(run, address)))
        (directory / "epp-config.yaml").write_text(yaml.safe_dump(llmd.epp_config(run.router)))
        (directory / "cleanup.json").write_text(json.dumps({"removed": ["x"], "remaining": []}))
        (directory / "components.json").write_text(json.dumps(placed(run)))
        epp, sidecar, envoy = [], [], []
        for request, (prefill, decode) in picked.items():
            at_prefill, at_decode = f"10.0.0.1:{8000 + prefill}", f"10.0.0.3:{8000 + decode}"
            epp += [picker(request, at_decode), picker(request, at_prefill),
                    {"body": "Request handled", "x-request-id": request, "trace_id": f"t{request}",
                     "incomingModelName": pd.MODEL_NAME, "endpoint": at_decode}]
            sidecar += [{"body": "SSRF protection: prefill target allowed", "trace_id": f"t{request}",
                         "target": at_prefill}, {"body": "running NIXL protocol V2"}]
            envoy.append({"access": "request", "request_id": request, "method": "POST", "path": "/v1/completions",
                          "status": "200", "upstream": at_decode})
        epp.append({"body": "Selecting endpoints from candidates sorted by max score", "scored-endpoints": [
            {"Endpoint": {"Name": e["name"], "MetricsHost": "m", "CacheNumBlocks": 100}}
            for e in llmd.endpoints(run, address)["endpoints"]]})
        for name, records in (("x-gateway-epp.log", epp), ("x-decode0-sidecar.log", sidecar),
                              ("x-gateway-envoy.log", envoy)):
            (directory / name).write_text("".join(json.dumps(r) + "\n" for r in records))
        (self.output / "bench.json").write_text(json.dumps({"completed": 2, "errors": []}))
        logs: dict[tuple[str, int], list[str]] = {(role, dp): [] for role in ("prefill", "decode") for dp in (0, 1)}
        for request, (prefill, decode) in ran.items():
            engine = f"cmpl-{request}-0-abcd1234"
            logs[("prefill", prefill)].append(json.dumps({"kv_events": {
                "admitted": [engine], "allocated": {engine: [1]}, "sends_done": [engine]}}))
            logs[("decode", decode)].append(json.dumps({"kv_events": {
                "admitted": [engine], "reads_posted": [engine], "reads_done": [engine]}}))
        for (role, dp), lines in logs.items():
            (self.output / getattr(run, role).step_log(role, dp, 0)).write_text("\n".join(lines) + "\n")

    def test_requests_on_the_ranks_the_epp_picked_pass(self) -> None:
        self.write()
        result = llmd.routing_evidence(self.output, self.run_)
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["evidence"]["routed"], {"prefill": {"prefill0-dp0": 1, "prefill0-dp1": 1},
                                                        "decode": {"decode0-dp1": 1, "decode0-dp0": 1}})
        self.assertEqual(result["evidence"]["pairs"], {"prefill0-dp0->decode0-dp1": 1, "prefill0-dp1->decode0-dp0": 1})

    def test_a_request_on_another_rank_than_picked_fails(self) -> None:
        self.write(ran={"a": (1, 1), "b": (1, 0)})
        problems = llmd.routing_evidence(self.output, self.run_)["problems"]
        self.assertTrue(any("a: the EPP picked prefill prefill0-dp0, it ran on ['prefill0-dp1']" in p for p in problems),
                        problems)
        self.write(ran={"a": (0, 0), "b": (1, 0)})
        problems = llmd.routing_evidence(self.output, self.run_)["problems"]
        self.assertTrue(any("a: the EPP picked decode decode0-dp1, it ran on ['decode0-dp0']" in p for p in problems),
                        problems)

    def test_endpoints_that_are_not_the_runs_fail(self) -> None:
        self.write()
        saved = self.output / "llmd" / "endpoints.yaml"
        found = yaml.safe_load(saved.read_text())
        for entry in found["endpoints"]:
            entry.pop("rankIndex")
        saved.write_text(yaml.safe_dump(found))
        problems = llmd.routing_evidence(self.output, self.run_)["problems"]
        self.assertTrue(any("the saved endpoints are not the run's" in p for p in problems), problems)


class PreciseIndexEvidenceTests(unittest.TestCase):
    """llm-d's precise prefix index against the engines it indexes."""

    NAMED = {"10.0.0.1:8000": {"name": "prefill0", "address": "10.0.0.1", "port": "8000"},
             "10.0.0.2:8000": {"name": "prefill1", "address": "10.0.0.2", "port": "8000"},
             "10.0.0.3:8000": {"name": "decode0", "address": "10.0.0.3", "port": "8000"}}

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shell.run(["rm", "-rf", str(self.directory)]))

    def evidence(self, *, subscribers=None, produced=None, metrics=None, steps=None, picks=None, block=16,
                 failures=0, reuse=True) -> dict:
        subscribers = {"prefill0": "10.0.0.1", "prefill1": "10.0.0.2", "decode0": "10.0.0.3"} if (
            subscribers is None) else subscribers
        # r1: nothing cached anywhere (cold); r2: its 20-block prefix on prefill0, where it goes (warm).
        produced = [("r1", 28, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}})] if produced is None else produced
        records = [{"body": "Ensured KV-events subscriber", "endpoint": f"default/{name}",
                    "zmq": f"tcp://{address}:5557"} for name, address in subscribers.items()]
        records += [{"body": "Failed to connect subscriber socket"}] * failures
        records += [{"body": "Produce completed", "x-request-id": rid, "blockKeys": keys, "matches": matches}
                    for rid, keys, matches in produced]
        (self.directory / "x-gateway-epp.log").write_text("".join(json.dumps(r) + "\n" for r in records))
        metrics = {"admissions": 56, "evictions": 0, "lookup_requests": 2} if metrics is None else metrics
        (self.directory / "epp-metrics.txt").write_text(
            "# TYPE llm_d_epp_kv_cache_index_admissions_total counter\n"
            + "".join(f"llm_d_epp_kv_cache_index_{name}_total {count}\n" for name, count in metrics.items())
            + 'llm_d_epp_kv_cache_index_lookup_latency_seconds_bucket{le="1"} 2\n')
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 320)], "prefill1": []} if steps is None else steps
        texts = {name: ["".join(json.dumps({"step": {"requests": [{
            "req_id": f"cmpl-{rid}-0-abcd1234", "num_prompt_tokens": prompt, "num_cached_tokens": cached}]}}) + "\n"
            for rid, prompt, cached in requests)] for name, requests in steps.items()}
        picks = {rid: "10.0.0.1:8000" for rid, _, _ in produced} if picks is None else picks
        return llmd.precise_index_evidence(self.directory, self.NAMED, set(picks), picks, texts, block, reuse)

    def assertProblem(self, evidence: dict, message: str) -> None:
        self.assertTrue(any(message in p for p in evidence["problems"]), evidence["problems"])

    def test_an_index_that_agrees_with_the_engines_passes(self) -> None:
        evidence = self.evidence()
        self.assertEqual(evidence["problems"], [])
        self.assertEqual((evidence["cold_exact"], evidence["warm"]), (1, {"exact": 1, "under": 0, "over": 0}))
        self.assertEqual((evidence["lookups_compared"], evidence["block_tokens"]), (2, 16))
        self.assertEqual((evidence["warm_validation"], evidence["unresolved_over"]), ("exercised", 0))
        self.assertEqual(evidence["index_blocks_at_pick"], {"total": 20, "max": 20})
        self.assertEqual(evidence["index"], {"admissions": 56, "evictions": 0, "lookup_requests": 2})
        self.assertEqual(evidence["subscribers"]["decode0"], "tcp://10.0.0.3:5557")

    def test_every_endpoint_is_subscribed_at_its_control_address(self) -> None:
        self.assertProblem(self.evidence(subscribers={"prefill0": "10.0.0.1", "decode0": "10.0.0.3"}),
                           "did not subscribe to prefill1's KV-cache events at tcp://10.0.0.2:5557")
        self.assertProblem(self.evidence(subscribers={"prefill0": "10.0.0.1", "prefill1": "10.9.9.9",
                                                      "decode0": "10.0.0.3"}), "did not subscribe to prefill1")
        self.assertEqual(self.evidence(failures=3)["subscriber_connect_failures"], 3)

    def test_an_index_no_event_reached_fails(self) -> None:
        self.assertProblem(self.evidence(metrics={"admissions": 0}), "admitted no block")
        self.assertProblem(self.evidence(metrics={}), "admitted no block")

    def test_the_lookups_block_count_matches_the_engines_whole_blocks(self) -> None:
        produced = [("r1", 27, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}})]
        self.assertProblem(self.evidence(produced=produced),
                           "r1: the EPP's lookup has 27 block key(s) of 16 tokens, the engine's prompt 448 tokens "
                           "(28 whole blocks)")
        # Only whole blocks are compared: tokens within a partial block are not.
        steps = {"prefill0": [("r1", 455, 0), ("r2", 455, 320)], "prefill1": []}
        self.assertEqual(self.evidence(steps=steps)["problems"], [])
        for keys in (None, -1, True, "28"):
            with self.subTest(keys=keys):
                self.assertProblem(self.evidence(produced=[("r1", 28, {}), ("r2", keys, {})]), "block key(s)")

    def test_every_completion_has_one_lookup_on_its_prefill_instance(self) -> None:
        picks = {"r1": "10.0.0.1:8000", "r2": "10.0.0.1:8000"}
        self.assertProblem(self.evidence(produced=[("r1", 28, {})], picks=picks),
                           "r2: the precise index logged 0 lookup(s)")
        twice = [("r1", 28, {}), ("r1", 28, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}})]
        self.assertProblem(self.evidence(produced=twice, picks=picks), "r1: the precise index logged 2 lookup(s)")
        self.assertProblem(self.evidence(picks={"r1": "10.0.0.1:8000", "r2": "10.0.0.2:8000"}),
                           "r2: no step of it on the prefill instance the EPP picked")

    def test_a_fully_cached_prompt_reuses_all_but_its_last_block(self) -> None:
        # The index holds all 28 blocks; vLLM recomputes the last token, in whole blocks: 432 tokens reused.
        produced = [("r1", 28, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 28}})]
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 432)], "prefill1": []}
        evidence = self.evidence(produced=produced, steps=steps)
        self.assertEqual((evidence["problems"], evidence["warm"]["exact"]), ([], 1))
        self.assertEqual(evidence["index_blocks_at_pick"]["max"], 28)
        # A prompt one token past a block boundary keeps all its full blocks.
        steps = {"prefill0": [("r1", 449, 0), ("r2", 449, 448)], "prefill1": []}
        self.assertEqual(self.evidence(produced=produced, steps=steps)["warm"]["exact"], 1)
        self.assertEqual([llmd.reusable_tokens(b, p, 16) for b, p in ((28, 448), (28, 449), (20, 448), (0, 448),
                                                                      (1, 1), (1, 17))], [432, 448, 320, 0, 0, 16])

    def test_cold_agreement_does_not_hide_wrong_warm_lookups(self) -> None:
        # Every lookup's matches emptied (the mutation of response 201): the cold one agrees, no warm one does.
        empty = [("r1", 28, {}), ("r2", 28, {}), ("r3", 28, {}), ("r4", 28, {})]
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 320), ("r3", 448, 320), ("r4", 448, 320)], "prefill1": []}
        evidence = self.evidence(produced=empty, steps=steps)
        self.assertEqual((evidence["cold_exact"], evidence["warm"]), (1, {"exact": 0, "under": 3, "over": 0}))
        self.assertProblem(evidence, "agreed with the engine on 0 of 3 warm lookup(s), not most")
        # A reuse workload where nothing was ever cached: warm agreement is unexercised, a problem.
        cold = [("r1", 28, {}), ("r2", 28, {})]
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 0)], "prefill1": []}
        self.assertProblem(self.evidence(produced=cold, steps=steps), "warm agreement was not exercised")
        # Not a reuse workload: reported, not a problem.
        unexercised = self.evidence(produced=cold, steps=steps, reuse=False)
        self.assertEqual((unexercised["problems"], unexercised["warm_validation"]), ([], "unexercised"))

    def test_a_stale_index_is_a_problem_unless_it_saw_evictions(self) -> None:
        # The index held r2's prefix; the engine no longer had it.
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 0), ("r3", 448, 320), ("r4", 448, 320)], "prefill1": []}
        produced = [("r1", 28, {})] + [(rid, 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}}) for rid in ("r2", "r3", "r4")]
        stale = self.evidence(steps=steps, produced=produced)
        self.assertEqual(stale["warm"], {"exact": 2, "under": 0, "over": 1})
        self.assertProblem(stale, "held more than the engine reused for 1 request(s)")
        # With evictions in the index it is reported as unresolved, not excused as explained.
        evicting = self.evidence(steps=steps, produced=produced, metrics={"admissions": 56, "evictions": 20})
        self.assertEqual((evicting["problems"], evicting["unresolved_over"]), ([], 1))
        # The engine had more than the index knew (an event not yet indexed): reported.
        held = {"10.0.0.1:8000": {"MatchedBlocks": 20}}
        behind = self.evidence(produced=[("r1", 28, {}), ("r2", 28, {}), ("r3", 28, held), ("r4", 28, held)],
                               steps={"prefill0": [("r1", 448, 0), ("r2", 448, 320), ("r3", 448, 320),
                                                   ("r4", 448, 320)], "prefill1": []})
        self.assertEqual((behind["problems"], behind["warm"]), ([], {"exact": 2, "under": 1, "over": 0}))
        # Another endpoint's match is not the picked one's.
        elsewhere = [("r1", 28, {}), ("r2", 28, {"10.0.0.2:8000": {"MatchedBlocks": 20}}),
                     ("r3", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}}), ("r4", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}})]
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 320), ("r3", 448, 320), ("r4", 448, 320)], "prefill1": []}
        self.assertEqual(self.evidence(produced=elsewhere, steps=steps)["warm"]["under"], 1)

    def test_impossible_matches_fail_even_with_evictions(self) -> None:
        evicting = {"admissions": 56, "evictions": 20}
        for value in (1_000_000, 29, -1, True, 2.5, "20", None):
            with self.subTest(value=value):
                produced = [("r1", 28, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": value}})]
                self.assertProblem(self.evidence(produced=produced, metrics=evicting), "r2: the EPP's lookup matched")
        # Another endpoint's impossible count is as wrong as the picked one's.
        produced = [("r1", 28, {"10.0.0.2:8000": {"MatchedBlocks": 1_000_000}}),
                    ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 20}})]
        self.assertProblem(self.evidence(produced=produced, metrics=evicting), "r1: the EPP's lookup matched")
        for matches in (None, [], {"10.0.0.1:8000": 20}):
            with self.subTest(matches=matches):
                problems = self.evidence(produced=[("r1", 28, {}), ("r2", 28, matches)])["problems"]
                self.assertTrue(any("r2: the EPP's lookup" in p for p in problems), problems)
        # The full block count is possible.
        produced = [("r1", 28, {}), ("r2", 28, {"10.0.0.1:8000": {"MatchedBlocks": 28}})]
        steps = {"prefill0": [("r1", 448, 0), ("r2", 448, 432)], "prefill1": []}
        self.assertEqual(self.evidence(produced=produced, steps=steps)["problems"], [])

    def test_the_index_needs_a_block_size(self) -> None:
        for block in (None, 0, True, "16", -16):
            self.assertProblem(self.evidence(block=block), "block size")


class LoadTests(unittest.TestCase):
    def test_a_router_gateway_is_required(self) -> None:
        raw = run_config()
        del raw["router"]
        with self.assertRaisesRegex(ValueError, "router.gateway"):
            load(raw)

    def test_unknown_router_keys_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "router"):
            load(run_config(scheduler="x"))

    def test_the_gateway_is_not_a_server(self) -> None:
        raw = run_config()
        raw["router"]["gateway"] = "decode0"
        with self.assertRaisesRegex(ValueError, "gateway"):
            load(raw)

    def test_the_gateway_is_not_the_client(self) -> None:
        raw = run_config()
        raw["router"]["gateway"] = "client"
        with self.assertRaisesRegex(ValueError, "a node of its own"):
            load(raw)

    def test_an_instance_may_have_dp_ranks_unequal_tp_or_speculation(self) -> None:
        # One endpoint each: the instance's vLLM balances its DP ranks.
        raw = run_config()
        raw["decode"]["data_parallel_size"] = 2
        self.assertEqual(load(raw).run.decode.data_parallel_size, 2)
        raw = run_config()
        raw["decode"]["tensor_parallel_size"] = 2
        self.assertEqual(load(raw).run.decode.tp(load(raw).run.server), 2)
        raw = run_config()
        raw["server"]["speculative"] = {"method": "mtp", "num_speculative_tokens": 1, "acceptance_rates": [1.0]}
        self.assertEqual(load(raw).run.server.speculative["method"], "mtp")
        # Several instances of a side stay one engine each.
        raw = run_config()
        raw["decode"] = {"node": ["decode0", "decode1"], "kv_network": "data0", "data_parallel_size": 2}
        with self.assertRaisesRegex(ValueError, "data_parallel_size 1"):
            load(raw)

    def test_an_instance_of_dp_ranks_is_one_endpoint(self) -> None:
        raw = run_config()
        raw["prefill"]["data_parallel_size"] = 4
        raw["decode"]["data_parallel_size"] = 8
        run = load(raw).run
        address = {"instances": {"prefill": {"prefill0": {"control": "10.0.0.1"}},
                                 "decode": {"decode0": {"control": "10.0.0.3"}}}}
        self.assertEqual([e["name"] for e in llmd.endpoints(run, address)["endpoints"]], ["prefill0", "decode0"])
        # The sidecar's own DP mode (an API server per rank) stays off.
        self.assertFalse(any("data-parallel" in a or "dp-size" in a for a in llmd.sidecar_args(run)))

    def test_pd_keys_still_validated(self) -> None:
        raw = run_config()
        raw["server"]["connector"] = "lmcache"
        with self.assertRaisesRegex(ValueError, "connector"):
            load(raw)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llmd = load(run_config())
        self.address = {"instances": {"prefill": {"prefill0": {"control": "10.0.0.1"}},
                                     "decode": {"decode0": {"control": "10.0.0.3"}}}}

    def test_images_are_pinned_by_digest(self) -> None:
        for name, image in llmd.IMAGES.items():
            with self.subTest(name=name):
                self.assertRegex(image, r":[^@:/]+@sha256:[0-9a-f]{64}$")

    def test_epp_config_adds_file_discovery_and_metrics(self) -> None:
        config = llmd.epp_config(self.llmd.router)
        self.assertEqual(config["apiVersion"], "llm-d.ai/v1")
        types = [p["type"] for p in config["plugins"]]
        self.assertEqual(types[:3], ["file-discovery", "metrics-data-source", "core-metrics-extractor"])
        self.assertIn("always-disagg-pd-decider", types)
        self.assertEqual(config["dataLayer"]["discovery"]["endpoints"]["pluginRef"], "file-discovery")
        self.assertEqual({p["name"] for p in config["schedulingProfiles"]}, {"prefill", "decode"})

    def test_a_calibrated_peak_prefill_throughput_replaces_the_guides(self) -> None:
        def throughput(config):
            return next(p["parameters"]["peakPrefillThroughput"] for p in config["plugins"]
                        if p["type"] == "prefix-cache-affinity-filter")
        self.assertEqual(throughput(llmd.epp_config(self.llmd.router)), llmd.UNCALIBRATED_PEAK_PREFILL_THROUGHPUT)
        self.llmd.router.peak_prefill_throughput = 46002.8
        self.assertEqual(throughput(llmd.epp_config(self.llmd.router)), 46002.8)

    def test_a_precise_index_is_fed_by_the_endpoints_events(self) -> None:
        self.llmd.router.prefix_index = "precise"
        self.llmd.router.peak_prefill_throughput = 46002.8
        config = llmd.epp_config(self.llmd.router, 32)
        plugins = {p["type"]: p.get("parameters") or {} for p in config["plugins"]}
        self.assertNotIn("approx-prefix-cache-producer", plugins)
        producer = plugins["precise-prefix-cache-producer"]
        self.assertEqual(producer["tokenProcessorConfig"], {"blockSizeTokens": 32})
        self.assertEqual(producer["kvEventsConfig"]["podDiscoveryConfig"],
                         {"socketPort": pd.KV_EVENTS_PORT, "replaySocketPort": pd.KV_EVENTS_REPLAY_PORT})
        self.assertTrue(producer["kvEventsConfig"]["discoverPods"])
        # The prompt tokenized by the servers themselves: prefill's port, decode's vLLM behind its sidecar.
        token = plugins["token-producer"]
        self.assertEqual(token["modelName"], pd.MODEL_NAME)
        self.assertEqual([(r["selector"]["matchLabels"][llmd.ROLE_LABEL], r["basePort"])
                          for r in token["vllm"]["endpointDiscovery"]["portRules"]],
                         [("prefill", pd.SERVER_PORT), ("decode", llmd.DECODE_ENGINE_PORT)])
        # The load and the affinity filter read the precise producer, with the run's throughput.
        named = {"prefixMatchInfoProducerName": "precise-prefix-cache-producer"}
        self.assertEqual(plugins["inflight-load-producer"], named)
        self.assertEqual(plugins["prefix-cache-affinity-filter"], {**named, "peakPrefillThroughput": 46002.8})
        # Endpoint notifications reach the producer (its subscribers), beside the metrics.
        self.assertEqual(config["dataLayer"]["sources"], [
            {"pluginRef": "metrics-source", "extractors": [{"pluginRef": "metrics-extractor"}]},
            {"pluginRef": "endpoint-notification-source",
             "extractors": [{"pluginRef": "precise-prefix-cache-producer"}]}])
        # The profiles are the approximate configuration's.
        self.llmd.router.prefix_index = "approximate"
        approximate = llmd.epp_config(self.llmd.router)
        self.assertEqual(config["schedulingProfiles"], approximate["schedulingProfiles"])
        self.assertEqual(len(approximate["dataLayer"]["sources"]), 1)

    def test_custom_plugins_take_plugins_and_profiles_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugins.yaml"
            path.write_text(yaml.safe_dump({"plugins": [], "schedulingProfiles": [], "dataLayer": {}}))
            self.llmd.router.epp_plugins = path
            with self.assertRaisesRegex(ValueError, "plugins and schedulingProfiles"):
                llmd.epp_config(self.llmd.router)

    def test_endpoints_name_prefill_vllm_and_decode_sidecar(self) -> None:
        found = {e["name"]: e for e in llmd.endpoints(self.llmd.run, self.address)["endpoints"]}
        self.assertEqual(found["prefill0"]["port"], str(pd.SERVER_PORT))
        self.assertEqual(found["decode0"]["port"], str(llmd.SIDECAR_PORT))
        self.assertEqual(found["prefill0"]["labels"][llmd.ROLE_LABEL], "prefill")
        self.assertEqual(found["decode0"]["labels"][llmd.ROLE_LABEL], "decode")
        self.assertEqual(found["decode0"]["address"], "10.0.0.3")

    def test_decode_vllm_is_behind_the_sidecar(self) -> None:
        front = llmd.Front(self.llmd.run)
        self.assertEqual(front.server_port("decode"), llmd.DECODE_ENGINE_PORT)
        self.assertEqual(front.server_port("prefill"), pd.SERVER_PORT)
        self.assertIn(f"--port {llmd.DECODE_ENGINE_PORT}",
                      " ".join(pd.server_command(self.llmd.run, "decode", ["10.1.0.1"], llmd.DECODE_ENGINE_PORT)))

    def test_sidecar_speaks_the_connector_in_plaintext(self) -> None:
        for connector, protocol in (("nixl", "nixlv2"), ("mooncake", "mooncake")):
            self.llmd.run.server.connector = connector
            args = llmd.sidecar_args(self.llmd.run)
            with self.subTest(connector=connector):
                self.assertIn(f"--kv-connector={protocol}", args)
                self.assertIn("--secure-serving=false", args)
                self.assertIn("--enable-ssrf-protection=false", args)
                self.assertIn(f"--mooncake-bootstrap-port={pd.BOOTSTRAP_PORT}", args)
                self.assertIn(f"--model-server-port={llmd.DECODE_ENGINE_PORT}", args)

    def test_epp_runs_plaintext_with_grpc_health(self) -> None:
        args = llmd.epp_args(self.llmd.router)
        self.assertIn("--secure-serving=false", args)
        self.assertIn("--health-checking=true", args)
        self.assertIn(f"--grpc-port={llmd.EPP_GRPC_PORT}", args)

    def test_envoy_listens_on_the_gateway_port_and_logs_each_request(self) -> None:
        config = llmd.envoy_config()
        listener = config["static_resources"]["listeners"][0]
        self.assertEqual(listener["address"]["socket_address"]["port_value"], llmd.GATEWAY_PORT)
        manager = listener["filter_chains"][0]["filters"][0]["typed_config"]
        fields = manager["access_log"][0]["typed_config"]["log_format"]["json_format"]
        self.assertEqual(fields["request_id"], "%REQ(X-REQUEST-ID)%")
        self.assertEqual(fields["upstream"], "%UPSTREAM_HOST%")


PREFILL, DECODE, GATEWAY = "10.0.0.1:8000", "10.0.0.3:8000", "10.0.0.5"


def picker(request: str, target: str) -> dict:
    address, port = target.split(":")
    return {"body": llmd.PICKED, "x-request-id": request,
            "result": {"TargetEndpoints": [{"Endpoint": {"Address": address, "Port": port}, "Score": 1}]}}


class EvidenceTests(unittest.TestCase):
    """`routing_evidence` on a synthetic run of two requests, then each way
    it can go wrong."""

    REQUESTS = ("aaaa-1111", "bbbb-2222")

    def setUp(self) -> None:
        self.llmd = load(run_config())
        self.output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shell.run(["rm", "-rf", str(self.output)]))
        self.write()

    def write(self, *, picked_prefill=PREFILL, picked_decode=DECODE, handled_at=None, dispatch=PREFILL,
              upstream=None, protocol="running NIXL protocol V2",
              decode_events=("admitted", "reads_posted", "reads_done"),
              prefill_events=("admitted", "allocated", "sends_done"), extra_engine=(), scored=True,
              completed=2, extra_epp=(), extra_sidecar=(), extra_envoy=(), decode_suffixes=("abcd1234",),
              ranks=1) -> None:
        directory = self.output / "llmd"
        directory.mkdir(exist_ok=True)
        (directory / "endpoints.yaml").write_text(yaml.safe_dump(llmd.endpoints(
            self.llmd.run, {"instances": {"prefill": {"prefill0": {"control": "10.0.0.1"}},
                                          "decode": {"decode0": {"control": "10.0.0.3"}}}})))
        (directory / "epp-config.yaml").write_text(yaml.safe_dump(llmd.epp_config(self.llmd.router)))
        (directory / "cleanup.json").write_text(json.dumps({"removed": ["x"], "remaining": []}))
        (directory / "components.json").write_text(json.dumps(placed(self.llmd.run)))
        epp, sidecar, envoy = [], [], []
        for index, request in enumerate(self.REQUESTS):
            trace = f"trace{index}"
            # As the EPP logs them: the decode profile's pick, then prefill's.
            epp.append(picker(request, picked_decode))
            epp.append(picker(request, picked_prefill))
            epp.append({"body": "Request handled", "x-request-id": request, "trace_id": trace,
                        "incomingModelName": pd.MODEL_NAME, "endpoint": handled_at or picked_decode})
            sidecar.append({"body": "SSRF protection: prefill target allowed", "trace_id": trace, "target": dispatch})
            sidecar.append({"body": protocol})
            envoy.append({"access": "request", "request_id": request, "method": "POST", "path": "/v1/completions",
                          "status": "200", "upstream": upstream or picked_decode})
        epp.append(picker("tok", DECODE))
        epp.append({"body": "Request handled", "x-request-id": "tok", "incomingModelName": "", "endpoint": DECODE})
        envoy.append({"access": "request", "request_id": "tok", "method": "POST", "path": "/tokenize",
                      "status": "200", "upstream": DECODE})
        if scored:
            epp.append({"body": "Selecting endpoints from candidates sorted by max score", "scored-endpoints": [
                {"Endpoint": {"Name": name, "MetricsHost": host, "CacheNumBlocks": 910, "RunningRequestsSize": 1}}
                for name, host in (("prefill0", PREFILL), ("decode0", DECODE))]})
        for name, records in (("x-gateway-epp.log", [*epp, *extra_epp]),
                              ("x-decode0-sidecar.log", [*sidecar, *extra_sidecar]),
                              ("x-gateway-envoy.log", [*extra_envoy, *envoy])):
            (directory / name).write_text("startup text\n" + "\n".join(json.dumps(r) for r in records) + "\n")

        def steps(events, extra=(), suffixes=("abcd1234",)):
            records = [{"kv_events": {kind: [f"cmpl-{r}-0-{suffix}" for r in (*self.REQUESTS, *extra)]
                                      for kind in events}} for suffix in suffixes]
            return "".join(json.dumps(r) + "\n" for r in records)

        for rank in range(ranks):
            suffix = f".tp{rank}" if rank else ""
            (self.output / f"prefill-prefill0-steps{suffix}.jsonl").write_text(steps(prefill_events, extra_engine))
            (self.output / f"decode-decode0-steps{suffix}.jsonl").write_text(
                steps(decode_events, extra_engine, decode_suffixes))
        (self.output / "bench.json").write_text(json.dumps({"completed": completed}))

    def problems(self) -> list[str]:
        return llmd.routing_evidence(self.output, self.llmd.run)["problems"]

    def assertProblem(self, text: str) -> None:
        problems = self.problems()
        self.assertTrue(any(text in p for p in problems), f"{text!r} not in {problems}")

    def test_a_correct_run_passes(self) -> None:
        result = llmd.routing_evidence(self.output, self.llmd.run)
        self.assertEqual(result["problems"], [])
        evidence = result["evidence"]
        self.assertEqual(evidence["completion_requests"], 2)
        self.assertEqual(evidence["other_requests"], 1)
        self.assertEqual(evidence["routed"], {"prefill": {"prefill0": 2}, "decode": {"decode0": 2}})
        self.assertEqual(evidence["gateway"]["others"], ["POST /tokenize 200"])
        self.assertEqual((evidence["producer_released"], evidence["consumer_received"]), (2, 2))
        self.assertIn("prefix-cache-affinity-filter.peakPrefillThroughput", evidence["uncalibrated"])

    def test_missing_routing_evidence_fails(self) -> None:
        shell.run(["rm", "-rf", str(self.output / "llmd")])
        self.assertProblem("missing routing evidence")

    def test_a_picked_prefill_the_sidecar_did_not_use_fails(self) -> None:
        # The reviewer's mutation: the picker's prefill address changed.
        self.write(picked_prefill="10.0.0.99:8000")
        self.assertProblem("picked 0 prefill endpoint(s)")

    def test_a_sidecar_dispatch_the_epp_did_not_pick_fails(self) -> None:
        self.write(dispatch=DECODE)
        self.assertProblem("the sidecar sent its prefill to 10.0.0.3:8000, the EPP picked 10.0.0.1:8000")

    def test_a_decode_pick_that_is_prefill_fails(self) -> None:
        self.write(picked_decode=PREFILL)
        self.assertProblem("picked 0 decode endpoint(s)")

    def test_the_handled_endpoint_must_be_the_picked_one(self) -> None:
        self.write(handled_at="10.0.0.9:8000")
        self.assertProblem("its picker chose")

    def test_envoy_disagreeing_with_the_epp_fails(self) -> None:
        self.write(upstream="10.0.0.9:8000")
        self.assertProblem("Envoy sent it to")

    def test_a_conflicting_sidecar_target_for_one_trace_fails(self) -> None:
        self.write(extra_sidecar=[{"body": "SSRF protection: prefill target allowed", "trace_id": "trace0",
                                   "target": "10.0.0.99:8000"}])
        self.assertProblem("dispatched its prefill 2 times")

    def test_an_earlier_failed_completion_with_the_same_id_fails(self) -> None:
        self.write(extra_envoy=[{"access": "request", "request_id": self.REQUESTS[0], "method": "POST",
                                 "path": "/v1/completions", "status": "503", "upstream": DECODE}])
        self.assertProblem("logged completion aaaa-1111 2 times (503, 200)")

    def test_a_request_the_epp_handled_twice_fails(self) -> None:
        self.write(extra_epp=[{"body": "Request handled", "x-request-id": self.REQUESTS[0], "trace_id": "trace0",
                               "incomingModelName": pd.MODEL_NAME, "endpoint": DECODE}])
        self.assertProblem("handled aaaa-1111 2 times")

    def test_a_second_picker_choice_fails(self) -> None:
        self.write(extra_epp=[picker(self.REQUESTS[0], PREFILL)])
        self.assertProblem("picked 2 prefill endpoint(s)")

    def test_two_engine_requests_under_one_routed_id_fail(self) -> None:
        self.write(decode_suffixes=("abcd1234", "ffff0000"))
        self.assertProblem("as 2 engine requests")

    def test_one_engine_request_logged_by_every_tp_rank_passes(self) -> None:
        raw = run_config()
        raw["server"]["tensor_parallel_size"] = 2
        self.llmd = load(raw)
        self.write(ranks=2, decode_suffixes=("abcd1234", "abcd1234"))
        self.assertEqual(self.problems(), [])

    def test_a_missing_kv_arrival_fails(self) -> None:
        self.write(decode_events=("admitted", "reads_posted"))
        self.assertProblem("never received its KV")

    def test_a_producer_that_never_released_fails(self) -> None:
        self.write(prefill_events=("admitted", "allocated"))
        self.assertProblem("never released")

    def test_an_engine_request_the_epp_never_routed_fails(self) -> None:
        self.write(extra_engine=("warmup-9999",))
        self.assertProblem("which the EPP never routed")

    def test_the_wrong_sidecar_protocol_fails(self) -> None:
        self.write(protocol="running Mooncake protocol")
        self.assertProblem("sidecar ran")

    def test_unscraped_endpoints_fail(self) -> None:
        self.write(scored=False)
        self.assertProblem("never scored")

    def test_a_picker_that_does_not_score_leaves_no_metrics_evidence(self) -> None:
        # The random control: no max-score picker, so no scored-endpoints records to require.
        self.llmd.router.epp_plugins = E2E / "epp" / "random.yaml"
        self.write(scored=False)
        result = llmd.routing_evidence(self.output, self.llmd.run)
        self.assertEqual(result["problems"], [])
        self.assertIsNone(result["evidence"]["metrics_seen_by_epp"])
        self.assertIn("random-picker", result["evidence"]["plugins"])
        self.assertEqual((result["evidence"]["uncalibrated"], result["evidence"]["peak_prefill_throughput"]), ({}, None))
        # The token-load arm scores, so its evidence is required.
        self.llmd.router.epp_plugins = E2E / "epp" / "token-load.yaml"
        self.write(scored=False)
        self.assertProblem("never scored")

    def test_a_calibrated_run_is_not_reported_uncalibrated(self) -> None:
        self.llmd.router.peak_prefill_throughput = 46002.8
        self.write()
        evidence = llmd.routing_evidence(self.output, self.llmd.run)["evidence"]
        self.assertEqual((evidence["uncalibrated"], evidence["peak_prefill_throughput"]), ({}, [46002.8]))
        # The saved config is the one the run's settings give: not another throughput's.
        self.llmd.router.peak_prefill_throughput = 50000.0
        self.assertProblem("saved EPP config is not the one")
        self.llmd.router.peak_prefill_throughput = None
        self.assertProblem("saved EPP config is not the one")

    def test_the_affinity_filters_decisions_are_reported(self) -> None:
        metrics = self.output / "llmd" / "epp-metrics.txt"
        self.write()
        self.assertIsNone(llmd.routing_evidence(self.output, self.llmd.run)["evidence"]["affinity_filter_decisions"])
        name = "llm_d_epp_prefix_cache_affinity_filter_decisions_total"
        metrics.write_text(f"# HELP {name} [ALPHA] decisions\n# TYPE {name} counter\n"
                           f'{name}{{outcome="no_match",plugin_name="prefix-cache-affinity-filter"}} 81\n'
                           f'{name}{{outcome="sticky",plugin_name="prefix-cache-affinity-filter"}} 3\n'
                           f'{name}{{outcome="sticky",plugin_name="another"}} 2\n'
                           f'other_total{{outcome="no_match"}} 9\n')
        evidence = llmd.routing_evidence(self.output, self.llmd.run)["evidence"]
        self.assertEqual(evidence["affinity_filter_decisions"], {"no_match": 81, "sticky": 5})
        # A configuration without the filter has none to report.
        self.llmd.router.epp_plugins = E2E / "epp" / "request-load.yaml"
        self.write()
        self.assertIsNone(llmd.routing_evidence(self.output, self.llmd.run)["evidence"]["affinity_filter_decisions"])

    def test_the_control_plugin_files_are_valid_picker_configs(self) -> None:
        for name, types in (("random", {"random-picker"}), ("token-load", {"token-load-scorer", "max-score-picker"}), ("request-load", {"active-request-scorer", "max-score-picker"})):
            self.llmd.router.epp_plugins = E2E / "epp" / f"{name}.yaml"
            config = llmd.epp_config(self.llmd.router)
            found = {p["type"] for p in config["plugins"]}
            with self.subTest(name=name):
                self.assertTrue(types <= found, found)
                # None names a prefix plugin. That alone does not make one prefix-blind: llm-d's token load
                # discounts cached prefixes (the EPP creates the approximate producer for it), so the
                # prefix-blind controls are the ones without the token-load scorer and its producer.
                self.assertFalse({t for t in found if "prefix" in t})
                self.assertEqual(not ({"token-load-scorer", "inflight-load-producer"} & found),
                                 name in ("random", "request-load"))
                self.assertEqual({p["name"] for p in config["schedulingProfiles"]}, {"prefill", "decode"})

    def test_client_and_router_counts_must_agree(self) -> None:
        self.write(completed=3)
        self.assertProblem("client completed 3")


class InstancesEvidenceTests(unittest.TestCase):
    """2P2D: each request's picked pair must be the instances that ran it,
    each instance checked on its own step logs."""

    ADDRESS = {"prefill": {"p0": "10.0.0.1", "p1": "10.0.0.2"}, "decode": {"d0": "10.0.0.3", "d1": "10.0.0.4"}}

    def setUp(self) -> None:
        raw = run_config()
        raw["prefill"]["node"], raw["decode"]["node"] = ["p0", "p1"], ["d0", "d1"]
        self.llmd = load(raw)
        self.output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shell.run(["rm", "-rf", str(self.output)]))

    def write(self, routes: dict[str, tuple[str, str]], ran: dict[str, tuple[str, str]] | None = None,
              also: dict[str, list[str]] | None = None) -> None:
        """``routes``: request -> its picked (prefill, decode) instance;
        ``ran``: where it actually ran (default: as picked); ``also``:
        another instance's extra requests."""
        directory = self.output / "llmd"
        directory.mkdir(exist_ok=True)
        address = {"instances": {role: {node: {"control": ip} for node, ip in nodes.items()}
                                 for role, nodes in self.ADDRESS.items()}}
        (directory / "endpoints.yaml").write_text(yaml.safe_dump(llmd.endpoints(self.llmd.run, address)))
        (directory / "epp-config.yaml").write_text(yaml.safe_dump(llmd.epp_config(self.llmd.router)))
        (directory / "cleanup.json").write_text(json.dumps({"removed": ["x"], "remaining": []}))
        (directory / "components.json").write_text(json.dumps(placed(self.llmd.run)))
        at = {node: f"{ip}:8000" for nodes in self.ADDRESS.values() for node, ip in nodes.items()}
        epp, sidecar, envoy = [], [], []
        engine: dict[str, list[str]] = {node: [] for nodes in self.ADDRESS.values() for node in nodes}
        for index, (request, (prefill, decode)) in enumerate(routes.items()):
            epp += [picker(request, at[decode]), picker(request, at[prefill]),
                    {"body": "Request handled", "x-request-id": request, "trace_id": f"t{index}",
                     "incomingModelName": pd.MODEL_NAME, "endpoint": at[decode]}]
            sidecar += [{"body": "SSRF protection: prefill target allowed", "trace_id": f"t{index}",
                         "target": at[prefill]}, {"body": "running NIXL protocol V2"}]
            envoy.append({"access": "request", "request_id": request, "method": "POST", "path": "/v1/completions",
                          "status": "200", "upstream": at[decode]})
            for node in (ran or routes)[request]:
                engine[node].append(request)
        for node, requests in (also or {}).items():
            engine[node] += requests
        epp.append({"body": "Selecting endpoints from candidates sorted by max score", "scored-endpoints": [
            {"Endpoint": {"Name": node, "MetricsHost": host, "CacheNumBlocks": 910}} for node, host in at.items()]})
        for name, records in (("x-gw-epp.log", epp), ("x-d0-sidecar.log", sidecar), ("x-gw-envoy.log", envoy)):
            (directory / name).write_text("\n".join(json.dumps(r) for r in records) + "\n")
        for role, nodes in self.ADDRESS.items():
            events = ("admitted", "allocated", "sends_done") if role == "prefill" else ("admitted", "reads_done")
            for node in nodes:
                ids = [f"cmpl-{r}-0-abcd1234" for r in engine[node]]
                (self.output / f"{role}-{node}-steps.jsonl").write_text(
                    json.dumps({"kv_events": {kind: ids for kind in events}}) + "\n")
        (self.output / "bench.json").write_text(json.dumps({"completed": len(routes)}))

    def evidence(self) -> dict:
        return llmd.routing_evidence(self.output, self.llmd.run)

    def test_each_request_runs_on_its_picked_pair(self) -> None:
        self.write({"a": ("p0", "d0"), "b": ("p1", "d1"), "c": ("p0", "d1")})
        result = self.evidence()
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["evidence"]["pairs"], {"p0->d0": 1, "p0->d1": 1, "p1->d1": 1})
        self.assertEqual(result["evidence"]["routed"], {"prefill": {"p0": 2, "p1": 1}, "decode": {"d0": 1, "d1": 2}})
        self.assertEqual(result["evidence"]["idle_instances"], [])

    def test_an_idle_instance_is_reported_not_failed(self) -> None:
        self.write({"a": ("p0", "d0"), "b": ("p0", "d1")})
        result = self.evidence()
        self.assertEqual(result["problems"], [])
        self.assertEqual(result["evidence"]["idle_instances"], ["p1"])

    def test_a_request_run_elsewhere_than_picked_fails(self) -> None:
        self.write({"a": ("p0", "d0"), "b": ("p1", "d1")}, ran={"a": ("p1", "d0"), "b": ("p1", "d1")})
        problems = self.evidence()["problems"]
        self.assertTrue(any("the EPP picked prefill p0, it ran on ['p1']" in p for p in problems), problems)

    def test_a_request_on_two_decode_instances_fails(self) -> None:
        self.write({"a": ("p0", "d0"), "b": ("p1", "d1")}, also={"d1": ["a"]})
        problems = self.evidence()["problems"]
        self.assertTrue(any("the EPP picked decode d0, it ran on ['d0', 'd1']" in p for p in problems), problems)


class RecheckTests(unittest.TestCase):
    """`pd.recheck` replays a saved run's routing evidence with its P/D
    checks (stubbed here): missing evidence is a problem."""

    def setUp(self) -> None:
        self.evidence = EvidenceTests("test_a_correct_run_passes")
        self.evidence.setUp()
        self.addCleanup(self.evidence.doCleanups)
        self.output = self.evidence.output
        run = self.evidence.llmd.run
        self.saved = {**dataclasses.asdict(run), "model": str(run.model), "addresses": {}}
        (self.output / "routers.json").write_text("{}")
        (self.output / "tent.json").write_text("{}")
        (self.output / "control-ports.json").write_text(json.dumps({"nodes": ["gateway"], "bytes": 0, "ports": {}}))
        stub = mock.patch.object(pd, "check", return_value={"problems": [], "kv_bytes_checked": 0, "transfers": {}})
        stub.start()
        self.addCleanup(stub.stop)

    def recheck(self, **saved) -> dict:
        (self.output / "run.json").write_text(json.dumps({**self.saved, **saved}, default=str))
        return pd.recheck(self.output)

    def test_a_run_replays_its_routing_evidence(self) -> None:
        verdict = self.recheck()
        self.assertEqual(verdict["problems"], [])
        self.assertEqual(verdict["llmd"]["completion_requests"], 2)
        self.assertNotIn("front", verdict)

    def test_the_exempt_node_is_the_saved_runs_gateway(self) -> None:
        self.recheck(router={"gateway": "elsewhere"})
        self.assertEqual(pd.check.call_args.args[4].router.gateway, "elsewhere")

    def test_missing_or_wrong_component_placement_fails(self) -> None:
        placement = self.output / "llmd" / "components.json"
        saved = json.loads(placement.read_text())
        self.assertEqual(self.recheck()["llmd"]["components"], saved)
        name = next(iter(saved))
        placement.write_text(json.dumps({**saved, name: {**saved[name], "cpuset": ""}}))
        self.assertTrue(any("ran on CPUs ''" in p for p in self.recheck()["problems"]))
        placement.write_text("null")
        self.assertTrue(any("placement evidence is missing" in p for p in self.recheck()["problems"]))
        placement.unlink()
        self.assertTrue(any("llmd/components.json" in p for p in self.recheck()["problems"]))

    def test_a_missing_cleanup_record_fails(self) -> None:
        (self.output / "llmd" / "cleanup.json").unlink()
        self.assertTrue(any("llmd/cleanup.json" in p for p in self.recheck()["problems"]))

    def test_a_run_without_its_evidence_fails(self) -> None:
        shell.run(["rm", "-rf", str(self.output / "llmd")])
        self.assertTrue(any("missing routing evidence" in p for p in self.recheck()["problems"]))

    def test_a_run_saved_without_a_router_is_refused(self) -> None:
        del self.saved["router"]
        with self.assertRaises(KeyError):
            self.recheck()

    def test_containers_left_after_the_run_fail(self) -> None:
        cleanup = self.output / "llmd" / "cleanup.json"
        cleanup.write_text(json.dumps({"removed": ["a", "b"], "remaining": []}))
        verdict = self.recheck()
        self.assertEqual((verdict["problems"], verdict["llmd"]["cleanup"]["removed"]), ([], ["a", "b"]))
        cleanup.write_text(json.dumps({"removed": ["a"], "remaining": ["b"]}))
        self.assertTrue(any("left after the run" in p for p in self.recheck()["problems"]))


@unittest.skipUnless(os.environ.get("E2E_DOCKER") == "1", "set E2E_DOCKER=1 to run docker topologies")
class CleanupTests(unittest.TestCase):
    """llm-d's containers join a node's network and carry the topology's
    label: ``down`` removes them, whether they run, exited or never started."""

    SPEC = spec.parse({
        "name": "llmd-cleanup-selftest",
        "image": "python:3.12-slim",
        "networks": {"ctl": {"subnet": "10.241.0.0/24"}},
        "nodes": {"gw": {"networks": ["ctl"], "cpus": "0"}},
    })

    def setUp(self) -> None:
        topology.down(self.SPEC.name)
        self.state = topology.up(self.SPEC)
        self.addCleanup(topology.down, self.SPEC.name)

    def _start_partially(self) -> list[str]:
        node = self.state["nodes"]["gw"]["container"]
        running = llmd.component_name(self.SPEC.name, "gw", "running")
        exited = llmd.component_name(self.SPEC.name, "gw", "exited")
        llmd.start_component(self.SPEC.name, node, running, "python:3.12-slim", ["sleep", "300"],
                             {"/etc/test": {"config.yaml": "a: 1\n"}})
        llmd.start_component(self.SPEC.name, node, exited, "python:3.12-slim", ["false"])
        with self.assertRaises(shell.CommandError):
            # Created, then failing to start: an unknown command.
            llmd.start_component(self.SPEC.name, node, llmd.component_name(self.SPEC.name, "gw", "broken"),
                                 "python:3.12-slim", ["/no/such/command"])
        return llmd.components(self.SPEC.name)

    def test_components_share_the_nodes_network_and_get_their_files(self) -> None:
        self._start_partially()
        running = llmd.component_name(self.SPEC.name, "gw", "running")
        mode = shell.run(["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", running]).strip()
        node = shell.run(["docker", "inspect", "-f", "{{.Id}}", self.state["nodes"]["gw"]["container"]]).strip()
        self.assertEqual(mode, f"container:{node}")
        # On its node's CPUs, as Docker and the kernel have it.
        container = self.state["nodes"]["gw"]["container"]
        self.assertEqual(llmd.placement([running]), {running: {
            "cpuset": "0", "network_mode": f"container:{node}", "node": container, "node_cpuset": "0"}})
        allowed = shell.run(["docker", "exec", running, "sh", "-c", "grep Cpus_allowed_list /proc/1/status"])
        self.assertEqual(allowed.split()[-1], "0")
        self.assertEqual(shell.run(["docker", "exec", running, "cat", "/etc/test/config.yaml"]), "a: 1\n")

    def test_down_removes_running_exited_and_unstarted_components(self) -> None:
        names = self._start_partially()
        self.assertEqual(len(names), 3)
        self.assertTrue(set(names) <= set(topology.containers(self.SPEC.name)))
        topology.down(self.SPEC.name)
        self.assertEqual(llmd.components(self.SPEC.name), [])

    def test_a_run_removes_its_own_components(self) -> None:
        self._start_partially()
        self.assertEqual(len(llmd.remove_components(self.SPEC.name)), 3)
        self.assertEqual(llmd.components(self.SPEC.name), [])
        # The node stays up.
        self.assertIn(self.state["nodes"]["gw"]["container"], topology.containers(self.SPEC.name))


if __name__ == "__main__":
    unittest.main()
