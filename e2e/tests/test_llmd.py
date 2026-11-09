"""llm-d runs: their configs, the router evidence check, and (E2E_DOCKER=1)
the llm-d containers' cleanup."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from harness import llmd, pd, shell, spec, topology


def run_config(**router) -> dict:
    return {"name": "llmd-test", "topology": "llmd-1p1d", "model": "model",
            "prefill": {"node": "prefill0", "kv_network": "data0"},
            "decode": {"node": "decode0", "kv_network": "data0"},
            "client": "client", "control_network": "control",
            "router": {"gateway": "gateway", **router},
            "server": {"connector": "nixl", "timing": "zero"}}


def load(raw: dict) -> llmd.LlmdRun:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "run.yaml"
        path.write_text(yaml.safe_dump(raw))
        return llmd.load(path)


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

    def test_one_engine_per_instance_for_now(self) -> None:
        raw = run_config()
        raw["decode"]["data_parallel_size"] = 2
        with self.assertRaisesRegex(ValueError, "data_parallel_size"):
            load(raw)

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
        front = llmd.LlmdFront(self.llmd)
        self.assertEqual(front.server_port("decode"), llmd.DECODE_ENGINE_PORT)
        self.assertEqual(front.server_port("prefill"), pd.SERVER_PORT)
        self.assertIn(f"--port {llmd.DECODE_ENGINE_PORT}",
                      " ".join(pd.server_command(self.llmd.run, "decode", ["10.1.0.1"], llmd.DECODE_ENGINE_PORT)))

    def test_sidecar_speaks_the_connector_in_plaintext(self) -> None:
        for connector, protocol in (("nixl", "nixlv2"), ("mooncake", "mooncake")):
            self.llmd.run.server.connector = connector
            args = llmd.sidecar_args(self.llmd.run, self.llmd.router)
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
    """`pd.recheck` replays a saved run's front end: llm-d's routing
    evidence for an llm-d run (its run.json's front), nothing more for a
    proxy run. The P/D checks themselves are stubbed."""

    def setUp(self) -> None:
        self.evidence = EvidenceTests("test_a_correct_run_passes")
        self.evidence.setUp()
        self.addCleanup(self.evidence.doCleanups)
        self.output = self.evidence.output
        run = self.evidence.llmd.run
        self.saved = {**dataclasses.asdict(run), "model": str(run.model), "proxy": str(run.proxy), "addresses": {}}
        (self.output / "routers.json").write_text("{}")
        (self.output / "tent.json").write_text("{}")
        (self.output / "control-ports.json").write_text(json.dumps({"nodes": ["gateway"], "bytes": 0, "ports": {}}))
        (self.output / "router.json").write_text(json.dumps({"gateway": "gateway"}))
        stub = mock.patch.object(pd, "check", return_value={"problems": [], "kv_bytes_checked": 0, "transfers": {}})
        stub.start()
        self.addCleanup(stub.stop)

    def recheck(self, **saved) -> dict:
        (self.output / "run.json").write_text(json.dumps({**self.saved, **saved}, default=str))
        return pd.recheck(self.output)

    def test_an_llmd_run_replays_its_routing_evidence(self) -> None:
        verdict = self.recheck(front="llmd")
        self.assertEqual((verdict["front"], verdict["problems"]), ("llmd", []))
        self.assertEqual(verdict["llmd"]["completion_requests"], 2)

    def test_the_exempt_nodes_come_from_the_saved_router(self) -> None:
        (self.output / "run.json").write_text(json.dumps({**self.saved, "front": "llmd"}, default=str))
        (self.output / "router.json").write_text(json.dumps({"gateway": "elsewhere"}))
        pd.recheck(self.output)
        self.assertEqual(pd.check.call_args.args[-1], ["elsewhere"])
        (self.output / "run.json").write_text(json.dumps({**self.saved, "front": "proxy"}, default=str))
        pd.recheck(self.output)
        self.assertEqual(pd.check.call_args.args[-1], [])

    def test_a_missing_cleanup_record_fails(self) -> None:
        (self.output / "llmd" / "cleanup.json").unlink()
        self.assertTrue(any("llmd/cleanup.json" in p for p in self.recheck(front="llmd")["problems"]))

    def test_an_llmd_run_without_its_evidence_fails(self) -> None:
        shell.run(["rm", "-rf", str(self.output / "llmd")])
        self.assertTrue(any("missing routing evidence" in p for p in self.recheck(front="llmd")["problems"]))

    def test_a_run_without_its_front_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            self.recheck()

    def test_containers_left_after_the_run_fail(self) -> None:
        cleanup = self.output / "llmd" / "cleanup.json"
        cleanup.write_text(json.dumps({"removed": ["a", "b"], "remaining": []}))
        verdict = self.recheck(front="llmd")
        self.assertEqual((verdict["problems"], verdict["llmd"]["cleanup"]["removed"]), ([], ["a", "b"]))
        cleanup.write_text(json.dumps({"removed": ["a"], "remaining": ["b"]}))
        self.assertTrue(any("left after the run" in p for p in self.recheck(front="llmd")["problems"]))

    def test_a_proxy_run_replays_as_before(self) -> None:
        verdict = self.recheck(front="proxy")
        self.assertEqual((verdict["front"], verdict["problems"]), ("proxy", []))
        self.assertNotIn("llmd", verdict)


@unittest.skipUnless(os.environ.get("E2E_DOCKER") == "1", "set E2E_DOCKER=1 to run docker topologies")
class CleanupTests(unittest.TestCase):
    """llm-d's containers join a node's network and carry the topology's
    label: ``down`` removes them, whether they run, exited or never started."""

    SPEC = spec.parse({
        "name": "llmd-cleanup-selftest",
        "image": "python:3.12-slim",
        "networks": {"ctl": {"subnet": "10.241.0.0/24"}},
        "nodes": {"gw": {"networks": ["ctl"]}},
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
