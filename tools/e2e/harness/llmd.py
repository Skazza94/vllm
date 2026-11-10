"""llm-d's router, the front end of every prefill/decode run (`harness.pd`),
without Kubernetes (llm-d v1.0's no-Kubernetes deployment).

Requests enter through Envoy on the gateway node, which asks llm-d's
endpoint picker (EPP, over ext_proc) where each goes. The EPP knows the
instances from an endpoints file (its ``file-discovery`` plugin), each
labelled with its ``llm-d.ai/role``, and scrapes their metrics; with
``always-disagg-pd-decider`` it picks a decode instance and a prefill
instance for every request. The decode instance's disaggregation sidecar
(llm-d's routing sidecar) takes the request: it sends the prefill request
to the prefill instance the EPP named and the decode request to its own
vLLM, the KV moving between them through the run's connector (sidecar
``nixlv2`` for NixlConnector, ``mooncake`` for MooncakeConnector).

llm-d's processes run in containers of their own images that join a node's
network namespace, as a pod's containers share its network (``--network
container:<node>``): the EPP and Envoy the gateway's, a sidecar each decode
node's. They carry the topology's label, so ``down`` removes them too; a run
removes them itself after saving their logs and configs. The servers, the
client's benchmark, the KV transfers and the run's checks are
`harness.pd`'s; this module starts the router and checks its own evidence.

A component container gets its node's CPU set (``docker --cpuset-cpus``, as
the node has it), so the router's processes run where the topology puts
that node: Envoy and the EPP on the gateway's cores, a sidecar on its
decode node's. What Docker reports for each is saved and checked.

An instance of several DP ranks is one endpoint. Who picks the rank depends
on the connector: under Mooncake the sidecar selects the producer (prefill)
DP rank itself, any one of those the prefill's bootstrap server lists, and
names it to both sides; decode's rank is vLLM's choice. Under NIXL vLLM
balances the ranks of both sides. The EPP routes to instances, not ranks.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import time

import yaml

from . import pd, shell, topology


IMAGES = {
    # llm-d v1.0.0 (guides/env.sh: ROUTER_EPP_VERSION; the sidecar of the same router release).
    "epp": "ghcr.io/llm-d/llm-d-router-endpoint-picker:v1.0.0"
           "@sha256:024188e998f516879f68532016ca24fabca00afbdcc2af9a61b17634e197df2c",
    "sidecar": "ghcr.io/llm-d/llm-d-router-disagg-sidecar:v1.0.0"
               "@sha256:83cdf20199485329baca89b69d78f695735855ed93412aeaeec325d2374ced04",
    # The no-Kubernetes guide's Envoy.
    "envoy": "envoyproxy/envoy:distroless-v1.33.2"
             "@sha256:85500e28ed088ec39ff0adc1be3d358a8ad062926aaa62c36b28bde00919e4e8",
}
LLMD_LABEL = "e2e.llmd"
GATEWAY_PORT = 8081
ENVOY_ADMIN_PORT = 19000
EPP_GRPC_PORT = 9002
EPP_HEALTH_PORT = 9003
EPP_METRICS_PORT = 9090
SIDECAR_PORT = pd.SERVER_PORT  # what the EPP knows a decode instance by
DECODE_ENGINE_PORT = 8200  # the sidecar's default --model-server-port
ROLE_LABEL = "llm-d.ai/role"
SIDECAR_CONNECTOR = {"nixl": "nixlv2", "mooncake": "mooncake"}
# The guide's value for its own fleet (gpt-oss-120b, 8 H200 prefill
# instances): not calibrated for any simulated model here.
UNCALIBRATED_PEAK_PREFILL_THROUGHPUT = 33821


def default_epp_plugins(peak_prefill_throughput: float | None = None, prefix_index: str = "approximate",
                        block_tokens: int = 16) -> dict:
    """llm-d v1.0's P/D router configuration
    (guides/pd-disaggregation/router/pd-disaggregation.values.yaml): every
    request disaggregated, prefill picked by prefix affinity and queued
    prefill tokens, decode by active requests.

    Two things about its prefix signal, as the pinned router has them:
    - The token load is itself prefix-aware: the in-flight load counts only
      the uncached part of each prompt on each endpoint, so an endpoint that
      holds a request's prefix looks less loaded for it. The affinity filter
      comes on top, and only for endpoints whose prefix score reaches its
      ``affinityThreshold`` (0.80 by default): below it the filter keeps
      every endpoint (its ``no_match`` outcome).
    - The approximate producer matches in blocks of at least 64 tokens,
      whatever the servers' block size (``approximateprefix`` clamps it): a
      coarser granularity than a server's own cache with smaller blocks."""

    throughput = UNCALIBRATED_PEAK_PREFILL_THROUGHPUT if peak_prefill_throughput is None else peak_prefill_throughput
    if prefix_index == "precise":
        # llm-d's precise P/D configuration (guides/models/glm-5-2/router/
        # precise-routing.values.yaml): the prompt tokenized by a server's
        # render endpoint (prefill's server port, decode's behind its
        # sidecar), the index fed by every instance's KV-cache events.
        named = {"prefixMatchInfoProducerName": "precise-prefix-cache-producer"}
        prefix = [
            {"type": "token-producer", "parameters": {"modelName": pd.MODEL_NAME, "vllm": {"endpointDiscovery": {
                "attemptTimeout": "2s",
                "portRules": [{"selector": {"matchLabels": {ROLE_LABEL: "prefill"}}, "basePort": pd.SERVER_PORT},
                              {"selector": {"matchLabels": {ROLE_LABEL: "decode"}}, "basePort": DECODE_ENGINE_PORT}]}}}},
            {"type": "endpoint-notification-source"},
            {"type": "precise-prefix-cache-producer", "parameters": {
                "tokenProcessorConfig": {"blockSizeTokens": block_tokens},
                "indexerConfig": {"kvBlockIndexConfig": {"enableMetrics": True}},
                "kvEventsConfig": {"engineType": "vllm", "topicFilter": "kv@", "discoverPods": True,
                                   "podDiscoveryConfig": {"socketPort": pd.KV_EVENTS_PORT,
                                                          "replaySocketPort": pd.KV_EVENTS_REPLAY_PORT}}}},
            {"type": "inflight-load-producer", "parameters": named},
            {"type": "prefix-cache-affinity-filter", "parameters": {**named, "peakPrefillThroughput": throughput}},
        ]
    else:
        prefix = [
            {"type": "approx-prefix-cache-producer", "parameters": {"maxPrefixTokensToMatch": 131072}},
            {"type": "inflight-load-producer"},
            {"type": "prefix-cache-affinity-filter", "parameters": {"peakPrefillThroughput": throughput}},
        ]
    return {
        "plugins": [
            {"type": "always-disagg-pd-decider"},
            {"type": "disagg-profile-handler", "parameters": {"deciders": {"prefill": "always-disagg-pd-decider"}}},
            {"type": "prefill-filter"},
            {"type": "decode-filter"},
            *prefix,
            {"type": "token-load-scorer"},
            {"type": "active-request-scorer"},
            {"type": "max-score-picker"},
        ],
        "schedulingProfiles": [
            {"name": "prefill", "plugins": [{"pluginRef": "prefill-filter"},
                                            {"pluginRef": "prefix-cache-affinity-filter"},
                                            {"pluginRef": "token-load-scorer"},
                                            {"pluginRef": "max-score-picker"}]},
            {"name": "decode", "plugins": [{"pluginRef": "decode-filter"},
                                           {"pluginRef": "active-request-scorer"},
                                           {"pluginRef": "max-score-picker"}]},
        ],
    }


def epp_config(router: pd.Router, block_tokens: int = 16) -> dict:
    """The EPP's config: the plugins, its endpoints from the file, their
    metrics scraped; with a precise prefix index (``block_tokens`` the
    servers' block size), the endpoints' notifications feeding it."""

    plugins = default_epp_plugins(router.peak_prefill_throughput, router.prefix_index, block_tokens) if (
        router.epp_plugins is None) else yaml.safe_load(router.epp_plugins.read_text(encoding="utf-8"))
    if set(plugins) - {"plugins", "schedulingProfiles"}:
        raise ValueError("router.epp_plugins takes plugins and schedulingProfiles only")
    return {
        "apiVersion": "llm-d.ai/v1",
        "kind": "EndpointPickerConfig",
        "plugins": [
            {"name": "file-discovery", "type": "file-discovery",
             "parameters": {"path": "/etc/epp/endpoints.yaml", "watchFile": False}},
            {"name": "metrics-source", "type": "metrics-data-source"},
            {"name": "metrics-extractor", "type": "core-metrics-extractor"},
            *plugins["plugins"],
        ],
        "schedulingProfiles": plugins["schedulingProfiles"],
        "dataLayer": {
            "injectDefaults": False,
            "discovery": {"endpoints": {"pluginRef": "file-discovery"}},
            "sources": [
                {"pluginRef": "metrics-source", "extractors": [{"pluginRef": "metrics-extractor"}]},
                # Each endpoint added or removed, to the plugins that follow them (the precise index's
                # event subscribers).
                *([{"pluginRef": "endpoint-notification-source",
                    "extractors": [{"pluginRef": "precise-prefix-cache-producer"}]}]
                  if any(p.get("type") == "precise-prefix-cache-producer" for p in plugins["plugins"]) else []),
            ],
        },
    }


def endpoints(run: pd.Run, address: dict) -> dict:
    """The instances the EPP knows: each prefill instance's vLLM and each
    decode instance's sidecar, by their control network addresses."""

    instances = address["instances"]
    found = []
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        base = pd.SERVER_PORT if role == "prefill" else SIDECAR_PORT
        for node in side.nodes:
            entry = {"name": node, "address": instances[role][node]["control"], "port": str(base),
                     "labels": {ROLE_LABEL: role, "model": pd.MODEL_NAME}}
            if not pd.rank_routed(run, side):
                found.append(entry)
                continue
            # Per-rank routing: each DP rank's API server (decode's: its
            # sidecar's listener for the rank), on the port after the last.
            for rank in range(side.data_parallel_size):
                found.append({**entry, "name": endpoint_name(node, rank), "port": str(base + rank),
                              "rankIndex": rank})
    return {"endpoints": found}


def endpoint_name(node: str, rank: int) -> str:
    """A per-rank endpoint's name."""
    return f"{node}-dp{rank}"


def envoy_config() -> dict:
    """llm-d v1.0's no-Kubernetes Envoy config
    (guides/no-kubernetes-deployment/router/envoy/envoy.yaml): the listener
    on GATEWAY_PORT, ext_proc to the EPP in the same network namespace, each
    request sent where the EPP's x-gateway-destination-endpoint says."""

    circuit = {"max_connections": 40000, "max_pending_requests": 40000, "max_requests": 40000}
    manager = {
        "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3."
                 "HttpConnectionManager",
        "stat_prefix": f"http-{GATEWAY_PORT}",
        "route_config": {"name": "vllm", "virtual_hosts": [{
            "name": "vllm-default", "domains": ["*"],
            "routes": [{
                "match": {"prefix": "/"},
                "route": {"cluster": "original_destination_cluster", "timeout": "86400s",
                          "idle_timeout": "86400s", "upgrade_configs": [{"upgrade_type": "websocket"}]},
                "typed_per_filter_config": {"envoy.filters.http.ext_proc": {
                    "@type": "type.googleapis.com/envoy.config.route.v3.FilterConfig", "config": {}}},
            }],
        }]},
        "http_filters": [
            {"name": "envoy.filters.http.ext_proc", "typed_config": {
                "@type": "type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor",
                "grpc_service": {"envoy_grpc": {"cluster_name": "ext_proc",
                                                "authority": f"localhost:{EPP_GRPC_PORT}"},
                                 "timeout": "10s"},
                "processing_mode": {"request_header_mode": "SEND", "response_header_mode": "SEND",
                                    "request_body_mode": "FULL_DUPLEX_STREAMED",
                                    "response_body_mode": "FULL_DUPLEX_STREAMED",
                                    "request_trailer_mode": "SEND", "response_trailer_mode": "SEND"},
                "message_timeout": "1000s"}},
            {"name": "envoy.filters.http.router", "typed_config": {
                "@type": "type.googleapis.com/envoy.extensions.filters.http.router.v3.Router",
                "suppress_envoy_headers": True}},
        ],
        "http2_protocol_options": {"max_concurrent_streams": 100, "initial_stream_window_size": 65536,
                                   "initial_connection_window_size": 1048576},
        "use_remote_address": True,
        # Not in the guide: one JSON line per request (the run's evidence).
        "access_log": [{"name": "envoy.access_loggers.stdout", "typed_config": {
            "@type": "type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog",
            "log_format": {"json_format": {
                "access": "request", "request_id": "%REQ(X-REQUEST-ID)%", "method": "%REQ(:METHOD)%",
                "path": "%REQ(:PATH)%", "status": "%RESPONSE_CODE%", "upstream": "%UPSTREAM_HOST%",
                "flags": "%RESPONSE_FLAGS%", "duration_ms": "%DURATION%",
                "bytes_received": "%BYTES_RECEIVED%", "bytes_sent": "%BYTES_SENT%"}}}}],
        "normalize_path": True,
        "merge_slashes": True,
        "server_header_transformation": "PASS_THROUGH",
        "common_http_protocol_options": {"headers_with_underscores_action": "REJECT_REQUEST"},
        "path_with_escaped_slashes_action": "UNESCAPE_AND_REDIRECT",
    }
    return {
        "admin": {"address": {"socket_address": {"address": "127.0.0.1", "port_value": ENVOY_ADMIN_PORT}}},
        "static_resources": {
            "listeners": [{
                "name": "vllm",
                "address": {"socket_address": {"address": "0.0.0.0", "port_value": GATEWAY_PORT}},
                "per_connection_buffer_limit_bytes": 32768,
                "filter_chains": [{"filters": [{"name": "envoy.filters.network.http_connection_manager",
                                                "typed_config": manager}]}],
            }],
            "clusters": [
                {"name": "original_destination_cluster", "type": "ORIGINAL_DST", "connect_timeout": "1000s",
                 "lb_policy": "CLUSTER_PROVIDED", "circuit_breakers": {"thresholds": [circuit]},
                 "original_dst_lb_config": {"use_http_header": True,
                                            "http_header_name": "x-gateway-destination-endpoint"}},
                {"name": "ext_proc", "type": "STATIC", "connect_timeout": "86400s", "lb_policy": "LEAST_REQUEST",
                 "circuit_breakers": {"thresholds": [{**circuit, "max_retries": 1024}]},
                 # The EPP serves gRPC health on its ext_proc port (--health-checking).
                 "health_checks": [{"timeout": "2s", "interval": "10s", "unhealthy_threshold": 3,
                                    "healthy_threshold": 2, "reuse_connection": True,
                                    "grpc_health_check": {
                                        "service_name": "envoy.service.ext_proc.v3.ExternalProcessor"}}],
                 "typed_extension_protocol_options": {
                     "envoy.extensions.upstreams.http.v3.HttpProtocolOptions": {
                         "@type": "type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions",
                         "explicit_http_config": {"http2_protocol_options": {
                             "initial_stream_window_size": 65536, "initial_connection_window_size": 1048576}}}},
                 "load_assignment": {"cluster_name": "ext_proc", "endpoints": [{"lb_endpoints": [{"endpoint": {
                     "address": {"socket_address": {"address": "127.0.0.1", "port_value": EPP_GRPC_PORT}}}}]}]}},
            ],
        },
    }


def epp_args(router: pd.Router) -> list[str]:
    """The EPP's flags (the no-Kubernetes guide's): plaintext, gRPC health
    on its ext_proc port for Envoy."""
    return ["--config-file=/etc/epp/config.yaml", "--pool-name=file-discovery", "--pool-namespace=default",
            f"--grpc-port={EPP_GRPC_PORT}", f"--grpc-health-port={EPP_HEALTH_PORT}",
            f"--metrics-port={EPP_METRICS_PORT}", "--secure-serving=false", "--health-checking=true",
            f"--v={router.epp_verbosity}"]


def sidecar_args(run: pd.Run) -> list[str]:
    """The decode sidecar's flags: plaintext, no InferencePool allowlist
    (there is none without Kubernetes), the run's connector protocol."""
    return [f"--port={SIDECAR_PORT}", f"--model-server-port={DECODE_ENGINE_PORT}",
            # Per-rank routing: a listener a decode rank, each to its rank's API server.
            *([f"--data-parallel-size={run.decode.data_parallel_size}"] if pd.rank_routed(run, run.decode) else []),
            f"--kv-connector={SIDECAR_CONNECTOR[run.server.connector]}",
            f"--mooncake-bootstrap-port={pd.BOOTSTRAP_PORT}", "--secure-serving=false",
            "--enable-ssrf-protection=false", f"--zap-log-level={run.router.sidecar_verbosity}"]


def component_name(topology_name: str, node: str, component: str) -> str:
    return f"{topology_name}-{node}-{component}"


def cpuset(container: str) -> str:
    """A container's CPU set as Docker has it ('' if unrestricted)."""
    return shell.run(["docker", "inspect", "-f", "{{.HostConfig.CpusetCpus}}", container]).strip()


def start_component(topology_name: str, node_container: str, name: str, image: str, args: list[str],
                    files: dict[str, dict[str, str]] | None = None, env: dict[str, str] | None = None) -> None:
    """A container of ``image`` in ``node_container``'s network namespace
    and on its CPU set (sharing the namespace alone would leave it on every
    core), labelled for ``down``; ``files`` maps a directory in it to the
    files (name: text) copied there before it starts."""

    cpus = cpuset(node_container)
    shell.run(["docker", "create", "--name", name, "--label", f"{topology.LABEL}={topology_name}",
               "--label", f"{LLMD_LABEL}={topology_name}", "--network", f"container:{node_container}",
               *(["--cpuset-cpus", cpus] if cpus else []),
               *(arg for name_, value in (env or {}).items() for arg in ("--env", f"{name_}={value}")),
               image, *args])
    for directory, contents in (files or {}).items():
        with tempfile.TemporaryDirectory() as local:
            # docker cp keeps modes (the owner becomes root): readable for
            # the images' non-root users.
            Path(local).chmod(0o755)
            for file, text in contents.items():
                (Path(local) / file).write_text(text, encoding="utf-8")
                (Path(local) / file).chmod(0o644)
            shell.run(["docker", "cp", f"{local}/.", f"{name}:{directory}"])
    shell.run(["docker", "start", name])


def _wait_component(client: str, url: str, name: str, timeout: float = 120) -> None:
    """Until ``url`` answers from the client node; a timeout reports the
    component container's state and log tail."""
    try:
        pd._wait(client, url, timeout, client, "/dev/null")
    except TimeoutError:
        state = shell.run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", name], check=False)
        tail = shell.run(["sh", "-c", f"docker logs --tail 20 {name} 2>&1"], check=False)
        raise TimeoutError(f"{url} did not answer within {timeout:.0f} s; {name} is {state.strip()}, "
                           f"its log ends with:\n{tail}") from None


def placement(names: list[str]) -> dict[str, dict]:
    """Where Docker runs each component: its own CPU set, and the node whose
    network namespace it joined with that node's CPU set."""
    found: dict[str, dict] = {}
    for name in names:
        described = json.loads(shell.run(["docker", "inspect", name]))[0]["HostConfig"]
        mode = described.get("NetworkMode") or ""
        node: dict = {"name": None, "cpuset": None}
        if mode.startswith("container:"):
            inspected = json.loads(shell.run(["docker", "inspect", mode.split(":", 1)[1]]))[0]
            node = {"name": inspected["Name"].lstrip("/"), "cpuset": inspected["HostConfig"].get("CpusetCpus") or ""}
        found[name] = {"cpuset": described.get("CpusetCpus") or "", "network_mode": mode,
                       "node": node["name"], "node_cpuset": node["cpuset"]}
    return found


def expected_components(run: pd.Run) -> dict[str, str]:
    """The run's component containers and the node container each belongs to."""
    gateway = run.router.gateway
    expected = {component_name(run.topology, gateway, kind): f"{run.topology}-{gateway}" for kind in ("epp", "envoy")}
    for node in run.decode.nodes:
        expected[component_name(run.topology, node, "sidecar")] = f"{run.topology}-{node}"
    return expected


def placement_problems(saved: object, run: pd.Run) -> list[str]:
    """The saved placement (``llmd/components.json``) against the run: every
    component present, in its node's network namespace and on that node's
    CPU set."""
    if not isinstance(saved, dict):
        return ["the router components' placement evidence is missing"]
    problems = []
    expected = expected_components(run)
    for name, node in expected.items():
        described = saved.get(name)
        if not isinstance(described, dict):
            problems.append(f"no placement evidence for the component {name}")
            continue
        if described.get("node") != node:
            problems.append(f"{name} joined {described.get('node')!r}, not its node {node}")
        if not isinstance(described.get("cpuset"), str) or described.get("cpuset") != described.get("node_cpuset"):
            problems.append(f"{name} ran on CPUs {described.get('cpuset')!r}, its node {node} on "
                            f"{described.get('node_cpuset')!r}")
    if unknown := sorted(set(saved) - set(expected)):
        problems.append(f"components the run does not have: {unknown}")
    return problems


def components(topology_name: str) -> list[str]:
    """The llm-d containers of a topology, running or not."""
    output = shell.run(["docker", "ps", "-a", "--filter", f"label={LLMD_LABEL}={topology_name}",
                        "--format", "{{.Names}}"])
    return sorted(line for line in output.splitlines() if line)


def remove_components(topology_name: str) -> list[str]:
    names = components(topology_name)
    if names:
        shell.run(["docker", "rm", "-f", *names])
    return names


class Front:
    """llm-d's router for a run: `harness.pd.execute` drives it."""

    def __init__(self, run: pd.Run) -> None:
        self.run = run
        self.router = run.router
        self.configs: dict[str, str] = {}
        self.gateway_address: str | None = None
        self.output: Path | None = None
        self.address: dict = {}

    def server_port(self, role: str) -> int:
        """The HTTP port ``role``'s vLLM serves on: a decode instance's
        sidecar has SIDECAR_PORT."""
        return DECODE_ENGINE_PORT if role == "decode" else pd.SERVER_PORT

    def endpoint_port(self, role: str) -> int:
        """The port the router knows ``role``'s instance by."""
        return SIDECAR_PORT if role == "decode" else pd.SERVER_PORT

    def health_url(self, role: str, control: str) -> str:
        """The URL that answers once ``role``'s server is up (from the
        client): with per-rank routing, once every rank's is (the
        instance's aggregated health)."""
        if pd.rank_routed(self.run, getattr(self.run, role)):
            return f"http://{control}:{pd.DP_SUPERVISOR_PORT}/health"
        return f"http://{control}:{self.server_port(role)}/health"

    def prepare(self, container: dict[str, str]) -> None:
        """Before the servers start: a previous run's router would route
        this run's requests."""
        remove_components(self.run.topology)

    def start(self, container: dict[str, str], address: dict) -> str:
        """Start the router once the servers are up; returns the base URL
        the client's benchmark sends to."""
        run, router = self.run, self.router
        self.address = address
        state = topology.load_state(run.topology)
        nodes = state["nodes"]
        gateway = nodes[router.gateway]["container"]
        self.gateway_address = nodes[router.gateway]["interfaces"][run.control_network]["address"]
        self.configs = {
            "epp-config.yaml": yaml.safe_dump(epp_config(router, run.server.block_size or 16), sort_keys=False),
            "endpoints.yaml": yaml.safe_dump(endpoints(run, address), sort_keys=False),
            "envoy.yaml": yaml.safe_dump(envoy_config(), sort_keys=False),
        }
        for node in run.decode.nodes:
            # The sidecar names its per-rank listeners by its pod address.
            start_component(run.topology, container[node], component_name(run.topology, node, "sidecar"),
                            IMAGES["sidecar"], sidecar_args(run),
                            env={"POD_IP": address["instances"]["decode"][node]["control"]})
        start_component(run.topology, gateway, component_name(run.topology, router.gateway, "epp"), IMAGES["epp"],
                        epp_args(router), {"/etc/epp": {"config.yaml": self.configs["epp-config.yaml"],
                                                        "endpoints.yaml": self.configs["endpoints.yaml"]}})
        start_component(run.topology, gateway, component_name(run.topology, router.gateway, "envoy"),
                        IMAGES["envoy"],
                        ["--service-node", "envoy-proxy", "--log-level", "info", "--concurrency", "2",
                         "--drain-strategy", "immediate", "--drain-time-s", "60", "-c", "/etc/envoy/envoy.yaml"],
                        {"/etc/envoy": {"envoy.yaml": self.configs["envoy.yaml"]}})
        client = container[run.client]
        # The sidecar answers /health itself; the EPP's metrics port; Envoy's
        # admin readiness (its listener up, the EPP's cluster checked).
        for node in run.decode.nodes:
            _wait_component(client, f"http://{address['instances']['decode'][node]['control']}:{SIDECAR_PORT}/health",
                            component_name(run.topology, node, "sidecar"))
        _wait_component(client, f"http://{self.gateway_address}:{EPP_METRICS_PORT}/metrics",
                        component_name(run.topology, router.gateway, "epp"))
        self._wait_routable(client)
        return f"http://{self.gateway_address}:{GATEWAY_PORT}"

    def _wait_routable(self, client: str, timeout: float = 120) -> None:
        """Until the EPP has every instance ready, so the first request
        finds them all: its pool metrics count them."""
        url = f"http://{self.gateway_address}:{EPP_METRICS_PORT}/metrics"
        fetch = ("import sys, urllib.request\n"
                 "try:\n    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())\n"
                 "except Exception as error:\n    print('NO ANSWER', error)\n")
        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            text = topology.exec_in(client, [pd.PYTHON, "-c", fetch, url])
            if _pool_ready(text, len(endpoints(self.run, self.address)["endpoints"])):
                return
            time.sleep(3)
        raise TimeoutError(f"the EPP did not report a ready pool within {timeout:.0f} s:\n"
                           + "\n".join(line for line in text.splitlines() if "pool" in line)[:3000])

    def collect(self, container: dict[str, str], output: Path) -> None:
        """Its configs and logs, whatever happened."""
        self.output = output
        directory = output / "llmd"
        directory.mkdir(parents=True, exist_ok=True)
        for file, text in self.configs.items():
            (directory / file).write_text(text, encoding="utf-8")
        (directory / "images.json").write_text(json.dumps(IMAGES, indent=2) + "\n")
        if self.gateway_address is not None:
            fetch = ("import sys, urllib.request\n"
                     "try:\n    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())\n"
                     "except Exception as error:\n    print('NO ANSWER', error)\n")
            text = topology.exec_in(container[self.run.client], [
                pd.PYTHON, "-c", fetch, f"http://{self.gateway_address}:{EPP_METRICS_PORT}/metrics"], check=False)
            (directory / "epp-metrics.txt").write_text(text, encoding="utf-8")
        names = components(self.run.topology)
        # Where each ran, from Docker (not from this module's arguments).
        (directory / "components.json").write_text(json.dumps(placement(names), indent=2) + "\n")
        # Stopped first: Envoy flushes its access log as it exits.
        for name in names:
            state = shell.run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", name],
                              check=False)
            (directory / f"{name}.state").write_text(state, encoding="utf-8")
        if names:
            shell.run(["docker", "stop", "-t", "15", *names], check=False, timeout=60)
        for name in names:
            logs = shell.run(["sh", "-c", f"docker logs {name} 2>&1"], check=False)
            (directory / f"{name}.log").write_text(logs, encoding="utf-8")

    def stop(self, container: dict[str, str]) -> None:
        """After collecting: its containers removed."""
        removed = remove_components(self.run.topology)
        if self.output is not None:
            # What the run removed, and that nothing of it is left.
            (self.output / "llmd" / "cleanup.json").write_text(json.dumps(
                {"removed": removed, "remaining": components(self.run.topology)}, indent=2) + "\n")


def _pool_ready(metrics: str, instances: int) -> bool:
    """The EPP's metrics show its endpoints scraped: a ready-endpoint gauge
    of at least the instances' count, whichever v1.0 names it."""
    for line in metrics.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        if re.search(r"(ready_pods|ready_endpoints|pool_ready)", name):
            try:
                return float(value) >= instances
            except ValueError:
                return False
    return False


ENGINE_REQUEST_ID = re.compile(r"^(?:cmpl|chatcmpl)-(?P<id>.+)-\d+-[0-9a-f]+$")
# A request's completion on each side, as the simulator's step logs name it
# for either connector: the producer released the request's blocks, the
# consumer has its KV (NIXL's decode read it; Mooncake's prefill wrote it).
PRODUCER_DONE = "sends_done"
CONSUMER_DONE = "reads_done"
SIDECAR_PROTOCOL = {"nixl": "running NIXL protocol V2", "mooncake": "running Mooncake protocol"}


def _json_lines(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
    return records


def _engine_requests(texts: list[str]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """An instance's requests from its ranks' step logs: each kv_events
    kind's, by the id the router gave them (``x-request-id``: vLLM names a
    request ``cmpl-<it>-0-<suffix>``), and each such id's full engine ids
    (one, however many ranks log it; two would be two engine requests)."""
    kinds: dict[str, set[str]] = {}
    full: dict[str, set[str]] = {}
    for text in texts:
        for record in pd._records(text):
            for kind, value in (record.get("kv_events") or {}).items():
                for request in value if isinstance(value, (list, dict)) else []:
                    match = ENGINE_REQUEST_ID.match(str(request))
                    routed = match["id"] if match else str(request)
                    kinds.setdefault(kind, set()).add(routed)
                    full.setdefault(routed, set()).add(str(request))
    return kinds, full


REQUIRED_EVIDENCE = ("llmd/endpoints.yaml", "llmd/epp-config.yaml", "llmd/*-epp.log", "llmd/*-sidecar.log",
                     "llmd/*-envoy.log", "llmd/cleanup.json", "llmd/components.json", "bench.json")
PICKED = "Completed running picker plugin successfully"


def _picked(record: dict) -> list[str]:
    """The endpoints (address:port) a successful picker record targets."""
    return [f"{(target.get('Endpoint') or {}).get('Address')}:{(target.get('Endpoint') or {}).get('Port')}"
            for target in ((record.get("result") or {}).get("TargetEndpoints") or [])]


def instance_steps(output: Path, run: pd.Run) -> dict[str, tuple[str, list[str]]]:
    """Each endpoint's role and its own ranks' step logs, by its name: an
    instance's node, or with per-rank routing each DP rank's
    (`endpoint_name`)."""
    found: dict[str, tuple[str, list[str]]] = {}
    for role in ("prefill", "decode"):
        side = getattr(run, role)
        for dp, tp in side.ranks(side.tp(run.server)):
            name = endpoint_name(side.node_of(dp), dp) if pd.rank_routed(run, side) else side.node_of(dp)
            found.setdefault(name, (role, []))[1].append((output / side.step_log(role, dp, tp)).read_text())
    return found


def routing_evidence(output: Path, run: pd.Run) -> dict:
    """What the router did with each request, from the client's results,
    Envoy's, the EPP's and the sidecar's logs and the instances' step logs.

    Every benchmark completion went through Envoy once and the EPP once; the
    EPP's picker chose exactly one decode and one prefill endpoint for it
    (``TargetEndpoints``); Envoy sent it to that decode endpoint, the
    sidecar its prefill to that prefill endpoint, once; it ran on exactly
    those instances, as one engine request each; its producer released it
    and its consumer received its KV. Every record is counted before it is
    matched: a repeated or conflicting one is a problem, as is missing
    evidence. Requests without a model (the benchmark's own, not
    completions) are counted apart."""

    directory = output / "llmd"
    missing = [pattern for pattern in REQUIRED_EVIDENCE if not list(output.glob(pattern))]
    if missing:
        return {"problems": [f"missing routing evidence: {', '.join(missing)}"], "evidence": {"missing": missing}}
    problems: list[str] = []
    named = {f"{e['address']}:{e['port']}": e for e in
             yaml.safe_load((directory / "endpoints.yaml").read_text())["endpoints"]}
    role_of = {address: e["labels"].get(ROLE_LABEL) for address, e in named.items()}
    # The endpoints the EPP was given are the run's: each instance, or with
    # per-rank routing each of its DP ranks by its port and rank index.
    def shape(found: list[dict]) -> list[tuple]:
        return sorted((str(e.get("name")), str(e.get("port")), str((e.get("labels") or {}).get(ROLE_LABEL)),
                       e.get("rankIndex")) for e in found)
    expected = endpoints(run, {"instances": {role: {node: {"control": None} for node in getattr(run, role).nodes}
                                             for role in ("prefill", "decode")}})["endpoints"]
    if shape(list(named.values())) != shape(expected) or len(named) != len(expected):
        problems.append(f"the saved endpoints are not the run's: {shape(list(named.values()))[:3]}..., "
                        f"not {shape(expected)[:3]}...")

    handled: dict[str, list[dict]] = {}
    picks: dict[str, list[list[str]]] = {}
    others: set[str] = set()
    scraped: dict[str, dict] = {}
    for record in (r for log in sorted(directory.glob("*-epp.log")) for r in _json_lines(log)):
        body = record.get("body")
        request = record.get("x-request-id")
        if body == "Request handled":
            if not record.get("incomingModelName"):
                others.add(request)
                continue
            handled.setdefault(request, []).append(record)
        elif body == PICKED and request:
            picks.setdefault(request, []).append(_picked(record))
        elif body == "Selecting endpoints from candidates sorted by max score":
            for scored in record.get("scored-endpoints") or []:
                endpoint = scored.get("Endpoint") or {}
                seen = scraped.setdefault(endpoint.get("Name"), {"metrics_host": endpoint.get("MetricsHost")})
                for key in ("CacheNumBlocks", "CacheBlockSize", "RunningRequestsSize", "WaitingQueueSize",
                            "KVCacheUsagePercent"):
                    value = endpoint.get(key)
                    if isinstance(value, (int, float)):
                        seen[f"max_{key}"] = max(seen.get(f"max_{key}", value), value)
    for request, records in handled.items():
        if len(records) > 1:
            problems.append(f"the EPP handled {request} {len(records)} times")

    dispatched: dict[str, list[str]] = {}
    protocols: dict[str, int] = {}
    for record in (r for log in sorted(directory.glob("*-sidecar.log")) for r in _json_lines(log)):
        if record.get("trace_id") and record.get("target"):
            dispatched.setdefault(record["trace_id"], []).append(record["target"])
        if str(record.get("body", "")).startswith("running ") and "protocol" in str(record.get("body")):
            protocols[record["body"]] = protocols.get(record["body"], 0) + 1

    bench = json.loads((output / "bench.json").read_text())
    completed = int(bench.get("completed") or 0)
    if len(handled) != completed:
        problems.append(f"the EPP handled {len(handled)} completion request(s), the client completed {completed}")
    expected = SIDECAR_PROTOCOL[run.server.connector]
    if protocols != {expected: len(handled)}:
        problems.append(f"the sidecar ran {protocols}, not {expected!r} once per completion ({len(handled)})")

    # The gateway's view: every request Envoy served, by path and status.
    accesses = [r for log in sorted(directory.glob("*-envoy.log")) for r in _json_lines(log)
                if r.get("access") == "request"]
    completions: dict[str, list[dict]] = {}
    for access in accesses:
        if access.get("method") == "POST" and str(access.get("path", "")).startswith(
                ("/v1/completions", "/v1/chat/completions")):
            completions.setdefault(access.get("request_id"), []).append(access)
    for request, records in completions.items():
        if len(records) > 1:
            problems.append(f"Envoy logged completion {request} {len(records)} times "
                            f"({', '.join(str(r.get('status')) for r in records)})")
    ok = sum(1 for records in completions.values() for r in records if str(r.get("status")) == "200")
    gateway = {"completions": len(completions), "completions_ok": ok,
               "others": sorted({f"{r.get('method')} {r.get('path')} {r.get('status')}" for r in accesses
                                 if r.get("request_id") not in completions})}
    if set(completions) != set(handled):
        problems.append(f"Envoy served {len(completions)} completion(s), the EPP handled {len(handled)} "
                        f"(differing: {sorted(set(completions) ^ set(handled))[:2]})")
    if ok != completed:
        problems.append(f"Envoy answered {ok} completion(s) with 200, the client completed {completed}")

    instances = instance_steps(output, run)
    engines = {name: _engine_requests(texts) for name, (_, texts) in instances.items()}
    by_address = {address: e["name"] for address, e in named.items()}
    owners: dict[str, dict[str, set[str]]] = {"prefill": {}, "decode": {}}
    for name, (role, _) in instances.items():
        kinds, full = engines[name]
        for request in kinds.get("admitted", set()):
            owners[role].setdefault(request, set()).add(name)
        for request, ids in full.items():
            if len(ids) > 1:
                problems.append(f"{name} ran {request} as {len(ids)} engine requests: {sorted(ids)[:2]}")
        for request in sorted(kinds.get("admitted", set()) - set(handled)):
            problems.append(f"{role} {name} ran {request}, which the EPP never routed")

    chosen: dict[str, dict[str, int]] = {"prefill": {}, "decode": {}}
    prefill_pick: dict[str, str] = {}
    pairs: dict[str, int] = {}
    released = received = 0
    for request, records in sorted(handled.items()):
        record = records[0]
        picked: dict[str, list[str]] = {}
        for targets in picks.get(request, []):
            if len(targets) != 1:
                problems.append(f"{request}: the EPP's picker targeted {targets}, not one endpoint")
            for target in targets:
                picked.setdefault(role_of.get(target, "unknown"), []).append(target)
        if set(picked) - {"prefill", "decode"}:
            problems.append(f"{request}: the EPP picked endpoints of no known role: {picked}")
        choice: dict[str, str] = {}
        for role in ("prefill", "decode"):
            targets = picked.get(role, [])
            if len(targets) != 1:
                problems.append(f"{request}: the EPP picked {len(targets)} {role} endpoint(s) {targets}, not one")
            else:
                choice[role] = targets[0]
        if "decode" in choice and record.get("endpoint") != choice["decode"]:
            problems.append(f"{request}: the EPP handled it at {record.get('endpoint')}, its picker chose "
                            f"{choice['decode']}")
        for access in completions.get(request, []):
            if "decode" in choice and access.get("upstream") != choice["decode"]:
                problems.append(f"{request}: Envoy sent it to {access.get('upstream')}, the EPP picked "
                                f"{choice['decode']}")
        targets = dispatched.get(record.get("trace_id") or "", [])
        if len(targets) != 1:
            problems.append(f"{request}: the sidecar dispatched its prefill {len(targets)} times {targets}")
        elif "prefill" in choice and targets[0] != choice["prefill"]:
            problems.append(f"{request}: the sidecar sent its prefill to {targets[0]}, the EPP picked "
                            f"{choice['prefill']}")
        if "prefill" in choice:
            prefill_pick[request] = choice["prefill"]
        if set(choice) == {"prefill", "decode"}:
            pair = f"{by_address[choice['prefill']]}->{by_address[choice['decode']]}"
            pairs[pair] = pairs.get(pair, 0) + 1
        for role, address in choice.items():
            name = by_address[address]
            chosen[role][name] = chosen[role].get(name, 0) + 1
            ran = owners[role].get(request, set())
            if ran != {name}:
                problems.append(f"{request}: the EPP picked {role} {name}, it ran on {sorted(ran) or 'none'}")
        if any(request in engines[name][0].get(PRODUCER_DONE, set()) for name in owners["prefill"].get(request, ())):
            released += 1
        else:
            problems.append(f"{request}: prefill never released it ({PRODUCER_DONE})")
        if any(request in engines[name][0].get(CONSUMER_DONE, set()) for name in owners["decode"].get(request, ())):
            received += 1
        else:
            problems.append(f"{request}: decode never received its KV ({CONSUMER_DONE})")
    saved_config = yaml.safe_load((directory / "epp-config.yaml").read_text())
    plugins = saved_config["plugins"]
    if run.router.epp_plugins is None and saved_config != epp_config(run.router, run.server.block_size or 16):
        # A run without its own plugins file ran the default ones, with its
        # router's settings (its calibrated throughput).
        problems.append("the saved EPP config is not the one the run's router settings give")
    # The engines' metrics as the EPP used them are in its max-score picker's
    # records: a configuration without that picker leaves no such evidence.
    scoring = any(p.get("type") == "max-score-picker" for p in plugins)
    for address, endpoint in named.items() if scoring else ():
        seen = scraped.get(endpoint["name"])
        if not seen or not seen.get("max_CacheNumBlocks"):
            problems.append(f"the EPP never scored {endpoint['name']} with its engine's metrics "
                            f"(cache blocks) from {address}")
    affinity = [p for p in plugins if p.get("type") == "prefix-cache-affinity-filter"]
    metrics = directory / "epp-metrics.txt"
    decisions = affinity_decisions(metrics.read_text(errors="replace")) if affinity and metrics.exists() else None
    throughputs = [(p.get("parameters") or {}).get("peakPrefillThroughput") for p in affinity]
    precise = None
    if any(p.get("type") == "precise-prefix-cache-producer" for p in plugins):
        precise = precise_index_evidence(
            directory, named, set(handled), prefill_pick,
            {name: texts for name, (role, texts) in instances.items() if role == "prefill"},
            next((p.get("parameters") or {}).get("tokenProcessorConfig", {}).get("blockSizeTokens")
                 for p in plugins if p.get("type") == "precise-prefix-cache-producer"),
            run.workload.prefix_repetition is not None)
        problems.extend(precise.pop("problems"))
    return {"problems": pd._capped(problems, 5), "evidence": {
        "completion_requests": len(handled), "other_requests": len(others), "gateway": gateway,
        "sidecar_protocols": protocols, "routed": chosen, "pairs": dict(sorted(pairs.items())),
        "idle_instances": sorted(name for role in ("prefill", "decode")
                                 for name in (e["name"] for e in named.values()
                                              if e["labels"].get(ROLE_LABEL) == role)
                                 if name not in chosen[role]),
        "metrics_seen_by_epp": scraped if scoring else None,
        "plugins": [p.get("type") for p in plugins],
        "producer_released": released, "consumer_received": received,
        # The llm-d guide's value (or the filter's own default, no value
        # given) is another fleet's.
        "peak_prefill_throughput": throughputs or None,
        # What the affinity filter did (the EPP's own counter, read after
        # the workload): only no_match means it never narrowed a choice, so
        # the run did not exercise its stickiness or its load gate.
        "affinity_filter_decisions": decisions,
        # The precise prefix index against the engines (null: an approximate
        # or no prefix signal).
        "precise_index": precise,
        "uncalibrated": {"prefix-cache-affinity-filter.peakPrefillThroughput": UNCALIBRATED_PEAK_PREFILL_THROUGHPUT}
        if any(t in (None, UNCALIBRATED_PEAK_PREFILL_THROUGHPUT) for t in throughputs) else {},
    }}


INDEX_METRIC = re.compile(r"^llm_d_epp_kv_cache_index_(?P<name>[a-z_]+)_total\s+(?P<count>[0-9.eE+]+)\s*$")


def _count(value: object) -> int | None:
    """A nonnegative integer count, else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def reusable_tokens(blocks: int, prompt: int, block_tokens: int) -> int:
    """The tokens of ``blocks`` cached blocks a prompt of ``prompt`` tokens
    can reuse: vLLM recomputes at least the last token for its logits, and
    in whole blocks (``KVCacheManager.get_computed_blocks``: at most
    ``prompt - 1`` tokens, rounded down to a block)."""
    return min(blocks * block_tokens, (max(prompt - 1, 0) // block_tokens) * block_tokens)


def precise_index_evidence(directory: Path, named: dict[str, dict], completions: set[str],
                           prefill_pick: dict[str, str], prefill_steps: dict[str, list[str]],
                           block_tokens: object, reuse_workload: bool = False) -> dict:
    """llm-d's precise prefix index against the engines it indexes, from the
    EPP's log (at TRACE: each request's lookup) and metrics and the prefill
    instances' step logs.

    - The EPP subscribed to every endpoint's KV-cache events at its control
      address (so file discovery fed the subscribers), and events reached
      the index.
    - Whole-block token-count consistency: each completion's lookup has as
      many block keys as the engine's prompt has whole blocks. This does not
      compare token ids, nor token counts within a partial block.
    - Each lookup's per-endpoint counts are nonnegative integers no larger
      than the request's block count; anything else is a problem, whatever
      the index evicted.
    - At the prefill instance the picker chose, the tokens the index's match
      lets the engine reuse (`reusable_tokens`: never the last token, whole
      blocks) are compared with what the engine's first step found cached.
      Cold comparisons (neither has anything) are counted apart from warm
      ones. A warm comparison is ``exact``, ``under`` (the engine had more:
      an event not yet indexed) or ``over`` (the index claimed blocks the
      engine did not reuse). Over-counting is a problem when the index saw
      no eviction; with evictions it is reported as unresolved (a global
      eviction counter does not establish its cause).
    - With a workload built for reuse (``reuse_workload``), warm comparisons
      must exist and most of them must agree; otherwise warm agreement is
      reported as unexercised."""

    problems: list[str] = []
    if _count(block_tokens) is None or not block_tokens:
        return {"problems": [f"the precise index's block size is {block_tokens!r}"]}
    subscribers: dict[str, str] = {}
    failures = 0
    produced: dict[str, list[dict]] = {}
    for record in (r for log in sorted(directory.glob("*-epp.log")) for r in _json_lines(log)):
        body = record.get("body")
        if body == "Ensured KV-events subscriber":
            subscribers[str(record.get("endpoint", "")).rsplit("/", 1)[-1]] = str(record.get("zmq"))
        elif body == "Failed to connect subscriber socket":
            failures += 1
        elif body == "Produce completed" and record.get("x-request-id"):
            produced.setdefault(record["x-request-id"], []).append(record)
    for address, endpoint in named.items():
        expected = f"tcp://{endpoint['address']}:{pd.KV_EVENTS_PORT}"
        if subscribers.get(endpoint["name"]) != expected:
            problems.append(f"the EPP did not subscribe to {endpoint['name']}'s KV-cache events at {expected} "
                            f"(it had {subscribers.get(endpoint['name'])!r})")
    index: dict[str, int] = {}
    metrics = directory / "epp-metrics.txt"
    for line in metrics.read_text(errors="replace").splitlines() if metrics.exists() else ():
        if match := INDEX_METRIC.match(line):
            index[match["name"]] = int(float(match["count"]))
    if not index.get("admissions"):
        problems.append(f"the precise index admitted no block (its metrics: {index or 'none'}): no KV-cache event "
                        "reached it")
    # Each request on its prefill instance: its prompt's tokens and what its
    # first step found cached.
    engine: dict[str, dict[str, tuple[int, int]]] = {}
    for name, texts in prefill_steps.items():
        for text in texts:
            for record in pd._records(text):
                step = record.get("step")
                for request in (step.get("requests") or []) if isinstance(step, dict) else []:
                    match = ENGINE_REQUEST_ID.match(str(request.get("req_id")))
                    if match:
                        engine.setdefault(match["id"], {}).setdefault(name, (
                            int(request.get("num_prompt_tokens") or 0), int(request.get("num_cached_tokens") or 0)))
    by_address = {address: e["name"] for address, e in named.items()}
    cold = 0
    warm = {"exact": 0, "under": 0, "over": 0}
    raw_blocks: list[int] = []
    for request in sorted(completions):
        records = produced.get(request, [])
        if len(records) != 1:
            problems.append(f"{request}: the precise index logged {len(records)} lookup(s), not one (its "
                            "records need router.epp_verbosity 5)")
            continue
        picked = prefill_pick.get(request)
        found = engine.get(request, {}).get(by_address.get(picked, ""))
        if found is None:
            problems.append(f"{request}: no step of it on the prefill instance the EPP picked ({picked})")
            continue
        prompt, cached = found
        keys = _count(records[0].get("blockKeys"))
        if keys != prompt // block_tokens:
            problems.append(f"{request}: the EPP's lookup has {records[0].get('blockKeys')!r} block key(s) of "
                            f"{block_tokens} tokens, the engine's prompt {prompt} tokens ({prompt // block_tokens} "
                            "whole blocks)")
            continue
        matches = records[0].get("matches")
        if not isinstance(matches, dict):
            problems.append(f"{request}: the EPP's lookup has no per-endpoint matches ({matches!r})")
            continue
        invalid = {endpoint: (match or {}).get("MatchedBlocks") if isinstance(match, dict) else match
                   for endpoint, match in matches.items()
                   if not isinstance(match, dict) or _count(match.get("MatchedBlocks")) is None
                   or match["MatchedBlocks"] > keys}
        if invalid:
            problems.append(f"{request}: the EPP's lookup matched {invalid} block(s), not a count of at most its "
                            f"{keys} block key(s)")
            continue
        blocks = (matches.get(picked) or {}).get("MatchedBlocks") or 0
        raw_blocks.append(blocks)
        reusable = reusable_tokens(blocks, prompt, block_tokens)
        if reusable == 0 and cached == 0:
            cold += 1
        else:
            warm["exact" if reusable == cached else "under" if reusable < cached else "over"] += 1
    if warm["over"] and not index.get("evictions"):
        problems.append(f"the precise index held more than the engine reused for {warm['over']} request(s) at "
                        "their prefill instance, with no eviction in the index")
    warm_compared = sum(warm.values())
    if reuse_workload:
        if not warm_compared:
            problems.append("no request of this reuse workload found a cached prefix on its prefill instance or "
                            "in the index: warm agreement was not exercised")
        elif warm["exact"] * 2 <= warm_compared:
            problems.append(f"the precise index agreed with the engine on {warm['exact']} of {warm_compared} warm "
                            "lookup(s), not most")
    return {"problems": pd._capped(problems, 5), "block_tokens": block_tokens, "subscribers": subscribers,
            "subscriber_connect_failures": failures, "index": index,
            "lookups_compared": cold + warm_compared, "cold_exact": cold, "warm": warm,
            "warm_validation": "exercised" if warm_compared else "unexercised",
            # Over-counts the index's evictions may or may not explain.
            "unresolved_over": warm["over"] if index.get("evictions") else 0,
            "index_blocks_at_pick": {"total": sum(raw_blocks), "max": max(raw_blocks, default=0)}}


AFFINITY_DECISIONS = re.compile(
    r'^llm_d_epp_prefix_cache_affinity_filter_decisions_total\{(?P<labels>[^}]*)\}\s+(?P<count>[0-9.eE+]+)\s*$')


def affinity_decisions(metrics: str) -> dict[str, int]:
    """The prefix-cache affinity filter's decisions by outcome, from the
    EPP's metrics text (summed over its instances)."""
    found: dict[str, int] = {}
    for line in metrics.splitlines():
        match = AFFINITY_DECISIONS.match(line)
        outcome = re.search(r'outcome="([^"]*)"', match["labels"]) if match else None
        if outcome:
            found[outcome[1]] = found.get(outcome[1], 0) + int(float(match["count"]))
    return found


def recheck(output: Path, run: pd.Run, verdict: dict) -> None:
    """A run's routing evidence, the router components' placement and their
    cleanup, checked into ``verdict``."""
    evidence = routing_evidence(output, run)
    cleanup = output / "llmd" / "cleanup.json"
    if cleanup.exists():
        evidence["evidence"]["cleanup"] = json.loads(cleanup.read_text())
        if evidence["evidence"]["cleanup"]["remaining"]:
            evidence["problems"].append(f"llm-d containers left after the run: "
                                        f"{evidence['evidence']['cleanup']['remaining']}")
    placed = output / "llmd" / "components.json"
    if placed.exists():
        saved = json.loads(placed.read_text())
        evidence["evidence"]["components"] = saved
        evidence["problems"].extend(placement_problems(saved, run))
    verdict["llmd"] = evidence["evidence"]
    verdict["problems"].extend(evidence["problems"])
