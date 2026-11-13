"""A prefill/decode run of the vLLM simulator in an up topology.

Prefill and decode each run ``vllm serve`` with simulated compute and real
KV transfers through MooncakeConnector (prefill writes), or NixlConnector
over its Mooncake, UCX/TCP or LIBFABRIC/TCP backend (decode reads; ``server.connector: nixl``).
The Mooncake paths use TENT's hp_tcp only; UCX and LIBFABRIC use each rank's resolved TCP
interface. NIXL's handshake uses the KV network too: one side channel per DP
engine, on rank 0's network, at ports of their own. HTTP and Mooncake's
bootstrap use the control network. Requests enter through llm-d's router
(`harness.llmd`: Envoy and the endpoint picker on the run's gateway node, a
disaggregation sidecar on each decode node); the client node runs ``vllm
bench serve`` against the gateway.

With tensor parallelism, every rank moves its own KV heads: on the side's KV
network, or on its own one (``kv_network`` a list, one per rank, the same
on both sides), and logs into its own step log (``steps.tp<r>.jsonl``);
each Mooncake/TENT rank serves metrics on its own port.

The run is then checked from what the servers and routers recorded: every
request completed, decode compared every byte it received with prefill's
pattern (``kv_check``) on every rank, the connector failed no transfer and
moved the bytes decode checked, each Mooncake rank's TENT saw its connector's
bytes (TENT is not applicable to UCX/LIBFABRIC), and the KV crossed its rank's network's router, not another's or the
control one's. A model of several KV cache groups (full and sliding-window
attention) is checked group by group against the rank's layout, a window's
pulls holding its last blocks only; the verdict keeps each group's evidence
(blocks and bytes checked, pulls clipped to the window, null places and
reused blocks in its allocations).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import json
import math
from pathlib import Path
import time

import yaml

from . import models, shell, topology


E2E = Path(__file__).resolve().parent.parent
REPOSITORY = E2E.parent.parent
CONNECTORS = ("mooncake", "nixl")
RESULTS = E2E / "results"
VLLM = "/opt/venv/bin/vllm"
PYTHON = "/opt/venv/bin/python"
MODEL_DIRECTORY = "/model"
MODEL_NAME = "sim-pd"
SERVER_PORT = 8000
BOOTSTRAP_PORT = 8998
HTTP_KEEP_ALIVE = 120  # seconds: the servers' HTTP keep-alive
# A server's KV-cache events (ZMQ PUB), for the router's precise prefix index,
# and their replay (ZMQ ROUTER). DP rank r publishes on each port plus r (vLLM
# offsets them; llm-d dials them by the endpoint's rank index), so the two
# are further apart than an instance has ranks.
KV_EVENTS_PORT = 5557
KV_EVENTS_REPLAY_PORT = 5657
PREFIX_INDEXES = ("approximate", "precise")
DP_ROUTINGS = ("instance", "rank")
DP_SUPERVISOR_PORT = 9256  # a per-rank instance's aggregated health (vLLM's default)
WORK = "/e2e"
STEP_LOG = f"{WORK}/steps.jsonl"
TENT_METRICS_PORT = 9100  # rank r of the host serves on TENT_METRICS_PORT + r
DISABLED_TENT_TRANSPORTS = ("tcp", "rdma", "shm", "gds", "ascend_direct")


@dataclass
class Side:
    node: str | list[str]
    """The instance's node; a list: several instances of this role, one per
    node, each one engine (data_parallel_size 1) of the side's TP size, a
    front end (llm-d's router) choosing among them. Each is then checked as
    its own owner of requests, as a DP rank is (`ranks`: the instance's
    index in place of the DP rank)."""
    # A list: one per tensor-parallel rank, which every DP rank's TP rank of
    # that index uses (any prefill DP rank may pair with any decode DP rank).
    kv_network: str | list[str]
    # > 1: an MoE model's data-parallel attention with expert parallelism.
    data_parallel_size: int = 1
    tensor_parallel_size: int | None = None
    """This instance's TP size; None: the server's. Unequal sizes (one
    dividing the other) re-shard KV heads between ranks: both instances on
    one switched network (``kv_network`` its name), rank r on its node's
    link r."""

    decode_context_parallel_size: int = 1
    """This instance's decode context parallelism (--decode-context-parallel-size):
    1 or its TP size, as vLLM requires over NIXL; a dense MLA model at equal
    TP sizes over NIXL, prefix caching off (`_check_dcp`). A rank's blocks
    are then global pages, DCP rank r every DCP-th from r (`_dcp_audit`)."""

    def tp(self, server: "Server") -> int:
        return self.tensor_parallel_size or server.tensor_parallel_size

    def rank_networks(self, ranks: int) -> list[str]:
        if isinstance(self.kv_network, str):
            return [self.kv_network] * ranks
        return list(self.kv_network)

    def ranks(self, tp_size: int) -> list[tuple[int, int]]:
        """(DP rank, TP rank) of each of the instance's ranks, in the host's
        order (rank_addresses, TENT metrics ports, step logs); of several
        instances, (instance index, TP rank)."""
        groups = len(self.nodes) if self.instances else self.data_parallel_size
        return [(dp, tp) for dp in range(groups) for tp in range(tp_size)]

    @property
    def nodes(self) -> list[str]:
        return [self.node] if isinstance(self.node, str) else list(self.node)

    @property
    def instances(self) -> bool:
        """Several instances (``node`` a list)."""
        return not isinstance(self.node, str)

    def node_of(self, dp: int) -> str:
        """The node of `ranks`' DP rank (or instance) ``dp``."""
        return self.nodes[dp] if self.instances else self.node

    def local_rank(self, dp: int, tp: int, tp_size: int) -> int:
        """A rank's index on its host (TENT metrics port offset)."""
        return tp if self.instances else dp * tp_size + tp

    def step_log(self, role: str, dp: int, tp: int) -> str:
        """A rank's step log's name in the results: its role, its node, its
        rank suffix on that node."""
        return f"{role}-{self.node_of(dp)}-steps{_rank_suffix(0 if self.instances else dp, tp)}.jsonl"

    def remote_step_log(self, dp: int, tp: int) -> str:
        """A rank's step log on its node."""
        return f"{WORK}/steps{_rank_suffix(0 if self.instances else dp, tp)}.jsonl"


@dataclass
class Server:
    gpu: str = "h100"
    block_size: int | None = 16
    """Tokens per KV block; None: the platform's (a hybrid model's)."""
    enable_prefix_caching: bool | None = None
    """None: off with kv_check, else vLLM's default (on). On with kv_check,
    a block's pattern is its content's, and prefix hits are checked too."""
    kv_cache_memory_bytes: int = 2 * 1024 ** 3
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    """The KV cache dtype (--kv-cache-dtype, e.g. fp8); auto: the model's."""
    quantization: str | None = None
    """Weights quantized as the model loads (--quantization, e.g. fp8); None:
    as published."""
    language_model_only: bool = False
    """A multimodal model served as its language model (--language-model-only):
    no modality, the encoders not built."""
    trust_remote_code: bool = False
    """Run the model repository's own code (--trust-remote-code, for the
    servers and the benchmark's tokenizer): Kimi's tokenizer is such code.
    Off unless a spec asks."""
    hp_tcp_args: dict = field(default_factory=dict)
    """TENT's hp_tcp transport settings (MC_TENT_CONF's transports.hp_tcp),
    e.g. {max_outstanding_tasks: 16384}. Admission, in the pinned TENT with
    hp_tcp alone and its runtime queue off: a submitted transfer's requests,
    after TENT merges those contiguous at both ends in the same
    registrations, are one task each, reserved together or not at all,
    without waiting, against the tasks still outstanding on that rank's
    transport (all its transfers in flight; each task frees its slot as it
    completes). Mooncake's Python binding retries a refused transfer once,
    at once. A transfer of more tasks than max_outstanding_tasks (4096 by
    default) is always refused; a smaller one may be, while others' tasks are
    outstanding: a Mooncake run whose unmerged one-request estimate is
    larger is refused before it starts (`_check_write_tasks`). A connector's
    descriptors are not those tasks (TENT may merge them), and staying under
    the task limit leaves the byte limits
    (max_outstanding_bytes, max_transfer_bytes) and the transport's health
    to pass too. hp_tcp sends protocol headers and payloads separately. Small
    head-sliced tasks may be sensitive to Nagle/delayed-ACK interactions;
    task size relative to the effective TCP MSS, and therefore MTU, can
    affect transfer time."""
    env: dict = field(default_factory=dict)
    timing: str | None = None
    """The simulator's timing provider (zero, linear, table, replay,
    aisimulate or a module.Class), required for a new run; None only in a
    saved run from before it was (timed by the former roofline provider)."""
    timing_args: dict | None = None
    """Its arguments; aisimulate's default to {gpu} (plus a target_model)."""
    connector: str = "mooncake"
    """The KV connector: mooncake (MooncakeConnector, prefill writes) or nixl
    (NixlConnector, decode reads). nixl_backend selects Mooncake/TENT, UCX/TCP or LIBFABRIC/TCP."""
    nixl_backend: str = "mooncake"
    """NIXL backend: mooncake (default), ucx or libfabric over CPU TCP."""
    speculative: dict | None = None
    """Native MTP: {method: mtp, num_speculative_tokens: k (1 to 5),
    acceptance_rates: [...], seed: 0}. Both instances take the same
    speculative config: NIXL's handshake checks their transfer
    compatibility, the draft among it; Mooncake does not (the simulator
    takes it for DeepSeek-V3, DeepSeek-V3.2 and GLM-5). The rates are the
    simulator's per-position conditional acceptance probabilities (the last
    repeats), sampled with the seed: a run input, not a model's."""


@dataclass
class Workload:
    input_len: int = 1024
    output_len: int = 16
    num_prompts: int = 40
    concurrency: int = 4
    seed: int = 0
    prefix_repetition: dict | None = None
    """Prompts sharing prefixes (the benchmark's prefix_repetition dataset):
    {prefix_len, suffix_len, num_prefixes}, input_len their sum. The text is
    tokenized again by the server: the sharing the checks report is the
    servers' own (their prefix hits), not this nominal split."""


PREFIX_REPETITION_KEYS = ("prefix_len", "suffix_len", "num_prefixes")


@dataclass
class Router:
    """llm-d's router, every run's front end (`harness.llmd`)."""
    gateway: str
    """The node Envoy and the endpoint picker (EPP) run in, one of its own:
    not the client, not a server's. Its control network address is the
    client's base URL."""
    epp_plugins: Path | None = None
    """An EndpointPickerConfig's plugins and schedulingProfiles (YAML), in
    place of `harness.llmd.default_epp_plugins`; the endpoints file and the
    metrics source are added to it either way."""
    epp_verbosity: int = 4
    sidecar_verbosity: int = 4
    prefix_index: str = "approximate"
    """Where the default plugins' prefix signal comes from. ``approximate``:
    the router's own estimate from the requests it sent (llm-d's P/D
    guide). ``precise``: the servers' KV-cache events — every server
    publishes them over ZMQ on the control network and serves the render
    endpoints, the EPP tokenizes each prompt through one of them and keeps
    an index of the blocks each instance holds (llm-d's precise P/D
    configuration)."""
    dp_routing: str = "instance"
    """Who picks the DP rank of an instance with several. ``instance``: the
    instance is one endpoint and its ``vllm serve`` balances its ranks
    (under Mooncake the sidecar names the prefill rank). ``rank``: llm-d's
    per-rank routing — the instance runs one API server per DP rank on
    consecutive ports (vLLM's multi-port external load balancing), each an
    endpoint of its own that the EPP picks; a decode instance's sidecar
    listens for each of its ranks."""
    peak_prefill_throughput: float | None = None
    """Tokens a second of one unloaded prefill chunk of
    ``max_num_batched_tokens`` on a prefill instance, as llm-d defines its
    prefix-cache affinity filter's ``peakPrefillThroughput`` (`harness
    peak-prefill` prints a saved run's). Here it is a timing input derived
    from the simulator's timing provider (the step's simulated cost), not a
    measured GPU or end-to-end throughput. It replaces the llm-d guide's
    value, calibrated for another fleet, in the default plugins; None keeps
    that one, which the verdict then reports as uncalibrated."""


@dataclass
class Run:
    name: str
    topology: str  # its name: the topology must be up
    model: Path  # the fetched copy (config and tokenizer) of the Hugging Face model the config names
    prefill: Side
    decode: Side
    client: str
    control_network: str
    router: Router
    kv_check: bool = True
    expect_fragmentation: bool = False
    """Require byte-checked transfers whose blocks are scattered on both
    instances (a tight KV budget's reuse); the counts are reported anyway."""
    expect_prefill_hit_requests: int | None = None
    """Require at least this many requests to hit a cached prefix on their
    prefill instance (prefix-aware routing keeps a prefix on the instances
    that hold it); the counts are reported anyway (``prefix_routing``)."""
    expect_decode_prefix_hits: bool = True
    """Whether decode is expected to reuse cached prefixes of its own in a
    byte-checked run with prefix caching. False for a model whose pulls
    leave decode nothing a later prompt can hit (sliding windows, of which
    only the last blocks are pulled): decode must then show no hit at all,
    so a surprise is a problem either way."""
    server: Server = field(default_factory=Server)
    workload: Workload = field(default_factory=Workload)
    startup_timeout: float = 600.0


def load(path: str | Path, *, fetch: bool = False) -> Run:
    """A run config; relative paths are the file's. ``fetch``: download a
    model named by its Hugging Face id if it is not here yet."""

    path = Path(path).resolve()
    return parse(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, path, fetch=fetch)


def parse(raw: dict, path: Path, *, fetch: bool = False) -> Run:
    """A run config's contents, read from ``path`` (relative paths are its)."""

    known = {"name", "topology", "model", "prefill", "decode", "client", "control_network", "kv_check",
             "expect_fragmentation", "expect_prefill_hit_requests", "expect_decode_prefix_hits", "server", "workload", "router",
             "startup_timeout"}
    if unknown := set(raw) - known:
        raise ValueError(f"unknown run key(s): {', '.join(sorted(unknown))}")
    router_raw = raw.get("router")
    if not isinstance(router_raw, dict) or not isinstance(router_raw.get("gateway"), str):
        raise ValueError("run config needs router.gateway (the node llm-d's Envoy and EPP run in)")
    try:
        router = Router(**router_raw)
    except TypeError as error:
        raise ValueError(f"router: {error}") from None
    if router.epp_plugins is not None:
        router.epp_plugins = (path.parent / router.epp_plugins).resolve()
    try:
        run = Run(
            name=raw["name"], topology=raw["topology"],
            model=models.resolve(str(raw["model"]), fetch=fetch),
            prefill=Side(**raw["prefill"]), decode=Side(**raw["decode"]),
            client=raw["client"], control_network=raw["control_network"], router=router,
            kv_check=bool(raw.get("kv_check", True)),
            expect_fragmentation=raw.get("expect_fragmentation", False),
            expect_prefill_hit_requests=raw.get("expect_prefill_hit_requests"),
            expect_decode_prefix_hits=raw.get("expect_decode_prefix_hits", True),
            server=Server(**(raw.get("server") or {})), workload=Workload(**(raw.get("workload") or {})),
            startup_timeout=float(raw.get("startup_timeout", 600)),
        )
    except KeyError as missing:
        raise ValueError(f"run config needs {missing}") from None
    except TypeError as error:
        raise ValueError(f"run config: {error}") from None
    if run.server.timing is None:
        raise ValueError("run config needs server.timing (zero, linear, table, replay, aisimulate or a "
                         "module.Class)")
    if not isinstance(run.server.trust_remote_code, bool):
        # YAML's "false" is a string: a true value in a flag's test.
        raise ValueError(f"server.trust_remote_code is true or false, not {run.server.trust_remote_code!r}")
    if run.server.connector not in CONNECTORS:
        raise ValueError(f"server.connector is one of {', '.join(CONNECTORS)}, not {run.server.connector!r}")
    _check_backend(run.server)
    if run.server.speculative is not None:
        _check_speculative(run.server)
    if not isinstance(run.expect_fragmentation, bool):
        raise ValueError(f"expect_fragmentation is true or false, not {run.expect_fragmentation!r}")
    if run.expect_fragmentation and not run.kv_check:
        raise ValueError("expect_fragmentation needs kv_check (it counts byte-checked transfers)")
    if not isinstance(run.expect_decode_prefix_hits, bool):
        raise ValueError(f"expect_decode_prefix_hits is true or false, not {run.expect_decode_prefix_hits!r}")
    if not run.expect_decode_prefix_hits and not (run.server.enable_prefix_caching and run.kv_check):
        raise ValueError("expect_decode_prefix_hits: false needs server.enable_prefix_caching and kv_check (it "
                         "counts byte-checked prefix hits)")
    expected_hits = run.expect_prefill_hit_requests
    if expected_hits is not None:
        if not isinstance(expected_hits, int) or isinstance(expected_hits, bool) or expected_hits < 1:
            raise ValueError(f"expect_prefill_hit_requests is a positive integer, not {expected_hits!r}")
        if not (run.server.enable_prefix_caching and run.kv_check):
            raise ValueError("expect_prefill_hit_requests needs server.enable_prefix_caching and kv_check (it "
                             "counts byte-checked prefix hits)")
        if expected_hits > run.workload.num_prompts:
            raise ValueError("expect_prefill_hit_requests is at most workload.num_prompts")
    throughput = run.router.peak_prefill_throughput
    if throughput is not None:
        if isinstance(throughput, bool) or not isinstance(throughput, (int, float)) or not (
                math.isfinite(throughput) and throughput > 0):
            raise ValueError(f"router.peak_prefill_throughput is a positive, finite number of tokens a second, not "
                             f"{throughput!r}")
        if run.router.epp_plugins is not None:
            raise ValueError("router.peak_prefill_throughput sets the default plugins' value: a run with "
                             "router.epp_plugins sets it in that file")
    repetition = run.workload.prefix_repetition
    if repetition is not None:
        if not isinstance(repetition, dict) or set(repetition) != set(PREFIX_REPETITION_KEYS) or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in repetition.values()):
            raise ValueError("workload.prefix_repetition takes positive prefix_len, suffix_len and num_prefixes")
        if repetition["prefix_len"] + repetition["suffix_len"] != run.workload.input_len:
            raise ValueError("workload.input_len is prefix_repetition's prefix_len + suffix_len")
    for side in (run.prefill, run.decode):
        if side.instances:
            if not side.nodes or not all(isinstance(n, str) for n in side.nodes) or len(set(side.nodes)) != len(
                    side.nodes):
                raise ValueError(f"a side's node list names distinct nodes, not {side.node!r}")
            if side.data_parallel_size != 1:
                raise ValueError("several instances of a side are one engine each (data_parallel_size 1)")
    if set(run.prefill.nodes) & set(run.decode.nodes):
        raise ValueError("prefill and decode instances are on distinct nodes")
    if run.router.gateway in (*run.prefill.nodes, *run.decode.nodes, run.client):
        raise ValueError("router.gateway is a node of its own, not the client or a server's")
    if run.router.dp_routing not in DP_ROUTINGS:
        raise ValueError(f"router.dp_routing is one of {', '.join(DP_ROUTINGS)}, not {run.router.dp_routing!r}")
    if run.router.dp_routing == "rank" and run.prefill.data_parallel_size == run.decode.data_parallel_size == 1:
        raise ValueError("router.dp_routing: rank needs a side with data_parallel_size > 1")
    if run.router.dp_routing == "rank" and run.server.connector != "nixl":
        # vLLM launches a Mooncake bootstrap server for each external-LB
        # prefill server (mooncake_connector.py's should_launch_bootstrap_
        # server), but llm-d v1.0's sidecar derives one bootstrap port from
        # the prefill host and picks a rank from that server's map
        # (connector_mooncake.go:49, 58-61), whichever endpoint the EPP
        # picked. Prefill DP 1 with rank-routed decode ranks is not that
        # case, and is not validated either.
        raise ValueError("router.dp_routing: rank needs server.connector: nixl (llm-d's sidecar selects one "
                         "Mooncake bootstrap server a prefill host, not one an endpoint)")
    if run.router.prefix_index not in PREFIX_INDEXES:
        raise ValueError(f"router.prefix_index is one of {', '.join(PREFIX_INDEXES)}, not "
                         f"{run.router.prefix_index!r}")
    if run.router.prefix_index == "precise":
        if run.router.epp_plugins is not None:
            raise ValueError("router.prefix_index selects the default plugins' prefix signal: a run with "
                             "router.epp_plugins configures it in that file")
        if not run.server.enable_prefix_caching:
            raise ValueError("router.prefix_index: precise needs server.enable_prefix_caching (the index is of "
                             "the servers' prefix caches)")
        if run.server.block_size is None:
            raise ValueError("router.prefix_index: precise needs server.block_size (the index's block size)")
        for role, side in (("prefill", run.prefill), ("decode", run.decode)):
            if side.data_parallel_size > 1 and not rank_routed(run, side):
                raise ValueError(f"router.prefix_index: precise needs router.dp_routing: rank for {role}'s "
                                 f"{side.data_parallel_size} DP ranks (each rank publishes its own events and "
                                 "holds its own cache; the index is of endpoints)")
        if run.router.epp_verbosity < 5:
            raise ValueError("router.prefix_index: precise needs router.epp_verbosity 5 (the index's lookups, "
                             "which the checks compare with the engines, are logged at that level)")
    if run.prefill.instances or run.decode.instances:
        if run.prefill.tp(run.server) != run.decode.tp(run.server):
            raise ValueError("several instances of a side take equal TP sizes")
        if run.server.speculative is not None:
            raise ValueError("several instances of a side take no speculative decoding")
    for side in (run.prefill, run.decode):
        size = side.tensor_parallel_size
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 1):
            raise ValueError(f"{side.nodes}: tensor_parallel_size is a positive integer")
        if not isinstance(side.kv_network, str) and len(side.kv_network) != side.tp(run.server):
            raise ValueError(f"{side.nodes}: kv_network needs one network per rank ({side.tp(run.server)})")
    prefill_tp, decode_tp = run.prefill.tp(run.server), run.decode.tp(run.server)
    if prefill_tp == decode_tp:
        if run.prefill.rank_networks(prefill_tp) != run.decode.rank_networks(decode_tp):
            # Each network routes only its own subnet.
            raise ValueError("prefill and decode rank r need the same KV network")
    else:
        if max(prefill_tp, decode_tp) % min(prefill_tp, decode_tp):
            raise ValueError(f"unequal TP sizes must divide one another, not {prefill_tp} and {decode_tp}")
        if not (isinstance(run.prefill.kv_network, str) and run.prefill.kv_network == run.decode.kv_network):
            raise ValueError("unequal TP sizes need one switched network for both instances (kv_network its "
                             "name, a link per rank)")
        if run.prefill.data_parallel_size > 1 or run.decode.data_parallel_size > 1:
            raise ValueError("unequal TP sizes without data parallelism")
    _check_dcp(run)
    if run.server.connector == "mooncake":
        _check_write_tasks(run)
    return run


def _decode_context_parallel_flags(side: Side) -> list[str]:
    return (["--decode-context-parallel-size", str(side.decode_context_parallel_size)]
            if side.decode_context_parallel_size > 1 else [])


def plain_mla_of(model: Path) -> bool:
    """A dense MLA model (one latent a layer, `kv_lora_rank`): not sparse MLA
    (an indexer, `index_topk`), not DeepSeek-V4's caches, not a hybrid."""
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    sources = (config, config.get("text_config") or {})
    return (any(source.get("kv_lora_rank") for source in sources)
            and not any("index_topk" in source for source in sources)
            and not any(source.get("model_type") in ("deepseek_v4", "kimi_k3", "kimi_linear")
                        for source in sources))


def _check_dcp(run: Run) -> None:
    """Decode context parallelism on either side: each side's DCP 1 or its TP
    size (vLLM's rule over NIXL), a dense MLA model over NIXL's pull
    connector, one instance a side without data parallelism, no speculative
    decoding, prefix caching on or off (one setting for both instances, as
    the byte checks need); at TP sizes one dividing the other, which,
    unequal, need one switched KV network (`parse`), as does a sharded
    producer read by a coarser consumer at equal sizes. The simulator checks
    the same scope at startup and at the handshake."""
    dcps = {role: side.decode_context_parallel_size for role, side in
            (("prefill", run.prefill), ("decode", run.decode))}
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        dcp = dcps[role]
        if not isinstance(dcp, int) or isinstance(dcp, bool) or dcp < 1:
            raise ValueError(f"{role}: decode_context_parallel_size is a positive integer")
        if dcp not in (1, side.tp(run.server)):
            raise ValueError(f"{role}: decode_context_parallel_size is 1 or the instance's TP size "
                             f"({side.tp(run.server)}) over NIXL, not {dcp}")
    if max(dcps.values()) == 1:
        return
    problems = []
    if run.server.connector != "nixl":
        problems.append(f"the {run.server.connector} connector (NIXL's pull connector only: Mooncake's "
                        "slices no page by DCP rank)")
    if run.prefill.instances or run.decode.instances or max(
            run.prefill.data_parallel_size, run.decode.data_parallel_size) > 1:
        problems.append("several instances or data parallelism")
    if run.server.speculative is not None:
        problems.append("speculative decoding")
    if not plain_mla_of(run.model):
        problems.append("the model's attention (a dense MLA model only)")
    if dcps["prefill"] > dcps["decode"] and not (
            isinstance(run.prefill.kv_network, str) and run.prefill.kv_network == run.decode.kv_network):
        # A decode rank reads every producer rank's slice (`dcp_sources`):
        # across ranks, as at unequal TP sizes.
        problems.append("a sharded producer read by a coarser consumer on per-rank KV networks (one switched "
                        "network for both instances: kv_network its name)")
    if problems:
        raise ValueError(f"decode context parallelism (prefill DCP {dcps['prefill']}, decode DCP "
                         f"{dcps['decode']}) is not supported with {', '.join(problems)}")


def _check_write_tasks(run: Run) -> None:
    """Refuses, before it starts, a Mooncake run whose unmerged estimate of
    one request's write (`request_write_tasks`) is above hp_tcp's task
    limit: a conservative policy. A write whose tasks, after the connector
    and TENT merge contiguous blocks, are above the limit is always refused,
    and the producer holds its blocks until their lease expires; the actual
    write can be much smaller, so a run whose blocks merge may be refused
    too. Several ready requests for one decode rank share a write, and
    writes overlap, as timing decides: not estimated here, the verdict
    reports them (`_hp_tcp_writes`)."""
    limit = hp_tcp_task_limit(run.server)
    bound = request_write_tasks(run)
    if limit is None or bound is None or bound[0] <= limit:
        return
    raise ValueError(f"one request's Mooncake write can be {bound[0]} hp_tcp tasks ({bound[1]}), above "
                     f"max_outstanding_tasks {limit}: a write of more tasks than that, once contiguous blocks "
                     f"merge, is always refused, and the producer holds its blocks until their lease expires "
                     f"(this estimate merges none: the run may merge enough). Raise it with server.hp_tcp_args: "
                     f"{{max_outstanding_tasks: N}} (N >= {bound[0]}), or shorten the prompt")


# These model types use the simulated CPU attention or dense MLA backend.
# Do not infer a startup layout for arbitrary model implementations.
_PLAIN_STARTUP_MODELS = (
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "minimax_m2", "deepseek_v2", "deepseek_v3",
)


def request_write_tasks(run: Run) -> tuple[int, str] | None:
    """An estimate of the tasks of one request's Mooncake write to one decode
    rank, merging no blocks, with its terms; None where it is not known
    before the run. An upper bound when the prompt is workload.input_len
    tokens; a prefix_repetition prompt can exceed it (the server tokenizes
    its joined text again: a block more seen), so there it is an estimate.
    connector copies each block of each registered region on its own
    (mooncake_connector.py:2066-2105), joining only blocks contiguous on both
    sides. TENT can merge more. A head slice changes a copy's length, not
    the count. Ordinary attention uses one region a layer (two for sparse
    MLA), including one native MTP layer when requested, whatever its depth.
    A known hybrid uses one raw page per state layer, separate from the
    attention blocks. Skip hybrid align mode and draft scratch, whose
    non-null state block counts are not inferred here. With no configured
    size, use 16 only for known ordinary attention or dense MLA; leave
    mixed layouts and startup-aligned hybrids unestimated."""
    block = run.server.block_size
    try:
        config = json.loads((Path(run.model) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    source = {**config, **(config.get("text_config") or config.get("llm_config") or {})}
    layers = source.get("num_hidden_layers")
    if type(layers) is not int or layers < 1:
        return None
    tokens = run.workload.input_len
    ordinary_dtype = run.server.kv_cache_dtype in ("auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2")
    if _has_states(source):
        hybrid = _hybrid_write_layers(source, layers)
        if (hybrid is None or type(block) is not int or block < 1 or not ordinary_dtype
                or _prefix_caching_enabled(run) is not False or run.server.speculative is not None):
            return None
        attention, state = hybrid
        # These hybrids use ordinary CPU attention or dense MLA: their
        # backends keep the aligned page whole. The platform may raise this
        # size, while with prefix caching off each state group has one page.
        blocks = -(-tokens // block)
        return attention * blocks + state, (f"{attention} attention layers x {blocks} blocks of {block} tokens "
                                            f"for {tokens} + {state} state layers x 1 page, none joined")
    startup = block is None
    if startup:
        if (source.get("model_type") not in _PLAIN_STARTUP_MODELS
                or source.get("sliding_window") or source.get("per_layer_config") or source.get("index_topk")
                or source.get("compress_ratios") or "sliding_attention" in (source.get("layer_types") or ())
                or not ordinary_dtype):
            return None
        # The simulated CPU attention and dense MLA backends accept whole
        # blocks in multiples of 16; the startup preference may be larger.
        block = 16
    if type(block) is not int or block < 1:
        return None
    regions = 2 if source.get("index_topk") else 1
    draft = 1 if run.server.speculative is not None else 0
    blocks = -(-tokens // block)
    named = f"{layers} layers" + (" + 1 MTP layer" if draft else "")
    size_note = " (startup lower bound)" if startup else ""
    return (layers + draft) * regions * blocks, (f"{named} x {regions} region(s) x {blocks} blocks of {block} tokens "
                                                 f"for {tokens}{size_note}, none joined")


def _hybrid_write_layers(config: dict, layers: int) -> tuple[int, int] | None:
    """Attention and state layers for hybrids with one region per page."""
    if any(config.get(key) for key in ("index_topk", "compress_ratios", "per_layer_config", "sliding_window")):
        return None
    pattern = config.get("hybrid_override_pattern")
    kinds = config.get("layer_types")
    linear = config.get("linear_attn_config")
    model_type = config.get("model_type")
    if (model_type == "nemotron_h" and isinstance(pattern, str) and len(pattern) == layers
            and set(pattern) <= set("M*E-")):
        attention, state = pattern.count("*"), pattern.count("M")
    elif (model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text")
          and isinstance(kinds, list) and len(kinds) == layers
          and all(kind in ("full_attention", "linear_attention") for kind in kinds)):
        attention, state = kinds.count("full_attention"), kinds.count("linear_attention")
    elif model_type in ("kimi_k3", "kimi_linear") and isinstance(linear, dict):
        kda = linear.get("kda_layers")
        if (not isinstance(kda, list) or not all(type(index) is int and 1 <= index <= layers for index in kda)
                or len(kda) != len(set(kda))):
            return None
        attention, state = layers - len(kda), len(kda)
    else:
        return None
    return (attention, state) if attention and state else None


def _has_states(config: dict) -> bool:
    """Whether a model config has recurrent-state (Mamba, linear attention)
    layers beside its attention."""
    kinds = set(config.get("layer_types") or []) | set(config.get("layers_block_type") or [])
    return bool(kinds - {"full_attention", "sliding_attention", "attention"}) or any(
        key in config for key in ("hybrid_override_pattern", "linear_attn_config", "ssm_state_size",
                                  "mamba_num_heads")) or 0 in (config.get("attn_type_list") or [])


SPECULATIVE_KEYS = {"method", "num_speculative_tokens", "acceptance_rates", "seed"}


def _check_speculative(server: Server) -> None:
    spec = server.speculative
    if not isinstance(spec, dict) or (unknown := set(spec) - SPECULATIVE_KEYS):
        raise ValueError(f"server.speculative takes {', '.join(sorted(SPECULATIVE_KEYS))}")
    k = spec.get("num_speculative_tokens")
    rates = spec.get("acceptance_rates")
    if spec.get("method") != "mtp" or not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 5:
        raise ValueError("server.speculative is native MTP (method mtp) of 1 to 5 tokens")
    if (not isinstance(rates, list) or not 1 <= len(rates) <= k
            or not all(isinstance(r, (int, float)) and not isinstance(r, bool) and 0 <= r <= 1 for r in rates)):
        raise ValueError(f"server.speculative.acceptance_rates: 1 to {k} probabilities in [0, 1]")
    if not isinstance(spec.get("seed", 0), int) or isinstance(spec.get("seed", 0), bool):
        raise ValueError("server.speculative.seed is an integer")


TENT_DEFAULT_MAX_OUTSTANDING_TASKS = 4096
"""hp_tcp's default max_outstanding_tasks in the TENT the image pins."""


def uses_ucx(server: Server) -> bool:
    return server.connector == "nixl" and server.nixl_backend == "ucx"


def uses_tent(server: Server) -> bool:
    return server.connector != "nixl" or server.nixl_backend == "mooncake"


def nixl_backend(server: Server) -> str:
    return {"mooncake": "Mooncake", "ucx": "UCX", "libfabric": "LIBFABRIC"}[server.nixl_backend]


def _check_backend(server: Server) -> None:
    if server.nixl_backend not in ("mooncake", "ucx", "libfabric"):
        raise ValueError("server.nixl_backend is one of mooncake, ucx, libfabric")
    if server.nixl_backend != "mooncake" and server.connector != "nixl":
        raise ValueError("UCX/LIBFABRIC needs server.connector: nixl")
    if not uses_tent(server):
        if server.hp_tcp_args or any(k.startswith(("MC_", "TENT_", "NIXL_MOONCAKE_")) for k in server.env):
            raise ValueError("UCX/LIBFABRIC does not accept TENT/Mooncake settings")
    if uses_ucx(server):
        if "UCX_TLS" in server.env and server.env["UCX_TLS"] != "tcp,self":
            raise ValueError("Managed UCX needs UCX_TLS=tcp,self")
        if any(k in server.env for k in ("UCX_NET_DEVICES", "UCX_CONFIG_FILE")):
            raise ValueError("UCX interface/config selection is managed per rank")
        if any(k.startswith("NIXL_UCX_") and k != "NIXL_UCX_WARNING_TIMEOUT" for k in server.env):
            raise ValueError("Managed UCX does not accept NIXL_UCX overrides")
    if server.nixl_backend == "libfabric":
        policy = {"FI_PROVIDER": "tcp", "FI_ADDR_FORMAT": "FI_SOCKADDR_IN", "FI_PREFER_SYSCONFIG": "0"}
        diagnostics = {"FI_LOG_LEVEL", "FI_LOG_PROV", "FI_LOG_SUBSYS"}
        for key, value in server.env.items():
            if key.startswith("NIXL_LIBFABRIC_") or (key.startswith("FI_") and key not in set(policy) | diagnostics):
                raise ValueError(f"Managed LIBFABRIC does not accept {key}")
            if key in policy and str(value) != policy[key]:
                raise ValueError(f"Managed LIBFABRIC needs {key}={policy[key]}")


def descriptor_limit(server: Server) -> int | None:
    return NIXL_MAX_DESCRIPTORS if uses_tent(server) else None


def hp_tcp_task_limit(server: Server) -> int | None:
    """The hp_tcp task limit the servers ran with: hp_tcp_args', or a
    server.env MC_TENT_CONF's that replaces the harness's; None if unknown
    (that config unreadable, or TENT not selected)."""

    def limit(hp_tcp: object) -> int | None:
        value = hp_tcp.get("max_outstanding_tasks", TENT_DEFAULT_MAX_OUTSTANDING_TASKS) if isinstance(
            hp_tcp, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if not uses_tent(server):
        return None
    env = {name: str(value) for name, value in server.env.items()}
    if env.get("MC_USE_TENT", "1") != "1":
        return None
    if "MC_TENT_CONF" in env:
        try:
            return limit(json.loads(env["MC_TENT_CONF"])["transports"]["hp_tcp"])
        except (ValueError, KeyError, TypeError):
            return None
    return limit(server.hp_tcp_args)


def tent_config(address: str, hp_tcp_args: dict) -> dict:
    """TENT with hp_tcp only, its listener, rails and control service at
    ``address``."""

    transports: dict = {name: {"enable": False} for name in DISABLED_TENT_TRANSPORTS}
    transports["hp_tcp"] = {"enable": True, "bind_address": address, "advertise_address": address,
                            "rail_addresses": [address], **hp_tcp_args}
    return {"rpc_server_hostname": address, "transports": transports}


def bench_command(run: Run, base_url: str) -> list[str]:
    """The client's benchmark: random prompts of the workload's lengths
    through the router's gateway at ``base_url``, tokenized with the
    model's tokenizer."""
    w = run.workload
    repetition = w.prefix_repetition
    dataset = ["--dataset-name", "random", "--random-input-len", str(w.input_len),
               "--random-output-len", str(w.output_len)] if repetition is None else [
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", str(repetition["prefix_len"]),
        "--prefix-repetition-suffix-len", str(repetition["suffix_len"]),
        "--prefix-repetition-num-prefixes", str(repetition["num_prefixes"]),
        "--prefix-repetition-output-len", str(w.output_len)]
    return [
        VLLM, "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
        "--base-url", base_url, "--model", MODEL_NAME, "--tokenizer", MODEL_DIRECTORY,
        *(["--trust-remote-code"] if run.server.trust_remote_code else []),
        *dataset, "--num-prompts", str(w.num_prompts),
        "--max-concurrency", str(w.concurrency), "--seed", str(w.seed), "--ignore-eos", "--disable-tqdm",
        "--save-result", "--save-detailed", "--result-dir", f"{WORK}/results", "--result-filename", "bench.json",
    ]


def rank_routed(run: Run, side: Side) -> bool:
    """The side's DP ranks are endpoints of their own (``router.dp_routing:
    rank``, several ranks)."""
    return run.router.dp_routing == "rank" and side.data_parallel_size > 1


def container_identities(container: dict[str, str]) -> dict[str, dict]:
    """Each node's container as Docker reports it: its image's id and name
    and its mounts (sources mounted over the image's would be here)."""
    found = {}
    for node, name in sorted(container.items()):
        inspected = json.loads(shell.run(["docker", "inspect", name]))[0]
        found[node] = {"container": name, "image_id": inspected["Image"], "image": inspected["Config"]["Image"],
                       "mounts": sorted(f"{m.get('Source')}:{m.get('Destination')}" for m in inspected["Mounts"])}
    return found


def kv_events_topic(control: str, endpoint_port: int) -> str:
    """The topic a server publishes its KV-cache events under, as llm-d's
    index reads it: the endpoint the router knows the instance by."""
    return f"kv@{control}:{endpoint_port}@{MODEL_NAME}"


def server_command(run: Run, role: str, kv_addresses: list[str], port: int = SERVER_PORT,
                   endpoint: tuple[str, int] | None = None) -> list[str]:
    """``vllm serve`` for ``role`` (prefill or decode) on HTTP ``port``, TP
    rank r's KV on ``kv_addresses[r]`` in every DP rank (their own:
    ``rank_addresses``, DP rank by TP rank). ``endpoint``: the control
    address and port the router knows the instance by (a precise prefix
    index names its events after it)."""

    _check_backend(run.server)
    ucx = uses_ucx(run.server)
    non_tent = not uses_tent(run.server)
    kv_address = kv_addresses[0]
    side = getattr(run, role)

    timing_args = run.server.timing_args
    if timing_args is None:
        timing_args = {"gpu": run.server.gpu} if run.server.timing == "aisimulate" else {}
    simulation = {"clock": "real", "timing": run.server.timing, "timing_args": timing_args,
                  "log_path": STEP_LOG, "kv_transfer_mode": "real"}
    if run.kv_check:
        simulation["kv_pattern_fill" if role == "prefill" else "kv_pattern_check"] = True
    spec = run.server.speculative
    if spec is not None:
        simulation["spec_acceptance_rates"] = spec["acceptance_rates"]
    nixl = run.server.connector == "nixl"
    kv_transfer = {"kv_connector": "NixlConnector" if nixl else "MooncakeConnector",
                   "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                   "kv_buffer_device": "cpu",
                   # Explicit selection prevents another installed backend from taking over.
                   "kv_connector_extra_config": {"backends": [nixl_backend(run.server)]} if nixl else {"mooncake_protocol": "tcp"}}
    if non_tent or len(set(kv_addresses)) > 1:
        kv_transfer["kv_connector_extra_config"]["rank_addresses"] = kv_addresses * side.data_parallel_size
    connector_env = {
        # NIXL's handshake and its Mooncake backend's segment on the KV network.
        "VLLM_NIXL_SIDE_CHANNEL_HOST": kv_address, "NIXL_MOONCAKE_IP_ADDR": kv_address,
        # A hybrid model's state moves as NIXL locates it: conv state DS (a
        # spec's env may set SD, which the simulator refuses).
        "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    } if nixl else {"VLLM_MOONCAKE_BOOTSTRAP_PORT": str(BOOTSTRAP_PORT)}
    env = {
        "TRITON_CPU_BACKEND": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "1",
        # llm-d's guides' value: vLLM's default (5 s) is shorter than the
        # sidecar's and Envoy's idle connection timeouts, so a server closes
        # a connection the router is about to reuse (the request is then
        # answered 502).
        "VLLM_HTTP_TIMEOUT_KEEP_ALIVE": str(HTTP_KEEP_ALIVE),
        # The address Mooncake's workers register (vLLM's get_ip).
        "VLLM_HOST_IP": kv_address,
        "MC_USE_TENT": "1",
        "MC_TENT_CONF": json.dumps(tent_config(kv_address, run.server.hp_tcp_args), separators=(",", ":")),
        **connector_env,
        # TENT's metrics, read from inside the node.
        "TENT_METRICS_HTTP_HOST": "127.0.0.1",
        "TENT_METRICS_HTTP_PORT": str(TENT_METRICS_PORT),
        "TENT_METRICS_REPORT_INTERVAL": "0",
        **{name: str(value) for name, value in run.server.env.items()},
    }
    if non_tent:
        env = {k: v for k, v in env.items() if not k.startswith(("MC_", "TENT_", "NIXL_MOONCAKE_"))}
        if ucx:
            env["UCX_TLS"] = "tcp,self"
        else:
            env.update(FI_PROVIDER="tcp", FI_ADDR_FORMAT="FI_SOCKADDR_IN", FI_PREFER_SYSCONFIG="0")
    s = run.server
    return [
        "env", *(f"{name}={value}" for name, value in env.items()),
        VLLM, "serve", MODEL_DIRECTORY, "--host", "0.0.0.0", "--port", str(port),
        "--served-model-name", MODEL_NAME, "--dtype", s.dtype, "--kv-cache-dtype", s.kv_cache_dtype,
        *(["--quantization", s.quantization] if s.quantization else []),
        *(["--block-size", str(s.block_size)] if s.block_size is not None else []),
        "--kv-cache-memory-bytes", str(s.kv_cache_memory_bytes), "--max-model-len", str(s.max_model_len),
        "--max-num-batched-tokens", str(s.max_num_batched_tokens), "--max-num-seqs", str(s.max_num_seqs),
        "--tensor-parallel-size", str(side.tp(s)),
        *_decode_context_parallel_flags(side),
        *(["--data-parallel-size", str(side.data_parallel_size), "--enable-expert-parallel"]
          if side.data_parallel_size > 1 else []),
        # Per-rank routing: an API server a rank, on ``port`` + rank.
        *(["--data-parallel-multi-port-external-lb", "--data-parallel-size-local", str(side.data_parallel_size),
           "--data-parallel-supervisor-port", str(DP_SUPERVISOR_PORT)] if rank_routed(run, side) else []),
        "--generation-config", "vllm", "--enforce-eager",
        *(["--language-model-only"] if s.language_model_only else []),
        *(["--trust-remote-code"] if s.trust_remote_code else []),
        # Off for the byte checks unless asked (then a block's pattern is
        # its content's).
        *_prefix_caching_flags(run),
        "--simulation-config", json.dumps(simulation),
        "--kv-transfer-config", json.dumps(kv_transfer),
        *(["--speculative-config", json.dumps({"method": spec["method"],
                                                "num_speculative_tokens": spec["num_speculative_tokens"]}),
           "--seed", str(spec.get("seed", 0))] if spec is not None else []),
        *_precise_index_flags(run, endpoint),
        "--simulate-forward",
    ]


def execute(run: Run, output: Path | None = None) -> dict:
    """Start the services and llm-d's router, run the workload, collect and
    check; returns the verdict (also in ``<output>/verdict.json``). Raises
    AssertionError when a check fails, after collecting."""

    from . import llmd  # it imports this module

    front = llmd.Front(run)
    state = topology.load_state(run.topology)
    nodes = state["nodes"]
    roles = (("prefill", run.prefill), ("decode", run.decode))
    container = {name: nodes[name]["container"] for name in (*run.prefill.nodes, *run.decode.nodes, run.client)}
    # Each instance's KV address per rank and its control address.
    instances = {role: {node: {"kv": _kv_addresses(nodes[node]["interfaces"], side, side.tp(run.server), node),
                               "control": nodes[node]["interfaces"][run.control_network]["address"]}
                        for node in side.nodes} for role, side in roles}
    address: dict = {"instances": instances}
    output = output or RESULTS / f"{run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps({**asdict(run), "model": str(run.model), "addresses": address},
                                                indent=2, default=str) + "\n")

    for name in container.values():
        # A previous run's servers would answer this run's requests.
        stop_services(name)
        topology.exec_in(name, ["sh", "-c", f"rm -rf {WORK}/logs {WORK}/results {WORK}/steps*.jsonl "
                                            f"&& mkdir -p {WORK}/logs"])
        # Only this run's model: a file the previous run's had and this
        # one's lacks (a quantization sidecar) would be read as its own.
        topology.exec_in(name, ["sh", "-c", f"rm -rf {MODEL_DIRECTORY} && mkdir -p {MODEL_DIRECTORY}"])
        shell.run(["docker", "cp", f"{run.model}/.", f"{name}:{MODEL_DIRECTORY}"])
    front.prepare(container)

    commands: dict[str, dict[str, list[str]]] = {}
    for role, side in roles:
        for node in side.nodes:
            command = server_command(run, role, instances[role][node]["kv"], front.server_port(role),
                                     (instances[role][node]["control"], front.endpoint_port(role)))
            commands.setdefault(role, {})[node] = command
            _background(container[node], command, f"{WORK}/logs/{role}.log")
    # Each server as it was started: its environment and flags.
    (output / "servers.json").write_text(json.dumps(commands, indent=2) + "\n")
    # And what it runs in: each node container's image, by id, and its mounts.
    (output / "containers.json").write_text(json.dumps(container_identities(container), indent=2) + "\n")
    try:
        # Every instance (the startup timeout counted from the first).
        deadline = time.monotonic() + run.startup_timeout
        for role, side in roles:
            for node in side.nodes:
                _wait(container[run.client], front.health_url(role, instances[role][node]["control"]),
                      max(1.0, deadline - time.monotonic()), container[node], f"{WORK}/logs/{role}.log")
        base_url = front.start(container, address)

        tent_before = _tent_metrics(run, container)
        control_before = network_ports(state, run.control_network)
        routers_before = router_bytes(state)
        ports_before = router_ports(state)
        bench = bench_command(run, base_url)
        (output / "bench.log").write_text(topology.exec_in(container[run.client], bench, timeout=3600))
        # The servers' step logs settle shortly after the last response.
        time.sleep(2)
        routers_after = router_bytes(state)
        ports_after = router_ports(state)
        control_after = network_ports(state, run.control_network)
        tent_after = _tent_metrics(run, container)
    finally:
        try:
            _collect(run, container, output)
            front.collect(container, output)
        finally:
            front.stop(container)
            for name in container.values():
                stop_services(name)
    routers = {network: routers_after[network] - routers_before[network] for network in routers_before}
    (output / "routers.json").write_text(json.dumps(routers, indent=2) + "\n")
    ports = _port_deltas(ports_before, ports_after)
    if ports:
        (output / "router-ports.json").write_text(json.dumps(ports, indent=2) + "\n")
    exempt = control_exemption(control_before, control_after, [run.router.gateway])
    (output / "control-ports.json").write_text(json.dumps(exempt, indent=2) + "\n")
    tent = {role: [_tent_delta(before, after) for before, after in zip(tent_before[role], tent_after[role])]
            for role in tent_before}
    if uses_tent(run.server):
        (output / "tent.json").write_text(json.dumps(tent, indent=2) + "\n")
    # As served after the workload: their labels (transports) included.
    if uses_tent(run.server):
        (output / "tent-after.json").write_text(json.dumps(tent_after, indent=2) + "\n")
    else:
        (output / "transport.json").write_text(json.dumps({"backend": nixl_backend(run.server), "tent_applicable": False}) + "\n")
    verdict = check(
        json.loads((output / "bench.json").read_text()),
        *[[(output / getattr(run, role).step_log(role, dp, tp)).read_text()
           for dp, tp in getattr(run, role).ranks(getattr(run, role).tp(run.server))]
          for role in ("prefill", "decode")],
        routers, run, tent, ports or None, exempt)
    llmd.recheck(output, run, verdict)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["problems"]:
        raise AssertionError(f"run {run.name} failed its checks ({output}): {'; '.join(verdict['problems'])}")
    return verdict


def recheck(output: str | Path) -> dict:
    """A saved run's checks again, from its results directory, with this
    harness: written to ``verdict-recheck.json``, the run's own
    ``verdict.json`` left as it was. The router's evidence is part of them:
    missing evidence is a problem."""
    from . import llmd  # it imports this module

    output = Path(output)
    run = saved_run(output)
    ports = output / "router-ports.json"
    # Required evidence: anything but an object (a `null` document) has no
    # ports, a problem of `control_evidence`, never `check`'s caller without
    # port evidence.
    control_saved = json.loads((output / "control-ports.json").read_text())
    if not isinstance(control_saved, dict):
        control_saved = {}
    verdict = check(
        json.loads((output / "bench.json").read_text()),
        *[[(output / getattr(run, role).step_log(role, dp, tp)).read_text()
           for dp, tp in getattr(run, role).ranks(getattr(run, role).tp(run.server))]
          for role in ("prefill", "decode")],
        json.loads((output / "routers.json").read_text()), run,
        None if not uses_tent(run.server) else json.loads((output / "tent.json").read_text()),
        json.loads(ports.read_text()) if ports.exists() else None,
        control_saved)
    llmd.recheck(output, run, verdict)
    (output / "verdict-recheck.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def saved_run(output: str | Path) -> Run:
    """A saved run's config (its ``run.json``)."""
    raw = json.loads((Path(output) / "run.json").read_text())
    raw.pop("addresses", None)
    router = Router(**raw["router"])
    if router.epp_plugins is not None:
        router.epp_plugins = Path(router.epp_plugins)
    return Run(**{**raw, "model": Path(raw["model"]), "router": router,
                  "prefill": Side(**raw["prefill"]), "decode": Side(**raw["decode"]),
                  "server": Server(**raw["server"]), "workload": Workload(**raw["workload"])})


def saved_peak_prefill_throughput(output: str | Path) -> dict:
    """`peak_prefill_throughput` of a saved run's prefill instances."""
    output = Path(output)
    run = saved_run(output)
    side = run.prefill
    return peak_prefill_throughput(
        [(output / side.step_log("prefill", dp, tp)).read_text() for dp, tp in side.ranks(side.tp(run.server))
         if tp == 0], run.server.max_num_batched_tokens)


def check(bench: dict, prefill_steps: list[str], decode_steps: list[str], routers: dict[str, int],
          run: Run, tent: dict[str, list[dict]] | None = None, ports: dict | None = None,
          control_exempt: dict | None = None) -> dict:
    """The run's checks, from the client's results, every rank's step logs
    (``prefill_steps[i]``, ``decode_steps[i]``, in each side's `Side.ranks`
    order), the bytes each router received during the workload and each
    rank's TENT counters over it (``tent[role][i]``). Requests are counted
    once, from the client; bytes per rank, then summed. Each request is
    produced by one prefill DP rank and consumed by one decode DP rank,
    the same on every TP rank."""

    problems = []
    _check_backend(run.server)
    if not uses_tent(run.server) and tent:
        problems.append(f"{nixl_backend(run.server)} received unexpected TENT accounting")
    w = run.workload
    sizes = {"prefill": run.prefill.tp(run.server), "decode": run.decode.tp(run.server)}
    # Equal sizes pair rank r with rank r; unequal ones re-shard heads
    # (`_hetero_tp`).
    tp_size = sizes["prefill"] if sizes["prefill"] == sizes["decode"] else None
    # With prefix caching the checks also compare each request's prefix
    # hits (`kind` hit), on both instances, apart from its transfer.
    prefix = bool(run.server.enable_prefix_caching) and run.kv_check
    errors = [error for error in bench.get("errors", []) if error]
    if bench.get("completed") != w.num_prompts or errors:
        problems.append(f"{bench.get('completed')} of {w.num_prompts} request(s) completed; errors: {errors[:3]}")

    sides = {"prefill": run.prefill, "decode": run.decode}
    # Several owners of requests per side: DP ranks, or instances.
    data_parallel = any(side.data_parallel_size > 1 or side.instances for side in sides.values())

    def where(dp: int | None, tp: int, side: Side | None = None) -> str:
        owner = f"instance {side.node_of(dp)}" if side is not None and side.instances and dp is not None else (
            f"DP rank {dp}")
        parts = ([owner] if data_parallel and dp is not None else []) + (
            [f"rank {tp}"] if max(sizes.values()) > 1 else [])
        return " ".join(parts) + ": " if parts else ""

    logs = {role: dict(zip(side.ranks(sizes[role]), map(_records, steps)))
            for role, side, steps in (("prefill", run.prefill, prefill_steps),
                                      ("decode", run.decode, decode_steps))}
    tents = {role: dict(zip(sides[role].ranks(sizes[role]), tent[role])) for role in tent} if tent and uses_tent(run.server) else {}
    # Each TP rank's steps, every DP rank's requests: under MoE DP only DP
    # rank 0 logs the joint steps, the others their own KV.
    joint_steps: dict[tuple[str, int], dict[str, list[tuple[int, int, int]]]] = {
        (role, tp): {} for role in sides for tp in range(sizes[role])}
    for role in sides:
        for (dp, tp), records in logs[role].items():
            joint_steps[(role, tp)].update(_request_steps(records))
    ranks: dict[str, dict[tuple[int, int], dict]] = {role: {} for role in sides}
    # Mooncake producer ranks without a write series (unequal TP only).
    unwritten: set[int] = set()
    # Instances' ranks without a moved series: given no request.
    idle: list[str] = []
    # Decode ranks without a read series under decode context parallelism.
    unread: set[int] = set()
    dcp_pair = max(side.decode_context_parallel_size for side in sides.values()) > 1
    for role, side in sides.items():
        for (dp, tp), records in logs[role].items():
            at = where(dp, tp, side)
            layout = next((r["kv_layout"] for r in records if "kv_layout" in r), None)
            # Each log must be this rank's (not another's supplied twice).
            if layout is None:
                problems.append(f"{at}{role} logged no KV layout")
            elif (layout.get("tp_rank"), layout.get("tp_size")) != (tp, sizes[role]):
                problems.append(f"{at}{role}'s log is TP rank {layout.get('tp_rank')} of "
                                f"{layout.get('tp_size')}, not {tp} of {sizes[role]}")
            elif data_parallel and layout.get("dp_rank") != (0 if side.instances else dp):
                problems.append(f"{at}{role}'s log is DP rank {layout.get('dp_rank')}, not "
                                f"{0 if side.instances else dp}")
            sent = {"count": 0, "bytes": 0, "failed": 0}
            for record in records:
                for transfers in (record.get("kv_transfers") or {}).values():
                    for key in sent:
                        sent[key] += int(transfers.get(key) or 0)
            if sent["failed"]:
                problems.append(f"{at}{role}'s connector failed {sent['failed']} transfer(s)")
            identity = (layout or {}).get("nixl_transport")
            if run.server.connector == "nixl" and (not uses_tent(run.server) or identity is not None):
                expected = [nixl_backend(run.server)]
                if not isinstance(identity, dict) or identity.get("backends") != expected:
                    problems.append(f"{at}{role}'s initialized NIXL backend is {identity}, expected {expected}")
                elif not uses_tent(run.server):
                    binding = identity.get("binding") or {}
                    ucx = uses_ucx(run.server)
                    settings_ok = (binding.get("UCX_TLS") == "tcp,self"
                                   and binding.get("UCX_NET_DEVICES") == binding.get("interface")) if ucx else (
                        binding.get("FI_PROVIDER") == "tcp"
                        and binding.get("FI_TCP_IFACE") == binding.get("interface")
                        and binding.get("FI_ADDR_FORMAT") == "FI_SOCKADDR_IN"
                        and binding.get("FI_PREFER_SYSCONFIG") == "0"
                        and binding.get("config_file") == "/opt/libfabric/etc/libfabric.conf"
                        and binding.get("config_policy") == "absent_or_empty")
                    if not binding.get("address") or not binding.get("interface") or not settings_ok:
                        problems.append(f"{at}{role} has no valid {nixl_backend(run.server)} TCP/interface evidence: {binding}")
                    if (identity.get("progress_thread") is not True
                            or type(identity.get("num_threads")) is not int or identity["num_threads"] < 0):
                        problems.append(f"{at}{role} has no valid {nixl_backend(run.server)} thread/progress evidence")
                    if not ucx and (identity.get("num_threads") != 4
                                    or identity.get("thread_policy_source") != "agent_config"
                                    or identity.get("native_provider_observed") is not False
                                    or identity.get("native_threads_observed") is not False
                                    or identity.get("backend_options") != {"num_threads": "4"}
                                    or identity.get("plugin_defaults") != {"num_threads": "0", "split_batch_size": "1024"}):
                        problems.append(f"{at}{role} lacks LIBFABRIC configured-policy/backend-options evidence")
                    addresses = {p["address"] for network in (ports or {}).values() for p in network.values()
                                 if p.get("node") == side.node_of(dp) and p.get("slot") == tp and p.get("address")}
                    if addresses and addresses != {binding.get("address")}:
                        problems.append(f"{at}{role}'s {nixl_backend(run.server)} address {binding.get('address')} differs from rank port {addresses}")
            logged = [record["kv_check"] for record in records if "kv_check" in record]
            # A transfer's check; a prefix hit's; a request the content
            # patterns could not describe (fill, write).
            checks = [c for c in logged if c.get("kind") in (None, "transfer")]
            hits = [c for c in logged if c.get("kind") == "hit"]
            outside = [c for c in logged if c.get("kind") not in (None, "transfer", "hit")]
            if run.kv_check:
                bad = [c for c in checks if not c.get("ok") or c.get("mismatched_bytes")]
                if bad:
                    problems.append(f"{at}{len(bad)} request(s) received KV bytes unlike prefill's: {bad[0]}")
                bad = [c for c in hits if not c.get("ok") or c.get("mismatched_bytes")]
                if bad:
                    problems.append(f"{at}{role}: {len(bad)} prefix hit(s) held bytes unlike their content's: "
                                    f"{bad[0]}")
                if outside:
                    problems.append(f"{at}{role}: {len(outside)} request(s) outside the pattern lifecycle: "
                                    f"{outside[0]}")
            if run.kv_check and role == "decode":
                if layout:
                    # Failed receives are reported above; every other check is validated.
                    problems.extend(f"{at}{problem}" for problem in _byte_check_problems(
                        layout, [c for c in checks if "error" not in c], w.input_len, _pull_plans(records),
                        {req_id: steps[0][2] for req_id, steps in joint_steps[("decode", tp)].items()}, prefix))
            admissions = _admissions(records)
            # The hits each request's admissions imply, checked: only those
            # reconciled count (`valid_hits`).
            valid_hits = []
            if prefix and layout:
                found, valid_hits = _hit_problems(layout, hits, admissions)
                problems.extend(f"{at}{role}: {problem}" for problem in found)
            verified = {c["req_id"] for c in checks
                        if "req_id" in c and "error" not in c and c.get("ok") and not c.get("mismatched_bytes")}
            pulled = {request for request, adms in admissions.items() if any(
                a["pull"] and any(block for _, blocks in a["pull"].values() for block in blocks) for a in adms)}
            if prefix and role == "decode":
                # A request admitted without a pull is consumed through its
                # own hit, checked, of its whole prompt but the last token
                # (the one decode computes: native caching hits at most N - 1,
                # a hybrid's replay is that token too), as admitted and as its
                # first step computes; one pulled, through its byte check.
                hit_ids = {c["req_id"] for c in valid_hits}
                first_steps = joint_steps[("decode", tp)]
                for request in sorted(set(admissions) - pulled):
                    if request not in hit_ids:
                        problems.append(f"{at}decode admitted {request} with neither a pull nor a checked prefix hit")
                        continue
                    computed = admissions[request][0]["computed"]
                    first = (first_steps.get(request) or [None])[0]
                    if first is None:
                        problems.append(f"{at}decode admitted {request} without a pull and logged no step of it")
                    elif min(computed, first[0]) < first[2] - 1:
                        problems.append(
                            f"{at}decode admitted {request} without a pull, {computed} of its {first[2]} prompt "
                            f"tokens cached, its first step computing from {first[0]}: it prefilled "
                            f"{first[2] - min(computed, first[0])} tokens itself, not only the last (a transfer "
                            "replaced by local prefill)")
                if unverified := sorted(pulled - verified):
                    problems.append(f"{at}{len(unverified)} pulled request(s) without a successful byte check: "
                                    f"{unverified[0]}")
                posted = {request for r in records for request in (r.get("kv_events") or {}).get("reads_posted") or []}
                if unadmitted := sorted(posted - set(admissions)):
                    problems.append(f"{at}{len(unadmitted)} pull(s) posted for requests decode never admitted: "
                                    f"{unadmitted[0]}")
                if unposted := sorted({c["req_id"] for c in checks if "req_id" in c} - pulled):
                    problems.append(f"{at}{len(unposted)} byte check(s) of requests whose pull was not logged: "
                                    f"{unposted[0]}")
            # The requests the rank owns: prefill's allocations; decode's
            # admissions (prefix caching), its byte checks, else its
            # allocations.
            if prefix and role == "decode":
                owned = {_base_request_id(request) for request in admissions}
            elif role == "decode" and run.kv_check:
                owned = {_base_request_id(c["req_id"]) for c in checks if "req_id" in c}
            else:
                owned = {_base_request_id(req_id) for r in records
                         for req_id in ((r.get("kv_events") or {}).get("allocated") or {})}
            counters = tents.get(role, {}).get((dp, tp)) if tents else None
            if tents:
                # Mooncake's prefill writes the KV, NIXL's decode reads it.
                mover, moved_counter = ("decode", "tent_read_bytes_total") if (
                    run.server.connector == "nixl") else ("prefill", "tent_write_bytes_total")
                required = ("tent_registered_buffer_bytes",) + ((moved_counter,) if role == mover else ())
                if counters is None:
                    problems.append(f"{at}{role}'s TENT metrics did not answer")
                elif missing := [name for name in required if name not in counters]:
                    if (side.instances or side.data_parallel_size > 1) and not owned and missing == [moved_counter]:
                        # An instance or a DP rank the router gave no request
                        # (the EPP picks the instance; the Mooncake sidecar
                        # any of the prefill's DP ranks, vLLM the others)
                        # moved nothing: its moved series was never
                        # incremented.
                        idle.append(f"{role} {side.node_of(dp)} rank {tp}" if side.instances
                                    else f"{role} DP rank {dp} rank {tp}")
                    elif tp_size is None and role == mover == "prefill" and missing == [moved_counter]:
                        # A replicated producer's rank that never writes never
                        # increments its write series: `_hetero_tp` accepts
                        # the absence only for a proven idle replica.
                        unwritten.add(tp)
                    elif dcp_pair and role == mover == "decode" and missing == [moved_counter]:
                        # A DCP rank whose pages are all past every handoff
                        # reads nothing: `_dcp_audit` accepts the absence only
                        # for a proven notify-only rank.
                        unread.add(tp)
                    else:
                        problems.append(f"{at}{role}'s TENT metrics lack {', '.join(missing)}")
                registered = (counters or {}).get("tent_registered_buffer_bytes")
                if registered is not None and layout and registered < layout["bytes"]:
                    problems.append(f"{at}{role}'s TENT registered {registered:.0f} bytes, less than its "
                                    f"{layout['bytes']} of KV")
                counted = (counters or {}).get(moved_counter)
                if role == mover and counted is not None and counted != sent["bytes"]:
                    problems.append(f"{at}TENT {'read' if mover == 'decode' else 'wrote'} {counted:.0f} bytes, "
                                    f"not the {sent['bytes']} its connector moved")
                if counters is not None:
                    # A healthy scrape omits a failure series never incremented.
                    failures = {key: int(counters.get(f"tent_{key}_total") or 0)
                                for key in ("task_failures", "transport_attempt_failures")}
                    if any(failures.values()):
                        problems.append(
                            f"{at}{role}'s TENT recorded {failures['task_failures']} failed task(s) and "
                            f"{failures['transport_attempt_failures']} failed transport attempt(s): not a clean "
                            f"transfer, whether or not a retry recovered it ({sent['failed']} connector transfer(s) "
                            "failed)")
            hp_tcp = None
            if tents and run.server.connector != "nixl" and role == "prefill":
                hp_tcp = _hp_tcp_writes(records, hp_tcp_task_limit(run.server))
            if tents and counters is not None:
                hp_tcp = {**(hp_tcp or {}), **{
                    key: int(counters.get(f"tent_{key}_total") or 0) if "failures" in key else (
                        None if counters.get(f"tent_{key}_total") is None else int(counters[f"tent_{key}_total"]))
                    for key in ("transport_attempts", "write_requests", "read_requests", "task_failures",
                                "transport_attempt_failures")}}
            ranks[role][(dp, tp)] = {
                "transfers": sent, "kv_checks": len(checks), "pulled": len(pulled),
                "checked_ids": {c["req_id"] for c in checks if "req_id" in c and "error" not in c},
                "verified_ids": verified,
                "allocated_ids": {req_id for r in records
                                  for req_id in ((r.get("kv_events") or {}).get("allocated") or {})},
                "hybrid": bool(layout and _prefill_backoff(layout)),
                "groups": layout and layout.get("groups") and _group_evidence(layout, checks, records),
                "kv_bytes_checked": sum(int(c.get("bytes") or 0) for c in checks),
                # Of them, NIXL's moves of regions holding none of a group's
                # layers (a shorter group's: a hybrid's state or attention).
                "kv_bytes_transfer_only": sum(int(g.get("transfer_only_bytes") or 0)
                                              for c in checks for g in c.get("groups") or []),
                "kv_bytes_spanned": _spanned_bytes(layout, checks) if layout else 0,
                # Prefix hits compared in place, reconciled with the
                # requests' admissions: no transfer moved them.
                "hit_ids": {_base_request_id(c["req_id"]) for c in valid_hits if "req_id" in c},
                "kv_hit_checks": len(valid_hits),
                "kv_hit_blocks": sum(c["blocks"] for c in valid_hits),
                "kv_bytes_hit_checked": sum(c["bytes"] for c in valid_hits),
                "requests": owned, "tent": counters, "hp_tcp": hp_tcp,
                "kv_layout": layout and {key: layout.get(key) for key in (
                    "dp_rank", "dp_local_rank", "tp_rank", "tp_size", "shape", "block_bytes", "bytes",
                    "tent_metrics_port")},
            }

    per_tp = []
    ownership: dict[str, dict[int, int]] = {"producers": {}, "consumers": {}}
    owners_by_tp: list[dict[str, dict[str, set[int]]]] = []
    hetero = None
    dcp = None
    if dcp_pair:
        # Each decode rank, with the producer ranks NIXL maps its DCP slice
        # to, at equal TP sizes or not.
        dcp = _dcp_audit(run, sizes, logs, unread)
        problems.extend(dcp.pop("problems"))
        hetero = {"sources": dcp.pop("sources"), "received": dcp.pop("received")}
    elif tp_size is None:
        # Each decode rank, with the producer ranks the native pairing gives it.
        hetero = _hetero_tp(run, sizes, logs, ranks, ports, unwritten)
        problems.extend(hetero.pop("problems"))
    for tp in range(sizes["decode"] if hetero else tp_size):
        at = where(None, tp)
        first = hetero["sources"][tp][0] if hetero else tp
        of = {role: {dp: summary for (dp, rank), summary in ranks[role].items()
                     if rank == (first if role == "prefill" else tp)} for role in sides}
        checked = sum(summary["kv_bytes_checked"] for summary in of["decode"].values())
        moved = hetero["received"].get(tp, 0) if hetero else sum(
            summary["transfers"]["bytes"] for role in sides for summary in of[role].values())
        # Decode checks every block the connector pulls: the transfer plan.
        # What the checked blocks span as the connector moves them (a packed
        # block whole), which a packed layout's checked pages do not fill.
        spanned = sum(summary["kv_bytes_spanned"] for summary in of["decode"].values())
        if moved < spanned or (run.kv_check and moved != spanned):
            what = "decode checked" if spanned == checked else f"decode's checked blocks span ({checked} checked)"
            problems.append(f"{at}the connector moved {moved} bytes, not the {spanned} {what}")
        owners: dict[str, dict[str, set[int]]] = {}
        for role in sides:
            owners[role] = {}
            for dp, summary in of[role].items():
                for request in summary["requests"]:
                    owners[role].setdefault(request, set()).add(dp)
        # Requests the decode DP ranks own, and the byte checks they logged
        # (none without kv_check).
        consumed = len(set().union(*(summary["requests"] for summary in of["decode"].values())))
        byte_checks = sum(summary["kv_checks"] for summary in of["decode"].values())
        # With prefix caching a request served from decode's own hits pulls
        # nothing: one byte check for each request that pulled.
        pulls = sum(summary["pulled"] for summary in of["decode"].values()) if prefix else consumed
        if run.kv_check:
            if consumed != w.num_prompts:
                problems.append(f"{at}decode {'admitted' if prefix else 'byte-checked'} {consumed} distinct "
                                f"request(s), not {w.num_prompts}")
            if byte_checks != pulls:
                problems.append(f"{at}decode logged {byte_checks} byte check(s) for {pulls} distinct "
                                f"request(s){' that pulled' if prefix else ''}")
        for role, label in (("prefill", "prefill DP ranks produced"), ("decode", "decode DP ranks consumed")):
            shared = sorted(request for request, dps in owners[role].items() if len(dps) > 1)
            if shared:
                problems.append(f"{at}{len(shared)} request(s) more than one of the {label}: {shared[0]}")
        unpaired = sorted(set(owners["prefill"]) ^ set(owners["decode"]))
        if unpaired:
            problems.append(f"{at}{len(unpaired)} request(s) not both produced and consumed: {unpaired[0]}")
        owners_by_tp.append(owners)
        handoff = None
        if any(summary["hybrid"] for summary in of["decode"].values()):
            # The requests decode consumed: its byte checks, else its
            # allocations (never its steps alone).
            consumed_ids = "checked_ids" if run.kv_check else "allocated_ids"
            handoff = _handoff_from_sources(
                joint_steps, tp, hetero["sources"][tp] if hetero else [tp],
                set().union(*(summary[consumed_ids] for summary in of["decode"].values())))
            problems.extend(f"{at}{problem}" for problem in handoff["problems"])
        hit = {role: {key: sum(summary[key] for summary in of[role].values())
                      for key in ("kv_hit_checks", "kv_hit_blocks", "kv_bytes_hit_checked")} for role in sides}
        if prefix:
            for role in sides:
                found = hit[role]["kv_hit_blocks"] and hit[role]["kv_bytes_hit_checked"]
                if role == "decode" and not run.expect_decode_prefix_hits:
                    if hit[role]["kv_hit_checks"]:
                        problems.append(f"{at}decode showed {hit[role]['kv_hit_checks']} prefix hit(s), the run "
                                        "expecting none (expect_decode_prefix_hits: false)")
                elif not found:
                    problems.append(f"{at}{role} showed no checked prefix hit: the run did not exercise prefix "
                                    "caching")
        fragmentation = _fragmentation(
            [records for (dp, rank), records in logs["prefill"].items() if rank == first],
            [records for (dp, rank), records in logs["decode"].items() if rank == tp],
            [ranks["decode"][(dp, rank)]["verified_ids"] for (dp, rank) in logs["decode"] if rank == tp])
        if tp == 0 and run.expect_fragmentation and not fragmentation["fragmented_both"]:
            problems.append(f"{at}no byte-checked transfer was fragmented on both instances: {fragmentation}")
        per_tp.append({
            "kv_checks": byte_checks, "consumed": consumed, "kv_bytes_checked": checked, "kv_bytes_spanned": spanned,
            "prefix_hits": hit if prefix else None, "fragmentation": fragmentation,
            "kv_bytes_transfer_only": sum(summary["kv_bytes_transfer_only"] for summary in of["decode"].values()),
            "transfers_bytes": moved,
            "kv_layout": {role: (of[role].get(0) or {}).get("kv_layout") for role in sides},
            "groups": {role: {dp: summary["groups"] for dp, summary in of[role].items()}
                       for role in sides} if any(s["groups"] for r in sides for s in of[r].values()) else None,
            "handoff": handoff and handoff["evidence"],
        })
    # Every TP rank of a request's owner is the same DP rank.
    for role, kind in (("prefill", "producers"), ("decode", "consumers")):
        first = owners_by_tp[0][role] if owners_by_tp else {}
        for tp, owners in enumerate(owners_by_tp[1:], start=1):
            differ = sorted(request for request in first if owners[role].get(request) != first[request])
            if differ:
                problems.append(f"rank {tp}: {len(differ)} request(s) with other {role} DP ranks than rank 0's")
        for dps in first.values():
            for dp in dps:
                ownership[kind][dp] = ownership[kind].get(dp, 0) + 1

    # Every KV byte crosses its TP rank's network's router once, and no
    # other: each carries its ranks' bytes and little more (headers,
    # acknowledgements).
    expected: dict[str, int] = {}
    # One side's connector counts them: Mooncake's prefill, NIXL's decode.
    for role in sides:
        for (dp, tp), summary in ranks[role].items():
            network = sides[role].rank_networks(sizes[role])[tp]
            expected[network] = expected.get(network, 0) + summary["transfers"]["bytes"]
    for network, carried_by_ranks in expected.items():
        carried = routers.get(network, 0)
        if carried < carried_by_ranks:
            problems.append(f"the KV network {network} carried {carried} bytes, fewer than the {carried_by_ranks} "
                            "its ranks moved")
        elif carried > carried_by_ranks * 1.05 + (1 << 20):
            problems.append(f"the KV network {network} carried {carried} bytes, more than its ranks' "
                            f"{carried_by_ranks} (another rank's KV, or traffic the connectors did not count)")
    moved = sum(summary["transfers"]["bytes"] for role in sides for summary in ranks[role].values())
    # The router's own traffic (the client's requests through the gateway,
    # the EPP's metrics scraping), measured at the gateway's ports, is not
    # KV: the guard is on the rest, both derived here from the same snapshot
    # of the control router's ports (`control_evidence`).
    if control_exempt is None:
        control, exempt = routers.get(run.control_network, 0), 0
    else:
        control, exempt, invalid = control_evidence(control_exempt, run)
        problems.extend(invalid)
    if moved and control - exempt > moved / 10:
        problems.append(f"the control network carried {control - exempt} bytes"
                        + (f" besides the front end's {exempt}" if exempt else "")
                        + f", too much beside {moved} of KV")
    sums = {key: sum(summary["transfers"][key] for role in sides for summary in ranks[role].values())
            for key in ("count", "bytes", "failed")}
    speculation = None
    if run.server.speculative is not None:
        speculation = _speculation(bench, logs["decode"], run)
        problems.extend(speculation.pop("problems"))
    prefix_routing = _prefix_routing(run, sides, ranks, bench) if prefix else None
    if prefix_routing and run.expect_prefill_hit_requests is not None and (
            prefix_routing["prefill"]["hit_requests"] < run.expect_prefill_hit_requests):
        problems.append(f"{prefix_routing['prefill']['hit_requests']} request(s) hit a cached prefix at prefill, "
                        f"fewer than the {run.expect_prefill_hit_requests} expected "
                        f"(by owner: {prefix_routing['prefill']['by_owner']})")
    return {
        **({"transport": {"backend": nixl_backend(run.server), "tent_applicable": False}} if not uses_tent(run.server) else {}),
        "problems": problems,
        "requests": bench.get("completed"),
        "kv_checks": per_tp[0]["kv_checks"] if per_tp else 0,
        "kv_bytes_checked": sum(rank["kv_bytes_checked"] for rank in per_tp),
        "kv_bytes_transfer_only": sum(rank.get("kv_bytes_transfer_only", 0) for rank in per_tp),
        "prefix_hits": _prefix_hit_totals(ranks) if prefix and per_tp else None,
        "fragmentation": per_tp[0]["fragmentation"] if per_tp else None,
        "prefix_routing": prefix_routing,
        "transfers": sums,
        "router_bytes": routers,
        "control_network": {"bytes": control, "front_end_bytes": exempt,
                            "front_end_nodes": [run.router.gateway],
                            "guarded_bytes": control - exempt},
        "ranks": per_tp,
        "ownership": ownership,
        **({"instances": {role: side.nodes for role, side in sides.items()}}
           if any(side.instances for side in sides.values()) else {}),
        **({"idle_ranks": idle} if idle or any(side.instances for side in sides.values()) else {}),
        "dp_ranks": {role: [{"dp_rank": dp, "tp_rank": tp,
                             **{key: value for key, value in summary.items()
                                if key not in ("requests", "checked_ids", "verified_ids", "allocated_ids", "hit_ids")},
                             "requests": len(summary["requests"])}
                            for (dp, tp), summary in ranks[role].items()] for role in sides},
        "speculation": speculation,
        "hetero_tp": hetero,
        "dcp": dcp,
        "hp_tcp_task_limit": hp_tcp_task_limit(run.server),
        "ttft_ms": {key: bench.get(f"{key}_ttft_ms") for key in ("mean", "median", "p99")},
        "tpot_ms": {key: bench.get(f"{key}_tpot_ms") for key in ("mean", "median", "p99")},
    }


def _prefix_hit_totals(ranks: dict[str, dict[tuple[int, int], dict]]) -> dict[str, dict[str, int]]:
    """A run's checked prefix hits by side: its TP rank 0's checks and blocks
    (every TP rank checks the same requests), and the bytes of each of its
    own (DP, TP) ranks once, whatever the other side's TP size."""

    return {role: {"kv_hit_checks": sum(s["kv_hit_checks"] for (_, tp), s in of.items() if not tp),
                   "kv_hit_blocks": sum(s["kv_hit_blocks"] for (_, tp), s in of.items() if not tp),
                   "kv_bytes_hit_checked": sum(s["kv_bytes_hit_checked"] for s in of.values())}
            for role, of in ranks.items()}


def _speculation(bench: dict, decode: dict[tuple[int, int], list[dict]], run: Run) -> dict:
    """Speculative decoding's evidence: every request returned exactly its
    output length (a last step's surplus drafts clipped); decode scheduled
    drafts, and each of a step's drafted requests has exactly one acceptance
    record (`spec_acceptance`) in its owning DP rank's log, for that step:
    its drafts, and accepted drafts within them, the step's own count where
    it has one (a joint MoE-DP step does not); all of them or none accepted
    at rates all 1 or all 0. TP ranks log the same: TP rank 0's logs."""

    spec = run.server.speculative
    problems = []
    lengths = bench.get("output_lens") or []
    if len(lengths) != run.workload.num_prompts or any(n != run.workload.output_len for n in lengths):
        problems.append(f"output lengths {sorted(set(lengths))} of {len(lengths)} request(s), not "
                        f"{run.workload.output_len} each")

    def count(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    # (owning DP rank, step, request): its drafts and the step's own count.
    scheduled: dict[tuple[int, int, str], tuple[int, int | None]] = {}
    logged: dict[tuple[int, int, str], list[object]] = {}
    duplicates, invalid = [], []
    steps = 0
    for (dp, tp), records in decode.items():
        if tp:
            continue
        for record in records:
            if "step" in record:
                step = record["step"]
                requests = step["requests"]
                drafts = [r["num_spec_tokens"] for r in requests]
                steps += any(drafts)
                counts = (record.get("cost") or {}).get("num_accepted")
                if counts is not None and any(drafts) and (
                        not isinstance(counts, list) or len(counts) != len(drafts)
                        or not all(count(a) and a <= d for a, d in zip(counts, drafts))):
                    problems.append(f"decode step {step['step_index']} accepted {counts} of drafts {drafts}")
                    counts = None
                for n, (request, d) in enumerate(zip(requests, drafts)):
                    if d:
                        owner = request.get("dp_rank", dp)
                        scheduled[(owner, step["step_index"], request["req_id"])] = (
                            d, counts[n] if counts is not None else None)
            if "spec_acceptance" in record:
                entry = record["spec_acceptance"]
                for request, pair in (entry.get("requests") or {}).items():
                    key = (dp, entry.get("step_index"), request)
                    if not isinstance(pair, list) or len(pair) != 2 or not all(map(count, pair)) or (
                            not 0 < pair[0] or pair[1] > pair[0]):
                        invalid.append(f"DP rank {dp} step {key[1]}: request {request} accepted {pair} "
                                       "(drafts, accepted)")
                    elif key in logged:
                        duplicates.append(f"DP rank {dp} step {key[1]}: request {request} logged twice")
                    else:
                        logged[key] = pair
    problems.extend(_capped(invalid) + _capped(duplicates))
    missing = [key for key in scheduled if key not in logged]
    extra = [key for key in logged if key not in scheduled]
    contradicted = [
        f"DP rank {key[0]} step {key[1]}: request {key[2]} logged {logged[key]}, the step drafted {d}"
        + (f" and accepted {a}" if a is not None else "")
        for key, (d, a) in scheduled.items() if key in logged
        and (logged[key][0] != d or (a is not None and logged[key][1] != a))]
    drafted = sum(d for d, _ in scheduled.values())
    accepted = sum(pair[1] for key, pair in logged.items() if key in scheduled)
    if not drafted:
        problems.append("decode scheduled no drafts")
    if missing:
        problems.append(f"{len(missing)} drafted request step(s) lack their owner's acceptance (spec_acceptance), "
                        f"e.g. DP rank {missing[0][0]} step {missing[0][1]} request {missing[0][2]}")
    if extra:
        problems.append(f"{len(extra)} acceptance record(s) for no drafted request step of their DP rank, e.g. "
                        f"DP rank {extra[0][0]} step {extra[0][1]} request {extra[0][2]}")
    problems.extend(_capped(contradicted))
    rates = spec["acceptance_rates"]
    if drafted and set(rates) == {1} and accepted != drafted:
        problems.append(f"{accepted} of {drafted} drafts accepted at acceptance rates 1")
    elif set(rates) == {0} and accepted:
        problems.append(f"{accepted} drafts accepted at acceptance rates 0")
    by_dp: dict[str, dict[str, int]] = {}
    for (owner, _, _), pair in sorted(logged.items(), key=lambda item: str(item[0])):
        rank = by_dp.setdefault(str(owner), {"drafted": 0, "accepted": 0})
        rank["drafted"] += pair[0]
        rank["accepted"] += pair[1]
    return {"problems": problems, "num_speculative_tokens": spec["num_speculative_tokens"],
            "acceptance_rates": rates, "seed": spec.get("seed", 0), "steps_with_drafts": steps,
            "drafted": drafted, "drafts_logged": sum(pair[0] for pair in logged.values()), "accepted": accepted,
            "dp_ranks": by_dp}

def _pull_plans(records: list[dict]) -> dict[str, dict[int, tuple[int, int]]]:
    """Decode's pulls as the connector posted them: for each request in a
    step's ``reads_posted``, that step's ``allocated_groups`` entries (the
    blocks it pulls, not the request's later scheduled table), as each
    group's first place and blocks holding KV."""
    plans: dict[str, dict[int, tuple[int, int]]] = {}
    for record in records:
        events = record.get("kv_events") or {}
        by_group = events.get("allocated_groups") or {}
        for request in events.get("reads_posted") or []:
            plans[request] = {entry["group"]: (entry["first"], sum(1 for block in entry["blocks"] if block))
                              for entry in by_group.get(request, [])}
    return plans


def _prefill_backoff(layout: dict) -> bool:
    """Whether decode loads N - 1 of a prompt's N tokens and computes the
    last: a recurrent state's, or (``prefill_backoff``) a window replayed
    after a load (DeepSeek-V4's)."""
    return layout.get("prefill_backoff") == 1 or any(
        group.get("kind") == "MambaSpec" for group in layout.get("groups") or [])


def _packed_problems(group: dict) -> list[str]:
    """How a packed group's layout (each block all of its layers' pages,
    DeepSeek-V4's) is malformed: each member names its kind, dtype, tokens a
    state, layer count and one offset per layer, its real bytes within its
    page; every page lies within the block stride, none overlapping another;
    the members count the group's layers, its checked bytes are their real
    pages and the bytes a block moves span the block. Empty for a group that
    is not packed."""
    if "kinds" in group and "members" not in group:
        return _kinds_problems(group)
    if not any(key in group for key in ("members", "checked_bytes_per_block", "transfer_bytes_per_block")):
        return []

    def size(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    members = group.get("members")
    checked = group.get("checked_bytes_per_block")
    moved = group.get("transfer_bytes_per_block")
    stride = group.get("block_stride")
    if not (isinstance(members, list) and members and size(checked) and size(moved) and size(stride)):
        return [f"has a malformed packed layout (members, checked_bytes_per_block {checked!r}, "
                f"transfer_bytes_per_block {moved!r}, block_stride {stride!r})"]
    problems, pages = [], []
    for index, member in enumerate(members):
        fields = ("kind", "dtype", "tokens_per_state", "layers", "offsets", "page_bytes", "real_page_bytes")
        if not isinstance(member, dict) or any(key not in member for key in fields):
            return [f"has a malformed packed member {index} (needs {', '.join(fields)})"]
        offsets, per_state = member["offsets"], member["tokens_per_state"]
        if not (isinstance(member["kind"], str) and isinstance(member["dtype"], str)
                and isinstance(per_state, (int, float)) and not isinstance(per_state, bool) and per_state > 0
                and size(member["layers"]) and size(member["page_bytes"]) and size(member["real_page_bytes"])
                and isinstance(offsets, list) and len(offsets) == member["layers"]
                and all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offsets)):
            return [f"has a malformed packed member {index} ({member!r})"]
        if member["real_page_bytes"] > member["page_bytes"]:
            problems.append(f"member {index} has {member['real_page_bytes']} real bytes in a "
                            f"{member['page_bytes']}-byte page")
        for offset in offsets:
            if offset + member["page_bytes"] > stride:
                problems.append(f"member {index}'s page at {offset} ends beyond the {stride}-byte block")
            pages.append((offset, offset + member["page_bytes"], index))
    pages.sort()
    for (_, end, first), (start, _, second) in zip(pages, pages[1:]):
        if start < end:
            problems.append(f"member {second}'s page at {start} overlaps member {first}'s, ending at {end}")
    layers = sum(member["layers"] for member in members)
    real = sum(member["real_page_bytes"] * member["layers"] for member in members)
    if layers != group.get("layers"):
        problems.append(f"packs {layers} layers, not its {group.get('layers')}")
    if real != checked:
        problems.append(f"checks {checked} bytes a block, not its members' {real} real bytes")
    if moved < stride:
        problems.append(f"moves {moved} bytes a block, less than its {stride}-byte block")
    return problems


def _kinds_problems(group: dict) -> list[str]:
    """How a group of layers of several pages, each layer in its own region
    (a DSA layer's latent and indexer keys), is malformed: each kind names
    its role, dtype, layer count, page and real bytes and block stride, its
    real bytes within its page and its page within its block; the kinds
    count the group's layers and its checked bytes a block are their pages
    (not the group's first tensor's)."""
    kinds = group.get("kinds")
    checked = group.get("checked_bytes_per_block")
    fields = ("role", "dtype", "layers", "page_bytes", "real_page_bytes", "block_stride")

    def size(value: object) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    if not (isinstance(kinds, list) and len(kinds) > 1 and size(checked)
            and all(isinstance(k, dict) and all(f in k for f in fields) for k in kinds)
            and all(size(k["layers"]) and size(k["page_bytes"]) and size(k["real_page_bytes"])
                    and size(k["block_stride"]) for k in kinds)):
        return [f"has a malformed layout of several pages (kinds {kinds!r}, checked_bytes_per_block {checked!r})"]
    problems = []
    if sum(k["layers"] for k in kinds) != group.get("layers"):
        problems.append(f"has kinds of {sum(k['layers'] for k in kinds)} layers, not its {group.get('layers')}")
    if checked != sum(k["page_bytes"] * k["layers"] for k in kinds):
        problems.append(f"checks {checked} bytes a block, not its kinds' pages")
    problems.extend(f"kind {k['role']} has {k['real_page_bytes']} real bytes in a {k['page_bytes']}-byte page"
                    for k in kinds if k["real_page_bytes"] > k["page_bytes"])
    problems.extend(f"kind {k['role']}'s {k['page_bytes']}-byte page exceeds its {k['block_stride']}-byte block"
                    for k in kinds if k["page_bytes"] > k["block_stride"])
    return problems


def _spanned_bytes(layout: dict, checks: list[dict]) -> int:
    """The bytes a rank's successful byte checks span as the connector moves
    them: their checked bytes, but a packed group's blocks whole, and without
    the sub-blocks a smaller producer block size left uncovered (NIXL zeroes
    them; `zeroed_bytes`)."""
    by_group = {index: group for index, group in enumerate(layout.get("groups") or [])}
    total = 0
    for check in checks:
        if "error" in check:
            continue
        nested = check.get("groups")
        if nested is None:
            total += int(check.get("bytes") or 0)
            continue
        for g in nested:
            group = by_group.get(g.get("group"), {})
            moved = group.get("transfer_bytes_per_block")
            packed = isinstance(moved, int) and not _packed_problems(group)
            total += g["blocks"] * moved if packed else (
                int(g.get("bytes") or 0) - int(g.get("zeroed_bytes") or 0))
    return total


def _byte_check_problems(layout: dict, checks: list[dict], input_len: int,
                         plans: dict[str, dict[int, tuple[int, int]]],
                         prompts: dict[str, int] | None = None, prefix: bool = False) -> list[str]:
    """How a rank's successful byte checks disagree with its KV layout.

    An older simulator's layout (no ``groups``): each check's bytes are its
    blocks times its layers times the layout's block bytes. Otherwise each
    check has blocks, bytes and nested results covering each KV cache group
    once, its blocks and bytes their sums; each group holds blocks (with
    prefix caching off a pull has some of every group, but a 1-token
    window's of a block-aligned prompt, all skipped), its bytes being
    its blocks times its layers' block bytes; a window pulls its last
    blocks only (Mooncake's cdiv(window, block) + 1), other attention from
    the first, all of the request's prompt (`prompts`, by decode's request
    id): cdiv(tokens, block) blocks of its N tokens, N - 1 when the
    decoder computes the last token (a hybrid model's, or a layout's
    ``prefill_backoff``); a gated delta-net's state
    is one block, at place 0. For several groups or a window, each group
    must also be the request's pull as posted (``_pull_plans``), and prompts
    longer than a window must have had some pull clipped. With prefix
    caching (`prefix`) a pull starts after decode's own prefix hit, any
    block, and takes the rest of the prompt's. A packed group
    (each block all of its layers' pages, DeepSeek-V4's) checks its layers'
    real pages, ``checked_bytes_per_block`` a block. Missing fields are
    problems."""
    groups = layout.get("groups")
    problems = []
    if groups is None:
        for check in checks:
            if any(key not in check for key in ("blocks", "layers", "bytes")) or (
                    check["bytes"] != check["blocks"] * check["layers"] * layout["block_bytes"]):
                problems.append(f"{check.get('req_id')}'s byte check disagrees with the rank's KV layout "
                                f"({layout['block_bytes']} bytes a block and layer)")
        return _capped(problems)
    grouped = len(groups) > 1 or any(spec.get("sliding_window") is not None for spec in groups)
    # The decoder computes the last of N > 1 prompt tokens (a recurrent
    # state, or a window replayed after a load).
    hybrid = _prefill_backoff(layout)
    if layout.get("prefill_backoff") not in (None, 1):
        problems.append(f"the layout's prefill_backoff is {layout.get('prefill_backoff')!r}, not 1")
    invalid = set()
    for index, spec in enumerate(groups):
        if found := _packed_problems(spec):
            invalid.add(index)
            problems.extend(f"group {index} {problem}" for problem in found)
    clipped = [0] * len(groups)
    for check in checks:
        request = check.get("req_id")
        if missing := [key for key in ("blocks", "bytes", "groups") if key not in check]:
            problems.append(f"{request}'s byte check lacks {', '.join(missing)}")
            continue
        nested = check["groups"]
        ids = sorted(g["group"] for g in nested)
        if ids != list(range(len(groups))):
            problems.append(f"{request}'s byte check covers groups {ids}, not each of the layout's "
                            f"{len(groups)} once")
        if (check["blocks"], check["bytes"]) != (sum(g["blocks"] for g in nested), sum(g["bytes"] for g in nested)):
            problems.append(f"{request}'s byte check is not its groups' sum")
        # With prefix caching a pull's plan is logged for one group too.
        plan = plans.get(request) if grouped or prefix else None
        if (grouped or prefix) and plan is None:
            problems.append(f"{request}'s pull plan was not logged (reads_posted with allocated_groups)")
        for g in nested:
            spec = groups[g["group"]] if 0 <= g["group"] < len(groups) else None
            if spec is None:
                continue
            window = spec.get("sliding_window")
            prompt = (prompts or {}).get(request)
            # A hybrid's decoder computes the last of N > 1 tokens; one
            # token's pages are pulled whole, the token computed again.
            tokens = prompt - 1 if hybrid and prompt is not None and prompt > 1 else prompt
            # The next token's 1-token window skips a block-aligned prompt whole.
            skipped = window == 1 and tokens is not None and tokens % spec["block_size"] == 0
            if skipped and (g["blocks"], g["first_logical_block"]) != (0, tokens // spec["block_size"]):
                problems.append(f"{request}'s 1-token window group {g['group']} checked {g['blocks']} blocks from "
                                f"{g['first_logical_block']}, not 0 from {tokens // spec['block_size']} (a "
                                f"block-aligned prompt's end)")
            elif g["blocks"] < 1 and not skipped and not _dcp_notify_only(g):
                problems.append(f"{request}'s group {g['group']} checked no blocks")
            if g["group"] in invalid:
                pass  # its malformed layout is reported above
            elif "transfer" in spec:
                # Several groups over NIXL: the native selection of each
                # block, its transfer-only regions and its state pages'
                # unmoved padding.
                transfer = spec["transfer"]
                want = {key: g["blocks"] * transfer[per] for key, per in (
                    ("bytes", "bytes_per_block"), ("transfer_only_bytes", "transfer_only_bytes_per_block"),
                    ("padding_bytes", "padding_bytes_per_block"), ("descriptors", "descriptors_per_block"))}
                got = {key: g.get(key) for key in want}
                if (g["layers"], got) != (spec["layers"], want):
                    problems.append(f"{request}'s group {g['group']} check disagrees with NIXL's selection "
                                    f"({spec['layers']} layers, {want}; checked {g['layers']} layers, {got})")
            elif "checked_bytes_per_block" in spec:
                # A packed block: its layers' real pages are checked.
                if (g["layers"], g["bytes"]) != (spec["layers"], g["blocks"] * spec["checked_bytes_per_block"]):
                    problems.append(f"{request}'s group {g['group']} check disagrees with the layout "
                                    f"({spec['layers']} layers, {spec['checked_bytes_per_block']} bytes checked "
                                    "a block)")
            elif (g["layers"], g["bytes"]) != (spec["layers"], g["blocks"] * spec["layers"] * spec["block_bytes"]):
                problems.append(f"{request}'s group {g['group']} check disagrees with the layout "
                                f"({spec['layers']} layers of {spec['block_bytes']} bytes a block)")
            # A state's running slot (before its native MTP's scratch
            # slots, the one NIXL moves) has clipped-read metadata index k
            # (`speculative_blocks`); any other group starts at 0.
            start = (spec.get("speculative_blocks") or 0) if spec.get("kind") == "MambaSpec" else 0
            if window is None and g["first_logical_block"] != start and not prefix:
                problems.append(f"{request}'s group {g['group']} ({spec.get('kind')}) was pulled from block "
                                f"{g['first_logical_block']}, not {start}")
            if spec.get("kind") == "MambaSpec":
                if g["blocks"] != 1:
                    problems.append(f"{request}'s state group {g['group']} checked {g['blocks']} blocks, not 1")
            elif window is None and prompt is None:
                problems.append(f"{request}'s group {g['group']} pages cannot be checked: decode logged no "
                                "steps of it")
            elif window is None and "payload_blocks" in g:
                # Decode context parallelism: a rank holds the request's
                # scheduler blocks (DCP pages each), its payload pages
                # reconciled with the handoff in `_dcp_audit`.
                # With prefix caching the pull starts past decode's own hit,
                # at its posted first block (`_pull_plans`).
                dcp = int((check.get("dcp") or {}).get("size") or 1)
                held = sum(int(g.get(key) or 0) for key in ("payload_blocks", "padding_blocks", "untouched_blocks"))
                if g["first_logical_block"] + held != -(-tokens // (spec["block_size"] * dcp)):
                    problems.append(f"{request}'s group {g['group']} holds {held} blocks from "
                                    f"{g['first_logical_block']}, not the rest of the cdiv({tokens}, "
                                    f"{spec['block_size']} * DCP {dcp}) of its prompt")
            elif window is None:
                hit = g["first_logical_block"] if prefix else 0
                if g["blocks"] != -(-tokens // spec["block_size"]) - hit:
                    problems.append(f"{request}'s group {g['group']} checked {g['blocks']} blocks, not the "
                                    f"cdiv({tokens}, {spec['block_size']}) of its prompt"
                                    + (f" past its {hit}-block prefix hit" if hit else ""))
            if window is not None:
                bound = -(-window // spec["block_size"]) + 1
                if g["blocks"] > bound:
                    problems.append(f"{request}'s window group {g['group']} pulled {g['blocks']} blocks, "
                                    f"more than its window's {bound}")
                clipped[g["group"]] += g["first_logical_block"] > 0
            # A pull posts no entry for a group it pulls nothing of. A DCP
            # check's blocks are its moved pages: its pull's are every
            # block it holds (untouched padding included).
            counted = (sum(int(g.get(key) or 0) for key in ("payload_blocks", "padding_blocks", "untouched_blocks"))
                       if "payload_blocks" in g else g["blocks"])
            if plan is not None and plan.get(g["group"], (tokens // spec["block_size"], 0) if skipped else None) != (
                    g["first_logical_block"], counted):
                first, blocks = plan.get(g["group"]) or (None, 0)
                problems.append(f"{request}'s group {g['group']} checked {counted} block(s) from "
                                f"{g['first_logical_block']}, not the pull's {blocks} from {first}")
    for index, spec in enumerate(groups):
        window = spec.get("sliding_window")
        if checks and window is not None and input_len > window + 2 * spec["block_size"] and not clipped[index]:
            problems.append(f"no pull of window group {index} was clipped, with prompts of {input_len} tokens "
                            f"beyond its {window}")
    return _capped(problems)


def _runs(blocks: list[int]) -> int:
    """Physically consecutive runs of a block list."""
    return len(blocks) and 1 + sum(b != a + 1 for a, b in zip(blocks, blocks[1:]))


def _fragmentation(prefill: list[list[dict]], decode: list[list[dict]], verified: list[set[str]]) -> dict:
    """How scattered one TP rank's pulled blocks were: for each pull decode
    posted, its blocks of each group (``allocated_groups`` of that step) and
    the producer's at the same places (its allocations, by base request id),
    each side's physical runs. A pull is fragmented on a side when a group's
    blocks there are not one run; jointly contiguous runs (consecutive on
    both sides at once) are what a connector can merge into one transfer.
    Only pulls with a successful byte check (`verified[i]`, decode's i-th
    log's) whose every block has the producer's are classified; the others
    are counted apart: unlogged (no allocations by group: one full-attention
    group without prefix caching), unchecked, unmatched (a producer place
    missing; `unmatched_groups`, their groups)."""
    produced: dict[str, dict[int, dict[int, int]]] = {}
    for records in prefill:
        for record in records:
            for request, entries in ((record.get("kv_events") or {}).get("allocated_groups") or {}).items():
                table = produced.setdefault(_base_request_id(request), {})
                for entry in entries:
                    places = table.setdefault(entry["group"], {})
                    for offset, block in enumerate(entry["blocks"]):
                        places[entry["first"] + offset] = block
    kinds = {(False, False): "contiguous", (True, False): "fragmented_prefill_only",
             (False, True): "fragmented_decode_only", (True, True): "fragmented_both"}
    counts = dict.fromkeys(kinds.values(), 0)
    joint = unlogged = unchecked = unmatched = unmatched_groups = 0
    for records, checked in zip(decode, verified):
        for record in records:
            events = record.get("kv_events") or {}
            for request in events.get("reads_posted") or []:
                table = produced.get(_base_request_id(request)) or {}
                entries = (events.get("allocated_groups") or {}).get(request)
                if not entries:
                    unlogged += 1
                    continue
                if request not in checked:
                    unchecked += 1
                    continue
                groups = [[(block, (table.get(entry["group"]) or {}).get(entry["first"] + offset))
                           for offset, block in enumerate(entry["blocks"]) if block] for entry in entries]
                missing = sum(any(p is None for _, p in pairs) for pairs in groups)
                if missing:
                    unmatched += 1
                    unmatched_groups += missing
                    continue
                sides = [False, False]
                for pairs in groups:
                    ours, theirs = [d for d, _ in pairs], [p for _, p in pairs]
                    sides[0] |= _runs(theirs) > 1
                    sides[1] |= _runs(ours) > 1
                    joint += len(pairs) and 1 + sum(
                        not (d2 == d1 + 1 and p2 == p1 + 1) for (d1, p1), (d2, p2) in zip(pairs, pairs[1:]))
                counts[kinds[tuple(sides)]] += 1
    return {"pulls": sum(counts.values()) + unlogged + unchecked + unmatched, **counts, "unlogged": unlogged,
            "unchecked": unchecked, "unmatched": unmatched, "jointly_contiguous_runs": joint,
            "unmatched_groups": unmatched_groups}


NIXL_MAX_DESCRIPTORS = 1024
"""A conservative benchmark guard on the descriptors vLLM selects for one
NIXL/Mooncake READ (before NIXL merges them): not a proven native limit,
since the plugin receives the merged list. This is not a UCX or LIBFABRIC limit."""


def kv_heads_of(model: Path) -> int | None:
    """The model's KV heads as its KV cache holds them, from its own config
    (a multimodal model's text config): an MLA model's (`kv_lora_rank`) one
    latent, the same on every rank, whatever attention heads it computes."""
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    sources = (config, config.get("text_config") or {}, config.get("llm_config") or {})
    if any(source.get("kv_lora_rank") for source in sources):
        return 1
    for source in sources:
        if isinstance(source.get("num_key_value_heads"), int):
            return source["num_key_value_heads"]
    return None


def global_heads(rank: int, tp: int, kv_heads: int) -> list[int]:
    """The global KV heads vLLM gives TP rank `rank` of `tp`: K/tp of them
    from rank * K/tp, or with tp > K one, head rank * K/tp (replicated)."""
    if tp <= kv_heads:
        return list(range(rank * kv_heads // tp, (rank + 1) * kv_heads // tp))
    return [rank * kv_heads // tp]


def group_kv_heads(model: Path, kinds: list) -> tuple[list[int] | None, str | None]:
    """Each KV cache group's total KV heads, from its layout `kind`: the
    model's own for all groups, except in the config family that gives its
    full-attention layers a count of their own (`num_global_key_value_heads`,
    Gemma4): a `FullAttentionSpec` group takes that, a `SlidingWindowSpec`
    group `num_key_value_heads`. Any other kind in such a config, or an
    invalid count, is an error: never a default. (None, error) then."""
    uniform = kv_heads_of(model)
    if uniform is None:
        return None, f"{model}/config.json gives no num_key_value_heads"
    try:
        config = json.loads((Path(model) / "config.json").read_text())
    except (OSError, ValueError):
        return None, f"{model}/config.json is unreadable"
    sources = (config, config.get("text_config") or {}, config.get("llm_config") or {})
    own = next((source["num_global_key_value_heads"] for source in sources
                if source.get("num_global_key_value_heads") is not None), None)
    if own is None:
        return [uniform] * len(kinds), None
    if isinstance(own, bool) or not isinstance(own, int) or own <= 0:
        return None, f"num_global_key_value_heads {own!r} is not a positive integer"
    totals = []
    for kind in kinds:
        if kind == "FullAttentionSpec":
            totals.append(own)
        elif kind == "SlidingWindowSpec":
            totals.append(uniform)
        else:
            return None, (f"a {kind} group has no KV head rule in a config with num_global_key_value_heads "
                          "(full-attention and sliding-window groups only)")
    return totals, None


def dcp_sources(rank: int, decode_tp: int, prefill_tp: int, prefill_dcp: int, decode_dcp: int) -> list[int]:
    """The producer ranks NIXL maps decode TP rank `rank` to under decode
    context parallelism (`compute_tp_mapping`, `TransferTopology.dcp_source_ranks`;
    each side's DCP 1 or its TP, the DCP rank the TP rank modulo DCP): a
    replicated producer (DCP 1) its rank `rank * prefill_tp // decode_tp`;
    a sharded one, for a consumer DCP at most the producer's, every rank r
    with r % decode DCP == the consumer's DCP rank (all of them for a
    replicated consumer), else (a finer consumer) the one rank of its DCP
    rank modulo the producer's DCP. A source holding none of the rank's
    pages is only notified (no READ)."""
    dcp_rank = rank % decode_dcp
    if prefill_dcp == 1:
        return [rank * prefill_tp // decode_tp]
    if decode_dcp <= prefill_dcp:
        return [r for r in range(prefill_tp) if r % decode_dcp == dcp_rank]
    return [dcp_rank % prefill_dcp]


def _dcp_notify_only(group: dict) -> bool:
    """A DCP check's group of no page to move: all its blocks past the
    handoff and untouched."""
    return ("payload_blocks" in group and group.get("payload_blocks") == group.get("padding_blocks") == 0
            and int(group.get("untouched_blocks") or 0) >= 1)


def _dcp_pulls(records: list[dict]) -> tuple[dict[str, int], dict[str, int], set[str]]:
    """A decode rank's pulls: each request's blocks as allocated when its
    read was posted, each request's prompt tokens (the producer's handoff)
    from its steps, and the requests whose reads finished."""
    held: dict[str, int] = {}
    prompts: dict[str, int] = {}
    done: set[str] = set()
    for record in records:
        events = record.get("kv_events") or {}
        for request in events.get("reads_posted") or []:
            held.setdefault(request, len((events.get("allocated") or {}).get(request) or []))
        done.update(events.get("reads_done") or [])
        for step in (record.get("step") or {}).get("requests") or []:
            if isinstance(step.get("num_prompt_tokens"), int):
                prompts.setdefault(step["req_id"], step["num_prompt_tokens"])
    return held, prompts, done


def _allocated_blocks(records: list[dict]) -> dict[str, int]:
    """Each request's blocks as a rank's steps allocated them: a new
    request's, then the blocks each later step adds (a chunk's, a decode
    token's), a resumed request's replacing its earlier ones."""
    held: dict[str, int] = {}
    for record in records:
        events = record.get("kv_events") or {}
        for request in events.get("resumed") or []:
            held[request] = 0
        for request, blocks in (events.get("allocated") or {}).items():
            held[request] = held.get(request, 0) + len(blocks)
    return held


def _dcp_audit(run: Run, sizes: dict[str, int], logs: dict, unread: set[int] = frozenset()) -> dict:
    """Decode context parallelism (`_check_dcp`), each rank's DCP rank its TP
    rank modulo the side's DCP size, its producer ranks NIXL's mapping
    (`dcp_sources`), at equal TP sizes or not: every rank's
    resolved geometry in its layout (one page size, the scheduler block DCP
    pages, the interleave the page); each decode check's global pages (DCP
    rank r's blocks pages k * DCP + r), those under the producer's handoff
    (its tokens over the page) the payload, the others padding moved or
    left; each READ of bytes from a producer rank `dcp_sources` names, each
    load record its READs' count and bytes, a rank's checked bytes its read
    ones (with ``kv_check``); every producer rank's sends done for every
    request it held (each released once all its readers notified it). With
    checks or without, each decode rank's pulls (`_dcp_pulls`) bound its read
    bytes: at least its payload pages (those of its blocks under each
    handoff), at most its blocks. A decode rank without a TENT read series
    (`unread`) must be notify-only: no READ, no payload page, every read
    finished, and (with ``kv_check``) its checks, at least one, of untouched
    blocks alone."""
    problems: list[str] = []
    tp = sizes["decode"]
    dcps = {"prefill": run.prefill.decode_context_parallel_size, "decode": run.decode.decode_context_parallel_size}
    # Every page size a rank records (its group's block size, its DCP
    # geometry's page): one for the pair, whichever ranks hold it.
    recorded: dict[int, list[str]] = {}
    for role in ("prefill", "decode"):
        for rank in range(sizes[role]):
            layout = next((r["kv_layout"] for r in logs[role].get((0, rank), []) if "kv_layout" in r), None)
            geometry = (layout or {}).get("decode_context_parallel")
            for group in (layout or {}).get("groups") or []:
                if isinstance(group.get("block_size"), int):
                    recorded.setdefault(group["block_size"], []).append(f"{role} rank {rank}")
            for value in ((geometry or {}).get("page_tokens") or [])[:1]:
                if isinstance(value, int):
                    recorded.setdefault(value, []).append(f"{role} rank {rank}")
            dcp = dcps[role]
            if dcp == 1:
                if geometry is not None:
                    problems.append(f"{role} rank {rank} logged decode context parallelism {geometry}, not none")
                continue
            own = (geometry or {}).get("page_tokens") or [None]
            want = {"size": dcp, "rank": rank % dcp, "page_tokens": own[:1],
                    "scheduler_block_tokens": [own[0] * dcp] if isinstance(own[0], int) else None,
                    "interleave": own[0]}
            if geometry != want or not isinstance(own[0], int):
                problems.append(f"{role} rank {rank}'s decode context parallelism is {geometry}, not {want}")
    prefill_dcp, decode_dcp = dcps["prefill"], dcps["decode"]
    common = sorted(recorded)
    if len(common) != 1:
        problems.append(f"the pair's ranks record page sizes {recorded or 'none'}, not one")
    page = common[0] if len(common) == 1 else None
    # Each producer rank's blocks of each request, as its steps allocated them.
    produced = {rank: {_base_request_id(request): blocks for request, blocks in
                       _allocated_blocks(logs["prefill"].get((0, rank), [])).items()}
                for rank in range(sizes["prefill"])}
    evidence: dict = {"dcp": dcps, "page_tokens": page, "ranks": {}}
    accepted: list[int] = []
    pairing: dict[str, dict[int, object]] = {"sources": {}, "received": {}}
    for rank in range(tp):
        records = logs["decode"].get((0, rank), [])
        dcp_rank = rank % decode_dcp
        sources = dcp_sources(rank, tp, sizes["prefill"], prefill_dcp, decode_dcp)
        loads = [r["kv_transfers"]["load"] for r in records if "load" in (r.get("kv_transfers") or {})]
        reads = [each for load in loads for each in load.get("each") or []]
        for load in loads:
            each = load.get("each") or []
            if (load.get("count"), load.get("bytes")) != (len(each), sum(int(e[0] or 0) for e in each if e)):
                problems.append(f"decode rank {rank}'s load of {load.get('count')} READ(s), {load.get('bytes')} "
                                f"bytes, is not its {len(each)} READ(s)' sum")
        unnamed = [each for each in reads if not each or int(each[0] or 0) > 0 and (
            len(each) < 6 or type(each[5]) is not int or each[5] not in sources)]
        if unnamed:
            problems.append(f"decode rank {rank} has {len(unnamed)} READ(s) of no producer rank among "
                            f"{sources}: {unnamed[0][:6]}")
        peers = sorted({each[5] for each in reads if len(each) > 5 and type(each[5]) is int})
        moved = sum(int(load.get("bytes") or 0) for load in loads)
        # Transfer checks only: a prefix hit's (`kind: hit`) holds cached
        # bytes, no transfer's (`_hit_problems`).
        transfers = [r["kv_check"] for r in records
                     if "kv_check" in r and r["kv_check"].get("kind") in (None, "transfer")]
        valid = [c for c in transfers if "error" not in c]
        checked = sum(int(c.get("bytes") or 0) for c in valid)
        if run.kv_check and checked != moved:
            problems.append(f"decode rank {rank} checked {checked} bytes, not the {moved} it read")
        notify_only = sum(1 for c in valid if (c.get("groups") or [])
                          and all(_dcp_notify_only(g) for g in c["groups"]))
        totals = {"payload": 0, "padding": 0, "untouched": 0, "padding_bytes": 0, "untouched_bytes": 0}
        miscounted = False
        held, prompts, done = _dcp_pulls(records)
        # Each pull's blocks as posted (past decode's own prefix hit): its
        # first place and count, all the request's blocks without a plan.
        plans = _pull_plans(records)
        spans = {request: plans.get(request, {}).get(0, (0, blocks)) for request, blocks in held.items()}
        for check in valid:
            found = check.get("dcp") or {}
            tokens = found.get("remote_num_tokens")
            if {key: found.get(key) for key in ("size", "rank", "remote_size")} != {
                    "size": decode_dcp, "rank": dcp_rank, "remote_size": prefill_dcp} or not isinstance(tokens, int):
                problems.append(f"decode rank {rank}'s check of {check.get('req_id')} is of decode context "
                                f"parallelism {found}, not size {decode_dcp} rank {dcp_rank} from size {prefill_dcp}")
                continue
            for group in check.get("groups") or []:
                count = sum(int(group.get(key) or 0) for key in ("payload_blocks", "padding_blocks", "untouched_blocks"))
                first = int(group.get("first_logical_block") or 0)
                handoff = -(-tokens // page) if page else None
                payload = sum(1 for k in range(first, first + count)
                              if handoff is not None and k * decode_dcp + dcp_rank < handoff)
                if group.get("payload_blocks") != payload:
                    miscounted = True
                    problems.append(f"decode rank {rank}'s check of {check.get('req_id')} has "
                                    f"{group.get('payload_blocks')} payload page(s), not the {payload} of its "
                                    f"{count} block(s) from {first} under the handoff's {handoff} page(s)")
                posted = spans.get(check.get("req_id"))
                if posted is not None and posted != (first, count):
                    miscounted = True
                    problems.append(f"decode rank {rank}'s check of {check.get('req_id')} covers {count} block(s) "
                                    f"from {first} (payload, moved and untouched padding), not its pull's "
                                    f"{posted[1]} from {posted[0]}")
                totals["payload"] += int(group.get("payload_blocks") or 0)
                totals["padding"] += int(group.get("padding_blocks") or 0)
                totals["untouched"] += int(group.get("untouched_blocks") or 0)
                totals["padding_bytes"] += int(group.get("padding_bytes") or 0)
                totals["untouched_bytes"] += int(group.get("untouched_bytes") or 0)
        layout = next((r["kv_layout"] for r in records if "kv_layout" in r), None) or {}
        groups = layout.get("groups") or []
        block_bytes = (int(groups[0]["block_bytes"]) * int(groups[0]["layers"])
                       if len(groups) == 1 and page else None)
        payload_blocks = 0
        if not held:
            problems.append(f"decode rank {rank} logged no pull")
        for request, blocks in held.items():
            tokens = prompts.get(request)
            if not blocks or tokens is None or not page:
                problems.append(f"decode rank {rank}'s pull of {request} logged no allocated blocks, prompt or page")
                continue
            first, count = spans[request]
            if first + count != -(-tokens // (page * decode_dcp)):
                problems.append(f"decode rank {rank}'s pull of {request} posts {count} block(s) from {first}, not "
                                f"the rest of its cdiv({tokens}, {page} * DCP {decode_dcp}) scheduler blocks")
            payload_blocks += sum(1 for k in range(first, first + count)
                                  if k * decode_dcp + dcp_rank < -(-tokens // page))
        pulled = sum(count for _, count in spans.values())
        if block_bytes is None:
            problems.append(f"decode rank {rank}'s layout is not one group of pages: {groups}")
        elif not payload_blocks * block_bytes <= moved <= pulled * block_bytes:
            problems.append(f"decode rank {rank} read {moved} bytes, not between its {payload_blocks} payload "
                            f"page(s) and its {pulled} pulled block(s) of {block_bytes} bytes")
        # The READs its pulls call for: one from each source holding some of
        # the pages its blocks hold (allocation padding included); a source
        # holding none is only notified.
        wanted: dict[int, int] = {}
        for request, (first, count) in spans.items():
            mine = {k * decode_dcp + dcp_rank for k in range(first, first + count)}
            for source in sources:
                theirs = produced.get(source, {}).get(_base_request_id(request), 0)
                if mine & {k * prefill_dcp + source % prefill_dcp for k in range(theirs)}:
                    wanted[source] = wanted.get(source, 0) + 1
        got: dict[int, int] = {}
        for each in reads:
            if each and int(each[0] or 0) > 0 and len(each) > 5 and type(each[5]) is int:
                got[each[5]] = got.get(each[5], 0) + 1
        if got != wanted:
            problems.append(f"decode rank {rank} read {dict(sorted(got.items()))} (READs by producer rank), not the "
                            f"{dict(sorted(wanted.items()))} its pulls' pages call for")
        pairing["sources"][rank] = sources
        pairing["received"][rank] = moved
        if unfinished := sorted(set(held) - done):
            problems.append(f"decode rank {rank} finished {len(set(held) & done)} of its {len(held)} pull(s): "
                            f"{unfinished[0]}")
        proven = not reads and held and not payload_blocks and not unfinished and (
            not run.kv_check or (valid and notify_only == len(valid) and not miscounted))
        if rank in unread and not proven:
            problems.append(f"decode rank {rank}'s TENT metrics lack tent_read_bytes_total, though it "
                            f"posted {len(reads)} READ(s), {len(held)} pull(s) with {payload_blocks} payload "
                            f"page(s), and {notify_only} of its {len(valid)} check(s) moved nothing")
        elif rank in unread:
            accepted.append(rank)
        evidence["ranks"][rank] = {"dcp_rank": dcp_rank, "sources": sources, "peers": peers, "reads": len(reads),
                                   "pulls": len(held), "payload_blocks": payload_blocks,
                                   "notify_only": notify_only, **totals}
    evidence["unread_accepted"] = accepted
    for rank in range(sizes["prefill"]):
        records = logs["prefill"].get((0, rank), [])
        held = {req for r in records for req in ((r.get("kv_events") or {}).get("allocated") or {})}
        done = {req for r in records for req in ((r.get("kv_events") or {}).get("sends_done") or [])}
        evidence.setdefault("released", {})[rank] = [len(held & done), len(held)]
        if held - done:
            problems.append(f"prefill rank {rank} finished sending {len(held & done)} of the {len(held)} "
                            "request(s) it held")
        # A finished send is also logged for a request its lease released:
        # released by its readers' notifications only, none after a lease.
        leases = [(r.get("kv_transfers") or {}).get("lease") or {} for r in records]
        expired = sum(int(lease.get("expired") or 0) for lease in leases)
        late = sum(int(lease.get("late_notifications") or 0) for lease in leases)
        if expired or late:
            problems.append(f"prefill rank {rank} released {expired} request(s) by lease expiry, with {late} "
                            "notification(s) after one: not by its readers' notifications")
    return {**evidence, **pairing, "problems": problems}


def _hetero_tp(run: Run, sizes: dict[str, int], logs: dict, ranks: dict, ports: dict | None,
               unwritten: set[int] = frozenset()) -> dict:
    """Unequal prefill and decode TP sizes, reconciled with the native
    pairing computed here: each decode rank's sources (decode TP >= prefill
    TP: rank d * P/D; else ranks d * P/D + i), each of its global heads sent
    by the lowest-ranked source holding it (a replicated producer's other
    ranks send nothing: Mooncake's answer without a payload, for each of
    the decode rank's requests); every rank's layout and decode check
    holding its own global heads; a head's bytes in a block of a layer (its
    slab) from each side's layout, the same on both; each pair's logged
    payload equal to the heads it carries; each Mooncake write's plan at
    those heads' places, in slabs; each NIXL READ's measured descriptors
    (as many as its hook saw, within NIXL's limit) each one layer's block of
    the heads it carries; each rank's router port carrying its own payload
    (prefill RX, decode TX); unique KV apart from the replicated payload.
    Producer ranks without a TENT write series (`unwritten`) must be such
    idle replicas."""

    problems: list[str] = []
    prefill_tp, decode_tp = sizes["prefill"], sizes["decode"]
    nixl = run.server.connector == "nixl"
    kv_heads = kv_heads_of(run.model)
    evidence: dict = {"sizes": sizes, "kv_heads": kv_heads}
    if kv_heads is None:
        return {**evidence, "problems": [f"{run.model}/config.json gives no num_key_value_heads"],
                "sources": {d: [0] for d in range(decode_tp)}, "received": {}}
    def layout(role: str, rank: int) -> dict | None:
        return next((r["kv_layout"] for r in logs[role].get((0, rank), []) if "kv_layout" in r), None)

    # Each group's kind, the same on every rank of both sides (a group's
    # correspondence across P and D), then its total KV heads.
    kind_lists = {(role, rank): [g.get("kind") for g in (layout(role, rank) or {}).get("groups") or []]
                  for role in sizes for rank in range(sizes[role])}
    kinds = next((k for k in kind_lists.values() if k), [])
    for (role, rank), found in sorted(kind_lists.items()):
        if found != kinds:
            problems.append(f"{role} rank {rank}'s groups are {found}, not the others' {kinds}")
    if "MambaSpec" in kinds:
        return _hetero_hybrid(run, sizes, logs, ranks, ports, unwritten, kinds, kv_heads, evidence, problems, layout)
    totals, why = group_kv_heads(run.model, kinds) if kinds else ([kv_heads], None)
    if why:
        return {**evidence, "problems": problems + [why], "sources": {d: [0] for d in range(decode_tp)},
                "received": {}}
    evidence["group_kv_heads"] = totals
    gheads = {role: {rank: [global_heads(rank, sizes[role], k) for k in totals] for rank in range(sizes[role])}
              for role in sizes}
    heads = {role: {rank: gheads[role][rank][0] for rank in gheads[role]} for role in sizes}
    ratio = max(prefill_tp, decode_tp) // min(prefill_tp, decode_tp)
    paired = {d: [d * prefill_tp // decode_tp] if decode_tp >= prefill_tp else
              [d * ratio + i for i in range(ratio)] for d in range(decode_tp)}
    gsender = [{(h, d): min(p for p in paired[d] if h in gheads["prefill"][p][g])
                for d in range(decode_tp) for h in gheads["decode"][d][g]} for g in range(len(totals))]
    sender = gsender[0]
    # NIXL reads each unique head once (its first rank); Mooncake's decode
    # pulls every paired producer.
    sources = {d: sorted({gsender[g][(h, d)] for g in range(len(totals)) for h in gheads["decode"][d][g]})
               if nixl else paired[d] for d in range(decode_tp)}
    evidence["heads"] = heads
    evidence["sources"] = sources

    # Each group's head slab: its page (padding excluded) over its heads,
    # the same on every rank of both sides.
    slabs: dict[tuple[str, int], tuple[int | None, ...]] = {}
    for role in sizes:
        for rank in range(sizes[role]):
            groups = (layout(role, rank) or {}).get("groups") or []
            mine_slabs = []
            for index, group in enumerate(groups):
                want = gheads[role][rank][index] if index < len(totals) else None
                if want is None or group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                    problems.append(f"{role} rank {rank}'s group {index} holds {group.get('pattern')} heads "
                                    f"{group.get('global_heads')}, not head-v1 heads {want}")
                    mine_slabs.append(None)
                    continue
                # A group of several page sizes (sparse MLA: a latent's and an
                # indexer's layers) has a slab for each.
                pages = sorted({kind.get("real_page_bytes") or kind.get("page_bytes")
                                for kind in group.get("kinds") or []}, key=str) or [
                    group.get("real_page_bytes") or group.get("block_bytes")]
                good = all(isinstance(page, int) and page > 0 and page % len(want) == 0 for page in pages)
                each = tuple(page // len(want) for page in pages) if good else None
                mine_slabs.append(each[0] if each and len(each) == 1 else each)
            slabs[(role, rank)] = tuple(mine_slabs) or (None,)
    distinct = set(slabs.values())
    gslabs = list(next(iter(distinct))) if len(distinct) == 1 else [None]
    if None in gslabs or len(distinct) != 1:
        problems.append(f"the ranks' layouts give no single head size in a block of a layer: "
                        f"{ {f'{role}{rank}': (n[0] if len(set(n)) == 1 else n) for (role, rank), n in sorted(slabs.items())} }")
        gslabs = [None] * len(totals)
    slab = gslabs[0] if len(set(gslabs)) == 1 else gslabs
    evidence["slab_bytes"] = list(slab) if isinstance(slab, tuple) else slab

    expected: dict[tuple[int, int], int] = {}
    unique = 0
    for d in range(decode_tp):
        mine = heads["decode"][d]
        for record in logs["decode"].get((0, d), []):
            check = record.get("kv_check")
            if not check or check.get("kind") not in (None, "transfer") or "error" in check:
                continue
            cgroups = check.get("groups") or []
            for position, group in enumerate(cgroups):
                index = group.get("group", position)
                want = gheads["decode"][d][index] if isinstance(index, int) and 0 <= index < len(totals) else None
                if want is None or group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                    problems.append(f"decode rank {d} checked {check.get('req_id')}'s heads "
                                    f"{group.get('global_heads')} ({group.get('pattern')}), not its {want}")
            # Bytes per head, each group's own (its bytes over its heads); a
            # check without per-group bytes counts one set of heads.
            per_group = []
            if cgroups and all(isinstance(g.get("bytes"), int) and isinstance(g.get("group", 0), int)
                               and 0 <= g.get("group", 0) < len(totals) for g in cgroups):
                for position, group in enumerate(cgroups):
                    index = group.get("group", position)
                    per_group.append((index, group["bytes"] // len(gheads["decode"][d][index])))
            else:
                per_group = [(0, int(check.get("bytes") or 0) // len(mine))]
            for index, per_head in per_group:
                if d == 0:
                    # Each request's KV once: every global head's bytes.
                    unique += per_head * totals[index]
                for p in {gsender[index][(h, d)] for h in gheads["decode"][d][index]}:
                    expected[(p, d)] = expected.get((p, d), 0) + per_head * sum(
                        1 for h in gheads["decode"][d][index] if gsender[index][(h, d)] == p)

    # Each decode rank's requests (its byte checks'), which each paired
    # producer answers.
    requested = {d: {_base_request_id(r["kv_check"]["req_id"]) for r in logs["decode"].get((0, d), [])
                     if "req_id" in (r.get("kv_check") or {}) and r["kv_check"].get("kind") in (None, "transfer")}
                 for d in range(decode_tp)}
    logged: dict[tuple[int, int], int] = {}
    answers: dict[tuple[int, int], list[str]] = {}
    # Producer ranks with an answer the pairing or its plan refutes.
    misanswered: set[int] = set()
    descriptors: list[int] = []
    merged: list[int] = []
    unknown = 0
    for role, mover in (("prefill", not nixl), ("decode", nixl)):
        if not mover:
            continue
        direction = "load" if nixl else "send"
        for (_, rank), records in logs[role].items():
            for record in records:
                transfer = (record.get("kv_transfers") or {}).get(direction) or {}
                for entry in transfer.get("each") or []:
                    peer = entry[5] if len(entry) > 5 else None
                    plan = entry[6] if len(entry) > 6 else None
                    if peer is None:
                        unknown += 1
                        continue
                    pair = (peer, rank) if nixl else (rank, peer)
                    logged[pair] = logged.get(pair, 0) + int(entry[0])
                    if nixl and isinstance(plan, list) and len(plan) > 1 and isinstance(plan[1], int):
                        descriptors.append(plan[1])
                    if nixl and isinstance(entry[3], int):
                        merged.append(entry[3])
                    problems.extend(_plan_problems(pair, entry, plan, nixl, gheads, gsender, sources, gslabs, descriptor_limit(run.server)))
                for answer in transfer.get("zero_payload") or []:
                    peer, requests, plan = (list(answer) + [None, None, None])[:3]
                    if not isinstance(peer, int) or rank not in paired.get(peer, ()):
                        problems.append(f"prefill rank {rank} answered decode rank {peer}, not one it pairs with")
                        misanswered.add(rank)
                        continue
                    if not isinstance(requests, list):
                        problems.append(f"prefill rank {rank} answered decode rank {peer} without a payload "
                                        f"for unnamed requests ({requests})")
                        misanswered.add(rank)
                        continue
                    answers.setdefault((rank, peer), []).extend(map(_base_request_id, requests))
                    if not plan or any(not isinstance(region, list) or region[0] for region in plan):
                        problems.append(f"prefill rank {rank} answered decode rank {peer} without a payload "
                                        f"under a plan that sends (or none): {plan}")
                        misanswered.add(rank)
    if unknown:
        problems.append(f"{unknown} transfer(s) logged without their peer rank")
    for pair in sorted(set(expected) | set(logged)):
        if logged.get(pair, 0) != expected.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]} moved {logged.get(pair, 0)} bytes, "
                            f"not the {expected.get(pair, 0)} of the heads it holds for it")
    idle: set[int] = set()
    if not nixl:
        # Paired producers that hold no head they send: a replica's answer
        # for each of the decode rank's requests, once.
        covered = {p: p not in misanswered for p in range(prefill_tp)}
        for d in range(decode_tp):
            for p in paired[d]:
                answered = answers.get((p, d), [])
                if (p, d) in expected:
                    if answered:
                        problems.append(f"prefill rank {p} answered decode rank {d} without a payload for "
                                        f"{len(answered)} request(s), though it holds heads it sends")
                    covered[p] = False
                    continue
                twice = sorted({r for r in answered if answered.count(r) > 1})
                missing = sorted(requested[d] - set(answered))
                extra = sorted(set(answered) - requested[d])
                for what, which in (
                        (f"answered decode rank {d} without a payload twice for", twice),
                        (f"never answered decode rank {d} (no payload and no zero-payload answer) for", missing),
                        (f"answered decode rank {d} without a payload for requests it never pulled:", extra)):
                    if which:
                        problems.append(f"prefill rank {p} {what} {len(which)} request(s), {which[0]}")
                if twice or missing or extra or not requested[d]:
                    covered[p] = False
        idle = {p for p, ok in covered.items() if ok and not any(pp == p for pp, _ in logged)
                and any(p in paired[d] for d in range(decode_tp))}
    for p in sorted(unwritten - idle):
        problems.append(f"prefill rank {p}'s TENT metrics lack tent_write_bytes_total, and it is not a replica "
                        "whose zero-payload answers cover each of its decode ranks' requests")
    evidence["idle_replicas"] = sorted(idle)
    evidence["unwritten_accepted"] = sorted(unwritten & idle)
    if nixl and descriptor_limit(run.server) is not None and descriptors and max(descriptors) > descriptor_limit(run.server):
        problems.append(f"a NIXL READ selected {max(descriptors)} descriptors, more than the "
                        f"{NIXL_MAX_DESCRIPTORS} guard")

    received = {d: sum(n for (p, dd), n in logged.items() if dd == d) for d in range(decode_tp)}
    sent = {p: sum(n for (pp, d), n in logged.items() if pp == p) for p in range(prefill_tp)}
    payload = sum(logged.values())
    mover_role = "decode" if nixl else "prefill"
    moved = sum(summary["transfers"]["bytes"] for summary in ranks[mover_role].values())
    if moved != payload:
        problems.append(f"the connector moved {moved} bytes, not the {payload} its transfers' peers account for")
    for p in range(prefill_tp):
        produced = ranks["prefill"].get((0, p), {}).get("requests") or set()
        if run.kv_check and len(produced) != run.workload.num_prompts:
            problems.append(f"prefill rank {p} produced {len(produced)} request(s), not {run.workload.num_prompts}")

    port_evidence = {}
    if ports is not None:
        network = ports.get(run.prefill.kv_network) or {}
        by_rank = {(described["node"], described["slot"]): described for described in network.values()}
        for role, node, carried, direction in (("prefill", run.prefill.node, sent, "rx"),
                                               ("decode", run.decode.node, received, "tx")):
            for rank, payload_bytes in carried.items():
                port = by_rank.get((node, rank))
                if port is None:
                    problems.append(f"{role} rank {rank}'s router port was not measured")
                    continue
                measured = port[direction]
                port_evidence[f"{role}{rank}"] = {"rx": port["rx"], "tx": port["tx"], "payload": payload_bytes}
                if measured < payload_bytes or measured > payload_bytes * 1.05 + (1 << 20):
                    problems.append(f"{role} rank {rank}'s router port {direction.upper()} {measured} bytes, not its "
                                    f"{payload_bytes} of KV and little more")
    else:
        problems.append("no router port counters: unequal TP needs the switched network's")

    evidence.update({
        "pairs": [{"prefill": p, "decode": d, "bytes": logged.get((p, d), 0), "expected": expected.get((p, d), 0),
                   "zero_payload_answers": len(answers.get((p, d), []))}
                  for p in range(prefill_tp) for d in range(decode_tp)
                  if (p, d) in expected or (p, d) in logged or (p, d) in answers],
        "unique_kv_bytes": unique, "payload_bytes": payload,
        "replication_factor": round(payload / unique, 6) if unique else None,
        "max_descriptors": max(descriptors, default=None), "max_merged_descriptors": max(merged, default=None),
        "ports": port_evidence or None,
        "received": received, "problems": problems,
    })
    return evidence


def _plan_problems(pair: tuple[int, int], entry: list, plan: list | None, nixl: bool, heads: dict,
                   sender: dict, sources: dict, slab: int | None,
                   max_descriptors: int | None = NIXL_MAX_DESCRIPTORS) -> list[str]:
    """A transfer's logged plan against the heads its pair carries, in
    slabs (a head's bytes in a block of a layer, from the layouts). NIXL's
    [local slot, selected, predicted]: the slot the source's order gives;
    the descriptors vLLM selected (each one layer's block of the carried
    heads, so they give the bytes; within the guard) and the hook's
    prediction of the descriptors NIXL merges them into, which must be what
    NIXL measured (`entry[3]`), never more than selected; a READ without
    that prediction fails. Each of Mooncake's regions [sends,
    source offset, destination offset, length]: the first carried head's
    places in the producer's and decode's pages, the carried heads' length."""
    p, d = pair
    at = f"prefill rank {p} -> decode rank {d}: "
    # `heads`, `sender` and `slab` are each KV cache group's (lists by group).
    by_group = [[h for h in heads["decode"][d][g] if sender[g][(h, d)] == p] for g in range(len(sender))]
    carried = next((c for c in by_group if c), [])
    if not carried:
        return [f"{at}a transfer, though decode takes none of its heads from it"]
    known = all(n is not None for n in slab)
    # A group's slabs: one, or one for each of its page sizes.
    slabs = [n if isinstance(n, tuple) else (n,) for n in slab]
    several = any(len(n) > 1 for n in slabs)
    spans = {len(c) * n for g, c in enumerate(by_group) if c and slab[g] is not None for n in slabs[g]}
    if not plan:
        return [f"{at}a transfer of {entry[0]} bytes without its plan ({plan})"]
    if nixl:
        problems = []
        slot = sources[d].index(p) if p in sources[d] and len(sources[d]) > 1 else 0
        if plan[0] != slot:
            problems.append(f"{at}a READ into slot {plan[0]}, not {slot}")
        measured = entry[3]
        selected, predicted = (list(plan) + [None, None, None])[1:3]

        def count(value) -> bool:
            return isinstance(value, int) and not isinstance(value, bool) and value > 0

        if not count(selected):
            problems.append(f"{at}a READ without its selected descriptors ({selected})")
            return problems
        if not count(measured):
            problems.append(f"{at}a READ without NIXL's measured descriptors ({measured}; {selected} selected)")
        if not count(predicted):
            problems.append(f"{at}a READ without the hook's predicted merged descriptors ({predicted}; "
                            f"{selected} selected, NIXL measured {measured})")
        elif predicted > selected:
            problems.append(f"{at}a READ predicted at {predicted} merged descriptors, more than its {selected} "
                            "selected")
        if count(measured) and count(predicted) and measured != predicted:
            problems.append(f"{at}a READ NIXL measured at {measured} descriptors, predicted at {predicted} "
                            f"({selected} selected)")
        if count(measured) and measured > selected:
            problems.append(f"{at}a READ NIXL measured at {measured} descriptors, more than its {selected} "
                            "selected")
        if max_descriptors is not None and selected > max_descriptors:
            problems.append(f"{at}a READ of {selected} selected descriptors, more than the "
                            f"{NIXL_MAX_DESCRIPTORS} guard")
        if known and several:
            # Descriptors of several sizes: the READ's own count of each.
            sizes = plan[4] if len(plan) > 4 and isinstance(plan[4], dict) else None
            counted = {int(size): n for size, n in (sizes or {}).items()}
            if sizes is None:
                problems.append(f"{at}a READ of several page sizes without its descriptors by size")
            elif (set(counted) != spans or sum(counted.values()) != selected
                  or sum(size * n for size, n in counted.items()) != int(entry[0])):
                problems.append(f"{at}a READ of {entry[0]} bytes in {selected} selected descriptors of sizes "
                                f"{counted}, not the carried spans {sorted(spans)} adding up to both")
        elif known and len(spans) != 1:
            problems.append(f"{at}a READ whose groups carry different spans {sorted(spans)}: the exact bytes "
                            "need one common carried span per descriptor")
        elif known:
            span = spans.pop()
            if int(entry[0]) != selected * span:
                if len({len(c) for c in by_group if c}) == 1 and len({n for n in slab}) == 1:
                    problems.append(f"{at}a READ of {entry[0]} bytes in {selected} selected descriptors, not "
                                    f"{len(carried)} head(s) of {slab[0]} bytes each")
                else:
                    problems.append(f"{at}a READ of {entry[0]} bytes in {selected} selected descriptors, not "
                                    f"{span} bytes (the carried span) each")
        return problems
    if not known:
        return []
    problems = []
    wants = []
    for g, c in enumerate(by_group):
        for n in slabs[g] if c else ():
            want = [True, heads["prefill"][p][g].index(c[0]) * n, heads["decode"][d][g].index(c[0]) * n,
                    len(c) * n]
            if want not in wants:
                wants.append(want)
    for region in plan:
        if list(region) not in wants:
            shown = wants[0] if len(wants) == 1 else wants
            problems.append(f"{at}plan {list(region)}, not {shown} (heads {carried}, "
                            f"{slabs[0][0] if not several else [list(n) for n in slabs]} bytes each)")
    if several and sorted(map(list, plan)) != sorted(wants):
        problems.append(f"{at}plan {[list(region) for region in plan]}, not a region for each of {wants}")
    return problems


def _state_spans(state: dict, rank: int, tp: int) -> list | None:
    """A rank's state spans [kind, component, offset, length, first, count]
    recomputed from the layout record's global components (the model's, the
    same on every rank): conv components in order, each its columns
    [rank*w/tp, ...) of `conv_rows * conv_itemsize` bytes, then the SSM
    heads. None when the layout does not split over tp."""
    column = state["conv_rows"] * state["conv_itemsize"]
    units = [w for _, w in state["components"]] + [state["ssm_heads"]]
    if tp < 1 or any(w % tp for w in units):
        return None
    spans, offset = [], 0
    for name, width in state["components"]:
        n = width // tp
        spans.append(["conv", name, offset, n * column, rank * n, n])
        offset += n * column
    n = state["ssm_heads"] // tp
    spans.append(["ssm", "ssm", offset, n * state["ssm_head_bytes"], rank * n, n])
    return spans


def _producers(rank: int, local_tp: int, remote_tp: int) -> list[int]:
    """The producers of a decode rank's pairing: one when the producer has
    no more ranks (floor(r*R/L)), else its R/L contiguous ones."""
    if remote_tp <= local_tp:
        return [rank * remote_tp // local_tp]
    k = remote_tp // local_tp
    return list(range(rank * k, (rank + 1) * k))


def _state_overlap(rank: int, local_tp: int, remote_tp: int, width: int) -> dict[int, tuple[int, int]]:
    """producer -> the global range [first, last) of a state component it
    supplies a decode rank: every candidate whose range overlaps the rank's."""
    lo, hi = rank * width // local_tp, (rank + 1) * width // local_tp
    out = {}
    for p in _producers(rank, local_tp, remote_tp):
        first, last = max(lo, p * width // remote_tp), min(hi, (p + 1) * width // remote_tp)
        if first < last:
            out[p] = (first, last)
    return out


def _hetero_hybrid(run: Run, sizes: dict[str, int], logs: dict, ranks: dict, ports: dict | None,
                   unwritten: set[int], kinds: list, kv_heads: int, evidence: dict, problems: list[str],
                   layout) -> dict:
    """Unequal TP sizes for a hybrid model over NIXL (its attention groups
    head-v1, its state groups state-v1), reconciled with ownership computed
    here: each state group's rank spans recomputed from the layout record's
    global components (never trusted), each decode rank's attention heads from
    the first candidate producer holding each (the union over heads) and its
    state ranges from every candidate whose range overlaps; each pair's bytes
    from the checks' per-group blocks, selected regions and spans (logical
    owned state, transfer-only regions and attention kept apart); each READ's
    exact bytes (its selected-byte sum and its length histogram) and the
    checker's per-READ record (producer, bytes by group, component and class,
    completion); NIXL's selected/predicted/measured descriptors. Mooncake does
    not pair a hybrid."""
    prefill_tp, decode_tp = sizes["prefill"], sizes["decode"]
    nixl = run.server.connector == "nixl"
    evidence["hybrid"] = True
    done = {"sources": {d: [0] for d in range(decode_tp)}, "received": {}}
    if not nixl:
        return {**evidence, "problems": problems + ["a hybrid model pairs unequal TP sizes over NIXL only"], **done}
    groups_of = lambda role, rank: (layout(role, rank) or {}).get("groups") or []
    state_idx = [i for i, k in enumerate(kinds) if k == "MambaSpec"]
    attn_idx = [i for i, k in enumerate(kinds) if k != "MambaSpec"]
    reference = None
    for role in sizes:
        for rank in range(sizes[role]):
            for index, group in enumerate(groups_of(role, rank)):
                if index in attn_idx:
                    want = global_heads(rank, sizes[role], kv_heads)
                    if group.get("pattern") != "head-v1" or group.get("global_heads") != want:
                        problems.append(f"{role} rank {rank}'s group {index} holds {group.get('pattern')} heads "
                                        f"{group.get('global_heads')}, not head-v1 heads {want}")
                    continue
                state = group.get("state")
                if group.get("pattern") != "state-v1" or not isinstance(state, dict):
                    problems.append(f"{role} rank {rank}'s group {index} is {group.get('pattern')}, not state-v1")
                    continue
                model = {k: state.get(k) for k in ("family", "components", "conv_rows", "conv_itemsize",
                                                   "ssm_heads", "ssm_head_bytes")}
                reference = reference or model
                if model != reference:
                    problems.append(f"{role} rank {rank}'s group {index} state layout {model} differs from {reference}")
                want_spans = _state_spans(reference, rank, sizes[role])
                if want_spans is None:
                    problems.append(f"{role} rank {rank}: the state does not split over TP {sizes[role]}")
                elif state.get("spans") != want_spans:
                    problems.append(f"{role} rank {rank}'s group {index} state spans {state.get('spans')}, "
                                    f"not the layout's {want_spans}")
    if reference is None or problems:
        return {**evidence, "problems": problems + ([] if reference else ["no state group's layout record"]), **done}
    evidence["state_layout"] = reference
    column = reference["conv_rows"] * reference["conv_itemsize"]

    # Attention sources: for each local head the first candidate holding it.
    def attention(d: int) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for h in global_heads(d, decode_tp, kv_heads):
            owner = next((q for q in _producers(d, decode_tp, prefill_tp)
                          if h in global_heads(q, prefill_tp, kv_heads)), None)
            if owner is None:
                problems.append(f"no producer of decode rank {d} holds head {h}")
            else:
                out.setdefault(owner, []).append(h)
        return out

    sources = {}
    for d in range(decode_tp):
        got = set(attention(d))
        for width in [w for _, w in reference["components"]] + [reference["ssm_heads"]]:
            got |= set(_state_overlap(d, decode_tp, prefill_tp, width))
        sources[d] = sorted(got)
    evidence["sources"] = sources
    expected: dict[tuple[int, int], int] = {}
    classes: dict[tuple[int, int, str], int] = {}
    attention_expected: dict[tuple[int, int], int] = {}
    totals = {"attention": 0, "state_logical": 0, "state_transfer_only": 0}
    unique = {"attention": 0, "state_logical": 0, "state_transfer_only": 0}
    checks: dict[int, list[dict]] = {d: [] for d in range(decode_tp)}
    for d in range(decode_tp):
        local = _state_spans(reference, d, decode_tp)
        span_bytes = sum(sp[3] for sp in local)
        for record in logs["decode"].get((0, d), []):
            check = record.get("kv_check")
            if not check or check.get("kind") not in (None, "transfer") or "error" in check:
                continue
            checks[d].append(check)
            coverage = check.get("coverage") or {}
            if not isinstance(coverage, dict) or "reads" not in coverage:
                problems.append(f"decode rank {d}'s check of {check.get('req_id')} has no READ coverage")
            for text in coverage.get("problems") or []:
                problems.append(f"decode rank {d} {check.get('req_id')}: {text}")
            if check.get("posted_matches_checked") is not True:
                problems.append(f"decode rank {d}'s check of {check.get('req_id')} did not match its posted READs")
            for read in coverage.get("reads") or []:
                if read.get("completed") is not True:
                    problems.append(f"decode rank {d}: a READ from {read.get('rank')} was not complete")
                if sum(c[1] for c in (read.get("classes") or {}).values()) != read.get("bytes"):
                    problems.append(f"decode rank {d}: a READ's classes sum to "
                                    f"{sum(c[1] for c in (read.get('classes') or {}).values())}, not {read.get('bytes')}")
            for position, group in enumerate(check.get("groups") or []):
                index = group.get("group", position)
                if index in attn_idx:
                    mine = global_heads(d, decode_tp, kv_heads)
                    if group.get("pattern") != "head-v1" or group.get("global_heads") != mine:
                        problems.append(f"decode rank {d} checked {check.get('req_id')}'s heads "
                                        f"{group.get('global_heads')} ({group.get('pattern')}), not its {mine}")
                    if not isinstance(group.get("bytes"), int) or group["bytes"] % len(mine):
                        problems.append(f"decode rank {d}: attention group {index} moved {group.get('bytes')} bytes "
                                        f"over {len(mine)} heads")
                        continue
                    # What the READs carried: the checked bytes but the
                    # sub-blocks NIXL zeroed (a smaller producer block size).
                    per_head = (group["bytes"] - int(group.get("zeroed_bytes") or 0)) // len(mine)
                    if d == 0:
                        unique["attention"] += per_head * kv_heads
                    totals["attention"] += group["bytes"]
                    for p, heads in attention(d).items():
                        expected[(p, d)] = expected.get((p, d), 0) + per_head * len(heads)
                        attention_expected[(p, d)] = attention_expected.get((p, d), 0) + per_head * len(heads)
                    continue
                if index not in state_idx or group.get("pattern") != "state-v1":
                    problems.append(f"decode rank {d} checked group {index} as {group.get('pattern')}, "
                                    "not a state-v1 state group")
                    continue
                blocks, regions = group.get("blocks"), group.get("regions")
                if not (isinstance(blocks, int) and isinstance(regions, int) and blocks > 0 and regions > 0):
                    problems.append(f"decode rank {d}: state group {index} reports {blocks} blocks, {regions} regions")
                    continue
                if group.get("bytes") != blocks * regions * span_bytes:
                    problems.append(f"decode rank {d}: state group {index} moved {group.get('bytes')} bytes, not "
                                    f"{blocks} blocks x {regions} regions x {span_bytes} of its spans")
                transfer_only = group.get("transfer_only_bytes", 0)
                if transfer_only % (blocks * span_bytes):
                    problems.append(f"decode rank {d}: state group {index}'s transfer-only bytes {transfer_only} "
                                    f"are not whole regions of {span_bytes}")
                    continue
                only_regions = transfer_only // (blocks * span_bytes)
                logical_regions = regions - only_regions
                totals["state_transfer_only"] += transfer_only
                totals["state_logical"] += group["bytes"] - transfer_only
                if d == 0:
                    whole = sum(w for _, w in reference["components"]) * column + (
                        reference["ssm_heads"] * reference["ssm_head_bytes"])
                    unique["state_logical"] += blocks * logical_regions * whole
                    unique["state_transfer_only"] += blocks * only_regions * whole
                for sp in local:
                    width = (dict((n, w) for n, w in reference["components"])[sp[1]] if sp[0] == "conv"
                             else reference["ssm_heads"])
                    unit = column if sp[0] == "conv" else reference["ssm_head_bytes"]
                    for p, (lo, hi) in _state_overlap(d, decode_tp, prefill_tp, width).items():
                        for kind, regs in (("logical", logical_regions), ("transfer-only", only_regions)):
                            n = blocks * regs * (hi - lo) * unit
                            if n:
                                classes[(p, d, f"g{index}.{sp[0]}.{sp[1]}.{kind}")] = (
                                    classes.get((p, d, f"g{index}.{sp[0]}.{sp[1]}.{kind}"), 0) + n)
                                expected[(p, d)] = expected.get((p, d), 0) + n

    # Per-READ telemetry: exact bytes and lengths, then NIXL's three counts.
    logged: dict[tuple[int, int], int] = {}
    seen_classes: dict[tuple[int, int, str], int] = {}
    attention_seen: dict[tuple[int, int], int] = {}
    telemetry: dict[int, list[tuple]] = {d: [] for d in range(decode_tp)}
    descriptors, merged, unknown = [], [], 0
    for (_, rank), records in logs["decode"].items():
        for record in records:
            for entry in ((record.get("kv_transfers") or {}).get("load") or {}).get("each") or []:
                peer = entry[5] if len(entry) > 5 else None
                plan = entry[6] if len(entry) > 6 else None
                if peer is None:
                    unknown += 1
                    continue
                at = f"prefill rank {peer} -> decode rank {rank}: "
                logged[(peer, rank)] = logged.get((peer, rank), 0) + int(entry[0])
                if not (isinstance(plan, list) and len(plan) > 4):
                    problems.append(f"{at}a READ without its selected bytes and lengths ({plan})")
                    continue
                selected, predicted, sel_bytes, table = plan[1], plan[2], plan[3], plan[4]
                measured = entry[3]
                if not (isinstance(table, dict) and isinstance(sel_bytes, int)):
                    problems.append(f"{at}a READ without its selected bytes and lengths ({plan[3:]})")
                    continue
                if int(entry[0]) != sel_bytes:
                    problems.append(f"{at}a READ of {entry[0]} bytes, its selected descriptors sum to {sel_bytes}")
                if sum(int(n) * c for n, c in table.items()) != sel_bytes or sum(table.values()) != selected:
                    problems.append(f"{at}a READ's length table {table} is not its {selected} descriptors of "
                                    f"{sel_bytes} bytes")
                count = lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0
                if not (count(selected) and count(measured) and count(predicted)):
                    problems.append(f"{at}a READ without counts: selected {selected}, predicted {predicted}, "
                                    f"measured {measured}")
                else:
                    descriptors.append(selected)
                    merged.append(measured)
                    if not predicted == measured <= selected:
                        problems.append(f"{at}a READ of {selected} selected descriptors predicted at {predicted} "
                                        f"and measured at {measured} (need predicted = measured <= selected)")
                    if descriptor_limit(run.server) is not None and selected > descriptor_limit(run.server):
                        problems.append(f"{at}a READ of {selected} selected descriptors, more than the "
                                        f"{NIXL_MAX_DESCRIPTORS} guard")
                telemetry[rank].append((peer, int(entry[0]), selected))
    if unknown:
        problems.append(f"{unknown} transfer(s) logged without their peer rank")
    for d in range(decode_tp):
        posted = sorted((r.get("rank"), r.get("bytes"), r.get("descriptors")) for c in checks[d]
                        for r in (c.get("coverage") or {}).get("reads") or [])
        for c in checks[d]:
            for r in (c.get("coverage") or {}).get("reads") or []:
                for name, (_, n) in (r.get("classes") or {}).items():
                    if ".attention." in name:
                        attention_seen[(r.get("rank"), d)] = attention_seen.get((r.get("rank"), d), 0) + n
                        continue
                    key = (r.get("rank"), d, name)
                    seen_classes[key] = seen_classes.get(key, 0) + n
        if posted != sorted(telemetry[d]):
            problems.append(f"decode rank {d}'s checked READs {posted[:3]}... are not its logged READs "
                            f"{sorted(telemetry[d])[:3]}... (producer, bytes, selected descriptors)")
    for key in sorted(set(classes) | set(seen_classes)):
        if classes.get(key, 0) != seen_classes.get(key, 0):
            problems.append(f"producer {key[0]} -> decode rank {key[1]}: {key[2]} carried {seen_classes.get(key, 0)} "
                            f"bytes, not the {classes.get(key, 0)} its owned range gives")
    for pair in sorted(set(attention_expected) | set(attention_seen)):
        if attention_expected.get(pair, 0) != attention_seen.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]}: attention carried "
                            f"{attention_seen.get(pair, 0)} bytes, not the {attention_expected.get(pair, 0)} "
                            "of the heads it holds for it")
    for pair in sorted(set(expected) | set(logged)):
        if logged.get(pair, 0) != expected.get(pair, 0):
            problems.append(f"prefill rank {pair[0]} -> decode rank {pair[1]} moved {logged.get(pair, 0)} bytes, "
                            f"not the {expected.get(pair, 0)} its owned heads and state ranges give")
    for p in sorted(unwritten):
        problems.append(f"prefill rank {p}'s TENT metrics lack tent_write_bytes_total")
    received = {d: sum(n for (p, dd), n in logged.items() if dd == d) for d in range(decode_tp)}
    sent = {p: sum(n for (pp, d), n in logged.items() if pp == p) for p in range(prefill_tp)}
    payload = sum(logged.values())
    moved = sum(summary["transfers"]["bytes"] for summary in ranks["decode"].values())
    if moved != payload:
        problems.append(f"the connector moved {moved} bytes, not the {payload} its transfers' peers account for")
    for p in range(prefill_tp):
        produced = ranks["prefill"].get((0, p), {}).get("requests") or set()
        if run.kv_check and len(produced) != run.workload.num_prompts:
            problems.append(f"prefill rank {p} produced {len(produced)} request(s), not {run.workload.num_prompts}")
    port_evidence = {}
    if ports is not None:
        network = ports.get(run.prefill.kv_network) or {}
        by_rank = {(described["node"], described["slot"]): described for described in network.values()}
        for role, node, carried, direction in (("prefill", run.prefill.node, sent, "rx"),
                                               ("decode", run.decode.node, received, "tx")):
            for rank, payload_bytes in carried.items():
                port = by_rank.get((node, rank))
                if port is None:
                    problems.append(f"{role} rank {rank}'s router port was not measured")
                    continue
                measured = port[direction]
                port_evidence[f"{role}{rank}"] = {"rx": port["rx"], "tx": port["tx"], "payload": payload_bytes}
                if measured < payload_bytes or measured > payload_bytes * 1.05 + (1 << 20):
                    problems.append(f"{role} rank {rank}'s router port {direction.upper()} {measured} bytes, not its "
                                    f"{payload_bytes} of KV and little more")
    else:
        problems.append("no router port counters: unequal TP needs the switched network's")
    evidence.update({
        "pairs": [{"prefill": p, "decode": d, "bytes": logged.get((p, d), 0), "expected": expected.get((p, d), 0)}
                  for p in range(prefill_tp) for d in range(decode_tp) if (p, d) in expected or (p, d) in logged],
        "byte_classes": totals, "unique_bytes_by_class": unique, "payload_bytes": payload,
        "max_descriptors": max(descriptors, default=None), "max_merged_descriptors": max(merged, default=None),
        "state_classes": {f"{p}->{d} {name}": n for (p, d, name), n in sorted(classes.items())},
        "ports": port_evidence or None, "received": received, "problems": problems,
    })
    return evidence



def _hp_tcp_writes(records: list[dict], limit: int | None) -> dict:
    """A Mooncake prefill rank's connector writes (``send``) against hp_tcp's
    task limit, reported, never judged: TENT may merge a write's descriptors
    into fewer tasks, and how many of another write's tasks are still
    outstanding when one is submitted is not logged. Each write spans
    [end - seconds, end]; a failed one [start, its record's time] (its end is
    not logged; the record reporting it is written after it). The
    overlapping-call bound is the most descriptors whose spans overlap: an
    upper bound on what was in flight, complete only if every write's span
    is known."""
    spans: list[tuple[float, float, int]] = []
    largest: list[int] = []
    complete = True
    written = failed = 0
    for record in records:
        send = (record.get("kv_transfers") or {}).get("send") or {}
        each = send.get("each") or []
        written += int(send.get("count") or 0)
        complete &= len(each) == int(send.get("count") or 0)
        for entry in each:
            descriptors = entry[3] if len(entry) > 4 else None
            if descriptors is None:
                complete = False
                continue
            largest.append(descriptors)
            if entry[1] is None or entry[4] is None:
                complete = False
                continue
            spans.append((entry[4] - entry[1], entry[4], descriptors))
        for entry in send.get("failed_each") or []:
            failed += 1
            descriptors, start = (entry[1], entry[2]) if len(entry) > 2 else (None, None)
            if descriptors is None:
                complete = False
                continue
            largest.append(descriptors)
            end = record.get("t")
            if start is None or end is None or end < start:
                complete = False
                continue
            spans.append((start, end, descriptors))
        # Failures counted but not each listed have no span.
        complete &= len(send.get("failed_each") or []) >= int(send.get("failed") or 0)
        failed += max(0, int(send.get("failed") or 0) - len(send.get("failed_each") or []))
    in_flight = peak = 0
    for _, change in sorted([(a, n) for a, _, n in spans] + [(b, -n) for _, b, n in spans],
                            key=lambda event: (event[0], event[1])):
        in_flight += change
        peak = max(peak, in_flight)
    notes = []
    biggest = max(largest, default=None)
    if limit is None:
        notes.append("hp_tcp's task limit is unknown (server.env replaces TENT's configuration)")
    else:
        if biggest is not None and biggest > limit:
            notes.append(f"a write of {biggest} connector descriptors exceeds the task limit {limit} (TENT may "
                         "merge them: its tasks are not logged)")
        if peak > limit:
            notes.append(f"writes of {peak} connector descriptors overlapped, above the task limit {limit} (an "
                         "upper bound: their tasks complete before their calls return)")
    return {"task_limit": limit, "connector_writes": written, "failed_connector_writes": failed,
            "connector_descriptors": sum(largest), "max_connector_descriptors_per_write": biggest,
            "overlapping_call_descriptor_bound": peak, "bound_complete": complete, "notes": notes}


def _admissions(records: list[dict]) -> dict[str, list[dict]]:
    """Each request's admissions in a rank's step log, in order: a new or
    resumed request's tokens computed when admitted (``admitted``), its
    blocks then by group (``allocated_groups`` from place 0; the null block,
    0, holding a window's clipped places) and, when a pull was posted for it
    before, that pull's (first place, blocks) by group."""
    admissions: dict[str, list[dict]] = {}
    posted: dict[str, dict[int, tuple[int, list[int]]]] = {}
    for record in records:
        events = record.get("kv_events") or {}
        by_group = events.get("allocated_groups") or {}
        for request in events.get("reads_posted") or []:
            posted[request] = {entry["group"]: (entry["first"], list(entry["blocks"]))
                               for entry in by_group.get(request, [])}
        for request, computed in (events.get("admitted") or {}).items():
            admissions.setdefault(request, []).append({
                "computed": computed,
                "table": {entry["group"]: list(entry["blocks"]) for entry in by_group.get(request, [])
                          if entry["first"] == 0},
                "pull": posted.pop(request, None)})
    return admissions


def _hit_problems(layout: dict, hits: list[dict], admissions: dict[str, list[dict]]) -> tuple[list[str], list[dict]]:
    """A rank's prefix hit checks against its requests' admissions, and the
    checks that agree with them. Each admission implies a hit of each
    group's places before its pull's first (decode, a pull posted), else
    before its computed tokens' full blocks: its blocks there, null ones
    (a window's clipped head) left out. Each admission with such blocks has
    one hit check, in order, of exactly those places and blocks, its bytes
    the layout's for them; no request has any other. A pull's blocks must be
    the request's at their places when admitted, its first within the
    admitted computed tokens."""
    groups = layout.get("groups")
    if not groups:
        return ["the layout lists no KV cache groups to check prefix hits by"], []
    # Decode context parallelism: a scheduler block, a hit's unit, spans DCP
    # pages, one a rank.
    scale = int((layout.get("decode_context_parallel") or {}).get("size") or 1)
    problems: list[str] = []
    expected: dict[str, list[dict[int, list[tuple[int, int]]]]] = {}
    for request, adms in admissions.items():
        for admission in adms:
            pull, table = admission["pull"], admission["table"]
            for group, (first, blocks) in (pull or {}).items():
                size = (groups[group]["block_size"] * scale) if 0 <= group < len(groups) else None
                if table.get(group, [])[first:first + len(blocks)] != blocks:
                    problems.append(f"{request}'s pull of group {group} ({len(blocks)} block(s) from {first}) is "
                                    "not its blocks there when admitted")
                elif size is None or first * size > admission["computed"]:
                    problems.append(f"{request}'s pull of group {group} starts at block {first}, past the "
                                    f"{admission['computed']} tokens it was admitted with")
            want = {}
            for group, spec in enumerate(groups):
                count = (pull.get(group, (0, []))[0] if pull is not None
                         else admission["computed"] // (spec["block_size"] * scale))
                row = table.get(group, [])
                places = [(place, row[place]) for place in range(min(count, len(row))) if row[place]]
                if places:
                    want[group] = places
            if want:
                expected.setdefault(request, []).append(want)
    logged: dict[str, list[dict]] = {}
    for check in hits:
        if "groups" in check:  # a hit outside the lifecycle has none, reported apart
            logged.setdefault(check.get("req_id"), []).append(check)
    valid = []
    for request in sorted(set(expected) | set(logged), key=str):
        want, got = expected.get(request, []), logged.get(request, [])
        if len(want) != len(got):
            problems.append(f"{request} logged {len(got)} prefix hit check(s), not the {len(want)} its "
                            f"admissions imply ({[sum(map(len, w.values())) for w in want]} block(s))")
        for places, check in zip(want, got):
            found = []
            seen = {}
            for g in check["groups"]:
                if any(key not in g for key in ("group", "logical_blocks", "block_ids", "blocks", "bytes")):
                    found.append(f"{request}'s hit check of group {g.get('group')} lacks its places, blocks or "
                                 "bytes")
                    continue
                spec = groups[g["group"]] if 0 <= g["group"] < len(groups) else None
                seen[g["group"]] = list(zip(g["logical_blocks"], g["block_ids"]))
                per_block = None if spec is None else (
                    spec["transfer"]["bytes_per_block"] if "transfer" in spec
                    else spec.get("checked_bytes_per_block") or spec["layers"] * spec["block_bytes"])
                if (len(g["logical_blocks"]), len(g["block_ids"])) != (g["blocks"], g["blocks"]) or (
                        per_block is None or g["bytes"] != g["blocks"] * per_block):
                    found.append(f"{request}'s hit check of group {g['group']} ({g['blocks']} block(s), "
                                 f"{g['bytes']} bytes) disagrees with the layout ({per_block} bytes a block)")
            if seen != places:
                found.append(f"{request}'s hit check covers {_places(seen)}, not its admitted hit "
                             f"{_places(places)}")
            if (check.get("blocks"), check.get("bytes")) != (sum(g.get("blocks", 0) for g in check["groups"]),
                                                             sum(g.get("bytes", 0) for g in check["groups"])):
                found.append(f"{request}'s hit check is not its groups' sum")
            if not (check.get("blocks") and check.get("bytes")):
                found.append(f"{request}'s hit check covers no bytes")
            problems.extend(found)
            if not found and check.get("ok") and not check.get("mismatched_bytes"):
                valid.append(check)
    return _capped(problems), valid


def _places(places: dict[int, list[tuple[int, int]]]) -> str:
    """Hit places by group, briefly: each group's count and first and last
    (place: block)."""
    return "; ".join(f"group {group}: {len(p)} block(s)" + (f" {p[0][0]}:{p[0][1]}..{p[-1][0]}:{p[-1][1]}" if p else "")
                     for group, p in sorted(places.items())) or "nothing"


def _request_steps(records: list[dict]) -> dict[str, list[tuple[int, int, int, int]]]:
    """Each request's steps in a step log: (computed, scheduled, prompt,
    speculative) tokens, in order; speculative, the drafts the step
    verifies among its scheduled tokens."""
    steps: dict[str, list[tuple[int, int, int, int]]] = {}
    for record in records:
        for request in (record.get("step") or {}).get("requests", []):
            steps.setdefault(request["req_id"], []).append(
                (request["num_computed_tokens"], request["num_scheduled_tokens"], request["num_prompt_tokens"],
                 request.get("num_spec_tokens") or 0))
    return steps


def _handoff_from_sources(joint_steps: dict, decode_rank: int, sources: list[int], consumed: set[str]) -> dict:
    """Validate the handoff on every producer that supplies this decode rank.

    TP ranks need not match: scatter shares a producer, while gather needs
    each producer's steps. Keep the request count per decode rank.
    """
    results = [_handoff_problems(joint_steps[("prefill", source)], joint_steps[("decode", decode_rank)], consumed)
               for source in sources]
    evidence = results[0]["evidence"]
    return {"evidence": evidence, "problems": [
        (f"prefill rank {source}: " if sources != [decode_rank] else "") + problem
        for source, result in zip(sources, results)
        for problem in result["problems"]]}


def _handoff_problems(prefill: dict[str, list[tuple[int, int, int, int]]],
                      decode: dict[str, list[tuple[int, int, int, int]]],
                      consumed: set[str]) -> dict:
    """A hybrid model's handoff, from both sides' steps: prefill computes
    the prompt but its last token (Mooncake truncates its N tokens to
    N - 1), decode resumes at N - 1 and computes the last one. Every
    request decode consumed (`consumed`, by its own request ids: its byte
    checks, or its allocations) needs its steps. Requests whose steps show
    one prompt token, which no side truncates, are not checked."""
    produced = {_base_request_id(request): steps for request, steps in prefill.items()}
    problems, prompts = [], []
    for request in sorted(consumed):
        steps = decode.get(request)
        if steps is None:
            problems.append(f"{request} has no decode steps")
            continue
        computed, scheduled, prompt, drafts = steps[0]
        if prompt <= 1:
            continue
        prompts.append(prompt)
        # The last prompt token, and with speculative decoding the drafts
        # the same step verifies.
        if (computed, scheduled) != (prompt - 1, 1 + drafts):
            problems.append(f"{request} resumed at {computed} tokens with {scheduled} scheduled, not at "
                            f"{prompt - 1} with {1 + drafts}")
        steps = produced.get(_base_request_id(request))
        if steps is None:
            problems.append(f"{request} has no prefill steps")
            continue
        computed, scheduled, prefilled, _ = steps[-1]
        if prefilled != prompt - 1 or computed + scheduled != prompt - 1:
            problems.append(f"{request}'s prefill took a {prefilled}-token prompt to {computed + scheduled} "
                            f"tokens, not its {prompt} tokens but the last")
    return {"problems": _capped(problems), "evidence": {
        "requests": len(prompts), "prompt_tokens": [min(prompts), max(prompts)] if prompts else None,
        "prefilled": "N - 1", "decode_resumes_at": "N - 1"}}


def _capped(problems: list[str], keep: int = 3) -> list[str]:
    """At most `keep` problems of one kind, then their count."""
    return problems if len(problems) <= keep else problems[:keep] + [
        f"and {len(problems) - keep} more such problems"]


def _group_evidence(layout: dict, checks: list[dict], records: list[dict]) -> list[dict]:
    """Each KV cache group of a rank: its layout, what decode checked of it
    (blocks, bytes, pulls clipped to a window and the furthest first block)
    and its allocations by place (`allocated_groups`): places held by the
    null block (a window's head), blocks a request got at more than one
    place (reused), and blocks given at more than one place of any requests
    (reallocated)."""
    given: list[set[tuple[str, int, int]]] = [set() for _ in layout["groups"]]
    for record in records:
        for request, entries in ((record.get("kv_events") or {}).get("allocated_groups") or {}).items():
            for entry in entries:
                given[entry["group"]].update(
                    (request, entry["first"] + offset, block) for offset, block in enumerate(entry["blocks"]))
    evidence = []
    for index, spec in enumerate(layout["groups"]):
        nested = [g for check in checks for g in check.get("groups", []) if g["group"] == index]
        firsts = [g["first_logical_block"] for g in nested]
        places: dict[int, set[tuple[str, int]]] = {}
        for request, place, block in given[index]:
            if block:
                places.setdefault(block, set()).add((request, place))
        evidence.append({
            **spec,
            "checked_blocks": sum(g["blocks"] for g in nested),
            "checked_bytes": sum(g["bytes"] for g in nested),
            "clipped_pulls": sum(first > 0 for first in firsts),
            "first_logical_block_max": max(firsts, default=None),
            "null_places": len({(request, place) for request, place, block in given[index] if not block}),
            "reused_blocks": sum(len(held) > len({request for request, _ in held}) for held in places.values()),
            "reallocated_blocks": sum(len(held) > 1 for held in places.values()),
        })
    return evidence


def _precise_index_flags(run: Run, endpoint: tuple[str, int] | None) -> list[str]:
    """What a precise prefix index needs of a server: the render endpoints
    (the router tokenizes through them) and its KV-cache events, published
    on the control network under the instance's endpoint."""
    if run.router.prefix_index != "precise":
        return []
    if endpoint is None:
        raise ValueError("a precise prefix index needs the instance's endpoint (its events' topic)")
    control, endpoint_port = endpoint
    # Wildcard endpoints: vLLM binds those and connects out to any other
    # (the router dials the instance's control address).
    events = {"enable_kv_cache_events": True, "publisher": "zmq",
              "endpoint": f"tcp://*:{KV_EVENTS_PORT}", "replay_endpoint": f"tcp://*:{KV_EVENTS_REPLAY_PORT}",
              "topic": kv_events_topic(control, endpoint_port)}
    return ["--enable-scale-out", "--kv-events-config", json.dumps(events)]


def _prefix_routing(run: Run, sides: dict[str, Side], ranks: dict, bench: dict) -> dict:
    """Where requests found their prefix cached: per role, the requests
    with a byte-checked prefix hit (TP rank 0's: every TP rank checks the
    same requests) and their owners (an instance, or a DP rank). With the
    workload's prefix groups, the most that can hit: each group's first
    request finds nothing, wherever it goes; a group spread over I owners
    misses once on each."""
    found: dict = {}
    for role, side in sides.items():
        by_owner: dict[str, int] = {}
        hit: set[str] = set()
        for (dp, tp), summary in ranks[role].items():
            if tp:
                continue
            owner = side.node_of(dp) if side.instances or side.data_parallel_size == 1 else f"{side.node}/dp{dp}"
            by_owner[owner] = len(summary["hit_ids"])
            hit |= summary["hit_ids"]
        found[role] = {"hit_requests": len(hit), "by_owner": by_owner}
    repetition = run.workload.prefix_repetition
    completed = int(bench.get("completed") or 0)
    found["ceiling"] = max(0, completed - repetition["num_prefixes"]) if repetition else None
    found["expected_at_prefill"] = run.expect_prefill_hit_requests
    return found


def peak_prefill_throughput(steps: list[str], chunk: int) -> dict:
    """llm-d's ``peakPrefillThroughput`` from a saved run's prefill step
    logs, by the simulator's own timing (each step's simulated cost, not a
    measurement of hardware): the tokens a second of an unloaded prefill
    chunk of ``chunk`` tokens (the server's ``max_num_batched_tokens``) — a step of one
    request, all of it prefill, nothing cached, exactly that many tokens.
    The best such step's; ``steps`` counts them (0: the run had none, so
    its prompts were shorter than a chunk)."""
    rates = []
    for text in steps:
        for record in _records(text):
            step, cost = record.get("step"), record.get("cost")
            if not isinstance(step, dict) or not isinstance(cost, dict):
                continue
            requests = step.get("requests") or []
            if (len(requests) == 1 and requests[0].get("is_prefill") and not requests[0].get("num_cached_tokens")
                    and step.get("num_prefill_tokens") == chunk and not step.get("num_decode_tokens")):
                seconds = float(cost.get("device_s") or 0) + float(cost.get("host_s") or 0)
                if seconds > 0:
                    rates.append(chunk / seconds)
    return {"chunk_tokens": chunk, "steps": len(rates),
            "tokens_per_second": round(max(rates), 1) if rates else None}


def _prefix_caching_enabled(run: Run) -> bool | None:
    """The server's prefix caching: as set, else off for the byte checks
    (their request patterns), else vLLM's default."""
    enabled = run.server.enable_prefix_caching
    return False if enabled is None and run.kv_check else enabled


def _prefix_caching_flags(run: Run) -> list[str]:
    enabled = _prefix_caching_enabled(run)
    if enabled is None:
        return []
    return ["--enable-prefix-caching" if enabled else "--no-enable-prefix-caching"]


def _base_request_id(request_id: str) -> str:
    """The id the router gives a request on both instances: vLLM appends its
    own random suffix to each (``cmpl-<uuid>-0-<suffix>``)."""
    return request_id.rsplit("-", 1)[0]


def is_service(cmdline: bytes, comm: bytes) -> bool:
    """A process of the run: vLLM runs on the image's Python;
    vLLM renames its engine core and workers (VLLM::EngineCore,
    VLLM::Worker_TP0, ...)."""
    return cmdline.startswith((b"/opt/venv/", b"VLLM::")) or comm.startswith(b"VLLM::")


STOP = inspect.getsource(is_service) + """
import os, signal, sys, time

me = os.getpid()


def ours():
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
            comm = open(f"/proc/{pid}/comm", "rb").read()
        except OSError:
            continue
        if int(pid) != me and is_service(cmdline, comm):
            found.append(int(pid))
    return found


for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
    for pid in ours():
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + wait
    while ours() and time.monotonic() < deadline:
        time.sleep(0.5)
    if not ours():
        sys.exit(0)
sys.exit(f"still running: {ours()}")
"""


def stop_services(container: str) -> None:
    """Stop the vLLM servers and their workers in ``container``."""

    topology.exec_in(container, [PYTHON, "-c", STOP])


def router_ports(state: dict) -> dict[str, dict[str, dict]]:
    """Each port's bytes received and sent so far, at the routers of networks
    some node joins with several links (a link per rank), with the node,
    link and address on it."""

    on = {}
    for node, described in state["nodes"].items():
        for interface in described["interfaces"].values():
            if interface.get("router_port") is not None:
                on.setdefault(interface["network"], {})[interface["router_port"]] = {
                    "node": node, "slot": interface["slot"], "address": interface["address"]}
    multi = {network for network, ports in on.items()
             if any(port["slot"] > 0 for port in ports.values())}
    totals = {}
    for network in sorted(multi):
        pid = int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}",
                             state["networks"][network]["router"]]).strip())
        links = json.loads(shell.in_netns(pid, "ip", "-j", "-s", "link", "show"))
        totals[network] = {link["ifname"]: {**on[network][link["ifname"]],
                                            "rx": link["stats64"]["rx"]["bytes"],
                                            "tx": link["stats64"]["tx"]["bytes"]}
                           for link in links if link["ifname"] in on[network]}
    return totals


def network_ports(state: dict, network: str) -> dict[str, dict]:
    """Every port's bytes received and sent so far at ``network``'s router,
    with the node, link and address on it."""

    on = {interface["router_port"]: {"node": node, "slot": interface["slot"], "address": interface["address"]}
          for node, described in state["nodes"].items() for interface in described["interfaces"].values()
          if interface.get("network") == network and interface.get("router_port") is not None}
    pid = int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}", state["networks"][network]["router"]]).strip())
    links = json.loads(shell.in_netns(pid, "ip", "-j", "-s", "link", "show"))
    return {link["ifname"]: {**on[link["ifname"]], "rx": link["stats64"]["rx"]["bytes"],
                             "tx": link["stats64"]["tx"]["bytes"]}
            for link in links if link["ifname"] in on}


def control_exemption(before: dict, after: dict, nodes: list[str]) -> dict:
    """The front end's share of the control network's bytes over the
    workload: every byte to or from its nodes crosses their ports once,
    received there (rx) or sent there (tx), and the router's total counts
    each byte once, where it was received."""
    deltas = {port: {**described, "rx": described["rx"] - before[port]["rx"],
                     "tx": described["tx"] - before[port]["tx"]} for port, described in after.items()}
    return {"nodes": nodes, "bytes": sum(p["rx"] + p["tx"] for p in deltas.values() if p["node"] in nodes),
            "ports": deltas}


def control_evidence(saved: dict, run: Run) -> tuple[int, int, list[str]]:
    """The control network's bytes over the workload and the front end's
    share of them, derived from the saved ports (``control-ports.json``),
    never taken from its stored subtotal: ``(total, exempt, problems)``.
    The front end's node is the run's gateway. Anything the
    guard cannot stand on is a problem: a server, client or front-end node
    without a port, a counter that is not a non-negative integer, a stored
    node list or subtotal that differs from the derived one, and a front
    end's share larger than the network's total."""
    problems: list[str] = []
    nodes = [run.router.gateway]
    ports = saved.get("ports")
    if not isinstance(ports, dict):
        return 0, 0, ["the control network's port evidence is missing"]
    valid = True
    for port, described in ports.items():
        if not isinstance(described, dict) or not isinstance(described.get("node"), str):
            problems.append(f"the control port {port} names no node")
            valid = False
            continue
        for key in ("rx", "tx"):
            value = described.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                problems.append(f"the control port {port} ({described['node']}) has an invalid {key} counter: {value!r}")
                valid = False
    seen = {described["node"] for described in ports.values()
            if isinstance(described, dict) and isinstance(described.get("node"), str)}
    required = [*run.prefill.nodes, *run.decode.nodes, run.client, *nodes]
    missing = sorted(set(required) - seen)
    if missing:
        problems.append(f"the control network's port evidence lacks the nodes {missing}")
        valid = False
    if saved.get("nodes") != nodes:
        problems.append(f"the saved control exemption is for the nodes {saved.get('nodes')!r}, "
                        f"the front end's are {nodes}")
    if not valid:
        return 0, 0, problems
    total = sum(described["rx"] for described in ports.values())
    exempt = sum(described["rx"] + described["tx"] for described in ports.values() if described["node"] in nodes)
    if saved.get("bytes") != exempt or isinstance(saved.get("bytes"), bool):
        problems.append(f"the saved control exemption is {saved.get('bytes')!r} bytes, "
                        f"its ports give {exempt}")
    if exempt > total:
        problems.append(f"the front end's {exempt} control bytes exceed the control network's {total}")
    return total, exempt, problems


def _port_deltas(before: dict, after: dict) -> dict:
    """Each port's RX and TX over the workload (never summed: a byte crosses
    the router in on one port and out on another)."""
    return {network: {port: {**described, "rx": described["rx"] - before[network][port]["rx"],
                             "tx": described["tx"] - before[network][port]["tx"]}
                      for port, described in ports.items()}
            for network, ports in after.items()}


def _kv_addresses(interfaces: dict, side: Side, tp: int, node: str | None = None) -> list[str]:
    """Each TP rank's KV address: its network's address, or on a network the
    node joins with several links, link r's."""
    addresses = []
    for rank, network in enumerate(side.rank_networks(tp)):
        links = sorted((i for i in interfaces.values() if i.get("network") == network),
                       key=lambda i: i.get("slot", 0)) or [interfaces[network]]
        if len(links) == 1:
            addresses.append(links[0]["address"])
        elif rank < len(links):
            addresses.append(links[rank]["address"])
        else:
            raise ValueError(f"{node or side.node} has {len(links)} links on {network}, not one per rank ({tp})")
    return addresses


def router_bytes(state: dict) -> dict[str, int]:
    """Bytes each network's router has received on its links so far."""

    totals = {}
    for network, described in state["networks"].items():
        pid = int(shell.run(["docker", "inspect", "-f", "{{.State.Pid}}", described["router"]]).strip())
        links = json.loads(shell.in_netns(pid, "ip", "-j", "-s", "link", "show"))
        totals[network] = sum(link["stats64"]["rx"]["bytes"] for link in links if link["ifname"].startswith("p"))
    return totals


def _records(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _background(container: str, command: list[str], log: str) -> None:
    script = " ".join(_quote(part) for part in command)
    shell.run(["docker", "exec", "-d", container, "sh", "-c", f"exec {script} > {log} 2>&1"])


def _quote(part: str) -> str:
    return "'" + part.replace("'", "'\"'\"'") + "'"


# The line a vLLM server logs when its engine dies at startup.
ENGINE_FAILED = "Engine core initialization failed"


def _wait(client: str, url: str, timeout: float, owner: str, log: str) -> None:
    """Until ``url`` answers from the client node; a timeout, or a vLLM
    server whose engine failed to start (its log says so), reports the tail
    of ``log`` in ``owner``."""

    probe = ("import sys, urllib.request\n"
             "try:\n    urllib.request.urlopen(sys.argv[1], timeout=5); print('up')\n"
             "except Exception as error:\n    print(error)\n")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if topology.exec_in(client, [PYTHON, "-c", probe, url]).strip() == "up":
            return
        if topology.exec_in(owner, ["grep", "-c", "-m", "1", ENGINE_FAILED, log], check=False).strip() == "1":
            tail = topology.exec_in(owner, ["tail", "-n", "30", log], check=False)
            raise RuntimeError(f"the server behind {url} failed to start; {owner}:{log} ends with:\n{tail}")
        time.sleep(3)
    tail = topology.exec_in(owner, ["tail", "-n", "30", log], check=False)
    raise TimeoutError(f"{url} did not answer within {timeout:.0f} s; {owner}:{log} ends with:\n{tail}")


def _rank_suffix(dp: int, tp: int) -> str:
    """The simulator's step log suffix for a rank: `.dp<i>` and `.tp<r>`."""
    return (f".dp{dp}" if dp else "") + (f".tp{tp}" if tp else "")


def _tent_metrics(run: Run, container: dict[str, str]) -> dict[str, list[str | None]]:
    """Every rank's TENT metrics, as served (None: no answer)."""

    fetch = ("import sys, urllib.request\n"
             "try:\n    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())\n"
             "except Exception:\n    print('NO ANSWER')\n")
    if not uses_tent(run.server):
        return {}
    metrics: dict[str, list[str | None]] = {}
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        metrics[role] = []
        # Every rank, in `Side.ranks` order, at its host's port for it.
        size = side.tp(run.server)
        for dp, tp in side.ranks(size):
            text = topology.exec_in(container[side.node_of(dp)], [
                PYTHON, "-c", fetch,
                f"http://127.0.0.1:{TENT_METRICS_PORT + side.local_rank(dp, tp, size)}/metrics"], check=False)
            metrics[role].append(None if text.strip() == "NO ANSWER" else text)
    return metrics


def _tent_delta(before: str | None, after: str | None) -> dict | None:
    """TENT's counters' change over the workload, summed over their labels
    (``tent_registered_buffer_bytes``: as after)."""

    if before is None or after is None:
        return None

    def totals(text: str) -> dict[str, float]:
        result: dict[str, float] = {}
        for metric in text.splitlines():
            if metric.startswith("tent_") and " " in metric:
                name, _, value = metric.rpartition(" ")
                name = name.split("{", 1)[0]
                if name.endswith("_total") or name == "tent_registered_buffer_bytes":
                    result[name] = result.get(name, 0.0) + float(value)
        return result

    start, end = totals(before), totals(after)
    return {name: value if name == "tent_registered_buffer_bytes" else value - start.get(name, 0.0)
            for name, value in end.items()}


def _collect(run: Run, container: dict[str, str], output: Path) -> None:
    """Logs, every rank's step log and the client's results, whatever
    happened: an instance's logs under ``logs/<role>/<node>``."""

    copies: list[tuple[str, str, Path]] = [
        (container[run.client], f"{WORK}/logs/.", output / "logs" / "client"),
        (container[run.client], f"{WORK}/results/bench.json", output / "bench.json")]
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        for node in side.nodes:
            logs = output / "logs" / role / node
            copies.append((container[node], f"{WORK}/logs/.", logs))
        for dp, tp in side.ranks(side.tp(run.server)):
            copies.append((container[side.node_of(dp)], side.remote_step_log(dp, tp),
                           output / side.step_log(role, dp, tp)))
    for name, remote, local in copies:
        local.parent.mkdir(parents=True, exist_ok=True)
        shell.run(["docker", "cp", f"{name}:{remote}", str(local)], check=False)
