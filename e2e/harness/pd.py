"""A prefill/decode run of the vLLM simulator in an up topology.

Prefill and decode each run ``vllm serve`` with simulated compute and real
KV transfers through MooncakeConnector over TENT's hp_tcp, bound to the
node's address on its KV network (other TENT transports off). HTTP, the
proxy and Mooncake's bootstrap use the control network. The client node runs
MooncakeConnector's proxy and ``vllm bench serve``.

With tensor parallelism, every rank moves its own KV heads: on the side's KV
network, or on its own one (``kv_network`` a list, one per rank, the same
on both sides), and logs into its own step log (``steps.tp<r>.jsonl``);
each serves TENT's metrics on its own port.

The run is then checked from what the servers and routers recorded: every
request completed, decode compared every byte it received with prefill's
pattern (``kv_check``) on every rank, the connector failed no transfer, each
rank's TENT saw the bytes its connector moved, and the KV crossed its
rank's network's router, not another's or the control one's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
import json
from pathlib import Path
import time

import yaml

from . import shell, topology


E2E = Path(__file__).resolve().parent.parent
REPOSITORY = E2E.parent
DEFAULT_PROXY = (REPOSITORY / "vllm" / "examples" / "disaggregated" / "mooncake_connector"
                 / "mooncake_connector_proxy.py")
RESULTS = E2E / "results"
VLLM = "/opt/venv/bin/vllm"
PYTHON = "/opt/venv/bin/python"
MODEL_DIRECTORY = "/model"
MODEL_NAME = "sim-pd"
SERVER_PORT = 8000
PROXY_PORT = 8080
BOOTSTRAP_PORT = 8998
WORK = "/e2e"
STEP_LOG = f"{WORK}/steps.jsonl"
TENT_METRICS_PORT = 9100  # rank r of the host serves on TENT_METRICS_PORT + r
DISABLED_TENT_TRANSPORTS = ("tcp", "rdma", "shm", "gds", "ascend_direct")


@dataclass
class Side:
    node: str
    # A list: one per tensor-parallel rank, which every DP rank's TP rank of
    # that index uses (any prefill DP rank may pair with any decode DP rank).
    kv_network: str | list[str]
    # > 1: an MoE model's data-parallel attention with expert parallelism.
    data_parallel_size: int = 1

    def rank_networks(self, ranks: int) -> list[str]:
        if isinstance(self.kv_network, str):
            return [self.kv_network] * ranks
        return list(self.kv_network)

    def ranks(self, tp_size: int) -> list[tuple[int, int]]:
        """(DP rank, TP rank) of each of the instance's ranks, in the host's
        order (rank_addresses, TENT metrics ports, step logs)."""
        return [(dp, tp) for dp in range(self.data_parallel_size) for tp in range(tp_size)]


@dataclass
class Server:
    gpu: str = "h100"
    block_size: int = 16
    kv_cache_memory_bytes: int = 2 * 1024 ** 3
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    hp_tcp_args: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    # The simulator's timing provider; roofline's arguments default to {gpu}.
    timing: str = "roofline"
    timing_args: dict | None = None


@dataclass
class Workload:
    input_len: int = 1024
    output_len: int = 16
    num_prompts: int = 40
    concurrency: int = 4
    seed: int = 0


@dataclass
class Run:
    name: str
    topology: str  # its name: the topology must be up
    model: Path  # a Hugging Face model directory (config and tokenizer)
    prefill: Side
    decode: Side
    client: str
    control_network: str
    kv_check: bool = True
    server: Server = field(default_factory=Server)
    workload: Workload = field(default_factory=Workload)
    proxy: Path = DEFAULT_PROXY
    startup_timeout: float = 600.0


def load(path: str | Path) -> Run:
    """A run config; relative paths are the file's."""

    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    known = {"name", "topology", "model", "prefill", "decode", "client", "control_network", "kv_check",
             "server", "workload", "proxy", "startup_timeout"}
    if unknown := set(raw) - known:
        raise ValueError(f"unknown run key(s): {', '.join(sorted(unknown))}")
    try:
        run = Run(
            name=raw["name"], topology=raw["topology"],
            model=(path.parent / raw["model"]).resolve(),
            prefill=Side(**raw["prefill"]), decode=Side(**raw["decode"]),
            client=raw["client"], control_network=raw["control_network"],
            kv_check=bool(raw.get("kv_check", True)),
            server=Server(**(raw.get("server") or {})), workload=Workload(**(raw.get("workload") or {})),
            proxy=(path.parent / raw["proxy"]).resolve() if "proxy" in raw else DEFAULT_PROXY,
            startup_timeout=float(raw.get("startup_timeout", 600)),
        )
    except KeyError as missing:
        raise ValueError(f"run config needs {missing}") from None
    except TypeError as error:
        raise ValueError(f"run config: {error}") from None
    ranks = run.server.tensor_parallel_size
    for side in (run.prefill, run.decode):
        if not isinstance(side.kv_network, str) and len(side.kv_network) != ranks:
            raise ValueError(f"{side.node}: kv_network needs one network per rank ({ranks})")
    if run.prefill.rank_networks(ranks) != run.decode.rank_networks(ranks):
        # Each network routes only its own subnet.
        raise ValueError("prefill and decode rank r need the same KV network")
    return run


def tent_config(address: str, hp_tcp_args: dict) -> dict:
    """TENT with hp_tcp only, its listener, rails and control service at
    ``address``."""

    transports: dict = {name: {"enable": False} for name in DISABLED_TENT_TRANSPORTS}
    transports["hp_tcp"] = {"enable": True, "bind_address": address, "advertise_address": address,
                            "rail_addresses": [address], **hp_tcp_args}
    return {"rpc_server_hostname": address, "transports": transports}


def server_command(run: Run, role: str, kv_addresses: list[str]) -> list[str]:
    """``vllm serve`` for ``role`` (prefill or decode), TP rank r's KV on
    ``kv_addresses[r]`` in every DP rank (their own: ``rank_addresses``, DP
    rank by TP rank)."""

    kv_address = kv_addresses[0]
    side = getattr(run, role)

    timing_args = run.server.timing_args
    if timing_args is None:
        timing_args = {"gpu": run.server.gpu} if run.server.timing == "roofline" else {}
    simulation = {"clock": "real", "timing": run.server.timing, "timing_args": timing_args,
                  "log_path": STEP_LOG, "kv_transfer_mode": "real"}
    if run.kv_check:
        simulation["kv_pattern_fill" if role == "prefill" else "kv_pattern_check"] = True
    kv_transfer = {"kv_connector": "MooncakeConnector",
                   "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                   "kv_buffer_device": "cpu", "kv_connector_extra_config": {"mooncake_protocol": "tcp"}}
    if len(set(kv_addresses)) > 1:
        kv_transfer["kv_connector_extra_config"]["rank_addresses"] = kv_addresses * side.data_parallel_size
    env = {
        "TRITON_CPU_BACKEND": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "OMP_NUM_THREADS": "1",
        # The address Mooncake's workers register (vLLM's get_ip).
        "VLLM_HOST_IP": kv_address,
        "MC_USE_TENT": "1",
        "MC_TENT_CONF": json.dumps(tent_config(kv_address, run.server.hp_tcp_args), separators=(",", ":")),
        "VLLM_MOONCAKE_BOOTSTRAP_PORT": str(BOOTSTRAP_PORT),
        # TENT's metrics, read from inside the node.
        "TENT_METRICS_HTTP_HOST": "127.0.0.1",
        "TENT_METRICS_HTTP_PORT": str(TENT_METRICS_PORT),
        "TENT_METRICS_REPORT_INTERVAL": "0",
        **{name: str(value) for name, value in run.server.env.items()},
    }
    s = run.server
    return [
        "env", *(f"{name}={value}" for name, value in env.items()),
        VLLM, "serve", MODEL_DIRECTORY, "--host", "0.0.0.0", "--port", str(SERVER_PORT),
        "--served-model-name", MODEL_NAME, "--dtype", s.dtype, "--block-size", str(s.block_size),
        "--kv-cache-memory-bytes", str(s.kv_cache_memory_bytes), "--max-model-len", str(s.max_model_len),
        "--max-num-batched-tokens", str(s.max_num_batched_tokens), "--max-num-seqs", str(s.max_num_seqs),
        "--tensor-parallel-size", str(s.tensor_parallel_size),
        *(["--data-parallel-size", str(side.data_parallel_size), "--enable-expert-parallel"]
          if side.data_parallel_size > 1 else []),
        "--generation-config", "vllm", "--enforce-eager",
        # The byte check's pattern is the request's: a block cached for
        # another request would not match it.
        *(["--no-enable-prefix-caching"] if run.kv_check else []),
        "--simulation-config", json.dumps(simulation),
        "--kv-transfer-config", json.dumps(kv_transfer),
        "--simulate-forward",
    ]


def execute(run: Run, output: Path | None = None) -> dict:
    """Start the services, run the workload, collect and check; returns the
    verdict (also in ``<output>/verdict.json``). Raises AssertionError when
    a check fails, after collecting."""

    state = topology.load_state(run.topology)
    nodes = state["nodes"]
    ranks = run.server.tensor_parallel_size
    container = {name: nodes[name]["container"] for name in (run.prefill.node, run.decode.node, run.client)}
    address = {
        **{f"{role}_kv": [nodes[side.node]["interfaces"][network]["address"]
                          for network in side.rank_networks(ranks)]
           for role, side in (("prefill", run.prefill), ("decode", run.decode))},
        "prefill_control": nodes[run.prefill.node]["interfaces"][run.control_network]["address"],
        "decode_control": nodes[run.decode.node]["interfaces"][run.control_network]["address"],
    }
    output = output or RESULTS / f"{run.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps({**asdict(run), "model": str(run.model), "proxy": str(run.proxy),
                                                 "addresses": address}, indent=2, default=str) + "\n")

    for name in container.values():
        # A previous run's servers would answer this run's requests.
        stop_services(name)
        topology.exec_in(name, ["sh", "-c", f"rm -rf {WORK}/logs {WORK}/results {WORK}/steps*.jsonl "
                                            f"&& mkdir -p {WORK}/logs"])
        topology.exec_in(name, ["mkdir", "-p", MODEL_DIRECTORY])
        shell.run(["docker", "cp", f"{run.model}/.", f"{name}:{MODEL_DIRECTORY}"])
    shell.run(["docker", "cp", str(run.proxy), f"{container[run.client]}:{WORK}/proxy.py"])

    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        _background(container[side.node], server_command(run, role, address[f"{role}_kv"]),
                    f"{WORK}/logs/{role}.log")
    try:
        for role in ("prefill", "decode"):
            _wait(container[run.client], f"http://{address[f'{role}_control']}:{SERVER_PORT}/health",
                  run.startup_timeout, container[getattr(run, role).node], f"{WORK}/logs/{role}.log")
        _background(container[run.client], [
            PYTHON, f"{WORK}/proxy.py", "--host", "127.0.0.1", "--port", str(PROXY_PORT),
            "--prefill", f"http://{address['prefill_control']}:{SERVER_PORT}", str(BOOTSTRAP_PORT),
            "--decode", f"http://{address['decode_control']}:{SERVER_PORT}"], f"{WORK}/logs/proxy.log")
        # The proxy has no health endpoint: FastAPI's docs page.
        _wait(container[run.client], f"http://127.0.0.1:{PROXY_PORT}/docs", 120,
              container[run.client], f"{WORK}/logs/proxy.log")

        tent_before = _tent_metrics(run, container)
        routers_before = router_bytes(state)
        w = run.workload
        bench = [
            VLLM, "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
            "--base-url", f"http://127.0.0.1:{PROXY_PORT}", "--model", MODEL_NAME, "--tokenizer", MODEL_DIRECTORY,
            "--dataset-name", "random", "--random-input-len", str(w.input_len),
            "--random-output-len", str(w.output_len), "--num-prompts", str(w.num_prompts),
            "--max-concurrency", str(w.concurrency), "--seed", str(w.seed), "--ignore-eos", "--disable-tqdm",
            "--save-result", "--save-detailed", "--result-dir", f"{WORK}/results", "--result-filename", "bench.json",
        ]
        (output / "bench.log").write_text(topology.exec_in(container[run.client], bench, timeout=3600))
        # The servers' step logs settle shortly after the last response.
        time.sleep(2)
        routers_after = router_bytes(state)
        tent_after = _tent_metrics(run, container)
    finally:
        _collect(run, container, output)
        for name in container.values():
            stop_services(name)
    routers = {network: routers_after[network] - routers_before[network] for network in routers_before}
    (output / "routers.json").write_text(json.dumps(routers, indent=2) + "\n")
    tent = {role: [_tent_delta(before, after) for before, after in zip(tent_before[role], tent_after[role])]
            for role in tent_before}
    (output / "tent.json").write_text(json.dumps(tent, indent=2) + "\n")
    verdict = check(
        json.loads((output / "bench.json").read_text()),
        *[[(output / _step_log_name(role, dp, tp)).read_text() for dp, tp in getattr(run, role).ranks(ranks)]
          for role in ("prefill", "decode")],
        routers, run, tent)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["problems"]:
        raise AssertionError(f"run {run.name} failed its checks ({output}): {'; '.join(verdict['problems'])}")
    return verdict


def check(bench: dict, prefill_steps: list[str], decode_steps: list[str], routers: dict[str, int],
          run: Run, tent: dict[str, list[dict]] | None = None) -> dict:
    """The run's checks, from the client's results, every rank's step logs
    (``prefill_steps[i]``, ``decode_steps[i]``, in each side's `Side.ranks`
    order), the bytes each router received during the workload and each
    rank's TENT counters over it (``tent[role][i]``). Requests are counted
    once, from the client; bytes per rank, then summed. Each request is
    produced by one prefill DP rank and consumed by one decode DP rank,
    the same on every TP rank."""

    problems = []
    w = run.workload
    tp_size = run.server.tensor_parallel_size
    errors = [error for error in bench.get("errors", []) if error]
    if bench.get("completed") != w.num_prompts or errors:
        problems.append(f"{bench.get('completed')} of {w.num_prompts} request(s) completed; errors: {errors[:3]}")

    sides = {"prefill": run.prefill, "decode": run.decode}
    data_parallel = any(side.data_parallel_size > 1 for side in sides.values())

    def where(dp: int | None, tp: int) -> str:
        parts = ([f"DP rank {dp}"] if data_parallel and dp is not None else []) + (
            [f"rank {tp}"] if tp_size > 1 else [])
        return " ".join(parts) + ": " if parts else ""

    logs = {role: dict(zip(side.ranks(tp_size), map(_records, steps)))
            for role, side, steps in (("prefill", run.prefill, prefill_steps),
                                      ("decode", run.decode, decode_steps))}
    tents = {role: dict(zip(sides[role].ranks(tp_size), tent[role])) for role in tent} if tent else {}
    ranks: dict[str, dict[tuple[int, int], dict]] = {role: {} for role in sides}
    for role, side in sides.items():
        for (dp, tp), records in logs[role].items():
            at = where(dp, tp)
            layout = next((r["kv_layout"] for r in records if "kv_layout" in r), None)
            # Each log must be this rank's (not another's supplied twice).
            if layout is None:
                problems.append(f"{at}{role} logged no KV layout")
            elif (layout.get("tp_rank"), layout.get("tp_size")) != (tp, tp_size):
                problems.append(f"{at}{role}'s log is TP rank {layout.get('tp_rank')} of "
                                f"{layout.get('tp_size')}, not {tp} of {tp_size}")
            elif data_parallel and layout.get("dp_rank") != dp:
                problems.append(f"{at}{role}'s log is DP rank {layout.get('dp_rank')}, not {dp}")
            sent = {"count": 0, "bytes": 0, "failed": 0}
            for record in records:
                for transfers in (record.get("kv_transfers") or {}).values():
                    for key in sent:
                        sent[key] += int(transfers.get(key) or 0)
            if sent["failed"]:
                problems.append(f"{at}{role}'s connector failed {sent['failed']} transfer(s)")
            checks = [record["kv_check"] for record in records if "kv_check" in record]
            if run.kv_check and role == "decode":
                bad = [c for c in checks if not c.get("ok") or c.get("mismatched_bytes")]
                if bad:
                    problems.append(f"{at}{len(bad)} request(s) received KV bytes unlike prefill's: {bad[0]}")
                if layout and any(c.get("bytes") != c["blocks"] * c["layers"] * layout["block_bytes"]
                                  for c in checks if "blocks" in c):
                    problems.append(f"{at}byte checks disagree with the rank's KV layout ({layout['block_bytes']} "
                                    "bytes a block and layer)")
            # The requests the rank owns: prefill's allocations; decode's
            # byte checks, else its allocations.
            owned = {_base_request_id(c["req_id"]) for c in checks if "req_id" in c} if (
                role == "decode" and run.kv_check) else {
                _base_request_id(req_id) for r in records
                for req_id in ((r.get("kv_events") or {}).get("allocated") or {})}
            counters = tents.get(role, {}).get((dp, tp)) if tents else None
            if tents:
                required = {"prefill": ("tent_registered_buffer_bytes", "tent_write_bytes_total"),
                            "decode": ("tent_registered_buffer_bytes",)}[role]
                if counters is None:
                    problems.append(f"{at}{role}'s TENT metrics did not answer")
                elif missing := [name for name in required if name not in counters]:
                    problems.append(f"{at}{role}'s TENT metrics lack {', '.join(missing)}")
                registered = (counters or {}).get("tent_registered_buffer_bytes")
                if registered is not None and layout and registered < layout["bytes"]:
                    problems.append(f"{at}{role}'s TENT registered {registered:.0f} bytes, less than its "
                                    f"{layout['bytes']} of KV")
                written = (counters or {}).get("tent_write_bytes_total")
                if role == "prefill" and written is not None and written != sent["bytes"]:
                    problems.append(f"{at}TENT wrote {written:.0f} bytes, not the {sent['bytes']} its connector "
                                    "moved")
            ranks[role][(dp, tp)] = {
                "transfers": sent, "kv_checks": len(checks),
                "kv_bytes_checked": sum(int(c.get("bytes") or 0) for c in checks),
                "requests": owned, "tent": counters,
                "kv_layout": layout and {key: layout.get(key) for key in (
                    "dp_rank", "dp_local_rank", "tp_rank", "tp_size", "shape", "block_bytes", "bytes",
                    "tent_metrics_port")},
            }

    per_tp = []
    ownership: dict[str, dict[int, int]] = {"producers": {}, "consumers": {}}
    owners_by_tp: list[dict[str, dict[str, set[int]]]] = []
    for tp in range(tp_size):
        at = where(None, tp)
        of = {role: {dp: summary for (dp, rank), summary in ranks[role].items() if rank == tp} for role in sides}
        checked = sum(summary["kv_bytes_checked"] for summary in of["decode"].values())
        moved = sum(summary["transfers"]["bytes"] for role in sides for summary in of[role].values())
        if moved < checked:
            problems.append(f"{at}the connector moved {moved} bytes, fewer than the {checked} decode checked")
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
        if run.kv_check:
            if consumed != w.num_prompts:
                problems.append(f"{at}decode byte-checked {consumed} distinct request(s), not {w.num_prompts}")
            if byte_checks != consumed:
                problems.append(f"{at}decode logged {byte_checks} byte check(s) for {consumed} distinct "
                                "request(s)")
        for role, label in (("prefill", "prefill DP ranks produced"), ("decode", "decode DP ranks consumed")):
            shared = sorted(request for request, dps in owners[role].items() if len(dps) > 1)
            if shared:
                problems.append(f"{at}{len(shared)} request(s) more than one of the {label}: {shared[0]}")
        unpaired = sorted(set(owners["prefill"]) ^ set(owners["decode"]))
        if unpaired:
            problems.append(f"{at}{len(unpaired)} request(s) not both produced and consumed: {unpaired[0]}")
        owners_by_tp.append(owners)
        per_tp.append({
            "kv_checks": byte_checks, "consumed": consumed, "kv_bytes_checked": checked,
            "transfers_bytes": moved,
            "kv_layout": {role: (of[role].get(0) or {}).get("kv_layout") for role in sides},
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
    for (dp, tp), summary in ranks["prefill"].items():
        network = run.prefill.rank_networks(tp_size)[tp]
        expected[network] = expected.get(network, 0) + summary["transfers"]["bytes"]
    for network, carried_by_ranks in expected.items():
        carried = routers.get(network, 0)
        if carried < carried_by_ranks:
            problems.append(f"the KV network {network} carried {carried} bytes, fewer than the {carried_by_ranks} "
                            "its ranks moved")
        elif carried > carried_by_ranks * 1.05 + (1 << 20):
            problems.append(f"the KV network {network} carried {carried} bytes, more than its ranks' "
                            f"{carried_by_ranks}: another rank's KV")
    moved = sum(summary["transfers"]["bytes"] for role in sides for summary in ranks[role].values())
    control = routers.get(run.control_network, 0)
    if moved and control > moved / 10:
        problems.append(f"the control network carried {control} bytes, too much beside {moved} of KV")
    sums = {key: sum(summary["transfers"][key] for role in sides for summary in ranks[role].values())
            for key in ("count", "bytes", "failed")}
    return {
        "problems": problems,
        "requests": bench.get("completed"),
        "kv_checks": per_tp[0]["kv_checks"] if per_tp else 0,
        "kv_bytes_checked": sum(rank["kv_bytes_checked"] for rank in per_tp),
        "transfers": sums,
        "router_bytes": routers,
        "ranks": per_tp,
        "ownership": ownership,
        "dp_ranks": {role: [{"dp_rank": dp, "tp_rank": tp,
                             **{key: value for key, value in summary.items() if key != "requests"},
                             "requests": len(summary["requests"])}
                            for (dp, tp), summary in ranks[role].items()] for role in sides},
        "ttft_ms": {key: bench.get(f"{key}_ttft_ms") for key in ("mean", "median", "p99")},
        "tpot_ms": {key: bench.get(f"{key}_tpot_ms") for key in ("mean", "median", "p99")},
    }


def _base_request_id(request_id: str) -> str:
    """The id the proxy gives a request on both instances: vLLM appends its
    own random suffix to each (``cmpl-<uuid>-0-<suffix>``)."""
    return request_id.rsplit("-", 1)[0]


def is_service(cmdline: bytes, comm: bytes) -> bool:
    """A process of the run: vLLM and the proxy run on the image's Python;
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
    """Stop the vLLM servers, their workers and the proxy in ``container``."""

    topology.exec_in(container, [PYTHON, "-c", STOP])


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


def _wait(client: str, url: str, timeout: float, owner: str, log: str) -> None:
    """Until ``url`` answers from the client node; a timeout reports the
    tail of ``log`` in ``owner``."""

    probe = ("import sys, urllib.request\n"
             "try:\n    urllib.request.urlopen(sys.argv[1], timeout=5); print('up')\n"
             "except Exception as error:\n    print(error)\n")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if topology.exec_in(client, [PYTHON, "-c", probe, url]).strip() == "up":
            return
        time.sleep(3)
    tail = topology.exec_in(owner, ["tail", "-n", "30", log], check=False)
    raise TimeoutError(f"{url} did not answer within {timeout:.0f} s; {owner}:{log} ends with:\n{tail}")


def _rank_suffix(dp: int, tp: int) -> str:
    """The simulator's step log suffix for a rank: `.dp<i>` and `.tp<r>`."""
    return (f".dp{dp}" if dp else "") + (f".tp{tp}" if tp else "")


def _step_log_name(role: str, dp: int, tp: int) -> str:
    return f"{role}-steps{_rank_suffix(dp, tp)}.jsonl"


def _tent_metrics(run: Run, container: dict[str, str]) -> dict[str, list[str | None]]:
    """Every rank's TENT metrics, as served (None: no answer)."""

    fetch = ("import sys, urllib.request\n"
             "try:\n    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())\n"
             "except Exception:\n    print('NO ANSWER')\n")
    metrics: dict[str, list[str | None]] = {}
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        metrics[role] = []
        # The host's ranks, in order: local DP rank by TP rank.
        for rank, _ in enumerate(side.ranks(run.server.tensor_parallel_size)):
            text = topology.exec_in(container[side.node], [
                PYTHON, "-c", fetch, f"http://127.0.0.1:{TENT_METRICS_PORT + rank}/metrics"], check=False)
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
    """Logs, every rank's step log and the client's results, whatever happened."""

    for role, node in (("prefill", run.prefill.node), ("decode", run.decode.node), ("client", run.client)):
        name = container[node]
        copies = [(f"{WORK}/logs/.", output / "logs" / role)]
        if role == "client":
            copies.append((f"{WORK}/results/bench.json", output / "bench.json"))
        else:
            side = run.prefill if role == "prefill" else run.decode
            copies += [(f"{WORK}/steps{_rank_suffix(dp, tp)}.jsonl", output / _step_log_name(role, dp, tp))
                       for dp, tp in side.ranks(run.server.tensor_parallel_size)]
        for remote, local in copies:
            local.parent.mkdir(parents=True, exist_ok=True)
            shell.run(["docker", "cp", f"{name}:{remote}", str(local)], check=False)
