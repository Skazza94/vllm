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
pattern (``kv_check``) on every rank, the connector failed no transfer and
moved the bytes decode checked, each rank's TENT saw the bytes its connector
moved, and the KV crossed its rank's network's router, not another's or the
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
    block_size: int | None = 16
    """Tokens per KV block; None: the platform's (a hybrid model's)."""
    enable_prefix_caching: bool | None = None
    """None: off with kv_check, else vLLM's default (on; a hybrid model's
    real transfers need it off)."""
    kv_cache_memory_bytes: int = 2 * 1024 ** 3
    max_model_len: int = 4096
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    hp_tcp_args: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    timing: str | None = None
    """The simulator's timing provider (zero, linear, table, replay,
    aisimulate or a module.Class), required for a new run; None only in a
    saved run from before it was (timed by the former roofline provider)."""
    timing_args: dict | None = None
    """Its arguments; aisimulate's default to {gpu} (plus a target_model)."""


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
    if run.server.timing is None:
        raise ValueError("run config needs server.timing (zero, linear, table, replay, aisimulate or a "
                         "module.Class)")
    if run.kv_check and run.server.enable_prefix_caching:
        raise ValueError("kv_check needs prefix caching off (server.enable_prefix_caching)")
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
        timing_args = {"gpu": run.server.gpu} if run.server.timing == "aisimulate" else {}
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
        "--served-model-name", MODEL_NAME, "--dtype", s.dtype,
        *(["--block-size", str(s.block_size)] if s.block_size is not None else []),
        "--kv-cache-memory-bytes", str(s.kv_cache_memory_bytes), "--max-model-len", str(s.max_model_len),
        "--max-num-batched-tokens", str(s.max_num_batched_tokens), "--max-num-seqs", str(s.max_num_seqs),
        "--tensor-parallel-size", str(s.tensor_parallel_size),
        *(["--data-parallel-size", str(side.data_parallel_size), "--enable-expert-parallel"]
          if side.data_parallel_size > 1 else []),
        "--generation-config", "vllm", "--enforce-eager",
        # The byte check's pattern is the request's: a block cached for
        # another request would not match it.
        *_prefix_caching_flags(run),
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


def recheck(output: str | Path) -> dict:
    """A saved run's checks again, from its results directory, with this
    harness: written to ``verdict-recheck.json``, the run's own
    ``verdict.json`` left as it was."""
    output = Path(output)
    raw = json.loads((output / "run.json").read_text())
    raw.pop("addresses", None)
    run = Run(**{**raw, "model": Path(raw["model"]), "proxy": Path(raw["proxy"]),
                 "prefill": Side(**raw["prefill"]), "decode": Side(**raw["decode"]),
                 "server": Server(**raw["server"]), "workload": Workload(**raw["workload"])})
    ranks = run.server.tensor_parallel_size
    verdict = check(
        json.loads((output / "bench.json").read_text()),
        *[[(output / _step_log_name(role, dp, tp)).read_text() for dp, tp in getattr(run, role).ranks(ranks)]
          for role in ("prefill", "decode")],
        json.loads((output / "routers.json").read_text()), run,
        json.loads((output / "tent.json").read_text()))
    (output / "verdict-recheck.json").write_text(json.dumps(verdict, indent=2) + "\n")
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
    # Each TP rank's steps, every DP rank's requests: under MoE DP only DP
    # rank 0 logs the joint steps, the others their own KV.
    joint_steps: dict[tuple[str, int], dict[str, list[tuple[int, int, int]]]] = {
        (role, tp): {} for role in sides for tp in range(tp_size)}
    for role in sides:
        for (dp, tp), records in logs[role].items():
            joint_steps[(role, tp)].update(_request_steps(records))
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
                if layout:
                    # Failed receives are reported above; every other check is validated.
                    problems.extend(f"{at}{problem}" for problem in _byte_check_problems(
                        layout, [c for c in checks if "error" not in c], w.input_len, _pull_plans(records),
                        {req_id: steps[0][2] for req_id, steps in joint_steps[("decode", tp)].items()}))
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
                "checked_ids": {c["req_id"] for c in checks if "req_id" in c and "error" not in c},
                "allocated_ids": {req_id for r in records
                                  for req_id in ((r.get("kv_events") or {}).get("allocated") or {})},
                "hybrid": bool(layout and any(g.get("kind") == "MambaSpec" for g in layout.get("groups") or [])),
                "groups": layout and layout.get("groups") and _group_evidence(layout, checks, records),
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
        # Decode checks every block the connector pulls: the transfer plan.
        if moved < checked or (run.kv_check and moved != checked):
            problems.append(f"{at}the connector moved {moved} bytes, not the {checked} decode checked")
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
        handoff = None
        if any(summary["hybrid"] for summary in of["decode"].values()):
            # The requests decode consumed: its byte checks, else its
            # allocations (never its steps alone).
            consumed_ids = "checked_ids" if run.kv_check else "allocated_ids"
            handoff = _handoff_problems(
                joint_steps[("prefill", tp)], joint_steps[("decode", tp)],
                set().union(*(summary[consumed_ids] for summary in of["decode"].values())))
            problems.extend(f"{at}{problem}" for problem in handoff["problems"])
        per_tp.append({
            "kv_checks": byte_checks, "consumed": consumed, "kv_bytes_checked": checked,
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
                            f"{carried_by_ranks} (another rank's KV, or traffic the connectors did not count)")
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
                             **{key: value for key, value in summary.items()
                                if key not in ("requests", "checked_ids", "allocated_ids")},
                             "requests": len(summary["requests"])}
                            for (dp, tp), summary in ranks[role].items()] for role in sides},
        "ttft_ms": {key: bench.get(f"{key}_ttft_ms") for key in ("mean", "median", "p99")},
        "tpot_ms": {key: bench.get(f"{key}_tpot_ms") for key in ("mean", "median", "p99")},
    }


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


def _byte_check_problems(layout: dict, checks: list[dict], input_len: int,
                         plans: dict[str, dict[int, tuple[int, int]]],
                         prompts: dict[str, int] | None = None) -> list[str]:
    """How a rank's successful byte checks disagree with its KV layout.

    An older simulator's layout (no ``groups``): each check's bytes are its
    blocks times its layers times the layout's block bytes. Otherwise each
    check has blocks, bytes and nested results covering each KV cache group
    once, its blocks and bytes their sums; each group holds blocks (with
    prefix caching off a pull has some of every group), its bytes being
    its blocks times its layers' block bytes; a window pulls its last
    blocks only (Mooncake's cdiv(window, block) + 1), other attention from
    the first, all of the request's prompt (`prompts`, by decode's request
    id): cdiv(tokens, block) blocks of its N tokens, N - 1 for a hybrid
    model, whose decoder computes the last token; a gated delta-net's state
    is one block, at place 0. For several groups or a window, each group
    must also be the request's pull as posted (``_pull_plans``), and prompts
    longer than a window must have had some pull clipped. Missing fields are
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
    hybrid = any(spec.get("kind") == "MambaSpec" for spec in groups)
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
        plan = plans.get(request) if grouped else None
        if grouped and plan is None:
            problems.append(f"{request}'s pull plan was not logged (reads_posted with allocated_groups)")
        for g in nested:
            spec = groups[g["group"]] if 0 <= g["group"] < len(groups) else None
            if spec is None:
                continue
            if g["blocks"] < 1:
                problems.append(f"{request}'s group {g['group']} checked no blocks")
            if (g["layers"], g["bytes"]) != (spec["layers"], g["blocks"] * spec["layers"] * spec["block_bytes"]):
                problems.append(f"{request}'s group {g['group']} check disagrees with the layout "
                                f"({spec['layers']} layers of {spec['block_bytes']} bytes a block)")
            window = spec.get("sliding_window")
            if window is None and g["first_logical_block"] != 0:
                problems.append(f"{request}'s group {g['group']} ({spec.get('kind')}) was pulled from block "
                                f"{g['first_logical_block']}, not 0")
            prompt = (prompts or {}).get(request)
            if spec.get("kind") == "MambaSpec":
                if g["blocks"] != 1:
                    problems.append(f"{request}'s state group {g['group']} checked {g['blocks']} blocks, not 1")
            elif window is None and prompt is None:
                problems.append(f"{request}'s group {g['group']} pages cannot be checked: decode logged no "
                                "steps of it")
            elif window is None:
                # A hybrid's decoder computes the last of N > 1 tokens; one
                # token's pages are pulled whole, the token computed again.
                tokens = prompt - 1 if hybrid and prompt > 1 else prompt
                if g["blocks"] != -(-tokens // spec["block_size"]):
                    problems.append(f"{request}'s group {g['group']} checked {g['blocks']} blocks, not the "
                                    f"cdiv({tokens}, {spec['block_size']}) of its prompt")
            if window is not None:
                bound = -(-window // spec["block_size"]) + 1
                if g["blocks"] > bound:
                    problems.append(f"{request}'s window group {g['group']} pulled {g['blocks']} blocks, "
                                    f"more than its window's {bound}")
                clipped[g["group"]] += g["first_logical_block"] > 0
            if plan is not None and plan.get(g["group"]) != (g["first_logical_block"], g["blocks"]):
                first, blocks = plan.get(g["group"]) or (None, 0)
                problems.append(f"{request}'s group {g['group']} checked {g['blocks']} block(s) from "
                                f"{g['first_logical_block']}, not the pull's {blocks} from {first}")
    for index, spec in enumerate(groups):
        window = spec.get("sliding_window")
        if checks and window is not None and input_len > window + 2 * spec["block_size"] and not clipped[index]:
            problems.append(f"no pull of window group {index} was clipped, with prompts of {input_len} tokens "
                            f"beyond its {window}")
    return _capped(problems)


def _request_steps(records: list[dict]) -> dict[str, list[tuple[int, int, int]]]:
    """Each request's steps in a step log: (computed, scheduled, prompt)
    tokens, in order."""
    steps: dict[str, list[tuple[int, int, int]]] = {}
    for record in records:
        for request in (record.get("step") or {}).get("requests", []):
            steps.setdefault(request["req_id"], []).append(
                (request["num_computed_tokens"], request["num_scheduled_tokens"], request["num_prompt_tokens"]))
    return steps


def _handoff_problems(prefill: dict[str, list[tuple[int, int, int]]],
                      decode: dict[str, list[tuple[int, int, int]]],
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
        computed, scheduled, prompt = steps[0]
        if prompt <= 1:
            continue
        prompts.append(prompt)
        if (computed, scheduled) != (prompt - 1, 1):
            problems.append(f"{request} resumed at {computed} tokens with {scheduled} scheduled, not at "
                            f"{prompt - 1} with 1")
        steps = produced.get(_base_request_id(request))
        if steps is None:
            problems.append(f"{request} has no prefill steps")
            continue
        computed, scheduled, prefilled = steps[-1]
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


def _prefix_caching_flags(run: Run) -> list[str]:
    """The server's prefix caching: as set, else off for the byte checks
    (the pattern is the request's), else vLLM's default."""
    enabled = run.server.enable_prefix_caching
    if enabled is None:
        enabled = False if run.kv_check else None
    if enabled is None:
        return []
    return ["--enable-prefix-caching" if enabled else "--no-enable-prefix-caching"]


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
