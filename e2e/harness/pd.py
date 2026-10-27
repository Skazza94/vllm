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
TENT_METRICS_PORT = 9100  # rank r serves on TENT_METRICS_PORT + r
DISABLED_TENT_TRANSPORTS = ("tcp", "rdma", "shm", "gds", "ascend_direct")


@dataclass
class Side:
    node: str
    kv_network: str | list[str]  # a list: one per tensor-parallel rank

    def rank_networks(self, ranks: int) -> list[str]:
        if isinstance(self.kv_network, str):
            return [self.kv_network] * ranks
        return list(self.kv_network)


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
    """``vllm serve`` for ``role`` (prefill or decode), rank r's KV on
    ``kv_addresses[r]`` (their own: ``rank_addresses``)."""

    kv_address = kv_addresses[0]

    simulation = {"clock": "real", "timing": "roofline", "timing_args": {"gpu": run.server.gpu},
                  "log_path": STEP_LOG, "kv_transfer_mode": "real"}
    if run.kv_check:
        simulation["kv_pattern_fill" if role == "prefill" else "kv_pattern_check"] = True
    kv_transfer = {"kv_connector": "MooncakeConnector",
                   "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
                   "kv_buffer_device": "cpu", "kv_connector_extra_config": {"mooncake_protocol": "tcp"}}
    if len(set(kv_addresses)) > 1:
        kv_transfer["kv_connector_extra_config"]["rank_addresses"] = kv_addresses
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
        [(output / _step_log_name("prefill", r)).read_text() for r in range(ranks)],
        [(output / _step_log_name("decode", r)).read_text() for r in range(ranks)],
        routers, run, tent)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if verdict["problems"]:
        raise AssertionError(f"run {run.name} failed its checks ({output}): {'; '.join(verdict['problems'])}")
    return verdict


def check(bench: dict, prefill_steps: list[str], decode_steps: list[str], routers: dict[str, int],
          run: Run, tent: dict[str, list[dict]] | None = None) -> dict:
    """The run's checks, from the client's results, every rank's step logs
    (``prefill_steps[r]``, ``decode_steps[r]``), the bytes each router
    received during the workload and each rank's TENT counters over it
    (``tent[role][r]``). Requests are counted once, from the client; bytes
    per rank, then summed."""

    problems = []
    w = run.workload
    ranks = run.server.tensor_parallel_size
    errors = [error for error in bench.get("errors", []) if error]
    if bench.get("completed") != w.num_prompts or errors:
        problems.append(f"{bench.get('completed')} of {w.num_prompts} request(s) completed; errors: {errors[:3]}")

    per_rank = []
    for rank in range(ranks):
        where = f"rank {rank}: " if ranks > 1 else ""
        decode = _records(decode_steps[rank])
        prefill = _records(prefill_steps[rank])
        checks = [record["kv_check"] for record in decode if "kv_check" in record]
        checked = sum(int(c.get("bytes") or 0) for c in checks)
        layouts = {role: next((r["kv_layout"] for r in records if "kv_layout" in r), None)
                   for role, records in (("prefill", prefill), ("decode", decode))}
        # Each log must be this rank's (not another's supplied twice).
        for role, layout in layouts.items():
            if layout is None:
                problems.append(f"{where}{role} logged no KV layout")
            elif (layout.get("tp_rank"), layout.get("tp_size")) != (rank, ranks):
                problems.append(f"{where}{role}'s log is TP rank {layout.get('tp_rank')} of "
                                f"{layout.get('tp_size')}, not {rank} of {ranks}")
        if run.kv_check:
            if len(checks) != w.num_prompts:
                problems.append(f"{where}decode byte-checked {len(checks)} request(s), not {w.num_prompts}")
            bad = [c for c in checks if not c.get("ok") or c.get("mismatched_bytes")]
            if bad:
                problems.append(f"{where}{len(bad)} request(s) received KV bytes unlike prefill's: {bad[0]}")
            layout = layouts["decode"]
            if layout and any(c.get("bytes") != c["blocks"] * c["layers"] * layout["block_bytes"]
                              for c in checks if "blocks" in c):
                problems.append(f"{where}byte checks disagree with the rank's KV layout ({layout['block_bytes']} "
                                "bytes a block and layer)")
        sent = {"count": 0, "bytes": 0, "failed": 0}
        for record in prefill + decode:
            for transfers in (record.get("kv_transfers") or {}).values():
                for key in sent:
                    sent[key] += int(transfers.get(key) or 0)
        if sent["failed"]:
            problems.append(f"{where}the connector failed {sent['failed']} transfer(s)")
        if sent["bytes"] < checked:
            problems.append(f"{where}the connector moved {sent['bytes']} bytes, fewer than the {checked} decode "
                            "checked")
        rank_tent = {role: tent[role][rank] for role in tent} if tent else {}
        required = {"prefill": ("tent_registered_buffer_bytes", "tent_write_bytes_total"),
                    "decode": ("tent_registered_buffer_bytes",)}
        for role, counters in rank_tent.items():
            if counters is None:
                problems.append(f"{where}{role}'s TENT metrics did not answer")
            elif missing := [name for name in required[role] if name not in counters]:
                problems.append(f"{where}{role}'s TENT metrics lack {', '.join(missing)}")
        for role, counters in rank_tent.items():
            registered = (counters or {}).get("tent_registered_buffer_bytes")
            layout = layouts[role]
            if registered is not None and layout and registered < layout["bytes"]:
                problems.append(f"{where}{role}'s TENT registered {registered:.0f} bytes, less than its "
                                f"{layout['bytes']} of KV")
        written = (rank_tent.get("prefill") or {}).get("tent_write_bytes_total")
        if tent and written is not None and written != sent["bytes"]:
            problems.append(f"{where}TENT wrote {written:.0f} bytes, not the {sent['bytes']} its connector moved")
        per_rank.append({
            "kv_checks": len(checks), "kv_bytes_checked": checked, "transfers": sent,
            "kv_layout": {role: layout and {key: layout.get(key) for key in
                                            ("tp_rank", "tp_size", "shape", "block_bytes", "bytes",
                                             "tent_metrics_port")}
                          for role, layout in layouts.items()},
            "tent": rank_tent,
        })

    moved = sum(rank["transfers"]["bytes"] for rank in per_rank)
    # Every KV byte crosses its rank's network's router once, and no other:
    # each carries its ranks' bytes and little more (headers, acknowledgements).
    expected: dict[str, int] = {}
    for rank, network in enumerate(run.prefill.rank_networks(ranks)):
        expected[network] = expected.get(network, 0) + per_rank[rank]["transfers"]["bytes"]
    for network, carried_by_ranks in expected.items():
        carried = routers.get(network, 0)
        if carried < carried_by_ranks:
            problems.append(f"the KV network {network} carried {carried} bytes, fewer than the {carried_by_ranks} "
                            "its ranks moved")
        elif carried > carried_by_ranks * 1.05 + (1 << 20):
            problems.append(f"the KV network {network} carried {carried} bytes, more than its ranks' "
                            f"{carried_by_ranks}: another rank's KV")
    control = routers.get(run.control_network, 0)
    if moved and control > moved / 10:
        problems.append(f"the control network carried {control} bytes, too much beside {moved} of KV")
    sums = {key: sum(rank["transfers"][key] for rank in per_rank) for key in ("count", "bytes", "failed")}
    return {
        "problems": problems,
        "requests": bench.get("completed"),
        "kv_checks": per_rank[0]["kv_checks"],
        "kv_bytes_checked": sum(rank["kv_bytes_checked"] for rank in per_rank),
        "transfers": sums,
        "router_bytes": routers,
        "ranks": per_rank,
        "ttft_ms": {key: bench.get(f"{key}_ttft_ms") for key in ("mean", "median", "p99")},
        "tpot_ms": {key: bench.get(f"{key}_tpot_ms") for key in ("mean", "median", "p99")},
    }


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


def _step_log_name(role: str, rank: int) -> str:
    return f"{role}-steps.jsonl" if rank == 0 else f"{role}-steps.tp{rank}.jsonl"


def _tent_metrics(run: Run, container: dict[str, str]) -> dict[str, list[str | None]]:
    """Every rank's TENT metrics, as served (None: no answer)."""

    fetch = ("import sys, urllib.request\n"
             "try:\n    print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())\n"
             "except Exception:\n    print('NO ANSWER')\n")
    metrics: dict[str, list[str | None]] = {}
    for role, side in (("prefill", run.prefill), ("decode", run.decode)):
        metrics[role] = []
        for rank in range(run.server.tensor_parallel_size):
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
            copies += [(STEP_LOG if rank == 0 else f"{WORK}/steps.tp{rank}.jsonl", output / _step_log_name(role, rank))
                       for rank in range(run.server.tensor_parallel_size)]
        for remote, local in copies:
            local.parent.mkdir(parents=True, exist_ok=True)
            shell.run(["docker", "cp", f"{name}:{remote}", str(local)], check=False)
