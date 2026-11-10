"""Command line: python -m harness {up,down,status,exec,pd,recheck,peak-prefill} ..."""

from __future__ import annotations

import argparse
import json
import sys

from . import pd
from . import spec as specs
from . import topology


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("up", help="create a topology from its spec")
    up.add_argument("spec")
    down = commands.add_parser("down", help="remove a topology")
    down.add_argument("name", help="the topology's name, or its spec")
    status = commands.add_parser("status", help="a topology's containers and addresses")
    status.add_argument("name", help="the topology's name, or its spec")
    run = commands.add_parser("exec", help="run a command in one of a topology's nodes")
    run.add_argument("name", help="the topology's name, or its spec")
    run.add_argument("node")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    prefill_decode = commands.add_parser(
        "pd", help="a prefill/decode run of the vLLM simulator behind llm-d's router, in an up topology")
    prefill_decode.add_argument("run", help="the run config")
    recheck = commands.add_parser("recheck", help="a saved prefill/decode run's checks again (verdict-recheck.json)")
    recheck.add_argument("results", help="the run's results directory")
    peak = commands.add_parser(
        "peak-prefill", help="llm-d's peakPrefillThroughput (router.peak_prefill_throughput) from a saved run")
    peak.add_argument("results", help="the run's results directory (prompts of at least max_num_batched_tokens)")
    args = parser.parse_args(argv)

    if args.command == "up":
        print(json.dumps(topology.up(specs.load(args.spec)), indent=2))
        return 0
    if args.command == "pd":
        print(json.dumps(pd.execute(pd.load(args.run)), indent=2))
        return 0
    if args.command == "peak-prefill":
        found = pd.saved_peak_prefill_throughput(args.results)
        print(json.dumps(found, indent=2))
        return 0 if found["tokens_per_second"] else 1
    if args.command == "recheck":
        verdict = pd.recheck(args.results)
        print(json.dumps({"problems": verdict["problems"], "kv_bytes_checked": verdict["kv_bytes_checked"],
                          "transfers": verdict["transfers"]}, indent=2))
        return 1 if verdict["problems"] else 0
    name = _name(args.name)
    if args.command == "down":
        removed = topology.down(name)
        print(f"removed {len(removed)} container(s)")
    elif args.command == "status":
        print(json.dumps({"containers": topology.containers(name), **topology.load_state(name)}, indent=2))
    else:
        container = topology.load_state(name)["nodes"][args.node]["container"]
        sys.stdout.write(topology.exec_in(container, args.argv, timeout=None))
    return 0


def _name(value: str) -> str:
    """A topology's name, given it or its spec file."""

    if value.endswith((".yaml", ".yml")):
        return specs.load(value).name
    return value


if __name__ == "__main__":
    sys.exit(main())
