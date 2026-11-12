# Simulated Execution

Simulated execution runs the real vLLM engine (scheduler, KV cache manager,
prefix caching, chunked prefill, preemption, async scheduling and the OpenAI
frontend) but replaces the model forward pass with a timing model. It predicts
a deployment's TTFT, TPOT and throughput without its hardware or weights, and
exercises scheduler and KV cache behavior deterministically on CPU-only hosts.

## How it works

* The model is built on the `meta` device, so KV cache specs, attention
  metadata and hybrid layouts are vLLM's own, with no parameters in memory.
* The KV cache is virtual: blocks are accounted as usual within
  `--kv-cache-memory-bytes`, but no tensors are allocated.
* Output tokens come from `SamplingParams.trace_decode_token_ids` (see
  [trace replay](../serving/online_serving/trace_replay.md)) or, without a
  trace, are `fallback_token_id`; such requests end by `max_tokens`.
* A *timing provider* prices each step: its device time (`device_s`) runs on
  a modeled device queue, overlapping the next step's scheduling as under
  async scheduling, and its optional host time (`host_s`, spent in
  `execute_model` and `sample_tokens`) holds the engine thread. The rest of
  the engine loop is the real code and takes its real time.

Simulation runs on CUDA-alike hosts and on the CPU platform (see
[Running without a GPU](#running-without-a-gpu)).

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulate-forward \
  --kv-cache-memory-bytes 8589934592 \
  --simulation-config '{"timing": "linear",
                        "timing_args": {"fixed_us": 3000,
                                        "prefill_us_per_token": 20,
                                        "decode_us_per_req": 40},
                        "log_path": "/tmp/sim_steps.jsonl"}'
```

Any client works unchanged, including `vllm bench serve`. Offline:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    simulation_config={"simulate_forward": True, "timing": "linear",
                       "timing_args": {"fixed_us": 3000, "decode_us_per_req": 40}},
    kv_cache_memory_bytes=4 * 1024**3,
)
out = llm.generate(
    [{"prompt_token_ids": list(range(300))}],
    SamplingParams(trace_decode_token_ids=[501, 502, 503]),
)
```

Set `--kv-cache-memory-bytes` to the KV budget of the modeled deployment, as
its startup log reports it: prefix-cache hit rates and preemption depend on it.

### Timing providers

| `timing` | behavior | `timing_args` |
| --- | --- | --- |
| `zero` | steps take no device time (default) | none |
| `linear` | `fixed_us + prefill_us_per_token * P + decode_us_per_req * D + decode_us_per_kv_token * K` | those four, plus `host_us` or `host_fit_log` |
| `table` | nearest-neighbor interpolation over a step log | `path`, `k` |
| `replay` | exact match on the step's batch composition, else `fallback` | `path`, `fallback` (`knn`/`zero`), `k`, `miss_fallback_s` |
| `aisimulate` | AISimulate's forward-pass estimate of the published model (below) | `gpu`, `target_model`, `backend_version`, `quant_modes`, `host_us` |
| `module.Class` | a `vllm.v1.worker.simulated.timing.TimingProvider` | its keyword arguments |

A provider receives a `StepDescriptor`: per request, the tokens scheduled in
the step, the KV length before it, the prompt length, prefix-cache hits at
admission, draft tokens, whether it is prefilling and, in an MoE-DP joint
step, its DP rank; plus batch totals, common-prefix blocks and the parallel
configuration. It returns a `StepCost`.

For a GPU's step times without a recording, `aisimulate` asks
[AISimulate](https://github.com/ai-dynamo/aisimulate)'s
forward-pass performance model for each step. It needs the optional
`aisimulate` package and prices the scheduler's actual batch, one workload
per attention data-parallel rank, from kernel timings measured on that GPU
under a vLLM release, with no analytical fallback.

`docker/Dockerfile.aisim` adds the pinned package to the CPU image (see
[Running without a GPU](#running-without-a-gpu) for `vllm:cpu`). From the
repository root:

```bash
docker build -f docker/Dockerfile.aisim -t vllm:cpu-aisim .
```

The deployment it times is the published model `target_model`, with that
model's published weight precision. The model's own Hugging Face id or a
`<org>--<name>/<commit>` directory names the target; a derived copy (a
directory with `DERIVED.md`) must name it explicitly. The local model only
holds the KV cache and serves the scheduler, so its KV and state geometry
must match the target's bundled config, and the local KV cache precision is
the one timed. The published precision is read from the target's full
config, a ModelOpt checkpoint's `hf_quant_config.json` included: block FP8
and MXFP4 experts by a fixed mapping; NVFP4, per-tensor FP8,
compressed-tensors and mixed per-layer precision as AISimulate itself reads
the checkpoint; any other is refused. Layers the server quantizes itself
are not the published weights. Of an unquantized checkpoint,
`--quantization fp8` (or `fp8_per_tensor`: per-tensor weights, dynamic
activations) and `fp8_per_block` are timed as AISimulate's `fp8` and
`fp8_block` precision of the published model, the target recorded as
modified with its `online_quantization`. The shorthand is taken alone, as
vLLM resolves it: every linear layer and every expert quantized its way.
A `--quantization-config` beside it that changes anything (layers ignored,
per-layer `targets`, another weight or activation precision) is refused
before the model is built, and the built layers must be that method's. Any
other online quantization (another method, `--quantization online` with a
config, layers quantized beside a quantized checkpoint) is refused at
startup rather than timed as another deployment. `quant_modes` overrides the published
precision, which
records the target as modified. DeepSeek-V4 is timed with its packed FP8
cache (`auto`, `fp8`, `fp8_e4m3` or `fp8_ds_mla`; others are rejected)
and its FP4 experts. A DSA model's FP8 cache (DeepSeek-V3.2's, GLM-5's),
under any of those FP8 names, is timed and recorded as the packed one too
(`kv_cache_format: fp8_ds_mla`, the configured dtype beside it). GLM-5 on
B200 has to name its cache (`fp8`, the GPU's default there, or `bfloat16`):
`auto` is FP8 on that GPU and BF16 on a CPU host. A bundled config AISimulate would misread is given to
it as a private normalized copy, the change and both configs' hashes in the
`timing` record: MiMo-V2-Flash's `hybrid_layer_pattern`, which marks its
sliding layers 1 where AISimulate reads global ones, is inverted. Startup
also times one prefill and one decode step: a target AISimulate lacks data
for at those shapes fails before serving,
but a later batch of another shape can still lack data and fail then.

Supported layouts are tensor parallelism within one node (TP 1, 2, 4
or 8) and an MoE model's local data-parallel attention with expert
parallelism (each rank of TP 1, 2 or 4, timed as vLLM lays it out: attention
on a rank's GPUs, the experts over all of them; up to 8 GPUs, all on one
host: launched by one server,
or a server each by a multi-port supervisor,
`--data-parallel-multi-port-external-lb`; ranks on other hosts are
rejected). Pipeline and context parallelism and EPLB are rejected, and so are models or layouts AISimulate has no timing
for. Speculative decoding is timed for the native single-module MTP of
DeepSeek-V3, DeepSeek-V3.2, GLM-5, DeepSeek-V4 and Qwen3.5 only (`method: mtp`, the checkpoint's one MTP layer,
1 to 5 tokens, the draft at the target's parallelism and precision), as
AISimulate's MTP depth: a decode step
verifies k + 1 tokens a request and runs k target-shaped draft layers, the
full width even when a step drafted fewer; draft weights and KV, prefill
draft work and host glue are omitted. Acceptance comes from
`spec_acceptance_rates` and does not change a step's cost. The `timing`
record names the target, the resolved estimator configuration and its
data's vLLM version.

`table` and `replay` read a recorded step log (see below). A `table` can also
be written by hand, as JSONL records of
`{"num_reqs", "num_tokens", "sum_past_kv", "device_s"[, "host_s"]}`: the
step's requests and scheduled tokens, and the sum of their KV lengths before
the step.

`HostTimeModel` prices host time as `fixed_us + per_req_us * R +
per_prefill_token_us * P`, fitted to a step log (`HostTimeModel.from_log`, or
`host_fit_log` for `linear`), for providers that model only the device. Under
async scheduling host time matters only where it exceeds device time, mostly
in prefills, so it mainly affects TTFT. Fit it to a recording of the same
parallel layout and speculative decoding setup, made under async scheduling;
terms the recording does not determine stay zero.

### Recording a real deployment

A real deployment writes the same step log, timed with CUDA events:

```bash
vllm serve Qwen/Qwen3-0.6B \
  --simulation-config '{"record_step_timing": true, "log_path": "/tmp/real_steps.jsonl"}'
```

`replay` with this log reproduces the recorded step times wherever the
simulated scheduler forms the same batches; `table` interpolates them for
other workloads. Both return the recorded device and host times. Host time
matters: large prefills are often host-bound, and pricing them by device time
alone underestimates TTFT. The recorded host time spans the engine thread
from `execute_model` to the end of `sample_tokens`, so it includes engine core
work between the two calls, such as structured-output bitmasks.

### Offline runs in simulated time

`vllm.v1.worker.simulated.offline` drives an in-process engine with timed
arrivals of requests and multi-turn sessions, and timestamps every output
token:

```python
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.simulated.offline import TimedRequest, simulate

requests = [
    TimedRequest(prompt_token_ids=list(range(512)),
                 sampling_params=SamplingParams(max_tokens=128, ignore_eos=True),
                 arrival_time=0.05 * i)
    for i in range(200)
]
result = simulate(
    EngineArgs(
        model="Qwen/Qwen3-0.6B",
        kv_cache_memory_bytes=8 * 1024**3,
        simulation_config={"simulate_forward": True, "clock": "fast_forward",
                           "timing": "replay",
                           "timing_args": {"path": "/tmp/real_steps.jsonl"}},
    ),
    requests,
)
print(result.summary())  # as vllm bench serve reports them
```

`simulation_config.clock` selects the time base:

| `clock` | waits (device time, idle gaps) | engine host loop | use |
| --- | --- | --- | --- |
| `real` (default) | slept | real time | serving |
| `fast_forward` | skipped | real time | sweeps: the latencies of `real`, faster |
| `virtual` | skipped | free | deterministic runs; underestimates host-bound workloads |

Predicted `host_s` holds the engine thread under every clock. The non-real
clocks need an engine from `make_engine()` (which `simulate()` uses), and a
run then takes as long as the engine's host work. `make_engine()` and `run()`
also drive a real engine on the real clock, to produce a reference run.
[`vllm bench sweep simulate`](../benchmarking/sweeps.md#simulated-sweep) runs a
`vllm bench sweep serve` experiment this way.

Only the engine core runs, so `frontend_latency_s` (of `run()` and
`simulate()`) adds the server's latency outside it (HTTP, API server, IPC) to
each request's arrival.

A `TimedSession` is a multi-turn conversation: each `Turn` appends its tokens
to the previous turn's prompt and output, and arrives `think_time` seconds
after the previous turn finished, so later turns reuse the conversation's KV
cache. Results have one timing per turn, in input order. Prompts are
concatenated token ids, so chat-template tokens belong in each turn's
`new_token_ids`, and `max_model_len` must fit a whole conversation.

```python
from vllm.v1.worker.simulated.offline import TimedSession, Turn

params = SamplingParams(max_tokens=128, ignore_eos=True)
session = TimedSession(
    turns=[Turn(first_prompt_ids, params)]
    + [Turn(user_ids, params, think_time=3.0) for user_ids in follow_ups],
    arrival_time=0.0,
)
```

### Speculative decoding

The scheduler schedules draft tokens as usual, and the simulated runner
decides how many are accepted. The draft model is built on the meta device
too, and its KV cache counts against the budget.

`spec_acceptance_rates` gives the probability that the draft at each position
is accepted given that all earlier ones were; the last value repeats. A step
log recorded with speculative decoding holds every request's accepted count,
and its path estimates the rates instead:

```python
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "spec_acceptance_rates": "/tmp/real_steps.jsonl",
}
```

A provider's `StepCost.num_accepted` overrides the rates. Acceptance is
sampled independently per step, so per-request acceptance, and with it P99
TPOT, varies less than on a real deployment. Trace replay is not available
with speculative decoding.

### Parallelism

Tensor, pipeline, data and context (prefill and decode) parallelism run as
deployed: one simulated worker per rank builds its shard of the model, so
per-rank KV capacity is vLLM's own; `--kv-cache-memory-bytes` is per rank.
No collective runs in a step: every rank derives the same timing from the
same scheduler output.

* **Pipeline parallelism**: `device_s` is a step's time through the pipeline,
  split evenly over the stages, so consecutive microbatches overlap. Steps
  recorded under pipeline parallelism carry the same through-pipeline time.
  The last stage's host time includes receiving intermediate tensors, which
  can bound the step rate under load; fit host time to a recording of the
  same layout.
* **Data parallelism**: dense models run independent engines, and DP rank
  `i > 0` logs to `<log_path>.dp<i>`. MoE models run their DP ranks in
  lockstep: wherever the real runner synchronizes them, the ranks gather
  their step descriptors into one joint step, priced once and logged by DP
  rank 0, busy or idle; each request keeps its DP rank. Joint steps do not
  record accepted drafts.

Only the output rank (last pipeline stage, first tensor-parallel rank) writes
the step log. On CUDA hosts each rank binds its own device, using a few MB of
its memory.

### KV cache offloading

CPU offloading by the `OffloadingConnector` (its CPU spec) runs with the real
scheduler side, so the CPU cache's hits, evictions and load and store jobs are
the real ones; a simulated worker replaces the copy streams. Each direction
is a serial queue, a transfer starts once the device has finished the step
running when it was submitted, and it takes `latency + bytes / bandwidth`.

`kv_transfer_timing` gives the bandwidths in 1e9 bytes/s, with optional
latencies:

```python
kv_transfer_config={
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"cpu_bytes_to_use": 32 << 30},
},
simulation_config={
    "simulate_forward": True,
    "timing": "replay",
    "timing_args": {"path": "/tmp/real_steps.jsonl"},
    "kv_transfer_timing": {"load_gbps": 25, "store_gbps": 25},
},
```

A step log recorded with offloading also holds the transfers completed in
each step; its path as `kv_transfer_timing` fits each direction's latency and
bandwidth by least squares. A transfer's size is its blocks' KV page size,
which overstates layouts where the real copy moves less (padded hybrid pages,
canonical-layout stores split across ranks); rates fitted to the same model
absorb this. A step that waits for stores the scheduler flushes pays the wait,
which its recorded host time also includes.

### Disaggregated prefill

`vllm.v1.worker.simulated.disaggregated` runs prefill and decode instances
connected by the `NixlConnector` (pull mode) in one process, in simulated
time:

```python
from vllm.v1.worker.simulated.disaggregated import simulate_disaggregated

result = simulate_disaggregated(
    prefill_args,  # EngineArgs with simulate_forward and a timing provider
    decode_args,
    requests,      # TimedRequests
    num_prefill=2,
    num_decode=1,
    handoff_latency_s=7e-3,
)
```

`EngineArgs` built in Python take the generic scheduler defaults rather than
`vllm serve`'s: set `max_num_batched_tokens` and `max_num_seqs` as the served
instances have them. (`vllm bench sweep simulate` applies `vllm serve`'s
defaults.)

The driver plays the proxy: a request goes to a prefill instance with
`max_tokens=1`, then to a decode instance with the `kv_transfer_params` it
returned. Instances are picked round-robin or, with
`routing="least_unfinished"`, by fewest unfinished requests;
`handoff_latency_s` is the proxy's latency between the two. The connector's
scheduler side is the real one; its worker side moves no data. A read takes
`latency + bytes / bandwidth` at `kv_transfer_timing`'s load rate, reads
queue on each instance's link, and a completed read frees the prefill
instance's blocks. Leases do not expire, since heartbeats run on the wall
clock. Each instance has one rank and its own virtual clock; the instance
furthest behind steps next.

Posting a step's reads holds a decode instance's engine thread. NIXL merges
blocks consecutive on both instances into one descriptor per KV cache
region, so posting slows once a decode instance's blocks scatter, after its
KV cache has filled once. The simulation charges `post_us_per_read` per read
plus `post_us_per_run` per run of consecutive blocks. A decode instance
recorded with the real `NixlConnector` logs its reads and postings, and its
step log as `kv_transfer_timing` fits the read rate and posting costs. Record
until its KV cache has filled, or the cost of scattered reads is
extrapolated.

#### Served instances

Simulated prefill and decode instances also run as servers, on the real
clock, with tensor parallelism, behind any proxy:

```bash
VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen3-8B --port 8100 \
  --tensor-parallel-size 2 --simulate-forward \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer"}' \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "p.jsonl"}, "kv_transfer_timing": "d.jsonl"}'
VLLM_NIXL_SIDE_CHANNEL_PORT=5700 vllm serve Qwen/Qwen3-8B --port 8200 \
  --tensor-parallel-size 2 --simulate-forward \
  --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}' \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "d.jsonl"}, "kv_transfer_timing": "d.jsonl"}'
```

A decode rank books its reads over ZMQ with the prefill ranks the real
connector maps it to. Each prefill rank listens on a port derived from its
instance's side-channel port and its rank, which must be free. The two
instances' tensor parallelism may differ as with the real connector, and a
request's reads from several ranks run concurrently. A read that cannot be
booked, or a layout the real connector rejects, fails its request, which the
decode instance then prefills itself. Pipeline parallelism (unsupported by
the real pull connector), decode context parallelism and hybrid (Mamba)
layouts are not modeled.

#### Push mode and pipeline parallelism

With the `NixlPushConnector`, a prefill instance with pipeline parallelism
writes into a decode instance without it. As with the real connector, the
proxy sends a request to both instances at once. Decode ranks register each
request with their paired prefill ranks in every stage and forward their
heartbeats; a prefill rank writes its share once the request is both
finished and registered, and frees the blocks when its writes end or the
lease of an unregistered request (a full prefix hit) expires. Here the
prefill instance records the transfers, so fit `kv_transfer_timing` to its
step log.

#### Real KV transfers

With `"kv_transfer_mode": "real"`, compute stays simulated but the KV cache
is real: CPU tensors that a native connector (`NixlConnector`,
`NixlPullConnector` or `MooncakeConnector`) registers and moves between the
instances, so transfers take their actual time over the actual network. It
needs an explicit `kv_cache_memory_bytes`. Supported layouts:

* Full-attention models, dense or MoE (experts hold no KV), and dense MLA
  models (one latent per token and layer); sliding-window attention too,
  alone or beside full attention (each layer kind a KV cache group, groups
  sharing memory, a block being one group's at a time, of different block
  sizes when their pages are equal: Gemma4's and MiMo-V2-Flash's full and
  sliding layers at most TP sizes): either connector moves each window's
  last blocks only, and says where they start among the request's blocks
  (NIXL: `local_first_block_indices`). Over NIXL a model of several groups
  is checked through NIXL's own selection: a group's block in each layer
  slot NIXL selects for it, a slot holding none of a shorter group's layers
  moved and checked too (transfer-only, reported apart); and hybrid
  models of Qwen's gated delta-net (Qwen3.5, served
  as its language model), of Kimi-K3's delta attention (KDA beside MLA,
  served as its language model; its SiTU experts the CPU platform builds
  only for a simulation) or of Mamba2 (`MambaMixer2`: Nemotron-H,
  Nemotron-3, whose non-gated relu^2 experts the CPU platform builds only
  for a simulation): each request's state is one block of its recurrent
  groups (`mamba_cache_mode` none), padded to the attention page, and
  prefill computes the prompt but its last token, which decode computes.
  With prefix caching vLLM keeps a state a block instead (`mamba_cache_mode`
  align): the state after the tokens up to the block's end, which a later
  prompt sharing them reuses, earlier states' places null; the connector
  moves the request's running state. The native runner copies a state from
  block to block around the forward; a simulated step computes none and
  copies none.
  `MooncakeConnector` moves the padded state pages. NIXL needs the DS
  conv-state layout (`VLLM_SSM_CONV_STATE_LAYOUT=DS`, set before startup)
  and moves each state as it selects it: the conv sub-projections and the
  SSM state of every state region, a region holding none of a group's
  layers included (transfer-only), the page's padding not; the pattern
  checks follow that selection. Each
  decoder layer holds one recurrent mixer at most, without attention beside
  it. Not other recurrent layers (Mamba, Kimi-Linear's or OLMo's
  delta-nets), a mixer beside attention (Falcon-H1, Zamba2), ReplaySSM or
  RecoverSSM, attention-free,
  other sparse-MLA (but DeepSeek-V4's, below), chunked local or RSWA
  attention models.
* DeepSeek-V3.2's and GLM-5's sparse MLA (DSA), over either connector, its
  latent cache BF16 (`auto` or `bfloat16`) or FlashMLA's packed FP8
  (`fp8`, `fp8_e4m3` or `fp8_ds_mla`: 656 bytes a token): each attention
  layer's latent (576 wide in BF16) and, on the layers
  holding an indexer (GLM-5's every few, by its config), the indexer's FP8
  keys and scales (132 bytes a token), pages of two sizes in one group, each
  layer its own region; the pattern checks cover each layer's page.
* DeepSeek-V4 over `NixlConnector` (or `NixlPullConnector`), on one rank,
  with tensor parallelism, alone or inside data-parallel attention with
  expert parallelism, each TP rank holding and moving the whole caches,
  with prefix caching or without, its cache the packed FP8 one (`fp8_ds_mla`): its
  sliding windows, compressed latents, indexer keys and compressor states
  are KV cache groups packed into one allocation, each block all of one
  group's layers; NIXL moves whole blocks. Decode loads all N prompt
  tokens and computes the last one again to sample; with
  bounded window replay on (a cache whose SWA spec sets `bounded_replay`,
  DeepSeek-V4.1's), NIXL loads N - 1 and decode computes the last.
* Speculative decoding only as the native single-module MTP of
  DeepSeek-V3, DeepSeek-V3.2, GLM-5, DeepSeek-V4 or Qwen3.5 (`method: mtp`, the checkpoint's one MTP layer, 1 to 5
  tokens, the draft at the target's parallelism and precision) over
  `NixlConnector` (or `NixlPullConnector`), whose handshake checks the
  instances' transfer compatibility, the draft among it (a peer without
  the MTP draft is refused), or, for DeepSeek-V3, DeepSeek-V3.2 and GLM-5,
  over `MooncakeConnector`, which has no such check (it refuses a producer
  layer the consumer lacks, not a consumer-only draft or another depth):
  both instances must run the same speculative configuration. The MTP
  layer's KV is a layer of the target's KV cache groups and moves with
  them. Qwen3.5's MTP stays NIXL's: Mooncake would also move each state's
  k scratch slots, which NIXL leaves. Drafts are not computed; their
  acceptance follows `spec_acceptance_rates`.
* One rank per instance, or:
    * tensor parallelism of the same size on both instances: each rank moves
      its own KV heads to its peer, or the whole MLA latent, which every rank
      holds;
    * tensor parallelism of different sizes, one dividing the other, for:
        * plain attention, sliding-window layers beside full ones included:
          the connector re-shards the KV heads either way, and each head's
          pattern is its own;
        * plain MLA (one latent a layer: DeepSeek-V3, Kimi-K2), the same
          page on every rank;
        * sparse MLA (DeepSeek-V3.2, GLM-5), BF16 or packed FP8: the latent's
          and the indexer's pages, each the same on every rank;
        * layers of different KV head counts (Gemma4's global layers beside
          its sliding ones), only where every layer's count is at least the
          larger size and divisible by both (Gemma4's 8 and 2: TP 1 with 2);
        * over NIXL only, a hybrid of Mamba2 or gated delta-net states
          whose state splits over both sizes and whose KV heads split or
          replicate evenly on each side; where the two sizes give it
          different attention block sizes (Nemotron-3-Nano: 1056 tokens at
          TP 4, 2112 at TP 2), only a decode block size that is a multiple
          of the prefill's, as NIXL supports: each decode block read as
          several prefill blocks, the states one to one, each attention
          page read whole from one producer (NIXL splits no head slice at
          a block-size ratio), without prefix caching for byte checks.
          This follows the simulated backend's FlashAttention-style kernel
          blocks (kernel block = attention block); a backend with small
          fixed kernel blocks (FlashInfer's), where NIXL pairs blocks by
          their physical ratios instead, is not modelled;

      and not for DeepSeek-V4's caches, an MLA hybrid (Kimi-K3) or
      speculative decoding. A NIXL decode instance refuses, as
      NIXL does, a larger prefill instance whose KV heads are replicated
      (more ranks than KV heads); MLA's latent is exempt. A refused pair
      fails at the handshake;
    * an MoE model's data-parallel attention with expert parallelism
      (`--data-parallel-size N --enable-expert-parallel`): each DP rank holds
      its own requests' KV; a request's prefill DP rank (pinned by
      Mooncake's proxy, else prefill's load balancer) names itself in its
      transfer parameters (NIXL: its engine id and side-channel port), and
      decode's load balancer picks its own, so the instances' DP sizes may
      differ. A DP rank may be tensor-parallel itself (equal TP sizes on
      both instances): each of its TP ranks moves its own KV, as above. With `--data-parallel-multi-port-external-lb` each rank has an
      API server of its own (the port plus its rank) and an external router
      picks the rank on both instances; this needs NIXL, as
      `MooncakeConnector`'s bootstrap server is one an instance.
* No pipeline or context parallelism, nor data parallelism of dense models
  (independent engines).
* Blocks of any multiple of 16 tokens, as on a GPU; MLA blocks are not
  forced to the CPU MLA kernel's 16.

`kv_pattern_fill` (prefill instance) writes a pattern into each request's KV
blocks, derived from the request, the TP rank, the layer and the block's
place among the request's blocks of its KV cache group, and
`kv_pattern_check` (decode instance) checks every byte it receives, logging
one `kv_check` record per request, each group's result nested in it (it
needs `log_path`). With prefix caching on, a block may hold KV computed for
another request, so its pattern is its content's instead: the tokens up to
the block's end, as prefix caching keys a full block. Prefix hits are then
compared too, read-only, on both instances (`kind` hit, apart from the
transferred blocks). Such a run takes token prompts only (no LoRA adapter,
`cache_salt`, prompt embeddings or multimodal inputs) and no speculative
decoding. A hybrid's state is its block's content
too: the tokens it has absorbed, which stop one short of the prompt on a
prefill instance (decode expects that). A hit ending inside a block is
reported as outside the checks. Decode reuses a cached prefix of its own
only where its pulls left one: not a sliding window's (the connector moves
a window's last blocks only: DeepSeek-V4, Gemma4), and for a hybrid its
attention blocks over NIXL, nothing over Mooncake. A group's blocks
are its layers' only, each block the page the connector moves (a state's
padding included; in DeepSeek-V4's packed blocks, each layer's real page,
its padding unchecked), and the null block that pads a window's head is
neither filled nor checked. The `kv_layout` record describes each group (its page,
the state's real bytes, the stride between blocks) and counts each storage
once, as the connector registers it.

`MooncakeConnector` takes one KV address per rank in its extra config,
`"rank_addresses": [...]` (IPv4), so each rank can use a NIC of its own: its
handshakes, TENT's RPC server and `hp_tcp` bind there. So does
`NixlConnector` over NIXL's Mooncake backend alone (`"backends":
["Mooncake"]`): each rank's backend segment and `hp_tcp` bind to its
address, while NIXL's handshake stays on the instance's side channel
(`VLLM_NIXL_SIDE_CHANNEL_HOST`). The list holds the
ranks of an engine, by TP rank; an MoE model's DP engines on one host share
it, host-local DP rank by TP rank (`local_kv_rank`), and so must be launched
together: by one server, or by one multi-port supervisor, which tells each
server it launches its place among the host's ranks
(`VLLM_DP_SUPERVISED_RANK_LOCAL`, `VLLM_DP_SUPERVISED_SIZE_LOCAL`; not to be
set by hand). Each rank's transfer engine serves TENT's metrics on the base
port (`MC_TENT_CONF`'s `metrics/http_port`, else `TENT_METRICS_HTTP_PORT`,
else 9100) plus its host-local DP rank times the TP size plus its TP rank.

NIXL can also select exactly `["UCX"]` or `["LIBFABRIC"]` for CPU TCP
transfers. Both require `rank_addresses` even for TP1; the selected IPv4
address must identify one non-loopback interface without IPv4 aliases.
UCX uses `tcp,self` and that interface. LIBFABRIC uses the direct `tcp`
provider, IPv4 only, and requires its interface to be UP and RUNNING.
Use the CPU image built with the corresponding plugins; the harness in
`tools/e2e` selects them with `server.nixl_backend: ucx` or `libfabric`
while retaining `mooncake` as its default.

Managed LIBFABRIC sets provider/interface policy before loading the native
library. It requires `/opt/libfabric/etc/libfabric.conf` to be absent or
empty and refuses configuration/provider overrides and rebinding a process
to another interface. `FI_LOG_LEVEL`, `FI_LOG_PROV` and `FI_LOG_SUBSYS` are
allowed for diagnostics. Its agent config uses four posting threads and
background progress; these configured values do not independently observe
live threads or provider/rail selection. Backend options and policy are
logged separately. TENT accounting applies to Mooncake, while UCX and
LIBFABRIC use NIXL telemetry, byte checks and network counters. For both,
the logged binding is the environment set before the agent is created;
the native transport and device selection is not observed
(`native_transport_observed: false` for UCX).

A managed UCX or LIBFABRIC `add_remote_agent` can block indefinitely
when the peer's side channel answers but its KV address is unreachable
(a wrong `rank_addresses` entry, a missing route, a dead peer). The call
cannot be interrupted and holds NIXL's single handshake thread: later
handshakes wait behind it and the process does not exit until it is
killed. After 30 seconds an error names the backend and the likely
cause; check the peer's KV address and routes, then restart both
instances.

Every rank of the last pipeline stage logs its own KV (`kv_layout`,
`kv_transfers`, `kv_check`, `kv_events` and `threads` records), and with
speculative decoding its own requests' drafts and accepted drafts each
step (`spec_acceptance`), into
`<log_path>[.dp<i>][.tp<r>]`, for its global DP rank `i` and TP rank `r`
when not 0; its `kv_layout` record names both and its host-local DP rank.
A `kv_transfers` record names each transfer's peer rank and its plan (for a
NIXL read, the descriptors selected and the count NIXL merges them into).
Under MoE DP the timing record and the joint steps stay in DP rank 0's logs,
also when a multi-port supervisor launched each rank's server. With
several KV cache groups or a sliding window, `kv_events` also lists each
request's new blocks by group (`allocated_groups`: the group, the place of
the first block among the request's blocks of that group, and the blocks,
null ones in place), the connector's pulls included.

### Running without a GPU

A CPU build of vLLM (`VLLM_TARGET_DEVICE=cpu`, which installs Triton's CPU
backend) runs simulations on any x86-64 host:

```bash
TRITON_CPU_BACKEND=1 vllm serve Qwen/Qwen3-235B-A22B --simulate-forward \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --kv-cache-memory-bytes 68719476736 \
  --simulation-config '{"timing": "replay", "timing_args": {"path": "steps.jsonl"}}'
```

The CPU image the examples here and the harness use is built from the
repository root:

```bash
docker build -f docker/Dockerfile.cpu --target vllm-openai \
  --build-arg VLLM_CPU_X86=true --build-arg ENABLE_AMX_FP8=0 \
  -t vllm:cpu .
```

Ranks need no devices, so a layout larger than any one GPU host runs on a
CPU host. Simulated workers share the host's cores instead of each taking a
NUMA node (nor does a `NixlConnector` worker pin itself to each node's last
core for its transfers), and keep the modeled GPU deployment's settings (async scheduling,
the in-process executor) rather than the CPU platform's: an MLA model keeps
its chunked prefill and prefix caching, and its blocks are not forced to the
CPU MLA kernel's 16 tokens. The KV block size must be a multiple of 16, a
GPU's smallest: pass `--block-size` to match the deployment.
Layers still pick the CPU platform's attention backends and kernels, which
keep no decode-context-parallel cache: `--decode-context-parallel-size` above
1 is refused there at startup.
Weight formats do not: a checkpoint published with quantized weights (FP8,
MXFP4, MXFP8 or NVFP4, ModelOpt's or compressed-tensors': DeepSeek-V3's,
DeepSeek-V3.2's, MiniMax-M2.5's, gpt-oss's, the `-NVFP4` checkpoints) builds
as published: serve it by its name, with no edited copy of its config. And
`--quantization` (`fp8`, `fp8_per_tensor`, `fp8_per_block`,
`fp8_per_channel`, `mxfp8`, `int8_per_channel_weight_only`) quantizes
the layers as they load, as on a GPU: they load on the meta device, so the
layers end with the deployment's weight dtypes and no data (not `mxfp4`,
whose emulation kernel needs the `amd-quark` package, nor `nvfp4_per_token`
on an MoE model, which vLLM takes on a Blackwell GPU only). A format the CPU
has no kernel for is built and never run: its linear layers take vLLM's
device-independent emulation kernel, NVFP4 experts the backend a GPU
deployment tries first (or `--moe-backend`), per-tensor FP8 experts the
Triton one. Weight quantization changes neither the KV cache nor what a
transfer moves. A checkpoint that also declares a quantized KV cache takes
that cache dtype as on its GPU: dense MLA keeps an FP8 cache (a byte an
element), DSA sparse MLA FlashMLA's packed one (`fp8_ds_mla`, 656 bytes a
token), whichever FP8 dtype is given. DSA sparse MLA
(DeepSeek-V3.2, GLM-5) builds in a simulation only, keeping the GPU
deployment's FlashMLA sparse backend for its caches (no kernel runs). The CPU build needs Python 3.11 or newer.

## Testing end to end

`tools/e2e` holds a local harness for simulated prefill/decode serving: it
builds a topology of Docker containers joined by separate control and KV
networks, serves a run behind [llm-d](https://llm-d.ai)'s router (Envoy, its
endpoint picker and its disaggregation sidecar, without Kubernetes), and
checks the result: every transferred byte, each rank's transfers and the
router's choices against what the engines ran. `tools/e2e/topologies/example.yaml`
and `tools/e2e/runs/example.yaml` document its options.

## Limitations

* On CUDA hosts each rank needs its own device (but little of its memory);
  the CPU platform has no such limit.
* Pipeline stages are assumed balanced.
* No LoRA, pooling or diffusion models, and speculative decoding without
  adaptive verification only.
* Simulated KV transfer is limited to CPU offloading by the
  `OffloadingConnector` and disaggregated prefill by the `NixlConnector`:
  in-process instances in pull mode with one rank each; served instances in
  pull mode without pipeline parallelism, or in push mode with it on the
  prefill instance. For real transfers, see
  [Real KV transfers](#real-kv-transfers).
* The `fast_forward` and `virtual` clocks need a single process with one
  rank; served and multi-rank simulations run on the real clock.
* Token ids are not meaningful; only their counts are.
