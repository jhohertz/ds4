# Add a `ds4` engine to the Arma Gateway (serve GLM-5.2 on 2× H200)

Apply to `/home/vcnngr/openrouter`. Verified backend: `ds4-server` distributed coordinator
already serves OpenAI `/v1` (2× H200 resident). This wires it into the on-demand router and
**solves the ~4-min cold start so requests don't time out**.

## 1. Engine dispatch — `execution/gateway/model_manager.py`

In `_start_worker` where the engine is branched, add:

```python
elif state.config.engine == "ds4":
    cmd = self._build_ds4_cmd(state.config)
```

Add the builder (mirrors `_build_vllm_cmd`):

```python
def _build_ds4_cmd(self, cfg: ModelConfig) -> list[str]:
    here = os.path.dirname(__file__)
    wrapper = os.path.join(here, "ds4_serve.sh")   # ship ds4_serve.sh alongside
    return [
        "bash", wrapper,
        cfg.ds4_bin or "/root/ds4/ds4-server",
        cfg.gguf_path,
        cfg.coordinator_layers,            # "0:36"
        cfg.worker_layers,                 # "37:output"
        cfg.host or "0.0.0.0",
        str(cfg.port),                     # OpenAI /v1 port (e.g. 8030)
        str(cfg.coord_listen_port or 28100),
        str(cfg.max_model_len or 2048),
        str(cfg.coordinator_gpu),          # 0
        str(cfg.worker_gpu),               # 1
        str(cfg.q8_f16_reserve_mb_coord or 20480),
        str(cfg.q8_f16_reserve_mb_worker or 8192),
        str(cfg.weight_cache_gb or 134),
    ]
```

Add the new fields to `ModelConfig` (in `config.py`): `gguf_path, ds4_bin, coordinator_layers,
worker_layers, coord_listen_port, coordinator_gpu, worker_gpu, q8_f16_reserve_mb_coord,
q8_f16_reserve_mb_worker, weight_cache_gb`.

## 2. Readiness probe — wait for "route ready", not just an open port

The DS4 server opens its port but only answers after the worker registers
(`distributed route ready`). The gateway's readiness check should poll the OpenAI surface:

```python
# in the health/readiness probe used by _wait_until_ready / _check_ready
GET http://127.0.0.1:{cfg.port}/v1/models   ->   ready when HTTP 200
```

(If `/v1/models` isn't implemented, probe a 1-token `/v1/chat/completions`; ready when it
returns 200 instead of connection-refused.)

## 3. Cold start (~4 min) — `on_demand` + SLURM queue (NOT always_on)

`always_on` is wrong here: GLM pins both H200s permanently and the other models could never
load. Use **`pool: on_demand`** and let **SLURM manage the queue and GPU allocation**. The
gateway already supports this — `_start_worker` calls `_submit_slurm_job(alias, cmd)` instead of
`_spawn_process` when SLURM mode is on (model_manager.py:415). So GLM is submitted as a SLURM
job requesting both GPUs (`--gres=gpu:2`); SLURM queues it until 2 GPUs are free (i.e. until the
other big models' jobs finish), runs the load (~4 min), serves, and on `idle_timeout` the
gateway cancels the job → GPUs return to SLURM for everyone else.

**Don't block a request for queue+load.** Queue time is unbounded (could be minutes), so a fixed
`ensure_model_ready` timeout can't cover it. Use the non-blocking pattern the router already
supports: trigger the load, keep `status = "loading"`, and **return 503 with `Retry-After`
immediately** (short `ensure_model_ready` timeout → TimeoutError → 503; the SLURM job keeps
loading in the background). Clients/gateway **retry**; each retry either gets a ready model or
another 503-loading. Never hold a single request for the whole queue+load. Give GLM a moderate
`idle_timeout_min` (e.g. 30–60) so a burst of traffic keeps it warm but it frees the GPUs when
truly idle.

Net: request → (SLURM queues GLM, 503-retry while queued+loading) → GLM scheduled → loads → ready
→ retries succeed → idle → SLURM frees both GPUs back to the pool.

## 4. SLURM job / GPU binding (wrapper)

GLM runs INSIDE a SLURM allocation of 2 GPUs (`--gres=gpu:2`), so it must NOT hardcode
`CUDA_VISIBLE_DEVICES=0/1` — the two GPUs come from SLURM. `ds4_serve.sh` should bind the worker
to the first allocated GPU and the coordinator to the second, e.g. derive them from
`$CUDA_VISIBLE_DEVICES` (SLURM sets it to the allocated device list) and split: worker →
`${GPUS[0]}`, coordinator → `${GPUS[1]}`. Pass `coordinator_gpu`/`worker_gpu` only for the
non-SLURM (`_spawn_process`) path. SLURM also enforces the exclusivity (GLM's `--gres=gpu:2`
won't co-schedule with `qwen3-235b`/`crimezero` TP=2 once the GPUs are taken).

## 5. `models.yaml` entry

```yaml
  glm-5.2-ds4:
    engine: ds4
    model_type: chat
    ds4_bin: /root/ds4/ds4-server
    gguf_path: /home/glm52_ds4_q2k_tok.gguf
    served_base_name: glm-5.2
    host: 0.0.0.0
    port: 8030
    coord_listen_port: 28100
    max_model_len: 2048
    coordinator_layers: "0:36"
    worker_layers: "37:output"
    coordinator_gpu: 0
    worker_gpu: 1
    q8_f16_reserve_mb_coord: 20480
    q8_f16_reserve_mb_worker: 8192
    weight_cache_gb: 134
    vram_gb: 286
    pool: on_demand          # NOT always_on (would pin both GPUs forever); SLURM queues it
    idle_timeout_min: 45     # keep warm across a burst, then free both GPUs back to SLURM
    slurm_gres: "gpu:2"      # SLURM allocates + queues; don't co-schedule with other TP=2 models
    description: GLM-5.2 (744B, q2) on DS4 — full-residency across 2x H200 (on_demand, SLURM-queued)
```

## Status of the DS4 side
- `ds4-server` distributed `/v1`: ✅ verified serving.
- GLM chat template in the server (`[gMASK]<sop>…<|system|>…`): ✅ added (`render_chat_prompt_text`
  `is_glm` branch + `special_token_at` GLM markers + `ds4_engine_is_glm`). Build:
  `make ds4-server CUDA_ARCH=native` from `vcnngr/ds4 @ glm-5.2-backend`.
- Reasoning is already split into `reasoning`/`content` by the server (give enough `max_tokens`
  so the model closes `</think>` — think mode needs headroom).
