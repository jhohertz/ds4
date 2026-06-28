# Serve GLM-5.2 (DS4) on the Arma Gateway (api.cdogen.cc) — integration spec

## How the gateway works (verified on 192.168.1.15)

OpenAI-compatible on-demand model router. Controller in
`/home/vcnngr/openrouter/execution/gateway/` (`router.py`, `model_manager.py`, `config.py`),
API on `:18000`, fronted by nginx (api.cdogen.cc → :18000). Models registered in
`config/models.yaml`. Each request spins up the model backend on its own port (pool
`on_demand`, `idle_timeout_min`), then proxies. Engine dispatch in `model_manager.py`:
`engine == "vllm" | "diffusers" | "tts"` → `_build_*_cmd`; launched via `_spawn_process`
(local subprocess) **or** `_submit_slurm_job` (SLURM). Every model today is `engine: vllm`
(`python -m vllm.entrypoints.openai.api_server --model … --served-model-name … --port …
--tensor-parallel-size …`).

## Why DS4 is the right backend for GLM-5.2 *here*

GLM-5.2 is 744B. On this **2× H200** box (286 GB HBM) a vLLM all-GPU serve doesn't fit at
vLLM's usual quants (4-bit AWQ ≈ 372 GB > 286; vLLM has no good ~2-bit path for `GlmMoeDsa`).
DS4's **q2 (~252 GB) fits fully resident across the 2 GPUs** (pipeline split, no CPU offload),
~16 t/s. So for *this* hardware, DS4 is the only full-residency option for GLM-5.2 — that's the
concrete reason to add a `ds4` engine, not just parity with vLLM.

## What's needed — gateway side

1. **New engine `ds4` in `model_manager.py`**: add `elif cfg.engine == "ds4": cmd =
   self._build_ds4_cmd(cfg)`. DS4 distributed serving is **2 processes on 2 GPUs**
   (coordinator + worker), unlike vLLM's single process — so `_build_ds4_cmd` should launch a
   **wrapper script** that starts the worker, waits, then starts the coordinator-as-server
   (model `model_manager` already supports a single `cmd`; the wrapper hides the 2-process
   detail). Pattern is the validated `run_glm_chat_native.sh` but with the coordinator running
   `ds4-server` instead of the one-shot CLI.
2. **Config fields** (new keys consumed by `_build_ds4_cmd`): `gguf_path`,
   `coordinator_layers` (`0:36`), `worker_layers` (`37:output`), `coordinator_gpu` (0),
   `worker_gpu` (1), `q8_f16_reserve_mb` (20480 coord / 8192 worker), `weight_cache_gb`.
3. **Lifecycle**: cold start is **~3–4 min** (252 GB load). `on_demand` + 4-min spin-up is
   rough → give GLM a long `idle_timeout_min` (e.g. 120) or a `pool: pinned`/always-on entry.
4. **GPU exclusivity**: GLM occupies **both** H200s fully → no other gateway model can run while
   GLM is up. Enforce via SLURM (reserve both GPUs for the GLM job) or a gateway mutex so it
   doesn't co-schedule with `qwen3-235b` (also TP=2) etc.
5. **Health check / readiness**: gateway must wait for `distributed route ready` before marking
   the model live (not just port open) — the coordinator serves only after the worker registers.

## What's needed — DS4 side

1. **`ds4-server` distributed coordinator + OpenAI `/v1` — ✅ VERIFIED WORKING (2026-06-28).**
   `ds4-server --role coordinator --layers 0:36 --listen 0.0.0.0 28100 --host 0.0.0.0 --port
   8030 -m glm52_ds4_q2k_tok.gguf -c 2048` + a worker (`--role worker --layers 37:output
   --coordinator 127.0.0.1 28100`) → server logs `listening on http://0.0.0.0:8030`,
   `session_create model_id=2 layers=78`, and `POST /v1/chat/completions` returns a valid
   OpenAI JSON (`model: glm-5.2`) generated on the 2× H200 resident pipeline. So the gateway
   adapter has a working backend to launch — no engine change needed for the transport.
   **One required server-side fix for CLEAN output** (the answer is correct but messy): the
   server renders the prompt with its own `render_chat_prompt_text` (`ds4_server.c:2312`,
   DeepSeek markers) → `ds4_tokenize_rendered_chat`, **not** the GLM-aware `encode_chat_prompt`.
   The DeepSeek marker strings map to GLM ids via `special_token_at`, but the **`[gMASK]<sop>`
   prefix is missing** and the `<think>…</think>` reasoning is returned raw (not separated). Fix:
   add an `is_glm` branch to `render_chat_prompt_text` emitting the GLM template
   (`[gMASK]<sop>[<|system|>…]<|user|>…<|assistant|><think>`) — extend `special_token_at` to
   also recognize the GLM marker strings (`[gMASK]`,`<sop>`,`<|system|>`,`<|user|>`,
   `<|assistant|>`) — and strip/separate the `<think>` block in the response (reasoning).
   Bounded server-side change in `ds4_server.c`.
2. **GGUF**: `/home/glm52_ds4_q2k_tok.gguf` (native GLM tokenizer, already built).
3. **Build**: DS4 with GLM support — `vcnngr/ds4 @ glm-5.2-backend`, `make ds4-server
   CUDA_ARCH=native` (verify the server target builds with the GLM changes).
4. **Chat template / stop**: the GLM template + multi-eos stop are wired in the engine
   (`encode_chat_prompt` / `ds4_token_is_eot`), so `/v1/chat/completions` messages should render
   correctly — verify the server uses `encode_chat_prompt` for the chat route.

## Example `models.yaml` entry (proposed)

```yaml
  glm-5.2-ds4:
    engine: ds4
    model_type: chat
    gguf_path: /home/glm52_ds4_q2k_tok.gguf
    served_base_name: glm-5.2
    port: 8030
    vram_gb: 286            # both H200s
    coordinator_layers: "0:36"
    worker_layers: "37:output"
    coordinator_gpu: 0
    worker_gpu: 1
    q8_f16_reserve_mb_coord: 20480
    q8_f16_reserve_mb_worker: 8192
    weight_cache_gb: 134
    pool: pinned            # or on_demand with idle_timeout_min: 120
    idle_timeout_min: 120
    description: GLM-5.2 (744B, q2) on DS4 — full-residency across 2x H200
```

## Open items / risks
- ds4-server distributed-coordinator API path: **verify / possibly small fix** (above).
- ~4-min cold start + dual-GPU exclusivity → schedule as pinned or SLURM-reserved, not casual
  on_demand alongside other models.
- This serves DS4-GLM; a benchmark **DS4 vs vLLM/llama.cpp** on the same box would say whether to
  keep DS4 here or just use vLLM where it fits (deferred — see PRO_FROM_GLM_HANDOFF.md).
