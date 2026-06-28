# DS4 — CUDA router fix, PRO findings, and a working GLM-5.2 (GlmMoeDsa) backend

External experiment on Linux + 2× NVIDIA H200 NVL (140 GB each, no NVLink), CUDA 13.2.
Four separable pieces, decreasing "ready to merge":

1. **CUDA MoE-router fix** — unblocks non-256-expert models. **Already a PR: #466.**
2. **DeepSeek-V4-Pro bug findings** (CUDA indexer + CPU Q8_0). Report-only.
3. **A GLM-5.2 (GlmMoeDsa, ~744B/40B-active) backend** that LOADS, computes CORRECT
   logits (corr 0.9994 vs HF transformers), GENERATES, and RUNS across 2 H200.
4. **A distributed `model id mismatch`** found while running GLM on 2 GPUs — root-caused to a
   launch footgun (coordinator started without `-m` → loads the default `ds4flash.gguf`), not
   an engine bug. Plus the memory-budget tuning for full 2-GPU residency.

Engine changes are confined to `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`, `ds4_cli.c`,
`ds4_distributed.c`. Converter/reference scripts are standalone Python (not in the tree).

---

## 1. CUDA MoE router hardcodes 256 experts / top-6 / scale 1.5  → PR #466

`ds4_gpu_router_select_tensor` / `_batch_tensor` reject any model whose routed-expert
count isn't 256, and the serial/warp/parallel kernels hardcode 256 + top-6 + 1.5 scale.
Fix (in #466): parametrize the serial kernel with `n_expert`/`scale`, relax the guards to
accept 256|384, size checks by `n_expert`; 256/top-6 stays on the fast warp path. Verified
zero-regression on Flash (logprob-vectors OK, 40 t/s).

GLM extends this further (in the GLM work below, not in #466): the serial kernel also gets a
`n_used` param (top-8) and a `scoring_sigmoid` flag (GLM uses sigmoid; DeepSeek uses
sqrt-softplus) threaded through both wrappers.

## 2. DeepSeek-V4-Pro on CUDA / CPU — bug findings (no fix)

- **CUDA DSA-indexer hardcoded for top_k=512** (Flash). Dispatch chain `if (top_k==512u…)`,
  `__shared__ uint32_t comp_rows[512]`, `comp_count>512u` truncation, `sort_512`,
  `cub::BlockRadixSort<uint64_t,512,16>`. Pro (top_k=1024) → truncated rows → garbage.
- **CPU backend rejects Pro's Q8_0 attention layout**: `make cpu` + Pro fails at
  `prefill layer 1/61` with `grouped Q8_0 tensor has an unexpected layout`.

## 3. GLM-5.2 (GlmMoeDsa) backend — works end-to-end on the real 744B

GLM-5.2 is DeepSeek-V3.2-family: MLA + DSA indexer + MoE(+shared) + MTP. Deltas from DS4's
DeepSeek path: **no mHC** (n_hc=1), **no V4 KV compressor** (dense, compress_ratio 0),
GLM stores explicit kv_b_proj + full o_proj vs DS4's absorbed-MLA, sigmoid routing, top-8,
rope_theta 8e6, untied output, intermediate_size≠moe_inter on dense layers, and the absorbed
head_dim = kv_lora+rope = **576** (not 512).

**Validated correctness** (the key result): a standalone converter maps GlmMoeDsa
safetensors → DS4-layout GGUF (absorbed-MLA: `attn_q_b[h]=W_k[h]^T·q_b_nope[h]` scaled,
`attn_kv=kv_a_proj`, `attn_output_a[h]=[W_v[h]|0]`, `output_b=o_proj`; proven equal to the
explicit GLM attention to 2e-08). With a GLM-5.2 shape variant + variant-gated engine
changes, DS4 reproduces the HF transformers GlmMoeDsa logits to **corr 0.9994 / max|Δ| 0.035**
(quant noise) on a small real-arch model, validated via `--dump-logits` vs a transformers
reference (numpy DS4-style full forward also matches HF to 7e-06).

Engine changes — all gated on the GLM variant (n_hc==1); the DeepSeek Flash/Pro path is
byte-identical (verified: Flash 40 t/s + logprob-vectors OK on the same build):
- **mHC n_hc==1 identity bypass** (the hc split/sinkhorn/weighted_sum primitives are
  hardcoded for n_hc==4): treat the single residual stream as the identity hyper-connection
  in attention/FFN encode (prefill + decode) and the 3 output heads.
- **MLA**: skip the per-head q RMSNorm GLM lacks; `glm_kv_norm` kernel (RMS-norm the
  kv_lora part only, pass the k_rope tail); absorbed-MLA dims (head_dim 576, n_value 512,
  n_out_group=n_head, n_lora_o=v_head); attn sinks → −inf; shexp layout check relaxed
  (dense layers carry intermediate_size in the shared-expert slot).
- **Router**: `n_used` (top-8) + `scoring_sigmoid` in the serial kernel (extends #466).
- **MoE Q2_K path**: added a `(gate=Q2_K,down=Q2_K)` dispatch + two kernels
  (`moe_gate_up_mid_q2K_qwarp32`, `moe_down_q2K_qwarp32`, reusing the existing
  `dev_dot_q2_K_q8_K_block`) so a q2 GLM fits 2×H200 (DS4 previously only had
  Q4_K and IQ2_XXS+Q2_K expert combos).
- **Fixed-size maxes were Flash-sized** and overflowed on GLM (heap corruption →
  double-free at model_close): `DS4_MAX_LAYER 61→96, MAX_VOCAB→160000, MAX_HEAD_DIM→640,
  MAX_OUT_GROUP→64, MAX_LORA_Q→2048, MAX_EXPERT_USED→8, MAX_FF_EXP→12288`. (Recurring
  theme: the CUDA/loader paths bake in Flash constants.)
- `--raw-ids` CLI flag (debug: inject token ids, bypass the tokenizer) for logit-level
  validation without a native GLM tokenizer.

**Real model**: zai-org/GLM-5.2 (1.5 TB) converted (streaming, vectorized Q2_K/Q4_K) to a
DS4 GGUF (q2 ≈ 252 GB / 775 B params). `--inspect` loads it; the forward produces sane,
peaked logits (entropy 5.26 vs uniform 11.95); it generates via decode. Ran 1-GPU
(SSD-streaming, slow ~0.01 t/s) and 2-GPU distributed (streaming). Real prompt → GLM
tokenizer (transformers) → `--raw-ids` → real generated token, decoded back to text.

**Native GLM tokenizer — now working** (real text in/out, no python bridge). Three parts:
- *GGUF*: a header-rewrite tool swaps `tokenizer.ggml.tokens`/`.merges` for the real GLM
  tokenizer (154880 tokens, 321649 merges from `tokenizer.json`) and copies the 252 GB tensor
  data verbatim (tensor offsets are relative to the data section, so they're unchanged) —
  ~minutes vs a ~1.5 h reconvert.
- *Engine* (`ds4.c`, variant-agnostic, gated by token presence so Flash/Pro are byte-identical):
  `vocab_load` detects GLM (`[gMASK]`+`<|assistant|>` present) and resolves the GLM special set
  (`[gMASK]`,`<sop>`,`<|system|>`,`<|user|>`,`<|assistant|>`,`<think>`/`</think>`,`<|endoftext|>`
  as eos) instead of hard-failing on the missing DeepSeek tokens; `encode_chat_prompt` renders
  the GLM template `[gMASK]<sop>[<|system|>…]<|user|>…<|assistant|><think>`. Added a non-fatal
  `vocab_lookup_opt`. Generation stop: DS4 broke only on a single eos; GLM ends an assistant
  turn with any of `<|endoftext|>`/`<|user|>`/`<|observation|>` (its `generation_config`
  eos set), so a `ds4_token_is_eot()` helper now stops on those for GLM (single-eos for
  DeepSeek unchanged) — otherwise the model runs on and hallucinates a fake user turn.
- *Pre-tokenizer*: **the open risk resolved itself** — DS4's `joyai-llm` byte-level BPE
  reproduces GLM's tokenization EXACTLY. `--dump-tokens` vs HF `AutoTokenizer` matched on 6/6
  diverse strings (punctuation, digits, code, Italian, multiple-spaces/tabs, accented unicode),
  e.g. `"The capital of France is"` → `[785,6722,315,9621,374]` on both. So `ds4 -p "text"`
  works natively on the GLM GGUF. (ds4-server supports the distributed coordinator role, so this
  also enables a persistent interactive 2-GPU chat endpoint instead of the per-turn CLI reload.)

## 4. Distributed `model id mismatch` — root-caused to a launch footgun, NOT an engine bug

Running the q2 GLM on 2 H200 fully resident failed the handshake with
`model id mismatch: worker=2 coordinator=0`. This was **initially mis-diagnosed as a
`g_ds4_shape` global-state reversion**. It is not. The engine's mismatch detection was
correct — the two processes really had loaded two different models.

Root cause (confirmed by instrumenting `config_validate_model` and `model_open`): the launch
script passed `-m <glm>` to the **worker** but **omitted `-m` on the coordinator**. With no
`-m`, the CLI default model path is `"ds4flash.gguf"` (`ds4_cli.c:1409`), so the coordinator
loaded the real DeepSeek-V4 **Flash** GGUF sitting in its CWD — variant 0, `block_count=43`,
`n_kv=62` — read straight out of `model_open` (before `vocab_load`/`config_validate`),
*not* a global that flipped later. The worker loaded the GLM file (variant 2, `block_count=78`,
`n_kv=36`). Both `ds4_engine_model_id()` calls returned exactly what each process had loaded;
the handshake rejection was correct behavior. (The "standalone coordinator sometimes prints
variant=2" observation was simply runs where `ds4flash.gguf` wasn't on the path / `-m` was
present.)

**Fix**: pass the same `-m <model>` to coordinator AND worker. After that the handshake
matches (`model_id=2` both sides), the route completes, and the pipeline runs. No engine
change is required.

Two follow-ups landed from this:
- **Better error message** (`ds4_distributed.c`): the `model id mismatch` text now hints that
  both nodes must load the same `-m` model and that a coordinator started without `-m` falls
  back to the default `ds4flash.gguf`. (One-line UX change; the detection logic is unchanged.)
- **Memory budget for full residency** — and a real finding: the q8→fp16 dequant cache
  reserve is too small on big cards. The q2 GLM is ~252 GB ≈ 3.2 GB/layer, so a ~half split
  puts ~122 GiB of weights on each 143 GB H200; the coordinator additionally holds the output
  head + embeddings + the 154880-vocab logit/prefill graph. The optional q8→fp16 dequant cache
  greedily fills HBM down to a fixed reserve, and `cuda_q8_f16_cache_reserve_bytes()`
  (`ds4_cuda.cu:512`) returns **only 512 MiB for total VRAM ≥ 112 GiB** (the 5%/4 GiB rule
  applies only below that). On the H200 the cache grew to ~14 GiB, left 0.5 GiB, and the GLM
  prefill graph OOM'd at execution (after the handshake/route/session-context all succeeded,
  `--raw-ids: 17 tokens injected`). `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` does NOT bound this cache
  (separate pool). Workaround per node: `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=20480` (coordinator)
  / `8192` (worker) to reserve real graph headroom — or `DS4_CUDA_Q8_F16_CACHE_MB=<n>` to cap
  the cache, or `DS4_CUDA_NO_Q8_F16_CACHE=1` to disable it (weights stay fully resident either
  way; attention just falls back to q8 kernels). **Suggested engine fix**: scale the ≥112 GiB
  reserve with model/graph size (or %) instead of a flat 512 MiB. Split used:
  `coordinator --layers 0:36`, `worker --layers 37:output`.

---

## Repro artifacts (on the test box, not in the tree)
- `glm5_to_ds4_stream.py` — GlmMoeDsa → DS4 GGUF (absorbed-MLA, vectorized Q2_K/Q4_K, lazy
  sharded load + 2-pass serialize for the 1.5 TB input).
- `glm_ref.py` / `glm_ds4_forward.py` / `glm_absorb_test.py` — transformers reference +
  numpy DS4-style forward (proves the recipe: 2e-08 attention, 7e-06 full forward).
- A small GLM-5.2 model + `glm_ref_logits.npy` for `--dump-logits` comparison.

Channel: GitHub. PR #466 is the only thing submitted; items 2–4 are reports/discussion.
