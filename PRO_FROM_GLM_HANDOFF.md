# Handoff: fix DeepSeek-V4-PRO on CUDA — what the GLM-5.2 work taught us

For a fresh-context agent. **Goal**: make **DeepSeek-V4-PRO** run *correctly* on CUDA (today it
is PARKED — loads/streams but emits garbage), then a 3-way benchmark **Flash vs GLM-5.2 vs PRO**.
GLM-5.2 is now DONE (runs resident on 2× H200, chats, native tokenizer); the techniques below
are what made it work and most transfer directly to PRO.

Box: `ssh root@192.168.1.15`, repo `/root/ds4`, CUDA 13.2, 2× H200 NVL (143 GB, no NVLink).
Build: `make ds4 CUDA_ARCH=native` (incremental; editing `ds4_cuda.cu` → nvcc recompile ~40 s).
Flash GGUF: `/root/ds4/ds4flash.gguf` (symlink → 81 GB IQ2XXS). Local repo:
`/Users/vincenzoingrosso/Documents/claude/ds4` (origin antirez/ds4, fork `vcnngr/ds4`).

## The PRO blocker (the one real bug)

**CUDA DSA indexer is hardcoded for `top_k = 512` (Flash).** PRO uses `top_k = 1024` → the
indexer truncates the candidate rows → wrong sparse attention → garbage. Same family of
hardcode I generalized elsewhere; this one is still open. Exact sites in `ds4_cuda.cu`:
- `__shared__ uint32_t comp_rows[512]` and `if (comp_count > 512u) comp_count = 512u` caps:
  ~lines 4855, 4892, 4897, 5011, 5052, 5226.
- `indexed_topk_sort_512_asc_kernel` (fixed 512-wide bitonic sort): ~7293–7310.
- top-k dispatch `if (top_k == 512u && n_comp <= {1024,2048,4096}u …)` with
  `cub::BlockRadixSort` variants: ~7522, 7529, 7536.

Generalize `top_k` (512 → also 1024, and 2048 for GLM's real sparsity): parametrize the
shared-mem `comp_rows[]`/caps, add sort/cub variants (or a generic path) for 1024/2048, and
relax the `== 512u` dispatch. **Do it once — it unblocks PRO (1024) and GLM real-sparsity
(2048).** (GLM currently dodges this with `compress_ratio 0` → dense, so GLM didn't need it yet;
PRO can't dodge it.) NB `head_dim == 512u` checks elsewhere are the MLA latent dim, *not* the
indexer top_k — don't touch those.

## What the GLM work already gives you (reuse, don't redo)

- **Router is already generalized** (`ds4_cuda.cu` serial kernel): `n_expert`, `n_used` (top-k),
  `scale`, and a `scoring_sigmoid` flag all parametrized; guards relaxed. PRO's expert
  count/top-k should flow through this. See PR **#466** (router, mine) and **#435** (draft, PRO
  router) — converge with them rather than re-fixing.
- **q8→f16 cache reserve fix = PR #472** (just opened from `vcnngr/ds4`): on ≥112 GiB cards the
  dequant cache starved the session graph → OOM after a successful load. PRO is large → it will
  hit this too. Merge/cherry-pick #472, or set `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` meanwhile.
- **2-GPU residency knobs** (252 GB GLM fit 2× H200): split by layers
  (`--role coordinator --layers 0:N` + `--role worker --layers N+1:output`), give the
  coordinator the smaller slice (it also holds output head + big logits), `WEIGHT_CACHE_LIMIT_GB`
  per node, reserve env above. **Both nodes MUST pass the same `-m`** (a coordinator without
  `-m` loads the default `ds4flash.gguf` → `model id mismatch`; this footgun cost a debugging
  session — see MAINTAINER_NOTES §4).

## Method that worked (apply to PRO)

1. **Variant-gate every change; verify Flash byte-identical after each.** GLM changes are gated
   on `DS4_N_HC == 1` / token presence so Flash/Pro paths are untouched. For PRO gate on
   `DS4_MODEL_VARIANT == DS4_VARIANT_PRO`. After every kernel edit, run the Flash regression:
   `./ds4 --cuda -m ds4flash.gguf -p "..." -n 24 --temp 0` (expect ~40 t/s gen, ~97 t/s prefill)
   and the logprob-vector check. A change that moves Flash numbers/logits is wrong.
2. **Instrument to localize, don't guess.** Temporary `getenv("DS4_DEBUG_*")`-gated `stderr`
   prints pinned the GLM bugs fast (e.g. printing metadata right after `model_open` exposed the
   wrong-model load in seconds). Use the same for PRO (dump indexer `top_k`, `comp_count`,
   per-layer row counts).
3. **Establish a trusted oracle, then `--dump-logits`.** GLM correctness was proven by dumping
   HF `transformers` reference logits on a *small real-arch* model and comparing DS4 via
   `--dump-logits` / `--raw-ids` (corr 0.9994). For PRO: the full model won't fit RAM for a CPU
   reference, so build a small real-arch DeepSeek-V4-Pro reference. **Lesson:** the DS4 CPU path
   has model-specific assumptions too (PRO failed CPU with `grouped Q8_0 tensor has an unexpected
   layout` at prefill layer 1) — budget time to get CPU correctness first as the oracle, or
   validate on `--cuda` once the indexer is fixed.

## PRO facts (from prior parked work / memory)

- PRO = DeepSeek-V4-Pro, 61 layers, indexer `top_k = 1024`, MLA latent head_dim 512.
- Runs on CUDA-streaming after the router patch, but the indexer 512-hardcode → garbage. That
  indexer generalization is THE remaining blocker.
- See memory `ds4-pro-cuda-flash-hardcodes` and `MAINTAINER_NOTES.md` §2.

## Disk before downloading PRO

`/home` 474 G free (holds two 253 G GLM GGUFs), `/` 833 G free (holds `/root/glm52-hf` = 1.4 TB
safetensors). The GLM **GGUF is self-contained** — the 1.4 TB HF safetensors are only needed to
re-convert. Free space for the PRO download by deleting: `glm52_ds4_q2k.gguf` (253 G, placeholder
tokenizer, **superseded** by `glm52_ds4_q2k_tok.gguf`) and, if no GLM re-quant is planned,
`/root/glm52-hf/*.safetensors` (1.4 TB; keep `config.json`/`tokenizer.json`). That frees ~1.65 TB.

## Reference docs (in the repo)
`MAINTAINER_NOTES.md` (clean writeup), `GLM52_BACKEND_PLAN.md` (full log P0–P5),
`GLM52_MAINTAINER_REPORT.md` (issue-ready report). Converter/run scripts: `glm5_to_ds4_stream.py`,
`glm_tok_rewrite.py`, `run_glm_res2.sh`, `run_glm_chat_native.sh`. PRs: #466 router, #472 reserve,
#435 draft PRO router.

## Benchmark (after PRO is correct)
**Two distinct benchmarks — keep them separate:**
1. **Intra-DS4** (the immediate goal): Flash vs GLM vs PRO on DS4 — prefill + generation t/s,
   VRAM residency / GPUs used, quality probe (logit corr vs HF ref, or a small task set). Match
   quant where feasible; note machine/backend/quant per CONTRIBUTING. Baselines so far: Flash
   prefill 97 / gen 40.8 t/s (single H200, IQ2XXS); GLM-5.2 q2 prefill ~34–39 / gen ~16 t/s
   (2× H200 resident).
2. **DS4 vs market (DEFERRED — user said "lo vediamo dopo")**: DS4-GLM vs **vLLM**-GLM vs
   **llama.cpp**-GLM (all-GPU, not Unsloth CPU-offload) on the same H200 box, same ~2-bit quant.
   This is the only thing that proves DS4 is *competitive* rather than just a working engine
   contribution. Unsloth's edge is cheap hardware (24 GB GPU + 256 GB RAM via MoE offload) — a
   different regime; DS4's regime is full GPU residency, where the real rivals are vLLM /
   llama.cpp-all-GPU. Possible DS4 levers to test: MTP speculative decode, distributed pipeline
   split, absorbed-MLA. Do this AFTER PRO + intra-DS4 bench.
