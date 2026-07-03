# GLM-5.2 (GlmMoeDsa, ~744B) runs on DS4 across 2× H200 — report + 2 upstreamable fixes

Ran **GLM-5.2** (`GlmMoeDsaForCausalLM`, Zhipu, MIT) on the DS4 engine. It loads, computes
correct logits, generates, runs fully resident across **2× H200 NVL** (no SSD streaming), and
chats with real text via a native GLM tokenizer in the GGUF.

Posting as a report (not a PR) because the engine diff is large and experimental. Two findings
inside it are **model-independent** and worth upstreaming on their own — details at the bottom.

## What works

- **Architecture**: GLM-5.2 is DeepSeek-V3.2-family (MLA + DSA indexer + MoE(+shared) + MTP),
  the path DS4 already implements. Deltas: no mHC (`n_hc=1`), no V4 KV compressor (dense), GLM
  stores explicit `kv_b_proj` + full `o_proj` vs DS4's absorbed-MLA, sigmoid routing, top-8,
  rope θ=1e6, and absorbed `head_dim = kv_lora+rope = 576`.
- **Correctness**: a standalone converter maps GlmMoeDsa safetensors → DS4-layout GGUF
  (absorbed-MLA, proven equal to explicit GLM attention to 2e-08). On a small real-arch model
  DS4 reproduces HF `transformers` GlmMoeDsa logits to **corr 0.9994 / max|Δ| 0.035** (quant
  noise), validated with `--dump-logits`.
- **Real 744B**: q2 GGUF (~252 GB) runs on 2× H200, **prefill ~34 t/s, generation ~16 t/s**,
  coherent output. `-p "What is the capital of France?"` → "The capital of France is Paris."
- **Native tokenizer**: real GLM tokenizer (154880 tokens / 321649 merges) in the GGUF; DS4's
  `joyai-llm` byte-level BPE reproduces GLM's tokenization **exactly** (`--dump-tokens` ==
  HF `AutoTokenizer` on 6/6 diverse strings).

All engine changes are **gated on the GLM variant (`n_hc==1`) / on token presence**, so the
DeepSeek **Flash/Pro paths are byte-identical** — verified: Flash 40 t/s + `--logprob-vectors`
OK on the same build. Engine changes (summary): mHC `n_hc==1` identity bypass; MLA q/kv-norm
GLM gates + absorbed dims (576/512/n_head/v_head); router `n_used` (top-8) + sigmoid scoring
(extends the router work, see below); MoE Q2_K kernels; GLM tokenizer special-token resolution
+ chat template + multi-token end-of-turn stop; Flash-sized fixed maxes bumped.

## Two model-independent fixes worth upstreaming

**1. `q8→fp16` dequant-cache reserve starves the session graph on ≥112 GiB cards.**
`cuda_q8_f16_cache_reserve_bytes()` returns a flat **512 MiB** reserve when total VRAM ≥ 112 GiB
(the 5%/4 GiB rule only applies below that). On H200 the optional cache filled HBM and left
0.5 GiB, so a large-model prefill graph OOM'd at session create *after* a successful distributed
handshake. `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` does not bound this cache. This is the **CUDA twin of
PR #446** (same bug on ROCm, q4q2). Fix submitted as **PR #472** (drop the ≥112 GiB special
case; use the same 5% / 4 GiB-min reserve). Evidence: before → OOM (cache 14 GiB, free 0.5 GiB);
after → free 7 GiB, runs 16 t/s; Flash non-regression 40.8 t/s. (Workaround:
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB`.)

**2. Distributed `model id mismatch` is easy to hit and the error doesn't say why.**
A coordinator started **without `-m`** silently loads the default `ds4flash.gguf`, then rejects
the workers with `model id mismatch: worker=2 coordinator=0`. The detection is correct; the
message just doesn't hint at the cause. One-line message improvement (mention same-`-m` / the
`ds4flash.gguf` default) saves a long debugging session. Tiny PR-able.

## Router note

The CUDA MoE router (256/top-6/scale-1.5 hardcodes) already has my PR **#466**; GLM needed it
plus `n_used` (top-8) and sigmoid scoring. #435 (draft) generalizes the same router for Pro.
Worth converging these so one router serves Flash/Pro/GLM.

Machine: Linux, 2× NVIDIA H200 NVL (143 GB, no NVLink), CUDA 13.2. Converter/reference scripts
are standalone Python (not in the tree); happy to share.
