# DS4 → GLM-5.2 Backend Plan

Goal: run **GLM-5.2** (`GlmMoeDsaForCausalLM`, Zhipu, ~744B/40B-active MoE, MIT) on
the DS4 (DwarfStar) engine, targeting **2x H200** (hpc-atlas01).

## Why this is feasible (key finding)

GLM-5.2 is **architecturally a DeepSeek-V4 clone** — the same family DS4 already
implements:

| Component        | DS4 (DeepSeek V4)              | GLM-5.2 (GlmMoeDsa)                 |
|------------------|-------------------------------|-------------------------------------|
| Attention        | MLA (latent KV)               | **MLA** kv_lora=512, q_lora=2048    |
| Decoupled RoPE   | rope/nope split               | qk_rope=64, qk_nope=192             |
| Sparse attention | DSA indexer                   | **DSA indexer** top_k=2048, hd=128  |
| MoE              | 256/384 exp, top-6, +1 shared | 256 exp, **top-8**, +1 shared       |
| Routed scaling   | 1.5                           | **2.5**                             |
| MTP              | yes                           | yes (1 nextn layer)                 |
| Layers           | 43 / 61                       | 78 (first 3 dense)                  |
| Hidden / vocab   | —                             | 6144 / 154880                       |

So this is a **re-parametrization of DS4's existing DeepSeek path**, NOT an
attention rewrite. The hard kernels (MLA, DSA indexer, MoE, MTP) already exist.

Sizing: GLM-5.2 q2 ≈ **~214 GB** → fits 2x H200 (281 GB HBM) **fully resident**
via 2-GPU pipeline split (~107 GB/GPU). **No SSD streaming** → avoids the PRO
streaming bugs entirely. This is a cleaner target than DS4-PRO.

## Reuse vs change

REUSE as-is: GGUF mmap loader, Metal/CUDA runtime + graph scheduler, MLA attention
kernels (re-dim'd), MoE expert compute (6 selected experts → 8), MTP draft head,
session/KV reuse, disk KV, server (OpenAI/Anthropic API), agent, CLI, sampling,
bench/eval harness, distributed pipeline split.

CHANGE (the actual work):
1. **Shape struct** (`ds4.c` `g_ds4_shape`, ~line 154-316): add a GLM-5.2 variant —
   hidden 6144, 78 layers, first_k_dense 3, vocab 154880, heads 64, kv_lora 512,
   q_lora 2048, qk_rope 64, qk_nope 192, v_head 256, n_expert 256, n_expert_used **8**,
   n_shared 1, expert_weight_scale **2.5**, moe_intermediate 2048, indexer top_k 2048,
   indexer head_dim 128, indexer n_heads 32.
2. **n_expert_used 6 → 8**: the CUDA router kernels hardcode `for j<6` topk
   (ds4_cuda.cu router_select_*). Same class as the PRO router patch already applied
   (parametrized n_expert + scale). Extend to parametrize `n_expert_used` (6→8) and
   size the selected/weights stride accordingly. `routed_scaling_factor` already
   flows as the `scale` param from the PRO patch.
3. **DSA indexer top_k 512 → 2048**: ds4_cuda.cu indexer topk dispatch is hardcoded
   `if (top_k == 512u ...)` with comp_rows[512] caps and sort_512/cub<512> kernels.
   Generalize to 2048 (shared mem, sort/cub variants, caps). **This is the same
   indexer work deferred for DS4-PRO** — do it once, serves both. See
   memory ds4-pro-cuda-flash-hardcodes.
4. **IndexShare**: GLM reuses the indexer top-k across layers (indexer_types
   full/shared, ~every 4). Map to DS4's per-layer indexer scheduling; "shared"
   layers skip the indexer score+topk and reuse the previous "full" layer's rows.
5. **MLA dims**: re-dim attention kernels for qk_nope=192, v_head=256 (DS4 Flash
   used head_dim=512 latent; verify GLM's exact latent layout from the GGUF).
6. **Tokenizer**: GLM vocab 154880 — load GLM tokenizer tokens/merges from GGUF.
7. **Prompt template**: GLM-5.2 chat template (roles, think/tool rendering) — new
   renderer; mirror the DeepSeek one in ds4_server/ds4_cli.
8. **GGUF metadata**: DS4 reads `deepseek4.*` keys. Either (a) emit GLM GGUFs with
   `deepseek4.*` keys remapped (simplest — DS4 stays single-arch), or (b) add a
   `glm5.*` key set + arch switch. Recommend (a) for the first bring-up.

## Hard prerequisite: the GGUF

DS4 only loads its own GGUFs. There is no GLM-5.2 GGUF in DS4 layout. Must generate
one with `gguf-tools/`: convert zai-org/GLM-5 → DS4-layout GGUF with the right
tensor names, the asymmetric 2-bit quant (IQ2 routed experts + Q8 attention/shared),
and an imatrix. This is a sub-project parallel to the engine changes.

## Validation strategy

Without official GLM logits locally, use **DS4's own method**: dump GLM-5.2
reference logits from the HF `transformers` GlmMoeDsa model (CPU/GPU, short prompt),
then compare DS4 engine logits via `--dump-logits` at several context sizes.
Lesson from PRO: the CPU reference path also has model-specific assumptions
(grouped Q8_0) — budget time to make CPU correctness work first as the oracle.

## Phased work

- **P0 Scaffold**: GLM-5.2 shape variant + config loader from a real GLM GGUF header
  (start with a tiny/random GLM-5 GGUF to exercise loading). `--inspect` must print
  correct dims.
- **P1 GGUF**: gguf-tools conversion of a small slice (few layers) → validate load.
- **P2 Kernels**: n_expert_used→8, indexer top_k→2048, IndexShare, MLA dims. Build
  CUDA, get past prefill layer 0 without garbage.
- **P3 Correctness**: logits vs HF transformers reference; iterate until match.
- **P4 Tokenizer + template**: end-to-end chat.
- **P5 Full model**: 2-GPU pipeline residency, bench, agent.

## Status
Doc created 2026-06-26. Engine already built on hpc-atlas01 (CUDA, Flash validated).
Router patch (n_expert/scale parametrized) already landed — directly reusable for
GLM's top-8/scale-2.5.

**P0 DONE (2026-06-26)**: GLM-5.2 shape scaffold landed in ds4.c — `DS4_VARIANT_GLM52`
enum, `DS4_SHAPE_GLM52` preset (confirmed GlmMoeDsaConfig values; MLA dims / n_out_group
/ n_swa / n_hash_layer marked VERIFY), selection branch in
`ds4_select_shape_from_metadata`, and a GLM52 case in `ds4_expected_layer_compress_ratio`
(placeholder, mirrors PRO). Compiles clean (-Wall -Wextra). Cannot runtime-test the
selection yet — no GLM-5.2 GGUF in DS4 layout exists.

### P1 findings (tiny-random/glm-5 inspected, real GlmMoeDsa tensors)

Conversion pipeline (gguf-tools): HF safetensors → F16 GGUF (deepseek4.* metadata
carried) → `deepseek4-quantize.c` (name_map + q8/q4/q2/iq2 quant + expert slicing +
imatrix). The name_map (top_map+layer_map, ~40 lines) is the main converter change.

GLM→DS4 tensor map (from real names):
- embed_tokens→token_embd, norm→output_norm (tie_word_embeddings=true → NO separate output.weight)
- input_layernorm→attn_norm, post_attention_layernorm→ffn_norm
- self_attn.q_a_proj→attn_q_a, q_a_layernorm→attn_q_a_norm, q_b_proj→attn_q_b
- kv_a_proj_with_mqa[kv_lora+rope=576]→attn_kv, kv_a_layernorm→attn_kv_a_norm, kv_b_proj[1792,512] SEPARATE (DS4 folds kv)
- self_attn.o_proj[full, n_head*v_head=1024] — GLM has NO output-lora (DS4 uses wo_a/wo_b)
- indexer.wk + k_norm + wq_b + weights_proj — GLM indexer differs (DS4 uses a compressor ape/kv/gate)
- mlp.gate.weight→ffn_gate_inp, mlp.gate.e_score_correction_bias→exp_probs_b (sigmoid + noaux_tc routing)
- mlp.experts.N.{gate,up,down}_proj→routed W1/W3/W2, mlp.shared_experts.*→ffn_*_shexp
- dense layers (first_k_dense_replace): plain mlp.{gate,up,down}_proj

**Key structural divergence — GLM-5.2 is DeepSeek-V3.2-style, NOT V4 HCA:**
- **No mHC** (no hc_* tensors) — DS4 requires them per layer/output. Fix: converter
  synthesizes identity hc tensors, and/or engine skips mHC when n_hc==1 (P2).
- **No KV compressor / compression ratios 4-128** — DS4 has V4 HCA (attn_compressor.*,
  per-layer ratios); GLM uses plain MLA latent + DSA indexer for sparsity. Engine needs
  an attention-assembly variant without the compressor path.
- **Output projection full o_proj** (no output-lora) vs DS4 wo_a/wo_b.
- **Indexer internals differ**: GLM wk + k_norm (direct), DS4 compressor (ape/kv/gate).
- config extras: scoring_func sigmoid, topk_method noaux_tc, rope_interleave true,
  indexer_rope_interleave true, qk_head_dim 256 (nope192+rope64), v_head 256, kv_b 1792.

So the port is NOT pure re-parametrization: MoE/MTP/MLA-q-lora/DSA-concept transfer, but
the attention assembly (no compressor/ratios, full o_proj, GLM indexer, no mHC) is a
real engine variant. Bounded (kernels exist, DSA concept shared) but more than P0 dims.

### P1 converter built + load gauntlet driven (2026-06-26)

Wrote `glm5_to_ds4.py` (HF GlmMoeDsa safetensors → DS4-layout F16/F32 GGUF): name_map,
identity-mHC synthesis (hc_*), attn_sinks zeros, output-lora split (a=I, b=o_proj),
tied output head, compress_ratios all 0 (dense bring-up), full deepseek4.* metadata.
Added DS4_SHAPE_GLM5_TINY preset + selection branch. Drove `./ds4 --inspect` on the
generated tiny GGUF through DS4's full validation gauntlet, fixing each gate:
- ✅ GGUF parses, ✅ shape selection matches GLM variant, ✅ ALL metadata validated
  (compress_ratios, swiglu_clamp_exp as float[NL], rope/eps defaults), ✅ F32/F16 tensor
  type framework works.
- WALL: DS4 requires **Q8_0** (32-elem blocks) for attn_q_a/q_b/kv, attn_output_a/b,
  ffn_*_shexp, output head, and routed experts. The tiny model is too degenerate:
  hidden_size=8 and moe_intermediate=32 are below/at the 32-block boundary (ne0=8 not
  ÷32), and n_head=4 < n_out_group=8 → n_head/n_out_group=0. tiny-random/glm-5 CANNOT
  complete a Q8_0 load — it was perfect for validating the metadata/shape/name/type
  pipeline (which all PASS) but its dims are sub-block-size.

Exact DS4 tensor layout map (from weights_validate_layout, ds4.c:3624-3757):
- Q8_0 2D: attn_q_a[N_EMBD,N_LORA_Q], attn_q_b[N_LORA_Q,q_dim=NH*qk_head], attn_kv[N_EMBD,N_HEAD_DIM],
  attn_output_a[N_HEAD_DIM*(NH/N_OUT_GROUP),out_low_dim], attn_output_b[out_low_dim,N_EMBD],
  ffn_*_shexp, output[N_EMBD,N_VOCAB]
- F16: token_embd[N_EMBD,N_VOCAB], ffn_gate_inp[N_EMBD,N_EXPERT], hc_*_fn
- F32: all norms, hc base/scale (output_hc_base[N_HC], hc_attn_scale[3], hc_attn_base[hc_mix_dim]),
  attn_sinks[N_HEAD], exp_probs_b[N_EXPERT]
- dims: hc_dim=N_EMBD*N_HC, hc_mix_dim=2*N_HC+N_HC^2, out_low_dim=N_OUT_GROUP*N_LORA_O

### P1b: Q8_0 added + synthetic non-degenerate model — load gauntlet driven deep (2026-06-26)

Added Q8_0 block quant to glm5_to_ds4.py; wrote gen_synth_glm.py (random-weight
non-degenerate GlmMoeDsa: hidden 256, 4 layers, 8 heads, 32 experts/top-8, ff 128,
vocab 512 — all ÷32, n_head≥n_out_group) + DS4_SHAPE_GLM5_SYNTH preset. Drove
`ds4 --inspect` on the synth GGUF, fixing each gate IN ORDER (all now PASS):
✅ metadata, ✅ shape select, ✅ F32/F16 types, ✅ hc tensor shapes
(output_hc_base[N_HC=1], output_hc_fn[N_HC,hc_dim], hc_*_base/scale[hc_mix=3],
hc_*_fn[hc_mix,hc_dim]), ✅ attn_kv=kv_a[:KVL], ✅ attn_output_a/b grouped-low-rank
shapes [N_HEAD_DIM*(NH/NOUTG),out_low] & [out_low,N_EMBD], ✅ Q8_0 for
attn_q_a/q_b/kv/output_a/b/shexp/output.
- **attn_q_b is absorbed-MLA**: DS4 wants [N_LORA_Q, NH*N_HEAD_DIM(=512)] = q into the
  512 latent; GLM q_b is NH*qk_head(256). Confirms DS4 uses ABSORBED MLA (kv_b folded
  into q_b + o into latent-space output); GLM stores explicit kv_b_proj + full o_proj.
  Real conversion needs the absorption math (P2). Synth uses random correct-shaped.
- **FINAL gate hit**: routed experts (ffn_gate/up/down_exps) must be a k-quant —
  `tensor_is_routed_expert_type` accepts only **IQ2_XXS / Q2_K / Q4_K** (NOT Q8_0/F16).
  ds4.c:3500. So the converter needs Q4_K (simplest) or Q2_K block quant for experts.

### P1c: GLM LOADS in DS4 — full load achieved (2026-06-27)

Added Q4_K block quant (superblock 256, 6-bit scale packing) for the 3 routed-expert
tensors and I32 for ffn_gate_tid2eid (hash table, [N_VOCAB,N_EXPERT_USED]) + the required
DeepSeek special tokens (BOS/EOS/User/Assistant/think/DSML) to the placeholder vocab.
Bumped synth moe_intermediate to 256 (Q4_K needs ÷256). RESULT:
- ✅ **`ds4 --inspect` FULLY SUCCEEDS**: model loads, all 103 tensors validated
  (f32 43, f16 14, q8_0 33, q4_k 12, i32 1). The GLM→DS4 converter produces a GGUF DS4
  loads end-to-end.
- ✅ **GPU forward STARTS**: CUDA backend init, model tensors prepared on device, context
  buffers allocated, graph-prefill begins.
- WALL: `gpu layer 0 attention batch encode failed`. The MLA attention kernel fails on
  the SYNTHESIZED random attn tensors (attn_q_b/output_a/b are random correct-shape, not
  valid absorbed-MLA), and/or the all-dense (compress_ratio 0 every layer) path. Expected:
  random attn ≠ valid MLA math. MoE/load/quant all work.

MILESTONE: GLM ingestion → DS4 GPU LOAD is DONE. The converter (name_map, mHC synth,
Q8_0/Q4_K/I32 quant, special tokens) + engine scaffold (3 GLM presets, router patch,
compress-ratio 0) produce a model DS4 loads and begins executing. The remaining gap to a
clean forward is **P2: real MLA attention assembly** — convert GLM's explicit
kv_b_proj/o_proj into DS4's absorbed-MLA attn_q_b/output_a/b (the absorption math), wire
the DSA indexer (top_k 2048, IndexShare), and drop mHC. Then run on a REAL GLM-5.2
checkpoint (hidden 6144 — non-degenerate) for correct output.

Artifacts on box: glm5_to_ds4.py (converter: name_map + mHC synth + Q8_0/Q4_K/I32 +
special tokens), gen_synth_glm.py, glm5_synth_ds4.gguf (LOADS), synth-glm5/, tiny-glm5/.

### P2 entry point pinpointed (2026-06-27): the forward blocker is mHC

`gpu layer 0 attention batch encode failed` is in the **hyper-connection (mHC)** stage,
NOT the core MLA. metal_graph_encode_layer_attention_batch (ds4.c:17440) ALWAYS runs the
mHC pipeline before attention: rms_norm(batch_cur_hc, hc_dim=N_HC*N_EMBD) → matmul
hc_attn_fn → hc_split (+sinkhorn, scale/base) → weighted_sum → attn_cur. It branches on
`DS4_N_HC == 4` (fuse_hc_norm, the tested path). GLM has **n_hc=1** (no mHC) — one of the
hc GPU ops fails on the n_hc=1 / synthesized-identity hc tensors. The ffn encode has the
same mHC structure + output uses hc_expand_add + a cur_hc/next_hc state swap.

**P2 task = add an n_hc==1 bypass**: when DS4_N_HC==1, skip hc_fn matmul / split / sinkhorn
/ weighted_sum and treat the hyper-connection as identity — attn_cur = batch_cur_hc (the
single residual stream), output add is a plain residual (no hc_expand). Touch points:
metal_graph_encode_layer_attention_batch (~17485-17560 hc-in, ~19225-19256 hc-out),
metal_graph_encode_layer_ffn_batch (mirror), and the decode-path equivalents. Plus drop
the synthesized hc tensors from the converter once the engine no longer requires them
(or keep them harmless). This is hot-path engine surgery — needs DS4's exact hc residual
semantics + GPU verification. After mHC bypass: next blocker likely the DSA indexer
(currently skipped via compress_ratio 0) for real attention sparsity, then absorbed-MLA
tensor conversion for a REAL GLM-5.2 (correct output).

JOURNEY SUMMARY: GLM-5.2 (DeepSeek-family: MLA+DSA+MoE+MTP) now LOADS into DS4 and the
forward executes up to the mHC stage. Remaining for a running synth forward: mHC n_hc==1
bypass. For correct real-GLM output: + DSA indexer wiring + absorbed-MLA conversion +
real GLM-5.2 checkpoint. The converter (name_map/quant/tokens) and scaffold (presets,
router patch, compress-ratio) are done and reusable.

### P2 DONE (2026-06-27): GLM synth runs a FULL forward end-to-end

Two engine changes landed; GLM synth now executes prefill→decode→output-head→sample
with NO encode failures (output is garbage tokens — expected, random synth weights).

1. **mHC n_hc==1 identity bypass** (`ds4.c`). DS4 always runs the mHC hyper-connection
   (rms_norm → hc_fn matmul → split/sinkhorn → weighted_sum on input; hc_expand on
   output); every hc primitive hard-returns 0 for `n_hc != 4` (`hc4_split_one` is fixed
   24-wide; `ds4_gpu_hc_split_*`/`weighted_sum`/`sinkhorn` all guard `n_hc != 4 → 0`).
   GLM has n_hc=1 → all fail → "attention/ffn batch encode failed". Fix: a
   `const bool hc_bypass = DS4_N_HC == 1;` branch at every hc site that treats the
   hyper-connection as identity:
   - input (attn & ffn, prefill batch + decode): `attn_cur/ffn_cur = ds4_gpu_tensor_copy`
     of the single residual stream (batch_cur_hc / batch_after_attn_hc), skipping
     norm/matmul/split/weighted_sum. The subsequent attn_norm/ffn_norm rms_norm_weight
     (the `!fuse_hc_norm` path) runs normally.
   - output (attn): `after_attn_hc = ds4_gpu_add_tensor(attn_out, cur_hc)` (plain residual).
   - output (ffn): `after_ffn_hc = routed_out + shared_out + after_attn_hc` (two adds).
   - 3 output heads (single, batch-spec, mtp): `output_embd = copy(cur_hc)`, skip
     hc_fn/weights/weighted_sum.
   - f16 fast paths gated off under bypass (`attn_out_f16`, `shared_down_f16`,
     `fuse_attn_out_hc`, `fuse_shared_down_hc`) so the plain f32 tensors my adds need
     are produced. ALL gated on DS4_N_HC==1 → n_hc==4 Flash/Pro path byte-for-byte intact.

2. **Router n_expert_used 6→generic** (`ds4_cuda.cu`). After mHC, ffn failed: the router
   guard rejected synth (32 exp / top-8) — `n_expert_used != 6 → return 0`, and the serial
   `router_select_kernel` hardcoded `*6` strides + `for j<6` top-k + `sel[6]`. Fix: added a
   `uint32_t n_used` param to the serial kernel (all 6→n_used: sel/weights/hash stride,
   top-k insert loop, weight normalize); relaxed both host guards to
   `n_expert==0||n_used==0||n_used>n_expert` only; dispatch `n_expert!=256 || n_used!=6`
   → serial(n_expert,n_used,scale); 256/top-6 still on the untouched fast warp/parallel
   path. Consumer (`routed_moe_batch`) + buffers already stride by N_EXPERT_USED, so top-8
   wires through. (This is the PR-#466 follow-up — the warp/parallel kernels remain top-6.)

VERIFIED on box (CUDA H200): GLM synth `./ds4 -m glm5_synth_ds4.gguf` runs all 4 layers +
generates. **Flash zero-regression** on the SAME GLM-modified build: generation 40.02 t/s
(baseline ~39-40), prefill 121 t/s, `ds4_test --logprob-vectors` OK. Both gates (n_hc==1,
256/top-6) keep Flash on identical code paths.

NEXT for CORRECT real-GLM output (synth output is meaningless by construction):
(a) DSA indexer wiring (currently skipped via compress_ratio 0 → dense; need top_k=2048 +
IndexShare for real sparsity); (b) absorbed-MLA conversion in glm5_to_ds4.py (GLM explicit
kv_b_proj + full o_proj → DS4 absorbed attn_q_b / output_a/b); (c) run on a REAL GLM-5.2
checkpoint (hidden 6144, non-degenerate) and validate logits vs HF GlmMoeDsa reference.

### P3 IN PROGRESS (2026-06-27): oracle established + absorbed-MLA PROVEN

Pivot from theoretical: stood up a TRUSTED reference. Box now has a CPU venv
`/root/glmref` (torch 2.12.1+cpu, transformers 5.12.1 — **GlmMoeDsa is in 5.12.1**).
`/root/glm_ref.py` builds a small REAL-arch GlmMoeDsa (hidden128/2L/fkd1/4head, kv_lora64,
q_lora48, rope16/nope32/v32, qk_head48, 8exp/top4, index_topk=1e5→dense) → saves weights
`/root/glm-ref-small/` + reference logits `/root/glm_ref_logits.npy` [8,100]. This is the
oracle for the whole port (the full 744B won't fit 1TB RAM for a CPU reference — 1.5TB
bf16 — so a small real-arch model is THE validation path).

**Absorbed-MLA math PROVEN** (`/root/glm_absorb_test.py`): explicit GLM attention vs the
DS4 absorbed form match to **max abs 2e-08 / rel 2e-07** (fp identical). Exact recipe
(verified against transformers GlmMoeDsa source modeling_glm_moe_dsa.py):
- GLM explicit attn: q=q_b(q_a_norm(q_a(x))) split nope/rope; compressed_kv=kv_a_mqa(x)
  split [c_kv(kv_lora) | k_rope]; c_kv→kv_a_norm→kv_b split [k_nope | v]; rope is
  **INTERLEAVE** (apply_rotary_pos_emb_interleave), k_rope shared across heads (MQA);
  score=(q_nope·k_nope + q_rope·k_rope)·**qk_head^-0.5**; o=softmax·v→o_proj. **NO sink.**
- Absorption: `W_k[h],W_v[h] = kv_b.view(H,nope+v,kv_lora)[:, :nope], [:, nope:]`.
  q_latent[h]=q_pass[h]@W_k[h] (∈ kv_lora). `q_ds4[h]=[q_latent | q_rope]`,
  `kv_ds4=[c_kv_normed | k_rope]` → **N_HEAD_DIM = kv_lora+rope = 576** (NOT 512!).
  ctx=softmax(q_ds4·kv_ds4·scale)@c_kv_normed; `o[h]=ctx@W_v[h]^T`; →o_proj.
- DS4 GGUF tensor recipe (exact, rank-exact output):
  · attn_q_a=q_a_proj, attn_q_a_norm=q_a_layernorm
  · attn_q_b[h]: nope rows = W_k[h]^T@q_b_nope[h] [kv_lora,q_lora]; rope rows = q_b_rope[h];
    stacked [NH*N_HEAD_DIM, q_lora] then transposed; **× sqrt(N_HEAD_DIM/qk_head)** to
    convert DS4's 1/sqrt(N_HEAD_DIM) scaling to GLM's 1/sqrt(qk_head) (real factor=1.5).
  · attn_kv = kv_a_proj_with_mqa (→[c_kv | k_rope]=N_HEAD_DIM), attn_kv_a_norm=kv_a_layernorm
  · **n_out_group=NH(64), n_lora_o=v_head(256)** → out_low=NH*v_head=o_proj-in.
    output_a[h]=W_v[h]@[I_{kv_lora}|0_{rope}] (per-head, group_heads=1); output_b=o_proj. EXACT.
- **GLM52 preset fixes needed**: n_head_dim 512→**576**, n_value_dim 256→**512**,
  n_out_group 8→**64**, n_lora_o 1024→**256**. (current values were Flash-copy placeholders.)
- **Engine GLM-attn divergences to gate on the variant** (DS4 ops GLM lacks): (1) **skip
  head_rms_norm on q** (layer_q_projection_*; GLM has none); (2) **attn_sinks → -inf**
  (GLM no sink; converter must emit -inf not 0); (3) kv_a_norm applies to **c_kv only**
  (first kv_lora dims), not the rope tail — DS4 rms-norms full N_HEAD_DIM; (4) confirm DS4
  rope_tail interleave matches GLM on the 576 tail-64. Each surfaces as a logit mismatch
  vs glm_ref_logits.npy and gets fixed there.

PLAN to converge: add a DS4 preset matching glm-ref-small dims (N_HEAD_DIM=80,n_value=64,
n_out_group=4,n_lora_o=32) + GLM-attn engine gates; extend glm5_to_ds4.py with the
absorbed recipe; convert glm-ref-small → GGUF; `./ds4 --dump-logits` vs glm_ref_logits.npy;
iterate gates until match; THEN apply to the real 744B (downloading to /root/glm52-hf).
Artifacts: /root/glm_ref.py, /root/glm_absorb_test.py, /root/glm-ref-small/, /root/glmref venv.

**FULL recipe PROVEN end-to-end (2026-06-27)**: `/root/glm_ds4_forward.py` reimplements
the COMPLETE forward in the DS4-absorbed style (absorbed-MLA + mHC=identity residual adds +
NO head_rms_norm + kv-norm on c_kv only + NO sink + MoE[sigmoid routing, noaux bias,
top-k norm, scale 2.5, shared expert] + interleave rope + RMS norms) and matches the
transformers GlmMoeDsa logits to **max 5e-05 / rel 8e-05, argmax identical at every
position**. This numpy file IS the executable spec the C engine must reproduce. Every
conversion + engine decision is now validated, not assumed.

### ✅ P3 CORRECTNESS ACHIEVED (2026-06-27): DS4 logits MATCH HF GlmMoeDsa

`./ds4 --cuda -m glm_ref_ds4.gguf --raw-ids "1..8" --dump-logits` vs the transformers
GlmMoeDsa reference (`glm_ref_logits.npy`): **argmax 451==451, corr 0.9994, max|Δ| 0.035**
(= Q8_0/Q4_K quant noise; F16 would be ~1e-3). DS4 now computes GLM-5.2 correctly
end-to-end on the real architecture. Localized + fixed via per-layer dumps
(DS4_METAL_GRAPH_DUMP_*): attn_norm/attn_out/ffn_shexp all corr~1.0 per layer → residual
stream correct → bug was the output head.

The full set of fixes that made it correct (all variant-gated on DS4_N_HC==1; Flash untouched):
- **Engine** (ds4.c/ds4_cuda.cu): mHC n_hc==1 bypass; router serial kernel **n_used** (top-8)
  + **scoring_sigmoid** (GLM uses sigmoid, DeepSeek sqrt-softplus) threaded through both
  router wrappers; **skip head_rms_norm on q** (force non-fused q_b path); **glm_kv_norm**
  kernel (RMS-norm c_kv/kv_lora only, pass k_rope tail) replacing the 576-wide kv-norm;
  preset n_head_dim 512→576, rms_eps→1e-5.
- **Converter** (glm5_to_ds4.py): absorbed-MLA tensors; metadata (key_length 576 etc.);
  attn_sinks=-1e30; dense layer = shared(dense MLP)+routed(0); **output.weight = lm_head if
  untied else embed** (was the final bug — glm-ref-small is untied; real GLM-5.2 is tied).
- **CLI** (ds4_cli.c): `--raw-ids "i,j,.."` injects token ids, bypassing the placeholder
  tokenizer, so DS4 and the transformers ref process identical ids for the logit compare.

REMAINING:
1. **Decode-path gates** for generation (only prefill/dump-logits validated): apply the same
   GLM gates to metal_graph_encode_decode_layer (head_rms_norm 15167, kv-norm 15115, router
   15713 sigmoid already done) — needed for multi-token generation, not for the logit match.
2. **Real GLM-5.2 (744B)**: downloading (~490G/1.5T). On completion: fix the GLM52 *real*
   preset (n_out_group 8→64, n_head_dim 512→576, n_value 256→512, n_lora_o 1024→256), note
   real model is **tied** (output=embed), stream-convert (shard-by-shard) since 1.5TB won't
   fit RAM, then run. (b) tokenizer + chat template for real text I/O.

### P3 progress (2026-06-27 cont'd): converter + preset DONE, absorbed GGUF LOADS

- glm5_to_ds4.py now emits the **absorbed-MLA** tensors (the recipe above): attn_q_b =
  per-head [W_k^T@q_b_nope ; q_b_rope]×1.5, attn_kv = kv_a_proj_with_mqa (576), attn_kv_a_norm
  = [kv_a_layernorm | ones(rope)], attn_output_a[h]=W_v[h]@[I|0], attn_output_b=o_proj,
  attn_sinks=-1e30, dense layer = shared(dense MLP)+routed(zeros). Metadata fixed:
  key_length=576, value_length=512, output_lora_rank=v_head, output_group_count=NH,
  rope.freq_base=cfg rope_theta, rms eps=cfg(1e-5).
- DS4 presets: GLM5_SYNTH n_head_dim 512→**576**; GLM5_SYNTH/TINY/GLM52 rms_eps→**1e-5**
  (GLM value). Reference model glm_ref.py re-dimmed to match GLM5_SYNTH exactly (H256/NH8/
  kv512/q64/rope64/nope192/v256/32exp-top8/ff256/voc512/4L/fkd1, index_topk=2048≥seq→dense).
- `./ds4 --inspect -m glm_ref_ds4.gguf` **fully loads** (103 tensors: f32 43/f16 14/q8_0 33/
  q4_k 12/i32 1) — converter+preset+absorbed-MLA shapes all consistent end-to-end.

REMAINING to a correct DS4 logit match (then 744B):
A. **Engine GLM-attn gates** (variant-gated; GPU path, exact sites):
   - skip **head_rms_norm on q**: GPU batch ds4.c:17749 (`ds4_gpu_head_rms_norm_tensor`)
     + the fused f16 path 17702 (`attn_q_b_f16_head_rms_rope_tail` → force non-fused for GLM);
     decode 15150/15167. CPU 6883/6902/6935/9529. (GLM has no per-head q norm.)
   - **kv_a_norm over kv_lora(512) only**, not full N_HEAD_DIM(576): GPU batch fused
     `ds4_gpu_dsv4_qkv_rms_norm_rows_tensor` 17666 (+ decode 15115); leave k_rope tail raw.
   - verify DS4 ropes the kv tail-64 (k_rope) the same as GLM (interleave); q rope at 17759.
B. **Token-id alignment** for the logit compare: feed DS4 the SAME ids the transformers ref
   used (torch ids [1..8], no BOS) — craft prompt or a raw-id path; DS4 --dump-logits vs
   glm_ref_logits.npy. Iterate gates until match.
C. Apply to the real 744B (downloading) + fix GLM52 real preset (n_out_group 8→64,
   n_head_dim 512→576, n_value 256→512, n_lora_o 1024→256) to match the absorbed layout.

ORIGINAL spec (kept for reference) — make C match glm_ds4_forward.py (all well-specified now):
1. Preset: GLM5_SYNTH/GLM52 n_head_dim 512→**576** (=kv_lora+rope). Others already correct
   (n_value=512, n_out_group=8/64, n_lora_o=256). Q8_0 divisibility holds (576÷32=18).
2. Engine GLM-attn gates (variant-gated, GPU batch+decode paths; mHC already bypassed):
   - **skip head_rms_norm on q** (layer_q_projection_* / GPU equiv) — GLM has none.
   - **kv_a_norm on first kv_lora dims only** (not the rope tail of N_HEAD_DIM) — DS4
     currently rms-norms the whole N_HEAD_DIM.
   - scaling: DS4 uses 1/sqrt(N_HEAD_DIM)=1/sqrt(576); GLM wants 1/sqrt(qk_head)=1/sqrt(256)
     → converter folds sqrt(576/256)=1.5 into attn_q_b (engine unchanged), OR gate the scale.
   - rope: confirm DS4 rope_tail interleave on the 576 tail-64 matches GLM (q AND kv tail).
3. Converter glm5_to_ds4.py absorbed recipe (replaces the random synth attn):
   - attn_q_b[h] = stack[ (W_k[h]^T @ q_b_nope[h]) ; q_b_rope[h] ] × 1.5, layout [N_LORA_Q, NH*576]
   - attn_kv = kv_a_proj_with_mqa (→[c_kv|k_rope]=576), attn_kv_a_norm = kv_a_layernorm
   - attn_output_a[h] = W_v[h] @ [I_512 | 0_64] (zeros rope tail), attn_output_b = o_proj
   - attn_sinks = -inf (large negative), exp_probs_b = mlp.gate.e_score_correction_bias
   Then convert glm-ref-small → GGUF, --dump-logits (CPU path needs CPU mHC n_hc==1 bypass
   too, OR validate on GPU --cuda which already has it), compare to glm_ref_logits.npy.

### ✅ P5 2-GPU FULL RESIDENCY ACHIEVED (2026-06-28): GLM-5.2 744B runs resident + fast

The real q2 GLM (252 GB GGUF `/home/glm52_ds4_q2k.gguf`, 775B params) RUNS fully resident
across 2× H200 with **no SSD streaming**: prefill 32.3 t/s, **generation 16.3 t/s**, coherent
English output. `--raw-ids "[gMASK]<sop>...The capital of France is<|assistant|><think>"` →
generated `"1.  **Analyze the Request:** The user provided"` (model reasoning correctly).

**The distributed "model id mismatch: worker=2 coordinator=0" was MIS-diagnosed earlier** (the
"global-state reversion" theory in §4 of MAINTAINER_NOTES was wrong). Real root cause, proven by
instrumenting `config_validate_model`/`model_open`: the launch script passed `-m` to the worker
but **omitted `-m` on the coordinator**, so the coordinator loaded the CLI default
`ds4flash.gguf` (`ds4_cli.c:1409`) — real DeepSeek-V4 Flash (variant 0, block_count=43). Right
after `model_open` the coordinator read block_count=43/n_kv=62 (Flash) vs worker 78/36 (GLM).
The engine's mismatch detection was correct. **Fix = same `-m` on both nodes** (no engine
change; only a clearer mismatch error string added in `ds4_distributed.c`).

**Second blocker (memory budget, real finding)**: after the handshake/route succeed, the GLM
prefill graph OOM'd. `cuda_q8_f16_cache_reserve_bytes()` (`ds4_cuda.cu:512`) reserves only
**512 MiB** free on cards ≥112 GiB, so the optional q8→fp16 dequant cache fills HBM and starves
the graph. `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB` does NOT bound it. Fix per node:
`DS4_CUDA_Q8_F16_CACHE_RESERVE_MB` (20480 coord / 8192 worker). Working split: coordinator
`--layers 0:36` (+output head), worker `--layers 37:output`. Script `/root/run_glm_res2.sh`.
Suggested engine fix: scale the ≥112 GiB reserve with model/graph size instead of a flat 512 MiB.

**Chat with real text**: working via GLM `apply_chat_template` → ids → `ds4 --raw-ids` → GLM
decode (`glm_chat_2gpu.sh`, `glm_chat.py`) — guaranteed-correct tokenization, no engine change.
Native-tokenizer-in-GGUF attempt: `glm_tok_rewrite.py` swaps tokenizer.ggml.tokens/merges and
copies the 252 GB data verbatim (~minutes vs ~1.5 h reconvert). Open risk: DS4's pre-tokenizer
is `joyai-llm`; GLM's differs — gate native tokenization on `--dump-tokens` vs AutoTokenizer
before trusting it. ds4-server supports the distributed coordinator role, so native tokenizer +
server = a persistent interactive 2-GPU chat endpoint (vs the per-turn reload of the CLI path).
