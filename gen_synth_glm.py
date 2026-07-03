#!/usr/bin/env python3
# Generate a SMALL but NON-DEGENERATE GlmMoeDsa model (real-ratio dims, all ÷32,
# n_head>=n_out_group) so DS4's Q8_0 layout can load it. Random weights — tests
# load + forward mechanics, not correctness.
import json, struct, os, numpy as np

OUT = "/root/ds4/synth-glm5"
os.makedirs(OUT, exist_ok=True)

H        = 256          # hidden (÷32)
NL       = 4
FKD      = 1            # first_k_dense_replace
NH       = 8            # heads
NOUTG    = 8            # n_head>=n_out_group, n_head%n_out_group==0
KVL      = 512          # kv_lora_rank (= DS4 n_head_dim)
QL       = 64           # q_lora_rank (÷32)
ROPE     = 64
NOPE     = 192
VHD      = 256
QKH      = NOPE+ROPE    # 256
NEXP     = 32           # routed experts (>= top-k)
NUSE     = 8
FFE      = 256          # moe_intermediate (÷256 for Q4_K experts)
INTER    = 256          # dense mlp (÷32)
VOC      = 512          # vocab (÷32)
IXH      = 8
IXD      = 128
IXK      = 2048

cfg = {
  "architectures":["GlmMoeDsaForCausalLM"],"model_type":"glm_moe_dsa","dtype":"bfloat16",
  "hidden_size":H,"num_hidden_layers":NL,"first_k_dense_replace":FKD,
  "num_attention_heads":NH,"num_key_value_heads":NH,"head_dim":ROPE,
  "kv_lora_rank":KVL,"q_lora_rank":QL,"qk_rope_head_dim":ROPE,"qk_nope_head_dim":NOPE,
  "qk_head_dim":QKH,"v_head_dim":VHD,
  "n_routed_experts":NEXP,"num_experts_per_tok":NUSE,"n_shared_experts":1,
  "moe_intermediate_size":FFE,"intermediate_size":INTER,"vocab_size":VOC,
  "index_n_heads":IXH,"index_head_dim":IXD,"index_topk":IXK,
  "routed_scaling_factor":2.5,"rms_norm_eps":1e-5,"norm_topk_prob":True,
  "scoring_func":"sigmoid","topk_method":"noaux_tc","n_group":1,"topk_group":1,
  "rope_parameters":{"rope_theta":1000000,"rope_type":"default"},
  "bos_token_id":0,"eos_token_id":1,"tie_word_embeddings":True,
}
json.dump(cfg, open(os.path.join(OUT,"config.json"),"w"), indent=1)

rng = np.random.default_rng(0)
def w(*shape): return (rng.standard_normal(shape)*0.02).astype(np.float32)

tens = {}
tens["model.embed_tokens.weight"] = w(VOC,H)
tens["model.norm.weight"] = w(H)
for il in range(NL):
    p=f"model.layers.{il}."
    tens[p+"input_layernorm.weight"]=w(H)
    tens[p+"post_attention_layernorm.weight"]=w(H)
    tens[p+"self_attn.q_a_proj.weight"]=w(QL,H)
    tens[p+"self_attn.q_a_layernorm.weight"]=w(QL)
    tens[p+"self_attn.q_b_proj.weight"]=w(NH*QKH,QL)
    tens[p+"self_attn.kv_a_proj_with_mqa.weight"]=w(KVL+ROPE,H)
    tens[p+"self_attn.kv_a_layernorm.weight"]=w(KVL)
    tens[p+"self_attn.kv_b_proj.weight"]=w(NH*(NOPE+VHD),KVL)
    tens[p+"self_attn.o_proj.weight"]=w(H,NH*VHD)
    tens[p+"self_attn.indexer.wk.weight"]=w(IXD,H)
    tens[p+"self_attn.indexer.wq_b.weight"]=w(IXH*IXD,QL)
    tens[p+"self_attn.indexer.weights_proj.weight"]=w(IXH,H)
    tens[p+"self_attn.indexer.k_norm.weight"]=w(IXD)
    tens[p+"self_attn.indexer.k_norm.bias"]=w(IXD)
    if il<FKD:
        tens[p+"mlp.gate_proj.weight"]=w(INTER,H)
        tens[p+"mlp.up_proj.weight"]=w(INTER,H)
        tens[p+"mlp.down_proj.weight"]=w(H,INTER)
    else:
        tens[p+"mlp.gate.weight"]=w(NEXP,H)
        tens[p+"mlp.gate.e_score_correction_bias"]=w(NEXP)
        tens[p+"mlp.shared_experts.gate_proj.weight"]=w(FFE,H)
        tens[p+"mlp.shared_experts.up_proj.weight"]=w(FFE,H)
        tens[p+"mlp.shared_experts.down_proj.weight"]=w(H,FFE)
        for e in range(NEXP):
            tens[p+f"mlp.experts.{e}.gate_proj.weight"]=w(FFE,H)
            tens[p+f"mlp.experts.{e}.up_proj.weight"]=w(FFE,H)
            tens[p+f"mlp.experts.{e}.down_proj.weight"]=w(H,FFE)

# write safetensors
hdr={}; off=0; blobs=[]
for k,v in tens.items():
    v=np.ascontiguousarray(v.astype(np.float32))
    b=v.tobytes()
    hdr[k]={"dtype":"F32","shape":list(v.shape),"data_offsets":[off,off+len(b)]}
    off+=len(b); blobs.append(b)
hjson=json.dumps(hdr).encode()
pad=(-len(hjson))%8; hjson+=b" "*pad
with open(os.path.join(OUT,"model.safetensors"),"wb") as f:
    f.write(struct.pack("<Q",len(hjson))); f.write(hjson)
    for b in blobs: f.write(b)
print(f"synth GlmMoeDsa: H={H} L={NL} heads={NH} experts={NEXP}/{NUSE} vocab={VOC} -> {OUT}")
print(f"shape preset values: n_layer={NL} n_embd={H} n_vocab={VOC} n_head={NH} n_head_dim={KVL} "
      f"n_rot={ROPE} n_out_group={NOUTG} n_lora_q={QL} n_lora_o={H} n_expert={NEXP} "
      f"n_expert_used={NUSE} n_ff_exp={FFE} n_hash_layer={FKD} indexer_head={IXH}")
