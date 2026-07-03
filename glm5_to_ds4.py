#!/usr/bin/env python3
# Convert a GlmMoeDsa (GLM-5.2) HF safetensors model to a DS4-layout F16 GGUF.
# First-load bring-up: compress_ratios all 0 (dense MLA, no DSA indexer / no V4
# compressor), synthesize the DeepSeek-V4 tensors GLM lacks (mHC hc_*, attn_sinks,
# output-lora split, tied output head). Goal: get DS4 to LOAD + run a forward pass.
# Correctness (DSA indexer, real attention assembly) is later (P2).
import json, struct, sys, os, numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/ds4/tiny-glm5"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/ds4/glm5_tiny_ds4.gguf"

cfg = json.load(open(os.path.join(SRC, "config.json")))
H   = cfg["hidden_size"]
NL  = cfg["num_hidden_layers"]
NH  = cfg["num_attention_heads"]
KVL = cfg["kv_lora_rank"]          # 512
QL  = cfg["q_lora_rank"]           # 32 (tiny) / 2048 (real)
ROPE= cfg["qk_rope_head_dim"]      # 64
NOPE= cfg["qk_nope_head_dim"]      # 192
VHD = cfg["v_head_dim"]            # 256
NEXP= cfg["n_routed_experts"]      # 256
NUSE= cfg["num_experts_per_tok"]   # 8
FFE = cfg["moe_intermediate_size"]
VOC = cfg["vocab_size"]
IXH = cfg["index_n_heads"]
IXD = cfg["index_head_dim"]
IXK = cfg["index_topk"]
SCALE = cfg.get("routed_scaling_factor", 2.5)
EPS = cfg.get("rms_norm_eps", 1e-5)
FKD = cfg.get("first_k_dense_replace", 1)
QKH = NOPE + ROPE                  # qk_head_dim 256
NHD = KVL + ROPE                   # DS4 n_head_dim = absorbed latent (c_kv 512 + k_rope 64) = 576
NLO = VHD                          # DS4 n_lora_o = v_head (per-head output low rank), group_heads=1
NOUTG = NH                         # n_out_group = NH so group_heads=1 -> exact per-head absorption
ROPE_THETA = float(cfg.get("rope_parameters",{}).get("rope_theta", cfg.get("rope_theta",10000.0)))
SCALE_FOLD = (NHD / QKH) ** 0.5    # DS4 scores use 1/sqrt(NHD); GLM uses 1/sqrt(QKH) -> fold into q_b
rng = np.random.default_rng(0)

# ---- load safetensors ----
def load_st(path):
    f=open(path,"rb"); n=struct.unpack("<Q",f.read(8))[0]; hdr=json.loads(f.read(n))
    base=f.tell()-0; data_off=8+n
    out={}
    for k,v in hdr.items():
        if k=="__metadata__": continue
        dt=v["dtype"]; sh=v["shape"]; a,b=v["data_offsets"]
        f.seek(data_off+a)
        raw=f.read(b-a)
        npdt={"BF16":np.uint16,"F16":np.float16,"F32":np.float32,"F64":np.float64,
              "I32":np.int32,"I64":np.int64,"U8":np.uint8}[dt]
        arr=np.frombuffer(raw,dtype=npdt).reshape(sh) if sh else np.frombuffer(raw,dtype=npdt)
        if dt=="BF16":
            arr=(arr.astype(np.uint32)<<16).view(np.float32)
        out[k]=np.asarray(arr,dtype=np.float32)
    return out

st = load_st(os.path.join(SRC,"model.safetensors"))
def g(name):
    if name not in st: raise KeyError(name)
    return st[name]
def has(name): return name in st

# ---- GGUF writer ----
GGUF_MAGIC=0x46554747; GGUF_VER=3
T_U32=4; T_F32=6; T_BOOL=7; T_STR=8; T_ARR=9; T_U64=10
GGML_F32=0; GGML_F16=1; GGML_Q8_0=8; GGML_Q4_K=12; GGML_I32=26

def quant_q4_K(arr):
    # Q4_K: superblock of 256 = 8 sub-blocks of 32. Quantize last axis (must be ÷256).
    # Layout per superblock (144B): fp16 d, fp16 dmin, scales[12], qs[128].
    a=np.ascontiguousarray(arr.astype(np.float32))
    ne0=a.shape[-1]; assert ne0%256==0, f"q4_K ne0={ne0} not mult 256"
    rows=a.reshape(-1,ne0); nr=rows.shape[0]; nsb=ne0//256
    out=bytearray()
    for r in range(nr):
        row=rows[r]
        for sb in range(nsb):
            x=row[sb*256:(sb+1)*256].reshape(8,32)
            lo=x.min(axis=1); hi=x.max(axis=1)
            dblk=(hi-lo)/15.0; dblk=np.where(dblk>1e-12,dblk,1e-12)
            q=np.clip(np.round((x-lo[:,None])/dblk[:,None]),0,15).astype(np.uint8)
            Bblk=np.clip(-lo,0,None)
            superd=max(dblk.max()/63.0,1e-12); superm=max(Bblk.max()/63.0,1e-12)
            sc=np.clip(np.round(dblk/superd),0,63).astype(np.int32)
            mn=np.clip(np.round(Bblk/superm),0,63).astype(np.int32)
            scales=bytearray(12)
            for j in range(4):
                scales[j]=int(sc[j]&63); scales[j+4]=int(mn[j]&63)
            for j in range(4,8):
                scales[j+4]=int((sc[j]&0xF)|((mn[j]&0xF)<<4))
                scales[j-4]=int(scales[j-4]|((sc[j]>>4)<<6))
                scales[j]  =int(scales[j]  |((mn[j]>>4)<<6))
            qs=bytearray(128)
            for pair in range(4):
                b0=2*pair; b1=2*pair+1
                lo_n=q[b0]; hi_n=q[b1]
                for t in range(32):
                    qs[pair*32+t]=int(lo_n[t] | (hi_n[t]<<4))
            out+=struct.pack("<e",float(np.float16(superd)))
            out+=struct.pack("<e",float(np.float16(superm)))
            out+=bytes(scales); out+=bytes(qs)
    return bytes(out)

def quant_q8_0(arr):
    # quantize along last axis (becomes GGUF ne0); must be multiple of 32.
    a=np.ascontiguousarray(arr.astype(np.float32))
    ne0=a.shape[-1]
    assert ne0%32==0, f"q8_0 ne0={ne0} not mult of 32"
    rows=a.reshape(-1,ne0)
    nb=ne0//32
    blk=rows.reshape(rows.shape[0],nb,32)
    amax=np.max(np.abs(blk),axis=2)            # [r,nb]
    d=amax/127.0
    dz=np.where(d>0,d,1.0)
    q=np.round(blk/dz[:,:,None]).clip(-127,127).astype(np.int8)  # [r,nb,32]
    dh=d.astype(np.float16)
    # interleave per block: fp16 d then 32 int8
    out=bytearray()
    qb=q.tobytes(); dbts=dh.tobytes()
    nr=rows.shape[0]
    # build via numpy structured concat for speed
    rec=np.empty((nr,nb,34),dtype=np.uint8)
    rec[:,:,0:2]=dh.view(np.uint8).reshape(nr,nb,2)
    rec[:,:,2:34]=q.view(np.uint8)
    return rec.tobytes()
def s(b): return struct.pack("<Q",len(b))+b
def kv_str(k,v): return s(k.encode())+struct.pack("<I",T_STR)+s(v.encode())
def kv_u32(k,v): return s(k.encode())+struct.pack("<I",T_U32)+struct.pack("<I",v)
def kv_u64(k,v): return s(k.encode())+struct.pack("<I",T_U64)+struct.pack("<Q",v)
def kv_f32(k,v): return s(k.encode())+struct.pack("<I",T_F32)+struct.pack("<f",v)
def kv_bool(k,v):return s(k.encode())+struct.pack("<I",T_BOOL)+struct.pack("<b",1 if v else 0)
def kv_arr_u32(k,vals):
    b=s(k.encode())+struct.pack("<I",T_ARR)+struct.pack("<I",T_U32)+struct.pack("<Q",len(vals))
    for x in vals: b+=struct.pack("<I",x)
    return b
def kv_arr_str(k,vals):
    b=s(k.encode())+struct.pack("<I",T_ARR)+struct.pack("<I",T_STR)+struct.pack("<Q",len(vals))
    for x in vals: b+=s(x.encode())
    return b
def kv_arr_f32(k,vals):
    b=s(k.encode())+struct.pack("<I",T_ARR)+struct.pack("<I",T_F32)+struct.pack("<Q",len(vals))
    for x in vals: b+=struct.pack("<f",x)
    return b

meta=[]
meta.append(kv_str("general.architecture","deepseek4"))
meta.append(kv_str("general.name","GLM 5.2 tiny"))
def m_u32(k,v): meta.append(kv_u32("deepseek4."+k,v))
def m_f32(k,v): meta.append(kv_f32("deepseek4."+k,v))
m_u32("block_count",NL)
m_u32("embedding_length",H)
m_u32("vocab_size",VOC)
m_u32("attention.head_count",NH)
m_u32("attention.key_length",NHD)        # n_head_dim = absorbed latent c_kv(512)+k_rope(64)=576
m_u32("attention.head_count_kv",1)
m_u32("attention.value_length",KVL)      # n_value_dim = c_kv latent (value) = 512
m_u32("rope.dimension_count",ROPE)
m_u32("attention.output_group_count",NOUTG)
m_u32("attention.q_lora_rank",QL)
m_u32("attention.output_lora_rank",NLO)  # n_lora_o = v_head (per-head output low rank)
m_u32("expert_count",NEXP)
m_u32("expert_used_count",NUSE)
m_u32("expert_feed_forward_length",FFE)
m_u32("expert_shared_count",1)
m_u32("hash_layer_count",FKD)
m_u32("expert_group_count",0)
m_u32("expert_group_used_count",0)
m_u32("attention.sliding_window",128)
m_u32("attention.indexer.head_count",IXH)
m_u32("attention.indexer.key_length",IXD)
m_u32("attention.indexer.top_k",IXK)
m_u32("hyper_connection.count",1)
m_u32("hyper_connection.sinkhorn_iterations",0)
meta.append(kv_arr_u32("deepseek4.attention.compress_ratios",[0]*NL))
# Bring-up: emit DS4 default rope/eps so they match the shape preset (load test
# only cares about matching; correctness later uses GLM's real rope_theta=1e6 etc).
m_f32("rope.freq_base",ROPE_THETA)
m_f32("attention.compress_rope_freq_base",160000.0)
m_f32("expert_weights_scale",SCALE)
m_f32("attention.layer_norm_rms_epsilon",EPS)
m_f32("hyper_connection.epsilon",1.0e-6)
meta.append(kv_arr_f32("deepseek4.swiglu_clamp_exp",[10.0]*NL))
meta.append(kv_bool("deepseek4.expert_weights_norm",True))
# tokenizer (minimal placeholder vocab so loader has tokens)
SPECIALS=["<｜begin▁of▁sentence｜>","<｜end▁of▁sentence｜>","<｜User｜>",
          "<｜Assistant｜>","<think>","</think>","｜DSML｜"]
toks=SPECIALS+[f"<tok{i}>" for i in range(VOC-len(SPECIALS))]
meta.append(kv_arr_str("tokenizer.ggml.tokens",toks))
meta.append(kv_arr_str("tokenizer.ggml.merges",[]))

# ---- build tensor dict (DS4 names -> (np array, ggml_type)) ----
# F32-typed tensors per DS4 weights_validate_layout (norms, hc base/scale, sinks,
# exp_probs_b). Everything else F16 (matrices, hc_fn, gate_inp, embed).
F32_SUFFIX={"output_hc_base.weight","output_hc_scale.weight","output_norm.weight",
            "hc_attn_scale.weight","hc_attn_base.weight","attn_norm.weight",
            "attn_q_a_norm.weight","attn_kv_a_norm.weight","attn_sinks.weight",
            "hc_ffn_scale.weight","hc_ffn_base.weight","ffn_norm.weight","exp_probs_b.bias"}
Q8_SUFFIX={"attn_q_a.weight","attn_q_b.weight","attn_kv.weight",
           "attn_output_a.weight","attn_output_b.weight",
           "ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight",
           "output.weight"}
Q4K_SUFFIX={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}
def _suf(name): return name.split(".",2)[-1] if name.startswith("blk.") else name
tensors={}
def put(name,arr):
    suf=_suf(name)
    if suf in F32_SUFFIX:
        tensors[name]=(np.ascontiguousarray(arr.astype(np.float32)),GGML_F32,None)
    elif suf in Q8_SUFFIX:
        tensors[name]=(arr,GGML_Q8_0,list(arr.shape))
    elif suf in Q4K_SUFFIX:
        tensors[name]=(arr,GGML_Q4_K,list(arr.shape))
    elif suf=="ffn_gate_tid2eid.weight":
        tensors[name]=(np.ascontiguousarray(arr.astype(np.int32)),GGML_I32,None)
    else:
        tensors[name]=(np.ascontiguousarray(arr.astype(np.float16)),GGML_F16,None)

# hc (hyper-connection) shapes for n_hc=1: hc_dim=H*1=H, hc_mix_dim=2*1+1*1=3
NHC=1
HC_DIM=H*NHC
HC_MIX=2*NHC+NHC*NHC   # 3
put("token_embd.weight", g("model.embed_tokens.weight"))
put("output_norm.weight", g("model.norm.weight"))
# output head: separate lm_head if untied, else tied to embeddings
put("output.weight", g("lm_head.weight") if has("lm_head.weight") else g("model.embed_tokens.weight"))
# output mHC identity: base[N_HC], fn[hc_dim,N_HC] F16, scale[1]
put("output_hc_base.weight", np.ones(NHC,dtype=np.float32))
put("output_hc_fn.weight", np.zeros((NHC,HC_DIM),dtype=np.float32))
put("output_hc_scale.weight", np.ones(1,dtype=np.float32))

for il in range(NL):
    p=f"model.layers.{il}."
    b=f"blk.{il}."
    put(b+"attn_norm.weight", g(p+"input_layernorm.weight"))
    put(b+"ffn_norm.weight", g(p+"post_attention_layernorm.weight"))
    put(b+"attn_q_a.weight", g(p+"self_attn.q_a_proj.weight"))
    put(b+"attn_q_a_norm.weight", g(p+"self_attn.q_a_layernorm.weight"))
    # --- absorbed-MLA (proven vs HF GlmMoeDsa to 7e-06, see glm_ds4_forward.py) ---
    qb  = g(p+"self_attn.q_b_proj.weight").reshape(NH, QKH, QL)              # [H,256,QL]
    kvb = g(p+"self_attn.kv_b_proj.weight").reshape(NH, NOPE+VHD, KVL)       # [H,448,512]
    Wk  = kvb[:, :NOPE, :]                                                    # [H,192,512]
    Wv  = kvb[:, NOPE:, :]                                                    # [H,256,512]
    # attn_q_b [NH*NHD, QL]: per head [ Wk[h]^T @ q_b_nope[h] (512) ; q_b_rope[h] (64) ] * fold
    q_b = np.zeros((NH*NHD, QL), dtype=np.float32)
    for hh in range(NH):
        q_b[hh*NHD : hh*NHD+KVL] = Wk[hh].T @ qb[hh, :NOPE]                   # [512,QL]
        q_b[hh*NHD+KVL : (hh+1)*NHD] = qb[hh, NOPE:]                          # [64,QL]
    put(b+"attn_q_b.weight", (q_b*SCALE_FOLD).astype(np.float32))
    # attn_kv [NHD, H] = kv_a_proj_with_mqa (rows = [c_kv(512) ; k_rope(64)])
    put(b+"attn_kv.weight", g(p+"self_attn.kv_a_proj_with_mqa.weight"))
    # kv_a_norm over NHD: c_kv part = GLM kv_a_layernorm, rope tail = 1 (engine norms c_kv only)
    put(b+"attn_kv_a_norm.weight",
        np.concatenate([g(p+"self_attn.kv_a_layernorm.weight"), np.ones(ROPE,dtype=np.float32)]))
    # GLM has no attention sink -> -inf so exp(sink-max)=0 (no phantom sink mass)
    put(b+"attn_sinks.weight", np.full(NH, -1e30, dtype=np.float32))
    # output: absorbed. output_a[h] = W_v[h] @ [I_512 | 0_64] ; output_b = o_proj.
    out_low = NOUTG*NLO                    # = NH*VHD = o_proj input dim
    a0 = NHD*(NH//NOUTG)                    # = NHD (group_heads=1)
    out_a = np.zeros((out_low, a0), dtype=np.float32)
    for hh in range(NH):
        out_a[hh*NLO:(hh+1)*NLO, :KVL] = Wv[hh]        # rope-tail cols stay 0
    put(b+"attn_output_a.weight", out_a)
    put(b+"attn_output_b.weight", g(p+"self_attn.o_proj.weight"))   # [H, NH*VHD]
    # mHC identity, correct shapes: base[hc_mix] scale[hc_mix] fn[hc_dim,hc_mix]
    for t in ("hc_attn_base","hc_attn_scale","hc_ffn_base","hc_ffn_scale"):
        put(b+t+".weight", np.ones(HC_MIX,dtype=np.float32))
    for t in ("hc_attn_fn","hc_ffn_fn"):
        put(b+t+".weight", np.zeros((HC_MIX,HC_DIM),dtype=np.float32))
    # ffn
    if il < FKD:
        # dense layer (GlmMoeDsaMLP, no routing): put the dense MLP in the shared expert
        # and ZERO the routed experts -> DS4 output = shared(dense) + routed(0) = dense MLP.
        put(b+"ffn_gate_shexp.weight", g(p+"mlp.gate_proj.weight"))
        put(b+"ffn_up_shexp.weight", g(p+"mlp.up_proj.weight"))
        put(b+"ffn_down_shexp.weight", g(p+"mlp.down_proj.weight"))
        gate=np.zeros((NEXP,FFE,H),dtype=np.float32)
        up=np.zeros((NEXP,FFE,H),dtype=np.float32)
        down=np.zeros((NEXP,H,FFE),dtype=np.float32)
        put(b+"ffn_gate_inp.weight", np.zeros((NEXP,H),dtype=np.float32))
        put(b+"exp_probs_b.bias", np.zeros(NEXP,dtype=np.float32))
        put(b+"ffn_gate_tid2eid.weight", np.zeros((VOC,NUSE),dtype=np.int32))
    else:
        put(b+"ffn_gate_shexp.weight", g(p+"mlp.shared_experts.gate_proj.weight"))
        put(b+"ffn_up_shexp.weight", g(p+"mlp.shared_experts.up_proj.weight"))
        put(b+"ffn_down_shexp.weight", g(p+"mlp.shared_experts.down_proj.weight"))
        put(b+"ffn_gate_inp.weight", g(p+"mlp.gate.weight"))
        put(b+"exp_probs_b.bias", g(p+"mlp.gate.e_score_correction_bias"))
        gate=np.stack([g(p+f"mlp.experts.{e}.gate_proj.weight") for e in range(NEXP)])
        up=np.stack([g(p+f"mlp.experts.{e}.up_proj.weight") for e in range(NEXP)])
        down=np.stack([g(p+f"mlp.experts.{e}.down_proj.weight") for e in range(NEXP)])
    put(b+"ffn_gate_exps.weight", gate)
    put(b+"ffn_up_exps.weight", up)
    put(b+"ffn_down_exps.weight", down)

# ---- serialize ----
def ggml_type(arr): return GGML_F16
names=list(tensors.keys())
tinfo=b""; data=b""; ALIGN=32
offset=0
blobs=[]
for nm in names:
    arr,gt,q8shape=tensors[nm]
    shape=q8shape if q8shape is not None else list(arr.shape)
    dims=list(shape)[::-1]  # GGUF stores dims in reverse (ne0 fastest)
    if not dims: dims=[1]
    tinfo+=s(nm.encode())+struct.pack("<I",len(dims))
    for d in dims: tinfo+=struct.pack("<Q",d)
    tinfo+=struct.pack("<I",gt)+struct.pack("<Q",offset)
    if gt==GGML_Q8_0: raw=quant_q8_0(arr)
    elif gt==GGML_Q4_K: raw=quant_q4_K(arr)
    else: raw=arr.tobytes()
    pad=(-len(raw))%ALIGN
    blobs.append(raw+b"\x00"*pad)
    offset+=len(raw)+pad

header=struct.pack("<IIQQ",GGUF_MAGIC,GGUF_VER,len(names),len(meta))
metab=b"".join(meta)
body=header+metab+tinfo
pad=(-len(body))%ALIGN
body+=b"\x00"*pad
with open(OUT,"wb") as f:
    f.write(body)
    for blob in blobs: f.write(blob)
print(f"wrote {OUT}: {len(names)} tensors, {len(meta)} meta kv, hidden={H} layers={NL} experts={NEXP}/{NUSE}")
