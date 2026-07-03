#!/usr/bin/env python3
# Streaming GlmMoeDsa (GLM-5.2) -> DS4-layout GGUF converter for the REAL 744B model.
# Same absorbed-MLA recipe as glm5_to_ds4.py (validated to corr 0.9994 vs HF), but:
#   - loads tensors LAZILY from sharded safetensors (model.safetensors.index.json),
#   - two-pass serialize (compute header offsets from shapes, then stream tensor data),
# so peak RAM stays ~one tensor-group (never the full 1.5TB).
import json, struct, sys, os, numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/glm52-hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/glm52_ds4.gguf"

cfg = json.load(open(os.path.join(SRC, "config.json")))
H=cfg["hidden_size"]; NL=cfg["num_hidden_layers"]; NH=cfg["num_attention_heads"]
KVL=cfg["kv_lora_rank"]; QL=cfg["q_lora_rank"]; ROPE=cfg["qk_rope_head_dim"]
NOPE=cfg["qk_nope_head_dim"]; VHD=cfg["v_head_dim"]; NEXP=cfg["n_routed_experts"]
NUSE=cfg["num_experts_per_tok"]; FFE=cfg["moe_intermediate_size"]; VOC=cfg["vocab_size"]
IXH=cfg["index_n_heads"]; IXD=cfg["index_head_dim"]; IXK=cfg["index_topk"]
SCALE=cfg.get("routed_scaling_factor",2.5); EPS=cfg.get("rms_norm_eps",1e-5)
FKD=cfg.get("first_k_dense_replace",3); QKH=NOPE+ROPE
NHD=KVL+ROPE; NLO=VHD; NOUTG=NH
ROPE_THETA=float(cfg.get("rope_parameters",{}).get("rope_theta", cfg.get("rope_theta",10000.0)))
SCALE_FOLD=(NHD/QKH)**0.5
TIED=cfg.get("tie_word_embeddings", False)

# ---- lazy sharded safetensors ----
idx_path=os.path.join(SRC,"model.safetensors.index.json")
if os.path.exists(idx_path):
    WMAP=json.load(open(idx_path))["weight_map"]
else:
    WMAP={k:"model.safetensors" for k in []}  # single-file fallback handled below
_HDRS={}   # filename -> (header dict, data_offset)
def _hdr(fn):
    if fn not in _HDRS:
        f=open(os.path.join(SRC,fn),"rb"); n=struct.unpack("<Q",f.read(8))[0]
        h=json.loads(f.read(n)); _HDRS[fn]=(h,8+n); f.close()
    return _HDRS[fn]
def _shard_of(name):
    if WMAP: return WMAP[name]
    return "model.safetensors"
def g(name):
    fn=_shard_of(name); h,base=_hdr(fn); v=h[name]
    dt=v["dtype"]; sh=v["shape"]; a,b=v["data_offsets"]
    with open(os.path.join(SRC,fn),"rb") as f:
        f.seek(base+a); raw=f.read(b-a)
    npdt={"BF16":np.uint16,"F16":np.float16,"F32":np.float32,"I32":np.int32,"I64":np.int64,"U8":np.uint8}[dt]
    arr=np.frombuffer(raw,dtype=npdt)
    if sh: arr=arr.reshape(sh)
    if dt=="BF16": arr=(arr.astype(np.uint32)<<16).view(np.float32)
    return np.asarray(arr,dtype=np.float32)
def has(name):
    return (name in WMAP) if WMAP else (name in _hdr("model.safetensors")[0])
def gshape(name):
    fn=_shard_of(name); h,_=_hdr(fn); return tuple(h[name]["shape"])

# ---- quant (same as validated converter) ----
def quant_q4_K(arr):
    # Fully vectorized Q4_K (numpy), same math as the validated loop version.
    a=np.ascontiguousarray(arr.astype(np.float32)); ne0=a.shape[-1]
    assert ne0%256==0
    N=a.size//256
    x=a.reshape(N,8,32)
    lo=x.min(2); hi=x.max(2)                                   # [N,8]
    dblk=np.maximum((hi-lo)/15.0,1e-12)
    q=np.clip(np.round((x-lo[:,:,None])/dblk[:,:,None]),0,15).astype(np.uint8)  # [N,8,32]
    Bblk=np.clip(-lo,0,None)
    superd=np.maximum(dblk.max(1)/63.0,1e-12); superm=np.maximum(Bblk.max(1)/63.0,1e-12)  # [N]
    sc=np.clip(np.round(dblk/superd[:,None]),0,63).astype(np.uint32)            # [N,8]
    mn=np.clip(np.round(Bblk/superm[:,None]),0,63).astype(np.uint32)
    rec=np.empty((N,144),dtype=np.uint8)
    rec[:,0:2]=superd.astype(np.float16).view(np.uint8).reshape(N,2)
    rec[:,2:4]=superm.astype(np.float16).view(np.uint8).reshape(N,2)
    sca=np.empty((N,12),dtype=np.uint32)
    sca[:,0:4]=(sc[:,0:4]&63)|((sc[:,4:8]>>4)<<6)
    sca[:,4:8]=(mn[:,0:4]&63)|((mn[:,4:8]>>4)<<6)
    sca[:,8:12]=(sc[:,4:8]&0xF)|((mn[:,4:8]&0xF)<<4)
    rec[:,4:16]=sca.astype(np.uint8)
    even=q[:,0::2,:]; odd=q[:,1::2,:]                          # [N,4,32]
    qs=(even|(odd<<4)).reshape(N,128)
    rec[:,16:144]=qs
    return rec.tobytes()
# Q2_K packing index tables (output pos -> qs byte + bit shift), built once.
_Q2K_BYTE=[]; _Q2K_SHIFT=[]
for _p in range(256):
    _g=_p//128; _loc=_p%128; _bi=_loc//16; _l=_loc%16
    _Q2K_SHIFT.append((_bi//2)*2); _Q2K_BYTE.append(_g*32 + (_bi%2)*16 + _l)
_Q2K_BYTE=np.array(_Q2K_BYTE); _Q2K_SHIFT=np.array(_Q2K_SHIFT)
def quant_q2_K(arr):
    # Vectorized Q2_K: superblock 256 = 16 sub-blocks of 16. 84 bytes/superblock (~2.625 bit).
    a=np.ascontiguousarray(arr.astype(np.float32)); ne0=a.shape[-1]; assert ne0%256==0
    N=a.size//256
    x=a.reshape(N,16,16)
    mn=x.min(2); mx=x.max(2)                                  # [N,16]
    scale=np.maximum((mx-mn)/3.0,1e-12)
    q=np.clip(np.round((x-mn[:,:,None])/scale[:,:,None]),0,3).astype(np.uint8)  # [N,16,16]
    Bmin=np.clip(-mn,0,None)
    superd=np.maximum(scale.max(1)/15.0,1e-12); superm=np.maximum(Bmin.max(1)/15.0,1e-12)  # [N]
    sc4=np.clip(np.round(scale/superd[:,None]),0,15).astype(np.uint8)   # [N,16]
    m4 =np.clip(np.round(Bmin/superm[:,None]),0,15).astype(np.uint8)
    rec=np.empty((N,84),dtype=np.uint8)
    rec[:,0:16]=sc4|(m4<<4)                                   # scales: 4-bit scale | 4-bit min
    qf=q.reshape(N,256)
    qs=np.zeros((N,64),dtype=np.uint8)
    for p in range(256):
        qs[:,int(_Q2K_BYTE[p])]|=(qf[:,p]<<int(_Q2K_SHIFT[p])).astype(np.uint8)
    rec[:,16:80]=qs
    rec[:,80:82]=superd.astype(np.float16).view(np.uint8).reshape(N,2)
    rec[:,82:84]=superm.astype(np.float16).view(np.uint8).reshape(N,2)
    return rec.tobytes()
def quant_q8_0(arr):
    a=np.ascontiguousarray(arr.astype(np.float32)); ne0=a.shape[-1]; assert ne0%32==0
    rows=a.reshape(-1,ne0); nb=ne0//32; blk=rows.reshape(rows.shape[0],nb,32)
    amax=np.max(np.abs(blk),2); d=amax/127.0; dz=np.where(d>0,d,1.0)
    q=np.round(blk/dz[:,:,None]).clip(-127,127).astype(np.int8); dh=d.astype(np.float16)
    nr=rows.shape[0]; rec=np.empty((nr,nb,34),dtype=np.uint8)
    rec[:,:,0:2]=dh.view(np.uint8).reshape(nr,nb,2); rec[:,:,2:34]=q.view(np.uint8)
    return rec.tobytes()

GGUF_MAGIC=0x46554747; GGUF_VER=3
T_U32=4;T_F32=6;T_BOOL=7;T_STR=8;T_ARR=9;T_U64=10
GGML_F32=0;GGML_F16=1;GGML_Q8_0=8;GGML_Q2_K=10;GGML_Q4_K=12;GGML_I32=26
EXPERT_GT = GGML_Q2_K if os.environ.get("EXPERT_Q2K") else GGML_Q4_K  # routed-expert quant
def s(b): return struct.pack("<Q",len(b))+b
def kv_str(k,v): return s(k.encode())+struct.pack("<I",T_STR)+s(v.encode())
def kv_u32(k,v): return s(k.encode())+struct.pack("<I",T_U32)+struct.pack("<I",v)
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

meta=[kv_str("general.architecture","deepseek4"), kv_str("general.name","GLM 5.2")]
def m_u32(k,v): meta.append(kv_u32("deepseek4."+k,v))
def m_f32(k,v): meta.append(kv_f32("deepseek4."+k,v))
m_u32("block_count",NL); m_u32("embedding_length",H); m_u32("vocab_size",VOC)
m_u32("attention.head_count",NH); m_u32("attention.key_length",NHD); m_u32("attention.head_count_kv",1)
m_u32("attention.value_length",KVL); m_u32("rope.dimension_count",ROPE)
m_u32("attention.output_group_count",NOUTG); m_u32("attention.q_lora_rank",QL); m_u32("attention.output_lora_rank",NLO)
m_u32("expert_count",NEXP); m_u32("expert_used_count",NUSE); m_u32("expert_feed_forward_length",FFE)
m_u32("expert_shared_count",1); m_u32("hash_layer_count",FKD)
m_u32("expert_group_count",0); m_u32("expert_group_used_count",0); m_u32("attention.sliding_window",128)
m_u32("attention.indexer.head_count",IXH); m_u32("attention.indexer.key_length",IXD); m_u32("attention.indexer.top_k",IXK)
m_u32("hyper_connection.count",1); m_u32("hyper_connection.sinkhorn_iterations",0)
meta.append(kv_arr_u32("deepseek4.attention.compress_ratios",[0]*NL))
m_f32("rope.freq_base",ROPE_THETA); m_f32("attention.compress_rope_freq_base",160000.0)
m_f32("expert_weights_scale",SCALE); m_f32("attention.layer_norm_rms_epsilon",EPS); m_f32("hyper_connection.epsilon",1.0e-6)
meta.append(kv_arr_f32("deepseek4.swiglu_clamp_exp",[10.0]*NL))
meta.append(kv_bool("deepseek4.expert_weights_norm",True))
# Real GLM tokenizer from tokenizer.json (DS4's joyai-llm byte-level BPE reproduces GLM's
# tokenization exactly — verified --dump-tokens == HF AutoTokenizer on diverse strings).
# Falls back to a placeholder vocab only if tokenizer.json is absent.
_tjp=os.path.join(SRC,"tokenizer.json")
if os.path.exists(_tjp):
    _tj=json.load(open(_tjp)); _vmap=_tj["model"]["vocab"]
    toks=["<unused_%d>"%i for i in range(VOC)]
    for _s,_id in _vmap.items():
        if 0<=_id<VOC: toks[_id]=_s
    for _at in _tj.get("added_tokens",[]):
        if 0<=_at["id"]<VOC: toks[_at["id"]]=_at["content"]
    merges=[(m if isinstance(m,str) else (m[0]+" "+m[1])) for m in _tj["model"].get("merges",[])]
    print(f"  tokenizer: {len(toks)} tokens, {len(merges)} merges (real GLM)",flush=True)
else:
    SPECIALS=["<｜begin▁of▁sentence｜>","<｜end▁of▁sentence｜>","<｜User｜>","<｜Assistant｜>","<think>","</think>","｜DSML｜"]
    toks=SPECIALS+[f"<tok{i}>" for i in range(VOC-len(SPECIALS))]; merges=[]
    print("  tokenizer: PLACEHOLDER (no tokenizer.json)",flush=True)
meta.append(kv_arr_str("tokenizer.ggml.tokens",toks)); meta.append(kv_arr_str("tokenizer.ggml.merges",merges))

NHC=1; HC_DIM=H*NHC; HC_MIX=2*NHC+NHC*NHC
F32S={"output_hc_base.weight","output_hc_scale.weight","output_norm.weight","hc_attn_scale.weight","hc_attn_base.weight",
      "attn_norm.weight","attn_q_a_norm.weight","attn_kv_a_norm.weight","attn_sinks.weight","hc_ffn_scale.weight",
      "hc_ffn_base.weight","ffn_norm.weight","exp_probs_b.bias"}
Q8S={"attn_q_a.weight","attn_q_b.weight","attn_kv.weight","attn_output_a.weight","attn_output_b.weight",
     "ffn_gate_shexp.weight","ffn_up_shexp.weight","ffn_down_shexp.weight","output.weight"}
Q4S={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}
def _suf(n): return n.split(".",2)[-1] if n.startswith("blk.") else n
def gtype(name):
    suf=_suf(name)
    if suf in F32S: return GGML_F32
    if suf in Q8S: return GGML_Q8_0
    if suf in Q4S: return EXPERT_GT
    if suf=="ffn_gate_tid2eid.weight": return GGML_I32
    return GGML_F16
def tbytes(gt,shape):
    ne=1
    for d in shape: ne*=d
    if gt==GGML_F32: return ne*4
    if gt==GGML_F16: return ne*2
    if gt==GGML_I32: return ne*4
    if gt==GGML_Q8_0: return ne//32*34
    if gt==GGML_Q2_K: return ne//256*84
    if gt==GGML_Q4_K: return ne//256*144
def encode(gt,arr):
    if gt==GGML_Q8_0: return quant_q8_0(arr)
    if gt==GGML_Q2_K: return quant_q2_K(arr)
    if gt==GGML_Q4_K: return quant_q4_K(arr)
    if gt==GGML_F32: return np.ascontiguousarray(arr.astype(np.float32)).tobytes()
    if gt==GGML_I32: return np.ascontiguousarray(arr.astype(np.int32)).tobytes()
    return np.ascontiguousarray(arr.astype(np.float16)).tobytes()

# ---- spec list: (ds4_name, shape, producer) -- producers are LAZY ----
SPECS=[]
def spec(name, shape, producer): SPECS.append((name, tuple(int(x) for x in shape), producer))

spec("token_embd.weight",(VOC,H), lambda: g("model.embed_tokens.weight"))
spec("output_norm.weight",(H,), lambda: g("model.norm.weight"))
spec("output.weight",(VOC,H), (lambda: g("model.embed_tokens.weight")) if (TIED or not has("lm_head.weight")) else (lambda: g("lm_head.weight")))
spec("output_hc_base.weight",(NHC,), lambda: np.ones(NHC,np.float32))
spec("output_hc_fn.weight",(NHC,HC_DIM), lambda: np.zeros((NHC,HC_DIM),np.float32))
spec("output_hc_scale.weight",(1,), lambda: np.ones(1,np.float32))

def mk_q_b(p):
    def f():
        qb=g(p+"self_attn.q_b_proj.weight").reshape(NH,QKH,QL)
        kvb=g(p+"self_attn.kv_b_proj.weight").reshape(NH,NOPE+VHD,KVL); Wk=kvb[:,:NOPE,:]
        out=np.zeros((NH*NHD,QL),np.float32)
        for hh in range(NH):
            out[hh*NHD:hh*NHD+KVL]=Wk[hh].T@qb[hh,:NOPE]
            out[hh*NHD+KVL:(hh+1)*NHD]=qb[hh,NOPE:]
        return out*SCALE_FOLD
    return f
def mk_out_a(p):
    def f():
        kvb=g(p+"self_attn.kv_b_proj.weight").reshape(NH,NOPE+VHD,KVL); Wv=kvb[:,NOPE:,:]
        out=np.zeros((NOUTG*NLO, NHD*(NH//NOUTG)),np.float32)
        for hh in range(NH): out[hh*NLO:(hh+1)*NLO,:KVL]=Wv[hh]
        return out
    return f

for il in range(NL):
    p=f"model.layers.{il}."; b=f"blk.{il}."
    spec(b+"attn_norm.weight",(H,), (lambda p=p: g(p+"input_layernorm.weight")))
    spec(b+"ffn_norm.weight",(H,), (lambda p=p: g(p+"post_attention_layernorm.weight")))
    spec(b+"attn_q_a.weight",(QL,H), (lambda p=p: g(p+"self_attn.q_a_proj.weight")))
    spec(b+"attn_q_a_norm.weight",(QL,), (lambda p=p: g(p+"self_attn.q_a_layernorm.weight")))
    spec(b+"attn_q_b.weight",(NH*NHD,QL), mk_q_b(p))
    spec(b+"attn_kv.weight",(NHD,H), (lambda p=p: g(p+"self_attn.kv_a_proj_with_mqa.weight")))
    spec(b+"attn_kv_a_norm.weight",(NHD,), (lambda p=p: np.concatenate([g(p+"self_attn.kv_a_layernorm.weight"),np.ones(ROPE,np.float32)])))
    spec(b+"attn_sinks.weight",(NH,), lambda: np.full(NH,-1e30,np.float32))
    spec(b+"attn_output_a.weight",(NOUTG*NLO,NHD*(NH//NOUTG)), mk_out_a(p))
    spec(b+"attn_output_b.weight",(H,NH*VHD), (lambda p=p: g(p+"self_attn.o_proj.weight")))
    for t in ("hc_attn_base","hc_attn_scale","hc_ffn_base","hc_ffn_scale"):
        spec(b+t+".weight",(HC_MIX,), lambda: np.ones(HC_MIX,np.float32))
    for t in ("hc_attn_fn","hc_ffn_fn"):
        spec(b+t+".weight",(HC_MIX,HC_DIM), lambda: np.zeros((HC_MIX,HC_DIM),np.float32))
    if il < FKD:
        sg=gshape(p+"mlp.gate_proj.weight"); sd=gshape(p+"mlp.down_proj.weight")
        spec(b+"ffn_gate_shexp.weight",sg, (lambda p=p: g(p+"mlp.gate_proj.weight")))
        spec(b+"ffn_up_shexp.weight",gshape(p+"mlp.up_proj.weight"), (lambda p=p: g(p+"mlp.up_proj.weight")))
        spec(b+"ffn_down_shexp.weight",sd, (lambda p=p: g(p+"mlp.down_proj.weight")))
        spec(b+"ffn_gate_inp.weight",(NEXP,H), lambda: np.zeros((NEXP,H),np.float32))
        spec(b+"exp_probs_b.bias",(NEXP,), lambda: np.zeros(NEXP,np.float32))
        spec(b+"ffn_gate_tid2eid.weight",(VOC,NUSE), lambda: np.zeros((VOC,NUSE),np.int32))
        spec(b+"ffn_gate_exps.weight",(NEXP,FFE,H), lambda: np.zeros((NEXP,FFE,H),np.float32))
        spec(b+"ffn_up_exps.weight",(NEXP,FFE,H), lambda: np.zeros((NEXP,FFE,H),np.float32))
        spec(b+"ffn_down_exps.weight",(NEXP,H,FFE), lambda: np.zeros((NEXP,H,FFE),np.float32))
    else:
        eg=gshape(p+"mlp.experts.0.gate_proj.weight"); ed=gshape(p+"mlp.experts.0.down_proj.weight")
        spec(b+"ffn_gate_shexp.weight",gshape(p+"mlp.shared_experts.gate_proj.weight"), (lambda p=p: g(p+"mlp.shared_experts.gate_proj.weight")))
        spec(b+"ffn_up_shexp.weight",gshape(p+"mlp.shared_experts.up_proj.weight"), (lambda p=p: g(p+"mlp.shared_experts.up_proj.weight")))
        spec(b+"ffn_down_shexp.weight",gshape(p+"mlp.shared_experts.down_proj.weight"), (lambda p=p: g(p+"mlp.shared_experts.down_proj.weight")))
        spec(b+"ffn_gate_inp.weight",gshape(p+"mlp.gate.weight"), (lambda p=p: g(p+"mlp.gate.weight")))
        spec(b+"exp_probs_b.bias",gshape(p+"mlp.gate.e_score_correction_bias"), (lambda p=p: g(p+"mlp.gate.e_score_correction_bias")))
        spec(b+"ffn_gate_exps.weight",(NEXP,)+eg, (lambda p=p: np.stack([g(p+f"mlp.experts.{e}.gate_proj.weight") for e in range(NEXP)])))
        spec(b+"ffn_up_exps.weight",(NEXP,)+gshape(p+"mlp.experts.0.up_proj.weight"), (lambda p=p: np.stack([g(p+f"mlp.experts.{e}.up_proj.weight") for e in range(NEXP)])))
        spec(b+"ffn_down_exps.weight",(NEXP,)+ed, (lambda p=p: np.stack([g(p+f"mlp.experts.{e}.down_proj.weight") for e in range(NEXP)])))

# remove tid2eid for moe layers (DS4 only wants it on hash/dense layers) -- keep simple: drop where il>=FKD
SPECS=[sp for sp in SPECS if not (sp[0].endswith("ffn_gate_tid2eid.weight") and int(sp[0].split(".")[1])>=FKD)]

# ---- pass 1: header (offsets from shapes) ----
ALIGN=32
tinfo=b""; offset=0; plan=[]
for name,shape,prod in SPECS:
    gt=gtype(name); raw_len=tbytes(gt,shape); pad=(-raw_len)%ALIGN
    dims=list(shape)[::-1] or [1]
    tinfo+=s(name.encode())+struct.pack("<I",len(dims))
    for d in dims: tinfo+=struct.pack("<Q",d)
    tinfo+=struct.pack("<I",gt)+struct.pack("<Q",offset)
    plan.append((name,gt,prod,raw_len,pad)); offset+=raw_len+pad
header=struct.pack("<IIQQ",GGUF_MAGIC,GGUF_VER,len(SPECS),len(meta))
body=header+b"".join(meta)+tinfo; body+=b"\x00"*((-len(body))%ALIGN)

# ---- pass 2: stream tensor data ----
with open(OUT,"wb") as f:
    f.write(body)
    for i,(name,gt,prod,raw_len,pad) in enumerate(plan):
        raw=encode(gt,prod())
        assert len(raw)==raw_len, f"{name}: {len(raw)}!={raw_len}"
        f.write(raw); f.write(b"\x00"*pad)
        if i%50==0: print(f"  [{i}/{len(plan)}] {name}",flush=True)
print(f"wrote {OUT}: {len(SPECS)} tensors, {len(meta)} meta, H={H} L={NL} exp={NEXP}/{NUSE} tied={TIED}")
