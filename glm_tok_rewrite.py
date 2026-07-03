#!/usr/bin/env python3
"""
Rewrite ONLY the tokenizer metadata of an existing DS4-layout GGUF in place of a
full reconvert. Replaces tokenizer.ggml.tokens / tokenizer.ggml.merges with the
REAL GLM-5.2 tokenizer (from tokenizer.json) and copies the (huge) tensor data
section verbatim.

Why this is safe/fast: GGUF tensor-info `offset` fields are RELATIVE to the start
of the data section (the aligned region after the header+KV+tensor-info block).
Swapping metadata only changes the size of that prefix; the relative tensor
offsets are unchanged, so we keep the tensor-info bytes verbatim and stream-copy
the entire data section. ~minutes (disk-bound) instead of ~1.5h reconvert.

Usage:
  python3 glm_tok_rewrite.py IN.gguf /root/glm52-hf OUT.gguf
"""
import json, os, struct, sys

IN, HF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ALIGN = 32

# ---- GGUF value-type sizes ----
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
_FIXED = {T_U8:1,T_I8:1,T_U16:2,T_I16:2,T_U32:4,T_I32:4,T_F32:4,T_BOOL:1,T_U64:8,T_I64:8,T_F64:8}

mm = open(IN, "rb")
buf = mm  # read incrementally

def rd(n):
    b = mm.read(n)
    if len(b) != n: raise EOFError
    return b
def ru32(): return struct.unpack("<I", rd(4))[0]
def ru64(): return struct.unpack("<Q", rd(8))[0]

magic = ru32(); ver = ru32()
assert magic == 0x46554747 and ver == 3, f"bad gguf magic/ver {magic:x}/{ver}"
n_tensors = ru64(); n_kv = ru64()

def skip_value(vt):
    """Advance past a value of type vt (cursor already after the type tag)."""
    if vt in _FIXED:
        rd(_FIXED[vt]); return
    if vt == T_STR:
        ln = ru64(); rd(ln); return
    if vt == T_ARR:
        at = ru32(); cnt = ru64()
        if at in _FIXED:
            rd(_FIXED[at]*cnt); return
        if at == T_STR:
            for _ in range(cnt):
                ln = ru64(); rd(ln)
            return
        raise ValueError(f"nested array type {at}")
    raise ValueError(f"unknown value type {vt}")

# ---- parse KVs, recording raw byte spans; drop the two tokenizer keys ----
kept_kv = []           # list of raw bytes for KVs we keep verbatim
DROP = {b"tokenizer.ggml.tokens", b"tokenizer.ggml.merges"}
for _ in range(n_kv):
    start = mm.tell()
    klen = ru64(); key = rd(klen)
    vt = ru32()
    skip_value(vt)
    end = mm.tell()
    mm.seek(start); raw = rd(end - start);
    if key not in DROP:
        kept_kv.append(raw)

# ---- tensor-info block: copy verbatim; just advance the cursor ----
ti_start = mm.tell()
for _ in range(n_tensors):
    nlen = ru64(); rd(nlen)
    ndims = ru32(); rd(8*ndims)
    ru32()           # ggml type
    ru64()           # offset (relative to data section)
ti_end = mm.tell()
mm.seek(ti_start); tensor_info = rd(ti_end - ti_start)

# data section starts at next ALIGN boundary after ti_end
data_start = (ti_end + ALIGN - 1) // ALIGN * ALIGN
data_size = os.path.getsize(IN) - data_start
print(f"[rewrite] n_kv={n_kv} n_tensors={n_tensors} ti_bytes={len(tensor_info)} "
      f"data_start={data_start} data_size={data_size/1e9:.1f}GB", flush=True)

# ---- build real GLM tokens + merges ----
tj = json.load(open(os.path.join(HF, "tokenizer.json")))
cfg = json.load(open(os.path.join(HF, "config.json")))
VOC = cfg["vocab_size"]
vocab = tj["model"]["vocab"]                      # {tokstr: id}
toks = ["<unused_%d>" % i for i in range(VOC)]
maxid = -1
for tokstr, tid in vocab.items():
    if 0 <= tid < VOC:
        toks[tid] = tokstr; maxid = max(maxid, tid)
for at in tj.get("added_tokens", []):
    tid = at["id"]
    if 0 <= tid < VOC:
        toks[tid] = at["content"]; maxid = max(maxid, tid)
merges_raw = tj["model"].get("merges", [])
merges = []
for mrg in merges_raw:
    merges.append(mrg if isinstance(mrg, str) else (mrg[0] + " " + mrg[1]))
print(f"[rewrite] tokens={len(toks)} (maxid={maxid}) merges={len(merges)}", flush=True)

def s(b): return struct.pack("<Q", len(b)) + b
def kv_arr_str(k, vals):
    out = bytearray()
    out += s(k.encode()); out += struct.pack("<I", T_ARR)
    out += struct.pack("<I", T_STR); out += struct.pack("<Q", len(vals))
    for x in vals: out += s(x.encode("utf-8"))
    return bytes(out)

new_kv = list(kept_kv)
new_kv.append(kv_arr_str("tokenizer.ggml.tokens", toks))
new_kv.append(kv_arr_str("tokenizer.ggml.merges", merges))
new_n_kv = len(new_kv)

header = struct.pack("<IIQQ", 0x46554747, 3, n_tensors, new_n_kv)
body = header + b"".join(new_kv) + tensor_info
pad = (-len(body)) % ALIGN
body += b"\x00" * pad

# ---- write: new header/meta/tensor-info + verbatim data copy ----
with open(OUT, "wb") as out:
    out.write(body)
    mm.seek(data_start)
    CHUNK = 256 * 1024 * 1024
    copied = 0
    while True:
        b = mm.read(CHUNK)
        if not b: break
        out.write(b); copied += len(b)
        if copied % (4*1024*1024*1024) < CHUNK:
            print(f"  copied {copied/1e9:.0f}/{data_size/1e9:.0f} GB", flush=True)
print(f"[rewrite] wrote {OUT} (new_n_kv={new_n_kv}, data {copied/1e9:.1f}GB copied)", flush=True)
