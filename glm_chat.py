#!/usr/bin/env python3
"""
GLM-5.2 on DS4 — real text chat WITHOUT a native GGUF tokenizer.

Bridges the GLM tokenizer (HF transformers) with DS4's `--raw-ids` debug path:
  user text -> GLM apply_chat_template -> token ids -> ds4 --raw-ids -> generated
  ids -> GLM decode -> assistant text.

Works on the 2-GPU fully-resident coordinator (fast) or a single-GPU streaming
run (slow). It launches one ds4 generation per turn; for the resident 2-GPU path
the worker must already be up (see run_glm_res2.sh) and listening, and this script
runs the coordinator. For a quick single-process test use --solo.

DS4 prints generated tokens as `<tokN>` placeholders where the real id is N+7
(7 DeepSeek special tokens precede the placeholder block); we parse those back.

Usage:
  # 2-GPU resident (worker already running from run_glm_res2.sh worker line):
  python3 glm_chat.py --hf /root/glm52-hf -m /home/glm52_ds4_q2k.gguf \
      --coordinator-args "--role coordinator --layers 0:36 --listen 0.0.0.0 28100" \
      --cache-gb 122 --gpu 0 -n 64 "What is the capital of France?"
"""
import argparse, os, re, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/root/glm52-hf", help="GLM HF dir (tokenizer)")
    ap.add_argument("-m", "--model", default="/home/glm52_ds4_q2k.gguf")
    ap.add_argument("--ds4", default="/root/ds4/ds4")
    ap.add_argument("--coordinator-args", default="",
                    help="extra ds4 args for the distributed coordinator role")
    ap.add_argument("--solo", action="store_true",
                    help="single-process run (no distributed); for small models")
    ap.add_argument("--cache-gb", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("-n", "--n-predict", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--system", default=None)
    ap.add_argument("--no-think", action="store_true",
                    help="close <think> immediately (skip reasoning)")
    ap.add_argument("prompt", nargs="+")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf, trust_remote_code=True)

    msgs = []
    if a.system:
        msgs.append({"role": "system", "content": a.system})
    msgs.append({"role": "user", "content": " ".join(a.prompt)})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if a.no_think:
        # GLM template opens reasoning with <think>; append </think> to skip it.
        end_think = tok.convert_tokens_to_ids("</think>")
        if end_think is not None and end_think >= 0:
            ids = list(ids) + [end_think]
    print(f"[glm_chat] prompt ids ({len(ids)}): {ids[:24]}{'...' if len(ids)>24 else ''}",
          file=sys.stderr)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    if a.cache_gb:
        env["DS4_CUDA_WEIGHT_CACHE_LIMIT_GB"] = str(a.cache_gb)
    env.setdefault("DS4_LOCK_FILE", "/tmp/ds4_chat.lock")

    cmd = [a.ds4, "-m", a.model, "--raw-ids", ",".join(str(i) for i in ids),
           "-n", str(a.n_predict), "--temp", str(a.temp), "-c", "512"]
    if not a.solo and a.coordinator_args:
        cmd += a.coordinator_args.split()
    print(f"[glm_chat] running: {' '.join(cmd)}", file=sys.stderr)

    out = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stderr.write(out.stderr[-2000:] if out.stderr else "")

    # DS4 prints generated tokens as <tokN> (placeholder id N -> real id N+7).
    gen_ids = [int(m) + 7 for m in re.findall(r"<tok(\d+)>", out.stdout)]
    # Some builds print raw ids after a marker; also try a "generated ids:" line.
    if not gen_ids:
        mm = re.search(r"generated ids?:?\s*([0-9,\s]+)", out.stdout)
        if mm:
            gen_ids = [int(x) for x in re.findall(r"\d+", mm.group(1))]
    if not gen_ids:
        print("[glm_chat] no generated ids parsed; raw stdout below:", file=sys.stderr)
        print(out.stdout)
        return 2

    text = tok.decode(gen_ids, skip_special_tokens=True)
    print("\n=== ASSISTANT ===")
    print(text)
    return 0

if __name__ == "__main__":
    sys.exit(main())
