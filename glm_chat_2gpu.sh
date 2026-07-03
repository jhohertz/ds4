#!/bin/bash
# Real-text chat with GLM-5.2 744B on the 2x H200 fully-resident path.
# Bridges the GLM tokenizer (HF transformers) to DS4's --raw-ids:
#   prompt -> apply_chat_template -> ids -> [worker 37:output | coordinator 0:36]
#   -> generated <tokN> ids -> GLM decode -> assistant text.
# One model load per turn (~3.5 min, inherent to the CLI one-shot path).
#
# Usage: ./glm_chat_2gpu.sh "What is the capital of France?" [n_predict]
set -e
PROMPT="${1:?usage: glm_chat_2gpu.sh \"your message\" [n_predict]}"
NPRED="${2:-64}"
M=/home/glm52_ds4_q2k.gguf
HF=/root/glm52-hf
PY=/root/glmref/bin/python3
DS4=/root/ds4/ds4
cd /root/ds4

# 1) prompt text -> GLM token ids (correct GLM tokenization, no pre-tok risk)
IDS=$("$PY" - "$HF" "$PROMPT" <<'PYEOF'
import sys
from transformers import AutoTokenizer
hf, prompt = sys.argv[1], sys.argv[2]
t = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
ids = t.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True)
print(",".join(str(i) for i in ids))
PYEOF
)
echo "[chat] prompt ids: $IDS" >&2

# 2) run the resident 2-GPU pipeline with those ids
pkill -9 -x ds4 2>/dev/null || true; sleep 2; rm -f /tmp/ds4_w.lock /tmp/ds4_c.lock
CUDA_VISIBLE_DEVICES=1 DS4_LOCK_FILE=/tmp/ds4_w.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=138 \
  DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=8192 \
  "$DS4" --role worker --layers 37:output --coordinator 127.0.0.1 28100 -m "$M" -c 512 \
  > /root/chat_worker.log 2>&1 &
WPID=$!
sleep 50
CUDA_VISIBLE_DEVICES=0 DS4_LOCK_FILE=/tmp/ds4_c.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=134 \
  DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=20480 \
  "$DS4" --role coordinator --layers 0:36 --listen 0.0.0.0 28100 -m "$M" \
  --raw-ids "$IDS" -n "$NPRED" --temp 0 -c 512 > /root/chat_coord.log 2>&1
kill -9 $WPID 2>/dev/null || true

# 3) generated <tokN> ids (real id = N+7) -> GLM decode -> text
"$PY" - "$HF" < /root/chat_coord.log <<'PYEOF'
import sys, re
from transformers import AutoTokenizer
hf = sys.argv[1]
data = sys.stdin.read()
ids = [int(m)+7 for m in re.findall(r"<tok(\d+)>", data)]
t = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
print("\n=== ASSISTANT ===")
print(t.decode(ids, skip_special_tokens=True) if ids else "[no tokens generated]")
PYEOF
grep -E "t/s|route ready|model id" /root/chat_coord.log >&2 || true
