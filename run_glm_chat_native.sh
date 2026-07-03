#!/bin/bash
# Native-tokenizer chat: GLM-5.2 744B, 2x H200 resident, real text in/out (no python bridge).
M=/home/glm52_ds4_q2k_tok.gguf
PROMPT="${1:-What is the capital of France? Answer in one sentence.}"
NPRED="${2:-80}"
cd /root/ds4
pkill -9 -x ds4 2>/dev/null; sleep 2; rm -f /tmp/ds4_w.lock /tmp/ds4_c.lock
CUDA_VISIBLE_DEVICES=1 DS4_LOCK_FILE=/tmp/ds4_w.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=138 DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=8192 \
  ./ds4 --role worker --layers 37:output --coordinator 127.0.0.1 28100 -m $M -c 512 > /root/chat_worker.log 2>&1 &
WPID=$!
sleep 50
CUDA_VISIBLE_DEVICES=0 DS4_LOCK_FILE=/tmp/ds4_c.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=134 DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=20480 \
  ./ds4 --role coordinator --layers 0:36 --listen 0.0.0.0 28100 -m $M -p "$PROMPT" -n "$NPRED" --temp 0 -c 512 > /root/chat_coord.log 2>&1
echo "COORD_RC=$?"
kill -9 $WPID 2>/dev/null
