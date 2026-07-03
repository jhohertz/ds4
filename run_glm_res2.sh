#!/bin/bash
# GLM-5.2 q2 (252GB) fully resident across 2x H200, no SSD streaming.
# Both nodes MUST load the same -m model (a coordinator without -m falls back to
# the default ds4flash.gguf -> "model id mismatch").
# On >=112 GiB cards the optional q8->fp16 dequant cache reserves only 512 MiB free
# by default and starves the prefill graph -> OOM. Reserve real headroom with
# DS4_CUDA_Q8_F16_CACHE_RESERVE_MB. Coordinator also holds output head + 154880-vocab
# logits, so it gets the bigger reserve and the smaller layer slice.
M=/home/glm52_ds4_q2k.gguf
cd /root/ds4
pkill -9 -x ds4 2>/dev/null; sleep 2
rm -f /tmp/ds4_w.lock /tmp/ds4_c.lock
CUDA_VISIBLE_DEVICES=1 DS4_LOCK_FILE=/tmp/ds4_w.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=138 DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=8192 ./ds4 --role worker --layers 37:output --coordinator 127.0.0.1 28100 -m $M -c 512 > /root/res_worker.log 2>&1 &
WPID=$!
sleep 50
CUDA_VISIBLE_DEVICES=0 DS4_LOCK_FILE=/tmp/ds4_c.lock DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=134 DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=20480 ./ds4 --role coordinator --layers 0:36 --listen 0.0.0.0 28100 -m $M --raw-ids "154822,154824,154826,25062,287,29905,371,25,7487,154827,785,6722,315,9621,374,154828,154841" -n 12 --temp 0 -c 512 > /root/res_coord.log 2>&1
echo "COORD_RC=$?"
kill -9 $WPID 2>/dev/null
