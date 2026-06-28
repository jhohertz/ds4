#!/bin/bash
# Launch a DS4 distributed GLM serve (worker + coordinator-API) as ONE tracked
# process for the Arma Gateway. The gateway spawns this; it backgrounds the
# worker and execs the coordinator-server in the foreground so the gateway tracks
# the coordinator PID (kill the process group to stop both).
#
# Args (from _build_ds4_cmd):
#   $1 ds4_server binary   $2 gguf   $3 coord_layers (0:36)  $4 worker_layers (37:output)
#   $5 api_host  $6 api_port  $7 coord_listen_port  $8 ctx
#   $9 coord_gpu  $10 worker_gpu  $11 coord_reserve_mb  $12 worker_reserve_mb  $13 weight_cache_gb
set -euo pipefail
DS4=$1; M=$2; CL=$3; WL=$4; HOST=$5; PORT=$6; CPORT=$7; CTX=$8
CGPU=$9; WGPU=${10}; CRES=${11}; WRES=${12}; WCACHE=${13}
LOCKW=/tmp/ds4_gw_w.lock; LOCKC=/tmp/ds4_gw_c.lock
rm -f "$LOCKW" "$LOCKC"

# Single-process mode (tensor_parallel_size==1, e.g. DeepSeek-V4-Flash 81GB on ONE
# H200): no worker/coordinator split. The gateway passes worker_layers="none".
# CUDA_VISIBLE_DEVICES is already set by the caller (SLURM shard allocation, or the
# gateway's _spawn_process) so we don't touch it.
if [ "$WL" = "none" ]; then
  exec env DS4_LOCK_FILE=$LOCKC \
    DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$WCACHE DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=$CRES \
    "$DS4" -m "$M" -c "$CTX" --host "$HOST" --port "$PORT" --prefill-chunk 512
fi

# Under SLURM the 2 GPUs come from the allocation (CUDA_VISIBLE_DEVICES set by
# SLURM), not the hardcoded 0/1. Bind worker -> first allocated, coordinator ->
# second. Outside SLURM, use the passed coordinator_gpu/worker_gpu.
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
    WGPU="${GPUS[0]}"; CGPU="${GPUS[1]:-${GPUS[0]}}"
fi

# worker (background, its own GPU)
CUDA_VISIBLE_DEVICES=$WGPU DS4_LOCK_FILE=$LOCKW \
  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$WCACHE DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=$WRES \
  "$DS4" --role worker --layers "$WL" --coordinator 127.0.0.1 "$CPORT" -m "$M" -c "$CTX" --prefill-chunk 512 &
WPID=$!
# stop the worker when this script (and thus the coordinator) dies
trap 'kill -9 $WPID 2>/dev/null || true' EXIT

# give the worker a head start on the model load (both read the same 252GB)
sleep 5

# coordinator-server in the FOREGROUND = the process the gateway tracks.
# Serves OpenAI /v1 on $HOST:$PORT; coordinates workers on :$CPORT.
exec env CUDA_VISIBLE_DEVICES=$CGPU DS4_LOCK_FILE=$LOCKC \
  DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=$WCACHE DS4_CUDA_Q8_F16_CACHE_RESERVE_MB=$CRES \
  "$DS4" --role coordinator --layers "$CL" --listen 0.0.0.0 "$CPORT" \
         --host "$HOST" --port "$PORT" -m "$M" -c "$CTX" --prefill-chunk 512
