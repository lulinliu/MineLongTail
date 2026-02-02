#!/usr/bin/env bash
set -euo pipefail

MODEL="/DATA2/lulin2/hf_cache/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"
VIDS="/DATA2/lulin2/ood/PhysicalAI-Autonomous-Vehicles/camera/ALL_mp4_symlinks"
# Only long-tail outputs now; keep a fresh folder to avoid mixing with old lighting files
OUT="/DATA2/lulin2/ood/qwen3vl_out_front_wide_split_longtail"

mkdir -p "$OUT"

# 可选：把控制台输出也保存一份 log
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node=4 /DATA2/lulin2/ood/PhysicalAI-Autonomous-Vehicles/qwen_filter.py \
  --model_path "$MODEL" \
  --video_dir "$VIDS" \
  --out_dir "$OUT" \
  --fps 1 \
  --max_frames 64 \
  --max_new_tokens 512 \
  --dtype bf16 \
  --attn_impl auto \
  2>&1 | tee "$LOG"
