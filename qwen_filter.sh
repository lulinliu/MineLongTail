#!/usr/bin/env bash
set -euo pipefail

MODEL="/scratch/10102/hh29499/MineLongTail/hf_cahce/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"
VIDS="/scratch/10102/hh29499/MineLongTail/longtail/PhysicalAI-Autonomous-Vehicles/camera/camera_front_wide_120fov/videos"
# Only long-tail outputs now; keep a fresh folder to avoid mixing with old lighting files
OUT="/scratch/10102/hh29499/MineLongTail/qwen3vl_out_front_wide_split_longtail"

mkdir -p "$OUT"

# 可选：把控制台输出也保存一份 log
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"

# Use MineLongTail prompt (9-class taxonomy)
torchrun --nproc_per_node=1 /DATA2/lulin2/ood/MineLongTail/qwen_filter.py \
  --model_path "$MODEL" \
  --video_dir "$VIDS" \
  --out_dir "$OUT" \
  --fps 1 \
  --max_frames 64 \
  --max_new_tokens 512 \
  --dtype bf16 \
  --attn_impl auto \
  2>&1 | tee "$LOG"
