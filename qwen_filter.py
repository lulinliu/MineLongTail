#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import imageio.v2 as iio
import imageio_ffmpeg
import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())

PROMPT_LONGTAIL = r"""
You are a data triage assistant for autonomous-driving video clips.

Goal:
1) Decide whether the clip contains a "long-tail" scenario.
2) If yes/unknown, assign one or more event labels from the 9-class taxonomy below.
3) Provide concise evidence tied to what is visible in the clip.

Definition of "long-tail":
A rare, unusual real-world driving scenario likely appearing in ~1% of driving data, involving uncommon agents,
unexpected behaviors, abnormal road events, unusual interactions, or safety-critical oddities.
IMPORTANT: Weather/lighting alone (rain, snow, fog, night, glare) is NOT long-tail unless it *causes* a rare event
(e.g., crash, road blockage, unusual agent behavior).

Conservatism rules:
- If the clip is ambiguous, partially occluded, or too short to confirm, output "unknown" and reduce confidence.
- Prefer fewer labels over many. Only label what you can justify with clear evidence.

Event taxonomy (9 classes; choose 0+ labels; only use labels listed here):

1) WORK_ZONES_TEMP_TRAFFIC_CONTROL
   - Construction/road works: cones/barriers/temporary lanes, workers, construction vehicles, road narrowing,
     detours, scaffolding, fresh asphalt

2) COMPLEX_INTERSECTION_INTERACTION
   - Multi-leg intersection/roundabout, confusing right-of-way, dense turning conflicts,
     abnormal merging/yielding behavior, aggressive cut-ins or sudden lane changes while merging

3) PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY
   - Crowding near roadway, groups crossing, jaywalking clusters, pedestrians close to ego path,
     school zone crowd

4) CYCLISTS_AND_MICROMOBILITY_COMPLEX
   - Multiple cyclists, unpredictable weaving, cyclists entering/exiting traffic,
     scooters/e-bikes in mixed traffic

5) ANIMALS_BIRDS_ROADKILL
   - Live animals near/on roadway affecting driving (dogs/deer/birds etc.)
   - Bird flock/swarm near roadway/overhead, swarm-like motion, multiple birds crossing path
   - Visible roadkill/carcass on roadway/shoulder

6) ROAD_DEBRIS_OR_SAFETY_TRACES
   - Road debris / abnormal objects: furniture, boxes, ladders, fallen cargo, large plastic sheets,
     cones displaced, any unusual object on road
   - Fresh skid marks / long tire traces suggesting sudden braking/near-miss/crash aftermath
     (must be visually evident)

7) EMERGENCY_INCIDENT_SCENE
   - Emergency/incident scene: stopped emergency vehicles, active incident response, flares,
     police directing traffic, crash scene

8) SPECIAL_OR_UNCOMMON_VEHICLE_BEHAVIOR
   - Oversized loads, tow trucks actively towing, construction convoy, farm equipment, street sweeper,
     unusual trailers, vehicles with abnormal behavior (wrong-way, stopped in lane), ceremonial/escort vehicles

9) OTHER_LONGTAIL
   - Rare event not covered above; must describe clearly in evidence

Output format:
Return ONLY a JSON object (no extra text) with EXACTLY these fields:
{
  "is_longtail": true/false/"unknown",
  "rarity_score_0_100": number,              // 0 common, 100 extremely rare
  "confidence": number,                      // 0.0 to 1.0
  "event_labels": [                          // 0 or more
    {
      "label": "ONE_OF_THE_9_ENUM_LABELS_ABOVE",
      "label_confidence": number,            // 0.0 to 1.0
      "evidence": [string, ...]              // short, concrete visual cues
    }
  ],
  "key_evidence": [string, ...]              // overall clip evidence; may overlap with label evidence
}

Scoring guide:
- rarity_score_0_100:
  0-20: normal driving, common interactions
  21-40: mildly uncommon but still frequent
  41-70: clearly unusual, likely long-tail
  71-100: extremely rare / safety-critical oddity
- confidence:
  high when evidence is clear and persistent across frames; low when brief/occluded/ambiguous.

Decision hints:
- If ANY label in taxonomy (1–8) is confidently present and affects driving context, is_longtail is likely true.
- If only weather/lighting changes without a rare event, is_longtail should be false.
""".strip()


def ddp_init():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def sample_frames(video_path: str, target_fps: float = 1.0, max_frames: int = 64) -> list[Image.Image]:
    reader = iio.get_reader(video_path, format="ffmpeg")
    try:
        meta = reader.get_meta_data()
        src_fps = meta.get("fps") or 30.0
        if not src_fps or np.isnan(src_fps) or src_fps <= 1e-6:
            src_fps = 30.0

        nframes = meta.get("nframes")
        duration_sec = meta.get("duration")
        if duration_sec is None and nframes and nframes > 0 and not np.isnan(nframes):
            duration_sec = nframes / src_fps

        frames: list[Image.Image] = []
        step = 1.0 / max(target_fps, 1e-6)

        t = 0.0
        while len(frames) < max_frames:
            if duration_sec is not None and t > duration_sec:
                break

            frame_idx = int(round(t * src_fps))
            try:
                frame = reader.get_data(frame_idx)
            except Exception:
                break

            frames.append(Image.fromarray(frame))
            t += step
    finally:
        reader.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames sampled from: {video_path}")
    return frames


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in output:\n{text}")
    return json.loads(m.group(0))


@torch.inference_mode()
def run_one(model, processor, video_frames: list[Image.Image], prompt: str, max_new_tokens: int) -> dict:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video"},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[video_frames],
        return_tensors="pt",
        padding=True
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    in_len = inputs["input_ids"].shape[1]
    out_trim = out_ids[:, in_len:]
    out_text = processor.batch_decode(
        out_trim,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return extract_json(out_text)


def load_model_local(model_path: str, local_rank: int, dtype: str, attn_impl: str):
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = "auto"

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if attn_impl != "auto":
        try:
            model.config.attn_implementation = attn_impl
        except Exception:
            pass

    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True, help="Local snapshot folder containing config.json")
    ap.add_argument("--video_dir", type=str, required=True, help="Directory containing mp4s (recursive)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for per-rank JSONL")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="bf16", help="auto|bf16|fp16")
    ap.add_argument("--attn_impl", type=str, default="auto", help="auto|flash_attention_2")
    args = ap.parse_args()

    rank, world_size, local_rank = ddp_init()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每个 rank 只输出 longtail JSONL
    out_longtail = out_dir / f"longtail_rank{rank:02d}_of_{world_size:02d}.jsonl"

    mp4s = sorted(video_dir.rglob("*.mp4"))
    if len(mp4s) == 0:
        raise RuntimeError(f"No .mp4 found under: {video_dir}")

    my_mp4s = mp4s[rank::world_size]

    if rank == 0:
        print(f"[DDP] world_size={world_size}, total_videos={len(mp4s)}")
    print(f"[rank {rank}] videos_assigned={len(my_mp4s)}")
    print(f"[rank {rank}] write longtail={out_longtail}")

    model, processor = load_model_local(args.model_path, local_rank, args.dtype, args.attn_impl)

    with out_longtail.open("w", encoding="utf-8") as f_lt:
        for p in my_mp4s:
            video_path = str(p.resolve())
            file_name = p.name

            try:
                frames = sample_frames(video_path, target_fps=args.fps, max_frames=args.max_frames)

                lt = run_one(model, processor, frames, PROMPT_LONGTAIL, args.max_new_tokens)

                rec_lt = {
                    "file_name": file_name,
                    "file_path": video_path,
                    "sample_fps": args.fps,
                    "num_frames_used": len(frames),
                    **lt
                }

                f_lt.write(json.dumps(rec_lt, ensure_ascii=False) + "\n")
                f_lt.flush()

            except Exception as e:
                err = {"file_name": file_name, "file_path": video_path, "error": str(e)}
                f_lt.write(json.dumps(err, ensure_ascii=False) + "\n")
                f_lt.flush()

    dist.barrier()
    if rank == 0:
        print(f"Done. Per-rank outputs in: {out_dir}")
        print("Merge longtail outputs:")
        print(f"  cat {out_dir}/longtail_rank*_of_*.jsonl > {out_dir}/longtail_merged.jsonl")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
