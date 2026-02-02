#!/usr/bin/env bash
# Parallel unzip + rename extracted videos with _chunkXXXX suffix
# Usage:
#   bash unzip_parallel_chunk_suffix.sh /path/to/camera_front_wide_120fov [parallel_jobs]
#
# Example:
#   bash unzip_parallel_chunk_suffix.sh \
#     /scratch/10102/hh29499/longtail/PhysicalAI-Autonomous-Vehicles/camera/camera_front_wide_120fov 10

set -euo pipefail

ROOT_DIR="${1:-/scratch/10102/hh29499/MineLongTail/longtail/PhysicalAI-Autonomous-Vehicles/camera/camera_front_wide_120fov}"
JOBS="${2:-16}"

cd "$ROOT_DIR"

mkdir -p videos _tmp_unzip

process_zip() {
  local z="$1"

  # Extract chunk number from filename: ...chunk_1550.zip -> 1550 (handles leading zeros)
  local chunk
  chunk="$(basename "$z" | sed -n 's/.*chunk_\([0-9]\+\)\.zip/\1/p')"
  if [[ -z "${chunk}" ]]; then
    echo "[WARN] cannot parse chunk from: $z" >&2
    return 0
  fi

  # Force base-10 to avoid octal issues (0008/0009)
  local chunk_num chunk_tag
  chunk_num=$((10#${chunk}))
  chunk_tag="$(printf "%04d" "$chunk_num")"

  local workdir
  workdir="_tmp_unzip/$(basename "${z%.zip}")"
  mkdir -p "$workdir"

  # Unzip into isolated workdir to avoid collisions under parallelism
  unzip -q "$z" -d "$workdir" || { echo "[ERR] unzip failed: $z" >&2; return 1; }

  # Move video files into videos/ and append _chunkXXXX suffix
  find "$workdir" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) -print0 \
    | while IFS= read -r -d '' f; do
        local base name ext out i
        base="$(basename "$f")"
        name="${base%.*}"
        ext="${base##*.}"

        out="videos/${name}_chunk${chunk_tag}.${ext}"

        # Avoid overwrite by adding _1/_2/... if needed
        if [[ -e "$out" ]]; then
          i=1
          while [[ -e "videos/${name}_chunk${chunk_tag}_$i.${ext}" ]]; do
            i=$((i+1))
          done
          out="videos/${name}_chunk${chunk_tag}_$i.${ext}"
        fi

        mv -n "$f" "$out"
      done

  rm -rf "$workdir"
}

export -f process_zip

# Find zips in the current folder only. If your zips are nested, remove -maxdepth 1.
find . -maxdepth 1 -type f -name '*.zip' -print0 \
  | xargs -0 -n 1 -P "$JOBS" bash -lc 'process_zip "$1"' _

echo "Done."
echo -n "Extracted file count: "
find videos -type f | wc -l
