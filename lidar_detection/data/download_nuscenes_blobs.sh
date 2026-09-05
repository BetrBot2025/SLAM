#!/usr/bin/env bash
#
# Download and extract the remaining nuScenes trainval blob archives (WSL Ubuntu).
#
# You already have blobs 1 2 3, so this fetches 4..10 by default.
# GNU wget (-c) resumes broken downloads, so you can re-run this safely.
#
# Usage:
#   chmod +x download_nuscenes_blobs.sh
#   ./download_nuscenes_blobs.sh                       # default dest + blobs 4..10
#   ./download_nuscenes_blobs.sh /home/me/nuscenes     # custom dest dir
#   BLOBS="4 5 6" ./download_nuscenes_blobs.sh         # pick specific blobs
#   CLEANUP=1 ./download_nuscenes_blobs.sh             # delete .tgz after extract

set -euo pipefail

# ---- config ----
DEST_DIR="${1:-/mnt/c/mohitk/AD/Code/lidar_detection/data/nuscenes_full}"   # /mnt/c/... = your Windows C: drive
BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
BLOBS="${BLOBS:-6 7 8 9 10}"                  # you already have 1 2 3
CLEANUP="${CLEANUP:-0}"                            # 1 = remove .tgz after extracting

# ---- checks ----
command -v wget >/dev/null 2>&1 || { echo "wget not found: sudo apt install wget"; exit 1; }
command -v tar  >/dev/null 2>&1 || { echo "tar not found: sudo apt install tar";  exit 1; }

#mkdir -p "$DEST_DIR"

for n in $BLOBS; do
    name=$(printf "v1.0-trainval%02d_blobs.tgz" "$n")
    url="$BASE_URL/$name"
    out="$DEST_DIR/$name"

    echo ""
    echo "=== Blob $n : $name ==="

    # download with resume
    wget -c "$url" -O "$out"

    # extract (all blobs merge into samples/ and sweeps/)
    echo "Extracting $name ..."
    tar -xzf "$out" -C "$DEST_DIR/$name"

    echo "Done: $name"

    if [ "$CLEANUP" = "1" ]; then
        rm -f "$out"
        echo "Removed archive $name"
    fi
done

echo ""
echo "All requested blobs processed."
echo "Dataset folder: $DEST_DIR"
