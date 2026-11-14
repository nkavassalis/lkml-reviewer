#!/usr/bin/env bash
# Copy v4's trained adapter (only) to data/train/run-4/, so the Mac build and
# multi-sample run don't wait for the cluster's evaluation and judging.
. "$(dirname "$0")/lib.sh"
on_master "docker exec linuspr-train chown -R 1000:1000 /work/run-4/adapter-final" || true
mkdir -p "$QROOT/data/train/run-4"
scp -q -r "$MASTER:$W/run-4/adapter-final" "$QROOT/data/train/run-4/" || exit 1
ls -la "$QROOT/data/train/run-4/adapter-final/adapter_model.safetensors"
