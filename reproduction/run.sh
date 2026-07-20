#!/usr/bin/env bash
set -euo pipefail

python -m pip install --quiet \
  "transformers==4.53.2" \
  "peft==0.16.0" \
  "sentencepiece==0.2.0"

mkdir -p /tmp/rlm_repro_results
torchrun --standalone --nproc_per_node=8 reproduction/experiment.py
