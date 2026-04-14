#!/bin/bash

set -euo pipefail

# Qwen3-VL example for the public Math Prompt-in-Image benchmark.
#
# This benchmark loads images directly from the public HF dataset:
#   YongxinWang/math-prompt-in-image
#
# Override any variable below at runtime if needed, e.g.:
#   MODEL=Qwen/Qwen3-VL-8B-Instruct TASKS=mathvision_testmini_prompt_in_image \
#   bash examples/models/qwen3vl_math_prompt_in_image.sh

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# pip install transformers==4.57.1
# pip install ".[qwen]"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12346}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
MAX_PIXELS="${MAX_PIXELS:-12845056}"
OUTPUT_PATH="${OUTPUT_PATH:-./logs/qwen3vl_math_prompt_in_image}"

# Section 1: run the public prompt-in-image benchmark group.
DEFAULT_GROUP_TASKS="math_prompt_in_image_testmini"

# Section 2: run the two tasks explicitly if you want per-task control.
DEFAULT_SINGLE_TASKS="mathvision_testmini_prompt_in_image,mathvista_testmini_prompt_in_image"

TASKS="${TASKS:-$DEFAULT_GROUP_TASKS}"

echo "=========================================="
echo "Math Prompt-in-Image Benchmark"
echo "=========================================="
echo "Model: $MODEL"
echo "Tasks: $TASKS"
echo "Num processes: $NUM_PROCESSES"
echo "Batch size: $BATCH_SIZE"
echo "Output path: $OUTPUT_PATH"
echo "------------------------------------------"
echo "Group tasks:   $DEFAULT_GROUP_TASKS"
echo "Single tasks:  $DEFAULT_SINGLE_TASKS"
echo "=========================================="

accelerate launch --num_processes="$NUM_PROCESSES" --main_process_port="$MAIN_PROCESS_PORT" -m lmms_eval \
  --model qwen3_vl \
  --model_args "pretrained=${MODEL},max_pixels=${MAX_PIXELS},attn_implementation=${ATTN_IMPL},interleave_visuals=False" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_PATH"
