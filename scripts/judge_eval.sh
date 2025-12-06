#!/bin/bash

BASE="base.jsonl"
FT="finetune.jsonl"

CUDA_VISIBLE_DEVICES=0 python judge_eval.py \
  --base "$BASE" \
  --ft "$FT" \
  --out qwen_judge.jsonl &

CUDA_VISIBLE_DEVICES=1 python judge_eval.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --base "$BASE" \
  --ft "$FT" \
  --out mistral_judge.jsonl &
  
wait