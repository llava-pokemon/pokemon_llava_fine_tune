#!/bin/bash

PROMPT="What can this pokemon do in the Pokemon Trading Card game? Explain in detail."
MAX_NEW_TOKENS=1024

CUDA_VISIBLE_DEVICES=0 python inference_dataset.py \
    --model-path checkpoints/llava-v1.5-13b-lora \
    --model-base liuhaotian/llava-v1.5-13b \
    --test-file data/pokemon_llava_dataset_test.jsonl \
    --image-folder data/pokemon_images \
    --query "$PROMPT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output-file finetune.jsonl

CUDA_VISIBLE_DEVICES=1 python inference_dataset.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --test-file data/pokemon_llava_dataset_test.jsonl \
    --image-folder data/pokemon_images \
    --query "$PROMPT" \
    --max_new_tokens $MAX_NEW_TOKENS \
    --output-file base.jsonl

wait