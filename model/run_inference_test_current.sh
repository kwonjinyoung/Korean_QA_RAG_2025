#!/bin/bash

# 현재 존재하는 모델로 테스트하는 스크립트
# Test script for currently available model

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "현재 존재하는 Qwen2.5-0.5B 모델로 테스트 중..."

uv run python run/inference.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --peft_model ./results/qwen2.5-0.5b-korean-qa-improved \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10 