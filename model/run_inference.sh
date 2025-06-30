#!/bin/bash

# 훈련된 모델 추론 테스트 스크립트
# Inference test script for fine-tuned model

export CUDA_VISIBLE_DEVICES=0

uv run python run/inference.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --peft_model ./results/qwen2.5-0.5b-korean-qa \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10 