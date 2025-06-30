#!/bin/bash

# 개선된 훈련 모델 추론 테스트 스크립트
# Inference test script for improved fine-tuned model

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# train.py에서 사용한 4bit 양자화 설정과 일치하도록 추론 실행
uv run python run/inference.py \
    --base_model Qwen/Qwen3-32B \
    --peft_model ./results/qwen3-32b-4bit-korean-qa-improved/checkpoint-160 \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10 \
    --use_4bit_quantization \
    --temperature 0.7 \
    --top_p 0.9 \
    --max_new_tokens 512 