#!/bin/bash

# Qwen3-8B 모델 검증 스크립트
# Qwen3-8B Model Validation Script

echo "========================================="
echo "Qwen3-8B 한국어 QA 모델 검증 시작"
echo "Qwen3-8B Korean QA Model Validation"
echo "========================================="

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정 완료:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

echo "검증 시작 시간: $(date)"
echo "========================================="

# Qwen3-8B 모델 검증 실행
uv run python validate_qwen3_8b.py \
    --base_model Qwen/Qwen3-8B \
    --peft_model ./results/qwen3-8b-korean-qa-improved \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10

echo ""
echo "========================================="
echo "검증 완료 시간: $(date)"
echo "Validation Completed!"
echo "=========================================" 