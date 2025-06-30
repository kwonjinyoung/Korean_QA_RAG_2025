#!/bin/bash

# Korean QA RAG API Server 실행 스크립트
# Korean QA RAG API Server Run Script

echo "========================================="
echo "한국어 QA RAG API 서버 시작"
echo "Korean QA RAG API Server Starting"
echo "========================================="

# 환경 변수 설정 (파인튜닝할 때와 동일)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정 완료:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

echo "GPU 상태 확인:"
nvidia-smi || echo "nvidia-smi 사용 불가"
echo ""

echo "Python 환경 확인:"
uv run python --version
echo ""

echo "서버 시작 시간: $(date)"
echo "모델: Qwen3-8B (4bit LoRA)"
echo "모델 경로: ../00_train/results/qwen3-8b-4bit-lora-korean-qa-rag/checkpoint-110"
echo "서버 주소: http://0.0.0.0:11435"
echo "API 문서: http://0.0.0.0:11435/docs"
echo "========================================="

# 서버 실행
uv run python server.py \
    --host 0.0.0.0 \
    --port 11435

echo ""
echo "========================================="
echo "서버 종료 시간: $(date)"
echo "Server Stopped!"
echo "=========================================" 