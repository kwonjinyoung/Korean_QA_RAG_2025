#!/bin/bash

# Korean QA RAG REST API 서버 실행 스크립트
# REST API Server startup script for Korean QA RAG model

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "Korean QA RAG REST API 서버를 시작합니다..."
echo "모델 로딩에 시간이 걸릴 수 있습니다."
echo ""
echo "서버 주소: http://localhost:11435"
echo "API 문서: http://localhost:11435/docs"
echo "헬스체크: http://localhost:11435/health"
echo ""

# 필요한 Python 패키지 설치 (처음 실행시)
echo "필요한 패키지를 확인하고 설치합니다..."
pip install fastapi uvicorn python-multipart

# 서버 실행
uv run python server.py \
    --host 0.0.0.0 \
    --port 11435 \
    --reload 