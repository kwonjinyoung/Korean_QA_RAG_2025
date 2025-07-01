#!/bin/bash

# 한국어 QA RAG 시스템 평가 자동화 스크립트
# Korean QA RAG System Evaluation Automation Script

# 기본값 설정
LIMIT=10

# 명령줄 인자 처리
if [ "$#" -ge 1 ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        LIMIT=$1
    else
        echo "❌ 오류: 첫 번째 인자는 숫자여야 합니다."
        echo "사용법: $0 [테스트할_데이터_개수]"
        exit 1
    fi
fi

echo "========================================="
echo "한국어 QA RAG 시스템 평가 시작"
echo "Korean QA RAG System Evaluation Started"
echo "테스트 데이터 개수: $LIMIT"
echo "========================================="

# 현재 시간 기록
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "시작 시간: $START_TIME"
echo ""

# 1. 모델 답변 생성 (rag_dev.py 실행)
echo "📝 1단계: 모델 답변 생성 중..."
cd ../04_dev_run
uv run python rag_dev.py --limit $LIMIT
if [ $? -ne 0 ]; then
    echo "❌ 모델 답변 생성 실패! 종료합니다."
    exit 1
fi
echo "✅ 모델 답변 생성 완료!"
echo ""

# 2. 평가 수행 (eval.py 실행)
echo "📊 2단계: 평가 수행 중..."
cd ../05_dev_evaluation
uv run python eval.py --limit $LIMIT
if [ $? -ne 0 ]; then
    echo "❌ 평가 실패! 종료합니다."
    exit 1
fi
echo "✅ 평가 완료!"
echo ""

# 종료 시간 기록
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "종료 시간: $END_TIME"

echo "========================================="
echo "한국어 QA RAG 시스템 평가 완료!"
echo "Korean QA RAG System Evaluation Completed!"
echo "테스트 데이터 개수: $LIMIT"
echo "========================================="

# 결과 파일 위치 안내
echo ""
echo "📁 결과 파일 위치:"
echo "- 모델 답변: ../04_dev_run/result.json"
echo "- 평가 입력: ./eval_input.json"
echo "- 평가 결과: ./evaluation_results.json"
echo "- 평가 요약: ./evaluation_results.csv"
echo "" 