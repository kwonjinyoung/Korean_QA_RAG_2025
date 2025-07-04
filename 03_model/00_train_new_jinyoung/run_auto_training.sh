#!/bin/bash

# 자동 성능 평가 및 반복 훈련 스크립트
# Auto Training with Performance Evaluation System

echo "========================================="
echo "자동 성능 평가 및 반복 훈련 시스템 시작"
echo "Auto Training with Performance Evaluation"
echo "목표 달성까지 자동으로 훈련을 반복합니다"
echo "========================================="

source .venv/bin/activate

# 가상환경 활성화
echo "=== 가상환경 활성화 ==="
if [ -d ".venv" ]; then
    echo "✓ .venv 가상환경을 활성화합니다."
    source .venv/bin/activate
    echo "✓ 가상환경 활성화 완료"
else
    echo "⚠ .venv 가상환경이 없습니다. 시스템 Python을 사용합니다."
fi
echo ""

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "환경 변수 설정 완료:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

# 필요한 라이브러리 확인 및 설치
#echo "=== 필요한 라이브러리 확인 및 설치 ==="
#python -c "import peft; print('✓ PEFT 라이브러리 확인됨')" || (echo "PEFT 라이브러리 설치 중..." && pip install peft)
#python -c "import bitsandbytes; print('✓ BitsAndBytes 라이브러리 확인됨')" || (echo "BitsAndBytes 라이브러리 설치 중..." && pip install bitsandbytes)
#python -c "import transformers; print('✓ Transformers 라이브러리 확인됨')" || (echo "Transformers 라이브러리 설치 중..." && pip install transformers)
#python -c "import rouge_score; print('✓ ROUGE 라이브러리 확인됨')" || (echo "ROUGE 라이브러리 설치 중..." && pip install rouge-score)
#python -c "import nltk; print('✓ NLTK 라이브러리 확인됨')" || (echo "NLTK 라이브러리 설치 중..." && pip install nltk)
#python -c "import sklearn; print('✓ Scikit-learn 라이브러리 확인됨')" || (echo "Scikit-learn 라이브러리 설치 중..." && pip install scikit-learn)
#echo ""

echo "GPU 상태 확인:"
nvidia-smi
echo ""

echo "Python 환경 확인:"
python --version
echo ""

# 목표 점수 설정
TARGET_EXACT_MATCH=0.75  # 정확도 75%
TARGET_F1_SCORE=0.80     # F1 점수 80%
TARGET_BLEU_SCORE=0.70   # BLEU 점수 70%
MAX_ITERATIONS=10        # 최대 10번 반복
MAX_EPOCHS=5            # 반복당 5 에포크

echo "=== 자동 훈련 설정 ==="
echo "목표 정확도 (Exact Match): ${TARGET_EXACT_MATCH}"
echo "목표 F1 점수: ${TARGET_F1_SCORE}" 
echo "목표 BLEU 점수: ${TARGET_BLEU_SCORE}"
echo "최대 반복 횟수: ${MAX_ITERATIONS}"
echo "반복당 에포크 수: ${MAX_EPOCHS}"
echo "출력 디렉토리: ./results/auto_training"
echo ""

# 데이터 파일 확인
echo "=== 데이터 파일 확인 ==="
TRAIN_DATA="../../resource/korean_language_rag_V1.0_train.json"
TEST_DATA="../../resource/korean_language_rag_V1.0_dev.json"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "✗ 훈련 데이터 파일을 찾을 수 없습니다: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -f "${TEST_DATA}" ]; then
    echo "✗ 테스트 데이터 파일을 찾을 수 없습니다: ${TEST_DATA}"
    exit 1
fi

echo "✓ 훈련 데이터: ${TRAIN_DATA}"
echo "✓ 테스트 데이터: ${TEST_DATA}"
echo ""

# 출력 디렉토리 생성
mkdir -p ./results/auto_training

echo "=== 자동 훈련 시작 ==="
echo "시작 시간: $(date)"
echo "목표 점수에 도달할 때까지 자동으로 훈련을 반복합니다..."
echo ""

# 자동 훈련 실행
uv run python auto_train_with_evaluation.py \
    --model_name "Qwen/Qwen3-8B" \
    --train_data_path "${TRAIN_DATA}" \
    --test_data_path "${TEST_DATA}" \
    --output_dir "./results/auto_training" \
    --target_exact_match ${TARGET_EXACT_MATCH} \
    --target_f1_score ${TARGET_F1_SCORE} \
    --target_bleu_score ${TARGET_BLEU_SCORE} \
    --max_epochs ${MAX_EPOCHS} \
    --max_iterations ${MAX_ITERATIONS}

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 자동 훈련 완료 시간: $(date)"
    echo "✓ 목표 점수 달성 또는 최대 반복 횟수 도달"
    echo "✓ 결과 저장 위치: ./results/auto_training"
    echo ""
    echo "훈련 기록 및 최고 성능 모델:"
    if [ -f "./results/auto_training/training_history.json" ]; then
        echo "✓ 훈련 기록: ./results/auto_training/training_history.json"
    fi
    echo ""
    echo "저장된 결과 확인:"
    ls -la ./results/auto_training/
    echo ""
    echo "🎉 자동 훈련 시스템 완료!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ 자동 훈련 실패!"
    echo "✗ 로그를 확인하세요: ./results/auto_training/"
    echo "Auto Training Failed!"
    echo "========================================="
    exit 1
fi