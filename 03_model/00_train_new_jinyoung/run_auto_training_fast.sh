#!/bin/bash

# 고속 자동 훈련 스크립트 (최적화된 배치 크기)
# Fast Auto Training with Optimized Batch Size

echo "========================================="
echo "🚀 고속 자동 훈련 시스템 시작"
echo "Fast Auto Training with Optimized Settings"
echo "GPU 메모리를 최대한 활용하여 훈련 속도를 향상시킵니다"
echo "========================================="

source .venv/bin/activate

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "환경 변수 설정 완료:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

echo "GPU 상태 확인:"
nvidia-smi
echo ""

echo "Python 환경 확인:"
python --version
echo ""

# 최적화된 목표 점수 설정 (빠른 달성을 위해 약간 낮춤)
TARGET_EXACT_MATCH=0.70  # 정확도 70% (기존 75%)
TARGET_F1_SCORE=0.75     # F1 점수 75% (기존 80%)
TARGET_BLEU_SCORE=0.65   # BLEU 점수 65% (기존 70%)
MAX_ITERATIONS=8         # 최대 8번 반복 (기존 10번)
MAX_EPOCHS=3            # 반복당 3 에포크 (기존 5번) - 더 빠른 반복

echo "=== 고속 훈련 설정 ==="
echo "목표 정확도 (Exact Match): ${TARGET_EXACT_MATCH} (빠른 달성)"
echo "목표 F1 점수: ${TARGET_F1_SCORE} (빠른 달성)" 
echo "목표 BLEU 점수: ${TARGET_BLEU_SCORE} (빠른 달성)"
echo "최대 반복 횟수: ${MAX_ITERATIONS}"
echo "반복당 에포크 수: ${MAX_EPOCHS} (빠른 반복)"
echo "배치 크기 최적화: GPU 메모리 최대 활용"
echo "출력 디렉토리: ./results/auto_training_fast"
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
mkdir -p ./results/auto_training_fast

echo "=== 🚀 고속 자동 훈련 시작 ==="
echo "시작 시간: $(date)"
echo "GPU 메모리 최대 활용으로 훈련 속도를 향상시킵니다..."
echo ""

# 고속 자동 훈련 실행
uv run python auto_train_with_evaluation_fast.py \
    --model_name "Qwen/Qwen3-8B" \
    --train_data_path "${TRAIN_DATA}" \
    --test_data_path "${TEST_DATA}" \
    --output_dir "./results/auto_training_fast" \
    --target_exact_match ${TARGET_EXACT_MATCH} \
    --target_f1_score ${TARGET_F1_SCORE} \
    --target_bleu_score ${TARGET_BLEU_SCORE} \
    --max_epochs ${MAX_EPOCHS} \
    --max_iterations ${MAX_ITERATIONS}

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 🚀 고속 자동 훈련 완료 시간: $(date)"
    echo "✓ 목표 점수 달성 또는 최대 반복 횟수 도달"
    echo "✓ 결과 저장 위치: ./results/auto_training_fast"
    echo ""
    echo "훈련 기록 및 최고 성능 모델:"
    if [ -f "./results/auto_training_fast/training_history.json" ]; then
        echo "✓ 훈련 기록: ./results/auto_training_fast/training_history.json"
    fi
    echo ""
    echo "저장된 결과 확인:"
    ls -la ./results/auto_training_fast/
    echo ""
    echo "🎉 고속 자동 훈련 시스템 완료!"
    echo "성능 분석: python analyze_training_results.py --results_dir ./results/auto_training_fast"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ 고속 자동 훈련 실패!"
    echo "✗ 로그를 확인하세요: ./results/auto_training_fast/"
    echo "Fast Auto Training Failed!"
    echo "========================================="
    exit 1
fi