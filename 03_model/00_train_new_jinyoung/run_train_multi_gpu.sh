#!/bin/bash

# Qwen3-8B Korean QA RAG Fine-tuning Script
# 8비트 양자화 + LoRA 멀티GPU 분산 훈련 (48GB x2)

echo "========================================="
echo "Qwen3-8B 한국어 QA RAG 파인튜닝 시작"
echo "8비트 양자화 + LoRA 멀티GPU 분산 훈련"
echo "Korean QA RAG Fine-tuning with 8-bit + LoRA (Multi-GPU)"
echo "========================================="

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
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo

echo "환경 변수 설정 완료:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

# 필요한 라이브러리 확인 및 설치
echo "=== 필요한 라이브러리 확인 및 설치 ==="
python -c "import peft; print('✓ PEFT 라이브러리 확인됨')" || (echo "PEFT 라이브러리 설치 중..." && pip install peft)
python -c "import bitsandbytes; print('✓ BitsAndBytes 라이브러리 확인됨')" || (echo "BitsAndBytes 라이브러리 설치 중..." && pip install bitsandbytes)
python -c "import transformers; print('✓ Transformers 라이브러리 확인됨')" || (echo "Transformers 라이브러리 설치 중..." && pip install transformers)
python -c "import rouge_score; print('✓ ROUGE 라이브러리 확인됨')" || (echo "ROUGE 라이브러리 설치 중..." && pip install rouge-score)
python -c "import nltk; print('✓ NLTK 라이브러리 확인됨')" || (echo "NLTK 라이브러리 설치 중..." && pip install nltk)
python -c "import sklearn; print('✓ Scikit-learn 라이브러리 확인됨')" || (echo "Scikit-learn 라이브러리 설치 중..." && pip install scikit-learn)
python -c "import torch; print('✓ PyTorch 라이브러리 확인됨')" || echo "✗ PyTorch 라이브러리 설치 필요"
echo ""

echo "GPU 상태 확인:"
nvidia-smi
echo ""

echo "분산 훈련 환경 확인:"
python -c "import torch; print(f'GPU 개수: {torch.cuda.device_count()}'); print(f'현재 GPU: {torch.cuda.current_device()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

echo "Python 환경 확인:"
python --version
echo ""

echo "훈련 시작 시간: $(date)"
echo "모델: Qwen3-8B (8비트 양자화 + LoRA)"
echo "GPU: RTX A6000 48GB x2 (멀티GPU 분산 훈련)"
echo "LoRA 설정: rank=16, alpha=32, dropout=0.1"
echo "에포크: 3"
echo "배치 크기: 128 (per_device=8 × 2 GPU × gradient_accumulation=8)"
echo "학습률: 2e-4 (LoRA 적용)"
echo "시퀀스 길이: 2048"
echo "========================================="

# 데이터 파일 확인
echo "=== 데이터 파일 확인 ==="
TRAIN_DATA="../../resource/korean_language_rag_V1.0_train.json"
EVAL_DATA="../../resource/korean_language_rag_V1.0_dev.json"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "✗ 훈련 데이터 파일을 찾을 수 없습니다: ${TRAIN_DATA}"
    exit 1
fi

if [ ! -f "${EVAL_DATA}" ]; then
    echo "✗ 평가 데이터 파일을 찾을 수 없습니다: ${EVAL_DATA}"
    exit 1
fi

echo "✓ 훈련 데이터: ${TRAIN_DATA}"
echo "✓ 평가 데이터: ${EVAL_DATA}"
echo ""

# 멀티GPU 분산 훈련 실행
echo "=== 멀티GPU 분산 훈련 시작 ==="
uv run torchrun --nproc_per_node=2 --master_port=29500 train_jinyoung.py \
    --model_name_or_path "Qwen/Qwen3-8B" \
    --train_data_path "${TRAIN_DATA}" \
    --eval_data_path "${EVAL_DATA}" \
    --output_dir "./results/qwen3-8b-korean-qa-rag-lora-multi-gpu" \
    --max_seq_length 2048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --seed 42 \
    --torch_dtype "float16" \
    --local_rank -1

# 훈련 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 멀티GPU 분산 훈련 완료 시간: $(date)"
    echo "✓ 8비트 양자화 + LoRA 멀티GPU 훈련 완료"
    echo "✓ LoRA 어댑터 저장 위치: ./results/qwen3-8b-korean-qa-rag-lora-multi-gpu"
    echo ""
    echo "저장된 모델 크기 확인:"
    du -sh ./results/qwen3-8b-korean-qa-rag-lora-multi-gpu/
    echo ""
    echo "저장된 파일 목록:"
    ls -la ./results/qwen3-8b-korean-qa-rag-lora-multi-gpu/
    echo ""
    echo "Multi-GPU Training Completed Successfully!"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ 멀티GPU 분산 훈련 실패!"
    echo "✗ 로그를 확인하세요: ./results/qwen3-8b-korean-qa-rag-lora-multi-gpu/logs"
    echo "Multi-GPU Training Failed!"
    echo "========================================="
    exit 1
fi