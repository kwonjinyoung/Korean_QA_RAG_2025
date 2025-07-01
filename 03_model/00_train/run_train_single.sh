#!/bin/bash

# Qwen3-8B 4bit LoRA 한국어 QA RAG 파인튜닝 스크립트
# Korean QA RAG Fine-tuning Script for Qwen3-8B with 4-bit LoRA

echo "========================================="
echo "한국어 QA RAG 파인튜닝 시작 (Qwen3-8B 4bit LoRA) (Loss 0.05 이하까지)"
echo "Korean QA RAG Fine-tuning Started (Qwen3-8B 4bit LoRA) (Until Loss < 0.05)"
echo "========================================="

# 환경 변수 설정 (1개 GPU 사용 - RTX 4090)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정 완료 (RTX 4090 1개 GPU 사용):"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

echo "GPU 상태 확인:"
nvidia-smi || echo "nvidia-smi 사용 불가"
echo ""

echo "Python 환경 확인:"
uv run python --version
echo ""

# Tensorboard 설치 확인 및 설치
if ! uv run pip list | grep -q tensorboard; then
  echo "Tensorboard가 설치되어 있지 않습니다. 설치 중..."
  uv run pip install tensorboard
  echo "Tensorboard 설치 완료"
fi

echo "훈련 시작 시간: $(date)"
echo "모델: Qwen3-8B (4bit LoRA)"
echo "목표: Loss 0.05 이하까지 훈련 (수동 모니터링 필요)"
echo "GPU: RTX 4090 1장 단일 훈련"
echo "배치 사이즈: 1 (per_device_train_batch_size=1 × 1 GPU)"
echo "그라디언트 누적 단계: 32 (총 유효 배치 사이즈: 32)"
echo "시퀀스 길이: 4096"
echo "메모리 최적화: gradient_checkpointing 활성화, 4비트 양자화 적용"
echo "========================================="

# Qwen3-8B 4bit LoRA로 Loss 0.05 이하까지 훈련하기 위한 파라미터 (1 GPU)
uv run python train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --train_data_path ../../02_makeDataset_for_train/final_dataset.json \
    --output_dir ./results/qwen3-8b-4bit-lora-korean-qa-rag \
    --overwrite_output_dir \
    --do_train \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --logging_dir ./logs \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --use_4bit_quantization \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_use_double_quant true \
    --seed 42 \
    --report_to none \
    --run_name korean-qa-rag-finetune \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --max_seq_length 4096 \
    --trust_remote_code \
    --ddp_find_unused_parameters false \
    --logging_first_step \
    --log_level info \
    --disable_tqdm false \
    --gradient_checkpointing true

echo ""
echo "========================================="
echo "훈련 완료 시간: $(date)"
echo "8B 모델 4bit LoRA + RTX 4090 1장 훈련으로 메모리 효율성 및 훈련 속도가 최적화되었습니다"
echo "배치 사이즈: 1 (per_device_train_batch_size=1 × 1 GPU)"
echo "그라디언트 누적 단계: 32 (총 유효 배치 사이즈: 32)"
echo "시퀀스 길이: 4096"
echo "메모리 최적화: gradient_checkpointing 활성화, 4비트 양자화 적용"
echo "Training Completed!"
echo "=========================================" 