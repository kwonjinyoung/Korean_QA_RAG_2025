#!/bin/bash

# Qwen3-32B 4bit 양자화 한국어 QA RAG 파인튜닝 스크립트
# Korean QA RAG Fine-tuning Script for Qwen3-32B with 4-bit Quantization

echo "========================================="
echo "한국어 QA RAG 파인튜닝 시작 (Qwen3-32B 4bit 양자화)"
echo "Korean QA RAG Fine-tuning Started (Qwen3-32B 4bit Quantization)"
echo "========================================="

# 환경 변수 설정 (2개 GPU 사용 - RTX A6000)
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정 완료 (RTX A6000 2개 GPU 사용):"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
echo ""

echo "GPU 상태 확인:"
nvidia-smi || echo "nvidia-smi 사용 불가"
echo ""

echo "Python 환경 확인:"
uv run python --version
echo ""

echo "훈련 시작 시간: $(date)"
echo "모델: Qwen3-32B (4bit 양자화)"
echo "에폭: 5"
echo "GPU: RTX A6000 2장 분산 훈련"
echo "배치 사이즈: 16 (per_device_train_batch_size=1 × 2 GPU × gradient_accumulation_steps=8)"
echo "그라디언트 누적 단계: 8 (총 유효 배치 사이즈: 16)"
echo "시퀀스 길이: 3500 (32B 모델에 맞게 조정)"
echo "양자화: 4bit (메모리 최적화)"
echo "메모리 최적화: gradient_checkpointing 활성화"
echo "========================================="

# Qwen3-32B 4bit 양자화로 5 에폭 훈련하기 위한 파라미터 (2 GPU 분산)
uv run python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
    --model_name_or_path Qwen/Qwen3-32B \
    --train_data_path ../../02_makeDataset_for_train/final_dataset.json \
    --output_dir ./results/qwen3-32b-4bit-korean-qa-rag \
    --overwrite_output_dir \
    --do_train \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --logging_dir ./logs \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --use_4bit_quantization \
    --bnb_4bit_compute_dtype float16 \
    --bnb_4bit_use_double_quant \
    --bnb_4bit_quant_type nf4 \
    --seed 42 \
    --report_to tensorboard \
    --run_name korean-qa-rag-finetune-32b \
    --max_seq_length 3500 \
    --trust_remote_code \
    --ddp_find_unused_parameters false \
    --logging_first_step \
    --log_level info \
    --disable_tqdm false \
    --gradient_checkpointing true

echo ""
echo "========================================="
echo "훈련 완료 시간: $(date)"
echo "32B 모델 4bit 양자화 + RTX A6000 2장 분산 훈련으로 메모리 효율성 및 훈련 속도가 최적화되었습니다"
echo "배치 사이즈: 16 (per_device_train_batch_size=1 × 2 GPU × gradient_accumulation_steps=8)"
echo "그라디언트 누적 단계: 8 (총 유효 배치 사이즈: 16)"
echo "시퀀스 길이: 3500 (32B 모델에 맞게 조정)"
echo "에폭: 5"
echo "학습률: 5e-5 (32B 모델에 맞게 조정)"
echo "양자화: 4bit NF4 (메모리 최적화)"
echo "메모리 최적화: gradient_checkpointing 활성화"
echo "Training Completed!"
echo "========================================="
 