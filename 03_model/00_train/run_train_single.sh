#!/bin/bash

# Qwen3-8B 16bit LoRA 한국어 QA RAG 파인튜닝 스크립트 (RTX 4090용 최적화)
# Korean QA RAG Fine-tuning Script for Qwen3-8B with 16-bit LoRA (Optimized for RTX 4090)

echo "========================================="
echo "한국어 QA RAG 파인튜닝 시작 (Qwen3-8B 16bit LoRA) (Loss 0.05 이하까지)"
echo "Korean QA RAG Fine-tuning Started (Qwen3-8B 16bit LoRA) (Until Loss < 0.05)"
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

echo "훈련 시작 시간: $(date)"
echo "모델: Qwen3-8B (16bit LoRA)"
echo "목표: Loss 0.05 이하까지 훈련 (수동 모니터링 필요)"
echo "GPU: RTX 4090 1장 훈련"
echo "배치 사이즈: 4"
echo "시퀀스 길이: 4096"
echo "========================================="

# Qwen3-8B 16bit LoRA로 Loss 0.05 이하까지 훈련하기 위한 파라미터 (RTX 4090 단일 GPU)
uv run python train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --train_data_path 02_makeDataset_for_train/final_dataset.json \
    --output_dir ./results/qwen3-8b-16bit-lora-korean-qa-rag \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --logging_dir ./logs \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 2 \
    --fp16 \
    --seed 42 \
    --report_to tensorboard \
    --run_name korean-qa-rag-finetune \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --max_seq_length 4096 \
    --trust_remote_code \
    --logging_first_step \
    --log_level info \
    --disable_tqdm false \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --validation_split_percentage 10

echo ""
echo "========================================="
echo "훈련 완료 시간: $(date)"
echo "주의: 0.05 loss 달성을 위해 tensorboard 로그를 모니터링하고"
echo "eval_loss가 0.05 이하가 되면 수동으로 중단하세요"
echo "8B 모델 16bit LoRA + RTX 4090 단일 GPU 훈련으로 메모리 효율성을 위해 최적화되었습니다"
echo "유효 배치 사이즈: 16 (per_device_train_batch_size=4 × gradient_accumulation_steps=4)"
echo "Training Completed!"
echo "=========================================" 