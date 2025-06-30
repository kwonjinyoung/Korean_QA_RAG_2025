#!/bin/bash

# Qwen3-8B 16bit LoRA 한국어 QA RAG 파인튜닝 스크립트
# Korean QA RAG Fine-tuning Script for Qwen3-8B with 16-bit LoRA

echo "========================================="
echo "한국어 QA RAG 파인튜닝 시작 (Qwen3-8B 16bit LoRA) (Loss 0.05 이하까지)"
echo "Korean QA RAG Fine-tuning Started (Qwen3-8B 16bit LoRA) (Until Loss < 0.05)"
echo "========================================="

# 환경 변수 설정 (2개 GPU 사용)
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정 완료 (2개 GPU 사용):"
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
echo "GPU: 2장 분산 훈련"
echo "========================================="

# Qwen3-8B 16bit LoRA로 Loss 0.05 이하까지 훈련하기 위한 파라미터 (2 GPU 분산)
uv run python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --train_data_path ../RAG/resource/korean_language_rag_V1.0_train_and_dev.json \
    --val_data_path ../RAG/resource/korean_language_rag_V1.0_dev.json \
    --output_dir ./results/qwen3-8b-16bit-lora-korean-qa-improved-test-dev-all-data \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --logging_dir ./logs/qwen3-8b-16bit-lora-improved \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 0 \
    --fp16 \
    --seed 42 \
    --report_to tensorboard \
    --run_name korean-qa-rag-qwen3-8b-16bit-lora-improved \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --max_seq_length 4096 \
    --trust_remote_code \
    --optim adamw_torch \
    --ddp_find_unused_parameters false \
    --logging_first_step \
    --log_level info \
    --disable_tqdm false \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --gradient_checkpointing

echo ""
echo "========================================="
echo "훈련 완료 시간: $(date)"
echo "주의: 0.05 loss 달성을 위해 tensorboard 로그를 모니터링하고"
echo "eval_loss가 0.05 이하가 되면 수동으로 중단하세요"
echo "8B 모델 16bit LoRA + 2 GPU 분산 훈련으로 메모리 효율성 및 훈련 속도가 최적화되었습니다"
echo "Training Completed!"
echo "=========================================" 