# Qwen3-8B 한국어 QA RAG 모델 훈련 및 검증 가이드

## 📋 개요

이 프로젝트는 Qwen3-8B 모델을 한국어 QA 데이터셋으로 파인튜닝하고 검증하는 완전한 파이프라인을 제공합니다.

## 🚀 시작하기

### 1. 환경 설정

```bash
# uv 가상환경 활성화 (이미 설정되어 있다고 가정)
# .venv 가상환경 사용

# 스크립트 권한 설정
./setup_permissions.sh
```

### 2. Qwen3-8B 모델 훈련

```bash
# Qwen3-8B 모델로 훈련 실행
./run_train_improved.sh
```

**주요 훈련 설정:**
- 모델: `Qwen/Qwen3-8B`
- LoRA 설정: r=32, alpha=64, dropout=0.05
- 훈련 epochs: 5
- 배치 크기: 1 (gradient_accumulation_steps=16)
- 학습률: 1e-4

### 3. 모델 검증

```bash
# 전용 검증 스크립트 실행
./run_validate_qwen3_8b.sh

# 또는 직접 Python 스크립트 실행
uv run python validate_qwen3_8b.py \
    --base_model Qwen/Qwen3-8B \
    --peft_model ./results/qwen3-8b-korean-qa-improved \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10
```

### 4. 추론 테스트

```bash
# 기존 추론 스크립트 실행 (훈련 완료 후)
./run_inference_improved.sh
```

## 📊 기대 성능

Qwen3-8B는 더 큰 모델이므로 Qwen2.5-0.5B 대비 다음과 같은 개선을 기대할 수 있습니다:

- **정확도 향상**: 더 복잡한 한국어 문법 규칙 이해
- **일관성 증대**: 더 안정적인 답변 생성
- **컨텍스트 이해**: 긴 문맥에서의 더 나은 추론

## 🗂️ 파일 구조

```
Korean_QA_RAG_2025/
├── train.py                      # Qwen3-8B 최적화된 훈련 스크립트
├── validate_qwen3_8b.py          # Qwen3-8B 전용 검증 스크립트
├── run_train_improved.sh         # Qwen3-8B 훈련 실행 스크립트
├── run_validate_qwen3_8b.sh      # Qwen3-8B 검증 실행 스크립트
├── run_inference_improved.sh     # Qwen3-8B 추론 실행 스크립트
└── results/
    └── qwen3-8b-korean-qa-improved/  # 훈련된 모델 저장 위치
```

## ⚙️ 하이퍼파라미터 설정

### LoRA 설정
- **lora_r**: 32 (Qwen3-8B에 최적화)
- **lora_alpha**: 64
- **lora_dropout**: 0.05
- **target_modules**: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### 훈련 설정
- **num_train_epochs**: 5
- **per_device_train_batch_size**: 1
- **gradient_accumulation_steps**: 16
- **learning_rate**: 1e-4
- **max_seq_length**: 2048

## 🔍 검증 결과 분석

검증 스크립트는 다음 메트릭을 제공합니다:

1. **정확도**: 정확한 답변 비율
2. **평균 생성 시간**: 답변당 평균 소요 시간
3. **키워드 매칭**: 부분적 정답 감지

## 🛠️ 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 조정
# run_train_improved.sh에서 다음 설정 수정:
--per_device_train_batch_size 1
--gradient_accumulation_steps 32  # 더 큰 값 사용
```

### 훈련 중단 시 재시작
```bash
# 체크포인트에서 자동 재시작됩니다
./run_train_improved.sh
```

## 📈 성능 모니터링

```bash
# TensorBoard로 훈련 과정 모니터링
tensorboard --logdir=./logs/improved
```

## 🤝 기여하기

1. 하이퍼파라미터 튜닝 제안
2. 새로운 검증 메트릭 추가
3. 추가적인 한국어 QA 데이터셋 지원

---

**참고**: Qwen3-8B는 대용량 모델이므로 충분한 GPU 메모리(최소 16GB 권장)가 필요합니다. 