# Korean QA RAG Fine-tuning Project

한국어 문법 및 언어 관련 질문-답변 데이터셋을 이용한 Qwen 2.5-0.5B 모델 파인튜닝 프로젝트입니다.

## 프로젝트 구조

```
Korean_QA_RAG_2025/
├── train.py                    # 메인 훈련 스크립트
├── run_train.sh               # 훈련 실행 스크립트
├── run_inference.sh           # 추론 실행 스크립트
├── pyproject.toml             # 프로젝트 의존성 설정
├── src/
│   ├── __init__.py
│   └── data.py                # 데이터 처리 모듈
├── run/
│   ├── __init__.py
│   ├── test.py
│   └── inference.py           # 추론 테스트 스크립트
└── resource/
    └── RAG/
        ├── korean_language_rag_V1.0_train.json  # 훈련 데이터
        ├── korean_language_rag_V1.0_dev.json    # 검증 데이터
        └── korean_language_rag_V1.0_test.json   # 테스트 데이터
```

## 설치 및 환경 설정

### 1. 의존성 설치

```bash
# uv를 사용하는 경우
uv sync

# 또는 pip를 사용하는 경우
pip install -e .
```

### 2. 필요한 패키지
- torch >= 2.0.0
- transformers >= 4.35.0
- peft >= 0.7.0
- datasets >= 2.14.0
- accelerate >= 0.24.0

## 사용법

### 1. 모델 훈련

#### 스크립트를 이용한 훈련:
```bash
chmod +x run_train.sh
./run_train.sh
```

#### 직접 명령어로 훈련:
```bash
python train.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --train_data_path resource/RAG/korean_language_rag_V1.0_train.json \
    --val_data_path resource/RAG/korean_language_rag_V1.0_dev.json \
    --output_dir ./results/qwen2.5-0.5b-korean-qa \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32
```

### 2. 모델 추론 테스트

#### 스크립트를 이용한 추론:
```bash
chmod +x run_inference.sh
./run_inference.sh
```

#### 직접 명령어로 추론:
```bash
python run/inference.py \
    --base_model Qwen/Qwen2.5-0.5B \
    --peft_model ./results/qwen2.5-0.5b-korean-qa \
    --test_data resource/RAG/korean_language_rag_V1.0_dev.json \
    --num_samples 10
```

## 데이터셋 구조

훈련 데이터는 다음과 같은 구조를 가집니다:

```json
[
    {
        "id": "1",
        "input": {
            "question_type": "선택형",
            "question": "다음 중 올바른 것을 선택하세요..."
        },
        "output": {
            "answer": "정답 및 설명..."
        }
    }
]
```

### 지원하는 질문 유형:
- **선다형**: 객관식 문제 (숫자로 답변)
- **서술형**: 서술식 답변이 필요한 문제
- **단답형**: 2단어 이내의 간단한 답변
- **교정형**: 문법 오류 교정 문제
- **선택형**: 주어진 선택지 중 선택 후 설명

## 주요 파라미터

### LoRA 설정
- `lora_r`: 16 (LoRA rank)
- `lora_alpha`: 32 (LoRA scaling factor)
- `lora_dropout`: 0.05
- `target_modules`: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 훈련 설정
- `learning_rate`: 2e-4
- `num_train_epochs`: 3
- `batch_size`: 2 (per device)
- `gradient_accumulation_steps`: 8
- `max_seq_length`: 2048

## 결과

훈련이 완료되면 다음 위치에 모델이 저장됩니다:
- LoRA 어댑터: `./results/qwen2.5-0.5b-korean-qa/`
- 훈련 로그: `./logs/`

## 모니터링

TensorBoard를 사용하여 훈련 과정을 모니터링할 수 있습니다:

```bash
tensorboard --logdir ./logs
```

## 문제 해결

### GPU 메모리 부족 시:
- `per_device_train_batch_size`를 1로 줄입니다
- `gradient_accumulation_steps`를 늘립니다
- `max_seq_length`를 줄입니다

### 훈련 속도 개선:
- `fp16` 또는 `bf16` 옵션을 사용합니다
- `dataloader_num_workers`를 조정합니다

## 라이센스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.


