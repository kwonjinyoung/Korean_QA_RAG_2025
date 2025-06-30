#!/usr/bin/env python3
"""
Korean QA RAG Fine-tuning Script for Qwen 3-32B with 4-bit Quantization
한국어 QA RAG 데이터셋을 이용한 Qwen 3-32B 모델 4bit 양자화 파인튜닝 스크립트

이 스크립트는 한국어 질문-답변 데이터셋을 사용하여 Qwen 3-32B 모델을 파인튜닝합니다.
메모리 효율성을 위해 4bit 양자화와 LoRA(Low-Rank Adaptation) 기법을 사용합니다.
"""

# 필요한 라이브러리들을 가져옵니다
import os  # 운영체제 관련 기능 (파일 경로, 디렉토리 확인 등)
import json  # JSON 파일 읽기/쓰기
import argparse  # 명령행 인수 파싱
import logging  # 로그 메시지 출력
from dataclasses import dataclass, field  # 데이터 클래스 생성을 위한 데코레이터
from typing import Optional, Dict, Sequence  # 타입 힌트
import warnings  # 경고 메시지 처리

# PyTorch 딥러닝 프레임워크
import torch
# Hugging Face Transformers 라이브러리 - 사전훈련된 모델 사용
import transformers
from transformers import (
    AutoConfig,  # 모델 설정 자동 로드
    AutoModelForCausalLM,  # 언어 생성 모델 자동 로드
    AutoTokenizer,  # 토크나이저 자동 로드
    HfArgumentParser,  # Hugging Face 인수 파서
    TrainingArguments,  # 훈련 관련 설정
    Trainer,  # 훈련 실행 클래스
    DataCollatorForLanguageModeling,  # 언어 모델링용 데이터 콜레이터
    set_seed,  # 재현 가능한 결과를 위한 시드 설정
    EarlyStoppingCallback,  # 조기 중단 콜백
    BitsAndBytesConfig,  # 양자화 설정
)
from transformers.trainer_utils import get_last_checkpoint  # 마지막 체크포인트 찾기
# PEFT(Parameter Efficient Fine-Tuning) 라이브러리 - LoRA 구현
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb  # 실험 추적 도구 (Weights & Biases)

# 사용자 정의 데이터 처리 모듈
from src.data import CustomDataset, DataCollatorForSupervisedDataset

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)  # 미래 버전 관련 경고
warnings.filterwarnings("ignore", category=UserWarning)  # 사용자 경고

# 로그 메시지 형식과 레벨 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # 시간-레벨-이름-메시지 형식
    datefmt="%m/%d/%Y %H:%M:%S",  # 날짜 형식: 월/일/년 시:분:초
    level=logging.INFO,  # INFO 레벨 이상만 표시
)
logger = logging.getLogger(__name__)  # 현재 모듈용 로거 생성


@dataclass
class ModelArguments:
    """
    모델과 관련된 파라미터들을 정의하는 클래스
    @dataclass 데코레이터를 사용하여 자동으로 __init__, __repr__ 등 메서드 생성
    """
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-32B",  # 기본값: Qwen 3-32B 모델
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,  # 기본값: None (모델 경로와 동일)
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,  # 기본값: None (모델 경로와 동일)
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,  # 기본값: None (기본 캐시 디렉토리 사용)
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,  # 빠른 토크나이저 사용 (Rust 기반)
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",  # 기본값: main 브랜치
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,  # 인증 토큰 사용 안함
        metadata={"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script with private models)."}
    )
    torch_dtype: Optional[str] = field(
        default="auto",  # 자동으로 데이터 타입 결정
        metadata={"help": "Override the default `torch.dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True,  # 원격 코드 실행 허용 (사용자 정의 모델을 위해)
        metadata={"help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"}
    )
    use_4bit_quantization: bool = field(
        default=True,  # 4bit 양자화 사용 (메모리 절약)
        metadata={"help": "Whether to use 4-bit quantization for memory efficiency"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",  # 4bit 양자화 시 계산용 데이터 타입
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",  # NormalFloat4 양자화 타입 (더 정확함)
        metadata={"help": "Quantization type for 4-bit (fp4 or nf4)"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,  # 이중 양자화 사용 (더 많은 메모리 절약)
        metadata={"help": "Whether to use double quantization for 4-bit"}
    )


@dataclass
class DataArguments:
    """
    데이터와 관련된 파라미터들을 정의하는 클래스
    훈련/검증 데이터 경로, 최대 시퀀스 길이 등을 설정
    """
    train_data_path: str = field(
        default="resource/RAG/korean_language_rag_V1.0_train.json",  # 훈련 데이터 파일 경로
        metadata={"help": "Path to the training data file"}
    )
    val_data_path: Optional[str] = field(
        default="resource/RAG/korean_language_rag_V1.0_dev.json",  # 검증 데이터 파일 경로
        metadata={"help": "Path to the validation data file"}
    )
    max_seq_length: int = field(
        default=2048,  # 최대 시퀀스 길이 (토큰 개수)
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,  # 전처리용 워커 수 (기본값: CPU 코어 수)
        metadata={"help": "The number of processes to use for the preprocessing."}
    )


@dataclass
class LoraArguments:
    """
    LoRA(Low-Rank Adaptation) 관련 파라미터들을 정의하는 클래스
    LoRA는 대용량 모델을 효율적으로 파인튜닝하는 기법입니다.
    전체 모델을 업데이트하는 대신 작은 어댑터만 훈련시킵니다.
    """
    use_lora: bool = field(
        default=True,  # LoRA 사용 여부
        metadata={"help": "Whether to use LoRA for parameter efficient fine-tuning"}
    )
    lora_r: int = field(
        default=32,  # LoRA rank (낮을수록 파라미터 수 적음, 높을수록 표현력 증가)
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=64,  # LoRA 스케일링 파라미터 (보통 rank의 2배로 설정)
        metadata={"help": "LoRA scaling parameter"}
    )
    lora_dropout: float = field(
        default=0.05,  # LoRA 드롭아웃 비율 (과적합 방지)
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        # LoRA를 적용할 레이어들 (Attention과 MLP 레이어)
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA adaptation (comma separated)"}
    )
    lora_bias: str = field(
        default="none",  # 바이어스 처리 방식 (none: 바이어스 업데이트 안함)
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )


def setup_model_and_tokenizer(model_args: ModelArguments):
    """
    모델과 토크나이저를 설정하는 함수
    
    Args:
        model_args: 모델 관련 설정을 담은 객체
    
    Returns:
        tuple: (모델, 토크나이저, 설정) 튜플 반환
    """
    
    # 토크나이저 로드를 위한 설정 딕셔너리
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,  # 캐시 디렉토리
        "use_fast": model_args.use_fast_tokenizer,  # 빠른 토크나이저 사용 여부
        "revision": model_args.model_revision,  # 모델 버전
        "use_auth_token": model_args.use_auth_token,  # 인증 토큰 사용 여부
        "trust_remote_code": model_args.trust_remote_code,  # 원격 코드 신뢰 여부
    }
    
    # 토크나이저 로드: 별도 토크나이저 이름이 있으면 사용, 없으면 모델과 동일한 것 사용
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    # 패딩 토큰 설정: 패딩 토큰이 없으면 종료 토큰을 패딩 토큰으로 사용
    # 패딩은 배치 내 시퀀스 길이를 맞추기 위해 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 종료 토큰을 패딩 토큰으로 설정
        tokenizer.pad_token_id = tokenizer.eos_token_id  # ID도 동일하게 설정

    # 모델 설정(config) 로드를 위한 설정 딕셔너리
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    # 모델 설정 로드: 별도 설정 이름이 있으면 사용, 없으면 모델과 동일한 것 사용
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    # 4bit 양자화 설정
    # 양자화는 모델의 가중치를 더 적은 비트로 표현하여 메모리 사용량을 줄입니다
    quantization_config = None
    if model_args.use_4bit_quantization:
        # 계산용 데이터 타입을 PyTorch 타입으로 변환 (예: "float16" -> torch.float16)
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # 4bit로 로드
            bnb_4bit_compute_dtype=compute_dtype,  # 계산용 데이터 타입
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,  # 양자화 타입 (nf4 또는 fp4)
            bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,  # 이중 양자화 사용
        )
        logger.info("4bit 양자화 설정이 활성화되었습니다.")

    # torch_dtype 설정
    # "auto"나 None이면 그대로 사용, 아니면 PyTorch 타입으로 변환
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    # 양자화를 사용할 때는 torch_dtype을 None으로 설정 (충돌 방지)
    if quantization_config is not None:
        torch_dtype = None

    # 실제 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,  # 모델 경로 또는 이름
        from_tf=bool(".ckpt" in model_args.model_name_or_path),  # TensorFlow 체크포인트 여부
        config=config,  # 모델 설정
        cache_dir=model_args.cache_dir,  # 캐시 디렉토리
        revision=model_args.model_revision,  # 모델 버전
        use_auth_token=model_args.use_auth_token,  # 인증 토큰
        torch_dtype=torch_dtype,  # 데이터 타입
        trust_remote_code=model_args.trust_remote_code,  # 원격 코드 신뢰
        quantization_config=quantization_config,  # 양자화 설정
        device_map=None
    )

    # 양자화된 모델을 k-bit 훈련을 위해 준비
    # 이 단계에서 그래디언트 계산이 가능하도록 모델을 수정합니다
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
        logger.info("양자화된 모델이 k-bit 훈련을 위해 준비되었습니다.")

    # 토크나이저 크기에 맞게 모델의 임베딩 레이어 크기 조정
    # 새로운 토큰이 추가된 경우 필요합니다
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer, config


def apply_lora(model, lora_args: LoraArguments):
    """
    모델에 LoRA(Low-Rank Adaptation)를 적용하는 함수
    
    LoRA는 대용량 모델을 효율적으로 파인튜닝하는 기법입니다.
    전체 모델의 가중치를 업데이트하는 대신, 작은 어댑터 레이어만 훈련시킵니다.
    
    Args:
        model: 원본 모델
        lora_args: LoRA 관련 설정
    
    Returns:
        LoRA가 적용된 모델 (또는 LoRA를 사용하지 않으면 원본 모델)
    """
    # LoRA를 사용하지 않으면 원본 모델 그대로 반환
    if not lora_args.use_lora:
        return model
    
    # LoRA를 적용할 모듈들을 쉼표로 분리하여 리스트로 변환
    # 예: "q_proj,k_proj,v_proj" -> ["q_proj", "k_proj", "v_proj"]
    target_modules = lora_args.lora_target_modules.split(",") if lora_args.lora_target_modules else None
    
    # LoRA 설정 생성
    lora_config = LoraConfig(
        r=lora_args.lora_r,  # LoRA rank (낮을수록 파라미터 적음)
        lora_alpha=lora_args.lora_alpha,  # 스케일링 파라미터
        target_modules=target_modules,  # LoRA를 적용할 모듈들
        lora_dropout=lora_args.lora_dropout,  # 드롭아웃 비율
        bias=lora_args.lora_bias,  # 바이어스 처리 방식
        task_type=TaskType.CAUSAL_LM,  # 작업 타입: 인과적 언어 모델링
    )
    
    # 모델에 LoRA 적용
    model = get_peft_model(model, lora_config)
    # 훈련 가능한 파라미터 수 출력 (전체 파라미터 대비 매우 적음)
    model.print_trainable_parameters()
    
    return model


def setup_datasets(data_args: DataArguments, tokenizer):
    """
    훈련 및 검증 데이터셋을 설정하는 함수
    
    Args:
        data_args: 데이터 관련 설정
        tokenizer: 토크나이저 객체
    
    Returns:
        tuple: (훈련 데이터셋, 검증 데이터셋) 튜플
    """
    
    # 훈련 데이터셋 생성
    # CustomDataset은 src/data.py에 정의된 사용자 정의 데이터셋 클래스
    train_dataset = CustomDataset(data_args.train_data_path, tokenizer)
    logger.info(f"훈련 데이터셋 크기: {len(train_dataset)}")
    
    # 검증 데이터셋 생성 (선택사항)
    # 검증 데이터 경로가 지정되고 파일이 존재하는 경우에만 생성
    eval_dataset = None
    if data_args.val_data_path and os.path.exists(data_args.val_data_path):
        eval_dataset = CustomDataset(data_args.val_data_path, tokenizer)
        logger.info(f"검증 데이터셋 크기: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    """
    메인 함수: 전체 훈련 프로세스를 실행합니다
    
    1. 명령행 인수 파싱
    2. 로깅 설정
    3. 체크포인트 확인
    4. 모델과 토크나이저 설정
    5. LoRA 적용
    6. 데이터셋 준비
    7. 훈련 실행
    8. 평가 및 모델 저장
    """
    
    # 명령행 인수를 파싱하여 각 설정 클래스의 인스턴스로 변환
    # HfArgumentParser는 dataclass를 기반으로 자동으로 인수를 파싱합니다
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # 로깅 레벨 설정 (훈련 설정의 로그 레벨을 사용)
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)  # Transformers 라이브러리 로그 레벨
    transformers.utils.logging.enable_default_handler()  # 기본 핸들러 활성화
    transformers.utils.logging.enable_explicit_format()  # 명시적 포맷 활성화

    # 마지막 체크포인트 확인
    # 이전에 중단된 훈련을 재개할 수 있는지 확인합니다
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # 출력 디렉토리가 비어있지 않은데 체크포인트가 없으면 오류
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            # 체크포인트가 발견되면 재개 알림
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 재현 가능한 결과를 위한 랜덤 시드 설정
    set_seed(training_args.seed)

    # 모델과 토크나이저 설정
    model, tokenizer, config = setup_model_and_tokenizer(model_args)
    logger.info(f"모델 로드 완료: {model_args.model_name_or_path}")

    # LoRA 적용 (메모리 효율적인 파인튜닝을 위해)
    model = apply_lora(model, lora_args)

    # 훈련 및 검증 데이터셋 설정
    train_dataset, eval_dataset = setup_datasets(data_args, tokenizer)

    # 데이터 콜레이터 설정
    # 배치 단위로 데이터를 처리할 때 패딩, 마스킹 등을 담당합니다
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Weights & Biases 초기화 (실험 추적을 위한 도구)
    if training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project="korean-qa-rag-finetune",  # 프로젝트 이름
            name=f"qwen3-32b-{training_args.run_name or 'default'}",  # 실험 이름
            config={  # 실험 설정 기록
                **vars(model_args),  # 모델 인수들
                **vars(data_args),  # 데이터 인수들
                **vars(training_args),  # 훈련 인수들
                **vars(lora_args),  # LoRA 인수들
            }
        )

    # 조기 중단(Early Stopping) 콜백 설정
    # Loss가 0.05 이하로 떨어지거나 개선이 없으면 훈련을 중단합니다
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,  # 5번의 평가에서 개선이 없으면 중단
        early_stopping_threshold=0.05  # Loss가 0.05 이하가 되면 목표 달성으로 간주
    )
    
    # Trainer 객체 생성
    # Trainer는 HuggingFace에서 제공하는 고수준 훈련 API입니다
    trainer = Trainer(
        model=model,  # 훈련할 모델
        args=training_args,  # 훈련 관련 설정
        train_dataset=train_dataset,  # 훈련 데이터셋
        eval_dataset=eval_dataset,  # 검증 데이터셋
        tokenizer=tokenizer,  # 토크나이저
        data_collator=data_collator,  # 데이터 콜레이터
        callbacks=[early_stopping_callback],  # 콜백 함수들
    )

    # 훈련 실행
    if training_args.do_train:
        # 재개할 체크포인트 결정
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        # 실제 훈련 실행
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 최종 모델 저장

        # 훈련 결과 메트릭 처리
        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # 훈련 메트릭 로그 및 저장
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()  # 훈련 상태 저장

    # 평가 실행
    if training_args.do_eval:
        logger.info("*** 평가 시작 ***")
        metrics = trainer.evaluate()  # 검증 데이터셋으로 평가

        # 평가 샘플 수 메트릭 추가
        max_eval_samples = len(eval_dataset) if eval_dataset else 0
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset)) if eval_dataset else 0

        # Perplexity 계산 (언어 모델의 성능 지표)
        # Perplexity = exp(loss), 낮을수록 좋음
        try:
            perplexity = torch.exp(torch.tensor(metrics["eval_loss"]))
        except OverflowError:
            perplexity = float("inf")  # 오버플로우 시 무한대
        metrics["perplexity"] = float(perplexity)

        # 평가 메트릭 로그 및 저장
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # LoRA 어댑터 저장
    # LoRA를 사용한 경우 어댑터만 저장하면 됩니다 (전체 모델 대비 매우 작음)
    if lora_args.use_lora:
        model.save_pretrained(training_args.output_dir)  # LoRA 어댑터 저장
        tokenizer.save_pretrained(training_args.output_dir)  # 토크나이저 저장
        logger.info(f"LoRA 어댑터가 {training_args.output_dir}에 저장되었습니다.")


def create_training_config():
    """
    기본 훈련 설정을 생성하는 함수
    
    이 함수는 프로그래밍 방식으로 훈련 설정을 생성할 때 사용할 수 있습니다.
    일반적으로는 명령행 인수나 설정 파일을 통해 설정하므로 선택적으로 사용됩니다.
    
    Returns:
        dict: 기본 훈련 설정 딕셔너리
    """
    return {
        # 모델 관련 설정
        "model_name_or_path": "Qwen/Qwen3-32B",  # 사용할 모델
        
        # 양자화 관련 설정 (메모리 절약)
        "use_4bit_quantization": True,  # 4bit 양자화 사용
        "bnb_4bit_compute_dtype": "float16",  # 계산용 데이터 타입
        "bnb_4bit_quant_type": "nf4",  # 양자화 타입
        "bnb_4bit_use_double_quant": True,  # 이중 양자화 사용
        
        # 데이터 관련 설정
        "train_data_path": "resource/RAG/korean_language_rag_V1.0_train.json",  # 훈련 데이터
        "val_data_path": "resource/RAG/korean_language_rag_V1.0_dev.json",  # 검증 데이터
        "max_seq_length": 2048,  # 최대 시퀀스 길이
        
        # 출력 및 로깅 설정
        "output_dir": "./results",  # 결과 저장 디렉토리
        "overwrite_output_dir": True,  # 기존 디렉토리 덮어쓰기
        "do_train": True,  # 훈련 실행
        "do_eval": True,  # 평가 실행
        "evaluation_strategy": "steps",  # 평가 전략
        "eval_steps": 100,  # 평가 간격
        "save_strategy": "steps",  # 저장 전략
        "save_steps": 100,  # 저장 간격
        "save_total_limit": 2,  # 최대 저장 모델 수
        "logging_steps": 10,  # 로깅 간격
        "logging_dir": "./logs",  # 로그 디렉토리
        
        # 훈련 하이퍼파라미터
        "num_train_epochs": 5,  # 훈련 에포크 수
        "per_device_train_batch_size": 1,  # 디바이스당 훈련 배치 크기
        "per_device_eval_batch_size": 2,  # 디바이스당 평가 배치 크기
        "gradient_accumulation_steps": 16,  # 그래디언트 누적 스텝
        "learning_rate": 1e-4,  # 학습률
        "weight_decay": 0.01,  # 가중치 감쇠
        "warmup_ratio": 0.03,  # 워밍업 비율
        "lr_scheduler_type": "cosine",  # 학습률 스케줄러
        
        # LoRA 관련 설정
        "use_lora": True,  # LoRA 사용
        "lora_r": 32,  # LoRA rank
        "lora_alpha": 64,  # LoRA alpha
        "lora_dropout": 0.05,  # LoRA 드롭아웃
        
        # 기타 설정
        "seed": 42,  # 랜덤 시드
        "dataloader_num_workers": 0,  # 데이터 로더 워커 수
        "report_to": "tensorboard",  # 리포트 도구
        "run_name": "korean-qa-rag-finetune",  # 실험 이름
    }


# 스크립트가 직접 실행될 때만 main() 함수 호출
# 다른 모듈에서 import할 때는 실행되지 않습니다
if __name__ == "__main__":
    main()
