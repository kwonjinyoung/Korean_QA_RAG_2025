#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B Korean QA RAG Fine-tuning Script
Qwen3-8B 한국어 QA RAG 파인튜닝 스크립트

Features:
- Qwen3-8B 모델 사용 (양자화 없음, 16bit)
- 멀티GPU 분산 훈련 (RTX A6000 48GB x2)
- 한국어 QA RAG 데이터셋 활용
- 중간 평가 기능
- LoRA 사용 안함 (Full Fine-tuning)
"""

import os
import json
import argparse
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import warnings
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from sklearn.metrics import f1_score

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    IntervalStrategy,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint

# PEFT 라이브러리 임포트
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# 평가 메트릭 라이브러리
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    print("Warning: ROUGE와 NLTK 라이브러리가 없습니다. 기본 평가 메트릭만 사용됩니다.")
    rouge_scorer = None
    sentence_bleu = None

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """모델 관련 파라미터"""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-8B",
        metadata={"help": "사용할 모델 경로 또는 모델 이름"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "모델 캐시 디렉토리"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "빠른 토크나이저 사용 여부"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "원격 코드 신뢰 여부"}
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "모델 데이터 타입"}
    )
    use_8bit_quantization: bool = field(
        default=True,
        metadata={"help": "8비트 양자화 사용 여부"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "LoRA 어댑터 사용 여부"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA 대상 모듈 (쉼표로 구분)"}
    )


@dataclass
class DataArguments:
    """데이터 관련 파라미터"""
    train_data_path: str = field(
        default="resource/korean_language_rag_V1.0_train.json",
        metadata={"help": "훈련 데이터 파일 경로"}
    )
    eval_data_path: str = field(
        default="resource/korean_language_rag_V1.0_dev.json",
        metadata={"help": "평가 데이터 파일 경로"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "최대 시퀀스 길이"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "전처리 워커 수"}
    )


class KoreanQADataset(Dataset):
    """한국어 QA RAG 데이터셋"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 데이터 전처리
        self.data = self._preprocess_data()
        
        logger.info(f"데이터 로드 완료: {len(self.data)}개")
    
    def _preprocess_data(self) -> List[Dict[str, str]]:
        """데이터 전처리"""
        processed_data = []
        
        for item in self.raw_data:
            question = item["input"]["question"]
            answer = item["output"]["answer"]
            question_type = item["input"]["question_type"]
            
            # 프롬프트 템플릿 적용
            prompt = self._create_prompt(question, question_type)
            
            processed_data.append({
                "prompt": prompt,
                "answer": answer,
                "question_type": question_type
            })
        
        return processed_data
    
    def _create_prompt(self, question: str, question_type: str) -> str:
        """한국어 QA RAG용 프롬프트 생성"""
        if question_type == "선택형":
            prompt = f"""다음은 한국어 문법 선택형 문제입니다. 주어진 문제를 읽고 정확한 답을 선택하여 그 이유와 함께 설명해주세요.

문제: {question}

답변:"""
        elif question_type == "교정형":
            prompt = f"""다음은 한국어 문법 교정형 문제입니다. 주어진 문제를 읽고 어문 규범에 맞게 교정하여 그 이유와 함께 설명해주세요.

문제: {question}

답변:"""
        else:
            prompt = f"""다음은 한국어 문법 문제입니다. 주어진 문제를 읽고 정확한 답을 제시해주세요.

문제: {question}

답변:"""
        
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        answer = item["answer"]
        
        # 전체 텍스트 생성 (프롬프트 + 답변)
        full_text = prompt + " " + answer + self.tokenizer.eos_token
        
        # 토큰화
        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 프롬프트 부분 토큰화 (라벨 마스킹용)
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids.squeeze(0)
        attention_mask = tokenized.attention_mask.squeeze(0)
        
        # 라벨 생성 (프롬프트 부분은 -100으로 마스킹)
        labels = input_ids.clone()
        prompt_length = prompt_tokenized.input_ids.shape[1]
        
        # 프롬프트 부분 마스킹
        if prompt_length < len(labels):
            labels[:prompt_length] = -100
        
        # 패딩 토큰 마스킹
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class KoreanQAEvaluator:
    """한국어 QA 전용 평가자"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False) if rouge_scorer else None
        self.smoothing_function = SmoothingFunction().method1 if sentence_bleu else None
    
    def extract_quoted_answer(self, text: str) -> str:
        """큰따옴표 안의 핵심 답변 추출"""
        import re
        
        # 1. 큰따옴표 안의 내용 추출 (한국어 문법 문제의 핵심 답변)
        quote_patterns = [
            r'"([^"]*)"',           # 일반 큰따옴표
            r'"([^"]*)"',           # 왼쪽 큰따옴표
            r'「([^」]*)」',          # 한국어 겹낫표
            r'『([^』]*)』'           # 한국어 겹꺾쇠표
        ]
        
        for pattern in quote_patterns:
            quotes = re.findall(pattern, text)
            if quotes:
                # 첫 번째 큰따옴표 내용이 실제 답변인 경우가 많음
                answer = quotes[0].strip()
                if len(answer) > 2:  # 너무 짧은 답변 제외
                    return answer
        
        # 2. 큰따옴표가 없으면 "~가 옳다" 패턴 찾기
        correct_patterns = [
            r'([^.]*가 옳다)',
            r'([^.]*이 맞다)',
            r'([^.]*가 정답이다)',
            r'([^.]*이 올바르다)'
        ]
        
        for pattern in correct_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0].strip()
        
        # 3. 첫 번째 문장 추출
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()
        
        return text.strip()
    
    def normalize_answer(self, text: str) -> str:
        """답변 정규화 (핵심 답변 추출 후 정규화)"""
        # 먼저 핵심 답변 추출
        core_answer = self.extract_quoted_answer(text)
        
        # 공백 정리
        core_answer = re.sub(r'\s+', ' ', core_answer.strip())
        
        # 불필요한 문구 제거
        remove_patterns = [
            r'가 옳다$',
            r'이 맞다$', 
            r'가 정답이다$',
            r'이 올바르다$',
            r'가 정확하다$'
        ]
        
        for pattern in remove_patterns:
            core_answer = re.sub(pattern, '', core_answer).strip()
        
        # 특수 문자는 유지하되 과도한 문자만 제거
        core_answer = re.sub(r'[""''""『』「」]', '', core_answer)  # 따옴표만 제거
        
        return core_answer.lower().strip()
    
    def extract_answer_from_generation(self, generated_text: str, prompt: str) -> str:
        """생성된 텍스트에서 답변 추출"""
        # 프롬프트 제거
        if prompt in generated_text:
            answer = generated_text.split(prompt)[-1].strip()
        else:
            answer = generated_text.strip()
        
        # 답변 마커 이후 텍스트 추출
        answer_markers = ["답변:", "답:", "정답:", "Answer:"]
        for marker in answer_markers:
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
                break
        
        # EOS 토큰 제거
        if self.tokenizer.eos_token:
            answer = answer.replace(self.tokenizer.eos_token, "")
        
        return answer.strip()
    
    def compute_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """정확한 매칭 점수 계산"""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if self.normalize_answer(pred) == self.normalize_answer(ref):
                exact_matches += 1
        return exact_matches / len(predictions) if predictions else 0.0
    
    def compute_f1_score(self, predictions: List[str], references: List[str]) -> float:
        """F1 점수 계산 (토큰 기반)"""
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self.normalize_answer(pred).split())
            ref_tokens = set(self.normalize_answer(ref).split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
            else:
                common_tokens = pred_tokens & ref_tokens
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)
        
        return np.mean(f1_scores) if f1_scores else 0.0
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """BLEU 점수 계산"""
        if not sentence_bleu:
            return 0.0
        
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = self.normalize_answer(pred).split()
            ref_tokens = [self.normalize_answer(ref).split()]
            
            if len(pred_tokens) == 0:
                bleu_scores.append(0.0)
            else:
                try:
                    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing_function)
                    bleu_scores.append(score)
                except:
                    bleu_scores.append(0.0)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def compute_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """ROUGE 점수 계산"""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        rouge_scores = defaultdict(list)
        for pred, ref in zip(predictions, references):
            pred_norm = self.normalize_answer(pred)
            ref_norm = self.normalize_answer(ref)
            
            if pred_norm and ref_norm:
                scores = self.rouge_scorer.score(ref_norm, pred_norm)
                rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
                rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
                rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
            else:
                rouge_scores["rouge1"].append(0.0)
                rouge_scores["rouge2"].append(0.0)
                rouge_scores["rougeL"].append(0.0)
        
        return {
            "rouge1": np.mean(rouge_scores["rouge1"]),
            "rouge2": np.mean(rouge_scores["rouge2"]),
            "rougeL": np.mean(rouge_scores["rougeL"])
        }


class KoreanQATrainer(Trainer):
    """한국어 QA 전용 트레이너"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # processing_class 또는 직접 전달된 tokenizer 사용
        tokenizer = getattr(self, 'processing_class', None) or self.tokenizer
        self.qa_evaluator = KoreanQAEvaluator(tokenizer)
        self.eval_dataset_raw = None  # 원본 평가 데이터셋 저장
    
    def set_eval_dataset_raw(self, eval_dataset_raw):
        """원본 평가 데이터셋 설정"""
        self.eval_dataset_raw = eval_dataset_raw
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """손실 계산"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # 언어 모델링 손실 계산
        if labels is not None:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """평가 실행 (QA 메트릭 포함)"""
        # 기본 평가 실행
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # QA 전용 평가 실행 (샘플링하여 속도 향상)
        if self.eval_dataset_raw is not None:
            logger.info("*** QA 전용 평가 시작 ***")
            qa_metrics = self.evaluate_qa_performance(sample_size=50)
            eval_results.update(qa_metrics)
            logger.info("*** QA 전용 평가 완료 ***")
        
        return eval_results
    
    def evaluate_qa_performance(self, sample_size: int = 50) -> Dict[str, float]:
        """QA 성능 평가"""
        if self.eval_dataset_raw is None:
            return {}
        
        # 샘플링
        eval_samples = random.sample(self.eval_dataset_raw, min(sample_size, len(self.eval_dataset_raw)))
        
        predictions = []
        references = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in eval_samples:
                prompt = sample["prompt"]
                true_answer = sample["answer"]
                
                # 토큰화 (deprecated 경고 해결)
                tokenizer = getattr(self, 'processing_class', None) or self.tokenizer
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.model.device)
                
                # 생성 (경고 메시지 해결을 위해 그리디 디코딩 사용)
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,  # 그리디 디코딩으로 변경
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # 디코딩
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_answer = self.qa_evaluator.extract_answer_from_generation(generated_text, prompt)
                    
                    predictions.append(predicted_answer)
                    references.append(true_answer)
                    
                except Exception as e:
                    logger.warning(f"생성 중 오류 발생: {e}")
                    predictions.append("")
                    references.append(true_answer)
        
        # 메트릭 계산
        if predictions and references:
            exact_match = self.qa_evaluator.compute_exact_match(predictions, references)
            f1_score = self.qa_evaluator.compute_f1_score(predictions, references)
            bleu_score = self.qa_evaluator.compute_bleu_score(predictions, references)
            rouge_scores = self.qa_evaluator.compute_rouge_score(predictions, references)
            
            qa_metrics = {
                "eval_qa_exact_match": exact_match,
                "eval_qa_f1_score": f1_score,
                "eval_qa_bleu_score": bleu_score,
                "eval_qa_rouge1": rouge_scores["rouge1"],
                "eval_qa_rouge2": rouge_scores["rouge2"],
                "eval_qa_rougeL": rouge_scores["rougeL"]
            }
            
            # 로깅
            logger.info(f"QA 정확도 (Exact Match): {exact_match:.4f}")
            logger.info(f"QA F1 점수: {f1_score:.4f}")
            logger.info(f"QA BLEU 점수: {bleu_score:.4f}")
            logger.info(f"QA ROUGE-1: {rouge_scores['rouge1']:.4f}")
            logger.info(f"QA ROUGE-2: {rouge_scores['rouge2']:.4f}")
            logger.info(f"QA ROUGE-L: {rouge_scores['rougeL']:.4f}")
            
            return qa_metrics
        
        return {}


def setup_model_and_tokenizer(model_args: ModelArguments, data_args: DataArguments):
    """모델과 토크나이저 설정"""
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"토크나이저 설정 완료: {tokenizer.vocab_size}개 토큰")
    
    # 모델 설정 로드
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # 8비트 양자화 설정
    quantization_config = None
    if model_args.use_8bit_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        logger.info("8비트 양자화가 활성화되었습니다.")
    
    # torch_dtype 설정 (양자화 사용 시 None으로 설정)
    torch_dtype = None if quantization_config else (
        getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None
    )
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        quantization_config=quantization_config,
        device_map="auto",  # 멀티GPU 자동 디바이스 매핑
    )
    
    # 8비트 양자화 모델을 훈련을 위해 준비
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
        logger.info("8비트 양자화 모델이 훈련을 위해 준비되었습니다.")
    
    # LoRA 설정
    if model_args.use_lora:
        target_modules = model_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA 어댑터가 적용되었습니다.")
    
    # 토크나이저 크기에 맞게 임베딩 크기 조정
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"모델 로드 완료: {model.num_parameters():,}개 파라미터")
    
    return model, tokenizer, config


def setup_datasets(data_args: DataArguments, tokenizer):
    """데이터셋 설정"""
    
    # 훈련 데이터셋
    train_dataset = KoreanQADataset(
        data_args.train_data_path,
        tokenizer,
        max_length=data_args.max_seq_length
    )
    
    # 평가 데이터셋
    eval_dataset = KoreanQADataset(
        data_args.eval_data_path,
        tokenizer,
        max_length=data_args.max_seq_length
    )
    
    # 원본 평가 데이터 로드 (QA 메트릭용)
    eval_dataset_raw = eval_dataset.data
    
    logger.info(f"훈련 데이터셋: {len(train_dataset)}개")
    logger.info(f"평가 데이터셋: {len(eval_dataset)}개")
    
    return train_dataset, eval_dataset, eval_dataset_raw


def main():
    """메인 함수"""
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description="Qwen3-8B Korean QA RAG Fine-tuning")
    
    # 모델 인수
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B",
                       help="모델 경로 또는 이름")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="모델 캐시 디렉토리")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       help="모델 데이터 타입")
    
    # 데이터 인수
    parser.add_argument("--train_data_path", type=str, 
                       default="resource/korean_language_rag_V1.0_train.json",
                       help="훈련 데이터 파일 경로")
    parser.add_argument("--eval_data_path", type=str,
                       default="resource/korean_language_rag_V1.0_dev.json",
                       help="평가 데이터 파일 경로")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="최대 시퀀스 길이")
    
    # 훈련 인수
    parser.add_argument("--output_dir", type=str, default="./results/qwen3-8b-korean-qa-rag",
                       help="출력 디렉토리")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="훈련 에포크 수")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="디바이스당 훈련 배치 크기")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="디바이스당 평가 배치 크기")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="그래디언트 누적 스텝")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="학습률")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="가중치 감쇠")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="워밍업 비율")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="로깅 간격")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="평가 간격")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="저장 간격")
    parser.add_argument("--save_total_limit", type=int, default=3,
                       help="최대 저장 체크포인트 수")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="로컬 랭크 (분산 훈련용)")
    
    args = parser.parse_args()
    
    # 분산 훈련 초기화 (필요한 경우에만)
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    
    # 시드 설정
    set_seed(args.seed)
    
    # 모델 및 데이터 인수 객체 생성
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=args.torch_dtype,
        use_8bit_quantization=True,  # 8비트 양자화 활성화
        use_lora=True,  # LoRA 활성화
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    
    data_args = DataArguments(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        max_seq_length=args.max_seq_length
    )
    
    # 모델과 토크나이저 설정
    model, tokenizer, config = setup_model_and_tokenizer(model_args, data_args)
    
    # 데이터셋 설정
    train_dataset, eval_dataset, eval_dataset_raw = setup_datasets(data_args, tokenizer)
    
    # 훈련 인수 설정 (메모리 최적화)
    training_args_dict = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "logging_dir": f"{args.output_dir}/logs",
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "tensorboard",
        "run_name": "qwen3-8b-korean-qa-rag",
        "fp16": True,
        "dataloader_num_workers": data_args.preprocessing_num_workers,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "seed": args.seed,
        # 메모리 최적화 설정
        "gradient_checkpointing": True,  # 메모리 vs 속도 트레이드오프
        "max_grad_norm": 1.0,  # 그래디언트 클리핑
        "dataloader_pin_memory": False,  # 핀 메모리 비활성화
        # "deepspeed": "./deepspeed_config.json",  # DeepSpeed 설정 (필요시 활성화)
    }
    
    # 분산 훈련 설정 (필요한 경우에만)
    if args.local_rank != -1:
        training_args_dict["local_rank"] = args.local_rank
        training_args_dict["ddp_backend"] = "nccl"
    
    training_args = TrainingArguments(**training_args_dict)
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 트레이너 설정
    trainer = KoreanQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 원본 평가 데이터셋 설정
    trainer.set_eval_dataset_raw(eval_dataset_raw)
    
    # 체크포인트 확인
    checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoint = get_last_checkpoint(training_args.output_dir)
        if checkpoint is not None:
            logger.info(f"체크포인트에서 재개: {checkpoint}")
    
    # 훈련 시작
    logger.info("*** 훈련 시작 ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 모델 저장 (LoRA 어댑터만 저장)
    if model_args.use_lora:
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info("LoRA 어댑터가 저장되었습니다.")
    else:
        trainer.save_model()
        trainer.save_state()
    
    # 훈련 메트릭 로그
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 최종 평가
    logger.info("*** 최종 평가 ***")
    eval_metrics = trainer.evaluate()
    
    # Perplexity 계산
    try:
        perplexity = torch.exp(torch.tensor(eval_metrics["eval_loss"]))
        eval_metrics["perplexity"] = float(perplexity)
    except OverflowError:
        eval_metrics["perplexity"] = float("inf")
    
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info("*** 훈련 완료 ***")
    logger.info(f"최종 평가 손실: {eval_metrics['eval_loss']:.4f}")
    logger.info(f"Perplexity: {eval_metrics['perplexity']:.4f}")


if __name__ == "__main__":
    main()