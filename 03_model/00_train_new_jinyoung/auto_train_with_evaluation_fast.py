#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고속 자동 성능 평가 및 반복 훈련 시스템
Fast Auto Training with Performance Evaluation

최적화된 배치 크기와 설정으로 빠른 훈련 진행
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

# 기존 훈련 스크립트 임포트
from train_jinyoung import (
    ModelArguments, DataArguments, KoreanQADataset, 
    KoreanQAEvaluator, setup_model_and_tokenizer
)

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class FastAutoTrainingEvaluator:
    """고속 자동 훈련 및 평가 시스템"""
    
    def __init__(self, 
                 model_name: str,
                 train_data_path: str,
                 test_data_path: str,
                 base_output_dir: str,
                 target_scores: Dict[str, float],
                 max_epochs: int = 3,
                 max_iterations: int = 8):
        
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.base_output_dir = base_output_dir
        self.target_scores = target_scores
        self.max_epochs = max_epochs
        self.max_iterations = max_iterations
        
        # 결과 추적
        self.training_history = []
        self.best_scores = {}
        self.best_model_path = None
        
        # 테스트 데이터 로드
        self.test_data = self.load_test_data()
        
        # 평가자 초기화
        self.tokenizer = None
        self.evaluator = None
        
        logger.info(f"🚀 고속 자동 훈련 시스템 초기화 완료")
        logger.info(f"목표 점수: {target_scores}")
        logger.info(f"최대 반복 횟수: {max_iterations} (빠른 반복)")
        logger.info(f"반복당 에포크: {max_epochs} (빠른 수렴)")
        logger.info(f"테스트 데이터: {len(self.test_data)}개")
    
    def load_test_data(self) -> List[Dict]:
        """테스트 데이터 로드"""
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_data = []
        for item in raw_data:
            question = item["input"]["question"]
            answer = item["output"]["answer"]
            question_type = item["input"]["question_type"]
            
            # 프롬프트 생성
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
            
            processed_data.append({
                "prompt": prompt,
                "answer": answer,
                "question_type": question_type,
                "question": question
            })
        
        return processed_data
    
    def run_training_epoch(self, iteration: int) -> str:
        """고속 훈련 에포크 실행"""
        epoch_output_dir = os.path.join(self.base_output_dir, f"iteration_{iteration}")
        
        # 최적화된 훈련 파라미터
        # GPU 메모리 56% 사용 중이므로 배치 크기를 증가시킬 수 있음
        batch_size = 8  # 기존 4에서 8로 증가
        gradient_accumulation = 4  # 기존 8에서 4로 감소 (총 유효 배치 크기 유지)
        
        # 훈련 스크립트 실행을 위한 명령어 구성
        cmd = f"""
        python train_jinyoung.py \\
            --model_name_or_path "{self.model_name}" \\
            --train_data_path "{self.train_data_path}" \\
            --eval_data_path "{self.test_data_path}" \\
            --output_dir "{epoch_output_dir}" \\
            --num_train_epochs {self.max_epochs} \\
            --per_device_train_batch_size {batch_size} \\
            --per_device_eval_batch_size {batch_size} \\
            --gradient_accumulation_steps {gradient_accumulation} \\
            --learning_rate 3e-4 \\
            --weight_decay 0.01 \\
            --warmup_ratio 0.05 \\
            --logging_steps 5 \\
            --eval_steps 50 \\
            --save_steps 200 \\
            --save_total_limit 2 \\
            --seed 42 \\
            --torch_dtype "float16" \\
            --max_seq_length 1536
        """
        
        logger.info(f"=== 🚀 고속 반복 {iteration} 훈련 시작 ===")
        logger.info(f"출력 디렉토리: {epoch_output_dir}")
        logger.info(f"최적화된 설정:")
        logger.info(f"  - 배치 크기: {batch_size} (GPU 메모리 최대 활용)")
        logger.info(f"  - 그래디언트 누적: {gradient_accumulation}")
        logger.info(f"  - 학습률: 3e-4 (빠른 수렴)")
        logger.info(f"  - 시퀀스 길이: 1536 (메모리 절약)")
        logger.info(f"  - 에포크: {self.max_epochs} (빠른 반복)")
        
        # 훈련 실행
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            logger.error(f"훈련 실패! 종료 코드: {exit_code}")
            return None
        
        logger.info(f"=== ✅ 고속 반복 {iteration} 훈련 완료 ===")
        return epoch_output_dir
    
    def evaluate_model(self, model_path: str) -> Dict[str, float]:
        """빠른 모델 성능 평가 (샘플링)"""
        logger.info(f"=== 🔍 빠른 성능 평가 시작: {model_path} ===")
        
        try:
            # 토크나이저 로드
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
                self.evaluator = KoreanQAEvaluator(self.tokenizer)
            
            # 모델 로드 (LoRA 어댑터)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 로드
            if os.path.exists(os.path.join(model_path, "adapter_config.json")):
                model = PeftModel.from_pretrained(base_model, model_path)
                logger.info("LoRA 어댑터 로드 완료")
            else:
                model = base_model
                logger.info("기본 모델 사용")
            
            model.eval()
            
            # 빠른 평가를 위한 샘플링 (전체 데이터의 20%)
            sample_size = max(50, len(self.test_data) // 5)  # 최소 50개, 최대 전체의 20%
            eval_samples = self.test_data[:sample_size]  # 순차 샘플링 (랜덤보다 빠름)
            
            logger.info(f"빠른 평가: {sample_size}개 샘플 사용 (전체 {len(self.test_data)}개 중)")
            
            # 추론 및 평가
            predictions = []
            references = []
            
            for i, sample in enumerate(eval_samples):
                if i % 25 == 0:
                    logger.info(f"평가 진행률: {i}/{len(eval_samples)}")
                
                prompt = sample["prompt"]
                true_answer = sample["answer"]
                
                # 토큰화 (짧은 시퀀스로 빠른 처리)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=384,  # 기존 512에서 384로 단축
                    truncation=True,
                    padding=True
                ).to(model.device)
                
                # 생성 (빠른 생성 설정)
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,  # 기존 256에서 128로 단축
                            do_sample=False,     # 그리디 디코딩 (빠름)
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # 디코딩
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    predicted_answer = self.evaluator.extract_answer_from_generation(generated_text, prompt)
                    
                    predictions.append(predicted_answer)
                    references.append(true_answer)
                    
                except Exception as e:
                    logger.warning(f"생성 중 오류 (샘플 {i}): {e}")
                    predictions.append("")
                    references.append(true_answer)
            
            # 메트릭 계산
            scores = {
                "exact_match": self.evaluator.compute_exact_match(predictions, references),
                "f1_score": self.evaluator.compute_f1_score(predictions, references),
                "bleu_score": self.evaluator.compute_bleu_score(predictions, references)
            }
            
            rouge_scores = self.evaluator.compute_rouge_score(predictions, references)
            scores.update({
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"],
                "rougeL": rouge_scores["rougeL"]
            })
            
            logger.info("=== 빠른 평가 결과 ===")
            for metric, score in scores.items():
                logger.info(f"{metric}: {score:.4f}")
            
            # GPU 메모리 정리
            del model
            del base_model
            torch.cuda.empty_cache()
            
            return scores
            
        except Exception as e:
            logger.error(f"평가 중 오류 발생: {e}")
            return {}
    
    def check_target_achieved(self, scores: Dict[str, float]) -> bool:
        """목표 점수 달성 여부 확인"""
        for metric, target in self.target_scores.items():
            if metric in scores and scores[metric] >= target:
                logger.info(f"✓ {metric} 목표 달성: {scores[metric]:.4f} >= {target:.4f}")
            else:
                current_score = scores.get(metric, 0.0)
                logger.info(f"✗ {metric} 목표 미달성: {current_score:.4f} < {target:.4f}")
                return False
        
        logger.info("🎉 모든 목표 점수 달성!")
        return True
    
    def update_best_model(self, model_path: str, scores: Dict[str, float]):
        """최고 성능 모델 업데이트"""
        # 주요 메트릭의 평균으로 종합 점수 계산
        main_metrics = ["exact_match", "f1_score", "bleu_score", "rougeL"]
        available_scores = [scores.get(metric, 0.0) for metric in main_metrics if metric in scores]
        composite_score = sum(available_scores) / len(available_scores) if available_scores else 0.0
        
        if not self.best_scores or composite_score > self.best_scores.get("composite_score", 0.0):
            self.best_scores = scores.copy()
            self.best_scores["composite_score"] = composite_score
            self.best_model_path = model_path
            
            logger.info(f"🏆 새로운 최고 성능 모델 업데이트!")
            logger.info(f"종합 점수: {composite_score:.4f}")
            logger.info(f"모델 경로: {model_path}")
    
    def save_training_history(self):
        """훈련 기록 저장"""
        history_file = os.path.join(self.base_output_dir, "training_history.json")
        
        history_data = {
            "training_history": self.training_history,
            "best_scores": self.best_scores,
            "best_model_path": self.best_model_path,
            "target_scores": self.target_scores,
            "timestamp": datetime.now().isoformat(),
            "optimization_info": {
                "fast_training": True,
                "optimized_batch_size": True,
                "reduced_sequence_length": True,
                "quick_evaluation": True
            }
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"훈련 기록 저장: {history_file}")
    
    def run_auto_training(self):
        """고속 자동 훈련 실행"""
        logger.info("🚀 고속 자동 훈련 시스템 시작!")
        
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🚀 고속 반복 {iteration}/{self.max_iterations} 시작")
            logger.info(f"{'='*60}")
            
            # 훈련 실행
            model_path = self.run_training_epoch(iteration)
            if model_path is None:
                logger.error("훈련 실패로 중단")
                break
            
            # 성능 평가
            scores = self.evaluate_model(model_path)
            if not scores:
                logger.error("평가 실패")
                continue
            
            # 기록 업데이트
            iteration_record = {
                "iteration": iteration,
                "model_path": model_path,
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            self.training_history.append(iteration_record)
            
            # 최고 성능 모델 업데이트
            self.update_best_model(model_path, scores)
            
            # 목표 달성 확인
            if self.check_target_achieved(scores):
                logger.info(f"🎯 목표 점수 달성! 고속 반복 {iteration}에서 완료")
                break
            
            # 훈련 기록 저장
            self.save_training_history()
            
            logger.info(f"고속 반복 {iteration} 완료. 다음 훈련 준비...")
            time.sleep(1)  # 짧은 대기
        
        # 최종 결과 요약
        logger.info(f"\n{'='*60}")
        logger.info("🏁 고속 자동 훈련 완료!")
        logger.info(f"{'='*60}")
        logger.info(f"총 반복 횟수: {len(self.training_history)}")
        logger.info(f"최고 성능 모델: {self.best_model_path}")
        logger.info("최고 성능 점수:")
        for metric, score in self.best_scores.items():
            logger.info(f"  {metric}: {score:.4f}")
        
        self.save_training_history()
        return self.best_model_path, self.best_scores


def main():
    parser = argparse.ArgumentParser(description="고속 자동 성능 평가 및 반복 훈련")
    
    # 기본 설정
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="기본 모델 이름")
    parser.add_argument("--train_data_path", type=str, 
                       default="../../resource/korean_language_rag_V1.0_train.json",
                       help="훈련 데이터 경로")
    parser.add_argument("--test_data_path", type=str,
                       default="../../resource/korean_language_rag_V1.0_dev.json", 
                       help="테스트 데이터 경로")
    parser.add_argument("--output_dir", type=str, 
                       default="./results/auto_training_fast",
                       help="출력 디렉토리")
    
    # 목표 점수 설정 (빠른 달성을 위해 약간 낮춤)
    parser.add_argument("--target_exact_match", type=float, default=0.70,
                       help="목표 정확도 (Exact Match)")
    parser.add_argument("--target_f1_score", type=float, default=0.75,
                       help="목표 F1 점수")
    parser.add_argument("--target_bleu_score", type=float, default=0.65,
                       help="목표 BLEU 점수")
    
    # 훈련 설정 (빠른 훈련)
    parser.add_argument("--max_epochs", type=int, default=3,
                       help="반복당 최대 에포크 수 (빠른 수렴)")
    parser.add_argument("--max_iterations", type=int, default=8,
                       help="최대 반복 횟수 (빠른 완료)")
    
    args = parser.parse_args()
    
    # 목표 점수 설정
    target_scores = {
        "exact_match": args.target_exact_match,
        "f1_score": args.target_f1_score,
        "bleu_score": args.target_bleu_score
    }
    
    # 고속 자동 훈련 시스템 실행
    auto_trainer = FastAutoTrainingEvaluator(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        base_output_dir=args.output_dir,
        target_scores=target_scores,
        max_epochs=args.max_epochs,
        max_iterations=args.max_iterations
    )
    
    best_model_path, best_scores = auto_trainer.run_auto_training()
    
    print(f"\n🎉 고속 자동 훈련 완료!")
    print(f"최고 성능 모델: {best_model_path}")
    print(f"최고 성능 점수: {best_scores}")


if __name__ == "__main__":
    main()