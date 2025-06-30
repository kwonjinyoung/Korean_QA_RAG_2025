#!/usr/bin/env python3
"""
훈련된 모델 추론 스크립트 (개선된 버전)
Enhanced inference script for fine-tuned Korean QA model
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig
)
from peft import PeftModel
import json
import argparse
import time
from datetime import datetime
import warnings

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(base_model_path, peft_model_path=None, use_4bit_quantization=True):
    """모델과 토크나이저를 로드합니다 (train.py와 동일한 방식)."""
    print(f"기본 모델 로딩: {base_model_path}")
    
    # 토크나이저 로드 (train.py와 동일한 설정)
    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": True,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **tokenizer_kwargs)
    
    # 패딩 토큰 설정 (train.py와 동일한 방식)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 모델 설정 로드
    config = AutoConfig.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    # 4bit 양자화 설정 (train.py와 동일한 설정)
    quantization_config = None
    if use_4bit_quantization:
        compute_dtype = torch.float16  # train.py 기본값
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",  # train.py 기본값
            bnb_4bit_use_double_quant=True,  # train.py 기본값
        )
        print("4bit 양자화 설정이 활성화되었습니다.")
    
    # torch_dtype 설정 (train.py와 동일한 방식)
    torch_dtype = None if quantization_config is not None else torch.float16
    
    # 모델 로드 (train.py와 동일한 방식)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    # LoRA 어댑터 로드
    if peft_model_path:
        print(f"LoRA 어댑터 로딩: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
        print(f"LoRA 어댑터 로드 완료: {peft_model_path}")
    
    # 토크나이저 크기에 맞게 모델의 임베딩 레이어 크기 조정 (train.py와 동일)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def create_prompt(question_type, question, other_info=None):
    """질문 타입에 맞는 프롬프트를 생성합니다."""
    
    type_instructions = {
        "선다형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
        ),
        "서술형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
        ),
        "단답형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        ),
        "교정형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        ),
        "선택형": (
            "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
            "[지침]\n"
            "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        )
    }
    
    instruction = type_instructions.get(question_type, "")
    
    chat_parts = [instruction]
    
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            info_list.append(f"- {key}: {value}")
        chat_parts.append("\n".join(info_list))
    
    chat_parts.append(f"[질문]\n{question}")
    
    return "\n\n".join(chat_parts)


def generate_answer(model, tokenizer, prompt, generation_config=None):
    """향상된 답변 생성 함수입니다."""
    
    SYSTEM_PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. \
        당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
        단, 동일한 문장을 절대 반복하지 마시오.'''
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)
    
    # 기본 생성 설정
    default_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    if generation_config:
        default_config.update(generation_config)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(text, **default_config)
    
    generation_time = time.time() - start_time
    
    # 프롬프트 부분 제거하고 생성된 답변만 추출
    generated_text = tokenizer.batch_decode(outputs[:, text.shape[-1]:], skip_special_tokens=True)[0]
    
    return generated_text.strip(), generation_time


def evaluate_answer(generated_answer, correct_answer):
    """간단한 답변 평가 함수입니다."""
    if not correct_answer:
        return {"evaluation": "정답 없음"}
    
    # 정확한 일치 검사
    exact_match = generated_answer.strip() == correct_answer.strip()
    
    # 주요 키워드 포함 여부 검사
    keywords = []
    if "가 옳다" in correct_answer:
        if "가 옳다" in generated_answer or "이 옳다" in generated_answer:
            keywords.append("적절한 형식")
    
    return {
        "exact_match": exact_match,
        "keywords_found": keywords,
        "evaluation": "정확" if exact_match else "부분적" if keywords else "부정확"
    }


def test_model(model, tokenizer, test_cases, generation_config=None):
    """테스트 케이스들로 모델을 평가합니다."""
    results = []
    total_time = 0
    correct_count = 0
    
    print(f"\n{'='*60}")
    print(f"추론 테스트 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    for i, test_case in enumerate(test_cases):
        print(f"\n=== 테스트 케이스 {i+1}/{len(test_cases)} ===")
        
        # 입력 정보
        input_data = test_case["input"]
        question_type = input_data["question_type"]
        question = input_data["question"]
        other_info = {k: v for k, v in input_data.items() if k not in ['question', 'question_type']}
        
        # 프롬프트 생성
        prompt = create_prompt(question_type, question, other_info if other_info else None)
        
        print(f"질문 유형: {question_type}")
        print(f"질문: {question}")
        
        # 답변 생성
        generated_answer, gen_time = generate_answer(model, tokenizer, prompt, generation_config)
        total_time += gen_time
        
        print(f"생성된 답변: {generated_answer}")
        print(f"생성 시간: {gen_time:.2f}초")
        
        # 정답과 비교 (있는 경우)
        evaluation = {}
        if "output" in test_case:
            correct_answer = test_case["output"].get("answer", "")
            print(f"정답: {correct_answer}")
            
            evaluation = evaluate_answer(generated_answer, correct_answer)
            print(f"평가: {evaluation['evaluation']}")
            
            if evaluation.get("exact_match", False):
                correct_count += 1
        
        results.append({
            "test_case_id": i + 1,
            "question_type": question_type,
            "question": question,
            "generated_answer": generated_answer,
            "correct_answer": test_case.get("output", {}).get("answer", ""),
            "generation_time": gen_time,
            "evaluation": evaluation
        })
        
        print("-" * 80)
    
    # 요약 통계
    avg_time = total_time / len(test_cases)
    accuracy = correct_count / len(test_cases) if test_cases else 0
    
    summary = {
        "total_cases": len(test_cases),
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "average_generation_time": avg_time,
        "total_time": total_time
    }
    
    print(f"\n{'='*60}")
    print(f"추론 테스트 완료")
    print(f"{'='*60}")
    print(f"총 테스트 케이스: {summary['total_cases']}")
    print(f"정답 수: {summary['correct_answers']}")
    print(f"정확도: {summary['accuracy']:.2%}")
    print(f"평균 생성 시간: {summary['average_generation_time']:.2f}초")
    print(f"총 소요 시간: {summary['total_time']:.2f}초")
    
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="훈련된 모델 추론 테스트 (개선된 버전)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B", help="기본 모델 경로")
    parser.add_argument("--peft_model", type=str, default="./results/qwen3-8b-korean-qa", help="LoRA 모델 경로")
    parser.add_argument("--test_data", type=str, default="resource/RAG/korean_language_rag_V1.0_dev.json", help="테스트 데이터 경로")
    parser.add_argument("--num_samples", type=int, default=10, help="테스트할 샘플 수")
    parser.add_argument("--temperature", type=float, default=0.7, help="생성 온도")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 값")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 생성 토큰 수")
    parser.add_argument("--use_4bit_quantization", action="store_true", default=True, help="4bit 양자화 사용")
    parser.add_argument("--no_4bit_quantization", dest="use_4bit_quantization", action="store_false", help="4bit 양자화 사용 안함")
    
    args = parser.parse_args()
    
    # 모델 로드
    print("모델 로딩 중...")
    model, tokenizer = load_model(args.base_model, args.peft_model, args.use_4bit_quantization)
    print("모델 로딩 완료!")
    
    # 테스트 데이터 로드
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 일부 샘플만 테스트
    test_samples = test_data[:args.num_samples]
    
    # 생성 설정
    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens
    }
    
    # 추론 테스트
    results, summary = test_model(model, tokenizer, test_samples, generation_config)
    
    # 결과 저장
    output_file = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "detailed_results": results,
            "config": {
                "base_model": args.base_model,
                "peft_model": args.peft_model,
                "test_data": args.test_data,
                "generation_config": generation_config
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n추론 결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main() 