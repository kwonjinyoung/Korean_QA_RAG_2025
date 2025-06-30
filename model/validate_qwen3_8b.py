#!/usr/bin/env python3
"""
Qwen3-8B 모델 검증 스크립트
Validation script for Qwen3-8B Korean QA model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import argparse
import time
from datetime import datetime
import os


def load_qwen3_8b_model(base_model_path="Qwen/Qwen3-8B", peft_model_path=None):
    """Qwen3-8B 모델과 토크나이저를 로드합니다."""
    print(f"Qwen3-8B 기본 모델 로딩: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    if peft_model_path and os.path.exists(peft_model_path):
        print(f"LoRA 어댑터 로딩: {peft_model_path}")
        model = PeftModel.from_pretrained(model, peft_model_path)
        print(f"LoRA 어댑터 로드 완료: {peft_model_path}")
    else:
        print(f"경고: LoRA 어댑터 경로가 존재하지 않습니다: {peft_model_path}")
        print("기본 Qwen3-8B 모델로 진행합니다.")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def validate_korean_qa(model, tokenizer, test_data_path, num_samples=10):
    """한국어 QA 데이터셋으로 모델을 검증합니다."""
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if num_samples > 0:
        test_data = test_data[:num_samples]
    
    results = []
    correct_count = 0
    total_time = 0
    
    print(f"\n{'='*60}")
    print(f"Qwen3-8B 모델 검증 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 {len(test_data)}개 샘플 검증")
    print(f"{'='*60}")
    
    for i, test_case in enumerate(test_data):
        print(f"\n=== 검증 케이스 {i+1}/{len(test_data)} ===")
        
        # 입력 정보
        input_data = test_case["input"]
        question_type = input_data["question_type"]
        question = input_data["question"]
        
        # 프롬프트 생성
        prompt = create_korean_qa_prompt(question_type, question)
        
        print(f"질문 유형: {question_type}")
        print(f"질문: {question}")
        
        # 답변 생성
        start_time = time.time()
        generated_answer = generate_korean_answer(model, tokenizer, prompt)
        gen_time = time.time() - start_time
        total_time += gen_time
        
        print(f"생성된 답변: {generated_answer}")
        print(f"생성 시간: {gen_time:.2f}초")
        
        # 정답과 비교
        evaluation = {}
        if "output" in test_case:
            correct_answer = test_case["output"].get("answer", "")
            print(f"정답: {correct_answer}")
            
            evaluation = evaluate_korean_answer(generated_answer, correct_answer)
            print(f"평가: {evaluation['evaluation']}")
            
            if evaluation.get("exact_match", False):
                correct_count += 1
        
        # 결과 저장
        result = {
            "test_case_id": i + 1,
            "question_type": question_type,
            "question": question,
            "generated_answer": generated_answer,
            "correct_answer": correct_answer if "output" in test_case else "",
            "generation_time": gen_time,
            "evaluation": evaluation
        }
        results.append(result)
    
    # 전체 결과 요약
    accuracy = correct_count / len(test_data) if test_data else 0
    avg_time = total_time / len(test_data) if test_data else 0
    
    summary = {
        "model": "Qwen3-8B",
        "total_cases": len(test_data),
        "correct_answers": correct_count,
        "accuracy": accuracy,
        "average_generation_time": avg_time,
        "total_time": total_time
    }
    
    print(f"\n{'='*60}")
    print(f"검증 완료!")
    print(f"정확도: {accuracy:.2%} ({correct_count}/{len(test_data)})")
    print(f"평균 생성 시간: {avg_time:.2f}초")
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"{'='*60}")
    
    return {
        "summary": summary,
        "detailed_results": results
    }


def create_korean_qa_prompt(question_type, question):
    """한국어 QA를 위한 프롬프트를 생성합니다."""
    
    type_instructions = {
        "선택형": (
            "[지침] 주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        ),
        "교정형": (
            "[지침] 주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
        ),
        "서술형": (
            "[지침] 질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
        ),
        "단답형": (
            "[지침] 질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
        ),
        "선다형": (
            "[지침] 주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
        )
    }
    
    instruction = type_instructions.get(question_type, "")
    return f"{instruction}[질문]\n{question}"


def generate_korean_answer(model, tokenizer, prompt):
    """Qwen3-8B로 한국어 답변을 생성합니다."""
    
    SYSTEM_PROMPT = '''당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하고 정확하게 답변해주세요.'''
    
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
    
    generation_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    with torch.no_grad():
        outputs = model.generate(text, **generation_config)
    
    generated_text = tokenizer.batch_decode(outputs[:, text.shape[-1]:], skip_special_tokens=True)[0]
    return generated_text.strip()


def evaluate_korean_answer(generated_answer, correct_answer):
    """한국어 답변을 평가합니다."""
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B 한국어 QA 모델 검증")
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B", help="기본 모델 경로")
    parser.add_argument("--peft_model", default="./results/qwen3-8b-korean-qa-improved", help="LoRA 어댑터 경로")
    parser.add_argument("--test_data", default="resource/RAG/korean_language_rag_V1.0_dev.json", help="테스트 데이터 경로")
    parser.add_argument("--num_samples", type=int, default=10, help="검증할 샘플 수 (0은 전체)")
    parser.add_argument("--output", default=None, help="결과 저장 파일명")
    
    args = parser.parse_args()
    
    # 모델 로드
    model, tokenizer = load_qwen3_8b_model(args.base_model, args.peft_model)
    
    # 검증 실행
    results = validate_korean_qa(model, tokenizer, args.test_data, args.num_samples)
    
    # 결과 저장
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        output_file = f"qwen3_8b_validation_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n검증 결과가 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    main() 