#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 Exact Match 로직 테스트
"""

import json
import re
from transformers import AutoTokenizer
from train_jinyoung import KoreanQAEvaluator

def test_exact_match_improvement():
    """개선된 Exact Match 로직 테스트"""
    
    # 테스트용 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    evaluator = KoreanQAEvaluator(tokenizer)
    
    # 실제 데이터에서 테스트 케이스 추출
    test_cases = [
        {
            "name": "선택형 - 큰따옴표 답변",
            "reference": "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다.",
            "prediction_correct": "\"나는 그를 본 적이 있음을 기억해 냈다.\"가 정답입니다.",
            "prediction_wrong": "\"나는 그를 본 적이 있음을 기억해냈다.\"가 맞습니다."
        },
        {
            "name": "교정형 - 큰따옴표 답변", 
            "reference": "\"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다' 등의 뜻이 있는 말은 '맞추다'로 적는다.",
            "prediction_correct": "\"오늘은 퍼즐 맞추기를 해 볼 거예요.\"",
            "prediction_wrong": "\"오늘은 퍼즐 마추기를 해 볼 거예요.\""
        },
        {
            "name": "패턴 테스트 - 가 옳다",
            "reference": "철수가 학교에 갔다가 맞다.",
            "prediction_correct": "철수가 학교에 갔다",
            "prediction_wrong": "철수가 학교에 간다"
        }
    ]
    
    print("=== 개선된 Exact Match 로직 테스트 ===\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"테스트 {i}: {case['name']}")
        print("-" * 50)
        
        # 정답 추출
        ref_extracted = evaluator.extract_quoted_answer(case['reference'])
        ref_normalized = evaluator.normalize_answer(case['reference'])
        
        print(f"📝 원본 정답: {case['reference'][:60]}...")
        print(f"🎯 추출된 핵심: {ref_extracted}")
        print(f"🔧 정규화 결과: {ref_normalized}")
        print()
        
        # 올바른 예측 테스트
        pred_correct_normalized = evaluator.normalize_answer(case['prediction_correct'])
        is_correct_match = ref_normalized == pred_correct_normalized
        
        print(f"✅ 올바른 예측: {case['prediction_correct']}")
        print(f"🔧 정규화 결과: {pred_correct_normalized}")
        print(f"🎯 매칭 결과: {'✅ 정답' if is_correct_match else '❌ 오답'}")
        print()
        
        # 틀린 예측 테스트
        pred_wrong_normalized = evaluator.normalize_answer(case['prediction_wrong'])
        is_wrong_match = ref_normalized == pred_wrong_normalized
        
        print(f"❌ 틀린 예측: {case['prediction_wrong']}")
        print(f"🔧 정규화 결과: {pred_wrong_normalized}")
        print(f"🎯 매칭 결과: {'✅ 정답' if is_wrong_match else '❌ 오답'}")
        print()
        
        # 결과 요약
        if is_correct_match and not is_wrong_match:
            print("🎉 테스트 통과: 올바른 구분!")
        else:
            print("⚠️ 테스트 실패: 구분 오류!")
        
        print("=" * 60)
        print()

def test_with_real_data():
    """실제 데이터로 테스트"""
    print("=== 실제 데이터 샘플 테스트 ===\n")
    
    # 실제 dev 데이터 로드
    with open('/home/jovyan/jinyoung-llm-models/Korean_QA_RAG_2025/resource/korean_language_rag_V1.0_dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 테스트용 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    evaluator = KoreanQAEvaluator(tokenizer)
    
    # 처음 5개 샘플로 테스트
    for i in range(min(5, len(data))):
        item = data[i]
        answer = item["output"]["answer"]
        
        print(f"샘플 {i+1} (ID: {item['id']}):")
        print(f"📝 원본 답변: {answer[:100]}...")
        
        extracted = evaluator.extract_quoted_answer(answer)
        normalized = evaluator.normalize_answer(answer)
        
        print(f"🎯 추출된 핵심: {extracted}")
        print(f"🔧 정규화 결과: {normalized}")
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_exact_match_improvement()
    test_with_real_data()