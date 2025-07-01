#!/usr/bin/env python3
"""
Korean QA RAG 모델 추론 모듈
Qwen3 모델 로딩 및 추론을 위한 함수들
"""

import os
import time
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, AsyncGenerator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.generation.streamers import TextIteratorStreamer
from threading import Thread

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(base_model_path: str, peft_model_path: Optional[str] = None, use_4bit: bool = True) -> Tuple[Any, Any]:
    """
    모델과 토크나이저를 로드합니다.
    
    Args:
        base_model_path: 기본 모델 경로 (예: "Qwen/Qwen3-32B")
        peft_model_path: PEFT 모델 경로 (예: "./results/qwen3-32b-4bit-korean-qa-improved/checkpoint-160")
        use_4bit: 4bit 양자화 사용 여부
        
    Returns:
        tuple: (model, tokenizer) 튜플
    """
    logger.info(f"모델 로딩 시작: {base_model_path}")
    start_time = time.time()
    
    try:
        # 4bit 양자화 설정
        quantization_config = None
        if use_4bit:
            logger.info("4bit 양자화 설정 적용")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        # PEFT 모델 로드 (있는 경우)
        if peft_model_path:
            logger.info(f"PEFT 모델 로딩: {peft_model_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, peft_model_path)
        
        # 모델을 추론 모드로 설정
        model.eval()
        
        elapsed_time = time.time() - start_time
        logger.info(f"모델 로딩 완료 (소요 시간: {elapsed_time:.2f}초)")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"모델 로딩 오류: {str(e)}")
        raise RuntimeError(f"모델 로딩 실패: {str(e)}")

def create_prompt(question_type: str, question: str, other_info: Optional[Dict[str, Any]] = None) -> str:
    """
    질문 유형에 따라 프롬프트를 생성합니다.
    
    Args:
        question_type: 질문 유형 (예: "서술형", "선택형", "단답형" 등)
        question: 질문 내용
        other_info: 추가 정보 (선택형 문제의 보기 등), None일 수 있음
        
    Returns:
        str: 형식화된 프롬프트
    """
    if question_type == "서술형":
        return f"""[질문]
{question}

[답변]"""
    
    elif question_type == "선택형":
        choices = other_info.get("choices", []) if other_info is not None else []
        choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
        
        return f"""[질문]
{question}

[보기]
{choices_text}

[답변]"""
    
    elif question_type == "단답형":
        return f"""[질문]
{question}

[답변]"""
    
    else:  # 기본 형식
        return f"""[질문]
{question}

[답변]"""

def generate_answer(model, tokenizer, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
    """
    주어진 프롬프트에 대한 답변을 생성합니다.
    
    Args:
        model: 로드된 모델
        tokenizer: 로드된 토크나이저
        prompt: 입력 프롬프트
        generation_config: 생성 설정 (temperature, top_p, max_new_tokens 등)
        
    Returns:
        tuple: (생성된 답변, 생성 시간)
    """
    try:
        # 기본 생성 설정
        if generation_config is None:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 512
            }
        
        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 답변 생성 시작 시간
        start_time = time.time()
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                temperature=generation_config.get("temperature", 0.7),
                top_p=generation_config.get("top_p", 0.9),
                max_new_tokens=generation_config.get("max_new_tokens", 512),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 생성 시간 계산
        generation_time = time.time() - start_time
        
        # 출력 디코딩 및 프롬프트 제거
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output[len(prompt):].strip()
        
        return answer, generation_time
    
    except Exception as e:
        logger.error(f"답변 생성 오류: {str(e)}")
        raise RuntimeError(f"답변 생성 실패: {str(e)}")

async def generate_answer_streaming(model, tokenizer, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
    """
    주어진 프롬프트에 대한 답변을 스트리밍 방식으로 생성합니다.
    
    Args:
        model: 로드된 모델
        tokenizer: 로드된 토크나이저
        prompt: 입력 프롬프트
        generation_config: 생성 설정 (temperature, top_p, max_new_tokens 등)
        
    Yields:
        str: 생성된 토큰들을 순차적으로 반환
    """
    try:
        # 기본 생성 설정
        if generation_config is None:
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 512
            }
        
        # 입력 토큰화
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 스트리머 생성
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 생성 설정
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            temperature=generation_config.get("temperature", 0.7),
            top_p=generation_config.get("top_p", 0.9),
            max_new_tokens=generation_config.get("max_new_tokens", 512),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # 별도 스레드에서 생성 실행
        thread = Thread(target=lambda: model.generate(**generation_kwargs))
        thread.start()
        
        # 생성된 토큰을 스트리밍
        collected_tokens = []
        for token in streamer:
            collected_tokens.append(token)
            # 토큰을 모아서 의미 있는 단위로 반환
            if len(collected_tokens) >= 1 or token.endswith((".", ",", "!", "?", "\n")):
                joined_tokens = "".join(collected_tokens)
                collected_tokens = []
                await asyncio.sleep(0)  # 다른 비동기 작업을 위한 양보
                yield joined_tokens
        
        # 남은 토큰이 있으면 반환
        if collected_tokens:
            yield "".join(collected_tokens)
    
    except Exception as e:
        logger.error(f"스트리밍 답변 생성 오류: {str(e)}")
        yield f"[오류 발생: {str(e)}]"

# 테스트 코드
if __name__ == "__main__":
    # 환경 변수 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 모델 경로 설정
    base_model = "Qwen/Qwen3-8B"
    peft_model = "../00_train/results/qwen3-8b-4bit-lora-korean-qa-rag/checkpoint-110"
    
    # 모델 로드 (파인튜닝할 때와 동일하게 4bit 양자화 사용)
    model, tokenizer = load_model(base_model, peft_model, use_4bit=True)
    
    # 테스트 질문
    test_question = "인공지능이란 무엇인가요?"
    
    # 프롬프트 생성
    prompt = create_prompt("서술형", test_question)
    print(f"프롬프트:\n{prompt}\n")
    
    # 답변 생성
    answer, gen_time = generate_answer(model, tokenizer, prompt)
    print(f"답변 (생성 시간: {gen_time:.2f}초):\n{answer}")
    
    # 스트리밍 테스트
    print("\n스트리밍 테스트:")
    
    async def test_streaming():
        async for token in generate_answer_streaming(model, tokenizer, prompt):
            print(token, end="", flush=True)
        print("\n스트리밍 완료")
    
    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_streaming()) 