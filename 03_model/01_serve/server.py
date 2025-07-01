#!/usr/bin/env python3
"""
Korean QA RAG 모델 REST API 서버
FastAPI를 사용한 추론 서버
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# 환경 변수 설정 (서버 시작 시 설정)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# inference.py에서 필요한 함수들 import
from inference import load_model, create_prompt, generate_answer, generate_answer_streaming

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic 모델들 정의
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: ChatMessage
    done: bool

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 512
    question_type: Optional[str] = "서술형"
    other_info: Optional[Dict[str, Any]] = None

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    generation_time: float

class ModelInfo(BaseModel):
    name: str
    base_model: str
    peft_model: str
    quantization: str
    status: str

# FastAPI 앱 생성
app = FastAPI(
    title="Korean QA RAG API Server",
    description="한국어 QA RAG 모델을 위한 REST API 서버",
    version="1.0.0"
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수들
MODEL = None
TOKENIZER = None
MODEL_LOADING = False
MODEL_LOADING_LOCK = asyncio.Lock()
MODEL_CONFIG = {
    "name": "qwen3-8b-korean-qa",
    "base_model": "Qwen/Qwen3-8B",
    "peft_model": "../00_train/results/qwen3-8b-16bit-lora-korean-qa-rag/checkpoint-85",  # 파인튜닝된 모델 경로
    "use_4bit_quantization": True,  # 파인튜닝할 때와 동일하게 4bit 양자화 사용
    "default_temperature": 0.7,
    "default_top_p": 0.9,
    "default_max_tokens": 512
}

class KoreanQAServer:
    def __init__(self, base_url: str = "http://localhost:11435", model_name: str = "qwen3:8b"):
        self.base_url = base_url
        self.model_name = model_name
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        self.model_loaded = False

    async def load_model_async(self):
        """비동기적으로 모델을 로드합니다. 동시성 제어를 통해 중복 로딩을 방지합니다."""
        global MODEL, TOKENIZER, MODEL_LOADING, MODEL_LOADING_LOCK
        
        # 이미 모델이 로드되어 있으면 바로 반환
        if MODEL is not None and TOKENIZER is not None:
            return MODEL, TOKENIZER
        
        # 동시성 제어를 위한 락 획득
        async with MODEL_LOADING_LOCK:
            # 락을 획득한 후 다시 확인 (다른 요청이 이미 모델을 로드했을 수 있음)
            if MODEL is not None and TOKENIZER is not None:
                return MODEL, TOKENIZER
                
            # 모델 로딩 중인지 확인
            if MODEL_LOADING:
                # 모델 로딩이 완료될 때까지 대기
                while MODEL_LOADING or (MODEL is None or TOKENIZER is None):
                    await asyncio.sleep(1)
                return MODEL, TOKENIZER
            
            # 모델 로딩 시작
            MODEL_LOADING = True
            logger.info("모델 로딩을 시작합니다...")
            
            try:
                MODEL, TOKENIZER = await asyncio.to_thread(
                    load_model,
                    MODEL_CONFIG["base_model"],
                    MODEL_CONFIG["peft_model"],
                    MODEL_CONFIG["use_4bit_quantization"]
                )
                self.model_loaded = True
                logger.info("모델 로딩이 완료되었습니다!")
                
            except Exception as e:
                logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
                raise HTTPException(status_code=500, detail=f"모델 로딩 실패: {str(e)}")
            finally:
                MODEL_LOADING = False
        
        return MODEL, TOKENIZER

# 서버 인스턴스 생성
server = KoreanQAServer(base_url="http://localhost:11435", model_name="qwen3:8b")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델을 로드합니다."""
    logger.info("서버가 시작됩니다. 모델을 로딩 중...")
    background_tasks = BackgroundTasks()
    background_tasks.add_task(server.load_model_async)

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Korean QA RAG API Server",
        "version": "1.0.0",
        "endpoints": ["/api/chat", "/api/generate", "/api/models"]
    }

@app.get("/api/models")
async def get_models():
    """사용 가능한 모델 정보를 반환합니다."""
    return {
        "models": [
            ModelInfo(
                name=MODEL_CONFIG["name"],
                base_model=MODEL_CONFIG["base_model"],
                peft_model=MODEL_CONFIG["peft_model"],
                quantization="4bit" if MODEL_CONFIG["use_4bit_quantization"] else "16bit",
                status="loaded" if server.model_loaded else "loading"
            )
        ]
    }

async def stream_chat_response(model, tokenizer, prompt, generation_config, request_model):
    """채팅 응답을 스트리밍합니다."""
    created_at = datetime.now().isoformat()
    
    async for token in generate_answer_streaming(model, tokenizer, prompt, generation_config):
        yield json.dumps({
            "model": request_model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": token},
            "done": False
        }) + "\n"
    
    # 완료 메시지 전송
    yield json.dumps({
        "model": request_model,
        "created_at": created_at,
        "message": {"role": "assistant", "content": ""},
        "done": True
    }) + "\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """채팅 API 엔드포인트 (Ollama 호환)"""
    global MODEL, TOKENIZER
    
    # 모델이 로드되지 않은 경우 로드
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = await server.load_model_async()
    
    try:
        # 마지막 사용자 메시지 추출
        user_message = None
        for message in reversed(request.messages):
            if message.role == "user":
                user_message = message.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지가 없습니다.")
        
        # 생성 설정
        generation_config = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0.1,  # 온도가 낮으면 샘플링 비활성화
            "repetition_penalty": 1.2  # 반복 방지 페널티 추가
        }
        
        # 기본 프롬프트로 답변 생성 (서술형으로 가정)
        prompt = create_prompt("서술형", user_message)
        
        # 스트리밍 응답 요청인 경우
        if request.stream:
            return StreamingResponse(
                stream_chat_response(MODEL, TOKENIZER, prompt, generation_config, request.model),
                media_type="text/event-stream"
            )
        
        # 일반 응답 요청인 경우
        generated_answer, generation_time = await asyncio.to_thread(
            generate_answer,
            MODEL, TOKENIZER, prompt, generation_config
        )
        
        # 응답 생성
        response = ChatResponse(
            model=request.model,
            created_at=datetime.now().isoformat(),
            message=ChatMessage(role="assistant", content=generated_answer),
            done=True
        )
        
        logger.info(f"Chat 응답 생성 완료 (소요시간: {generation_time:.2f}초)")
        return response
        
    except Exception as e:
        logger.error(f"Chat API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {str(e)}")

async def stream_generate_response(model, tokenizer, prompt, generation_config, request_model):
    """생성 응답을 스트리밍합니다."""
    created_at = datetime.now().isoformat()
    start_time = time.time()
    
    async for token in generate_answer_streaming(model, tokenizer, prompt, generation_config):
        generation_time = time.time() - start_time
        yield json.dumps({
            "model": request_model,
            "created_at": created_at,
            "response": token,
            "done": False,
            "generation_time": generation_time
        }) + "\n"
    
    # 완료 메시지 전송
    generation_time = time.time() - start_time
    yield json.dumps({
        "model": request_model,
        "created_at": created_at,
        "response": "",
        "done": True,
        "generation_time": generation_time
    }) + "\n"

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """텍스트 생성 API 엔드포인트 (Ollama 호환)"""
    global MODEL, TOKENIZER
    
    # 모델이 로드되지 않은 경우 로드
    if MODEL is None or TOKENIZER is None:
        MODEL, TOKENIZER = await server.load_model_async()
    
    try:
        # 생성 설정
        generation_config = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0.1,  # 온도가 낮으면 샘플링 비활성화
            "repetition_penalty": 1.2  # 반복 방지 페널티 추가
        }
        
        # 프롬프트가 이미 형식화된 경우와 질문만 있는 경우 구분
        if request.question_type and not request.prompt.startswith("[질문]"):
            # 질문 타입에 맞는 프롬프트 생성
            prompt = create_prompt(
                request.question_type, 
                request.prompt, 
                request.other_info if request.other_info is not None else {}
            )
        else:
            # 이미 형식화된 프롬프트 사용
            prompt = request.prompt
        
        # 스트리밍 응답 요청인 경우
        if request.stream:
            return StreamingResponse(
                stream_generate_response(MODEL, TOKENIZER, prompt, generation_config, request.model),
                media_type="text/event-stream"
            )
        
        # 일반 응답 요청인 경우
        generated_answer, generation_time = await asyncio.to_thread(
            generate_answer,
            MODEL, TOKENIZER, prompt, generation_config
        )
        
        # 응답 생성
        response = GenerateResponse(
            model=request.model,
            created_at=datetime.now().isoformat(),
            response=generated_answer,
            done=True,
            generation_time=generation_time
        )
        
        logger.info(f"Generate 응답 생성 완료 (소요시간: {generation_time:.2f}초)")
        return response
        
    except Exception as e:
        logger.error(f"Generate API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "model_loaded": server.model_loaded,
        "timestamp": datetime.now().isoformat()
    }

# 개발용 실행 함수
def run_server(host: str = "0.0.0.0", port: int = 11435, reload: bool = False):
    """개발 서버를 실행합니다."""
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Korean QA RAG API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--port", type=int, default=11435, help="서버 포트")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (자동 재시작)")
    
    args = parser.parse_args()
    
    print("Korean QA RAG API Server 시작 중...")
    print(f"서버 주소: http://{args.host}:{args.port}")
    print(f"API 문서: http://{args.host}:{args.port}/docs")
    
    run_server(host=args.host, port=args.port, reload=args.reload)
