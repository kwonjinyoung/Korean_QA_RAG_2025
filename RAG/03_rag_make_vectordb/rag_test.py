import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
import time
import logging

# HTTP API 기반 Qdrant 클라이언트 임포트
from makeDB_http import KoreanRAGVectorDB_HTTP

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaQwenChat:
    """Ollama Qwen3:8b-fp16 모델을 사용한 채팅 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen3:8b-fp16"):
        self.base_url = base_url
        self.model_name = model_name
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        
    def chat_with_context(
        self, 
        query: str, 
        context: str, 
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """컨텍스트를 포함한 채팅 응답 생성"""
        try:
            # 시스템 프롬프트 설정
            if system_prompt is None:
                system_prompt = """당신은 한국어 언어학 및 문법 전문가입니다. 
주어진 컨텍스트 정보를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.
컨텍스트에 없는 정보는 추측하지 말고, 컨텍스트 기반으로만 답변하세요."""

            # 프롬프트 구성
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"""다음 컨텍스트 정보를 참고하여 질문에 답변해주세요.

**컨텍스트:**
{context}

**질문:** {query}

**답변:**"""
                }
            ]
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            logger.info(f"Qwen3 모델에게 질문: {query[:50]}...")
            start_time = time.time()
            
            response = requests.post(self.chat_url, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("message", {}).get("content", "")
            
            end_time = time.time()
            logger.info(f"✅ Qwen3 응답 생성 완료 (소요시간: {end_time - start_time:.2f}초)")
            
            return answer.strip() if answer else None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Ollama API 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ 채팅 응답 생성 중 오류: {e}")
            return None
    
    def simple_generate(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """간단한 텍스트 생성"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(self.generate_url, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"❌ 텍스트 생성 실패: {e}")
            return None

class KoreanRAGSystem:
    """한국어 RAG 시스템 (HTTP API 기반)"""
    
    def __init__(
        self,
        collection_name: str = "korean_rag_http_collection",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        llm_model: str = "qwen3:8b-fp16"
    ):
        """
        RAG 시스템 초기화
        
        Args:
            collection_name: Qdrant 컬렉션 이름
            qdrant_host: Qdrant 서버 호스트
            qdrant_port: Qdrant 서버 포트
            ollama_host: Ollama 서버 호스트
            ollama_port: Ollama 서버 포트
            llm_model: 사용할 LLM 모델명
        """
        # 벡터 DB 초기화 (기존 컬렉션 로드)
        logger.info("기존 벡터 데이터베이스에 연결합니다...")
        self.vector_db = self._connect_existing_vectordb(
            collection_name, qdrant_host, qdrant_port, ollama_host, ollama_port
        )
        
        # Qwen3 채팅 모델 초기화
        logger.info(f"Ollama {llm_model} 모델을 초기화합니다...")
        self.chat_model = OllamaQwenChat(
            base_url=f"http://{ollama_host}:{ollama_port}",
            model_name=llm_model
        )
        
        # 모델 연결 테스트
        self._test_model_connection()
        
        logger.info("✅ 한국어 RAG 시스템 초기화 완료!")
    
    def _connect_existing_vectordb(
        self, collection_name: str, qdrant_host: str, qdrant_port: int,
        ollama_host: str, ollama_port: int
    ) -> KoreanRAGVectorDB_HTTP:
        """기존 벡터 데이터베이스에 연결 및 BM25 데이터 로드"""
        try:
            # 기존 구현을 수정하여 컬렉션을 새로 생성하지 않도록 함
            vector_db = KoreanRAGVectorDB_HTTP.__new__(KoreanRAGVectorDB_HTTP)
            vector_db.collection_name = collection_name
            
            # Ollama BGE-M3 임베더 초기화
            from makeDB_http import OllamaBGEEmbedder, SimpleBM25Wrapper, QdrantHTTPClient
            ollama_base_url = f"http://{ollama_host}:{ollama_port}"
            vector_db.embedder = OllamaBGEEmbedder(base_url=ollama_base_url)
            
            # BM25 래퍼 초기화
            vector_db.bm25_wrapper = SimpleBM25Wrapper()
            
            # Qdrant HTTP 클라이언트 초기화
            vector_db.client = QdrantHTTPClient(host=qdrant_host, port=qdrant_port)
            
            # 연결 테스트
            collections = vector_db.client.get_collections()
            collection_names = [col["name"] for col in collections["result"]["collections"]]
            
            if collection_name not in collection_names:
                raise Exception(f"컬렉션 '{collection_name}'이 존재하지 않습니다. 먼저 makeDB_http.py를 실행하세요.")
            
            # 컬렉션 정보 확인
            collection_info = vector_db.client.get_collection_info(collection_name)
            points_count = collection_info["result"]["points_count"]
            vector_config = collection_info["result"]["config"]["params"]["vectors"]
            vector_db.dense_vector_size = vector_config["size"]
            
            # 기존 벡터화된 데이터 로드하여 BM25용 문서 텍스트 추출
            logger.info("BM25 검색을 위해 기존 벡터화된 데이터를 로드합니다...")
            current_dir = Path(__file__).parent
            vectorized_data_path = current_dir.parent / "02_make_vector_data" / "vectorized_data.json"
            
            if vectorized_data_path.exists():
                with open(vectorized_data_path, 'r', encoding='utf-8') as f:
                    vectorized_data = json.load(f)
                
                # 문서 텍스트 추출 및 저장 (BM25용)
                documents_text = [item['content'] for item in vectorized_data]
                vector_db.documents = documents_text
                vector_db.document_metadata = vectorized_data
                
                # BM25 모델 학습
                vector_db.bm25_wrapper.fit(documents_text)
                logger.info(f"✅ BM25 모델 학습 완료: {len(documents_text)}개 문서")
            else:
                logger.warning("⚠️ 벡터화된 데이터 파일을 찾을 수 없습니다. BM25 검색이 제한됩니다.")
                vector_db.documents = []
                vector_db.document_metadata = []
            
            logger.info(f"✅ 기존 벡터 DB 연결 성공: {points_count}개 벡터, {vector_db.dense_vector_size}차원")
            return vector_db
            
        except Exception as e:
            logger.error(f"❌ 기존 벡터 DB 연결 실패: {e}")
            raise
    
    def _test_model_connection(self):
        """모델 연결 테스트"""
        try:
            test_response = self.chat_model.simple_generate("안녕하세요. 간단히 인사해주세요.")
            if test_response:
                logger.info(f"✅ Qwen3 모델 연결 확인: {test_response[:50]}...")
            else:
                raise Exception("모델 응답을 받지 못했습니다.")
        except Exception as e:
            logger.error(f"❌ Qwen3 모델 연결 실패: {e}")
            raise
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5, 
        search_type: str = "hybrid",
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """질문에 관련된 컨텍스트 검색"""
        try:
            logger.info(f"컨텍스트 검색 중: '{query}' (방식: {search_type}, top_k: {top_k})")
            
            if search_type == "hybrid":
                results = self.vector_db.search_hybrid(query, top_k=top_k, alpha=alpha)
            elif search_type == "dense":
                results = self.vector_db.search_dense_only(query, top_k=top_k)
            elif search_type == "sparse":
                results = self.vector_db.search_sparse_only(query, top_k=top_k)
            else:
                raise ValueError(f"지원하지 않는 검색 방식: {search_type}")
            
            logger.info(f"✅ {len(results)}개 관련 문서 검색 완료")
            return results
            
        except Exception as e:
            logger.error(f"❌ 컨텍스트 검색 실패: {e}")
            return []
    
    def format_context(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 컨텍스트 텍스트로 포맷팅"""
        if not search_results:
            return "관련 정보를 찾을 수 없습니다."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            score = result.get('score', 0)
            content = result.get('content', '')
            context_parts.append(f"[문서 {i}] (관련도: {score:.3f})\n{content}")
        
        return "\n\n".join(context_parts)
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        alpha: float = 0.7,
        temperature: float = 0.7,
        system_prompt: str = None
    ) -> Dict[str, Any]:
        """RAG 기반 질문 답변"""
        try:
            start_time = time.time()
            
            # 1. 컨텍스트 검색
            logger.info(f"🔍 질문: {question}")
            search_results = self.retrieve_context(
                query=question, 
                top_k=top_k, 
                search_type=search_type,
                alpha=alpha
            )
            
            if not search_results:
                return {
                    'question': question,
                    'answer': "죄송합니다. 관련된 정보를 찾을 수 없어서 답변드릴 수 없습니다.",
                    'context': [],
                    'search_type': search_type,
                    'processing_time': time.time() - start_time
                }
            
            # 2. 컨텍스트 포맷팅
            context_text = self.format_context(search_results)
            
            # 3. LLM 답변 생성
            answer = self.chat_model.chat_with_context(
                query=question,
                context=context_text,
                system_prompt=system_prompt,
                temperature=temperature
            )
            
            if not answer:
                answer = "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."
            
            end_time = time.time()
            
            result = {
                'question': question,
                'answer': answer,
                'context': search_results,
                'search_type': search_type,
                'search_params': {
                    'top_k': top_k,
                    'alpha': alpha if search_type == "hybrid" else None,
                    'temperature': temperature
                },
                'processing_time': end_time - start_time
            }
            
            logger.info(f"✅ RAG 답변 완료 (소요시간: {result['processing_time']:.2f}초)")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG 답변 생성 실패: {e}")
            return {
                'question': question,
                'answer': f"오류가 발생했습니다: {str(e)}",
                'context': [],
                'search_type': search_type,
                'processing_time': time.time() - start_time,
                'error': str(e)
            }
    
    def batch_test(self, test_questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """여러 질문에 대한 배치 테스트"""
        results = []
        total_start_time = time.time()
        
        logger.info(f"🚀 배치 테스트 시작: {len(test_questions)}개 질문")
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- 테스트 {i}/{len(test_questions)} ---")
            result = self.answer_question(question, **kwargs)
            results.append(result)
            
            # 중간 결과 출력
            print(f"\n{'='*60}")
            print(f"질문 {i}: {question}")
            print(f"답변: {result['answer']}")
            print(f"검색된 문서 수: {len(result.get('context', []))}")
            print(f"처리 시간: {result['processing_time']:.2f}초")
            
            if result.get('context'):
                print(f"\n관련 문서 (상위 3개):")
                for j, ctx in enumerate(result['context'][:3], 1):
                    score = ctx.get('score', 0)
                    content = ctx.get('content', '')[:100]
                    print(f"  {j}. [점수: {score:.3f}] {content}...")
        
        total_time = time.time() - total_start_time
        logger.info(f"\n✅ 배치 테스트 완료 (총 소요시간: {total_time:.2f}초)")
        
        return results
    
    def interactive_chat(self):
        """대화형 RAG 채팅"""
        print("\n" + "="*60)
        print("🤖 한국어 RAG 시스템 대화형 테스트")
        print("- 'quit' 또는 'exit'를 입력하면 종료됩니다")
        print("- 'help'를 입력하면 도움말을 확인할 수 있습니다")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ 질문을 입력하세요: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료']:
                    print("👋 RAG 시스템을 종료합니다.")
                    break
                
                if question.lower() == 'help':
                    print("""
📖 도움말:
- 한국어 언어학, 문법, 표준어 규정 등에 대해 질문하세요
- 예시 질문들:
  * "표준어 규정이 뭐야?"
  * "한국어 문법에서 조사는 어떻게 사용해?"
  * "맞춤법 규칙을 알려줘"
  * "외래어 표기법은 어떻게 되나요?"
                    """)
                    continue
                
                if not question:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                # RAG 답변 생성
                result = self.answer_question(
                    question=question,
                    top_k=5,
                    search_type="hybrid",
                    alpha=0.7,
                    temperature=0.7
                )
                
                # 결과 출력
                print(f"\n🤖 답변:")
                print(result['answer'])
                
                print(f"\n📊 검색 정보:")
                print(f"- 검색된 문서: {len(result.get('context', []))}개")
                print(f"- 검색 방식: {result['search_type']}")
                print(f"- 처리 시간: {result['processing_time']:.2f}초")
                
                # 관련 문서 미리보기
                if result.get('context'):
                    print(f"\n📝 관련 문서 미리보기:")
                    for i, ctx in enumerate(result['context'][:2], 1):
                        score = ctx.get('score', 0)
                        content = ctx.get('content', '')[:150]
                        print(f"  {i}. [관련도: {score:.3f}] {content}...")
                
            except KeyboardInterrupt:
                print("\n\n👋 사용자가 중단했습니다. RAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")
                continue

def main():
    """메인 테스트 함수"""
    logger.info("=== 한국어 RAG 시스템 테스트 ===")
    
    try:
        # RAG 시스템 초기화
        rag_system = KoreanRAGSystem(
            collection_name="korean_rag_http_collection",
            qdrant_host="localhost",
            qdrant_port=6333,
            ollama_host="localhost",
            ollama_port=11434,
            llm_model="qwen3:8b-fp16"
        )
        
        # 테스트 질문들
        test_questions = [
            "표준어 규정이 무엇인가요?",
            "한국어 문법에서 조사의 역할은 무엇인가요?",
            "외래어 표기법의 기본 원칙을 알려주세요",
            "맞춤법에서 띄어쓰기 규칙을 설명해주세요",
            "한글 맞춤법의 기본 원리는 무엇인가요?"
        ]
        
        print("\n" + "="*60)
        print("🧪 배치 테스트 vs 대화형 테스트 선택")
        print("1. 배치 테스트 (미리 정의된 질문들)")
        print("2. 대화형 테스트 (직접 질문 입력)")
        print("="*60)
        
        choice = input("선택하세요 (1 또는 2): ").strip()
        
        if choice == "1":
            # 배치 테스트 실행
            logger.info("배치 테스트를 시작합니다...")
            results = rag_system.batch_test(
                test_questions=test_questions,
                top_k=5,
                search_type="hybrid",
                alpha=0.7,
                temperature=0.7
            )
            
            # 결과 요약
            print(f"\n{'='*60}")
            print("📊 배치 테스트 결과 요약")
            print(f"{'='*60}")
            
            total_time = sum(r['processing_time'] for r in results)
            avg_time = total_time / len(results)
            
            print(f"총 질문 수: {len(results)}")
            print(f"총 처리 시간: {total_time:.2f}초")
            print(f"평균 처리 시간: {avg_time:.2f}초")
            
            # 각 질문별 요약
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['question']}")
                print(f"   답변 길이: {len(result['answer'])}자")
                print(f"   검색 문서: {len(result.get('context', []))}개")
                print(f"   처리 시간: {result['processing_time']:.2f}초")
        
        elif choice == "2":
            # 대화형 테스트 실행
            rag_system.interactive_chat()
        
        else:
            print("❌ 잘못된 선택입니다. 1 또는 2를 입력해주세요.")
            return
        
        logger.info("=== RAG 시스템 테스트 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ RAG 시스템 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    main()