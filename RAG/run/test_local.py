import argparse
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

# Qdrant 로컬 모드를 위한 라이브러리
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# ARM64 호환성을 위한 라이브러리 선택
import re
from rank_bm25 import BM25Okapi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 로컬 Qdrant DB 관련 클래스들

class SimpleBM25Wrapper:
    """rank_bm25를 사용한 BM25 래퍼 클래스 (ARM64 호환)"""
    
    def __init__(self):
        self.bm25 = None
        self.tokenized_docs = []
        self.documents = []
        
    def tokenize(self, text: str) -> List[str]:
        """한국어 친화적 토크나이저"""
        # 한국어, 영어, 숫자만 남기고 나머지 제거
        text = re.sub(r'[^\w가-힣\s]', ' ', text)
        # 공백으로 분할하고 길이가 1보다 큰 토큰만 선택
        tokens = [token.lower() for token in text.split() if len(token) > 1]
        return tokens
    
    def fit(self, documents: List[str]):
        """BM25 모델 학습"""
        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info(f"✅ BM25 모델 학습 완료: {len(documents)}개 문서")
    
    def get_scores(self, query: str) -> List[float]:
        """쿼리에 대한 BM25 점수 계산"""
        if self.bm25 is None:
            logger.error("BM25 모델이 학습되지 않았습니다.")
            return [0.0] * len(self.documents)
        
        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        return scores.tolist()

class OllamaBGEEmbedder:
    """Ollama BGE-M3 모델을 사용한 텍스트 임베딩 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "bge-m3"):
        self.base_url = base_url
        self.model_name = model_name
        self.embed_url = f"{base_url}/api/embeddings"
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """텍스트에 대한 임베딩 벡터를 생성합니다."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(self.embed_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding", [])
            
            if not embedding:
                logger.error(f"임베딩 결과가 비어있습니다: {text[:50]}...")
                return None
                
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            return None

class KoreanRAGVectorDB_Local:
    """로컬 파일 시스템 기반 한국어 RAG 하이브리드 Qdrant 벡터 데이터베이스"""
    
    def __init__(
        self,
        collection_name: str = "korean_rag_local_collection",
        db_path: str = "./qdrant_storage",
        ollama_host: str = "localhost",
        ollama_port: int = 11434
    ):
        """
        초기화
        
        Args:
            collection_name: Qdrant 컬렉션 이름
            db_path: Qdrant 로컬 데이터베이스 저장 경로
            ollama_host: Ollama 서버 호스트
            ollama_port: Ollama 서버 포트
        """
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        
        # 로컬 DB 저장 디렉토리 생성
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Ollama BGE-M3 임베더 초기화
        ollama_base_url = f"http://{ollama_host}:{ollama_port}"
        self.embedder = OllamaBGEEmbedder(base_url=ollama_base_url)
        
        # ARM64 호환 BM25 모델 초기화
        self.bm25_wrapper = SimpleBM25Wrapper()
        
        # Qdrant 로컬 클라이언트 초기화
        try:
            self.client = QdrantClient(path=str(self.db_path))
            logger.info(f"✅ Qdrant 로컬 모드 초기화 성공: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ Qdrant 로컬 모드 초기화 실패: {e}")
            raise ConnectionError(f"Qdrant 로컬 모드를 초기화할 수 없습니다: {self.db_path}")
        
        # Ollama BGE-M3 모델 연결 확인 및 벡터 차원 확인
        self.dense_vector_size = self._check_ollama_model()
        
        # 문서 저장용 (BM25 계산을 위해)
        self.documents = []
        self.document_metadata = []
        
        # 기존 컬렉션이 있으면 로드, 없으면 생성
        self._initialize_collection()
    
    def _check_ollama_model(self) -> int:
        """Ollama BGE-M3 모델 연결 확인 및 벡터 차원 반환"""
        try:
            test_embedding = self.embedder.get_embedding("테스트")
            if test_embedding and len(test_embedding) > 0:
                vector_size = len(test_embedding)
                logger.info(f"✅ Ollama BGE-M3 모델 연결 확인. 임베딩 차원: {vector_size}")
                return vector_size
            else:
                raise Exception("BGE-M3 모델에서 유효한 임베딩을 생성하지 못했습니다.")
        except Exception as e:
            logger.error(f"❌ Ollama BGE-M3 모델 연결 실패: {e}")
            logger.error("다음을 확인해주세요:")
            logger.error("1. Ollama가 실행 중인지: ollama serve")
            logger.error("2. BGE-M3 모델이 설치되어 있는지: ollama pull bge-m3")
            raise ConnectionError(f"Ollama BGE-M3 모델에 연결할 수 없습니다: {e}")
    
    def _initialize_collection(self):
        """Qdrant 컬렉션 초기화 (기존 컬렉션 확인 후 생성 또는 로드)"""
        try:
            # 기존 컬렉션 확인
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                # 기존 컬렉션 로드
                logger.info(f"📂 기존 컬렉션 '{self.collection_name}' 로드")
                collection_info = self.client.get_collection(self.collection_name)
                points_count = collection_info.points_count
                logger.info(f"✅ 기존 컬렉션 로드 완료: {points_count}개 벡터")
                
                # 기존 데이터로 BM25 초기화
                self._load_existing_data_for_bm25()
            else:
                # 새 컬렉션 생성
                logger.info(f"🆕 새 컬렉션 '{self.collection_name}' 생성")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dense_vector_size, 
                        distance=Distance.COSINE
                    ),
                    # 성능 최적화 설정
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 200
                    },
                    optimizers_config={
                        "default_segment_number": 2,
                        "max_segment_size": 20000,
                        "memmap_threshold": 20000,
                        "indexing_threshold": 20000,
                        "flush_interval_sec": 5,
                        "max_optimization_threads": 2
                    }
                )
                logger.info(f"✅ 컬렉션 '{self.collection_name}' 생성 완료")
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 초기화 실패: {e}")
            raise
    
    def _load_existing_data_for_bm25(self):
        """기존 벡터화된 데이터를 로드하여 BM25 초기화"""
        try:
            # 벡터화된 데이터 파일 경로
            current_dir = Path(__file__).parent
            vectorized_data_path = current_dir.parent / "02_make_vector_data" / "vectorized_data.json"
            
            if vectorized_data_path.exists():
                with open(vectorized_data_path, 'r', encoding='utf-8') as f:
                    vectorized_data = json.load(f)
                
                # 문서 텍스트 추출 및 저장 (BM25용)
                documents_text = [item['content'] for item in vectorized_data]
                self.documents = documents_text
                self.document_metadata = vectorized_data
                
                # BM25 모델 학습
                if documents_text:
                    self.bm25_wrapper.fit(documents_text)
                    logger.info(f"✅ BM25 모델 초기화 완료: {len(documents_text)}개 문서")
                else:
                    logger.warning("⚠️ 로드된 문서가 없습니다.")
            else:
                logger.warning("⚠️ 벡터화된 데이터 파일을 찾을 수 없습니다. BM25 검색이 제한됩니다.")
                self.documents = []
                self.document_metadata = []
                
        except Exception as e:
            logger.error(f"❌ BM25 데이터 로드 실패: {e}")
            self.documents = []
            self.document_metadata = []
    
    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행 (Dense + Sparse, HTTP API 사용)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            alpha: Dense와 Sparse 가중치 (0.7 = Dense 70%, Sparse 30%)
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 1. Dense 검색 (Semantic Search, HTTP API)
            query_embedding = self.embedder.get_embedding(query)
            if query_embedding is None:
                logger.error("❌ 쿼리 임베딩 생성 실패")
                return []
            
            # 로컬 클라이언트를 통해 Dense 검색
            dense_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=min(top_k * 2, len(self.documents)) if self.documents else top_k,
                with_payload=True
            )
            
            # 2. BM25 검색 (Sparse, ARM64 호환)
            bm25_scores = self.bm25_wrapper.get_scores(query) if self.documents else []
            
            # 3. 결과 통합
            combined_results = {}
            
            # Dense 결과 처리
            for point in dense_results:
                chunk_id = point.payload["chunk_id"]
                combined_results[chunk_id] = {
                    'payload': point.payload,
                    'dense_score': float(point.score),
                    'sparse_score': 0.0
                }
            
            # Sparse 결과 처리
            for i, sparse_score in enumerate(bm25_scores):
                if i < len(self.document_metadata):
                    chunk_id = self.document_metadata[i]['id']
                    if chunk_id in combined_results:
                        combined_results[chunk_id]['sparse_score'] = float(sparse_score)
                    elif sparse_score > 0:  # BM25 점수가 있는 문서만
                        combined_results[chunk_id] = {
                            'payload': {
                                'chunk_id': chunk_id,
                                'content': self.document_metadata[i]['content'],
                                'original_content': self.document_metadata[i]['original_content'],
                                'length': self.document_metadata[i]['length'],
                                'embedding_dim': self.document_metadata[i].get('embedding_dim', 0)
                            },
                            'dense_score': 0.0,
                            'sparse_score': float(sparse_score)
                        }
            
            # 4. 하이브리드 점수 계산 및 정렬
            final_results = []
            max_sparse_score = max(bm25_scores) if bm25_scores else 1.0
            
            for chunk_id, result in combined_results.items():
                # 점수 정규화 (0-1 범위)
                normalized_dense = min(1.0, max(0.0, result['dense_score']))
                normalized_sparse = min(1.0, max(0.0, result['sparse_score'] / max_sparse_score)) if max_sparse_score > 0 else 0.0
                
                # 하이브리드 점수 계산
                hybrid_score = alpha * normalized_dense + (1 - alpha) * normalized_sparse
                
                payload = result['payload']
                final_result = {
                    'chunk_id': payload['chunk_id'],
                    'content': payload['content'],
                    'original_content': payload['original_content'],
                    'length': payload['length'],
                    'score': hybrid_score,
                    'dense_score': result['dense_score'],
                    'sparse_score': result['sparse_score'],
                    'embedding_dim': payload.get('embedding_dim', 0),
                    'search_type': 'hybrid'
                }
                final_results.append(final_result)
            
            # 하이브리드 점수로 정렬하고 상위 결과 반환
            final_results.sort(key=lambda x: x['score'], reverse=True)
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ 하이브리드 검색 실패: {e}")
            return []
    
    def search_dense_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense 벡터만 사용한 검색 (HTTP API)"""
        try:
            query_embedding = self.embedder.get_embedding(query)
            if query_embedding is None:
                logger.error("❌ 쿼리 임베딩 생성 실패")
                return []
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for point in search_results:
                result = {
                    'chunk_id': point.payload['chunk_id'],
                    'content': point.payload['content'],
                    'original_content': point.payload['original_content'],
                    'length': point.payload['length'],
                    'score': point.score,
                    'embedding_dim': point.payload.get('embedding_dim', 0),
                    'search_type': 'dense_only'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Dense 검색 실패: {e}")
            return []

class QuestionPromptGenerator:
    """질문 타입별 프롬프트 생성기"""
    
    def __init__(self):
        self.type_instructions = {
            "선다형": (
                "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                "[지침]\n"
                "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
                "[예시]\n"
                "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
                "1) 주사위 놀이\n"
                "2) 검무\n"
                "3) 격구\n"
                "4) 영고\n"
                "5) 무애무\n"
                "답변: 3"
            ),
            "서술형": (
                "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                "[지침]\n"
                "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
                "[예시]\n"
                "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
                "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
            ),
            "단답형": (
                "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                "[지침]\n"
                "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
                "[예시]\n"
                "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
                "답변: 정약용"
            ),
            "교정형": (
                "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                "[지침]\n"
                "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
                "[예시]\n"
                "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
                "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
            ),
            "선택형": (
                "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                "[지침]\n"
                "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
                "[예시]\n"
                "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
                "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
            )
        }
    
    def generate_prompt(self, question_data: Dict[str, Any]) -> str:
        """질문 데이터로부터 프롬프트 생성"""
        # 디버깅을 위한 로그
        logger.info(f"[DEBUG] 프롬프트 생성 중: {question_data}")
        
        # question_type에 따른 instruction 선택
        question_type = question_data.get('question_type', '')
        instruction = self.type_instructions.get(question_type, "")
        
        # 디버깅: question_type과 instruction 확인
        logger.info(f"[DEBUG] question_type: '{question_type}', instruction 존재: {bool(instruction)}")
        
        # 기타 정보 생성 (question과 question_type 제외)
        other_info = {k: v for k, v in question_data.items() if k not in ['question', 'question_type']}
        
        # 기타 정보가 있는 경우에만 추가
        chat_parts = [instruction]
        if other_info:
            info_list = ["[기타 정보]"]
            for key, value in other_info.items():
                info_list.append(f"- {key}: {value}")
            chat_parts.append("\n".join(info_list))

        # 질문 추가
        question = question_data.get('question', '')
        chat_parts.append(f"[질문]\n{question}")

        # 최종 프롬프트 생성
        chat = "\n\n".join(chat_parts)
        
        logger.info(f"[DEBUG] 생성된 프롬프트 길이: {len(chat)}")
        
        return chat

class OllamaQwenChat:
    """Ollama qwen3:14b 모델을 사용한 채팅 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "qwen3:14b"):
        self.base_url = base_url
        self.model_name = model_name
        self.chat_url = f"{base_url}/api/chat"
        self.generate_url = f"{base_url}/api/generate"
        
    def chat_with_context(
        self, 
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """컨텍스트를 포함한 채팅 응답 생성"""
        try:
            # 시스템 프롬프트 설정
            if system_prompt is None:
                system_prompt = """# Instructions:
당신은 한국어 언어학 및 문법 전문가입니다. 
주어진 Context 정보를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.
Context에 없는 정보는 추측하지 말고, 컨텍스트 기반으로만 답변하세요."""

            # 프롬프트 구성
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": f"""다음 컨텍스트 정보를 참고하여 질문에 답변해주세요.

# Context:
{context}

# Question: {query}

"""
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
    """한국어 RAG 시스템 (로컬 파일 시스템 기반)"""
    
    def __init__(
        self,
        collection_name: str = "korean_rag_local_collection",
        db_path: str = "./qdrant_storage",
        ollama_host: str = "localhost",
        ollama_port: int = 11434,
        llm_model: str = "qwen3:14b"
    ):
        """
        RAG 시스템 초기화
        
        Args:
            collection_name: Qdrant 컬렉션 이름
            db_path: Qdrant 로컬 데이터베이스 저장 경로
            ollama_host: Ollama 서버 호스트
            ollama_port: Ollama 서버 포트
            llm_model: 사용할 LLM 모델명
        """
        # 벡터 DB 초기화
        logger.info("벡터 데이터베이스를 초기화합니다...")
        self.vector_db = KoreanRAGVectorDB_Local(
            collection_name=collection_name,
            db_path=db_path,
            ollama_host=ollama_host,
            ollama_port=ollama_port
        )
        
        # Qwen3 채팅 모델 초기화
        logger.info(f"Ollama {llm_model} 모델을 초기화합니다...")
        self.chat_model = OllamaQwenChat(
            base_url=f"http://{ollama_host}:{ollama_port}",
            model_name=llm_model
        )
        
        # 프롬프트 생성기 초기화
        self.prompt_generator = QuestionPromptGenerator()
        
        # 모델 연결 테스트
        self._test_model_connection()
        
        logger.info("✅ 한국어 RAG 시스템 초기화 완료!")
    
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
            content = result.get('content', '')
            context_parts.append(f"<Content>\n{content}\n</Content>")
        
        return "\n\n".join(context_parts)
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        alpha: float = 0.7,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
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
            
            context_text = "<Context>" + context_text + "</Context>"
            
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
    
    def process_question_data(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """질문 데이터를 처리하여 RAG 답변 생성"""
        try:
            logger.info(f"[DEBUG] 처리할 질문 데이터: {question_data}")
            
            # 프롬프트 생성
            prompt = self.prompt_generator.generate_prompt(question_data)
            question = question_data.get('question', '')
            
            logger.info(f"[DEBUG] 생성된 프롬프트:\n{prompt}")
            logger.info(f"[DEBUG] 추출된 질문: {question}")
            
            # RAG 답변 생성
            result = self.answer_question(
                question=question,
                top_k=5,
                search_type="hybrid",
                alpha=0.7,
                temperature=0.7,
                system_prompt=prompt
            )
            
            # 결과에 원본 데이터 정보 추가
            result['original_data'] = question_data
            result['generated_prompt'] = prompt
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 질문 데이터 처리 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return {
                'question': question_data.get('question', ''),
                'answer': f"처리 중 오류 발생: {str(e)}",
                'context': [],
                'error': str(e),
                'original_data': question_data
            }
    
    def batch_test(self, test_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """배치 테스트 실행"""
        results = []
        total_start_time = time.time()
        
        logger.info(f"🚀 배치 테스트 시작: {len(test_data)}개 질문")
        
        for i, raw_data in enumerate(test_data, 1):
            logger.info(f"\n--- 테스트 {i}/{len(test_data)} ---")
            
            # 데이터 구조 변환: {"id": "1", "input": {...}} -> {...}
            if 'input' in raw_data:
                question_data = raw_data['input'].copy()
                question_data['id'] = raw_data.get('id', str(i))
            else:
                question_data = raw_data
            
            result = self.process_question_data(question_data)
            results.append(result)
            
            # 중간 결과 출력
            print(f"\n{'='*60}")
            print(f"질문 {i}: {result['question']}")
            print(f"답변: {result['answer']}")
            print(f"검색된 문서 수: {len(result.get('context', []))}")
            print(f"처리 시간: {result.get('processing_time', 0):.2f}초")
            
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

# fmt: off
parser = argparse.ArgumentParser(prog="rag_test", description="한국어 RAG 시스템 테스트")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, help="입력 JSON 파일 경로")
g.add_argument("--output", type=str, help="출력 JSON 파일 경로")
g.add_argument("--collection_name", type=str, default="korean_rag_local_collection", help="Qdrant 컬렉션 이름")
g.add_argument("--db_path", type=str, default="./qdrant_storage", help="Qdrant 로컬 데이터베이스 저장 경로")
g.add_argument("--ollama_host", type=str, default="localhost", help="Ollama 서버 호스트")
g.add_argument("--ollama_port", type=int, default=11434, help="Ollama 서버 포트")
g.add_argument("--llm_model", type=str, default="qwen3:14b", help="사용할 LLM 모델")
g.add_argument("--mode", type=str, choices=["batch", "interactive"], default="interactive", help="실행 모드")
# fmt: on

def main(args):
    """메인 실행 함수"""
    logger.info("=== 한국어 RAG 시스템 테스트 ===")
    
    try:
        # RAG 시스템 초기화
        rag_system = KoreanRAGSystem(
            collection_name=args.collection_name,
            db_path=args.db_path,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            llm_model=args.llm_model
        )
        
        if args.mode == "batch" and args.input:
            # 배치 테스트 모드
            logger.info(f"배치 테스트 모드: {args.input}")
            
            # 입력 데이터 로드
            with open(args.input, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 배치 테스트 실행
            results = rag_system.batch_test(test_data)
            
            # 결과 저장
            if args.output:
                import re
                import os
                
                # 디버그 파일명 생성
                base_name = os.path.splitext(args.output)[0]
                ext = os.path.splitext(args.output)[1]
                debug_file = f"{base_name}_debug{ext}"
                
                # 1. 원본 결과를 디버그 파일로 저장 (전체 상세 정보)
                with open(debug_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"디버그 결과 저장 완료: {debug_file}")
                
                # 2. 정제된 결과를 메인 파일로 저장
                def clean_answer(answer_text):
                    """answer에서 <think>...</think> 부분을 제거하고 좌우 여백 제거"""
                    if not answer_text:
                        return ""
                    
                    # <think>...</think> 패턴 제거 (개행 문자 포함)
                    cleaned = re.sub(r'<think>.*?</think>', '', answer_text, flags=re.DOTALL)
                    
                    # 좌우 여백 제거
                    cleaned = cleaned.strip()
                    
                    return cleaned
                
                # 정제된 결과 생성
                cleaned_results = []
                for result in results:
                    original_data = result.get('original_data', {})
                    
                    cleaned_result = {
                        "id": original_data.get('id', ''),
                        "input": original_data.get('input', original_data),
                        "output": {
                            "answer": clean_answer(result.get('answer', ''))
                        }
                    }
                    
                    # input이 별도로 없는 경우 원본 데이터 구조 유지
                    if 'input' not in original_data:
                        cleaned_result["input"] = {
                            k: v for k, v in original_data.items() 
                            if k not in ['id']
                        }
                    
                    cleaned_results.append(cleaned_result)
                
                # 정제된 결과 저장
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
                logger.info(f"정제된 결과 저장 완료: {args.output}")
            
            # 결과 요약
            print(f"\n{'='*60}")
            print("📊 배치 테스트 결과 요약")
            print(f"{'='*60}")
            
            total_time = sum(r.get('processing_time', 0) for r in results)
            avg_time = total_time / len(results) if results else 0
            
            print(f"총 질문 수: {len(results)}")
            print(f"총 처리 시간: {total_time:.2f}초")
            print(f"평균 처리 시간: {avg_time:.2f}초")
            
            # 각 질문별 요약
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['question']}")
                print(f"   답변 길이: {len(result['answer'])}자")
                print(f"   검색 문서: {len(result.get('context', []))}개")
                print(f"   처리 시간: {result.get('processing_time', 0):.2f}초")
        
        else:
            # 대화형 테스트 모드
            rag_system.interactive_chat()
        
        logger.info("=== RAG 시스템 테스트 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ RAG 시스템 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    exit(main(parser.parse_args()))