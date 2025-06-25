import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import requests
import time
import logging
import re

# ARM64 호환성을 위한 라이브러리 선택 (HTTP API 사용)
from rank_bm25 import BM25Okapi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QdrantHTTPClient:
    """Qdrant HTTP API 클라이언트 (ARM64 호환)"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """HTTP 요청 실행"""
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=60)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=60)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=60)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=60)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Qdrant HTTP API 요청 실패: {e}")
            raise
    
    def get_collections(self) -> Dict:
        """컬렉션 목록 조회"""
        return self._make_request("GET", "/collections")
    
    def create_collection(self, collection_name: str, vector_size: int) -> Dict:
        """컬렉션 생성"""
        config = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine"
            },
            "hnsw_config": {
                "m": 16,
                "ef_construct": 200
            },
            "optimizers_config": {
                "default_segment_number": 2,
                "max_segment_size": 20000,
                "memmap_threshold": 20000,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": 2
            }
        }
        return self._make_request("PUT", f"/collections/{collection_name}", config)
    
    def delete_collection(self, collection_name: str) -> Dict:
        """컬렉션 삭제"""
        return self._make_request("DELETE", f"/collections/{collection_name}")
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """컬렉션 정보 조회"""
        return self._make_request("GET", f"/collections/{collection_name}")
    
    def upsert_points(self, collection_name: str, points: List[Dict]) -> Dict:
        """포인트 업서트"""
        data = {"points": points}
        return self._make_request("PUT", f"/collections/{collection_name}/points", data)
    
    def search_points(self, collection_name: str, query_vector: List[float], 
                     limit: int = 10, with_payload: bool = True) -> Dict:
        """포인트 검색"""
        data = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": with_payload
        }
        return self._make_request("POST", f"/collections/{collection_name}/points/search", data)

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

class KoreanRAGVectorDB_HTTP:
    """HTTP API를 사용한 ARM64 호환 한국어 RAG 하이브리드 Qdrant 벡터 데이터베이스"""
    
    def __init__(
        self,
        collection_name: str = "korean_rag_http_collection",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        ollama_host: str = "localhost",
        ollama_port: int = 11434
    ):
        """
        초기화
        
        Args:
            collection_name: Qdrant 컬렉션 이름
            qdrant_host: Qdrant 서버 호스트
            qdrant_port: Qdrant 서버 포트
            ollama_host: Ollama 서버 호스트
            ollama_port: Ollama 서버 포트
        """
        self.collection_name = collection_name
        
        # Ollama BGE-M3 임베더 초기화
        ollama_base_url = f"http://{ollama_host}:{ollama_port}"
        self.embedder = OllamaBGEEmbedder(base_url=ollama_base_url)
        
        # ARM64 호환 BM25 모델 초기화
        self.bm25_wrapper = SimpleBM25Wrapper()
        
        # Qdrant HTTP 클라이언트 초기화
        try:
            self.client = QdrantHTTPClient(host=qdrant_host, port=qdrant_port)
            # 연결 테스트
            collections = self.client.get_collections()
            logger.info(f"✅ Qdrant HTTP API 연결 성공: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"❌ Qdrant HTTP API 연결 실패: {e}")
            raise ConnectionError(f"Qdrant HTTP API에 연결할 수 없습니다: {qdrant_host}:{qdrant_port}")
        
        # Ollama BGE-M3 모델 연결 확인 및 벡터 차원 확인
        self.dense_vector_size = self._check_ollama_model()
        
        # 문서 저장용 (BM25 계산을 위해)
        self.documents = []
        self.document_metadata = []
        
        # 컬렉션 생성
        self._create_collection()
    
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
    
    def _create_collection(self):
        """Qdrant 컬렉션 생성 (HTTP API 사용)"""
        try:
            # 기존 컬렉션 확인 및 삭제
            collections_response = self.client.get_collections()
            collection_names = [col["name"] for col in collections_response["result"]["collections"]]
            
            if self.collection_name in collection_names:
                logger.warning(f"⚠️ 컬렉션 '{self.collection_name}'이 이미 존재합니다. 삭제 후 재생성합니다.")
                self.client.delete_collection(self.collection_name)
            
            # Dense 벡터 컬렉션 생성
            response = self.client.create_collection(self.collection_name, self.dense_vector_size)
            
            if response["status"] == "ok":
                logger.info(f"✅ 컬렉션 '{self.collection_name}' 생성 완료")
            else:
                raise Exception(f"컬렉션 생성 실패: {response}")
            
        except Exception as e:
            logger.error(f"❌ 컬렉션 생성 실패: {e}")
            raise
    
    def load_vectorized_data(self, json_path: str) -> List[Dict[str, Any]]:
        """벡터화된 JSON 파일에서 데이터 로드"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ {len(data)}개 벡터화된 데이터 로드 완료: {json_path}")
            return data
            
        except FileNotFoundError:
            logger.error(f"❌ 파일을 찾을 수 없습니다: {json_path}")
            raise
        except Exception as e:
            logger.error(f"❌ 벡터화된 데이터 로드 실패: {e}")
            raise
    
    def build_vector_database_from_json(self, json_path: str, batch_size: int = 100):
        """
        벡터화된 JSON 파일로부터 하이브리드 벡터 데이터베이스 구축 (HTTP API 사용)
        
        Args:
            json_path: vectorized_data.json 파일 경로
            batch_size: 배치 처리 크기
        """
        logger.info("🚀 HTTP API 기반 하이브리드 벡터 데이터베이스 구축 시작...")
        
        # 벡터화된 데이터 로드
        vectorized_data = self.load_vectorized_data(json_path)
        
        if not vectorized_data:
            logger.error("❌ 로드된 벡터화 데이터가 없습니다.")
            return
        
        # 문서 텍스트 추출 및 저장 (BM25용)
        documents_text = [item['content'] for item in vectorized_data]
        self.documents = documents_text
        self.document_metadata = vectorized_data
        
        # ARM64 호환 BM25 모델 학습
        logger.info("ARM64 호환 BM25 모델을 학습합니다...")
        self.bm25_wrapper.fit(documents_text)
        
        # 배치 단위로 처리
        total_batches = (len(vectorized_data) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="벡터 DB 업로드"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(vectorized_data))
            batch_data = vectorized_data[start_idx:end_idx]
            
            try:
                # HTTP API용 포인트 생성 (기존 임베딩 사용)
                points = []
                for item in batch_data:
                    point = {
                        "id": item['id'],
                        "vector": item['embedding'],  # 기존에 생성된 임베딩 사용
                        "payload": {
                            "chunk_id": item['id'],
                            "content": item['content'],
                            "length": item['length'],
                            "original_content": item['original_content'],
                            "embedding_dim": item.get('embedding_dim', len(item['embedding']))
                        }
                    }
                    points.append(point)
                
                # HTTP API를 통해 Qdrant에 업로드
                response = self.client.upsert_points(self.collection_name, points)
                
                if response["status"] != "ok":
                    logger.error(f"❌ 배치 {batch_idx + 1} 업로드 실패: {response}")
                
            except Exception as e:
                logger.error(f"❌ 배치 {batch_idx + 1} 처리 실패: {e}")
                continue
        
        # 컬렉션 정보 출력
        collection_info = self.client.get_collection_info(self.collection_name)
        points_count = collection_info["result"]["points_count"]
        
        logger.info(f"✅ HTTP API 기반 하이브리드 벡터 데이터베이스 구축 완료!")
        logger.info(f"   - 총 벡터 수: {points_count}")
        logger.info(f"   - 컬렉션 이름: {self.collection_name}")
        logger.info(f"   - Dense 벡터 차원: {self.dense_vector_size}")
        logger.info(f"   - 하이브리드 모드: BGE-M3 + rank_bm25 (HTTP API)")
    
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
            
            # HTTP API를 통해 Dense 검색
            search_response = self.client.search_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=min(top_k * 2, len(self.documents)),
                with_payload=True
            )
            
            dense_results = search_response["result"]
            
            # 2. BM25 검색 (Sparse, ARM64 호환)
            bm25_scores = self.bm25_wrapper.get_scores(query)
            
            # 3. 결과 통합
            combined_results = {}
            
            # Dense 결과 처리
            for point in dense_results:
                chunk_id = point["payload"]["chunk_id"]
                combined_results[chunk_id] = {
                    'payload': point["payload"],
                    'dense_score': float(point["score"]),
                    'sparse_score': 0.0
                }
            
            # Sparse 결과 처리
            for i, sparse_score in enumerate(bm25_scores):
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
                    'embedding_dim': payload['embedding_dim'],
                    'search_type': 'hybrid_http_api'
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
            
            search_response = self.client.search_points(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            results = []
            for point in search_response["result"]:
                result = {
                    'chunk_id': point["payload"]['chunk_id'],
                    'content': point["payload"]['content'],
                    'original_content': point["payload"]['original_content'],
                    'length': point["payload"]['length'],
                    'score': point["score"],
                    'embedding_dim': point["payload"]['embedding_dim'],
                    'search_type': 'dense_only_http'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Dense 검색 실패: {e}")
            return []
    
    def search_sparse_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25만 사용한 검색 (ARM64 호환)"""
        try:
            bm25_scores = self.bm25_wrapper.get_scores(query)
            
            # 점수와 인덱스를 함께 정렬
            scored_docs = [(score, i) for i, score in enumerate(bm25_scores)]
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            
            results = []
            for score, i in scored_docs[:top_k]:
                if score > 0:  # 점수가 있는 문서만
                    item = self.document_metadata[i]
                    result = {
                        'chunk_id': item['id'],
                        'content': item['content'],
                        'original_content': item['original_content'],
                        'length': item['length'],
                        'score': float(score),
                        'embedding_dim': item.get('embedding_dim', 0),
                        'search_type': 'sparse_only_http'
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Sparse 검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회 (HTTP API)"""
        try:
            return self.client.get_collection_info(self.collection_name)
        except Exception as e:
            logger.error(f"❌ 컬렉션 정보 조회 실패: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 및 Qdrant 서버 정보 조회"""
        try:
            # 시스템 아키텍처 정보
            import platform
            system_info = {
                'architecture': platform.machine(),
                'platform': platform.platform(),
                'processor': platform.processor()
            }
            
            # 컬렉션 목록
            collections_response = self.client.get_collections()
            collection_names = [col["name"] for col in collections_response["result"]["collections"]]
            
            # 현재 컬렉션 정보
            collection_info = None
            points_count = 0
            if self.collection_name in collection_names:
                collection_info = self.client.get_collection_info(self.collection_name)
                points_count = collection_info["result"]["points_count"]
            
            return {
                'server_status': 'connected',
                'api_type': 'HTTP API',
                'system_info': system_info,
                'collections': collection_names,
                'current_collection': self.collection_name,
                'current_collection_exists': self.collection_name in collection_names,
                'points_count': points_count,
                'dense_vector_size': self.dense_vector_size,
                'hybrid_mode': True,
                'sparse_embedding': 'rank_bm25 (ARM64 호환)',
                'dense_embedding': 'BGE-M3',
                'implementation': 'HTTP API (ARM64 Optimized)'
            }
            
        except Exception as e:
            logger.error(f"❌ 시스템 정보 조회 실패: {e}")
            return {
                'server_status': 'error',
                'error': str(e)
            }
    
    def clear_collection(self):
        """현재 컬렉션 삭제 (HTTP API)"""
        try:
            response = self.client.delete_collection(self.collection_name)
            if response["status"] == "ok":
                logger.info(f"✅ 컬렉션 '{self.collection_name}' 삭제 완료")
            else:
                logger.error(f"❌ 컬렉션 삭제 실패: {response}")
        except Exception as e:
            logger.error(f"❌ 컬렉션 삭제 실패: {e}")

def main():
    """메인 실행 함수"""
    logger.info("=== HTTP API 기반 ARM64 호환 한국어 RAG 하이브리드 벡터 데이터베이스 구축 ===")
    
    # 경로 설정
    current_dir = Path(__file__).parent
    vectorized_data_path = current_dir.parent / "02_make_vector_data" / "vectorized_data.json"
    
    try:
        # HTTP API 기반 하이브리드 벡터 DB 클래스 초기화
        logger.info("HTTP API 기반 하이브리드 벡터 데이터베이스를 초기화합니다...")
        vector_db = KoreanRAGVectorDB_HTTP(
            collection_name="korean_rag_http_collection",
            qdrant_host="localhost",  # Docker 컨테이너 (host 네트워크)
            qdrant_port=6333,         # HTTP 포트
            ollama_host="localhost",  # Ollama 서버 호스트
            ollama_port=11434         # Ollama 서버 포트
        )
        
        # 벡터화된 데이터로부터 하이브리드 벡터 DB 구축
        logger.info("벡터화된 데이터로부터 하이브리드 데이터베이스를 구축합니다...")
        vector_db.build_vector_database_from_json(str(vectorized_data_path))
        
        # 컬렉션 정보 출력
        collection_info = vector_db.get_collection_info()
        if collection_info and collection_info["status"] == "ok":
            points_count = collection_info["result"]["points_count"]
            logger.info(f"최종 컬렉션 정보:")
            logger.info(f"  - 벡터 개수: {points_count}")
            logger.info(f"  - Dense 벡터 차원: {vector_db.dense_vector_size}")
            logger.info(f"  - 하이브리드 모드: BGE-M3 + rank_bm25 (HTTP API)")
        
        # 하이브리드 검색 테스트
        logger.info("\n=== HTTP API 하이브리드 검색 테스트 ===")
        test_queries = ["표준어 규정", "한국어 문법", "언어학 이론"]
        
        for query in test_queries:
            logger.info(f"\n테스트 쿼리: '{query}'")
            
            # 하이브리드 검색 (Dense 70%, Sparse 30%)
            hybrid_results = vector_db.search_hybrid(query, top_k=3, alpha=0.7)
            if hybrid_results:
                logger.info(f"하이브리드 검색 결과 ({len(hybrid_results)}개):")
                for i, result in enumerate(hybrid_results, 1):
                    logger.info(f"  {i}. [하이브리드: {result['score']:.4f}, Dense: {result['dense_score']:.4f}, Sparse: {result['sparse_score']:.4f}]")
                    logger.info(f"     {result['content'][:100]}...")
            
            # Dense only 검색
            dense_results = vector_db.search_dense_only(query, top_k=3)
            if dense_results:
                logger.info(f"Dense only 검색 결과 ({len(dense_results)}개):")
                for i, result in enumerate(dense_results, 1):
                    logger.info(f"  {i}. [Dense: {result['score']:.4f}] {result['content'][:100]}...")
            
            # Sparse only 검색
            sparse_results = vector_db.search_sparse_only(query, top_k=3)
            if sparse_results:
                logger.info(f"Sparse only 검색 결과 ({len(sparse_results)}개):")
                for i, result in enumerate(sparse_results, 1):
                    logger.info(f"  {i}. [Sparse: {result['score']:.4f}] {result['content'][:100]}...")
            
        # 시스템 정보 출력
        system_info = vector_db.get_system_info()
        logger.info(f"\n=== 시스템 및 서버 정보 ===")
        logger.info(f"아키텍처: {system_info.get('system_info', {}).get('architecture')}")
        logger.info(f"플랫폼: {system_info.get('system_info', {}).get('platform')}")
        logger.info(f"API 타입: {system_info.get('api_type')}")
        logger.info(f"서버 상태: {system_info.get('server_status')}")
        logger.info(f"하이브리드 모드: {system_info.get('hybrid_mode')}")
        logger.info(f"Dense 임베딩: {system_info.get('dense_embedding')}")
        logger.info(f"Sparse 임베딩: {system_info.get('sparse_embedding')}")
        logger.info(f"구현 방식: {system_info.get('implementation')}")
            
        logger.info("\n=== HTTP API 기반 하이브리드 벡터 데이터베이스 구축 완료 ===")
        
    except Exception as e:
        logger.error(f"❌ 프로세스 실행 실패: {e}")
        raise

if __name__ == "__main__":
    main() 