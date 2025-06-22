import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm
import requests
import time

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SparseVector,
    NamedVector, SparseVectorParams, VectorParamsDiff,
    CollectionInfo, SearchRequest, QueryRequest, Query,
    SparseIndices, SparseValues, RecommendRequest,
    HnswConfigDiff, OptimizersConfigDiff
)


class KoreanRAGVectorDB:
    """한국어 RAG를 위한 BGE-M3 REST API + Qdrant 하이브리드 벡터 데이터베이스"""
    
    def __init__(
        self,
        collection_name: str = "korean_rag_collection",
        qdrant_storage_path: str = "./qdrant_storage",
        api_host: str = "localhost",
        api_port: int = 8008,
        dense_vector_size: int = 1024,
        max_seq_length: int = 8192
    ):
        """
        초기화
        
        Args:
            collection_name: Qdrant 컬렉션 이름
            qdrant_storage_path: Qdrant 데이터 저장 경로
            api_host: BGE-M3 API 서버 호스트
            api_port: BGE-M3 API 서버 포트
            dense_vector_size: Dense 벡터 차원
            max_seq_length: 최대 시퀀스 길이
        """
        self.collection_name = collection_name
        self.dense_vector_size = dense_vector_size
        self.max_seq_length = max_seq_length
        
        # BGE-M3 API 서버 설정
        self.api_base_url = f"http://{api_host}:{api_port}"
        self.api_timeout = 300  # 5분 타임아웃
        
        # Qdrant 클라이언트 초기화 (파일 시스템 기반)
        self.qdrant_storage_path = qdrant_storage_path
        os.makedirs(self.qdrant_storage_path, exist_ok=True)
        
        try:
            # 파일 시스템 기반 Qdrant 클라이언트 생성
            self.client = QdrantClient(path=self.qdrant_storage_path)
            print(f"✅ Qdrant 파일 시스템 모드로 실행: {os.path.abspath(self.qdrant_storage_path)}")
        except Exception as e:
            print(f"⚠️ Qdrant 파일 시스템 초기화 실패, 메모리 모드로 실행합니다: {e}")
            self.client = QdrantClient(":memory:")
        
        # BGE-M3 API 서버 연결 확인
        self._check_api_server()
        
        # 컬렉션 생성
        self._create_collection()
    
    def _check_api_server(self):
        """BGE-M3 API 서버 연결 확인"""
        try:
            health_url = f"{self.api_base_url}/health"
            response = requests.get(health_url, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ BGE-M3 API 서버 연결 확인: {self.api_base_url}")
            else:
                print(f"⚠️ BGE-M3 API 서버 응답 이상: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ BGE-M3 API 서버 연결 실패: {self.api_base_url}")
            print(f"   에러: {e}")
            print("   API 서버가 실행 중인지 확인해주세요.")
            raise ConnectionError(f"BGE-M3 API 서버에 연결할 수 없습니다: {self.api_base_url}")
    
    def _create_collection(self):
        """Qdrant 컬렉션 생성 (하이브리드 검색용)"""
        try:
            # 기존 컬렉션 확인
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                print(f"⚠️ 컬렉션 '{self.collection_name}'이 이미 존재합니다. 삭제 후 재생성합니다.")
                self.client.delete_collection(self.collection_name)
            
            # 하이브리드 검색을 위한 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    # Dense vector (의미적 유사성)
                    "dense": VectorParams(
                        size=self.dense_vector_size,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=16,  # 연결 수
                            ef_construct=200,  # 구성 시 탐색 깊이
                        )
                    )
                },
                sparse_vectors_config={
                    # Sparse vector (키워드 매칭, BM25 스타일)
                    "sparse": SparseVectorParams(
                        index=SparseIndices.default()
                    )
                },
                optimizers_config=OptimizersConfigDiff(
                    default_segment_number=2,
                    max_segment_size=20000,
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=2
                )
            )
            
            print(f"✅ 하이브리드 컬렉션 '{self.collection_name}' 생성 완료")
            
        except Exception as e:
            print(f"❌ 컬렉션 생성 실패: {e}")
            raise
    
    def encode_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        REST API를 통해 텍스트를 BGE-M3로 인코딩 (Dense + Sparse)
        
        Args:
            texts: 인코딩할 텍스트 리스트
            
        Returns:
            인코딩 결과 (dense_vecs, sparse_vecs)
        """
        try:
            # API 요청 데이터 준비
            request_data = {
                "texts": texts,
                "batch_size": 32,
                "max_length": self.max_seq_length,
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": False
            }
            
            # BGE-M3 API 호출
            encode_url = f"{self.api_base_url}/encode"
            response = requests.post(
                encode_url,
                json=request_data,
                timeout=self.api_timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"API 응답 오류: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # API 응답 형식에 맞게 변환
            return {
                'dense_vecs': np.array(result['dense_vecs']),
                'sparse_vecs': result['sparse_vecs']
            }
            
        except requests.exceptions.Timeout:
            print(f"❌ API 요청 타임아웃 ({self.api_timeout}초)")
            raise
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            raise
        except Exception as e:
            print(f"❌ 텍스트 인코딩 실패: {e}")
            raise
    
    def _convert_sparse_embedding(self, sparse_embedding: Dict) -> SparseVector:
        """BGE-M3 sparse embedding을 Qdrant SparseVector로 변환"""
        try:
            # BGE-M3 sparse 형식: {token_id: weight, ...}
            indices = list(sparse_embedding.keys())
            values = list(sparse_embedding.values())
            
            return SparseVector(
                indices=indices,
                values=values
            )
        except Exception as e:
            print(f"❌ Sparse 벡터 변환 실패: {e}")
            raise
    
    def load_chunks_from_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """JSONL 파일에서 청크 데이터 로드"""
        chunks = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Line {line_num} JSON 파싱 오류: {e}")
                        continue
            
            print(f"✅ {len(chunks)}개 청크 로드 완료: {jsonl_path}")
            return chunks
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {jsonl_path}")
            raise
        except Exception as e:
            print(f"❌ 청크 로드 실패: {e}")
            raise
    
    def build_vector_database(self, jsonl_path: str, batch_size: int = 16):
        """
        JSONL 파일로부터 하이브리드 벡터 데이터베이스 구축
        
        Args:
            jsonl_path: chunks.jsonl 파일 경로
            batch_size: 배치 처리 크기
        """
        print("🚀 벡터 데이터베이스 구축 시작...")
        
        # 청크 데이터 로드
        chunks = self.load_chunks_from_jsonl(jsonl_path)
        
        if not chunks:
            print("❌ 로드된 청크가 없습니다.")
            return
        
        # 배치 단위로 처리
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc="벡터화 및 업로드"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            # 텍스트 추출
            texts = [chunk['content'] for chunk in batch_chunks]
            
            # BGE-M3 인코딩
            try:
                embeddings = self.encode_texts(texts)
                dense_vecs = embeddings['dense_vecs']
                sparse_vecs = embeddings['sparse_vecs']
                
                # Qdrant 포인트 생성
                points = []
                for i, chunk in enumerate(batch_chunks):
                    # Sparse 벡터 변환
                    sparse_vector = self._convert_sparse_embedding(sparse_vecs[i])
                    
                    # 포인트 생성
                    point = PointStruct(
                        id=hash(chunk['id']),  # 문자열 ID를 해시로 변환
                        vector={
                            "dense": dense_vecs[i].tolist(),
                            "sparse": sparse_vector
                        },
                        payload={
                            "chunk_id": chunk['id'],
                            "content": chunk['content'],
                            "source": chunk.get('source', ''),
                            "chunk_index": chunk.get('chunk_index', 0),
                            "length": chunk.get('length', len(chunk['content']))
                        }
                    )
                    points.append(point)
                
                # Qdrant에 업로드
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
            except Exception as e:
                print(f"❌ 배치 {batch_idx + 1} 처리 실패: {e}")
                continue
        
        # 컬렉션 정보 출력
        collection_info = self.client.get_collection(self.collection_name)
        print(f"✅ 벡터 데이터베이스 구축 완료!")
        print(f"   - 총 벡터 수: {collection_info.points_count}")
        print(f"   - 컬렉션 이름: {self.collection_name}")
        print(f"   - Dense 벡터 차원: {self.dense_vector_size}")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행 (Dense + Sparse)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            dense_weight: Dense 검색 가중치
            sparse_weight: Sparse 검색 가중치
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 인코딩
            query_embeddings = self.encode_texts([query])
            query_dense = query_embeddings['dense_vecs'][0]
            query_sparse = self._convert_sparse_embedding(query_embeddings['sparse_vecs'][0])
            
            # Qdrant 1.10+ Query API 사용한 하이브리드 검색
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=Query(
                    fusion=Query.Fusion.RRF,  # Reciprocal Rank Fusion
                    prefetch=[
                        # Dense 검색
                        Query(
                            nearest=query_dense.tolist(),
                            using="dense",
                            limit=top_k * 2
                        ),
                        # Sparse 검색
                        Query(
                            nearest=query_sparse,
                            using="sparse", 
                            limit=top_k * 2
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )
            
            # 결과 포맷팅
            results = []
            for point in search_results.points:
                result = {
                    'chunk_id': point.payload['chunk_id'],
                    'content': point.payload['content'],
                    'source': point.payload['source'],
                    'chunk_index': point.payload['chunk_index'],
                    'score': point.score,
                    'length': point.payload['length']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ 하이브리드 검색 실패: {e}")
            return []
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """컬렉션 정보 조회"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            print(f"❌ 컬렉션 정보 조회 실패: {e}")
            return None
    
    def get_storage_info(self) -> Dict[str, Any]:
        """저장소 정보 조회"""
        try:
            storage_info = {
                'storage_path': os.path.abspath(self.qdrant_storage_path),
                'storage_exists': os.path.exists(self.qdrant_storage_path),
                'storage_size_mb': 0,
                'files': []
            }
            
            if os.path.exists(self.qdrant_storage_path):
                # 저장소 크기 계산
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.qdrant_storage_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total_size += os.path.getsize(filepath)
                            storage_info['files'].append({
                                'name': filename,
                                'path': filepath,
                                'size_bytes': os.path.getsize(filepath)
                            })
                
                storage_info['storage_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            return storage_info
            
        except Exception as e:
            print(f"❌ 저장소 정보 조회 실패: {e}")
            return {}
    
    def clear_storage(self):
        """저장소 초기화 (모든 데이터 삭제)"""
        try:
            import shutil
            if os.path.exists(self.qdrant_storage_path):
                shutil.rmtree(self.qdrant_storage_path)
                print(f"✅ 저장소 초기화 완료: {self.qdrant_storage_path}")
            
            # 저장소 재생성
            os.makedirs(self.qdrant_storage_path, exist_ok=True)
            
            # 클라이언트 재초기화
            self.client = QdrantClient(path=self.qdrant_storage_path)
            
        except Exception as e:
            print(f"❌ 저장소 초기화 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    print("🇰🇷 한국어 RAG 벡터 데이터베이스 구축 시작")
    print("=" * 50)
    
    # 벡터 데이터베이스 초기화
    vector_db = KoreanRAGVectorDB(
        collection_name="korean_rag_hybrid",
        qdrant_storage_path="./qdrant_storage",  # 파일 시스템 저장 경로
        api_host="localhost",
        api_port=8008,
        dense_vector_size=1024
    )
    
    # chunks.jsonl 파일 경로
    chunks_path = "../00_rag_make_dataset/chunks.jsonl"
    
    if not os.path.exists(chunks_path):
        print(f"❌ 청크 파일을 찾을 수 없습니다: {chunks_path}")
        return
    
    # 벡터 데이터베이스 구축
    vector_db.build_vector_database(chunks_path, batch_size=8)
    
    # 컬렉션 정보 출력
    collection_info = vector_db.get_collection_info()
    if collection_info:
        print("\n📊 컬렉션 정보:")
        print(f"   - 벡터 수: {collection_info.points_count}")
        print(f"   - 상태: {collection_info.status}")
    
    # 저장소 정보 출력
    storage_info = vector_db.get_storage_info()
    if storage_info:
        print("\n💾 저장소 정보:")
        print(f"   - 저장 경로: {storage_info['storage_path']}")
        print(f"   - 저장소 크기: {storage_info['storage_size_mb']} MB")
        print(f"   - 파일 수: {len(storage_info['files'])}")
        if storage_info['files']:
            print("   - 주요 파일들:")
            for file_info in storage_info['files'][:5]:  # 처음 5개 파일만 표시
                size_kb = round(file_info['size_bytes'] / 1024, 1)
                print(f"     * {file_info['name']}: {size_kb} KB")
    
    # 테스트 검색
    print("\n🔍 테스트 검색 수행...")
    test_queries = [
        "한글 맞춤법 규정",
        "외래어 표기법",
        "문장 부호 사용법"
    ]
    
    for query in test_queries:
        print(f"\n검색어: '{query}'")
        results = vector_db.hybrid_search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [점수: {result['score']:.4f}] {result['content'][:100]}...")
    
    print("\n✅ 벡터 데이터베이스 구축 및 테스트 완료!")


if __name__ == "__main__":
    main()
