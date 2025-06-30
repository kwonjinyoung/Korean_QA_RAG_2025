"""
Qdrant VectorDB 구축 - 하이브리드 방식 (Dense + Sparse BM25)
"""

import json
import os
from typing import List, Dict, Any
from uuid import uuid4

# LangChain 및 Qdrant 관련 임포트
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams


def load_vectorized_data(json_file_path: str) -> List[Dict[str, Any]]:
    """vectorized_data.json 파일에서 데이터를 로드합니다."""
    print(f"데이터 로딩 시작: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"총 {len(data)}개의 문서가 로드되었습니다.")
    return data


def create_documents_from_data(data: List[Dict[str, Any]]) -> tuple[List[Document], List[str]]:
    """데이터를 LangChain Document 객체로 변환합니다."""
    documents = []
    uuids = []
    
    for item in data:
        doc = Document(
            page_content=item['content'],
            metadata={
                'id': item['id'],
                'length': item['length'],
                'original_content': item['original_content']
            }
        )
        documents.append(doc)
        uuids.append(str(uuid4()))
    
    print(f"{len(documents)}개의 Document 객체가 생성되었습니다.")
    return documents, uuids


def setup_qdrant_hybrid_vectorstore():
    """Qdrant 하이브리드 벡터스토어를 설정합니다."""
    
    # 1. 임베딩 모델 설정 (Ollama BGE-M3)
    print("임베딩 모델 설정 중...")
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    # 2. Sparse 임베딩 설정 (BM25)
    print("Sparse 임베딩 설정 중...")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # 3. Qdrant 클라이언트 생성 (로컬 저장소)
    print("Qdrant 클라이언트 생성 중...")
    client = QdrantClient(path="./qdrant_local_db")
    
    collection_name = "korean_qa_hybrid"
    
    # 4. 기존 컬렉션이 있으면 삭제
    try:
        client.delete_collection(collection_name)
        print(f"기존 컬렉션 '{collection_name}' 삭제 완료")
    except Exception as e:
        print(f"기존 컬렉션 삭제 실패 (존재하지 않음): {e}")
    
    # 5. 새 컬렉션 생성 (Dense + Sparse 벡터)
    print("새 컬렉션 생성 중...")
    
    # BGE-M3 모델의 임베딩 차원은 1024입니다
    embedding_dim = 1024
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    
    # 6. QdrantVectorStore 생성
    print("QdrantVectorStore 생성 중...")
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    
    return qdrant_store


def build_qdrant_vectordb():
    """Qdrant VectorDB 구축 메인 함수"""
    try:
        # 1. 데이터 로드
        json_file_path = "../02_make_vector_data/vectorized_data.json"
        
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {json_file_path}")
        
        data = load_vectorized_data(json_file_path)
        
        # 2. Document 객체 생성
        documents, uuids = create_documents_from_data(data)
        
        # 3. Qdrant 하이브리드 벡터스토어 설정
        qdrant_store = setup_qdrant_hybrid_vectorstore()
        
        # 4. 문서 추가 (배치 처리)
        print("문서를 벡터스토어에 추가 중...")
        
        # 대용량 데이터를 위한 배치 처리
        batch_size = 100
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_ids = uuids[i:i+batch_size]
            
            print(f"배치 처리 중... ({i+1}-{min(i+batch_size, total_docs)}/{total_docs})")
            
            qdrant_store.add_documents(
                documents=batch_docs,
                ids=batch_ids
            )
        
        print("✅ Qdrant VectorDB 구축이 완료되었습니다!")
        
        # 5. 간단한 테스트 검색
        print("\n🔍 테스트 검색 수행 중...")
        test_query = "표준어는 무엇인가요?"
        results = qdrant_store.similarity_search(test_query, k=3)
        
        print(f"\n테스트 쿼리: '{test_query}'")
        print("검색 결과:")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:200]}...")
            print(f"   메타데이터: {doc.metadata}")
            print()
        
        return qdrant_store
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise


if __name__ == "__main__":
    print("=== Qdrant VectorDB 구축 시작 ===")
    print("설정:")
    print("- 임베딩 모델: Ollama BGE-M3")
    print("- 검색 방식: 하이브리드 (Dense + Sparse BM25)")
    print("- 저장소: 로컬 Qdrant DB")
    print("=" * 50)
    
    vectorstore = build_qdrant_vectordb()
    
    print("\n=== 구축 완료 ===")
    print("벡터 데이터베이스가 './qdrant_local_db' 디렉토리에 저장되었습니다.")
    print("이제 이 벡터스토어를 사용하여 RAG 시스템을 구축할 수 있습니다.")
