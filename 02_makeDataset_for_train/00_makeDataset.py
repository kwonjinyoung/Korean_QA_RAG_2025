"""
트레이닝 데이터를 처리하고 Qdrant를 사용하여 리트리버 결과를 얻는 코드
"""

import os
import json
import re
from typing import List, Dict, Any
import time

from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_qdrant import RetrievalMode, FastEmbedSparse


def load_train_data(file_path: str = "../resource/korean_language_rag_V1.0_train.json") -> List[Dict]:
    """한국어 QA 트레이닝 데이터를 로드합니다."""
    print("📚 한국어 QA 트레이닝 데이터 로드 중...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"트레이닝 데이터 파일이 존재하지 않습니다: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"✅ 트레이닝 데이터 로드 완료: {len(train_data)}개 문항")
    return train_data


def extract_core_words(question: str) -> str:
    """질문에서 큰따옴표 안에 있는 문장만 추출합니다."""
    pattern = r'"([^"]*)"'
    matches = re.findall(pattern, question)
    
    if matches:
        return matches[0]
    else:
        # 큰따옴표 안의 문장이 없으면 질문 전체를 반환
        return question


def load_vectorstore():
    """기존에 구축된 Qdrant 벡터스토어를 로드합니다."""
    print("🔄 기존 Qdrant 벡터스토어 로드 중...")
    
    # DB 경로 확인
    db_path = "../qdrant_local_db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Qdrant DB가 존재하지 않습니다: {db_path}")
    
    # 임베딩 모델 설정
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    # Sparse 임베딩 설정
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Qdrant 클라이언트 생성
    client = QdrantClient(path=db_path)
    
    collection_name = "korean_qa_hybrid"
    
    # 컬렉션 존재 확인
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]
    
    if collection_name not in collection_names:
        raise ValueError(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")
    
    # 하이브리드 벡터스토어 생성
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    
    print("✅ 벡터스토어 로드 완료!")
    return qdrant_store


def retrieve_context(vectorstore, query: str, k: int = 5) -> str:
    """쿼리에 대한 컨텍스트를 검색합니다."""
    print(f"🔍 쿼리 검색 중: {query}")
    
    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # 검색 실행
    docs = retriever.invoke(query)
    
    # 검색 결과 포맷팅
    context = "\n".join("<Content>\n" + doc.page_content + "\n</Content>" for doc in docs)
    
    return context


def process_data_and_save(train_data: List[Dict], vectorstore) -> None:
    """데이터를 처리하고 새 JSON 파일로 저장합니다."""
    print("🔄 데이터 처리 중...")
    
    processed_data = []
    total = len(train_data)
    
    for i, item in enumerate(train_data, 1):
        if i % 10 == 0:
            print(f"진행 중: {i}/{total} ({i/total:.1%})")
        
        question_id = item["id"]
        question_type = item["input"]["question_type"]
        question = item["input"]["question"]
        answer = item["output"]["answer"]
        
        # 핵심 단어 추출
        core_words = extract_core_words(question)
        
        # 리트리버 결과 얻기
        context = retrieve_context(vectorstore, core_words)
        
        # 새 데이터 포맷 생성
        processed_item = {
            "question_type": question_type,
            "context": context,
            "question": question,
            "answer": answer
        }
        
        processed_data.append(processed_item)
    
    # 결과 저장
    output_file = "processed_train_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 처리된 데이터가 저장되었습니다: {output_file}")
    print(f"   - 총 {len(processed_data)}개 문항 결과 저장")
    print(f"   - 파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")


def main():
    """메인 함수"""
    try:
        print("🚀 한국어 QA 데이터셋 처리 시작")
        print("=" * 80)
        
        # 1. 트레이닝 데이터 로드
        train_data = load_train_data()
        
        # 테스트용으로 일부 데이터만 처리 (필요시 주석 해제)
        # train_data = train_data[:10]
        # print(f"🧪 테스트 모드: {len(train_data)}개 문항만 처리합니다.")
        
        # 2. 벡터스토어 로드
        vectorstore = load_vectorstore()
        
        # 3. 데이터 처리 및 저장
        process_data_and_save(train_data, vectorstore)
        
        print("\n✅ 모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
