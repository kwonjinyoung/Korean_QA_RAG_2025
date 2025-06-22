# 한국어 RAG 벡터 데이터베이스 구축

BGE-M3 모델과 Qdrant를 사용한 하이브리드 검색 벡터 데이터베이스 구축 도구입니다.

## 주요 특징

- 🇰🇷 **한국어 특화**: `upskyy/bge-m3-korean` 모델 사용
- 🔄 **하이브리드 검색**: Dense + Sparse 벡터 결합
- ⚡ **최신 기술**: Qdrant 1.10+ Query API, RRF(Reciprocal Rank Fusion)
- 🚀 **고성능**: 배치 처리 및 GPU 가속 지원

## 설치 및 실행

### 1. 의존성 설치

```bash
# 프로젝트 루트에서 실행
uv sync
```

### 2. Qdrant 설정

```bash
# 파일 시스템 모드로 실행 (기본값)
# 벡터 데이터가 ./qdrant_storage 폴더에 저장됩니다.
# 별도의 서버 설치나 실행이 필요하지 않습니다.

# 선택사항: Docker로 Qdrant 서버 실행
cd 01_rag_make_vectordb
docker-compose up -d
```

### 3. 벡터 데이터베이스 구축

```bash
cd 01_rag_make_vectordb
python main.py
```

## 코드 구조

### KoreanRAGVectorDB 클래스

주요 메서드:
- `__init__()`: BGE-M3 모델 로드 및 Qdrant 연결
- `build_vector_database()`: JSONL 파일로부터 벡터 DB 구축
- `hybrid_search()`: Dense + Sparse 하이브리드 검색
- `encode_texts()`: BGE-M3를 사용한 텍스트 인코딩

### 하이브리드 검색 구조

```
Query → BGE-M3 Encoding → Dense Vector + Sparse Vector
                           ↓
Qdrant Query API → RRF Fusion → Top-K Results
```

## 사용 예시

```python
from main import KoreanRAGVectorDB

# 벡터 DB 초기화
vector_db = KoreanRAGVectorDB(
    collection_name="korean_rag_hybrid",
    qdrant_storage_path="./qdrant_storage",  # 파일 저장 경로
    embedding_model="upskyy/bge-m3-korean"
)

# 벡터 DB 구축
vector_db.build_vector_database("../00_rag_make_dataset/chunks.jsonl")

# 검색 수행
results = vector_db.hybrid_search("한글 맞춤법", top_k=5)
for result in results:
    print(f"점수: {result['score']:.4f}")
    print(f"내용: {result['content'][:100]}...")
```

## 기술 세부사항

### BGE-M3 모델
- **Dense Vector**: 의미적 유사성 (1024차원)
- **Sparse Vector**: 키워드 매칭 (BM25 스타일)
- **한국어 특화**: `upskyy/bge-m3-korean` 모델 사용

### Qdrant 설정
- **Distance**: Cosine 유사도
- **HNSW**: m=16, ef_construct=200
- **Optimizer**: 성능 최적화 설정
- **RRF Fusion**: Dense + Sparse 결과 통합

### 성능 최적화
- 배치 처리 (기본 8개씩)
- GPU 가속 (CUDA 사용 가능시)
- FP16 정밀도 (메모리 절약)
- 점진적 업로드 (대용량 데이터 처리)

## 데이터 저장 및 관리

### 파일 시스템 저장
- 벡터 데이터는 `./qdrant_storage` 폴더에 저장됩니다
- 프로그램 종료 후에도 데이터가 유지됩니다
- 데이터 백업: 전체 `qdrant_storage` 폴더를 복사하면 됩니다

### 저장소 관리
```python
# 저장소 정보 확인
storage_info = vector_db.get_storage_info()
print(f"저장소 크기: {storage_info['storage_size_mb']} MB")

# 저장소 초기화 (모든 데이터 삭제)
vector_db.clear_storage()
```

## 문제 해결

### 1. CUDA 메모리 부족
```python
# 배치 크기 줄이기
vector_db.build_vector_database(chunks_path, batch_size=4)
```

### 2. 저장소 권한 문제
- `qdrant_storage` 폴더의 읽기/쓰기 권한 확인
- 필요시 `chmod 755 qdrant_storage` 실행

### 3. 모델 다운로드 실패
- 인터넷 연결 확인
- Hugging Face 모델 캐시 삭제 후 재시도

### 4. 디스크 공간 부족
- 저장소 크기 확인: `vector_db.get_storage_info()`
- 불필요한 컬렉션 삭제 또는 저장소 초기화

## 참고 자료

- [BGE-M3 논문](https://arxiv.org/abs/2402.03216)
- [Qdrant 문서](https://qdrant.tech/documentation/)
- [한국어 BGE-M3 모델](https://huggingface.co/upskyy/bge-m3-korean) 