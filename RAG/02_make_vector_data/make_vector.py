import json
import os
import requests
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """배치로 여러 텍스트의 임베딩을 생성합니다."""
        embeddings = []
        for text in tqdm(texts, desc="임베딩 생성 중"):
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 로드합니다."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.error(f"라인 {line_num}에서 JSON 파싱 오류: {e}")
                    continue
        
        logger.info(f"총 {len(data)}개의 데이터를 로드했습니다.")
        return data
        
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except Exception as e:
        logger.error(f"파일 로드 중 오류 발생: {e}")
        return []

def save_vector_data(data: List[Dict[str, Any]], embeddings: List[Optional[List[float]]], output_path: str):
    """임베딩과 원본 데이터를 함께 저장합니다."""
    try:
        vector_data = []
        
        for i, (item, embedding) in enumerate(zip(data, embeddings)):
            if embedding is not None:
                vector_item = {
                    'id': i,
                    'content': item['content'],
                    'length': item['length'],
                    'original_content': item['original_content'],
                    'embedding': embedding,
                    'embedding_dim': len(embedding)
                }
                vector_data.append(vector_item)
            else:
                logger.warning(f"인덱스 {i}의 임베딩이 None입니다. 건너뜁니다.")
        
        # JSON 파일로 저장
        json_output_path = output_path.replace('.npy', '.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        
        # NumPy 배열로도 저장 (임베딩만)
        valid_embeddings = [item['embedding'] for item in vector_data]
        if valid_embeddings:
            embeddings_array = np.array(valid_embeddings)
            np.save(output_path, embeddings_array)
            
            logger.info(f"벡터 데이터를 저장했습니다:")
            logger.info(f"  - JSON 파일: {json_output_path}")
            logger.info(f"  - NumPy 파일: {output_path}")
            logger.info(f"  - 저장된 항목 수: {len(vector_data)}")
            logger.info(f"  - 임베딩 차원: {embeddings_array.shape[1] if len(embeddings_array) > 0 else 0}")
        else:
            logger.error("유효한 임베딩이 없어 파일을 저장하지 못했습니다.")
            
    except Exception as e:
        logger.error(f"벡터 데이터 저장 중 오류 발생: {e}")

def check_ollama_model(embedder: OllamaBGEEmbedder) -> bool:
    """Ollama에서 BGE-M3 모델이 사용 가능한지 확인합니다."""
    try:
        # 간단한 테스트 임베딩으로 모델 상태 확인
        test_embedding = embedder.get_embedding("테스트")
        if test_embedding is not None and len(test_embedding) > 0:
            logger.info(f"BGE-M3 모델이 정상적으로 작동합니다. 임베딩 차원: {len(test_embedding)}")
            return True
        else:
            logger.error("BGE-M3 모델에서 유효한 임베딩을 생성하지 못했습니다.")
            return False
    except Exception as e:
        logger.error(f"모델 확인 중 오류 발생: {e}")
        return False

def main():
    """메인 실행 함수"""
    # 경로 설정
    current_dir = Path(__file__).parent
    input_file = current_dir.parent / "00_rag_make_dataset" / "rechunked_data.jsonl"
    output_dir = current_dir
    output_file = output_dir / "vectorized_data.npy"
    
    # 출력 디렉토리 생성
    output_dir.mkdir(exist_ok=True)
    
    logger.info("=== BGE-M3 벡터화 프로세스 시작 ===")
    
    # 1. Ollama 임베더 초기화
    logger.info("Ollama BGE-M3 임베더를 초기화합니다...")
    embedder = OllamaBGEEmbedder()
    
    # 2. 모델 상태 확인
    logger.info("BGE-M3 모델 상태를 확인합니다...")
    if not check_ollama_model(embedder):
        logger.error("BGE-M3 모델을 사용할 수 없습니다. 다음을 확인해주세요:")
        logger.error("1. Ollama가 실행 중인지 확인: ollama serve")
        logger.error("2. BGE-M3 모델이 설치되어 있는지 확인: ollama pull bge-m3")
        return
    
    # 3. 데이터 로드
    logger.info(f"데이터를 로드합니다: {input_file}")
    data = load_jsonl_data(str(input_file))
    
    if not data:
        logger.error("로드할 데이터가 없습니다.")
        return
    
    # 4. content 텍스트 추출
    logger.info("content 텍스트를 추출합니다...")
    texts = [item['content'] for item in data]
    logger.info(f"총 {len(texts)}개의 텍스트를 처리합니다.")
    
    # 5. 임베딩 생성
    logger.info("임베딩을 생성합니다...")
    embeddings = embedder.get_embeddings_batch(texts)
    
    # 6. 결과 확인
    valid_embeddings = [e for e in embeddings if e is not None]
    logger.info(f"성공적으로 생성된 임베딩: {len(valid_embeddings)}/{len(embeddings)}")
    
    if not valid_embeddings:
        logger.error("생성된 임베딩이 없습니다.")
        return
    
    # 7. 벡터 데이터 저장
    logger.info("벡터 데이터를 저장합니다...")
    save_vector_data(data, embeddings, str(output_file))
    
    logger.info("=== 벡터화 프로세스 완료 ===")

if __name__ == "__main__":
    main()
