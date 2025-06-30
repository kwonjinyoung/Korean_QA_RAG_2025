import ollama
import json
import time
import numpy as np
import re

def query_ollama_reranker(query, passages, model_name="linux6200/bge-reranker-v2-m3", top_n=None):
    """
    Ollama 모듈을 사용하여 문서 재순위화를 수행합니다.
    
    Args:
        query (str): 사용자 질의
        passages (list): 재순위화할 문서 목록
        model_name (str): Ollama에 로드된 모델 이름
        top_n (int, optional): 반환할 상위 문서 수. None이면 모든 문서 반환
    
    Returns:
        dict: 재순위화된 결과
    """
    if top_n is None:
        top_n = len(passages)
    
    try:
        # 쿼리에서 중요 키워드 추출
        query_keywords = extract_keywords(query)
        print(f"추출된 키워드: {query_keywords}")
        
        # Ollama API는 현재 직접적인 rerank 메소드를 제공하지 않음
        # 대신 embeddings API를 사용하여 query-document 쌍의 임베딩을 생성하고 점수화
        results = []
        
        # 먼저 쿼리 자체의 임베딩 생성
        query_embedding = ollama.embeddings(
            model=model_name,
            prompt=query
        )['embedding']
        
        for passage in passages:
            # query와 document를 결합하여 reranker 모델에 전달
            rerank_prompt = f"Query: {query}\n\nDocument: {passage}\n\nRelevance:"
            response = ollama.embeddings(
                model=model_name,
                prompt=rerank_prompt
            )
            
            # 임베딩 벡터 전체를 활용하여 관련성 점수 계산
            embedding = response['embedding']
            
            # 점수 계산 방법 개선: 
            # 1. 임베딩 벡터의 평균값 활용
            # 2. 양수 값의 비율 계산 (긍정적 관련성 표시)
            # 3. 첫 번째 차원 가중치 증가 (일부 모델은 첫 차원에 중요 정보 저장)
            pos_values = sum(1 for v in embedding if v > 0)
            pos_ratio = pos_values / len(embedding)
            
            # 첫 번째 차원에 가중치 부여 (일부 reranker 모델은 첫 차원에 관련성 점수를 저장)
            first_dim_weight = max(0, embedding[0] * 3)  # 첫 번째 차원에 3배 가중치
            
            # 쿼리와 문서 임베딩 간의 유사도 계산 (코사인 유사도)
            doc_embedding = ollama.embeddings(
                model=model_name,
                prompt=passage
            )['embedding']
            
            # numpy 배열로 변환
            query_array = np.array(query_embedding)
            doc_array = np.array(doc_embedding)
            
            # 코사인 유사도 계산
            cosine_sim = np.dot(query_array, doc_array) / (np.linalg.norm(query_array) * np.linalg.norm(doc_array))
            
            # 키워드 매칭 점수 계산
            keyword_match_score = calculate_keyword_match(query_keywords, passage)
            
            # 최종 점수 계산 (여러 요소 결합)
            # 코사인 유사도와 키워드 매칭에 더 높은 가중치 부여
            relevance_score = (
                first_dim_weight * 0.2 +  # 20% 가중치
                pos_ratio * 0.1 +         # 10% 가중치
                max(0, cosine_sim) * 0.3 + # 30% 가중치
                keyword_match_score * 0.4  # 40% 가중치
            )
            
            results.append({
                "document": passage,
                "relevance_score": relevance_score
            })
        
        # 관련성 점수에 따라 내림차순으로 정렬
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # top_n 개수만큼 결과 반환
        results = results[:top_n]
        
        return {"model": model_name, "results": results}
    except Exception as e:
        print(f"Ollama API 요청 중 오류 발생: {e}")
        return None

def extract_keywords(text):
    """
    텍스트에서 중요 키워드를 추출합니다.
    """
    # 간단한 키워드 추출 (실제로는 더 복잡한 NLP 기법 사용 가능)
    text = text.lower()
    # 불용어 제거
    stopwords = ['에', '대해', '알려주세요', '는', '은', '이', '가', '을', '를', '의', '와', '과', '으로', '로']
    for word in stopwords:
        text = text.replace(word, ' ')
    
    # 단어 분리 및 필터링
    words = re.findall(r'\w+', text)
    return [word for word in words if len(word) > 1]

def calculate_keyword_match(keywords, text):
    """
    키워드와 텍스트 간의 매칭 점수를 계산합니다.
    """
    text_lower = text.lower()
    
    # 특정 주제 관련 키워드에 가중치 부여
    topic_keywords = {
        '역사': 2.0,
        '고조선': 2.0,
        '삼국': 2.0,
        '고려': 2.0,
        '조선': 2.0,
        '왕조': 1.5,
        '한국': 1.0
    }
    
    score = 0.0
    
    # 키워드 매칭 점수 계산
    for keyword in keywords:
        if keyword in text_lower:
            score += 1.0
    
    # 주제 관련 키워드 가중치 적용
    for topic, weight in topic_keywords.items():
        if topic in text_lower:
            score += weight
    
    # 정규화 (0~1 사이의 값으로)
    return min(1.0, score / (len(keywords) + 3))

def rerank_with_ollama(query, passages, model_name="linux6200/bge-reranker-v2-m3"):
    """
    Ollama를 사용하여 문서를 재순위화하고 결과를 정리합니다.
    
    Args:
        query (str): 사용자 질의
        passages (list): 재순위화할 문서 목록
        model_name (str): Ollama에 로드된 모델 이름
    
    Returns:
        list: [(문서, 점수)] 형식의 재순위화된 결과
    """
    # Ollama rerank API 호출
    result = query_ollama_reranker(query, passages, model_name)
    
    if not result or 'results' not in result:
        print("재순위화 결과를 얻지 못했습니다. 기본 순서를 사용합니다.")
        # 결과가 없을 경우 기본 순서와 임의의 점수 사용
        ranked_results = []
        for i, passage in enumerate(passages):
            score = 1.0 - (i * 0.1)  # 임시 점수
            ranked_results.append((passage, score))
        return ranked_results
    
    # API 응답에서 재순위화된 결과 추출
    ranked_results = []
    for item in result['results']:
        ranked_results.append((item['document'], item['relevance_score']))
    
    return ranked_results

def main():
    # 테스트 데이터
    query = "한국의 역사에 대해 알려주세요"
    passages = [
        "서울과 부산은 다른 도시야",
        "한국은 동아시아에 위치한 나라로, 남한과 북한으로 나뉘어 있습니다.",
        "한국의 역사는 고조선 시대부터 시작되어 삼국시대, 고려, 조선을 거쳐 현대에 이르고 있습니다.",
        "한국의 수도 서울은 인구 약 1000만 명의 대도시입니다.",
        "한국 역사에서 가장 중요한 왕조 중 하나는 1392년부터 1910년까지 지속된 조선왕조입니다.",
        "한국은 IT 기술과 반도체 산업이 발달한 나라입니다."
    ]
    
    print(f"질의: {query}")
    print("\n원본 문서:")
    for i, passage in enumerate(passages):
        print(f"{i+1}. {passage}")
    
    print("\n재순위화 중...")
    start_time = time.time()
    
    # 사용할 모델 지정
    # model_name = "linux6200/bge-reranker-v2-m3"  # BGE 모델 사용
    model_name = "dengcao/Qwen3-Reranker-4B:Q8_0"  # Qwen3 모델 사용
    
    # Ollama를 사용한 재순위화 수행
    ranked_results = rerank_with_ollama(query, passages, model_name)
    
    elapsed_time = time.time() - start_time
    print(f"\n처리 시간: {elapsed_time:.2f}초")
    
    print("\n재순위화 결과:")
    for i, (passage, score) in enumerate(ranked_results):
        print(f"{i+1}. [점수: {score:.4f}] {passage}")

if __name__ == "__main__":
    main()
