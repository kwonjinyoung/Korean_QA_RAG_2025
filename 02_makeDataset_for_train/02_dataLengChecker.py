import json
from transformers import AutoTokenizer
import os

def main():
    # Qwen3-8B 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    
    # 현재 디렉토리 확인
    dataset_path = "final_dataset.json"
    
    # 파일이 존재하는지 확인
    if not os.path.exists(dataset_path):
        print(f"파일을 찾을 수 없습니다: {dataset_path}")
        return
    
    print(f"데이터셋 파일 경로: {dataset_path}")
    
    # 데이터셋 로드
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"파일 로드 중 오류 발생: {e}")
        return
    
    print(f"데이터셋 항목 수: {len(dataset)}")
    
    # 각 항목의 question+answer 문자열에 대한 토큰 크기 계산
    token_sizes = []
    for idx, item in enumerate(dataset):
        question = item.get('question', '')
        answer = item.get('answer', '')
        combined_text = question + answer
        
        # 토큰화
        tokens = tokenizer.encode(combined_text)
        token_count = len(tokens)
        
        token_sizes.append({
            'index': idx,
            'token_count': token_count,
            'question': question,
            'answer': answer
        })
    
    # 토큰 크기 기준으로 내림차순 정렬
    sorted_token_sizes = sorted(token_sizes, key=lambda x: x['token_count'], reverse=True)
    
    # Top 10 출력
    print("\n=== 토큰 크기 기준 상위 10개 항목 ===")
    for i, item in enumerate(sorted_token_sizes[:10], 1):
        print(f"{i}. 인덱스: {item['index']}, 토큰 수: {item['token_count']}")
        print(f"   질문: {item['question'][:50]}..." if len(item['question']) > 50 else f"   질문: {item['question']}")
        print(f"   답변: {item['answer'][:50]}..." if len(item['answer']) > 50 else f"   답변: {item['answer']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
