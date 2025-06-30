import json
import os

def read_file_content(file_path):
    """파일 내용을 읽어서 문자열로 반환합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # 필요한 파일 경로 설정
    train_data_path = "processed_train_data.json"
    instruction_path = "../prompt/00_prompt_Instruction.md"
    context_path = "../prompt/01_prompt_context.md"
    correction_path = "../prompt/02_prompt_few_shot_교정형.md"
    selection_path = "../prompt/02_prompt_few_shot_선택형.md"
    output_path = "final_dataset.json"
    
    # 파일 내용 읽기
    instruction = read_file_content(instruction_path)
    context_template = read_file_content(context_path)
    correction_examples = read_file_content(correction_path)
    selection_examples = read_file_content(selection_path)
    
    # 학습 데이터 읽기
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 결과 데이터셋 생성
    result_dataset = []
    
    for item in train_data:
        question_type = item["question_type"]
        context = item["context"]
        question = item["question"]
        answer = item["answer"]
        
        # context 템플릿에 실제 컨텍스트 삽입
        formatted_context = context_template.replace("{Context}", context)
        
        # 문제 유형에 따라 다른 예시 사용
        examples = correction_examples if question_type == "교정형" else selection_examples
        
        # 최종 질문 생성
        final_question = f"{instruction}\n\n{formatted_context}\n\n{examples}\n\n질문: {question}\n\n답변: "
        
        # 결과 데이터셋에 추가
        result_dataset.append({
            "question": final_question,
            "answer": answer
        })
    
    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"데이터셋 생성 완료: {output_path}")
    print(f"총 {len(result_dataset)}개의 항목이 생성되었습니다.")

if __name__ == "__main__":
    main()
