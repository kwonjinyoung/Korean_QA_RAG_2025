"""
Qdrant VectorDB 테스트 코드
- 한국어 QA RAG 시스템으로 실제 테스트 수행
- 테스트 결과를 JSON 파일로 저장
"""

import os
import time
import json
import numpy as np
import re
from typing import List, Dict, Any, Tuple
import signal
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client import QdrantClient


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(duration):
    """컨텍스트 매니저를 사용한 타임아웃 처리"""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"작업이 {duration}초 내에 완료되지 않았습니다.")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def load_test_data(file_path: str = "../../RAG/resource/korean_language_rag_V1.0_test.json") -> List[Dict]:
    """한국어 QA 테스트 데이터를 로드합니다."""
    print("📚 한국어 QA 테스트 데이터 로드 중...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"테스트 데이터 파일이 존재하지 않습니다: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✅ 테스트 데이터 로드 완료: {len(test_data)}개 문항")
    return test_data


def load_existing_vectorstore():
    """기존에 구축된 Qdrant 벡터스토어를 로드합니다."""
    print("🔄 기존 Qdrant 벡터스토어 로드 중...")
    
    # DB 경로 확인
    db_path = "./qdrant_local_db"
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
        raise ValueError(f"컬렉션 '{collection_name}'이 존재하지 않습니다. 먼저 makeDB_local_2.py를 실행하세요.")
    
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
    return qdrant_store, client, embeddings


def clean_model_output(text: str) -> str:
    """모델 출력에서 think 태그와 불필요한 부분을 제거합니다."""
    if not text:
        return text
    
    original_text = text
    
    # <think>...</think> 부분 제거 (다중 줄 포함)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # think 태그가 잘못 닫힌 경우 처리
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 불필요한 답변 라벨 제거
    text = re.sub(r'^\s*답변:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*\*답변:\*\*\s*', '', text, flags=re.MULTILINE)
    
    # 추가적인 정리
    text = text.strip()
    
    # 너무 짧은 답변이나 의미없는 답변인 경우 원본에서 다른 부분 찾기
    if len(text) < 10 or text in ['...', '..', '.']:
        # 원본에서 quote 부분 찾기
        quote_pattern = r'"[^"]*"[가가]?\s*옳다[.]?'
        quote_matches = re.findall(quote_pattern, original_text)
        if quote_matches:
            # 첫 번째 quote와 "옳다" 부분 사용
            text = quote_matches[0]
            
            # 이유 설명 부분도 찾기
            reason_pattern = r'옳다[.]?\s*([^<]*?)(?:\.|$)'
            reason_match = re.search(reason_pattern, original_text)
            if reason_match:
                reason = reason_match.group(1).strip()
                if reason and len(reason) > 5:
                    text += " " + reason
                    if not text.endswith('.'):
                        text += '.'
    
    # 빈 줄 제거하고 깔끔하게 정리
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('<'):
            lines.append(line)
    
    text = '\n'.join(lines)
    
    # 여전히 너무 짧은 경우 기본 응답
    if len(text.strip()) < 5:
        text = "답변을 생성할 수 없습니다."
    
    return text


def create_rag_chain(vectorstore: QdrantVectorStore):
    """RAG 체인을 생성합니다."""
    print("🔗 RAG 체인 생성 중...")
    
    # Qwen3:8b 모델 설정 (think 태그 허용)
    llm = ChatOllama(
        model="qwen3:8b",
        base_url="http://localhost:11435",
        temperature=0.1,  # 적당한 창의성 유지
        num_predict=4096*2,  # 출력 토큰 수 대폭 증가
        num_ctx=4096*2,     # 컨텍스트 길이 대폭 증가 (퓨샷 예시 포함)
        timeout=300,       # 요청 타임아웃 300초로 증가
    )
    
    # 질문 유형별 프롬프트 템플릿 정의
    type_instructions = {
        "교정형": """# Instruction:
1. 당신은 한국어 언어학 전문가입니다. 한국어 표준어 규정에 따라 생각하고 답변하세요.
2. [질문]을 잘 읽고 답변을 생성하시오. 문장을 정확히 이해하고 올바른 답을 작성해야 합니다.
3. 질문과 답변 예시:
    예시 1:
    질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.
    "오늘은 화요일 입니다."    
    답변: "오늘은 화요일입니다."가 옳다. '입니다'는 '이다'의 활용형이고 '이다'는 서술격 조사이다. 조사는 하나의 단어이지만 자립성이 없기 때문에 앞말에 붙여 쓴다. 따라서 '화요일입니다'와 같이 앞말에 붙여 써야 한다.

    예시 2:
    질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.
    "면허 년월일을 기입해 주세요."    
    답변: "면허 연월일을 기입해 주세요."가 옳다. '녀, 뇨, 뉴, 니'를 포함하는 한자어 음절은 단어 첫머리에 오면 '여, 요, 유, 이'의 형태로 실현되는데 이를 국어의 두음 법칙이라고 한다. 단, 의존 명사는 이러한 두음 법칙이 적용되지 않는다. 따라서 '연월일(年月日)'는 '년월일'이 아닌 '연월일'로 적는다. 한편 '年度'와 같이 명사로 쓰이기도 하고 의존 명사로 쓰이기도 하는 한자어의 경우 명사일 때는 '연도', 의존 명사일 때는 '년도'로 적는다.

5. Context를 참고하여 질문에 대한 답변을 생성하세요.

---

<Context>
{context}
</Context>

---

질문: {question}
답변: """,

        "선택형": """# Instruction:
1. 당신은 한국어 언어학 전문가입니다. 한국어 표준어 규정에 따라 생각하고 답변하세요.
2. [질문]을 잘 읽고 답변을 생성하시오. 문장을 정확히 이해하고 올바른 답을 작성해야 합니다.
3. 질문과 답변 예시:
    예시 1:
    질문: "여왕개미는 {{수개미/숫개미}}보다 더 크다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.
    답변: "여왕개미는 수개미보다 더 크다."가 옳다. 수컷을 이르는 접두사는 '수-'로 통일한다. 다만 '수-'가 '강아지, 개, 것, 기와, 닭, 당나귀, 돌쩌귀, 돼지, 병아리'와 결합할 때는 접두사 다음에서 나는 거센소리를 인정하고, '양, 염소, 쥐'와 결합하는 경우는 예외적으로 '숫-'을 쓴다. '개미'는 '숫-'을 쓰는 예외에 속하지도 않고 접두사 다음에서 나는 거센소리를 인정하지도 않으므로 '수개미'가 옳다.

    예시 2:
    질문: "저기 서 있는 저 나무 {{한그루가/한 그루가}} 몹시 쓸쓸해 보였다." 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.
    답변: "저기 서 있는 저 나무 한 그루가 몹시 쓸쓸해 보였다."가 옳다. 단위를 나타내는 말은 의존 명사이든 자립 명사이든 하나의 단어로 인정되는 명사이므로 앞말과 띄어 써야 한다.

5. Context를 참고하여 질문에 대한 답변을 생성하세요.

---

<Context>
{context}
</Context>

---

질문: {question}
답변: """
    }
    
    # 기본 프롬프트 (타입이 명시되지 않은 경우)
    default_prompt = ChatPromptTemplate.from_template("""
당신은 한국어 언어학 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 대해 정확한 답변을 제공해주세요.

---

<Context>
{context}
</Context>

---

질문: {question}

답변:""")
    
    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # 검색 결과 수를 다시 3개로 증가
    )
    
    # 문서 포맷팅 함수
    def format_docs(docs):
        return "\n".join("<Content>\n" + doc.page_content + "\n</Content>" for doc in docs)
    
    # 질문 유형을 추출하는 함수
    def extract_question_type(question: str) -> str:
        """질문에서 유형을 추출하거나 패턴으로 판단"""
        # 선택형 패턴 감지
        if "{" in question and "}" in question and "/" in question:
            return "선택형"
        # 교정형 패턴 감지
        elif any(keyword in question for keyword in ["교정", "고치", "올바르게", "어문 규범", "부합하지 않는"]):
            return "교정형"
        # 기본값
        return "기본"
    
    # 동적 프롬프트 선택 함수
    def get_dynamic_prompt(inputs):
        question = inputs["question"]
        context = inputs["context"]
        
        # 질문 유형 추출
        question_type = extract_question_type(question)
        
        if question_type in type_instructions:
            # 해당 유형의 프롬프트 사용
            prompt_template = type_instructions[question_type]
            return prompt_template.format(context=context, question=question)
        else:
            # 기본 프롬프트 사용
            return default_prompt.format_messages(context=context, question=question)[0].content
    
    # RAG 체인 구성 (동적 프롬프트 적용 + 후처리 + 재시도 로직)
    from langchain_core.runnables import RunnableLambda
    
    def create_dynamic_chain():
        def process_query(question):
            max_retries = 2  # 최대 2회 재시도
            
            for attempt in range(max_retries + 1):
                try:
                    # 1. 검색 수행
                    docs = retriever.invoke(question)
                    context = format_docs(docs)

                    #print("----- Context -----\n", context, "\n----- Context -----\n\n" )
                    
                    
                    # 2. 동적 프롬프트 생성
                    prompt_text = get_dynamic_prompt({"question": question, "context": context})
                    
                    # 3. LLM 호출
                    result = llm.invoke(prompt_text)
                    
                    
                    # 4. 결과 정리 (think 태그 제거)
                    raw_output = result.content if hasattr(result, 'content') else str(result)
                    
                    # 디버깅을 위해 원본 출력 확인
                    if attempt > 0:
                        print(f"    🔄 재시도 {attempt}회차 - 원본 출력 길이: {len(raw_output)}")
                    
                    cleaned_output = clean_model_output(raw_output)
                    print(f"답변: {cleaned_output}")
                    
                    # log 파일에 prompt_text append 추가. 없으면 파일 생성도
                    if not os.path.exists("prompt_text.log"):
                        with open("prompt_text.log", "w", encoding="utf-8") as f:
                            f.write("prompt_text.log 파일 생성\n")
                        
                    with open("prompt_text.log", "a", encoding="utf-8") as f:
                        f.write(prompt_text + cleaned_output + "\n\n")
                    
                    # 5. 성공적인 답변인지 확인
                    if cleaned_output != "답변을 생성할 수 없습니다." and len(cleaned_output.strip()) > 10:
                        return cleaned_output
                    elif attempt < max_retries:
                        print(f"    ⚠️ 답변 품질 불량, 재시도 중... ({attempt + 1}/{max_retries})")
                        # 재시도 시 약간의 무작위성 추가
                        llm.temperature = min(0.3, llm.temperature + 0.1)
                        continue
                    else:
                        # 최종 시도에서도 실패 시, 원본 출력을 더 관대하게 처리
                        print(f"    🚨 최종 시도 실패, 원본 출력 사용 시도")
                        if len(raw_output.strip()) > 5:
                            # think 태그만 제거하고 나머지는 보존
                            fallback_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE)
                            fallback_output = fallback_output.strip()
                            if len(fallback_output) > 10:
                                return fallback_output
                        
                        return cleaned_output  # 마지막 수단
                        
                except Exception as e:
                    print(f"    ❌ 시도 {attempt + 1} 실패: {e}")
                    if attempt < max_retries:
                        continue
                    else:
                        return "답변 생성 중 오류가 발생했습니다."
            
            return "답변을 생성할 수 없습니다."
        
        return RunnableLambda(process_query)
    
    rag_chain = create_dynamic_chain()
    
    print("✅ RAG 체인 생성 완료!")
    return rag_chain, retriever


def run_test_exam(rag_chain, test_data: List[Dict]) -> List[Dict]:
    """RAG 시스템으로 테스트를 수행합니다."""
    print(f"\n🎯 RAG 시스템 테스트 시작 (총 {len(test_data)}문항)")
    print("=" * 80)
    
    results = []
    
    for i, item in enumerate(test_data, 1):
        question_id = item["id"]
        question = item["input"]["question"]
        question_type = item["input"]["question_type"]
        
        print(f"\n📝 문제 {i}/{len(test_data)} (ID: {question_id})")
        print(f"유형: {question_type}")
        #print(f"질문: {question[:100]}...")
        print(f"질문: {question}")
        
        # RAG로 답변 생성
        #print(f"🤖 답변 생성 중...")
        start_time = time.time()
        
        try:
            # 타임아웃을 적용하여 RAG 답변 생성
            with timeout(300):  # 300초 타임아웃
                generated_answer = rag_chain.invoke(question)
            
            response_time = time.time() - start_time
            
            #print(f"생성 답변: {generated_answer}")
            print(f"응답 시간: {response_time:.2f}초")
            
            # 결과 저장
            result = {
                "id": question_id,
                "input": {
                    "question_type": question_type,
                    "question": question
                },
                "output": {
                    "answer": generated_answer
                }
            }
            results.append(result)
            
        except TimeoutException as te:
            print(f"⏰ 답변 생성 타임아웃: {te}")
            result = {
                "id": question_id,
                "input": {
                    "question_type": question_type,
                    "question": question
                },
                "output": {
                    "answer": f"답변 생성 시간 초과: {str(te)}"
                }
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ 답변 생성 실패: {e}")
            result = {
                "id": question_id,
                "input": {
                    "question_type": question_type,
                    "question": question
                },
                "output": {
                    "answer": f"답변 생성 오류: {str(e)}"
                }
            }
            results.append(result)
        
        print("-" * 80)
        print(f"✅ 문제 {i}/{len(test_data)} 완료")
    
    return results


def save_test_results(results: List[Dict]):
    """테스트 결과를 result.json 파일로 저장합니다."""
    output_file = "result.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 테스트 결과가 저장되었습니다: {output_file}")
        print(f"   - 총 {len(results)}개 문항 결과 저장")
        print(f"   - 파일 크기: {os.path.getsize(output_file) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ 결과 파일 저장 실패: {e}")


def print_test_summary(results: List[Dict]):
    """테스트 결과 요약을 출력합니다."""
    print("\n📊 테스트 결과 요약")
    print("=" * 50)
    
    total_questions = len(results)
    successful_answers = 0
    failed_answers = 0
    timeout_answers = 0
    
    type_stats = {}
    
    for result in results:
        answer = result["output"]["answer"]
        question_type = result["input"]["question_type"]
        
        # 유형별 통계 초기화
        if question_type not in type_stats:
            type_stats[question_type] = {
                "total": 0,
                "success": 0,
                "timeout": 0,
                "error": 0
            }
        
        type_stats[question_type]["total"] += 1
        
        # 답변 상태 분류
        if "답변 생성 시간 초과" in answer:
            timeout_answers += 1
            type_stats[question_type]["timeout"] += 1
        elif "답변 생성 오류" in answer or "답변을 생성할 수 없습니다" in answer:
            failed_answers += 1
            type_stats[question_type]["error"] += 1
        else:
            successful_answers += 1
            type_stats[question_type]["success"] += 1
    
    print(f"총 문항 수: {total_questions}")
    print(f"성공한 답변: {successful_answers}개 ({successful_answers/total_questions:.1%})")
    print(f"타임아웃: {timeout_answers}개 ({timeout_answers/total_questions:.1%})")
    print(f"실패한 답변: {failed_answers}개 ({failed_answers/total_questions:.1%})")
    
    print(f"\n📈 유형별 결과:")
    for q_type, stats in type_stats.items():
        success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {q_type}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        if stats["timeout"] > 0:
            print(f"    - 타임아웃: {stats['timeout']}개")
        if stats["error"] > 0:
            print(f"    - 오류: {stats['error']}개")


def main():
    """메인 테스트 함수"""
    try:
        print("🚀 한국어 QA RAG 시스템 테스트 시작")
        print("=" * 80)
        
        # 1. 테스트 데이터 로드
        test_data = load_test_data()
        
        # 테스트용으로 1개 문항만 처리
        #test_data = test_data[:1]
        #print(f"🧪 테스트 모드: {len(test_data)}개 문항만 처리합니다.")
        
        # 2. 기존 벡터스토어 로드
        vectorstore, client, embeddings = load_existing_vectorstore()
        
        # 3. RAG 체인 생성
        rag_chain, retriever = create_rag_chain(vectorstore)
        
        # 4. 간단한 테스트
        # print("\n🧪 간단한 테스트 실행")
        # test_question = "표준어는 무엇인가요?"
        # print(f"테스트 질문: {test_question}")
        
        # try:
        #     with timeout(30):  # 30초 타임아웃
        #         answer = rag_chain.invoke(test_question)
        #     print(f"생성된 답변: {answer}")
        # except TimeoutException:
        #     print("⏰ 간단한 테스트 타임아웃 - 실제 테스트로 넘어갑니다.")
        
        # 5. 실제 테스트 수행
        results = run_test_exam(rag_chain, test_data)
        
        # 6. 결과 저장
        save_test_results(results)
        
        # 7. 결과 요약 출력
        print_test_summary(results)
        
        print("\n✅ 모든 테스트가 완료되었습니다!")
        print("결과가 result.json 파일에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
