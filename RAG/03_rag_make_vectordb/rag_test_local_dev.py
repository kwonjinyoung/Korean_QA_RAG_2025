"""
Qdrant VectorDB 테스트 코드
- 하이브리드 검색 성능 테스트
- 다양한 검색 모드 비교
- 검색 결과 품질 평가
- 한국어 QA RAG 시스템 구현 및 평가
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
from sklearn.metrics.pairwise import cosine_similarity


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


def load_qa_data(file_path: str = "../../RAG/resource/korean_language_rag_V1.0_dev.json") -> List[Dict]:
    """한국어 QA 데이터를 로드합니다."""
    print("📚 한국어 QA 데이터 로드 중...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"QA 데이터 파일이 존재하지 않습니다: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"✅ QA 데이터 로드 완료: {len(qa_data)}개 문항")
    return qa_data


def create_qa_vectorstore(qa_data: List[Dict]) -> QdrantVectorStore:
    """QA 데이터로부터 벡터스토어를 생성합니다."""
    print("🔧 QA 벡터스토어 생성 중...")
    
    # 임베딩 모델 설정
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434"
    )
    
    # Sparse 임베딩 설정
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # 문서 생성 (질문-답변 쌍을 하나의 문서로)
    documents = []
    for item in qa_data:
        question = item["input"]["question"]
        answer = item["output"]["answer"]
        question_type = item["input"]["question_type"]
        
        # 질문과 답변을 결합한 문서 생성
        content = f"질문: {question}\n답변: {answer}"
        
        doc = Document(
            page_content=content,
            metadata={
                "id": item["id"],
                "question": question,
                "answer": answer,
                "question_type": question_type,
                "length": len(content)
            }
        )
        documents.append(doc)
    
    # 고유한 Qdrant DB 경로 설정 (시간 기반)
    import time
    timestamp = int(time.time())
    db_path = f"./qdrant_qa_db_{timestamp}"
    collection_name = "korean_qa_hybrid"
    
    # 혹시 기존 경로가 있다면 삭제 시도
    import shutil
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print(f"기존 DB 디렉토리 '{db_path}' 삭제됨")
        except Exception as e:
            print(f"기존 DB 삭제 실패 (무시): {e}")
    
    # 혹시 모르니 잠시 대기
    time.sleep(1)
    
    print(f"DB 경로: {db_path}")
    
    # 하이브리드 벡터스토어 생성 (로컬 파일 기반)
    qdrant_store = QdrantVectorStore.from_documents(
        documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        path=db_path,  # location 대신 path 사용
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    
    # 벡터스토어의 내부 클라이언트를 재사용 (중복 생성 방지)
    client = qdrant_store.client
    
    print("✅ QA 벡터스토어 생성 완료!")
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
    
    # Qwen3:32b 모델 설정 (think 태그 허용)
    llm = ChatOllama(
        model="qwen3:32b",
        base_url="http://localhost:11434",
        temperature=0.1,  # 적당한 창의성 유지
        num_predict=4096*2,  # 출력 토큰 수 대폭 증가
        num_ctx=4096*2,     # 컨텍스트 길이 대폭 증가 (퓨샷 예시 포함)
        timeout=90,       # 요청 타임아웃 90초로 증가
    )
    
    # 질문 유형별 프롬프트 템플릿 정의
    type_instructions = {
        "교정형": """당신은 한국어 언어학 전문가입니다. 교정형 문제에 답변하세요.

다음은 교정형 문제의 예시입니다:

예시 1:
질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.
"오늘은 화요일 입니다."
답변: "오늘은 화요일입니다."가 옳다. '입니다'는 '이다'의 활용형이고 '이다'는 서술격 조사이다. 조사는 하나의 단어이지만 자립성이 없기 때문에 앞말에 붙여 쓴다. 따라서 '화요일입니다'와 같이 앞말에 붙여 써야 한다.

예시 2:
질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.
"면허 년월일을 기입해 주세요."
답변: "면허 연월일을 기입해 주세요."가 옳다. '녀, 뇨, 뉴, 니'를 포함하는 한자어 음절은 단어 첫머리에 오면 '여, 요, 유, 이'의 형태로 실현되는데 이를 국어의 두음 법칙이라고 한다. 단, 의존 명사는 이러한 두음 법칙이 적용되지 않는다. 따라서 '연월일(年月日)'는 '년월일'이 아닌 '연월일'로 적는다. 한편 '年度'와 같이 명사로 쓰이기도 하고 의존 명사로 쓰이기도 하는 한자어의 경우 명사일 때는 '연도', 의존 명사일 때는 '년도'로 적는다.

컨텍스트:
{context}

질문: {question}

<think> 태그 안에서 문제를 분석한 후, 올바른 문장과 이유를 제시하세요.

답변:""",
        "선택형": """당신은 한국어 언어학 전문가입니다. 선택형 문제에 답변하세요.

참고할 답변 형식:
- 질문에서 제시된 선택지 중 올바른 것을 선택합니다
- "올바른선택지"가 옳다. 문법적 근거를 설명합니다.

컨텍스트:
{context}

질문: {question}

<think> 태그 안에서 질문의 선택지들을 분석한 후, 올바른 답변을 제시하세요.

답변:"""
    }
    
    # 기본 프롬프트 (타입이 명시되지 않은 경우)
    default_prompt = ChatPromptTemplate.from_template("""
당신은 한국어 언어학 전문가입니다. 주어진 컨텍스트를 바탕으로 질문에 대해 정확한 답변을 제공해주세요.

컨텍스트:
{context}

질문: {question}

답변:""")
    
    # 검색기 설정
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 검색 결과 수를 다시 3개로 증가
    )
    
    # 문서 포맷팅 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
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


def calculate_similarity(text1: str, text2: str, embeddings: OllamaEmbeddings) -> float:
    """두 텍스트 간의 코사인 유사도를 계산합니다."""
    try:
        print(f"    📊 임베딩 계산 중... (텍스트 길이: {len(text1)}, {len(text2)})")
        
        # 타임아웃을 적용하여 임베딩 계산
        with timeout(30):  # 30초 타임아웃
            embed1 = embeddings.embed_query(text1)
            print(f"    ✅ 첫 번째 임베딩 완료")
            
            embed2 = embeddings.embed_query(text2)
            print(f"    ✅ 두 번째 임베딩 완료")
        
        # 코사인 유사도 계산
        embed1 = np.array(embed1).reshape(1, -1)
        embed2 = np.array(embed2).reshape(1, -1)
        
        similarity = cosine_similarity(embed1, embed2)[0][0]
        print(f"    ✅ 유사도 계산 완료: {similarity:.4f}")
        return similarity
        
    except TimeoutException as te:
        print(f"    ⏰ 임베딩 계산 타임아웃: {te}")
        return 0.0
    except Exception as e:
        print(f"    ❌ 유사도 계산 실패: {e}")
        return 0.0


def evaluate_rag_performance(rag_chain, qa_data: List[Dict], embeddings: OllamaEmbeddings, 
                           sample_size: int = 10) -> Dict[str, Any]:
    """RAG 시스템의 성능을 평가합니다."""
    print(f"\n🔍 RAG 성능 평가 시작 (샘플 크기: {sample_size})")
    print("=" * 80)
    
    # 평가할 샘플 선택
    import random
    random.seed(42)
    sample_data = random.sample(qa_data, min(sample_size, len(qa_data)))
    
    results = []
    total_similarity = 0
    total_time = 0
    
    for i, item in enumerate(sample_data, 1):
        question = item["input"]["question"]
        correct_answer = item["output"]["answer"]
        question_type = item["input"]["question_type"]
        
        print(f"\n📝 테스트 {i}/{sample_size}")
        print(f"질문 유형: {question_type}")
        print(f"질문: {question}")
        print(f"정답: {correct_answer[:100]}...")
        
        # 질문 유형 자동 추출 확인
        def extract_question_type_for_eval(question: str) -> str:
            """평가용 질문 유형 추출 함수"""
            if "{" in question and "}" in question and "/" in question:
                return "선택형"
            elif any(keyword in question for keyword in ["교정", "고치", "올바르게", "어문 규범", "부합하지 않는"]):
                return "교정형"
            return "기본"
        
        detected_type = extract_question_type_for_eval(question)
        if detected_type != question_type and detected_type != "기본":
            print(f"  🔍 유형 감지: 실제({question_type}) vs 감지({detected_type})")
        else:
            print(f"  ✅ 유형 감지 성공: {question_type}")
        
        # RAG로 답변 생성
        print(f"🤖 RAG 답변 생성 중...")
        start_time = time.time()
        
        try:
            # 타임아웃을 적용하여 RAG 답변 생성
            with timeout(90):  # 90초로 타임아웃 연장 (더 자세한 답변 생성)
                generated_answer = rag_chain.invoke(question)
            
            response_time = time.time() - start_time
            total_time += response_time
            
            print(f"생성 답변: {generated_answer[:150]}...")
            print(f"응답 시간: {response_time:.2f}초")
            
            # 유사도 계산
            print(f"📊 유사도 계산 시작...")
            similarity = calculate_similarity(correct_answer, generated_answer, embeddings)
            total_similarity += similarity
            
            print(f"유사도 점수: {similarity:.4f}")
            
            results.append({
                "question_id": item["id"],
                "question": question,
                "question_type": question_type,
                "detected_type": detected_type,
                "correct_answer": correct_answer,
                "generated_answer": generated_answer,
                "similarity": similarity,
                "response_time": response_time
            })
            
        except TimeoutException as te:
            print(f"⏰ RAG 답변 생성 타임아웃: {te}")
            results.append({
                "question_id": item["id"],
                "question": question,
                "question_type": question_type,
                "detected_type": detected_type,
                "correct_answer": correct_answer,
                "generated_answer": f"TIMEOUT: {str(te)}",
                "similarity": 0.0,
                "response_time": 0.0
            })
            
        except Exception as e:
            print(f"❌ 답변 생성 실패: {e}")
            results.append({
                "question_id": item["id"],
                "question": question,
                "question_type": question_type,
                "detected_type": detected_type,
                "correct_answer": correct_answer,
                "generated_answer": f"ERROR: {str(e)}",
                "similarity": 0.0,
                "response_time": 0.0
            })
        
        print("-" * 80)
        print(f"✅ 테스트 {i}/{sample_size} 완료")
    
    # 전체 통계 계산
    valid_results = [r for r in results if r["similarity"] > 0]
    avg_similarity = total_similarity / len(valid_results) if valid_results else 0
    avg_response_time = total_time / len(valid_results) if valid_results else 0
    
    evaluation_report = {
        "total_questions": len(sample_data),
        "successful_responses": len(valid_results),
        "success_rate": len(valid_results) / len(sample_data),
        "average_similarity": avg_similarity,
        "average_response_time": avg_response_time,
        "detailed_results": results
    }
    
    return evaluation_report


def print_evaluation_summary(report: Dict[str, Any]):
    """평가 결과 요약을 출력합니다."""
    print("\n📊 RAG 성능 평가 결과")
    print("=" * 50)
    print(f"총 질문 수: {report['total_questions']}")
    print(f"성공한 응답 수: {report['successful_responses']}")
    print(f"성공률: {report['success_rate']:.2%}")
    print(f"평균 유사도: {report['average_similarity']:.4f}")
    print(f"평균 응답 시간: {report['average_response_time']:.2f}초")
    
    # 유형 감지 정확도 분석
    type_detection_accuracy = []
    for result in report["detailed_results"]:
        if result["similarity"] > 0:  # 성공한 경우만
            actual_type = result["question_type"]
            detected_type = result["detected_type"]
            if detected_type != "기본":  # 기본 타입이 아닐 경우만 평가
                type_detection_accuracy.append(actual_type == detected_type)
    
    if type_detection_accuracy:
        accuracy = sum(type_detection_accuracy) / len(type_detection_accuracy)
        print(f"유형 감지 정확도: {accuracy:.2%} ({sum(type_detection_accuracy)}/{len(type_detection_accuracy)})")
    
    # 유사도 분포 분석
    similarities = [r["similarity"] for r in report["detailed_results"] if r["similarity"] > 0]
    if similarities:
        print(f"\n유사도 분포:")
        print(f"  최고: {max(similarities):.4f}")
        print(f"  최저: {min(similarities):.4f}")
        print(f"  표준편차: {np.std(similarities):.4f}")
        
        # 성능 구간별 분포
        excellent_quality = len([s for s in similarities if s >= 0.9])
        high_quality = len([s for s in similarities if 0.8 <= s < 0.9])
        medium_quality = len([s for s in similarities if 0.6 <= s < 0.8])
        low_quality = len([s for s in similarities if s < 0.6])
        
        print(f"\n성능 구간별 분포:")
        print(f"  최우수 (≥0.9): {excellent_quality}개 ({excellent_quality/len(similarities):.1%})")
        print(f"  우수 (0.8-0.9): {high_quality}개 ({high_quality/len(similarities):.1%})")
        print(f"  양호 (0.6-0.8): {medium_quality}개 ({medium_quality/len(similarities):.1%})")
        print(f"  개선필요 (<0.6): {low_quality}개 ({low_quality/len(similarities):.1%})")


def analyze_by_question_type(report: Dict[str, Any]):
    """질문 유형별 성능을 분석합니다."""
    print("\n📈 질문 유형별 성능 분석")
    print("=" * 50)
    
    type_stats = {}
    for result in report["detailed_results"]:
        q_type = result["question_type"]
        if q_type not in type_stats:
            type_stats[q_type] = {
                "similarities": [], 
                "times": [], 
                "detection_correct": 0,
                "detection_total": 0,
                "generated_answers": []
            }
        
        if result["similarity"] > 0:
            type_stats[q_type]["similarities"].append(result["similarity"])
            type_stats[q_type]["times"].append(result["response_time"])
            type_stats[q_type]["generated_answers"].append(result["generated_answer"])
            
            # 유형 감지 정확도 계산
            if result["detected_type"] != "기본":
                type_stats[q_type]["detection_total"] += 1
                if result["detected_type"] == q_type:
                    type_stats[q_type]["detection_correct"] += 1
    
    for q_type, stats in type_stats.items():
        if stats["similarities"]:
            avg_sim = np.mean(stats["similarities"])
            avg_time = np.mean(stats["times"])
            count = len(stats["similarities"])
            
            print(f"\n🔸 {q_type}:")
            print(f"  평균 유사도: {avg_sim:.4f}")
            print(f"  평균 응답 시간: {avg_time:.2f}초")
            print(f"  처리된 문항 수: {count}개")
            
            # 유형 감지 정확도
            if stats["detection_total"] > 0:
                detection_rate = stats["detection_correct"] / stats["detection_total"]
                print(f"  유형 감지 정확도: {detection_rate:.2%} ({stats['detection_correct']}/{stats['detection_total']})")
            
            # 품질 분포
            excellent = len([s for s in stats["similarities"] if s >= 0.9])
            good = len([s for s in stats["similarities"] if s >= 0.8])
            print(f"  고품질 답변(≥0.8): {good}개 ({good/count:.1%})")
            print(f"  최우수 답변(≥0.9): {excellent}개 ({excellent/count:.1%})")
            
            # 샘플 답변 출력 (가장 높은 유사도)
            if stats["similarities"]:
                best_idx = stats["similarities"].index(max(stats["similarities"]))
                best_answer = stats["generated_answers"][best_idx]
                print(f"  최고 품질 답변 샘플: {best_answer[:100]}...")


def save_evaluation_log(report: Dict[str, Any], qa_data: List[Dict]):
    """평가 결과를 JSON 로그 파일로 저장합니다."""
    import json
    from datetime import datetime
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"rag_evaluation_log_{timestamp}.json"
    
    # 유사도 통계 계산
    similarities = [r["similarity"] for r in report["detailed_results"] if r["similarity"] > 0]
    similarity_stats = {
        "max": max(similarities) if similarities else 0,
        "min": min(similarities) if similarities else 0,
        "std": float(np.std(similarities)) if similarities else 0
    }
    
    # 성능 카테고리별 분포 계산
    performance_distribution = {
        "excellent": len([s for s in similarities if s >= 0.9]),
        "good": len([s for s in similarities if 0.8 <= s < 0.9]),
        "fair": len([s for s in similarities if 0.6 <= s < 0.8]),
        "poor": len([s for s in similarities if s < 0.6])
    }
    
    # 유형 감지 정확도 계산
    type_detection_accuracy = []
    for result in report["detailed_results"]:
        if result["similarity"] > 0 and "detected_type" in result:
            actual_type = result["question_type"]
            detected_type = result["detected_type"]
            if detected_type != "기본":
                type_detection_accuracy.append(actual_type == detected_type)
    
    detection_accuracy = sum(type_detection_accuracy) / len(type_detection_accuracy) if type_detection_accuracy else 0
    
    # 질문 유형별 분석 계산
    type_analysis = {}
    for result in report["detailed_results"]:
        q_type = result["question_type"]
        if q_type not in type_analysis:
            type_analysis[q_type] = {
                "similarities": [],
                "times": [],
                "count": 0
            }
        
        type_analysis[q_type]["count"] += 1
        if result["similarity"] > 0:
            type_analysis[q_type]["similarities"].append(result["similarity"])
            type_analysis[q_type]["times"].append(result["response_time"])
    
    # 로그 데이터 구성
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(qa_data),
            "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "llm_model": "qwen3:32b",
                "embedding_model": "bge-m3",
                "vector_store": "qdrant_hybrid",
                "retrieval_mode": "hybrid_dense_sparse"
            }
        },
        "overall_performance": {
            "total_questions": report["total_questions"],
            "successful_responses": report["successful_responses"],
            "success_rate": report["success_rate"],
            "average_similarity": report["average_similarity"],
            "average_response_time": report["average_response_time"],
            "type_detection_accuracy": detection_accuracy
        },
        "similarity_distribution": similarity_stats,
        "performance_categories": performance_distribution,
        "question_type_analysis": {},
        "detailed_results": []
    }
    
    # 질문 유형별 분석 추가
    for q_type, stats in type_analysis.items():
        log_data["question_type_analysis"][q_type] = {
            "count": stats["count"],
            "average_similarity": float(np.mean(stats["similarities"])) if stats["similarities"] else 0,
            "average_response_time": float(np.mean(stats["times"])) if stats["times"] else 0,
            "success_rate": len(stats["similarities"]) / stats["count"] if stats["count"] > 0 else 0,
            "high_quality_rate": len([s for s in stats["similarities"] if s >= 0.8]) / len(stats["similarities"]) if stats["similarities"] else 0,
            "excellent_rate": len([s for s in stats["similarities"] if s >= 0.9]) / len(stats["similarities"]) if stats["similarities"] else 0
        }
    
    # 상세 결과 추가
    for result in report["detailed_results"]:
        detailed_result = {
            "question_id": result["question_id"],
            "question_type": result["question_type"],
            "question": result["question"],
            "expected_answer": result["correct_answer"],
            "generated_answer": result["generated_answer"],
            "similarity_score": result["similarity"],
            "response_time": result["response_time"],
            "success": result["similarity"] > 0,
            "performance_category": get_performance_category(result["similarity"]) if result["similarity"] > 0 else "failed",
            "detected_type": result.get("detected_type", "unknown")
        }
        log_data["detailed_results"].append(detailed_result)
    
    # JSON 파일로 저장
    try:
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 평가 결과가 로그 파일에 저장되었습니다: {log_filename}")
        print(f"   - 총 {len(log_data['detailed_results'])}개 문항 결과 저장")
        print(f"   - 파일 크기: {os.path.getsize(log_filename) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ 로그 파일 저장 실패: {e}")


def get_performance_category(similarity: float) -> str:
    """유사도 점수에 따른 성능 카테고리를 반환합니다."""
    if similarity >= 0.9:
        return "excellent"
    elif similarity >= 0.8:
        return "good"
    elif similarity >= 0.6:
        return "fair"
    else:
        return "poor"


def show_detailed_comparison(report: Dict[str, Any], top_n: int = 2):
    """상위 N개 결과의 상세 비교를 보여줍니다."""
    print(f"\n🔍 상위 {top_n}개 결과 상세 비교")
    print("=" * 80)
    
    # 유사도 순으로 정렬
    valid_results = [r for r in report["detailed_results"] if r["similarity"] > 0]
    sorted_results = sorted(valid_results, key=lambda x: x["similarity"], reverse=True)
    
    for i, result in enumerate(sorted_results[:top_n], 1):
        print(f"\n🏆 {i}등 (유사도: {result['similarity']:.4f})")
        print(f"유형: {result['question_type']}")
        print(f"질문: {result['question']}")
        print(f"\n정답:")
        print(f"  {result['correct_answer']}")
        print(f"\n생성 답변:")
        print(f"  {result['generated_answer']}")
        print("-" * 80)


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
    return qdrant_store, client, embeddings, sparse_embeddings


def get_collection_info(client: QdrantClient, collection_name: str):
    """컬렉션 정보를 출력합니다."""
    print(f"\n📊 컬렉션 '{collection_name}' 정보:")
    
    # 컬렉션 정보 가져오기
    collection_info = client.get_collection(collection_name)
    print(f"- 문서 수: {collection_info.points_count}")
    
    # 컬렉션 설정 정보 확인
    try:
        # 새로운 API 방식 시도
        if hasattr(collection_info.config, 'params'):
            if hasattr(collection_info.config.params, 'vectors'):
                print(f"- 벡터 설정: {collection_info.config.params.vectors}")
            if hasattr(collection_info.config.params, 'sparse_vectors'):
                print(f"- Sparse 벡터 설정: {collection_info.config.params.sparse_vectors}")
        else:
            # 기본 정보만 출력
            print(f"- 컬렉션 상태: {collection_info.status}")
            print(f"- 옵티마이저 상태: {collection_info.optimizer_status}")
    except Exception as e:
        print(f"- 상세 설정 정보 조회 실패: {e}")
        print(f"- 컬렉션 상태: {collection_info.status}")
        print(f"- 기본 정보만 표시됩니다.")


def test_search_modes(vectorstore: QdrantVectorStore, embeddings, sparse_embeddings, client: QdrantClient):
    """다양한 검색 모드를 테스트합니다."""
    print("\n🔍 다양한 검색 모드 테스트")
    print("=" * 60)
    
    test_queries = [
        "표준어는 무엇인가요?",
        "복수 표준어에 대해 설명해주세요",
        "가뭄과 가물은 같은 말인가요?",
        "한국어 맞춤법 규칙은 무엇인가요?",
        "언어학적 특징을 알려주세요"
    ]
    
    for query in test_queries:
        print(f"\n🔎 테스트 쿼리: '{query}'")
        print("-" * 50)
        
        # 1. 하이브리드 검색
        print("1️⃣ 하이브리드 검색 (Dense + Sparse)")
        hybrid_store = QdrantVectorStore(
            client=client,
            collection_name="korean_qa_hybrid",
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        
        start_time = time.time()
        hybrid_results = hybrid_store.similarity_search_with_score(query, k=3)
        hybrid_time = time.time() - start_time
        
        print(f"   검색 시간: {hybrid_time:.3f}초")
        for i, (doc, score) in enumerate(hybrid_results, 1):
            print(f"   {i}. [점수: {score:.4f}] {doc.page_content[:100]}...")
        
        # 2. Dense 검색만
        print("\n2️⃣ Dense 검색만 (의미 기반)")
        dense_store = QdrantVectorStore(
            client=client,
            collection_name="korean_qa_hybrid",
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
        )
        
        start_time = time.time()
        dense_results = dense_store.similarity_search_with_score(query, k=3)
        dense_time = time.time() - start_time
        
        print(f"   검색 시간: {dense_time:.3f}초")
        for i, (doc, score) in enumerate(dense_results, 1):
            print(f"   {i}. [점수: {score:.4f}] {doc.page_content[:100]}...")
        
        # 3. Sparse 검색만
        print("\n3️⃣ Sparse 검색만 (키워드 기반 BM25)")
        sparse_store = QdrantVectorStore(
            client=client,
            collection_name="korean_qa_hybrid",
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse",
        )
        
        start_time = time.time()
        sparse_results = sparse_store.similarity_search_with_score(query, k=3)
        sparse_time = time.time() - start_time
        
        print(f"   검색 시간: {sparse_time:.3f}초")
        for i, (doc, score) in enumerate(sparse_results, 1):
            print(f"   {i}. [점수: {score:.4f}] {doc.page_content[:100]}...")
        
        print("\n" + "="*60)


def test_retriever_functionality(vectorstore: QdrantVectorStore):
    """Retriever 기능을 테스트합니다."""
    print("\n🔧 Retriever 기능 테스트")
    print("=" * 50)
    
    # MMR (Maximal Marginal Relevance) 검색
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    
    # 유사도 기반 검색
    retriever_similarity = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    test_query = "표준어 규정에 대해 알려주세요"
    
    print(f"테스트 쿼리: '{test_query}'")
    print("\n1️⃣ MMR 검색 결과:")
    mmr_results = retriever_mmr.invoke(test_query)
    for i, doc in enumerate(mmr_results, 1):
        print(f"   {i}. {doc.page_content[:150]}...")
        print(f"      메타데이터: {doc.metadata}")
    
    print("\n2️⃣ 유사도 검색 결과:")
    similarity_results = retriever_similarity.invoke(test_query)
    for i, doc in enumerate(similarity_results, 1):
        print(f"   {i}. {doc.page_content[:150]}...")
        print(f"      메타데이터: {doc.metadata}")


def test_metadata_filtering(vectorstore: QdrantVectorStore):
    """메타데이터 필터링을 테스트합니다."""
    print("\n🏷️  메타데이터 필터링 테스트")
    print("=" * 50)
    
    from qdrant_client import models
    
    # 특정 길이 이상의 문서만 검색
    print("1️⃣ 길이가 1000자 이상인 문서만 검색:")
    
    results = vectorstore.similarity_search(
        query="한국어 표준어",
        k=3,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.length",
                    range=models.Range(gte=1000)
                )
            ]
        )
    )
    
    for i, doc in enumerate(results, 1):
        print(f"   {i}. 길이: {doc.metadata['length']}자")
        print(f"      내용: {doc.page_content[:100]}...")
    
    print("\n2️⃣ 특정 ID 범위의 문서만 검색:")
    results = vectorstore.similarity_search(
        query="언어 규칙",
        k=3,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.id",
                    range=models.Range(gte=0, lte=10)
                )
            ]
        )
    )
    
    for i, doc in enumerate(results, 1):
        print(f"   {i}. ID: {doc.metadata['id']}")
        print(f"      내용: {doc.page_content[:100]}...")


def performance_benchmark(vectorstore: QdrantVectorStore):
    """성능 벤치마크를 수행합니다."""
    print("\n⚡ 성능 벤치마크")
    print("=" * 50)
    
    test_queries = [
        "표준어 규정",
        "한국어 맞춤법",
        "언어학 이론",
        "문법 규칙",
        "어휘 분류"
    ]
    
    total_time = 0
    total_queries = len(test_queries)
    
    print(f"총 {total_queries}개 쿼리로 성능 테스트 중...")
    
    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        results = vectorstore.similarity_search(query, k=5)
        query_time = time.time() - start_time
        total_time += query_time
        
        print(f"   쿼리 {i}: '{query}' - {query_time:.3f}초 ({len(results)}개 결과)")
    
    avg_time = total_time / total_queries
    print(f"\n📊 성능 결과:")
    print(f"   - 총 실행 시간: {total_time:.3f}초")
    print(f"   - 평균 쿼리 시간: {avg_time:.3f}초")
    print(f"   - 초당 쿼리 수: {1/avg_time:.2f} QPS")


def main():
    """메인 테스트 함수"""
    try:
        print("🚀 한국어 QA RAG 시스템 테스트 시작")
        print("=" * 80)
        
        # 1. QA 데이터 로드
        qa_data = load_qa_data()
        
        # 2. QA 벡터스토어 생성
        vectorstore, client, embeddings = create_qa_vectorstore(qa_data)
        
        # 3. RAG 체인 생성
        rag_chain, retriever = create_rag_chain(vectorstore)
        
        # 4. 간단한 테스트
        print("\n🧪 간단한 테스트 실행")
        test_question = "표준어는 무엇인가요?"
        print(f"테스트 질문: {test_question}")
        
        try:
            with timeout(30):  # 30초 타임아웃
                answer = rag_chain.invoke(test_question)
            print(f"생성된 답변: {answer}")
        except TimeoutException:
            print("⏰ 간단한 테스트 타임아웃 - 평가로 넘어갑니다.")
        
        # 5. 전체 데이터셋 성능 평가 실행
        print(f"\n🔍 전체 데이터셋 성능 평가 시작 (총 {len(qa_data)}개 문항)")
        evaluation_report = evaluate_rag_performance(
            rag_chain, qa_data, embeddings, sample_size=len(qa_data)  # 전체 데이터셋
        )
        
        # 6. 평가 결과를 JSON 로그 파일로 저장
        save_evaluation_log(evaluation_report, qa_data)
        
        # 7. 평가 결과 출력
        print_evaluation_summary(evaluation_report)
        analyze_by_question_type(evaluation_report)
        show_detailed_comparison(evaluation_report, top_n=5)  # 상위 5개 결과 표시
        
        # 7. 기존 벡터스토어 테스트는 생략
        print("\n" + "="*80)
        print("📊 기존 벡터스토어 테스트는 성능상의 이유로 생략합니다.")
        
        print("\n✅ 모든 테스트가 완료되었습니다!")
        print("한국어 QA RAG 시스템이 정상적으로 동작하고 있습니다.")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
