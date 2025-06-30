import json
import torch
from torch.utils.data import Dataset
import random
import sys
import os
from pathlib import Path

# RAG 시스템 임포트
sys.path.append(str(Path(__file__).parent.parent / "rag_data_maker"))
try:
    from qdrant_client import QdrantClient
    from maker import BGEM3Embeddings
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ RAG 모듈을 찾을 수 없습니다. RAG 기능 없이 진행합니다.")
    RAG_AVAILABLE = False


class RAGRetriever:
    """RAG 벡터 데이터베이스 검색기"""
    
    def __init__(self, collection_name="korean_rag_collection", qdrant_url="http://localhost:6333", top_k=3):
        self.collection_name = collection_name
        self.top_k = top_k
        self.client = None
        self.embeddings = None
        self.is_initialized = False
        
        if not RAG_AVAILABLE:
            print("⚠️ RAG 시스템을 사용할 수 없습니다.")
            return
        
        try:
            # Qdrant 클라이언트 초기화
            self.client = QdrantClient(url=qdrant_url, check_compatibility=False)
            
            # 컬렉션 존재 확인
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_name not in collection_names:
                print(f"⚠️ 컬렉션 '{collection_name}'이 존재하지 않습니다.")
                return
            
            # BGE-M3 임베딩 모델 로드
            self.embeddings = BGEM3Embeddings()
            self.is_initialized = True
            print(f"✅ RAG 시스템 초기화 완료 (컬렉션: {collection_name})")
            
        except Exception as e:
            print(f"⚠️ RAG 시스템 초기화 실패: {e}")
            self.is_initialized = False
    
    def retrieve(self, query: str) -> str:
        """질문에 대한 관련 문서 검색"""
        if not self.is_initialized:
            return ""
        
        try:
            # Dense 검색 수행
            query_embedding = self.embeddings.embed_query(query)
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                limit=self.top_k
            )
            
            if not search_results:
                return ""
            
            # 검색 결과를 컨텍스트로 변환
            context_parts = []
            for i, result in enumerate(search_results, 1):
                text = result.payload.get("text", "").strip()
                score = result.score
                if text and score > 0.3:  # 최소 유사도 임계값
                    context_parts.append(f"[참고자료 {i}] (유사도: {score:.3f})\n{text}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"⚠️ RAG 검색 실패: {e}")
            return ""


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer, max_length=2048, use_rag=True, rag_top_k=3):
        IGNORE_INDEX = -100
        self.inp = []
        self.label = []
        self.max_length = max_length
        self.use_rag = use_rag

        # RAG 시스템 초기화
        self.rag_retriever = None
        if use_rag and RAG_AVAILABLE:
            self.rag_retriever = RAGRetriever(top_k=rag_top_k)
            if not self.rag_retriever.is_initialized:
                print("⚠️ RAG 시스템을 사용할 수 없어 일반 모드로 진행합니다.")
                self.use_rag = False

        PROMPT = '''You are a helpful AI assistant specialized in Korean language, culture, history, grammar, and various academic fields. \
            당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트입니다. \
            제공된 참고자료를 바탕으로 사용자의 질문에 대해 정확하고 친절하게 답변해주세요. \
            참고자료가 질문과 관련이 없다면 일반적인 지식을 바탕으로 답변하되, 추측이나 불확실한 정보는 제공하지 마세요. \
            단, 동일한 문장을 절대 반복하지 마시오.'''

        with open(fname, "r", encoding="utf-8") as f:
            data = json.load(f)

        def make_chat(inp):
            # question type별 instruction 정의
            type_instructions = {
                "선다형": (
                    "[질문]을 잘 읽고 참고자료를 활용하여 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n"
                    "참고자료가 있다면 이를 근거로 판단하고, 없다면 일반적인 지식을 활용하시오.\n\n"
                    "[예시]\n"
                    "질문: 다음 한국의 전통 놀이 중 '조선시대'에 행한 놀이는?\n"
                    "1) 주사위 놀이\n"
                    "2) 검무\n"
                    "3) 격구\n"
                    "4) 영고\n"
                    "5) 무애무\n"
                    "답변: 3"
                ),
                "서술형": (
                    "[질문]을 잘 읽고 참고자료를 활용하여 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "질문에 대한 답변을 완성된 문장으로 서술하시오.\n"
                    "참고자료가 있다면 이를 근거로 설명하고, 없다면 일반적인 지식을 활용하시오.\n\n"
                    "[예시]\n"
                    "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
                    "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
                ),
                "단답형": (
                    "[질문]을 잘 읽고 참고자료를 활용하여 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n"
                    "참고자료가 있다면 이를 근거로 판단하고, 없다면 일반적인 지식을 활용하시오.\n\n"
                    "[예시]\n"
                    "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
                    "답변: 정약용"
                ),
                "교정형": (
                    "[질문]을 잘 읽고 참고자료를 활용하여 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n"
                    "참고자료의 문법 규정을 우선적으로 활용하시오.\n\n"
                    "[예시]\n"
                    "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
                    "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
                ),
                "선택형": (
                    "[질문]을 잘 읽고 참고자료를 활용하여 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n"
                    "참고자료의 문법 규정을 우선적으로 활용하시오.\n\n"
                    "[예시]\n"
                    "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
                    "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
                )
            }

            # question type에 따른 instruction 선택
            instruction = type_instructions.get(inp['question_type'], "")

            # RAG 검색 수행
            context = ""
            if self.use_rag and self.rag_retriever and self.rag_retriever.is_initialized:
                context = self.rag_retriever.retrieve(inp['question'])

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}
            
            # 프롬프트 구성 요소들
            chat_parts = [instruction]
            
            # 참고자료 추가 (RAG 검색 결과)
            if context:
                chat_parts.append(f"[참고자료]\n{context}")
            
            # 기타 정보가 있는 경우에만 추가
            if other_info:
                info_list = ["[기타 정보]"]
                for key, value in other_info.items():
                    info_list.append(f"- {key}: {value}")
                chat_parts.append("\n".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문]\n{inp['question']}")

            # 최종 프롬프트 생성
            chat = "\n\n".join(chat_parts)

            return chat
        
        print(f"데이터 로딩 중... 총 {len(data)}개 샘플")
        if self.use_rag:
            print("🔍 RAG 시스템을 사용하여 관련 문서를 검색합니다.")
        else:
            print("📚 일반 instruction 모드로 진행합니다.")
        
        for i, example in enumerate(data):
            if i % 50 == 0:  # RAG 검색 때문에 더 자주 출력
                print(f"처리 중: {i}/{len(data)}")
                
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )

            target = example.get("output", {}).get("answer", "")
            if target != "":
                target += tokenizer.eos_token
            
            target_tokens = tokenizer(
                target,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target_tokens["input_ids"] = target_tokens["input_ids"].type(torch.int64)

            # 길이 제한 체크 (RAG 컨텍스트 때문에 더 엄격하게)
            total_length = source[0].shape[0] + target_tokens["input_ids"][0].shape[0]
            if total_length > self.max_length:
                # 너무 긴 경우 target을 줄임
                max_target_length = self.max_length - source[0].shape[0] - 10
                if max_target_length > 0:
                    target_tokens["input_ids"] = target_tokens["input_ids"][:, :max_target_length]
                else:
                    print(f"⚠️ 샘플 {i} 건너뛰기: 길이 초과 (총 길이: {total_length})")
                    continue  # 건너뛰기

            input_ids = torch.concat((source[0], target_tokens["input_ids"][0]))
            labels = torch.concat((
                torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), 
                target_tokens["input_ids"][0]
            ))
            
            self.inp.append(input_ids)
            self.label.append(labels)
        
        print(f"데이터 로딩 완료! 총 {len(self.inp)}개 샘플 처리됨")
        if self.use_rag and self.rag_retriever and self.rag_retriever.is_initialized:
            print("✅ RAG 기반 컨텍스트가 포함된 데이터셋이 생성되었습니다.")

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inp[idx],
            "labels": self.label[idx]
        }


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        
        # 패딩 적용
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
