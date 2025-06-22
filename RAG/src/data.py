import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. \
            당신은 한국의 전통 문화와 역사, 문법, 사회, 과학기술 등 다양한 분야에 대해 잘 알고 있는 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요. \
            단, 동일한 문장을 절대 반복하지 마시오.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            # question type별 instruction 정의
            type_instructions = {
                "선다형": (
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 보기 중에서 가장 적절한 답을 숫자로만 응답하시오.\n\n"
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
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "질문에 대한 답변을 완성된 문장으로 서술하시오.\n\n"
                    "[예시]\n"
                    "질문: 대한민국의 행정구역 체계를 서술하세요.\n"
                    "답변: 대한민국의 행정구역은 여러 종류의 지역 단위로 나뉘어 구성되어 있으며, 먼저 특별시와 광역시부터 살펴볼 수 있다. 특별시로는 수도인 서울특별시가 있으며, 광역시에는 인천광역시, 부산광역시, 대전광역시, 광주광역시, 대구광역시, 울산광역시 등이 포함된다. 이 외에도 대한민국은 일반 도 단위로 6개의 도를 두고 있는데, 그 이름은 경기도, 충청북도, 충청남도, 전라남도, 경상북도, 경상남도로 구성되어 있다. 특별한 자치권을 부여받은 도인 특별자치도로는 제주특별자치도, 전북특별자치도, 강원특별자치도가 있다. 마지막으로 특별자치시로는 세종특별자치시가 존재한다."
                ),
                "단답형": (
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "질문에 대한 답을 2단어 이내로 간단히 답하시오.\n\n"
                    "[예시]\n"
                    "질문: 조선 후기의 실학 사상가로 목민심서를 쓴 인물은?\n"
                    "답변: 정약용"
                ),
                "교정형": (
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 문장이 올바른지 판단하고, 틀린 경우 올바르게 교정하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
                    "[예시]\n"
                    "질문: 다음 문장에서 어문 규범에 부합하지 않는 부분을 찾아 고치고, 그 이유를 설명하세요.\n\"오늘은 퍼즐 마추기를 해 볼 거예요.\"\n"
                    "답변: \"오늘은 퍼즐 맞추기를 해 볼 거예요.\"가 옳다. '제자리에 맞게 붙이다, 주문하다, 똑바르게 하다, 비교하다' 등의 뜻이 있는 말은 '마추다'가 아닌 '맞추다'로 적는다."
                ),
                "선택형": (
                    "[질문]을 잘 읽고 답변을 생성하시오. 문제를 그대로 출력하지 마시오.\n"
                    "[지침]\n"
                    "주어진 보기들 중에서 가장 적절한 것을 선택하여 \"~가 옳다.\" 형태로 답변하고, 그 이유를 설명하시오.\n\n"
                    "[예시]\n"
                    "질문: \"나는 그를 본 적이 있음을 {기억해냈다/기억해 냈다}.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요.\n"
                    "답변: \"나는 그를 본 적이 있음을 기억해 냈다.\"가 옳다. '기억해 냈다'는 '기억하-+-아+냈다'의 구성이다. 이처럼 '본용언+-아/-어+보조 용언' 구성인 경우 본용언과 보조 용언을 붙여 쓰는 것이 허용되지만, 이러한 구성을 갖더라도 앞말이 3음절 이상의 합성어나 파생어라면 보조 용언을 붙여 쓰는 것이 허용되지 않는다. '기억하다'는 '기억'과 '-하다'가 결합한 파생어이며 '기억해'는 3음절이다. 따라서 '기억해'와 '냈다'는 띄어 써야 한다."
                )
            }

            # question type에 따른 instruction 선택
            instruction = type_instructions.get(inp['question_type'], "")

            # 기타 정보 생성 (question과 question_type 제외)
            other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}
            
            # 기타 정보가 있는 경우에만 추가
            chat_parts = [instruction]
            if other_info:
                info_list = ["[기타 정보]"]
                for key, value in other_info.items():
                    info_list.append(f"- {key}: {value}")
                chat_parts.append("\n".join(info_list))

            # 질문 추가
            chat_parts.append(f"[질문]\n{inp['question']}")

            # 최종 프롬프트 생성
            chat = "\n\n".join(chat_parts)

            # print(f'[DBG] chat: {chat}')

            return chat
        
        for example in data:
            user_prompt = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            if False: print(f'[DBG] message: {message}')
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=False
            )

            target = example.get("output", "")
            if target != "":
                target += tokenizer.eos_token
            target = tokenizer(
                target,
                return_attention_mask=False,
                add_special_tokens=False,
                return_tensors="pt"
            )
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
