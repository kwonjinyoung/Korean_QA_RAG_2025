# 국어 지식 기반 생성(RAG) Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 경진 대회 과제 중 '국어 지식 기반 생성(RAG)'에 대한 베이스라인 모델의 추론과 평가를 재현하기 위한 코드를 포함하고 있습니다.


추론 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

### Baseline
|Model|Exact Match|ROUGE|BERTScore|BLEURT|
|:---:|---|---|---|---|
|qwen3 8b|||||
|Bllossom 3b|||||
|HyperCLOVAX Text 1.5B|||||

 - '{선택/교정 문장}'이 옳다 부분 : EM
 - 이유 생성: ROUGE, BERTScore, BLEURT


평가 코드 : https://github.com/teddysum/korean_evaluation.git


## Directory Structure
```
# 평가에 필요한 데이터가 들어있습니다.
resource
└── QA

# 실행 가능한 python 스크립트가 들어있습니다.
run
└── test.py

# 학습에 사용될 커스텀 함수들이 구현되어 있습니다.
src
└── data.py   
```

## Data Format
```
{
    "id": "1",
    "input": {
        "question_type": "선택형",
        "question": "\"우동이 {불을/불} 것 같아 걱정이다.\" 가운데 올바른 것을 선택하고, 그 이유를 설명하세요."
    },
    "output": {
        "answer": "\"우동이 불을 것 같아 걱정이다.\"가 옳다. '붇다'의 어간 끝 받침 'ㄷ'은 모음으로 시작하는 어미 앞에서 'ㄹ'로 바뀐다. 따라서 '붇다'에 관형형 어미 '-(으)ㄹ'이 결합하면 '불을'이 된다. 마찬가지로 '깨닫다'는 '깨달을', '듣다'는 '들을'으로 활용한다. 다만, '곧다', '뜯다' 등은 '곧을', '뜯을'과 같이 어간이 바뀌지 않는 형태로 활용한다."
    }
},
```

## How to Run
### Inference
```
python -m run.test \
    --input $testset \
    --output $output \
    --model_id $model \
    --device cuda:1 \
```



## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Bllossome (Teddysum) (https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)


