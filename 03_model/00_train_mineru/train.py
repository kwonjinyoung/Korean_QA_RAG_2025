import os
import orjson
import fire
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from trl import DPOConfig, SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

import wandb

class CustomDataset(torch.utils.data.Dataset):
    """
    한국어 QA RAG 데이터셋을 위한 사용자 정의 데이터셋 클래스
    
    데이터 구조:
    [
      {
        "question": "Instruction:\n...\n질문: ...",
        "answer": "..."
      },
      ...
    ]
    """
    def __init__(self, data_path, tokenizer, max_length=2048):
        """
        데이터셋 초기화
        
        Args:
            data_path: 데이터 파일 경로
            tokenizer: 토크나이저 객체
            max_length: 최대 시퀀스 길이
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # JSON 파일에서 데이터 로드
        with open(data_path, mode="r", encoding="utf-8") as f:
            raw_data = orjson.loads(f.read())
        
        # 데이터 전처리: 특수 토큰 제거
        self.data = self._preprocess_data(raw_data)
        
        print(f"{data_path}에서 {len(self.data)} 개의 데이터를 로드했습니다.")
    
    def _preprocess_data(self, raw_data):
        """
        데이터 전처리 함수
        
        Args:
            raw_data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        processed_data = []
        special_tokens = ["[답변]", "[/답변]", "[답문]", "[/답문]"]
        
        for item in raw_data:
            question = item["question"]
            answer = item["answer"]
            
            # 답변에서 특수 토큰 제거
            for token in special_tokens:
                answer = answer.replace(token, "")
            
            # 공백 정리
            answer = answer.strip()
            
            # 전처리된 데이터 추가
            processed_data.append({
                "question": question,
                "answer": answer
            })
            
        return processed_data
    
    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터 항목 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            dict: 토큰화된 입력 및 라벨
        """
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        input_text = question
        
        # 전체 텍스트: 질문 + 구분자 + 답변 + EOS 토큰
        # 답변이 완전히 끝나도록 EOS 토큰을 명시적으로 추가
        full_text = input_text + answer + self.tokenizer.eos_token
        
        # 입력 부분(질문 + 구분자) 토큰화
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )
        
        # 전체 텍스트(질문 + 구분자 + 답변) 토큰화
        # add_special_tokens=True로 설정하여 EOS 토큰이 자동으로 추가되도록 함
        full_encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # 배치 차원 제거
        input_ids = full_encodings.input_ids.squeeze(0)
        attention_mask = full_encodings.attention_mask.squeeze(0)
        
        # 라벨 설정: 입력 부분은 -100으로 마스킹, 답변 부분만 실제 라벨
        labels = input_ids.clone()
        
        # 입력 부분(질문 + 구분자)의 길이 계산
        input_length = input_encodings.input_ids.shape[1]
        
        # 입력 부분은 -100으로 마스킹 (loss 계산에서 제외)
        if input_length < len(labels):
            labels[:input_length] = -100
        
        # 패딩 토큰에 대한 라벨도 -100으로 설정
        padding_mask = attention_mask == 0
        labels[padding_mask] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }



class TrainCli:
    def __init__(self):
        self.token = os.getenv("HF_TOKEN")
        self.write_token = os.getenv("HF_WRITE_TOKEN")

    def run(
        self,
        is_lora: bool = False,
        is_sft: bool = True,
        is_dpo: bool = False,
        is_unsloth: bool = False,
        model_path: str = "unsloth/gemma-3-12b-it",
        # model_path: str = "unsloth/Qwen3-8B",
        # model_path: str = "unsloth/Qwen3-14B",
    ):
        if is_unsloth:
            is_lora = False
            if not (
                model_path.find("unsloth") != -1
            ):
                raise Exception("베이스 모델이 unsloth이 아닙니다.")
        if is_lora:
            is_unsloth = False
        self.model_name = model_path

        if is_unsloth:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                dtype=None,
                device_map="auto",
                load_in_4bit=False,
                max_seq_length=4096,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        if is_lora:
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "up_proj",
                    "o_proj",
                    "k_proj",
                    "down_proj",
                    "gate_proj",
                    "v_proj",
                ],
                lora_dropout=0.5,
                use_dora=False,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
        if is_unsloth:
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=16,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=0.5,
                bias="none",
                use_gradient_checkpointing="unsloth",
                use_rslora=False,
                loftq_config=None,
            )
				# Todo : 데이터셋 다시 로드 작업 해주셔야 합니다.
        
        train_data_path = "/home/n/Korean_QA_RAG_2025/02_makeDataset_for_train/final_dataset.json"
        tokenized_datasets = CustomDataset(train_data_path, tokenizer, max_length=4096)
        wandb_project: str = "{}".format(self.model_name.split("/")[-1])
        wandb_entity: str = "mineru"
        wandb_run_name: str = ""
        wandb_watch: str = ""
        wandb_log_model: str = ""
        wandb.login()

        use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
        )
        if len(wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = wandb_watch
        if len(wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = wandb_log_model

        EOS_TOKEN = tokenizer.eos_token

        if torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        wandb.init(entity=wandb_entity, project=wandb_project)

        if is_unsloth:
            if is_sft:
                training_args = SFTConfig(
                    eval_strategy="no",
                    output_dir=f"./output",
                    warmup_steps=1000,
                    logging_steps=1,
                    learning_rate=1e-5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    optim="adamw_8bit",
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    save_strategy="steps",
                    save_total_limit=3,
                    save_steps=1000,
                    report_to="wandb" if use_wandb else None,
                    run_name=wandb_run_name if use_wandb else None,
                    lr_scheduler_type="linear",
                )
            if is_dpo:
                training_args = DPOConfig(
                    eval_strategy="no",
                    output_dir=f"./output",
                    warmup_steps=1000,
                    logging_steps=1,
                    learning_rate=1e-5,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    optim="adamw_8bit",
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    save_strategy="steps",
                    save_total_limit=3,
                    save_steps=1000,
                    report_to="wandb" if use_wandb else None,
                    run_name=wandb_run_name if use_wandb else None,
                    lr_scheduler_type="linear",
                )
        else:
            if is_sft:
                training_args = SFTConfig(
                    eval_strategy="no",
                    output_dir=f"./output",
                    warmup_steps=1000,
                    logging_steps=1,
                    learning_rate=1e-5,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    optim="adamw_torch",
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    save_strategy="steps",
                    save_total_limit=3,
                    save_steps=1000,
                    report_to="wandb" if use_wandb else None,
                    run_name=wandb_run_name if use_wandb else None,
                    lr_scheduler_type="linear",
                )
            if is_dpo:
                training_args = DPOConfig(
                    eval_strategy="no",
                    output_dir=f"./output",
                    warmup_steps=1000,
                    logging_steps=1,
                    learning_rate=1e-5,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    num_train_epochs=1,
                    optim="adamw_torch",
                    fp16=not torch.cuda.is_bf16_supported(),
                    bf16=torch.cuda.is_bf16_supported(),
                    save_strategy="steps",
                    save_total_limit=3,
                    save_steps=1000,
                    report_to="wandb" if use_wandb else None,
                    run_name=wandb_run_name if use_wandb else None,
                    lr_scheduler_type="linear",
                )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            packing=False,
            data_collator=data_collator,
            peft_config=peft_config if is_lora else None,
        )
        trainer.train()

        repo_name = self.model_name
        if is_unsloth:
            model.push_to_hub_merged(
                repo_name,
                tokenizer,
                save_method="merged_16bit",
                token=self.write_token,
            )
        else:
            if is_lora:
                model = model.merge_and_unload()
            model.save_pretrained(f"./result/{wandb_project}")
            tokenizer.save_pretrained(f"./result/{wandb_project}")

            model.push_to_hub(repo_name, token=self.write_token)
            tokenizer.push_to_hub(repo_name, token=self.write_token)


if __name__ == "__main__":
    fire.Fire(TrainCli)
