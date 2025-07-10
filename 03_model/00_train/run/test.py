import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.data import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--use_auth_token", type=str, help="Hugging Face token for accessing gated models")
g.add_argument("--use_4bit_quantization", action="store_true", default=False, help="4bit 양자화 사용")
g.add_argument("--bnb_4bit_compute_dtype", type=str, default="float16", help="4bit 양자화 연산 데이터 타입")
g.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="4bit 양자화 타입")
g.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True, help="이중 양자화 사용")
# fmt: on


def main(args):
    # 4bit 양자화 설정
    quantization_config = None
    if args.use_4bit_quantization:
        # 데이터 타입 설정
        if args.bnb_4bit_compute_dtype == "float16":
            compute_dtype = torch.float16
        elif args.bnb_4bit_compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        print("4bit 양자화 설정이 활성화되었습니다.")
    
    # torch_dtype 설정 (양자화 사용 시 None, 아니면 기본값 사용)
    torch_dtype = None if quantization_config is not None else torch.bfloat16
    
    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": args.device,
    }
    
    # 양자화 설정 추가
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    if args.use_auth_token:
        model_kwargs["use_auth_token"] = args.use_auth_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs
    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    # Prepare tokenizer loading kwargs
    tokenizer_kwargs = {}
    if args.use_auth_token:
        tokenizer_kwargs["use_auth_token"] = args.use_auth_token
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    file_test = args.input
    dataset = CustomDataset(file_test, tokenizer)

    with open(file_test, "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        data_item = dataset[idx]
        inp = data_item["input_ids"]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05,
            temperature=0.7,
            top_p=0.8,
            # do_sample=False,
        )

        output_text = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        
        # 출력에서 "답변: " 접두어 제거
        if output_text.startswith("답변: "):
            output_text = output_text[4:]
        elif output_text.startswith("답변:"):
            output_text = output_text[3:]
            
        result[idx]["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))