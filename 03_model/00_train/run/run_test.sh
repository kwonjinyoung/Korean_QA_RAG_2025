uv run python -m run.test \
  --input resource/RAG/korean_language_rag_V1.0_test.json \
  --output result555.json \
  --model_id results/qwen3-32b-4bit-korean-qa-improved-2/checkpoint-720 \
  --device cuda:0 \
  --use_4bit_quantization \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_use_double_quant