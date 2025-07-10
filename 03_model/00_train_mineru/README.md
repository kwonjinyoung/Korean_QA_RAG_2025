## 패키지 설치

```bash
pip install fire wandb
pip install unsloth==2025.6.12 unsloth_zoo==2025.6.8
pip install tf-keras
pip install fastembed
```

## 모델 양자화

```bash
git clone https://github.com/ggml-org/llama.cpp

cd llama.cpp

python llama.cpp/convert_hf_to_gguf.py models/Mineru/Qwen3-8B-1epoch \
  --outfile models/Mineru/Qwen3-8B-1epoch.gguf \
  --outtype f16

./llama.cpp/build/bin/llama-quantize \
    models/Mineru/Qwen3-8B-1epoch-f16.gguf \
    models/Mineru/Qwen3-8B-1epoch-q4_k.gguf \
    Q4_K

ollama create qwen3-8b-1ep-juju:q4 -f Modelfile.q4
ollama create qwen3-8b-1ep-juju:f16 -f Modelfile.f16
```

```bash
python llama.cpp/convert_hf_to_gguf.py models/Mineru/Qwen3-8B-3epoch \
  --outfile models/Mineru/Qwen3-8B-3epoch.gguf \
  --outtype f16

./llama.cpp/build/bin/llama-quantize \
    models/Mineru/Qwen3-8B-3epoch-f16.gguf \
    models/Mineru/Qwen3-8B-3epoch-q4_k.gguf \
    Q4_K

ollama create qwen3-8b-3ep-juju:q4 -f Modelfile.q4
ollama create qwen3-8b-3ep-juju:f16 -f Modelfile.f16
```