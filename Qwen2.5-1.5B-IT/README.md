
# Qwen2.5-1.5B-IT Weights

This repo excludes:
- `models/Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel/` (AX models)
- any `*.npy` (e.g., `model.embed_tokens.weight.npy`)
- runtime `kvcache/*.bin`

Fetch weights from the official source and drop them into the indicated folders.
Use `infer.py` or `chat.py` after placing files.
