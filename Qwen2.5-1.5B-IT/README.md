
# Qwen2.5-1.5B-IT-Int8

This example is from the Axera HF model zoo.  You'll need to clone that repo to get the model and tokenizer files and it looks like Axera has just moved this python example over into it's own clean repo, which sort of makes this example unneccessary!  

https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct-python

```
git clone https://huggingface.co/AXERA-TECH/Qwen2.5-1.5B-Instruct-python
```

For these scripts, move the model files to:
- `models/Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel/` (AX models)
- copy or move the Qwen2.5-tokenizer directory into this directory

Fetch weights from the official source and drop them into the indicated folders.
Use `infer.py` or `chat.py` after placing files.
