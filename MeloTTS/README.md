# MeloTTS Weights

This repo does not include large model files.
Download the encoder/decoder models and language binaries from their official sources (e.g., Hugging Face or vendor links) and place them here:

- `decoder-ax650/` → AX650 decoder(s)
- `encoder-onnx/` → ONNX encoder(s)
- `python/models/` → `decoder-*.axmodel`, `encoder-*.onnx`
- `g-en.bin`, `g-jp.bin`, `g-zh_mix_en.bin` → language resources

> After placing files, run `python/melotts.py` or `melotts_onnx.py` per your workflow.
