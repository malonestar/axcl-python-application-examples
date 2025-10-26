# MiniCPM-V-4 Weights

Missing large files:
- `minicpm-v-4_axmodel/` many `llama_p320_l*_together.axmodel`, `llama_post.axmodel`
- `utils/embed_tokens.pth`, `utils/resampler.axmodel`, `utils/siglip.axmodel`

Download from official model sources and place in the paths above. Then run:
```bash
python run_axmodel.py
# or
python minicpmv4_picam.py
