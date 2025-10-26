#!/usr/bin/env python3
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import soundfile
import onnxruntime as ort
import axengine as axe
import argparse
import time
from split_utils import split_sentence
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from symbols import LANG_TO_SYMBOL_MAP
import re
import gc

# ---------- your original helpers ----------
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def get_text_for_tts_infer(text, language_str, symbol_to_id=None):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)
    phone = intersperse(phone, 0)
    tone = intersperse(tone, 0)
    language = intersperse(language, 0)
    phone = np.array(phone, dtype=np.int32)
    tone = np.array(tone, dtype=np.int32)
    language = np.array(language, dtype=np.int32)
    word2ph = np.array(word2ph, dtype=np.int32) * 2
    if word2ph.size > 0:
        word2ph[0] += 1
    return phone, tone, language, norm_text, word2ph

def split_sentences_into_pieces(text, language, quiet=False):
    # Mirror ZH_MIX_EN split behavior for English
    split_lang = "ZH_MIX_EN" if language == "EN" else language
    texts = split_sentence(text, language_str=split_lang)
    if not quiet:
        print(" > Text split to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
    return texts

def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments

def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]
    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices, zp_slices = [], []
    while pn_end < len(word2pronoun):
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices

# ---------- synth steps ----------
def synth_sentence(se, language, dec_len, speed, sample_rate, sess_enc, sess_dec, g_vec, symbol_to_id):
    if language in ['EN', 'ZH_MIX_EN']:
        se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)

    phones, tones, lang_ids, norm_text, word2ph = get_text_for_tts_infer(se, language, symbol_to_id=symbol_to_id)

    t0 = time.time()
    z_p, pronoun_lens, audio_len = sess_enc.run(
        None,
        input_feed={
            'phone': phones, 'g': g_vec,
            'tone': tones, 'language': lang_ids,
            'noise_scale': np.array([0], dtype=np.float32),
            'length_scale': np.array([1.0 / speed], dtype=np.float32),
            'noise_scale_w': np.array([0], dtype=np.float32),
            'sdp_ratio': np.array([0], dtype=np.float32)
        }
    )
    print(f"encoder run take {(time.time() - t0)*1000:.2f}ms")

    word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
    pn_slices, zp_slices = generate_slices(word2pronoun, dec_len)

    audio_len = int(audio_len[0])
    sub_audio_list = []
    for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
        zp_slice = z_p[..., zs]
        sub_dec_len = zp_slice.shape[-1]
        sub_audio_len = 512 * sub_dec_len
        if sub_dec_len < dec_len:
            zp_slice = np.concatenate(
                (zp_slice, np.zeros((*zp_slice.shape[:-1], dec_len - sub_dec_len), dtype=np.float32)),
                axis=-1
            )
        t1 = time.time()
        audio = sess_dec.run(None, input_feed={"z_p": zp_slice, "g": g_vec})[0].flatten()

        audio_start = 0
        if len(sub_audio_list) > 0 and pn_slices[i - 1].stop > ps.start:
            audio_start = 512 * word2pronoun[ps.start]
        audio_end = sub_audio_len
        if i < len(pn_slices) - 1 and ps.stop > pn_slices[i + 1].start:
            audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]
        audio = audio[audio_start:audio_end]
        print(f"Decode slice[{i}]: decoder run take {(time.time() - t1)*1000:.2f}ms")
        sub_audio_list.append(audio)

    return merge_sub_audio(sub_audio_list, 0, audio_len)

# ---------- CLI / main ----------
def get_args():
    p = argparse.ArgumentParser(prog="melotts_repl_accel", description="MeloTTS preloaded + accelerator + REPL")
    p.add_argument("--wav", "-w", default="output.wav")
    p.add_argument("--encoder", "-e", default=None)
    p.add_argument("--decoder", "-d", default=None)
    p.add_argument("--dec_len", type=int, default=128)
    p.add_argument("--sample_rate", "-sr", type=int, default=44100)
    p.add_argument("--speed", type=float, default=1.2)
    p.add_argument("--language", "-l", choices=["ZH", "ZH_MIX_EN", "JP", "EN", "KR", "ES", "SP", "FR"], default="EN")
    p.add_argument("--text", "-s", help="Optional one-shot text before REPL")
    return p.parse_args()

def main():
    args = get_args()
    language = "ZH_MIX_EN" if args.language == "ZH" else args.language

    enc_model = args.encoder or ("models/encoder-zh.onnx" if "ZH" in language else f"models/encoder-{language.lower()}.onnx")
    dec_model = args.decoder or ("models/decoder-zh.axmodel" if "ZH" in language else f"models/decoder-{language.lower()}.axmodel")
    assert os.path.exists(enc_model), f"Encoder model ({enc_model}) not exist!"
    assert os.path.exists(dec_model), f"Decoder model ({dec_model}) not exist!"

    print(f"sample_rate: {args.sample_rate}")
    print(f"encoder: {enc_model}")
    print(f"decoder: {dec_model}")
    print(f"language: {language}")

    symbol_to_id = {s: i for i, s in enumerate(LANG_TO_SYMBOL_MAP[language])}

    # ---------- Preload language (moves the 8–9s hit to startup)
    t_lang = time.time()
    try:
        clean_text("预热 warmup" if language == "ZH_MIX_EN" else "warmup", language)
    except Exception:
        pass
    print(f"Load language module take {(time.time()-t_lang)*1000:.6f}ms")

    sess_enc = sess_dec = None
    try:
        # IMPORTANT: build AXEngine decoder first (initializes AX stack)
        t0 = time.time()
        sess_dec = axe.InferenceSession(dec_model)

        # Now open ORT encoder with AXCLRT EP (like your working script)
        sess_enc = ort.InferenceSession(
            enc_model,
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions()
        )
        print(f"load models take {(time.time()-t0)*1000:.3f}ms")

        # conditioning vector
        g_path = f"../g-{language.lower()}.bin"
        assert os.path.exists(g_path), f"Missing conditioning vector: {g_path}"
        g_vec = np.fromfile(g_path, dtype=np.float32).reshape(1, 256, 1)

        def run_once(text: str):
            t = time.time()
            sens = split_sentences_into_pieces(text, language, quiet=True)
            print(f"split_sentences_into_pieces take {(time.time()-t)*1000:.3f}ms")
            audio_list = []
            for idx, se in enumerate(sens):
                print(f"\nSentence[{idx}]: {se}")
                audio_list.append(
                    synth_sentence(se, language, args.dec_len, args.speed, args.sample_rate,
                                   sess_enc, sess_dec, g_vec, symbol_to_id)
                )
            audio = audio_numpy_concat(audio_list, sr=args.sample_rate, speed=args.speed)
            soundfile.write(args.wav, audio, args.sample_rate)
            print(f"Save to {args.wav}")
            print(f"[Total] {(time.time()-t):.3f}s")

        # one-shot
        if args.text:
            run_once(args.text)

        # REPL
        print("\n[MeloTTS] Ready. Type text and press Enter. Commands: :q to quit")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Exit]")
                break
            if line in (":q", ":quit", ":exit"):
                print("[Exit]")
                break
            if not line:
                continue
            run_once(line)

    finally:
        try: del sess_enc
        except: pass
        try: del sess_dec
        except: pass
        gc.collect()

if __name__ == "__main__":
    main()
