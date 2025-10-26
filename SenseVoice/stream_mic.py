# stream_mic.py
import queue, time, sys, os
import numpy as np
import sounddevice as sd

from utils.SenseVoiceAx import SenseVoiceAx
from utils.tokenizer import SentencepiecesTokenizer
from utils.print_utils import rich_transcription_postprocess

SAMPLE_RATE = 16000

# --------- VAD / segmentation tuning ---------
BLOCK_MS           = 30
BLOCK_SAMPLES      = SAMPLE_RATE * BLOCK_MS // 1000

RMS_THRESH_START   = 0.010   # higher threshold to START speech
RMS_THRESH_END     = 0.006   # slightly lower to END speech (hysteresis)
MIN_SPEECH_MS      = 200     # require at least this much speech to start
END_SIL_MS         = 350     # commit after this much silence
MAX_UTTER_MS       = 5000    # hard cut at this length even without silence
TAIL_CONTEXT_MS    = 250     # small overlap carried to the next chunk

# Optional interim inference (kept off by default; set >0 to enable)
INTERIM_EVERY_MS   = 0       # e.g., 750 for periodic interim
# ---------------------------------------------

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x, dtype=np.float32)) + 1e-12)

def ms_to_samples(ms: int) -> int:
    return int(SAMPLE_RATE * ms / 1000)

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, "models")
    model_path = os.path.join(models_dir, "sensevoice.axmodel")
    bpe_path   = os.path.join(models_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")

    tok = SentencepiecesTokenizer(bpemodel=bpe_path)
    asr = SenseVoiceAx(model_path, max_len=256, language="auto", use_itn=True, tokenizer=tok)

    q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata.mean(axis=1).astype(np.float32, copy=False)
        q.put(mono)

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        dtype="float32",
        callback=audio_cb,
        latency="low"
    )

    speaking = False
    speech_buf = []
    tail = np.empty((0,), dtype=np.float32)   # small overlap tail we prepend after a cut

    above_ms = 0
    silence_ms = 0
    utter_ms = 0
    last_interim_ms = 0
    last_final_text = ""

    print(" Listening… speak into the mic (Ctrl+C to stop)")
    with stream:
        try:
            while True:
                block = q.get()
                speech_buf.append(block)

                e = rms(block)
                # Hysteresis: different thresholds for start vs end
                if speaking:
                    is_speech = e > RMS_THRESH_END
                else:
                    is_speech = e > RMS_THRESH_START

                if is_speech:
                    above_ms += BLOCK_MS
                    silence_ms = 0
                else:
                    silence_ms += BLOCK_MS
                    # small decay on "above" while silent
                    above_ms = max(0, above_ms - BLOCK_MS // 2)

                if not speaking and above_ms >= MIN_SPEECH_MS:
                    speaking = True
                    utter_ms = 0
                    last_interim_ms = 0

                if speaking:
                    utter_ms += BLOCK_MS
                    last_interim_ms += BLOCK_MS

                    # Optional interim inference (does NOT print final)
                    if INTERIM_EVERY_MS and last_interim_ms >= INTERIM_EVERY_MS:
                        last_interim_ms = 0
                        # Take last up to MAX_UTTER_MS window for speed
                        audio = np.concatenate([tail] + speech_buf, axis=0)
                        if audio.size > ms_to_samples(MAX_UTTER_MS):
                            audio = audio[-ms_to_samples(MAX_UTTER_MS):]
                        # You could run asr.infer(audio) here and show partial in UI
                        # Skipping print to keep terminal clean.

                    need_commit = False
                    reason = ""

                    # Normal end of utterance on silence
                    if silence_ms >= END_SIL_MS:
                        need_commit, reason = True, "silence"

                    # Hard cut if user keeps talking forever
                    elif utter_ms >= MAX_UTTER_MS:
                        need_commit, reason = True, "maxlen"

                    if need_commit:
                        # Build audio with small context tail
                        audio_full = np.concatenate([tail] + speech_buf, axis=0)

                        # Keep a tiny tail for next utterance continuity
                        tail_len = ms_to_samples(TAIL_CONTEXT_MS)
                        tail = audio_full[-tail_len:].copy() if tail_len < audio_full.size else audio_full.copy()

                        # Commit region excludes the new tail (avoid duplication)
                        commit_audio = audio_full
                        if tail_len > 0 and commit_audio.size > tail_len:
                            commit_audio = commit_audio[:-tail_len]

                        # Reset state for next utterance
                        speaking = False
                        speech_buf = []
                        above_ms = 0
                        silence_ms = 0
                        utter_ms = 0
                        last_interim_ms = 0

                        # Ignore super short blips
                        if commit_audio.size < ms_to_samples(200):
                            continue

                        t0 = time.time()
                        res = asr.infer(commit_audio, print_rtf=False)
                        dt = time.time() - t0
                        text = " ".join(rich_transcription_postprocess(s) for s in res).strip()

                        # Avoid printing identical repeats
                        if text and text != last_final_text:
                            print(f"\n {text}  (latency: {dt:.3f}s; {reason})\n", flush=True)
                            last_final_text = text

        except KeyboardInterrupt:
            print("\nBye!")

if __name__ == "__main__":
    main()
