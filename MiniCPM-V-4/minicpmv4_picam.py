import os
import time
import gc
import argparse
import threading
import numpy as np
from copy import deepcopy
from PIL import Image

# ----- Torch / HF / AX Engine -----
import torch
from ml_dtypes import bfloat16
import axengine as ort
from transformers import AutoProcessor, AutoTokenizer, AutoConfig

# ----- Camera / UI -----
import cv2
from picamera2 import Picamera2

# =============================
#   MiniCPM-V4 runtime (adapted from your run_axmodel.py)
# =============================

def post_process(data, topk=1, topp=0.9, temperature=0.6):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32).flatten()
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    candidate_value /= temperature
    candidate_soft = softmax(candidate_value)
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return np.array(next_token), candidate_index, candidate_soft


class MiniCPMV:
    def __init__(self, siglip_path, resampler_path, embed_token_path, llm_ax_path, config):
        self.config = config
        self.vpm = ort.InferenceSession(siglip_path)
        self.resampler = ort.InferenceSession(resampler_path)
        self.embed_tokens = torch.load(embed_token_path, weights_only=False)

        self.prefill_slice_len = 320
        self.kv_cache_len = 1023
        self.prefill_decoder_sessions = []
        for i in range(self.config.num_hidden_layers):
            sess = ort.InferenceSession(
                f"{llm_ax_path}/llama_p{self.prefill_slice_len}_l{i}_together.axmodel"
            )
            self.prefill_decoder_sessions.append(sess)

        self.post_process_session = ort.InferenceSession(
            f"{llm_ax_path}/llama_post.axmodel"
        )
        self.kv_dim = 256
        self.terminators = ['<|im_end|>', '</s>']
        print("[MiniCPM] All sessions loaded.")

    def get_position_ids(self, pixel_values: torch.FloatTensor,
                         patch_attention_mask: torch.BoolTensor, tgt_sizes: torch.IntTensor=None):
        batch_size = pixel_values.size(0)
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h = max_im_h // self.config.vision_config.patch_size
        max_nb_patches_w = max_im_w // self.config.vision_config.patch_size
        num_patches_per_side = self.config.vision_config.image_size // self.config.vision_config.patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(size=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        return position_ids

    @torch.no_grad()
    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = torch.float32
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            for pixel_values in pixel_values_list:
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            if all_pixel_values:  # has images
                tgt_sizes = [t for t in tgt_sizes if isinstance(t, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])), dtype=torch.bool)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                all_pixel_values = all_pixel_values.type(dtype)

                position_ids = self.get_position_ids(all_pixel_values,
                                                     patch_attention_mask=patch_attn_mask,
                                                     tgt_sizes=tgt_sizes)
                siglip_inputs = {
                    "all_pixel_values": all_pixel_values.numpy(),
                    "position_ids": position_ids.numpy().astype(np.int32),
                }
                vision_embedding = self.vpm.run(None, input_feed=siglip_inputs)[0]
                resampler_inputs = { "vision_embedding": vision_embedding }
                vision_embedding = self.resampler.run(None, input_feed=resampler_inputs)[0]
                vision_embedding = torch.from_numpy(vision_embedding)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:
                vision_hidden_states = [[] for _ in range(len(pixel_values_list))]
        else:
            vision_hidden_states = data['vision_hidden_states']

        vllm_embedding = self.embed_tokens(data['input_ids'])
        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
                                for i in vision_hidden_states]

        bs = len(data['input_ids'])
        embed_dim = vllm_embedding.shape[-1]

        new_vllm_embeddings = []
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            cur_vllm_emb = vllm_embedding[i]
            if len(cur_vs_hs) == 0:
                new_vllm_embeddings.append(cur_vllm_emb); continue

            cur_image_bound = data['image_bound'][i]
            if len(cur_image_bound) > 0:
                image_indices = torch.stack([
                    torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound
                ], dim=0).flatten()
                indices_expanded = image_indices.view(-1, 1).expand(-1, embed_dim)
                vision_features = cur_vs_hs.view(-1, embed_dim)
                updated_emb = cur_vllm_emb.scatter(0, indices_expanded, vision_features)
                new_vllm_embeddings.append(updated_emb)
            else:
                new_vllm_embeddings.append(cur_vllm_emb)

        vllm_embedding = torch.stack(new_vllm_embeddings, dim=0)
        return vllm_embedding, vision_hidden_states

    def decode(self, inputs_embeds, tokenizer, attention_mask, input_ids_init, max_new_tokens=64, stream_print=False):
        token_ids = input_ids_init.tolist()[0]
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]

        k_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
                    for _ in range(self.config.num_hidden_layers)]
        v_caches = [np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
                    for _ in range(self.config.num_hidden_layers)]

        new_tokens = 0
        token_len = inputs_embeds.shape[1]
        prefill_slice_len = self.prefill_slice_len
        slice_indexs = [e for e in range(token_len // prefill_slice_len + 1)]
        prefill_len = prefill_slice_len * slice_indexs[-1] if slice_indexs[-1] != 0 else prefill_slice_len

        # ----- Prefill -----
        if prefill_len > 0:
            for slice_index in slice_indexs:
                indices = np.array(list(range(slice_index * prefill_slice_len,
                                            (slice_index + 1) * prefill_slice_len)), np.uint32).reshape((1, prefill_slice_len))
                mask = (np.zeros((1, prefill_slice_len, prefill_slice_len * (slice_index + 1))) - 65536).astype(bfloat16)
                data = np.zeros((1, prefill_slice_len, self.config.hidden_size)).astype(bfloat16)

                for i, t in enumerate(range(slice_index * prefill_slice_len, (slice_index + 1) * prefill_slice_len)):
                    if t < token_len:
                        mask[:, i, : slice_index * prefill_slice_len + i + 1] = 0
                        data[:, i:i+1, :] = inputs_embeds[0][t].reshape((1, 1, self.config.hidden_size)).astype(bfloat16)

                remain_len = token_len - slice_index * prefill_slice_len if slice_index == slice_indexs[-1] else prefill_slice_len

                for i in range(self.config.num_hidden_layers):
                    input_feed = {
                        "K_cache": (k_caches[i][:, 0: prefill_slice_len * slice_index, :]
                                    if slice_index else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)),
                        "V_cache": (v_caches[i][:, 0: prefill_slice_len * slice_index, :]
                                    if slice_index else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    outputs = self.prefill_decoder_sessions[i].run(None, input_feed, shape_group=slice_index + 1)
                    k_caches[i][:, slice_index * prefill_slice_len: slice_index * prefill_slice_len + remain_len, :] = outputs[0][:, :remain_len, :]
                    v_caches[i][:, slice_index * prefill_slice_len: slice_index * prefill_slice_len + remain_len, :] = outputs[1][:, :remain_len, :]
                    data = outputs[2]

            post_out = self.post_process_session.run(None, {
                "input": data[:, token_len - (len(slice_indexs) - 1) * prefill_slice_len - 1, None, :]
            })[0]
            next_token, _, _ = post_process(post_out)
            token_ids.append(next_token)
            token_ids_cached = [next_token] if stream_print else []
            generated_tokens = [int(next_token)]
            new_tokens += 1
            if next_token in terminators or new_tokens >= max_new_tokens:
                if stream_print:
                    if len(token_ids_cached) > 0:
                        msg = tokenizer.decode(token_ids_cached).replace("\ufffd", "")
                        print(msg, end='\n<|im_end|>\n', flush=True)
                else:
                    msg = tokenizer.decode(generated_tokens).replace("\ufffd", "")
                    print(msg, end='\n<|im_end|>\n', flush=True)
                return
        else:
            # No prefill path
            token_ids_cached = []        # used only for streaming prints
            generated_tokens = []        # accumulate for non-stream print

        # Build initial mask after prefill
        mask = (np.zeros((1, 1, self.kv_cache_len + 1), dtype=np.float32) - 65536).astype(bfloat16)
        if prefill_len > 0:
            mask[:, :, :token_len + 1] = 0

        # ----- Decode -----
        for start_indice in range(self.kv_cache_len):
            if prefill_len > 0 and start_indice < token_len:
                continue

            next_token = token_ids[start_indice]
            indices = np.array([start_indice], np.uint32).reshape((1, 1))
            data = self.embed_tokens(torch.from_numpy(next_token)).reshape((1, 1, self.config.hidden_size)).detach().numpy().astype(bfloat16)
            for i in range(self.config.num_hidden_layers):
                input_feed = {
                    "K_cache": k_caches[i],
                    "V_cache": v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.prefill_decoder_sessions[i].run(None, input_feed, shape_group=0)
                k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]

            mask[..., start_indice + 1] = 0

            if start_indice >= token_len - 1:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, _, _ = post_process(post_out)
                token_ids.append(next_token)
                generated_tokens.append(int(next_token))
                new_tokens += 1

                # Stop conditions
                if next_token in terminators or new_tokens >= max_new_tokens:
                    if stream_print:
                        if len(token_ids_cached) > 0:
                            msg = tokenizer.decode(token_ids_cached).replace("\ufffd", "")
                            print(msg, end='\n<|im_end|>\n', flush=True)
                    else:
                        msg = tokenizer.decode(generated_tokens).replace("\ufffd", "")
                        print(msg, end='\n<|im_end|>\n', flush=True)
                    return

                # Streaming chunk prints
                if stream_print:
                    token_ids_cached.append(next_token)
                    if len(token_ids_cached) >= 10:
                        msg = tokenizer.decode(token_ids_cached).replace("\ufffd", "")
                        print(msg, end=" ", flush=True)
                        token_ids_cached.clear()


# =============================
#   Camera + VQA loop
# =============================

def setup_picam2(display_w=1280, display_h=720, main_w=2304, main_h=1296):
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (main_w, main_h)},                                   # high-res stream
        lores={"size": (display_w, display_h), "format": "RGB888"}         # fast RGB lores for display
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)
    return picam2

def build_inputs(processor, image_pil, question):
    msgs = [{'role': 'user', 'content': [image_pil, question]}]

    copy_msgs = deepcopy(msgs)
    images = []
    for i, msg in enumerate(copy_msgs):
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        cur_msgs = []
        for c in content:
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
        msg["content"] = "\n".join(cur_msgs)

    prompt = processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor([prompt], [images], max_slice_nums=None, use_image_id=None, return_tensors="pt", max_length=32768)
    inputs.pop("image_sizes", None)

    model_inputs = {
        "input_ids": inputs.input_ids,
        "image_bound": inputs.image_bound,
        "pixel_values": inputs.pixel_values,
        "tgt_sizes": inputs.tgt_sizes
    }
    return model_inputs, inputs.attention_mask


# ---------- async inference worker ----------
def infer_worker(pil_image, question, processor, minicpm, tokenizer, result_dict, lock, max_new_tokens):
    try:
        model_inputs, attn_mask = build_inputs(processor, pil_image, question)
        inputs_embeds, _ = minicpm.get_vllm_embedding(model_inputs)

        start = time.time()
        print("\n[MiniCPM] Answer (running)...")
        minicpm.decode(
            inputs_embeds.detach().numpy(),
            tokenizer,
            attn_mask,
            model_inputs["input_ids"],
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - start
        print(f"\n[MiniCPM] Inference time: {elapsed:.2f}s")

        with lock:
            result_dict["answer"] = f"done in {elapsed:.2f}s"
    except Exception as e:
        with lock:
            result_dict["answer"] = f"[error] {e}"
    finally:
        with lock:
            result_dict["busy"] = False
        gc.collect()



def main():
    ap = argparse.ArgumentParser(description="MiniCPM-V4 Live Camera VQA")
    ap.add_argument("--hf_model_path", type=str, default="utils/minicpmv4_tokenizer")
    ap.add_argument("--siglip_axmodel", type=str, default="./utils/siglip.axmodel")
    ap.add_argument("--resampler_axmodel", type=str, default="./utils/resampler.axmodel")
    ap.add_argument("--embed_token_path", type=str, default="./utils/embed_tokens.pth")
    ap.add_argument("--minicpm_axmodel", type=str, default="./minicpm-v-4_axmodel")
    ap.add_argument("--question", "-q", type=str, default="Describe the scene.")
    ap.add_argument("--save_dir", type=str, default="./captures")
    ap.add_argument("--snap_stream", choices=["main","lores"], default="lores")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--width", type=int, default=1280)   # lores/display width
    ap.add_argument("--height", type=int, default=720)   # lores/display height
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    question = args.question

    print("[Init] Loading HF processor/tokenizer/config...")
    processor = AutoProcessor.from_pretrained(args.hf_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    processor.image_processor.slice_mode = False

    print("[Init] Loading MiniCPM sessions...")
    minicpm = MiniCPMV(args.siglip_axmodel, args.resampler_axmodel,
                       args.embed_token_path, args.minicpm_axmodel, config)

    # ---- Camera setup (YOLO-style main+lores) ----
    print("[Init] Starting Picamera2 (main=2304x1296, lores RGB888 for display)...")
    picam = setup_picam2(display_w=args.width, display_h=args.height, main_w=2304, main_h=1296)

    cv2.namedWindow("MiniCPM-V-4 Live", cv2.WINDOW_AUTOSIZE)

    last = time.time()
    fps = 0.0
    frame_count = 0

    # non-blocking inference state
    result = {"busy": False, "answer": ""}
    rlock = threading.Lock()

    # color toggle (your setup looks correct with NO conversion)
    swap_rb = True  # default: show RGB directly; press 'c' to toggle if colors look off


    print("\n[Ready]")
    print("  a = ask current question on FULL-RES snapshot (non-blocking)")
    print("  e = edit question (terminal)")
    print("  s = save current DISPLAY frame (lores)")
    print("  c = toggle color swap")
    print("  q = quit\n")
    print(f"[Question] {question}")

    try:
        while True:
            # Live preview from lores
            rgb_lores = picam.capture_array("lores")  # (H,W,3) RGB888

            # Display image (OpenCV expects BGR)
            if swap_rb:
                # force swap to mimic different channel order
                bgr_lores = rgb_lores  # intentionally no conversion (RGB shown as BGR)
            else:
                bgr_lores = cv2.cvtColor(rgb_lores, cv2.COLOR_RGB2BGR)

            # FPS overlay
            now = time.time()
            dt = now - last
            frame_count += 1
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                last = now

            # HUD: FPS, question, controls, status/answer
            y = 30
            cv2.putText(bgr_lores, f"FPS: {fps:.1f}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA); y += 30
            cv2.putText(bgr_lores, f"Q: {question}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA); y += 24
            cv2.putText(bgr_lores, "Keys: [a] ask  [s] save  [e] edit  [c] color  [q] quit",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2, cv2.LINE_AA); y += 24

            with rlock:
                status = "thinking..." if result["busy"] else ("ready" if not result["answer"] else f"ans: {result['answer']}")
            cv2.putText(bgr_lores, f"Status: {status}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("MiniCPM-V-4 Live", bgr_lores)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                swap_rb = not swap_rb
                print(f"[Color] swap_rb = {swap_rb}")

            elif key == ord('s'):
                out_path = os.path.join(args.save_dir, f"snap_lores_{int(time.time())}.jpg")
                cv2.imwrite(out_path, bgr_lores)
                print(f"[Saved] {out_path}")

            elif key == ord('e'):
                print("\nEnter new question (press Enter to keep current):")
                try:
                    new_q = input("> ").strip()
                    if new_q:
                        question = new_q
                        print(f"[Question updated] {question}")
                except EOFError:
                    pass

            elif key == ord('a'):
                with rlock:
                    if result["busy"]:
                        print("[MiniCPM] Busy; ignore new request.")
                        continue
                    result["busy"] = True
                    result["answer"] = ""

                # Take a FULL-RES snapshot from main stream for best VQA quality
                stream_name = None if args.snap_stream == "main" else "lores"
                snap_rgb = picam.capture_array(stream_name) if stream_name else picam.capture_array()
                pil = Image.fromarray(snap_rgb).convert("RGB").resize((448, 448))


                t = threading.Thread(
                    target=infer_worker,
                    args=(pil, question, processor, minicpm, tokenizer, result, rlock, args.max_new_tokens),
                    daemon=True
                )
                t.start()

    finally:
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
