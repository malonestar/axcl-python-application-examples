# import llm_utils
import dataclasses
import json
from transformers import AutoTokenizer, AutoConfig
import torch
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import argparse


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

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft


def generate_slice_indices(token_len, prefill=128, expand=512):
    remaining = max(0, token_len - prefill)
    extra_blocks = (remaining + expand - 1) // expand
    return list(range(extra_blocks + 1))


if __name__ == "__main__":

    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="models/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="models/Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("-q", "--question", type=str, default="Please calculate the derivative of the function y=2x^2.",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()

    device = "cpu"
    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path

    cfg = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True, use_fast=False)

    prompt = args.question

    messages = [
        {"role": "system", "content": (
                "You are Frank, a helpful assistant powered by Raspberry Pi 5 and the Axera 8850. "
                "Provide brief and helpful information and if asked to perform calculations, briefly share the steps. "
                "Keep answers short and only share necessary information. "
            )
        },
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    token_ids = model_inputs.input_ids[0].cpu().numpy().tolist()

    embeds = np.load(f"{axmodel_path}/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    token_len = len(token_ids)
    import pdb; pdb.set_trace()

    ##################
    lastN = 2559

    kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
    k_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]

    prefill_decoder_sessins = []

    for i in tqdm(range(cfg.num_hidden_layers), desc="Init InferenceSession"):
        session = InferenceSession(
            f"{axmodel_path}/qwen2_p128_l{i}_together.axmodel"
        )
        prefill_decoder_sessins.append(session)

    post_process_session = InferenceSession(
        f"{axmodel_path}/qwen2_post.axmodel"
    )
    print("model load done!")
    print("prefill token_len: ", token_len)

    """
    Model input shape:
        - kv_cache: g1:[1, 1, hidden_size] -> g2:[1, kv_mask_expand_lens, kv_dim] -> g3:[1, kv_mask_expand_lens * 2, kv_dim]
        - mask: g1:[1, input_prefill_len, input_prefill_len] -> g2:[1, input_prefill_len, input_prefill_len+kv_mask_expand_lens] -> g3:[1, input_prefill_len, input_prefill_len+kv_mask_expand_lens*2]
        - indices: g1:[1, input_prefill_len] -> g2:[1, input_prefill_len] -> g3:[1, input_prefill_len]
        - input: g1:[1, input_prefill_len, hidden_size] -> g2:[1, input_prefill_len, hidden_size] -> g3:[1, input_prefill_len, hidden_size]
    """

    input_prefill_len = 128
    kv_mask_expand_len = 128 # 512

    """
    Model output shape:
        - kv_cache: g1:[1, input_prefill_len, kv_dim] -> g2:[1, input_prefill_len, kv_dim] -> g3:[1, input_prefill_len, kv_dim]
        - output: g1:[1, input_prefill_len, hidden_size] -> g2:[1, input_prefill_len, hidden_size] -> g3:[1, input_prefill_len, hidden_size]
    """
    slice_indexs = generate_slice_indices(token_len, input_prefill_len, input_prefill_len)
    print(f"slice_indexs is {slice_indexs}")

    """
        prefill
    """
    if input_prefill_len > 0:
        for slice_index in slice_indexs:
            if slice_index == 0:
                current_slice_len = input_prefill_len
            else:
                current_slice_len = kv_mask_expand_len

            indices = np.array(
                list(
                    range(
                        slice_index * input_prefill_len,
                        (slice_index + 1) * input_prefill_len,
                    )
                ),
                np.uint32,
            ).reshape((1, input_prefill_len))

            mask = (
                np.zeros((1, input_prefill_len, current_slice_len * slice_index + input_prefill_len))
                - 65536
            )
            data = np.zeros((1, input_prefill_len, cfg.hidden_size)).astype(bfloat16)
            for i, t in enumerate(
                range(
                    slice_index * input_prefill_len,
                    (slice_index + 1) * input_prefill_len,
                )
            ):
                if t < len(token_ids):
                    mask[:, i, : slice_index * input_prefill_len + i + 1] = 0
                    data[:, i : i + 1, :] = (
                        prefill_data[t]
                        .reshape((1, 1, cfg.hidden_size))
                        .astype(bfloat16)
                    )

            if slice_index == slice_indexs[-1]:
                curlen_procd = token_len - slice_index * input_prefill_len # curlen_procd 是当前处理数据的长度
            else:
                curlen_procd = input_prefill_len

            mask = mask.astype(bfloat16)
            for i in range(cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": (
                        k_caches[i][:, 0: current_slice_len * slice_index, :]
                        if slice_index
                        else np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16)
                    ),
                    "V_cache": (
                        v_caches[i][:, 0: current_slice_len * slice_index, :]
                        if slice_index
                        else np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16)
                    ),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=slice_index + 1)

                k_caches[i][
                    :,
                    slice_index
                    * input_prefill_len : slice_index
                    * input_prefill_len + curlen_procd, # current_slice_len
                    :,
                ] = outputs[0][:, :curlen_procd, :]

                v_caches[i][
                    :,
                    slice_index
                    * input_prefill_len : slice_index
                    * input_prefill_len + curlen_procd, # current_slice_len
                    :,
                ] = outputs[1][:, :curlen_procd, :]

                data = outputs[2]

            print("slice prefill done", slice_index)

        post_out = post_process_session.run(
            None,
            {
                "input": data[
                    :, token_len - (len(slice_indexs) - 1) * input_prefill_len - 1, None, :
                ]
            }
        )[0]
        next_token, posssible_tokens, possible_soft = post_process(post_out)
        posibles = [tokenizer.decode([t]) for t in posssible_tokens]
        posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
        token_ids.append(next_token)

    print("answer >>", tokenizer.decode(token_ids[token_len], skip_special_tokens=True), end='', flush=True)
    # print("answer >>", end='', flush=True)

    # set to decoder
    kv_cache_len = lastN
    mask = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :kv_cache_len] -= 65536
    if input_prefill_len > 0:
        mask[:, :, :token_len] = 0

    # for start_indice in tqdm(range(kv_cache_len), desc="Decode"):
    for start_indice in range(kv_cache_len):
        if input_prefill_len > 0 and start_indice < token_len:
            continue

        next_token = token_ids[start_indice]
        indices = np.array([start_indice], np.uint32).reshape((1, 1))
        data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)
        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": k_caches[i],
                "V_cache": v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
            k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
            v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
            data = outputs[2]
        mask[..., start_indice] = 0
        if start_indice < token_len - 1:
            pass
        else:
            post_out = post_process_session.run(None, {"input": data})[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            token_ids.append(next_token)
            if next_token == tokenizer.eos_token_id and next_token > token_len:
                break

        print(tokenizer.decode(next_token, skip_special_tokens=True), end='', flush=True)
