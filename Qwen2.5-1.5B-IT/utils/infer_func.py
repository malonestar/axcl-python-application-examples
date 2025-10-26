import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm
from axengine import InferenceSession
from ml_dtypes import bfloat16
from transformers import AutoTokenizer, AutoConfig
import json
from loguru import logger


class KVCacheTools:
    """
    k, v cache 的本地保存和加载
    """
    def __init__(self, axmodel_num: int, dtype=np.float32):
        self.axmodel_num = axmodel_num
        self.dtype = dtype

    def save_kvcache(
        self,
        target_dir: str,
        system_prompt: str,
        precompute_len: int,
        k_caches: List[np.ndarray],
        v_caches: List[np.ndarray],
        metadata: Optional[Dict] = None
    ) -> bool:
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)

            for i, (k, v) in enumerate(zip(k_caches, v_caches)):
                k.astype(self.dtype).tofile(target_path / f"k_cache_{i}.bin")
                v.astype(self.dtype).tofile(target_path / f"v_cache_{i}.bin")

            config = {
                "precompute_len": precompute_len,
                "system_prompt": system_prompt,
                "axmodel_num": self.axmodel_num,
                "dtype": str(self.dtype),
                "metadata": metadata or {},
            }
            with open(target_path / "config.json", "w", encoding="utf8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Save failed: {str(e)}")
            return False

    def load_kvcache(
        self,
        cache_dir: str
    ) -> Tuple[
        List[np.ndarray], 
        List[np.ndarray], 
        str, 
        int,
        Dict
    ]:
        try:
            cache_path = Path(cache_dir)
            k_caches, v_caches = [], []

            with open(cache_path / "config.json") as f:
                config = json.load(f)

            if config["axmodel_num"] != self.axmodel_num:
                raise ValueError(
                    f"Model layer mismatch: "
                    f"Expected {self.axmodel_num}, got {config['axmodel_num']}"
                )

            for i in range(self.axmodel_num):
                k_data = np.fromfile(cache_path / f"k_cache_{i}.bin", dtype=self.dtype).reshape(1, -1, 256)
                v_data = np.fromfile(cache_path / f"v_cache_{i}.bin", dtype=self.dtype).reshape(1, -1, 256)
                k_caches.append(k_data)
                v_caches.append(v_data)

            return (
                (k_caches, v_caches),
                config["system_prompt"],
                config["precompute_len"],
                config.get("metadata", {})
            )
        except Exception as e:
            print(f"Load failed: {str(e)}")
            exit()


class InferManager:    
    def __init__(self, hf_model_path: str, axmodel_path: str):
        self.device = "cpu"
        self.hf_model_path = hf_model_path
        self.axmodel_path = axmodel_path

        self.hf_config = AutoConfig.from_pretrained(self.hf_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path, trust_remote_code=True, use_fast=False)
        self.system_prompt = (
            "You are Frank, a helpful assistant. "
            "For complex questions or math problems, think step-by-step first "
            "to ensure you have the right answer. After your reasoning, "
            "provide the final, concise answer in two sentences or fewer."
        )
        self.embeds = np.load(f"{self.axmodel_path}/model.embed_tokens.weight.npy")

    def build_system_prompt(self):

        messages = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        self.system_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        self.system_input_ids = self.system_inputs.input_ids[0].cpu().numpy().tolist()
        self.system_input_embeds = np.take(self.embeds, self.system_input_ids, axis=0)
        self.system_input_ids_len = len(self.system_input_ids)
        self.model_inputs = {
            "input_ids": self.system_input_ids,
            "input_embeds": self.system_input_embeds,
            "input_ids_len": self.system_input_ids_len
        }
        self.precompute_len = self.system_input_ids_len
        # logger.info(f"system prompt prompt ids len: {self.system_input_ids_len}")

    def encoder_prompt(self, prompt):

        text = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        input_ids = model_inputs.input_ids[0].cpu().numpy().tolist()
        input_embeds = np.take(self.embeds, input_ids, axis=0)
        input_ids_len = len(input_ids)
        # logger.info(f"user prompt token_len: {input_ids_len}")

        model_inputs = {
            "message": text,
            "model_inputs": model_inputs,
            "input_ids": input_ids,
            "input_embeds": input_embeds,
            "input_ids_len": input_ids_len
        }
        return model_inputs

    def build_kvcache(self, kv_cache_len: int = 2559):

        kv_dim = self.hf_config.hidden_size // self.hf_config.num_attention_heads * self.hf_config.num_key_value_heads
        self.k_caches = [
            np.zeros((1, kv_cache_len, kv_dim), dtype=bfloat16)
            for _ in range(self.hf_config.num_hidden_layers)
        ]
        self.v_caches = [
            np.zeros((1, kv_cache_len, kv_dim), dtype=bfloat16)
            for _ in range(self.hf_config.num_hidden_layers)
        ]

    def get_kvcache(self):
        return [self.k_caches, self.v_caches]

    def update_kvcache(self, update_kv_cache):
        self.k_caches = update_kv_cache[0]
        self.v_caches = update_kv_cache[1]

    def get_tokenizer(self):
        return self.tokenizer

    def get_system_prompt(self):
        return self.system_prompt

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def build_infer_model(self, ):
        self.prefill_decoder_sessins = []

        for i in tqdm(range(self.hf_config.num_hidden_layers), desc="Init InferenceSession"):
            session = InferenceSession(
                f"{self.axmodel_path}/qwen2_p128_l{i}_together.axmodel"
            )
            self.prefill_decoder_sessins.append(session)

        self.post_process_session = InferenceSession(
            f"{self.axmodel_path}/qwen2_post.axmodel"
        )
        print("The models have been loaded!")

    def get_infer_session(self):
        return [self.prefill_decoder_sessins, self.post_process_session]

    @staticmethod
    def _top_p(probs: np.ndarray, p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)
        filtered = probs.copy()
        cumulative = 0
        for idx in sorted_indices[::-1]:
            if cumulative >= p:
                filtered[idx] = 0
            cumulative += filtered[idx]
        return filtered / cumulative

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return (exp_logits / np.sum(exp_logits)).astype(np.float64)

    def post_process(self, logits, top_k=1, top_p=0.9, temperature=0.6):
        logits = logits.astype(np.float32).flatten()
        candidate_indices = np.argpartition(logits, -top_k)[-top_k:]
        candidate_logits = logits[candidate_indices] / temperature
        candidate_probs = self._softmax(candidate_logits)
        candidate_probs = self._top_p(candidate_probs, top_p)
        candidate_probs = candidate_probs.astype(np.float64) / candidate_probs.sum()
        chosen_idx = np.random.multinomial(1, candidate_probs).argmax()
        next_token = candidate_indices[chosen_idx]
        return next_token, candidate_indices, candidate_probs

    def gen_slice_indices(self, token_len, prefill=128, expand=128):
        remaining = max(0, token_len - prefill)
        extra_blocks = (remaining + expand - 1) // expand
        return list(range(extra_blocks + 1))

    def prefill(
        self,
        model_inputs,
        slice_len=128,
        precompute_len=0, # system prompt prefill 的时候, 只能设置为 0
    ):
        """
        Prefill step for chunked inference.
        """
        token_ids = model_inputs["input_ids"]
        token_embeds = model_inputs["input_embeds"]
        token_len = model_inputs["input_ids_len"]

        seq_len = len(token_ids)
        slice_indices = [i for i in range(seq_len // slice_len + 1)]
        print(f"slice_indices: {slice_indices}")
        # total_prefill_len = (
        #     slice_len * slice_indices[-1]
        #     if slice_indices[-1] != 0
        #     else slice_len
        # )
        # slice_indices = self.gen_slice_indices(seq_len)
        total_prefill_len = slice_len * (slice_indices[-1] + 1)
        kv_mask_expand_len = 128

        if total_prefill_len > 0:
            for slice_index in slice_indices:
                if slice_index == 0:
                    current_slice_len = slice_len
                else:
                    current_slice_len = kv_mask_expand_len

                indices = np.array(
                    list(
                        range(
                            precompute_len + slice_index * slice_len,
                            precompute_len + (slice_index + 1) * slice_len,
                        )
                    ),
                    np.uint32,
                ).reshape((1, slice_len))
                indices[:, min(token_len, slice_len):] = 0

                mask = (
                    np.zeros((1, slice_len, current_slice_len * slice_index + slice_len))
                    - 65536
                )
                data = np.zeros((1, slice_len, self.hf_config.hidden_size)).astype(bfloat16)

                for i, t in enumerate(
                    range(
                        slice_index * slice_len,
                        (slice_index + 1) * slice_len,
                    )
                ):
                    if t < len(token_ids):
                        # mask[:, i, 0: slice_index * slice_len + i + 1] = 0
                        data[:, i : i + 1, :] = (
                            token_embeds[t]
                            .reshape((1, 1, self.hf_config.hidden_size))
                            .astype(bfloat16)
                        )
                    if t < len(token_ids) + precompute_len:
                        mask[:, i, 0: slice_index * slice_len + i + 1] = 0

                if slice_index == slice_indices[-1]:
                    curlen_procd = token_len - slice_index * slice_len # curlen_procd 是当前处理数据的长度
                else:
                    curlen_procd = slice_len

                mask = mask.astype(bfloat16)
                for i in range(self.hf_config.num_hidden_layers):
                    input_feed = {
                        "K_cache": (
                            self.k_caches[i][:, 0: current_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.hf_config.hidden_size), dtype=bfloat16)
                        ),
                        "V_cache": (
                            self.v_caches[i][:, 0: current_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.hf_config.hidden_size), dtype=bfloat16)
                        ),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=slice_index + 1)
                    self.k_caches[i][
                        :,
                        slice_index
                        * slice_len + precompute_len : slice_index
                        * slice_len + curlen_procd + precompute_len,
                        :,
                    ] = outputs[0][:, :curlen_procd, :]

                    self.v_caches[i][
                        :,
                        slice_index
                        * slice_len + precompute_len: slice_index
                        * slice_len + curlen_procd + precompute_len,
                        :,
                    ] = outputs[1][:, :curlen_procd, :]

                    data = outputs[2]

                print("slice prefill done", slice_index)
        else:
            print("No prefill needed.")
        # return "Calculated the kv cache of the system prompt."
        return (self.k_caches, self.v_caches)

    def decode(
        self,
        token_ids,
        prefill_len=128,
        slice_len=128
    ):
        token_len = len(token_ids)
        # set to decoder
        print("answer: >> ", end='', flush=True)
        kv_cache_len = 2559
        mask = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :kv_cache_len] -= 65536
        if prefill_len > 0:
            mask[:, :, :token_len + self.precompute_len] = 0

        for start_indice in range(kv_cache_len):
            if self.precompute_len > 0 and start_indice < self.precompute_len:
                continue
            next_token = token_ids[start_indice - self.precompute_len]
            indices = np.array([start_indice], np.uint32).reshape((1, 1))
            data = self.embeds[next_token, :].reshape((1, 1, self.hf_config.hidden_size)).astype(bfloat16)
            for i in range(self.hf_config.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[i],
                    "V_cache": self.v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.prefill_decoder_sessins[i].run(None, input_feed, shape_group=0)
                self.k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
                self.v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
                data = outputs[2]
            mask[..., start_indice] = 0
            if start_indice < token_len + self.precompute_len - 1:
                pass
            else:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, posssible_tokens, possible_soft = self.post_process(post_out)
                token_ids.append(next_token)
                print(self.tokenizer.decode(next_token, skip_special_tokens=True), end='', flush=True)

                if next_token == self.tokenizer.eos_token_id and start_indice > token_len + self.precompute_len:
                    # print("\n>> HINT: The next_token encountered EOS token, generation completed.")
                    break
        print("\n")
        self.precompute_len = len(token_ids) + self.precompute_len - 1
        return self.tokenizer.decode(token_ids[self.precompute_len - 1:], skip_special_tokens=True)

