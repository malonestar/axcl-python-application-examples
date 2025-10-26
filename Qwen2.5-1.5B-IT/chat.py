import sys
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from ml_dtypes import bfloat16
import dataclasses
from transformers import AutoTokenizer, AutoConfig
import torch
from torchvision.transforms.functional import InterpolationMode
from axengine import InferenceSession
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import argparse
from loguru import logger
from copy import deepcopy
from utils.infer_func import InferManager, KVCacheTools


class LlamaChatSession:
    def __init__(self, builder_instance):
        self.system_prompt = builder_instance.system_prompt
        self.builder_instance = builder_instance
        self.last_reply = ""

    def encode(self, prompt: str) -> Tuple[List[int], List[int]]:
        """
        keys: "message", "model_inputs", "input_ids", "input_embeds", "input_ids_len"
        """
        return self.builder_instance.encoder_prompt(prompt)

    def get_kvcache(self) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        return self.builder_instance.k_caches, self.builder_instance.v_caches

    def generate(self,
        model_inputs
    ):
        token_ids = model_inputs["input_ids"]
        self.builder_instance.decode(token_ids)
        return None

    def run(self, model_inputs) -> str:
        response = self.generate(
            model_inputs
        )
        return response

    def reset_context(self, system_prompt: str = None):
        if system_prompt is not None:
            self.system_prompt = system_prompt

        self.builder_instance.precompute_len = self.builder_instance.system_input_ids_len

        for i in range(len(self.builder_instance.k_caches)):
            self.builder_instance.k_caches[i][:, self.builder_instance.precompute_len:, :].fill(0)
            self.builder_instance.v_caches[i][:, self.builder_instance.precompute_len:, :].fill(0)

    def chat_loop(self, live_print: bool = False):

        if self.system_prompt:
            print(f">>> System Prompt: {self.system_prompt}")

        logger.info("Type 'q' to exit, Ctrl+c to stop current generation\n")

        while True:
            try:
                prompt = input("prompt (Type q to Quit) >> ")

                if prompt.lower() == "q" or prompt.lower() == "exit":
                    print("\nOK, Exited Conversation.")
                    return

                if prompt.lower() == "debug":
                    print(f"\n>>> DEBUG INFO >>>\n precompute_len is {self.builder_instance.precompute_len}\n<<< DEBUG INFO <<<\n")
                    continue

                if not prompt.strip():
                    print(f"\n{self.system_prompt}")
                    continue

                if prompt.strip() == "reset":
                    self.reset_context()
                    print("Context has been reset.")
                    continue

                model_inputs = self.encode(prompt)

                if self.builder_instance.precompute_len + 128 >= 2559:
                    logger.info("ERROR: Context window is full! Please use the reset command to Reset the context")
                    continue

                response = self.run(model_inputs)

            except KeyboardInterrupt:
                # Ctrl+C
                print("\nOkay, successfully exited the conversation..")
                exit()
                
            except Exception as e:
                print(f"ERROR: {str(e)}")


if __name__ == "__main__":

    hf_model_path = 'models/Qwen2.5-1.5B-Instruct-GPTQ-Int8/'
    axmodel_model_path = 'models/Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel/'

    builder = InferManager(hf_model_path, axmodel_model_path) # init tokenizer & hf_config & system prompt
    builder.build_system_prompt()
    builder.build_kvcache()
    builder.build_infer_model()

    cache_manager = KVCacheTools(axmodel_num=28, dtype=bfloat16)

    if not os.path.exists("./kvcache"):
        # system prompt k,v 
        update_kv_cache = builder.prefill(
            builder.model_inputs,
            slice_len=128,
        )
        if cache_manager.save_kvcache(
            target_dir="./kvcache",
            system_prompt=builder.system_prompt,
            precompute_len=builder.system_input_ids_len,
            k_caches=update_kv_cache[0],
            v_caches=update_kv_cache[1],
            metadata={"model_version": "v0.1"}
        ):
            logger.info(">>> Pre-calculating system prompt KV cache, saving to ./kvcache directory. The cache can be loaded directly on the next startup. <<<")
        else:
            logger.error(">>> kvcache Cache save failed, program exiting! <<<")
            exit()
    else:
        update_kv_cache, prompt, plen, meta = cache_manager.load_kvcache("./kvcache")
        builder.update_kvcache(update_kv_cache)

    logger.debug(">>> LlamaChatSession >>>")

    session = LlamaChatSession(
        builder_instance=builder
    )
    session.chat_loop(live_print=False)
