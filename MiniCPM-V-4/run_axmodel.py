import os
import torch
import argparse
import gc
from copy import deepcopy
from ml_dtypes import bfloat16
import axengine as ort
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig


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
    return np.array(next_token), candidate_index, candidate_soft


class MiniCPMV:
    def __init__(self, siglip_onnx_path, resampler_onnx_path, embed_token_path, llm_axmodel_path, config) -> None:
        self.config = config
        self.vpm = ort.InferenceSession(siglip_onnx_path)
        self.resampler = ort.InferenceSession(resampler_onnx_path)
        self.embed_tokens = torch.load(embed_token_path, weights_only=False) # llm embedding
        
        self.prefill_slice_len=320
        self.kv_cache_len=1023
        self.prefill_decoder_sessions = []
        
        for i in tqdm(range(self.config.num_hidden_layers), desc="Init InferenceSession"):
            session = ort.InferenceSession(
                f"{llm_axmodel_path}/llama_p{self.prefill_slice_len}_l{i}_together.axmodel"
            )
            self.prefill_decoder_sessions.append(session)
        
        self.post_process_session = ort.InferenceSession(
            f"{llm_axmodel_path}/llama_post.axmodel"
        )
        print("model load done!")
        
        self.kv_dim = 256 # self.config.hidden_size // self.config.num_attention_heads * self.config.num_key_value_heads
        
        self.terminators = ['<|im_end|>', '</s>']
        
    def get_position_ids(self, pixel_values: torch.FloatTensor, patch_attention_mask: torch.BoolTensor, tgt_sizes: torch.IntTensor=None):
        batch_size = pixel_values.size(0)
        
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.config.vision_config.patch_size, max_im_w // self.config.vision_config.patch_size
        num_patches_per_side = self.config.vision_config.image_size // self.config.vision_config.patch_size
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

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
            # position_ids[batch_idx] = torch.where(p_attn_mask.view(-1).cpu(), pos_ids, position_ids[batch_idx])
    
        return position_ids
    
    @torch.no_grad()
    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = torch.float32
            device = "cpu"
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                                                                   padding_value=0.0)
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype).to(device=device)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else: # 走这里
                    position_ids = self.get_position_ids(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes)
                    siglip_inputs = {
                        "all_pixel_values": all_pixel_values.numpy(),
                        "position_ids": position_ids.numpy().astype(np.int32),
                    }
                    
                    vision_embedding = self.vpm.run(None, input_feed=siglip_inputs)[0]
                resampler_inputs = {
                    "vision_embedding": vision_embedding,
                    # "tgt_sizes": tgt_sizes.type(torch.int64).numpy(),
                }
                
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
            else: # no image
                if self.training:
                    dummy_image = torch.zeros(
                        (1, 3, 224, 224),
                        device=device, dtype=dtype
                    )
                    tgt_sizes = torch.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']       
        
        vllm_embedding = self.embed_tokens(data['input_ids'])
        
        vision_hidden_states = [i.type(vllm_embedding.dtype) if isinstance(
            i, torch.Tensor) else i for i in vision_hidden_states]

        bs = len(data['input_ids'])
        device = vllm_embedding.device
        embed_dim = vllm_embedding.shape[-1]

        new_vllm_embeddings = []
        
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            cur_vllm_emb = vllm_embedding[i]

            if len(cur_vs_hs) == 0:
                new_vllm_embeddings.append(cur_vllm_emb)
                continue
                
            cur_image_bound = data['image_bound'][i]

            if len(cur_image_bound) > 0:
                image_indices = torch.stack([
                    torch.arange(r[0], r[1], dtype=torch.long) 
                    for r in cur_image_bound
                ], dim=0).flatten().to(device)

                indices_expanded = image_indices.view(-1, 1).expand(-1, embed_dim)
                vision_features = cur_vs_hs.view(-1, embed_dim)
                
                updated_emb = cur_vllm_emb.scatter(0, indices_expanded, vision_features)
                new_vllm_embeddings.append(updated_emb)
            elif self.training:
                dummy_term = cur_vs_hs[0].sum() * 0 
                new_vllm_embeddings.append(cur_vllm_emb + dummy_term)
            else:
                new_vllm_embeddings.append(cur_vllm_emb)

        vllm_embedding = torch.stack(new_vllm_embeddings, dim=0)
        return vllm_embedding, vision_hidden_states

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        token_ids = model_inputs["input_ids"].tolist()[0]
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        
        k_caches = [
            np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
            for _ in range(self.config.num_hidden_layers)
        ]
        v_caches = [
            np.zeros((1, self.kv_cache_len, self.kv_dim), dtype=bfloat16)
            for _ in range(self.config.num_hidden_layers)
        ]
        
        token_len = inputs_embeds.shape[1]  # (B, L, D)
        """
            prefill
        """
        prefill_slice_len = self.prefill_slice_len
        # slice_indexs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        slice_indexs = [
            e for e in range(token_len // prefill_slice_len + 1)
        ]
        prefill_len = prefill_slice_len * slice_indexs[-1] if slice_indexs[-1] != 0 else prefill_slice_len

        if prefill_len > 0:
            for slice_index in tqdm(slice_indexs, desc="prefill"):
                indices = np.array(
                    list(
                        range(
                            slice_index * prefill_slice_len,
                            (slice_index + 1) * prefill_slice_len,
                        )
                    ),
                    np.uint32,
                ).reshape((1, prefill_slice_len))       
        
                mask = (
                    np.zeros((1, prefill_slice_len, prefill_slice_len * (slice_index + 1)))
                    - 65536
                )            
                data = np.zeros((1, prefill_slice_len, self.config.hidden_size)).astype(bfloat16)
                for i, t in enumerate(
                    range(
                        slice_index * prefill_slice_len,
                        (slice_index + 1) * prefill_slice_len,
                    )
                ):
                    if t < token_len:
                        mask[:, i, : slice_index * prefill_slice_len + i + 1] = 0
                        data[:, i : i + 1, :] = (
                            inputs_embeds[0][t]
                            .reshape((1, 1, self.config.hidden_size))
                            .astype(bfloat16)
                        )
                if slice_index == slice_indexs[-1]:
                    remain_len = token_len - slice_index * prefill_slice_len
                else:
                    remain_len = prefill_slice_len
                mask = mask.astype(bfloat16)
                for i in range(self.config.num_hidden_layers):        
                    input_feed = {
                        "K_cache": (
                            k_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "V_cache": (
                            v_caches[i][:, 0 : prefill_slice_len * slice_index, :]
                            if slice_index
                            else np.zeros((1, 1, self.config.hidden_size), dtype=bfloat16)
                        ),
                        "indices": indices,
                        "input": data,
                        "mask": mask,
                    }
                    outputs = self.prefill_decoder_sessions[i].run(None, input_feed, shape_group=slice_index + 1)
                    k_caches[i][
                        :,
                        slice_index
                        * prefill_slice_len : slice_index
                        * prefill_slice_len + remain_len,
                        :,
                    ] = outputs[0][:, :remain_len, :]
                    v_caches[i][
                        :,
                        slice_index
                        * prefill_slice_len : slice_index
                        * prefill_slice_len + remain_len,
                        :,
                    ] = outputs[1][:, :remain_len, :]
                    data = outputs[2]    
                    
            post_out = self.post_process_session.run(
                None,
                {
                    "input": data[
                        :, token_len - (len(slice_indexs) - 1) * prefill_slice_len - 1, None, :
                    ]
                }
            )[0]                            
        
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            posibles = [tokenizer.decode([t]) for t in posssible_tokens]
            posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
            token_ids.append(next_token)
        
        # set to decoder
        token_ids_cached = []
        token_ids_cached.append(next_token)

        mask = np.zeros((1, 1, self.kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :self.kv_cache_len + 1] -= 65536
        if prefill_len > 0:
            mask[:, :, :token_len + 1] = 0
        
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
            if start_indice < token_len - 1:
                pass
            else:
                post_out = self.post_process_session.run(None, {"input": data})[0]
                next_token, posssible_tokens, possible_soft = post_process(post_out)
                token_ids.append(next_token)

                if next_token in terminators:
                    if len(token_ids_cached) > 0:
                        msg = tokenizer.decode(token_ids_cached)
                        token_ids_cached.clear()
                        if "\ufffd" in msg:
                            msg = msg.replace("\ufffd", "")
                        print(msg, end='\n<|im_end|>\n', flush=True)
                    return

                token_ids_cached.append(next_token)

                if len(token_ids_cached) >= 10:
                    msg = tokenizer.decode(token_ids_cached)
                    token_ids_cached.clear()
                    if "\ufffd" in msg:
                        msg = msg.replace("\ufffd", "")
                    print(msg, end=" ", flush=True)
        return                
        
        
if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniCPM-v4 axmodel demo")
    parser.add_argument("--hf_model_path", type=str, default="utils/minicpmv4_tokenizer",
                        help="Path to HuggingFace model")
    parser.add_argument("--siglip_axmodel", type=str, default="./utils/siglip.axmodel")
    parser.add_argument("--resampler_axmodel", type=str, default="./utils/resampler.axmodel")
    parser.add_argument("--embed_token_path", type=str, default="./utils/embed_tokens.pth")
    parser.add_argument("--minicpm_axmodel", type=str, default="./minicpm-v-4_axmodel")
    
    parser.add_argument("-i", "--image", type=str, default="./show_demo.jpg",
                        help="Path to the test image.")
    parser.add_argument("-q", "--question", type=str, default="What is the landform in the picture?",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()
    
    
    
    hf_model_path = args.hf_model_path
    img_path = args.image
    image = Image.open(img_path).convert('RGB').resize((448, 448))
    question = args.question

    msgs = [{'role': 'user', 'content': [image, question]}]
    
    resampler_axmodel = args.resampler_axmodel
    siglip_axmodel = args.siglip_axmodel
    embed_token_path = args.embed_token_path
    llm_axmodel_path = args.minicpm_axmodel
    
    processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)
    
    processor.image_processor.slice_mode = False # 不对图像做切分操作
    
    minicpm_axmodel = MiniCPMV(siglip_axmodel, resampler_axmodel, embed_token_path, llm_axmodel_path, config)
    msgs_list = [msgs]

    prompts_lists = []
    input_images_lists = []
    for msgs in msgs_list:
        copy_msgs = deepcopy(msgs)
        images = []
        for i, msg in enumerate(copy_msgs):
            role = msg["role"]
            content = msg["content"]
            assert role in ["user", "assistant"]
            if i == 0:
                assert role == "user", "The role of first msg should be user"
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

        prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
        input_images_lists.append(images)
    
    inputs = processor(
            prompts_lists, 
            input_images_lists, 
            max_slice_nums=None,
            use_image_id=None,
            return_tensors="pt", 
            max_length=32768
    )
    generation_config = {
        "top_p": 0.8,
        "top_k": 100,
        "temperature": 0.7,
        "do_sample": True,
        "repetition_penalty": 1.05
    }
    inputs.pop("image_sizes")
    
    model_inputs = {
        "input_ids": inputs.input_ids,
        "image_bound": inputs.image_bound,
    }
    model_inputs["pixel_values"] = inputs.pixel_values
    model_inputs['tgt_sizes'] = inputs.tgt_sizes
    
    model_inputs["inputs_embeds"], vision_hidden_states = minicpm_axmodel.get_vllm_embedding(model_inputs)

    del minicpm_axmodel.vpm, minicpm_axmodel.resampler, vision_hidden_states
    gc.collect()
    
    result = minicpm_axmodel._decode(model_inputs["inputs_embeds"].detach().numpy(), tokenizer, inputs.attention_mask, decode_text=True)

