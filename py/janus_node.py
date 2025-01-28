import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, 'Janus'))
from PIL import Image
import torch
from docutils.nodes import target
from transformers import AutoModelForCausalLM
import random
import numpy as np
from .janus.models import MultiModalityCausalLM, VLChatProcessor
from .janus.models import MultiModalityCausalLM, VLChatProcessor
from .janus.utils.io import load_pil_images
import folder_paths
from comfy.utils import ProgressBar
from tqdm import tqdm

def pil2tensor(image:Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def check_and_download_model(model_path, repo_id):
    model_path = os.path.join(folder_paths.models_dir, model_path)

    if not os.path.exists(model_path):
        print(f"Downloading {repo_id} model...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=model_path, ignore_patterns=["*.md", "*.txt", ".git"])
    return model_path

class DZ_LoadJanusModel:
    def __init__(self):
        self.NODE_NAME = 'DZ_LoadJanusModel'
    @classmethod
    def INPUT_TYPES(self):
        model_list = ["Janus-Pro-7B", "Janus-Pro-1B"]
        return {
            "required": {
                "model": (model_list,),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("JanusModel",)
    RETURN_NAMES = ("janus_model",)
    FUNCTION = 'load_janus_model'
    CATEGORY = '😺dzNodes/Janus'

    def load_janus_model(self, model):
        model_path = os.path.join(folder_paths.models_dir, "Janus-Pro", model)
        check_and_download_model(model_path, f"deepseek-ai/{model}")
        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        return ({"vl_gpt": vl_gpt, "vl_chat_processor": vl_chat_processor},)


class DZ_JanusT2I:

    def __init__(self):
        self.NODE_NAME = 'JanusT2I'


    @classmethod
    def INPUT_TYPES(self):
        default_prompt = ("A massive blue whale soaring like a bird above a deep blue ocean, slicing through silky waves. "
                          "The sky is illed with golden and purple auroras, and thewhale's body glimmers with iridescent ights. "
                          "its tail fin skims the ocean surace,leaving a tral of glowing streams, "
                          "The scene exudes a futuristic and surrealvibe, with floating islands and glowing crystals in the background. "
                          "The overall composition is breathtaking and fantastical.")
        return {
            "required": {
                "janus_model": ("JanusModel",),
                "prompt": ("STRING",{"default": default_prompt, "multiline": True},),
                "size": ("INT", {"default": 384, "min": 384, "max": 384, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2, "step": 0.1}),
                "cfg": ("FLOAT", {"default": 5, "min": 0, "max": 32, "step": 0.1}),
                "token_num": ("INT", {"default": 576, "min": 576, "max": 576, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 1024, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'janus_t2i'
    CATEGORY = '😺dzNodes/Janus'

    def janus_t2i(self, janus_model, prompt, size, temperature, cfg, token_num, batch_size, seed):
        ret_images = []

        vl_chat_processor = janus_model["vl_chat_processor"]
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt = janus_model["vl_gpt"]

        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + vl_chat_processor.image_start_tag

        @torch.inference_mode()
        def generate(
                mmgpt: MultiModalityCausalLM,
                vl_chat_processor: VLChatProcessor,
                prompt: str,
                temperature: float = 1,
                parallel_size: int = 1,
                cfg_weight: float = 5,
                image_token_num_per_image: int = 576,
                img_size: int = 384,
                patch_size: int = 16,
                seed: int = 0,
        ):

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            input_ids = vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()

            for i in range(parallel_size * 2):
                tokens[i, :] = input_ids
                if i % 2 != 0:
                    tokens[i, 1:-1] = vl_chat_processor.pad_id

            inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

            generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

            comfy_pbar = ProgressBar(image_token_num_per_image)
            tqdm_pbar = tqdm(total=image_token_num_per_image, desc="Generating Images")
            for i in range(image_token_num_per_image):
                outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True,
                                                     past_key_values=outputs.past_key_values if i != 0 else None)
                hidden_states = outputs.last_hidden_state

                logits = mmgpt.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]

                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)

                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)
                comfy_pbar.update(1)
                tqdm_pbar.update(1)

            dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                     shape=[parallel_size, 8, img_size // patch_size,
                                                            img_size // patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

            dec = np.clip((dec + 1) / 2 * 255, 0, 255)

            visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec

            return visual_img

        visual_img = generate(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            prompt=prompt,
            temperature=temperature,
            parallel_size=batch_size,
            cfg_weight=cfg,
            image_token_num_per_image=token_num,
            img_size=size,
            seed=seed
        )

        for i in visual_img:
            img = Image.fromarray(i)
            ret_images.append(pil2tensor(img))

        return (torch.cat(ret_images, dim=0),)

class DZ_JanusI2T:

    def __init__(self):
        self.NODE_NAME = 'JanusI2T'


    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "janus_model": ("JanusModel",),
                "image": ("IMAGE",),
                "question": ("STRING",{"default": "describe this image", "multiline": True},),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2, "step": 0.1}),
                "max_new_tokens": ("INT", {"default": 512, "min": 8, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1e18, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'janus_i2t'
    CATEGORY = '😺dzNodes/Janus'
    OUTPUT_IS_LIST = (True,)

    def janus_i2t(self, janus_model, image, question, temperature, max_new_tokens, seed):

        ret_texts = []
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        vl_chat_processor = janus_model["vl_chat_processor"]
        tokenizer = vl_chat_processor.tokenizer
        vl_gpt = janus_model["vl_gpt"]

        ret_text = []

        for i in image:
            img = tensor2pil(i).convert("RGB")
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{question}",
                    "images": [img],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            prepare_inputs = vl_chat_processor(
                conversations=conversation, images=[img], force_batchify=True
            ).to(vl_gpt.device)

            # # run image encoder to get the image embeddings
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

            # # run the model to get the response
            outputs = vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            # print(f"{prepare_inputs['sft_format'][0]}", answer)
            ret_text.append(answer)

        return (ret_text,)


NODE_CLASS_MAPPINGS = {
    "JanusTextToImage": DZ_JanusT2I,
    "JanusImage2Text": DZ_JanusI2T,
    "LoadJanusModel": DZ_LoadJanusModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JanusTextToImage": "Janus Text To Image (Generation)",
    "JanusImage2Text": "Janus Image To Text (Understanding)",
    "LoadJanusModel": "Load Janus Model",
}