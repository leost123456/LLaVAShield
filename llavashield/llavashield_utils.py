import argparse
import torch
import os
import io
import json
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_models
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token

from utils.prompt_template import llava_prefix_prompt, llava_task_prompt, llava_note_prompt, llava_input_parameter_prompt, llava_output_format_prompt, llava_dialogue_history_prompt
from utils.policy import policy_prompt_construct, usage_policy_prompt

class LlavaShieldProcessor:
    """
    Processes input data, converting messages and images into model-required input_ids and image_tensors.
    """
    def __init__(self, tokenizer, image_processor, usage_policy=None):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.usage_policy = usage_policy if usage_policy is not None else ['Violence & Harm', 'Hate & Harassment', 'Sexual Content', 'Self-Harm & Suicide', 'Illegal Activities', 'Deception & Misinformation', 'Privacy Violation', 'Malicious Disruption']

    def get_conversations(self, messages, usage_policy):
        conv_history = []
        image_list = []
        cum_index = 1
        
        for message in messages:
            if message['role'] == 'user':
                content_list = message['content']
                cur_image_list = []
                text_list = []
                for content in content_list:
                    if content['type'] == 'image':
                        cur_image_list.append(content['image'])
                    elif content['type'] == 'text':
                        text_list.append(content['text'])
                
                # Image1: <image>...
                image_prefix = ' '.join([f'Image{cum_index+index}: <image>;' if index != len(cur_image_list)-1 else f'Image{cum_index+index}: <image>.' for index in range(len(cur_image_list))])
                text = '\n'.join(text_list)
                conv_history.append({"role": "user", "content": image_prefix + ' ' + text})

                cum_index += len(cur_image_list)
                image_list.extend(cur_image_list) 

            elif message['role'] == 'assistant':
                conv_history.append({"role": "assistant", "content": message['content']})

        llava_prompt_template = llava_prefix_prompt + '\n' + llava_task_prompt + '\n' + llava_note_prompt + '\n' + f'**Policy Dimensions**:\n{policy_prompt_construct(usage_policy)}\n' + llava_input_parameter_prompt  + '\n' + llava_output_format_prompt + '\n' + llava_dialogue_history_prompt + '\n'
        prompt = llava_prompt_template.format(
            usage_policy=usage_policy_prompt(usage_policy=usage_policy),
            conversations=json.dumps(conv_history, ensure_ascii=False)
        )

        llava_conversations = [
            {
                "from": "human",
                "value": prompt
            }
        ]
        return llava_conversations, image_list

    def __call__(self, messages, usage_policy=None, device="cuda"):
        policy = usage_policy if usage_policy is not None else self.usage_policy
        if policy is None:
            raise ValueError("usage_policy must be provided either in initialization or processor call.")

        conversations, image_files = self.get_conversations(messages, policy)
        qs = conversations[0]["value"]

        conv_mode = "qwen_1_5"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        image_tensors = []
        for image_file in image_files:
            if isinstance(image_file, str):
                image = Image.open(image_file)
            elif isinstance(image_file, Image.Image):
                image = image_file
            elif isinstance(image_file, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image_file))
            else:
                raise ValueError(f"Unsupported image type: {type(image_file)}")
                
            image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().to(device))
            
        return {"input_ids": input_ids, "image_tensors": image_tensors}

class LlavaShieldModel:
    """
    Model class responsible for inference with processed tensors.
    """
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.to(self.device)

    @torch.inference_mode()
    def generate(self, input_ids, image_tensors, max_new_tokens=1024,do_sample=True, temperature=0.7, top_p=1.0, num_beams=1):
        outputs = self.model.generate(
            input_ids.to(self.device),
            images=image_tensors,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )

        return outputs

def load_shield(model_path, usage_policy=None, device=None, dtype='float16', device_map='auto', attn_implementation="flash_attention_2"):
    """
    Factory function: Loads model weights and returns processor and model for clean usage.
    """
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_models(model_path, device_map=device_map, torch_dtype=dtype, attn_implementation=attn_implementation)

    processor = LlavaShieldProcessor(tokenizer, image_processor, usage_policy=usage_policy)
    shield_model = LlavaShieldModel(model, device=device)

    return processor, shield_model
