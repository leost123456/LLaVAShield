import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN,IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init,rank0_print
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, PretrainedConfig
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

#修正config文件，删除一些不必要的字段(transformers4.45及以上版本保存模型config会自动加入text_config和vision_config，会导致后续模型推理出现问题，这样操作保证和原版llava-ov的config文件保持一致，支持源码推理)
def config_revise(output_dir):
    config=AutoConfig.from_pretrained(output_dir,trust_remote_code=True).to_dict()
    config.pop("text_config",None)
    config.pop("vision_config",None)
    config.pop("image_seq_length",None)
    with open(os.path.join(output_dir,'config.json'),'w') as f:
        json.dump(config,f,ensure_ascii=False, indent=2)
                    
#模型导入（注意下面是专门为qwen写的一个模型导入和lora，原版的load_pretrained_model是没有专门qwen的lora的）
def load_llava_qwen_models(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map
    #lora操作
    if "lora" in model_name.lower() and model_base is not None:
        rank0_print("Loading LLaVA from base model...")
        from llava.model.language_model.llava_qwen import LlavaQwenConfig
        #lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # if isinstance(getattr(lora_cfg_pretrained, "text_config", None), dict):
        #      lora_cfg_pretrained.text_config = PretrainedConfig.from_dict(lora_cfg_pretrained.text_config)
        # if isinstance(getattr(lora_cfg_pretrained, "vision_config", None), dict):
        #      lora_cfg_pretrained.vision_config = PretrainedConfig.from_dict(lora_cfg_pretrained.vision_config)
        model = LlavaQwenForCausalLM.from_pretrained(
            model_base,low_cpu_mem_usage=False,
            config=lora_cfg_pretrained,
            attn_implementation=attn_implementation, **kwargs
        )
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        rank0_print("Loading additional LLaVA weights...")

        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        rank0_print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        rank0_print("Merging LoRA weights...")
        model = model.merge_and_unload()
        rank0_print("Model is loaded...")

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device="cuda", dtype=kwargs['torch_dtype']) #使用torch.float16精度类型
    image_processor = vision_tower.image_processor
    
    # 4) 参数设置
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len

def merge_lora(model_path=None,model_base=None,save_model_path=None):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_llava_qwen_models(model_path, model_base, model_name, device_map=None)

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    
    #进行config revise
    config_revise(save_model_path)

if __name__ == "__main__":
    model_path='/data3/guolei.huang/multimodal-dialogue-safety-detection/LLaVA-NeXT/checkpoints/test1/llava-shield-qwen2-0.5b-ov-lora-default_shuffle_mask_mmdu-test1_2'#lora权重位置
    model_base="/data3/guolei.huang/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-0.5b-ov/snapshots/381d9947148efb1e58a577f451c05705ceec666e" #原始模型权重位置
    save_model_path='/data3/guolei.huang/multimodal-dialogue-safety-detection/LLaVA-NeXT/checkpoints/test1/merged_llava-shield-qwen2-0.5b-ov-lora-default_shuffle_mask_mmdu-test1_2' #合并后的模型权重保存位置
    merge_lora(model_path=model_path,model_base=model_base,save_model_path=save_model_path)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str,default='/data3/guolei.huang/multimodal-dialogue-safety-detection/LLaVA-NeXT/checkpoints/test1/llava-shield-qwen2-0.5b-ov-lora-default_shuffle_mask_mmdu-test1_1') #lora权重位置
    # parser.add_argument("--model-base", type=str,default="/data3/guolei.huang/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-0.5b-ov/snapshots/381d9947148efb1e58a577f451c05705ceec666e") #原始模型权重位置
    # parser.add_argument("--save-model-path", type=str,default='/data3/guolei.huang/multimodal-dialogue-safety-detection/LLaVA-NeXT/checkpoints/test1/merged_llava-shield-qwen2-0.5b-ov-lora-default_shuffle_mask_mmdu-test1_1') #合并后的模型权重保存位置

    # args = parser.parse_args()

#MODEL BASE
#LLAVA-OV-0.5B="/data3/guolei.huang/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-0.5b-ov/snapshots/381d9947148efb1e58a577f451c05705ceec666e"
#LLAVA-OV-7B="/data3/guolei.huang/huggingface/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov/snapshots/0b07bf7565e244cf4f39982249eafe8cd799d6dd"