import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ..constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ..conversation import conv_templates, SeparatorStyle
from ..model.builder import load_pretrained_model
from ..utils import disable_torch_init
from ..mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from ..constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids #[L]

def eval_model(args):
    
    # Model （模型导入，包括tokenizer，model，image processor）
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path) #模型路径
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Data 数据集导入（问题集(json)和回答集）加载并切块问题集
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        idx = line["sample_id"]
        question_type = line["metadata"]["question_type"]
        dataset_name = line["metadata"]["dataset"]
        gt = line["conversations"][1]["value"] #label标签

        image_files = line["image"] #图像文件夹
        qs = line["conversations"][0]["value"]
        cur_prompt = args.extra_prompt + qs

        args.conv_mode = "qwen_1_5"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() # FIXME 这里为什么不直接用完整的prompt输入一个能够直接处理进行分词的功能函数得到input_ids呢？
        
        #构造得到输入的input_dis
        input_ids = preprocess_qwen([line["conversations"][0],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
        img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX) #统计IMAGE_TOKEN_INDEX的数量

        image_tensors = []
        #取出并处理每个样本的images
        for image_file in image_files:
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            #生成输出
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        #输出内容分词
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        #写入答案
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
                                   "dataset": dataset_name, #数据集名称
                                   "sample_id": idx, # id
                                   "prompt": cur_prompt, #输入prompt
                                   "pred_response": outputs, #输出
                                   "gt_response": gt, #真实回答（label标签）
                                   "shortuuid": ans_id,
                                   "model_id": model_name,
                                   "question_type": question_type,
                                   }) + "\n")
        ans_file.flush()

        if len(line["conversations"]) > 2: #如果是多轮对话

            for i in range(2, len(line["conversations"]), 2): #取每轮human部分输入
                input_ids = torch.cat((input_ids, output_ids), dim=1) #模型前一轮对话的问题和回答

                gt = line["conversations"][i + 1]["value"] #新一轮的真实回答labels
                qs = line["conversations"][i]["value"] #新一轮问题
                cur_prompt = args.extra_prompt + qs

                args.conv_mode = "qwen_1_5"

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids_new = preprocess_qwen([line["conversations"][i],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
                input_ids = torch.cat((input_ids, input_ids_new), dim=1)
                img_num = list(input_ids_new.squeeze()).count(IMAGE_TOKEN_INDEX)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        # no_repeat_ngram_size=3,
                        max_new_tokens=1024,
                        use_cache=True)
        
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()

                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({
                                        "dataset": dataset_name,
                                        "sample_id": idx,
                                        "prompt": cur_prompt,
                                        "pred_response": outputs,
                                        "gt_response": gt,
                                        "shortuuid": ans_id,
                                        "model_id": model_name,
                                        "question_type": question_type,
                                        }) + "\n")
                ans_file.flush()


    ans_file.close()

if __name__ == "__main__":
    #针对数据集进行evaluate（单/多图多轮对话,但是注意图像一直用的是第一轮输入的，后面的不能输入图像），数据集中包括多轮对话内容，会根据原始的对话问题模型进行回答，并将原始的gpt内容作为label同时进行记录。
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m") #模型权重与底座；
    parser.add_argument("--model-base", type=str, default=None) 
    parser.add_argument("--image-folder", type=str, default="") #图像目录
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl") #输入问题 JSONl 文件；
    parser.add_argument("--answers-file", type=str, default="answer.jsonl") #输出 JSONL文件
    parser.add_argument("--conv-mode", type=str, default="llava_v1") #对话模板（代码里强制改成 qwen_1_5 来取分隔符，外部传入值实际上没被使用）；
    parser.add_argument("--num-chunks", type=int, default=1) #默认是不切分
    parser.add_argument("--chunk-idx", type=int, default=0) #整块内容
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)