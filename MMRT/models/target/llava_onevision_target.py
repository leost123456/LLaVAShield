import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from config import Config
from models.target.base_target import TargetModel

from utils.utils import check_response_refused, to_data_url
from utils.logger import get_logger

logger = get_logger(__name__)  

class LlavaOnevisionTarget(TargetModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
                model_path,
                min_pixels=256*28*28,  
                max_pixels=768*28*28
        )
        
    def respond(self, task_id, next_question, image_ids=None, generated_image_ids=None, history=None):
        local_history = list(history or [])
        system_prompt = "You are a helpful assistant."
        
        # Load image content
        image_content = self._load_images(image_ids, task_id, image_dir=Config.images_dir)
        image_content += self._load_images(generated_image_ids, task_id, image_dir=Config.images_output_dir)

        # Build user prompt
        user_content = self._build_user_content(next_question, image_content)

        # Build conversation messages
        messages = self._build_messages(system_prompt, user_content, local_history)

        # Call model
        response_content = self._call_model(messages, image_content, task_id)

        return response_content, user_content
    
    def _load_images(self, image_ids, task_id, image_dir):
        tag = f"[Task {task_id}] "
        if not image_ids:
            return []

        image_content = []
        for image_id in image_ids:

            image_path = os.path.join(image_dir, str(task_id), image_id)
            if not os.path.exists(image_path):
                logger.warning("%sTarget skip missing image: %s", tag, image_path)
                continue
            
            image_content.append({"type": "image", "image": image_path})

        return image_content

    def _build_user_content(self, text, images):
        """
        构建用户输入内容
        """

        return images + [{"type": "text", "text": text}]

    
    def _build_messages(self, system_prompt, user_content, history):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_model(self, messages, image_content, task_id):
        """
        Call Llava to generate response
        """
        tag = f"[Task {task_id}] "
        for attempt in range(self.max_retries):
            try:
                # Model input preprocessing
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                images = []
                for msg in messages:
                    if msg["role"] == "user":
                        for content in msg["content"]:
                            if content["type"] == "image":
                                image_file = content["image"]
                                image = Image.open(image_file).convert("RGB")
                                images.append(image)
                if images:
                    inputs = self.processor(
                        text=[text],
                        images=images,
                        padding=True,
                        return_tensors="pt"
                    ).to(0, torch.float16)
                
                else:
                    inputs = self.processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt"
                    ).to(0, torch.float16)

                # Model inference
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.config.max_tokens, 
                        temperature=self.config.temperature,
                    )
                    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
                    response_content = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
                
                return response_content
            
            except Exception as e:
                logger.error("%sTarget Exception, retry %d: %s", tag, attempt+1, e)

        return "I'm sorry, I can't assist with that."

    def update_history(self, user_content, response_content, history):
        """
        更新target端对话历史信息
        """
        updated_history = list(history)
        updated_history.append({"role": "user", "content": user_content})
        updated_history.append({"role": "assistant", "content": response_content})
        
        return updated_history
