import re
import os
import torch
from transformers import AutoModel, AutoTokenizer

from config import Config
from models.target.base_target import TargetModel
from utils.intern_vl_utils import load_image

from utils.logger import get_logger
logger = get_logger(__name__)  

class InternVL3Target(TargetModel):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map="auto"
            ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.generation_config = dict(max_new_tokens=self.config.max_tokens, temperature=self.config.temperature)

    def respond(self, task_id, next_question, image_ids=None, generated_image_ids=None, history=None):
        local_history = list(history or [])

        # Load image content
        image_content_1, pixel_values_list_1, num_patches_list_1 = self._load_images(image_ids, task_id, image_dir=Config.images_dir)
        image_content_2, pixel_values_list_2, num_patches_list_2 = self._load_images(generated_image_ids, task_id, image_dir=Config.images_output_dir)
        
        image_content = (image_content_1 or "") + (image_content_2 or "")
        
        # Merge pixel values and patch counts
        pixel_values_list = (pixel_values_list_1 or []) + (pixel_values_list_2 or [])
        num_patches_list  = (num_patches_list_1 or []) + (num_patches_list_2 or [])

        if pixel_values_list:
            pixel_values = torch.cat(pixel_values_list, dim=0)
        else:
            pixel_values = None
            num_patches_list = None  # Pass None when there are no images

        # Build user prompt
        user_content = self._build_user_content(next_question, image_content)

        # Call model
        response_content, history = self._call_model(user_content, pixel_values, num_patches_list, local_history, task_id)

        return response_content, user_content, history
    
    def _load_images(self, image_ids, task_id,  image_dir):
        tag = f"[Task {task_id}] "
        if not image_ids:
            return "", None, None

        image_content = ""
        pixel_values_list = []
        num_patches_list = []
        for image_id in image_ids:

            image_path = os.path.join(image_dir, str(task_id), image_id)
            if not os.path.exists(image_path):
                logger.warning("%sTarget skip missing image: %s", tag, image_path)
                continue

            pv = load_image(image_path, max_num=12).to(torch.bfloat16).to(torch.bfloat16).cuda()

            image_content += f"Image-{image_ids}: <image>\n"
            pixel_values_list.append(pv)
            num_patches_list.append(pv.size(0))

        return image_content, pixel_values_list, num_patches_list

    def _build_user_content(self, text, images):
        return images + text

    def _call_model(self, user_content, pixel_values, num_patches_list, history, task_id):
        tag = f"[Task {task_id}] "
        for attempt in range(self.max_retries):
            try:
                # Model input preprocessing
                response, new_history = self.model.chat(self.tokenizer, 
                                            pixel_values, 
                                            user_content, 
                                            self.generation_config,
                                            num_patches_list=num_patches_list,
                                            history=history,
                                            return_history=True)
            
                return response, new_history

            except Exception as e:
                logger.error("%sTarget Exception, retry %d: %s", tag, attempt+1, e)

        return "I'm sorry, I can't assist with that.", []

    def _update_history(self, history):
        self.history = history
