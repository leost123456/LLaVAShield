import os
import time
import random
import openai
from openai import APIConnectionError

from ...config import Config
from .base_target import TargetModel

from ...utils.utils import to_data_url
from ...utils.logger import get_logger

logger = get_logger(__name__)  

class Qwen2_5VLInstructTarget(TargetModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),

        )

    def respond(self, task_id, next_question, image_ids=None, generated_image_ids=None, history=None):
        local_history = list(history or [])
        system_prompt = "You are a helpful assistant."
        
        # Load image content
        image_content = self._load_images(task_id, image_ids, image_dir=Config.images_dir)
        image_content += self._load_images(task_id, generated_image_ids, image_dir=Config.images_output_dir)

        # Build user prompt
        user_content = self._build_user_content(next_question, image_content)

        # Build conversation messages
        messages = self._build_messages(system_prompt, user_content, local_history)

        # Call OpenAI API
        response_content = self._call_model(messages, task_id)

        return response_content, user_content
    
    def _load_images(self, task_id, image_ids, image_dir):
        tag = f"[Task {task_id}] "
        if not image_ids:
            return []

        image_content = []
        for image_id in image_ids:
            
            image_path = os.path.join(image_dir, str(task_id), image_id)

            if not (isinstance(image_id, str) and image_id.strip() and os.path.exists(image_path)):
                logger.warning("%sTarget skip missing image: %s", tag, image_path)
                continue

            image_content.append({"type": "image_url", "image_url":  {"url": to_data_url(image_path)}})

        return image_content

    def _build_user_content(self, text, images):
        return images + [{"type": "text", "text": text}]
    
    def _build_messages(self, system_prompt, user_content, history):
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _call_model(self, messages, task_id):
        tag = f"[Task {task_id}] "
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="qwen2.5-vl-72b-instruct",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                response_content = response.choices[0].message.content

                return response_content
            
            except APIConnectionError as e:
                wait = 2 ** (attempt+1) + random.random()
                logger.error("%sTarget APIConnectionError, retry %d in %.1fs: %s", tag, attempt+1, wait, e)
                time.sleep(wait)
                
            except Exception as e:
                logger.error("%sTarget Exception, retry %d: %s", tag, attempt+1, e)

        return "I'm sorry, I can't assist with that."
    
    def update_history(self, user_content, response_content, history):
        updated_history = list(history)
        updated_history.append({"role": "user", "content": user_content})
        updated_history.append({"role": "assistant", "content": response_content})

        return updated_history
