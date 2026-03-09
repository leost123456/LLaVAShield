import os
import re
import json
import time
import random
import httpx
import openai
from openai import APIConnectionError

from ...config import Config
from ...prompts.evaluate_prompt import EVALUATE_SYSTEM_PROMPT

from .base_evaluator import EvaluatorModel

from ...utils.utils import check_response_refused
from ...utils.logger import get_logger

logger = get_logger(__name__)  
def _normalize_history(history):
    turns = []

    for item in history or []:
        if isinstance(item, dict) and "role" in item and "content" in item:
            role = str(item["role"]).strip().lower()
            content = "" if item["content"] is None else str(item["content"])
            if role in ("assistant", "user"):
                turns.append({"role": role, "content": content})
            else:
                turns.append({"role": "assistant", "content": content})
            continue

        if isinstance(item, (tuple, list)) and len(item) == 2:
            u, a = item
            if u is not None and str(u).strip():
                turns.append({"role": "user", "content": str(u)})
            if a is not None and str(a).strip():
                turns.append({"role": "assistant", "content": str(a)})
            continue

        if isinstance(item, str):
            if item.strip():
                turns.append({"role": "assistant", "content": item})
            continue

    return turns

class GPT4oEvaluator(EvaluatorModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )


    def evaluate(self, task, history):
        normalized_history = _normalize_history(history)

        response_history = [t for t in normalized_history if t["role"] == "assistant"]

        system_prompt = EVALUATE_SYSTEM_PROMPT.format(
            task=task, 
            response_history=response_history
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":"Generate your evaluation."}
        ]
                
        output = self._call_model(messages)

        return output
    
    def _parse_json(self, text):
        output = ""
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            output = json.loads(json_str)
                
        return output
    
    def _call_model(self, messages):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                response_content = response.choices[0].message.content

                if not response_content or check_response_refused(response_content):
                    logger.warning("Evaluator refuse to respond. Retrying...")
                    continue
                
                try:
                    output = self._parse_json(response_content)
                    if output and output.get("score"):

                        return output  
                    
                except json.JSONDecodeError:
                    logger.error("Evaluator output invalid JSON format. Retrying...")
            
            except APIConnectionError as e:
                wait = 2 ** attempt + random.random()
                logger.warning("Evaluator APIConnectionError, retry %d in %.1fs: %s", attempt+1, wait, e)
                time.sleep(wait)
        
        return {"think": "", "score": 1}
                


