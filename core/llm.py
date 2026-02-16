import os
import json
import re
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

T = TypeVar("T", bound=BaseModel)

class LLMService:
    def __init__(self, base_url: str = None, api_key: str = None, model: str = "deepseek-chat", timeout: float = 60.0):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=self.api_key,
            timeout=timeout
        )
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_structured(self, system_prompt: str, user_prompt: str, response_model: Type[T], temperature: float = 0.0) -> tuple[T, dict]:
        """
        Generates a response... returns (model_instance, usage_dict)
        """
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        
        # Enhanced system prompt
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "You MUST respond with valid JSON strictly conforming to this schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Do not include any text outside the JSON object."
        )

        try:
            start_t = asyncio.get_event_loop().time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=temperature
            )
            latency = (asyncio.get_event_loop().time() - start_t) * 1000
            
            if not response or not hasattr(response, 'choices') or not response.choices:
                raise ValueError(f"Invalid LLM response: {response}")

            content = response.choices[0].message.content
            
            # Extract Usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "latency_ms": latency,
                "model": self.model
            }

            return self._parse_json(content, response_model), usage
            
        except Exception as e:
            # Tenacity will handle specified exceptions, but we log others
            # print(f"LLM Error: {e}") 
            raise

    def _parse_json(self, content: str, model: Type[T]) -> T:
        try:
            # 1. Try direct parse
            data = json.loads(content)
            return model.model_validate(data)
        except json.JSONDecodeError:
            # 2. Try extracting from markdown ```json ... ```
            match = re.search(r"```json(.*?)```", content, re.DOTALL)
            if match:
                clean_content = match.group(1).strip()
                data = json.loads(clean_content)
                return model.model_validate(data)
            else:
                # 3. Last ditch: try to find start/end of json object
                try:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != -1:
                        data = json.loads(content[start:end])
                        return model.model_validate(data)
                except:
                    pass
                
                raise ValueError(f"Could not parse JSON from LLM response: {content[:100]}...")
