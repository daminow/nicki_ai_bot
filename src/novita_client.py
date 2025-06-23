from openai import OpenAI
from .config import settings


class NovitaClient:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.bot_base_url, api_key=settings.novita_api_key
        )
        self.model = settings.novita_model

    async def chat(self, messages: list[dict], stream: bool = False) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            max_tokens=2048,
            temperature=0.7,
            top_p=1,
            extra_body={"top_k": 40, "min_p": 0},
        )
        # Stream disabled in current implementation – collect full text
        if stream:
            out = ""
            async for chunk in response:
                out += chunk.choices[0].delta.content or ""
            return out
        return response.choices[0].message.content