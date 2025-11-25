# app/aac_graph/llm_clients.py
import os
from typing import Any
from openai import OpenAI
import google.generativeai as genai

# ========= OpenAI GPT (ChatGPT) =========

class GPTClient:
    def __init__(self, api_key: str, default_model: str):
        self.client = OpenAI(api_key=api_key)
        self.default_model = default_model

    def chat(self, prompt: str, model: str | None = None, temperature: float = 0.4) -> str:
        m = model or self.default_model
        resp = self.client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

# ========= Google Gemini =========

class GeminiClient:
    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text.strip()

# ========= 환경변수에서 설정 읽기 =========

GPT_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1-2025-04-14")
GPT_INTENT_MODEL = os.getenv("GPT_INTENT_MODEL", "gpt-4.1-mini-2025-04-14i")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# 기본 GPT 클라이언트 (문장 생성/리파인용)
gpt_client = GPTClient(api_key=GPT_API_KEY, default_model=GPT_MODEL)

# 의도 분류 전용 helper
def gpt_intent_chat(prompt: str) -> str:
    """
    Intent classifier용 GPT 호출. 모델은 GPT_INTENT_MODEL 사용.
    """
    return gpt_client.chat(prompt, model=GPT_INTENT_MODEL, temperature=0.1)

# Gemini 클라이언트 (룰 체크용)
gemini_client = GeminiClient(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
