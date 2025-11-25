# app/aac_graph/nodes.py
from typing import List
import json
from .state import GraphState, RuleStatus
from .llm_clients import gpt_client, gemini_client, gpt_intent_chat
from app.redis_client import redis_client  # 네가 쓰는 redis 래퍼라고 가정

# 3-1. load_recent_phrases
def load_recent_phrases(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    key = f"phrases:{user_id}"
    recent_phrases: List[str] = redis_client.lrange(key, -10, -1) or []
    return {**state, "raw_phrases": recent_phrases}

# 3-2. normalize_phrases
def normalize_phrases(state: GraphState) -> GraphState:
    raw = state.get("raw_phrases", [])

    normalized: List[str] = []
    for token in raw:
        if not normalized or normalized[-1] != token:
            normalized.append(token)

    return {**state, "normalized_phrases": normalized}

# 3-3. intent_classifier (GPT mini)
def intent_classifier(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])
    prompt = f"""
너는 AAC 제스처 문장 시스템의 의도 분류기야.
아래 입력을 보고 의도를 EMERGENCY, REQUEST, STATUS, OTHER 중 하나로만 결정해.

입력 토큰 리스트: {phrases}

출력 형식 (JSON 한 줄):
{{"intent": "EMERGENCY"}}
처럼 intent 키만 포함해서 출력해.
"""

    resp_text = gpt_intent_chat(prompt)

    # JSON 파싱 시도
    intent = "OTHER"
    try:
        data = json.loads(resp_text)
        val = str(data.get("intent", "")).upper()
        if val in ["EMERGENCY", "REQUEST", "STATUS", "OTHER"]:
            intent = val
    except Exception:
        # 혹시 모델이 그냥 문자열만 반환했을 때를 대비
        val = resp_text.strip().upper()
        if val in ["EMERGENCY", "REQUEST", "STATUS", "OTHER"]:
            intent = val

    return {
        **state,
        "intent": intent,
        "is_emergency": (intent == "EMERGENCY"),
    }

# 3-4. emergency_generate (GPT – gpt-4.1)
def emergency_generate(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])
    prompt = f"""
너는 응급 상황 AAC 문장을 만들어야 하는 한국어 도우미야.

- 입력은 제스처로 인식된 토큰 리스트다.
- 이 토큰을 참고해서 긴급 상황을 매우 짧고 분명하게 표현해라.
- 최대 2문장, 30자 이내.
- 존댓말을 사용해라.

입력 토큰: {phrases}

출력: 조건을 만족하는 한글 문장만 출력해.
"""
    sentence = gpt_client.chat(prompt).strip()
    return {**state, "draft_sentence": sentence}

# 3-5. emergency_check (Gemini – gemini-2.5-flash)
def emergency_check(state: GraphState) -> GraphState:
    sentence = state.get("draft_sentence", "")
    prompt = f"""
너는 AAC용 긴급 문장을 검수하는 심사관이다.

검수 기준:
1) 1~2문장인지
2) 30자 이내인지
3) 매우 직관적인 긴급 도움 요청인지
4) 존댓말인지

아래 중 하나만 출력해라:
- "OK"
- "REWRITE"

검수할 문장: "{sentence}"
"""
    status = gemini_client.chat(prompt).strip()
    rule_status: RuleStatus = "OK" if status == "OK" else "REWRITE"
    return {**state, "rule_status": rule_status}

# 3-6. normal_generate (GPT – gpt-4.1)
def normal_generate(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])
    prompt = f"""
너는 AAC 사용자를 도와주는 문장 생성기다.

- 입력: 제스처로 인식된 토큰 리스트
- 출력: 일상적인 상황에서 사용할 짧은 문장 1개
- 존댓말, 1문장, 30자 이내
- 의미는 최대한 유지하되, 너무 복잡한 표현은 쓰지 마라.

입력 토큰: {phrases}

출력: 조건을 만족하는 한글 문장만 출력해.
"""
    sentence = gpt_client.chat(prompt).strip()
    return {**state, "draft_sentence": sentence}

# 3-7. refine_sentence (GPT – gpt-4.1)
def refine_sentence(state: GraphState) -> GraphState:
    draft = state.get("draft_sentence", "")
    prompt = f"""
너는 AAC 문장을 다듬는 보정기다.

아래 문장을:
- 더 공손하게
- 더 단순하게
- 1문장, 30자 이내
로 바꿔라.

입력: "{draft}"

출력: 조건을 만족하는 한글 문장만 출력해.
"""
    refined = gpt_client.chat(prompt).strip()
    return {**state, "refined_sentence": refined}

# 3-8. normal_check (Gemini – gemini-2.5-flash)
def normal_check(state: GraphState) -> GraphState:
    sentence = state.get("refined_sentence") or state.get("draft_sentence", "")
    prompt = f"""
너는 AAC 일상 문장을 검수하는 심사관이다.

기준:
1) 1~2문장
2) 40자 이하
3) 공손한 존댓말
4) 의미가 명확하고 단순함

아래 중 하나만 출력해라:
- "OK"
- "TOO_LONG"
- "NOT_POLITE"
- "UNCLEAR"

검수할 문장: "{sentence}"
"""
    tag = gemini_client.chat(prompt).strip()
    rule_status: RuleStatus = "OK"
    if tag in ["TOO_LONG", "NOT_POLITE", "UNCLEAR"]:
        rule_status = tag  # type: ignore

    return {
        **state,
        "rule_status": rule_status,
        "final_sentence": sentence if rule_status == "OK" else state.get("final_sentence", ""),
    }
