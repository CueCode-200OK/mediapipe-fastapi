# app/aac_graph/nodes.py
from typing import List, Any
import json
import logging

from .state import GraphState, RuleStatus
from .llm_clients import gpt_client, gemini_client, gpt_intent_chat
from app.redis_client import redis_client, lrange  # 너가 쓰는 redis 래퍼라고 가정
from redis.exceptions import ResponseError

logger = logging.getLogger(__name__)


def _append_debug(state: GraphState, step: str, info: dict) -> GraphState:
    """
    각 노드에서 호출해서 debug_trace 리스트에 기록을 추가한다.
    """
    trace = list(state.get("debug_trace", []))  # type: ignore
    trace.append({"step": step, **info})
    new_state: GraphState = {**state, "debug_trace": trace}
    return new_state


# # 3-1. load_recent_phrases
# def load_recent_phrases(state: GraphState) -> GraphState:
#     user_id = state["user_id"]
#     key = f"phrases:{user_id}"
#
#     # 1) 안전한 wrapper로 LIST 시도 (나중에 구조 바꿀 때 대비)
#     recent_phrases: List[str] = lrange(key, -10, -1) or []
#
#     # 2) LIST가 비어 있으면 JSON string fallback (지금 우리가 쓰는 형태)
#     if not recent_phrases:
#         try:
#             raw_json = redis_client.get(key)
#             if raw_json:
#                 data = json.loads(raw_json)
#                 if isinstance(data, list):
#                     recent_phrases = [str(x) for x in data]
#         except Exception as e:
#             logger.warning(
#                 "[load_recent_phrases] JSON fallback parse failed for key=%s: %s",
#                 key,
#                 e,
#             )
#
#     logger.info(
#         "[load_recent_phrases] user_id=%s key=%s phrases=%s",
#         user_id,
#         key,
#         recent_phrases,
#     )
#
#     new_state: GraphState = {**state, "raw_phrases": recent_phrases}
#     return _append_debug(new_state, "load_recent_phrases", {
#         "user_id": user_id,
#         "redis_key": key,
#         "raw_phrases": recent_phrases,
#         "count": len(recent_phrases),
#     })
# 3-1. load_recent_phrases (consume queue)
def load_recent_phrases(state: GraphState) -> GraphState:
    user_id = state["user_id"]
    key = f"phrases:{user_id}"

    consumed: List[str] = []
    used_list_mode = False

    # --- 1) LIST 기반 큐에서 LPOP으로 소비 시도 ---
    MAX_BATCH = 200  # 한 번의 문장 생성에서 처리할 최대 개수

    try:
        for _ in range(MAX_BATCH):
            item = redis_client.lpop(key)  # LIST가 아니면 여기서 WRONGTYPE 터짐
            if item is None:
                break
            if isinstance(item, bytes):
                try:
                    item = item.decode("utf-8")
                except Exception:
                    item = item.decode("utf-8", errors="ignore")
            consumed.append(str(item))

        used_list_mode = True

    except ResponseError as e:
        # phrases:{user} 가 LIST가 아니라 (예전 JSON string 등) 인 경우
        logger.warning(
            "[load_recent_phrases] LPOP failed for key=%s, fallback to JSON. err=%s",
            key,
            e,
        )

    # --- 2) LIST에서 아무것도 못 가져왔으면 JSON 문자열 fallback ---
    if not consumed:
        try:
            raw_json = redis_client.get(key)
            if raw_json:
                if isinstance(raw_json, bytes):
                    raw_json = raw_json.decode("utf-8", errors="ignore")
                data = json.loads(raw_json)
                if isinstance(data, list):
                    consumed = [str(x) for x in data]

                # JSON 모드에서도 한 번 읽었으면 지워줌 (consume 의미 맞추기)
                redis_client.delete(key)
        except Exception as e:
            logger.warning(
                "[load_recent_phrases] JSON fallback parse failed for key=%s: %s",
                key,
                e,
            )

    logger.info(
        "[load_recent_phrases] user_id=%s key=%s consumed_phrases=%s",
        user_id,
        key,
        consumed,
    )

    new_state: GraphState = {**state, "raw_phrases": consumed}
    return _append_debug(
        new_state,
        "load_recent_phrases",
        {
            "user_id": user_id,
            "redis_key": key,
            "raw_phrases": consumed,
            "count": len(consumed),
            "used_list_mode": used_list_mode,
        },
    )

# 3-2. normalize_phrases
def normalize_phrases(state: GraphState) -> GraphState:
    raw = state.get("raw_phrases", [])

    normalized: List[str] = []
    for token in raw:
        if not normalized or normalized[-1] != token:
            normalized.append(token)

    logger.info(
        "[normalize_phrases] raw=%s normalized=%s",
        raw,
        normalized,
    )

    new_state: GraphState = {**state, "normalized_phrases": normalized}
    return _append_debug(new_state, "normalize_phrases", {
        "raw_phrases": raw,
        "normalized_phrases": normalized,
    })


# 3-3. intent_classifier (GPT mini)
def intent_classifier(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])

    if not phrases:
        # 아예 LLM 안 부르고 바로 OTHER로
        intent = "OTHER"
        logger.info("[intent_classifier] empty phrases -> intent=OTHER (skip LLM)")
        new_state: GraphState = {
            **state,
            "intent": intent,
            "is_emergency": False,
        }
        return _append_debug(new_state, "intent_classifier", {
            "phrases": phrases,
            "skipped_llm": True,
            "final_intent": intent,
            "is_emergency": False,
        })

    prompt = f"""
    너는 AAC 제스처 문장 시스템의 의도 분류기야.
    아래 입력 토큰들을 보고 의도를 네 가지 중 하나로 분류해라.

    [라벨 정의]
    - EMERGENCY: 생명 위협, 급한 응급상황.
      예) "숨을 못 쉬어요", "심정지예요", "쓰러졌어요", "의식이 없어요", "피를 많이 토해요"
    - REQUEST: 뭔가를 요구/요청하는 표현.
      예) "물 주세요", "도와줘", "간호사 불러주세요"
    - STATUS: 몸 상태, 통증, 불편함을 설명하지만 즉시 죽을 것 같지는 않은 상황.
      예) "머리가 아파요", "배가 아파요", "피곤해요", "기침이 나와요"
    - OTHER: 잡담, 인사, 시스템 테스트 등 위 범주에 들어가지 않는 모든 것.

    [입력 토큰 리스트]
    {phrases}

    [출력 형식]
    반드시 다음 JSON 한 줄만 출력해라.
    예시: {{"intent": "EMERGENCY"}}

    조건:
    - intent 값은 EMERGENCY, REQUEST, STATUS, OTHER 중 하나여야 한다.
    - JSON 이외의 설명, 추가 문장은 절대 쓰지 마라.
    """

    resp_text = gpt_intent_chat(prompt)

    intent: str = "OTHER"
    parsed_val: Any = None

    try:
        data = json.loads(resp_text)
        val = str(data.get("intent", "")).upper()
        parsed_val = val
        if val in ["EMERGENCY", "REQUEST", "STATUS", "OTHER"]:
            intent = val
    except Exception:
        # 혹시 모델이 그냥 문자열만 반환했을 때를 대비
        val = resp_text.strip().upper()
        parsed_val = val
        if val in ["EMERGENCY", "REQUEST", "STATUS", "OTHER"]:
            intent = val

    logger.info(
        "[intent_classifier] phrases=%s raw_resp=%s intent=%s",
        phrases,
        resp_text,
        intent,
    )

    new_state: GraphState = {
        **state,
        "intent": intent,                     # "EMERGENCY"/"REQUEST"/...
        "is_emergency": (intent == "EMERGENCY"),
    }
    return _append_debug(new_state, "intent_classifier", {
        "phrases": phrases,
        "prompt_preview": prompt[:300],
        "raw_response": resp_text,
        "parsed_intent": parsed_val,
        "final_intent": intent,
        "is_emergency": intent == "EMERGENCY",
    })


# 3-4. emergency_generate (GPT – gpt-4.1)
def emergency_generate(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])
    prompt = f"""
    너는 응급 상황 AAC 문장을 만들어야 하는 한국어 도우미야.

    - 입력은 제스처로 인식된 토큰 리스트다.
    - 이 토큰을 참고해서 긴급 상황을 매우 짧고 분명하게 표현해라.
    - 최대 2문장, 30자 이내.
    - 존댓말을 사용해라.
    - 아래 예시 문장들의 톤과 길이를 따라라.

    [자주 사용하는 응급 문장 예시]
    - "의식이 없어요"
    - "쓰러졌어요"
    - "숨을 못 쉬어요"
    - "심정지예요"
    - "출혈이 심해요"
    - "경련을 해요"
    - "죽을 것 같아요"

    이런 느낌으로 아주 직관적인 긴급 문장을 만들어라.

    [입력 토큰 리스트]
    {phrases}

    [출력 형식]
    조건을 만족하는 한글 문장만 한 줄로 출력해라.
    부연 설명이나 따옴표는 쓰지 마라.
    """
    sentence = gpt_client.chat(prompt).strip()

    logger.info(
        "[emergency_generate] phrases=%s -> draft_sentence=%s",
        phrases,
        sentence,
    )

    new_state: GraphState = {**state, "draft_sentence": sentence}
    return _append_debug(new_state, "emergency_generate", {
        "phrases": phrases,
        "prompt_preview": prompt[:300],
        "draft_sentence": sentence,
    })


# 3-5. emergency_check (Gemini – gemini-2.5-flash)
def emergency_check(state: GraphState) -> GraphState:
    sentence = state.get("draft_sentence", "") or ""

    # 재시도 횟수 가져오기 (없으면 0)
    retry = int(state.get("emergency_retry", 0))
    MAX_RETRY = 2  # 필요하면 1~3 사이로 조절

    prompt = f"""
    너는 AAC용 긴급 문장을 검수하는 심사관이다.

    [검수 기준]
    1) 1~2문장인지
    2) 30자 이내인지
    3) 매우 직관적인 긴급 도움 요청인지
    4) 존댓말인지
    5) "죽을 것 같아요", "숨을 못 쉬어요"처럼 응급성이 분명해야 한다.

    [OK 예시]
    - "숨을 못 쉬어요"
    - "쓰러져서 의식이 없어요"
    - "피를 많이 토해요"
    - "심정지예요"

    [REWRITE 예시]
    - "조금 아픈 것 같아요"  (응급성이 약함)
    - "제가 어제부터 컨디션이 안 좋아서 그런데요..." (불필요하게 길고 복잡함)

    아래 중 하나만 출력해라 (큰따옴표 제외):
    OK
    REWRITE

    검수할 문장:
    {sentence}
    """

    status = gemini_client.chat(prompt).strip()

    # LLM이 실제로 판단한 값 (원본)
    raw_rule_status: RuleStatus = "OK" if status == "OK" else "REWRITE"

    # --- 재시도 로직 ---
    if raw_rule_status == "REWRITE" and retry < MAX_RETRY:
        # 아직 재시도 여유 있음 → 그래프에서 다시 emergency_generate 호출하도록 REWRITE 유지
        rule_status: RuleStatus = "REWRITE"
        final_sentence = state.get("final_sentence", "")
        next_retry = retry + 1
    else:
        # (1) OK 를 받았거나
        # (2) REWRITE지만 MAX_RETRY를 넘었으면 → 그냥 OK로 보고 종료
        rule_status = "OK"
        final_sentence = sentence
        next_retry = retry  # 더 증가시키지 않음

    logger.info(
        "[emergency_check] sentence=%s status=%s raw_rule_status=%s rule_status=%s retry=%s",
        sentence,
        status,
        raw_rule_status,
        rule_status,
        retry,
    )

    new_state: GraphState = {
        **state,
        "rule_status": rule_status,
        "final_sentence": final_sentence,
        "emergency_retry": next_retry,
    }
    return _append_debug(new_state, "emergency_check", {
        "checked_sentence": sentence,
        "raw_response": status,
        "raw_rule_status": raw_rule_status,
        "rule_status": rule_status,
        "final_sentence": final_sentence,
        "retry": next_retry,
    })

# 3-6. normal_generate (GPT – gpt-4.1)
def normal_generate(state: GraphState) -> GraphState:
    phrases = state.get("normalized_phrases", [])
    prompt = f"""
    너는 AAC 사용자를 도와주는 일상 문장 생성기다.

    [역할]
    - 입력: 제스처로 인식된 토큰 리스트
    - 출력: 일상적인 상황에서 사용할 짧은 문장 1개
    - 존댓말, 1문장, 30자 이내
    - 의미는 최대한 유지하되, 너무 복잡한 표현은 쓰지 마라.

    [좋은 예시]
    - "배가 아파요"
    - "머리가 아파요"
    - "물 좀 주세요"
    - "화장실에 가고 싶어요"
    - "기침이 나와요"
    - "피곤해서 쉬고 싶어요"

    이런 스타일로 간단하고 공손한 문장을 만들어라.

    [입력 토큰 리스트]
    {phrases}

    [출력 형식]
    조건을 만족하는 한글 문장만 한 줄로 출력해라.
    """
    sentence = gpt_client.chat(prompt).strip()

    logger.info(
        "[normal_generate] phrases=%s -> draft_sentence=%s",
        phrases,
        sentence,
    )

    new_state: GraphState = {**state, "draft_sentence": sentence}
    return _append_debug(new_state, "normal_generate", {
        "phrases": phrases,
        "prompt_preview": prompt[:300],
        "draft_sentence": sentence,
    })


# 3-7. refine_sentence (GPT – gpt-4.1)
def refine_sentence(state: GraphState) -> GraphState:
    draft = state.get("draft_sentence", "") or ""
    prompt = f"""
    너는 AAC 문장을 다듬는 보정기다.

    [목표]
    - 더 공손하게
    - 더 단순하게
    - 1문장, 30자 이내

    [수정 예시]
    - 입력: "배가 너무 아파 죽겠어요"
      출력: "배가 너무 아파요"
    - 입력: "좀 도와주실 수 있나요?"
      출력: "도와주세요"
    - 입력: "제 상태가 별로 좋은 것 같지가 않아요"
      출력: "몸이 안 좋아요"

    이제 아래 문장을 위 기준에 맞게 고쳐라.

    입력: {draft}

    [출력 형식]
    조건을 만족하는 한글 문장만 한 줄로 출력해라.
    """
    refined = gpt_client.chat(prompt).strip()

    logger.info(
        "[refine_sentence] draft=%s -> refined=%s",
        draft,
        refined,
    )

    new_state: GraphState = {**state, "refined_sentence": refined}
    return _append_debug(new_state, "refine_sentence", {
        "draft_sentence": draft,
        "prompt_preview": prompt[:300],
        "refined_sentence": refined,
    })


# 3-8. normal_check (Gemini – gemini-2.5-flash)
def normal_check(state: GraphState) -> GraphState:
    # refined가 있으면 그걸 우선 사용, 없으면 draft 사용
    sentence = state.get("refined_sentence") or state.get("draft_sentence", "") or ""
    prompt = f"""
    너는 AAC 일상 문장을 검수하는 심사관이다.

    [검수 기준]
    - 기본 규칙:
      1) 1~2문장
      2) 40자 이하
      3) 공손한 존댓말
      4) 의미가 명확하고 단순함

    [태그 정의]
    - OK: 위 기준을 모두 만족.
    - TOO_LONG: 문장이 40자를 넘거나, 불필요한 설명이 길게 붙어 있음.
    - NOT_POLITE: 반말이거나, 너무 명령조/무례한 표현.
    - UNCLEAR: 문장이 모호해서 상황을 이해하기 어렵거나, 주어/행동이 애매함.

    [예시]
    - "물 좀 주세요" → OK
    - "제가 어제부터 여기저기 너무 많이 아파가지고요" → TOO_LONG
    - "물 줘" → NOT_POLITE
    - "그거 좀 그렇게 해줘요" → UNCLEAR (무엇을 말하는지 모호함)

    아래 중 하나만 출력해라 (큰따옴표 쓰지 말 것):
    OK
    TOO_LONG
    NOT_POLITE
    UNCLEAR

    검수할 문장:
    {sentence}
    """

    tag = gemini_client.chat(prompt).strip()

    rule_status: RuleStatus = "OK"
    if tag in ["TOO_LONG", "NOT_POLITE", "UNCLEAR"]:
        rule_status = tag  # type: ignore

    final_sentence = sentence if rule_status == "OK" else state.get("final_sentence", "")

    logger.info(
        "[normal_check] sentence=%s tag=%s rule_status=%s",
        sentence,
        tag,
        rule_status,
    )

    new_state: GraphState = {
        **state,
        "rule_status": rule_status,
        "final_sentence": final_sentence,
    }
    return _append_debug(new_state, "normal_check", {
        "checked_sentence": sentence,
        "raw_response": tag,
        "rule_status": rule_status,
        "final_sentence": final_sentence,
    })
