# app/aac_graph/routes.py
from .state import GraphState

def route_intent(state: GraphState) -> str:
    if state.get("is_emergency"):
        return "emergency_generate"
    else:
        return "normal_generate"

def route_emergency_check(state: GraphState) -> str:
    if state.get("rule_status") == "OK":
        return "finish"
    else:
        return "emergency_generate"

def route_normal_check(state: GraphState) -> str:
    if state.get("rule_status") == "OK":
        return "finish"
    else:
        return "refine_sentence"


def route_after_normalize(state: GraphState) -> str:
    phrases = state.get("normalized_phrases") or []
    # 아무 단어도 없으면 그래프 종료
    if len(phrases) == 0:
        return "finish"
    # 단어가 있으면 기존 플로우 계속
    return "continue"
