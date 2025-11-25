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
