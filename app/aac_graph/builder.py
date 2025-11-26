# app/aac_graph/builder.py
from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    load_recent_phrases,
    normalize_phrases,
    intent_classifier,
    emergency_generate,
    emergency_check,
    normal_generate,
    refine_sentence,
    normal_check,
)
from .routes import route_intent, route_emergency_check, route_normal_check, route_after_normalize

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("load_recent_phrases", load_recent_phrases)
    builder.add_node("normalize_phrases", normalize_phrases)
    builder.add_node("intent_classifier", intent_classifier)

    builder.add_node("emergency_generate", emergency_generate)
    builder.add_node("emergency_check", emergency_check)

    builder.add_node("normal_generate", normal_generate)
    builder.add_node("refine_sentence", refine_sentence)
    builder.add_node("normal_check", normal_check)

    builder.set_entry_point("load_recent_phrases")

    # builder.add_edge("load_recent_phrases", "normalize_phrases")
    # builder.add_edge("normalize_phrases", "intent_classifier")

    builder.add_edge("load_recent_phrases", "normalize_phrases")

    builder.add_conditional_edges(
        "normalize_phrases",
        route_after_normalize,
        {
            "finish": END,                 # phrases 비어 있으면 그래프 종료
            "continue": "intent_classifier",  # 아니면 기존대로 intent_classifier로
        },
    )

    builder.add_conditional_edges(
        "intent_classifier",
        route_intent,
        {
            "emergency_generate": "emergency_generate",
            "normal_generate": "normal_generate",
        },
    )

    builder.add_edge("emergency_generate", "emergency_check")
    builder.add_conditional_edges(
        "emergency_check",
        route_emergency_check,
        {
            "finish": END,
            "emergency_generate": "emergency_generate",
        },
    )

    builder.add_edge("normal_generate", "refine_sentence")
    builder.add_edge("refine_sentence", "normal_check")
    builder.add_conditional_edges(
        "normal_check",
        route_normal_check,
        {
            "finish": END,
            "refine_sentence": "refine_sentence",
        },
    )

    return builder.compile()
