# app/services/sentence_service.py
from app.aac_graph.builder import build_graph
from app.aac_graph.state import GraphState

graph = build_graph()

# def build_sentence_for_user(user_id: str) -> str:
#     initial: GraphState = {"user_id": user_id}
#     result = graph.invoke(initial)
#     return result.get("final_sentence") or ""


def build_sentence_for_user(user_id: str) -> GraphState:
    initial: GraphState = {"user_id": user_id}
    result = graph.invoke(initial)
    return result