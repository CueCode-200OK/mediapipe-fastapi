# app/aac_graph/state.py
from typing import TypedDict, List, Literal, Optional

Intent = Literal["EMERGENCY", "REQUEST", "STATUS", "OTHER"]
RuleStatus = Literal["OK", "REWRITE", "TOO_LONG", "NOT_POLITE", "UNCLEAR"]

class GraphState(TypedDict, total=False):
    user_id: str

    # 입력/중간 상태
    raw_phrases: List[str]             # ex) ["나", "물", "물", "주세요"]
    normalized_phrases: List[str]      # ex) ["나", "물", "주세요"]
    intent: Intent                     # ex) "EMERGENCY"
    draft_sentence: str                # 첫 생성 문장
    refined_sentence: str              # Refiner 후 문장
    final_sentence: str                # 최종 문장
    rule_status: RuleStatus            # 체크 결과
    is_emergency: bool                 # 브랜치 플래그

    # 오류 / 진단용
    error_message: Optional[str]