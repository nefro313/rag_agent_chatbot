from typing import Literal
from src.core.state import AgentState

# ── Routing helpers ─────────────────────────────────────────────────
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
     return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
     return st["route"]

def after_web(_) -> Literal["answer"]:
     return "answer"
