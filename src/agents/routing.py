import logging
from typing import Literal
from src.core.state import AgentState

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Routing helpers ─────────────────────────────────────────────────
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    route = st["route"]
    logger.info(f"Routing from router: {route}")
    return route

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    route = st["route"]
    logger.info(f"Routing after RAG: {route}")
    return route

def after_web(_) -> Literal["answer"]:
    logger.info("Routing after web search to answer")
    return "answer"
