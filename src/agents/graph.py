import logging
from langgraph.graph import StateGraph, END
from src.agents.nodes import router_node, rag_node, web_node, answer_node
from src.core.state import AgentState
from src.agents.routing import from_router, after_rag
from langgraph.checkpoint.memory import MemorySaver

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Build graph ─────────────────────────────────────────────────────
logger.info("Building agent graph...")
g = StateGraph(AgentState)
g.add_node("router", router_node)
g.add_node("rag_lookup", rag_node)
g.add_node("web_search", web_node)
g.add_node("answer", answer_node)

g.set_entry_point("router")
g.add_conditional_edges("router", from_router,
                        {"rag": "rag_lookup", "answer": "answer", "end": END})
g.add_conditional_edges("rag_lookup", after_rag,
                        {"answer": "answer", "web": "web_search"})
g.add_edge("web_search", "answer")
g.add_edge("answer", END)

logger.info("Compiling agent graph...")
graph_agent = g.compile(checkpointer=MemorySaver())
logger.info("Agent graph compiled successfully.")
