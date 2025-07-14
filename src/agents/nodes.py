import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.core.llm_config import router_llm, judge_llm, answer_llm
from src.core.state import AgentState, RouteDecision, RagJudge
from src.tools.rag import rag_search_tool
from src.tools.web_search import web_search_tool

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Node 1: decision/router ─────────────────────────────────────────
def router_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering router_node")
        user_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
        query = user_msgs[-1].content if user_msgs else ""
        logger.info(f"Router query: {query}")

        file_ctx = state.get("upload_file_content", "").strip()
        has_file = bool(file_ctx)

        system_prompt = (
            "You are a router that decides how to handle user queries.\n"
            "- If the user is just greeting or small‑talk, return route='end'.\n"
            "- If upload_file_content is non‑empty, return route='answer' so the assistant "
            "can answer directly from that content.\n"
            "- Otherwise, if the question needs domain knowledge from the KB, return route='rag'.\n"
            "- If RAG lookup is not enough, return route='web'.\n"
            "- If you can answer without any external info, return route='answer'.\n\n"
            f"Upload present? {'Yes' if has_file else 'No'}"
        )

        decision: RouteDecision = router_llm.invoke([
            ("system", system_prompt),
            ("user", query)
        ])
        logger.info(f"Router decision: {decision.route}")

        new_msgs = state.get("messages", []) + [HumanMessage(content=query)]
        out: AgentState = {
            **state,
            "messages": new_msgs,
            "route": decision.route
        }

        if decision.route == "end":
            out["messages"] = new_msgs + [
                AIMessage(content=decision.reply or "Hello!")
            ]

        logger.info("Exiting router_node")
        return out
    except Exception as e:
        logger.error(f"Error in router_node: {e}")
        raise

# ── Node 2: RAG lookup ───────────────────────────────────────────────
def rag_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering rag_node")
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        logger.info(f"RAG query: {query}")

        chunks = rag_search_tool.invoke({"query": query})
        logger.info(f"RAG retrieved {len(chunks)} chunks")

        judge_messages = [
            ("system", ("You are a judge evaluating if the retrieved information is sufficient "
                "to answer the user's question. Consider both relevance and completeness."
            )),
            ("user", f"Question: {query}\n\nRetrieved info: {chunks}\n\nIs this sufficient to answer the question?")
        ]

        verdict: RagJudge = judge_llm.invoke(judge_messages)
        logger.info(f"RAG judge verdict: {'sufficient' if verdict.sufficient else 'insufficient'}")

        logger.info("Exiting rag_node")
        return {
            **state,
            "rag": chunks,
            "route": "answer" if verdict.sufficient else "web"
        }
    except Exception as e:
        logger.error(f"Error in rag_node: {e}")
        raise

# ── Node 3: web search ───────────────────────────────────────────────
def web_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering web_node")
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        logger.info(f"Web search query: {query}")

        snippets = web_search_tool.invoke({"query": query})
        logger.info(f"Web search retrieved {len(snippets)} snippets")

        logger.info("Exiting web_node")
        return {**state, "web": snippets, "route": "answer"}
    except Exception as e:
        logger.error(f"Error in web_node: {e}")
        raise

# ── Node 4: final answer ─────────────────────────────────────────────
def answer_node(state: AgentState) -> AgentState:
    try:
        logger.info("Entering answer_node")
        user_q = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            ""
        )
        logger.info(f"Answer node query: {user_q}")

        chat_history = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in state["messages"]
        )

        file_ctx = state.get("upload_file_content", "").strip()
        rag_ctx = state.get("rag", "").strip()
        web_ctx = state.get("web", "").strip()

        if file_ctx:
            context = f"Uploaded PDF (first ≤2 pages):\n{file_ctx}"
        else:
            parts = []
            if rag_ctx:
                parts.append("Knowledge Base:\n" + rag_ctx)
            if web_ctx:
                parts.append("Web Search:\n" + web_ctx)
            context = "\n\n".join(parts) if parts else "No external context available."
        logger.info(f"Answer node context: {context}")

        prompt = f"""
You are a helpful assistant. Use the information below to answer the user's latest question.

Conversation so far:
{chat_history}

Context (prioritize uploaded file if present):
{context}

Now answer clearly and concisely:
Question: {user_q}
"""

        answer = answer_llm.invoke([HumanMessage(content=prompt)]).content
        logger.info(f"Generated answer: {answer}")

        logger.info("Exiting answer_node")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=answer)]
        }
    except Exception as e:
        logger.error(f"Error in answer_node: {e}")
        raise
