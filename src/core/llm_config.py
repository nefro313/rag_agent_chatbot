import os
from langchain_openai import ChatOpenAI
from src.core.state import RouteDecision, RagJudge

from dotenv import load_dotenv
load_dotenv()
# ── Load environment variables ──────────────────────────────────────── 
os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGSMITH_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
os.environ["LANGSMITH_PROJECT"] = 'Rag_agent_project'      
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# ── LLM instances with structured output where needed ───────────────
router_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)\
             .with_structured_output(RouteDecision)
judge_llm  = ChatOpenAI(model="gpt-4.1-mini", temperature=0)\
             .with_structured_output(RagJudge)
answer_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)