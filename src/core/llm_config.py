import os
import logging
from langchain_openai import ChatOpenAI
from src.core.state import RouteDecision, RagJudge
from dotenv import load_dotenv

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Load environment variables ────────────────────────────────────────
try:
    logger.info("Loading environment variables...")
    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = 'true'
    os.environ["LANGSMITH_ENDPOINT"] = 'https://api.smith.langchain.com'
    os.environ["LANGSMITH_API_KEY"] = os.getenv('LANGSMITH_API_KEY')
    os.environ["LANGSMITH_PROJECT"] = 'Rag_agent_project'
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    logger.info("Environment variables loaded successfully.")
except Exception as e:
    logger.error(f"Error loading environment variables: {e}")
    raise

# ── LLM instances with structured output where needed ───────────────
try:
    logger.info("Initializing LLM instances...")
    router_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)\
                 .with_structured_output(RouteDecision)
    judge_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)\
                .with_structured_output(RagJudge)
    answer_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    logger.info("LLM instances initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM instances: {e}")
    raise
