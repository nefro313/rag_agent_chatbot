import os
import logging
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# Initialize Tavily search tool
# Note: Ensure you have the correct API key set in your environment for Tavily
try:
    load_dotenv()
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    tavily = TavilySearch(max_results=3, topic="general")
    logger.info("TavilySearch tool initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing TavilySearch tool: {e}")
    raise

@tool
def web_search_tool(query: str) -> str:
    """Up-to-date web info via Tavily"""
    try:
        logger.info(f"Performing web search for query: {query}")
        result = tavily.invoke({"query": query})

        # Extract and format the results from Tavily response
        if isinstance(result, dict) and 'results' in result:
            formatted_results = []
            for item in result['results']:
                title = item.get('title', 'No title')
                content = item.get('content', 'No content')
                url = item.get('url', '')
                formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")

            logger.info(f"Web search found {len(formatted_results)} results for query: {query}")
            return "\n\n".join(formatted_results) if formatted_results else "No results found"
        else:
            logger.warning(f"Web search returned unexpected result type for query '{query}': {type(result)}")
            return str(result)
    except Exception as e:
        logger.error(f"Error during web search for query '{query}': {e}")
        return f"WEB_ERROR::{e}"

