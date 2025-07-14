import streamlit as st
from datetime import datetime, timedelta
from uuid import uuid4
from langchain.schema import HumanMessage, AIMessage
from src.agents.graph import graph_agent
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
import logging
from src.core.logging_config import setup_logging

# ── LOGGING ─────────────────────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)

# ── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Assistant", layout="centered")

# ── SIDEBAR SETUP ────────────────────────────────────────────────────
st.sidebar.title("Do Chat 💬")
test_pdf_path = "data/testfile.pdf"
if os.path.exists(test_pdf_path):
    st.sidebar.markdown("### 📄 Upload this sample PDF for testing")
    with open(test_pdf_path, "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.download_button(
        label="Download sample PDF",
        data=pdf_bytes,
        file_name="testfile.pdf",
        mime="application/pdf",
    )
# ── RAG KNOWLEDGE BASE ────────────────────────────────────────────────
test_rag_file_path = "data/starxai.pdf"
if os.path.exists(test_rag_file_path):
    st.sidebar.markdown("### 🗄️ Look at the RAG Knowledge Base used")
    with open(test_rag_file_path, "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.download_button(
        label="Download Starxai PDF",
        data=pdf_bytes,
        file_name="starxai.pdf",
        mime="application/pdf",
    )

# ── NAVIGATION SETUP ────────────────────────────────────────────────
PAGES = ["🏠 Home", "💬 Chat"]

# Initialize a page state if missing
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# Sidebar radio to switch pages
st.sidebar.radio("Navigate", PAGES, key="page")


if st.session_state.page == "🏠 Home":
    # ── WELCOME PAGE ────────────────────────────────────────────────
    st.title("Welcome to RAG AGENT Chatbot 🚀")
    # Architecture diagram

    st.markdown(
    """
    **What is RAG AGENT Chatbot this?**  
    This is a Retrieval‑Augmented Generation (RAG) assistant powered by:
    - **PDF upload**: Uploada a PDF for instant context  
    - **Vector‑store RAG**: I’ve pre‑loaded **`starxai.pdf`** with Starx AI technology info  
    - **Web search**: fallback to live web when needed  

    **RAG knowledge base source**  
    Used `starxai.pdf` (about Starx AI Technology and its company details).

    **How to test**  
    1. **To test the Rag** - Download `starxai.pdf` from the sidebar ask this Sample questions.  
        - “What is the aim of the Starx AI technology?”  
        - “Who is the CEO of Starx AI technology?”  
    2. **Ask a web search question** to pull fresh info:  
       - “What is the feedback of the new 2025 F1 movie released starring Brad Pitt?”  
       - “Who is the 2025 President of USA?”  
       
    3. **Use the test PDF** (`testfile.pdf`) to check upload-only flows:  
       - “Why is Sanna Vaara saying she is worried?”  
       - "Who is Sanna Vaara?”  
    """)
    # Architecture diagram
    st.image("data/agent_graph.png", caption="My LangGraph Agent Architecture Flow", use_container_width=True)
else: 
    st.set_page_config(page_title="RAG Assistant", layout="centered")
    st.title("🤖 RAG_AGENT Chatbot")
    st.markdown("Upload a PDF or ask me anything — I’ll leverage your document, RAG knowledge base, and web search to deliver the best answer.")


    SESSION_TIMEOUT = timedelta(minutes=10)
    now = datetime.now()

    if "thread_id" not in st.session_state or "last_active" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.last_active = now
        logger.info(f"New session started with thread_id: {st.session_state.thread_id}")
    elif now - st.session_state.last_active > SESSION_TIMEOUT:
        logger.warning(f"Session timed out. Creating new session.")
        st.session_state.thread_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.last_active = now
        logger.info(f"New session started with thread_id: {st.session_state.thread_id}")

    st.session_state.last_active = now

    


    for msg in st.session_state.messages:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input(
        placeholder="Ask a question…",
        accept_file=True,
        file_type=["pdf"],
    )

    if user_input:
        # Show user text
        question = user_input.text or ""
        st.chat_message("user").markdown(question)
        logger.info(f"User question: {question}")

        # Extract PDF text if uploaded
        file_ctx = ""
        if user_input.files:
            pdf = user_input.files[0]
            st.chat_message("user").markdown(f"📄 Uploaded: **{pdf.name}**")
            logger.info(f"User uploaded file: {pdf.name}")
            try:
                suffix = os.path.splitext(pdf.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(pdf.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                pages = loader.load()[:2]        # first two pages only
                os.unlink(tmp_path)
                file_ctx = "\n\n".join(p.page_content for p in pages)
                logger.info(f"Successfully loaded and processed PDF file: {pdf.name}")
            except Exception as e:
                logger.error(f"Error processing uploaded PDF file: {e}")
                st.chat_message("assistant").markdown(f"❌ Error processing PDF: `{e}`")
                file_ctx = ""


        full_payload = (file_ctx + "\n\n" + question).strip()
        user_msg = HumanMessage(content=full_payload)
        st.session_state.messages.append(user_msg)

        try:
            logger.info("Invoking graph agent")
            result = graph_agent.invoke(
                {
                    "messages": [user_msg],
                    "upload_file_content": file_ctx
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            logger.info("Graph agent invocation successful")
            # extract assistant reply
            ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
            if ai_msg:
                st.chat_message("assistant").markdown(ai_msg.content)
                st.session_state.messages.append(ai_msg)
                logger.info(f"AI response: {ai_msg.content}")
            else:
                st.chat_message("assistant").markdown("⚠️ No response generated.")
                logger.warning("No response generated by the AI.")
        except Exception as e:
            logger.error(f"Error invoking graph agent: {e}")
            st.chat_message("assistant").markdown(f"❌ Error: `{e}`")
