from pydantic import BaseModel, Field
from typing import List, Optional,Literal,TypedDict
from langchain_core.messages import BaseMessage

class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool
    

# ── Shared state type ────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    upload_file_content: Optional[str]
    route:    Literal["rag", "answer", "end"]
    rag:      str
    web:      str