from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from services.telemetry_ml import CognitiveState

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    retrieved_docs: List[str]
    prerequisites: List[str]
    matched_concepts: List[str]  # Concepts directly relevant to the user's query
    current_load_state: CognitiveState
    graph_nodes: Dict[str, Any] # Memory agent needs this
    missing_nodes: List[str] # Output from memory agent
    courseid: str
