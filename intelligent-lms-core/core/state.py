from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from services.telemetry_ml import CognitiveState

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    retrieved_docs: List[str]
    prerequisites: List[str]
    matched_concepts: List[str]
    current_load_state: CognitiveState
    graph_nodes: Dict[str, Any]
    missing_nodes: List[str]
    courseid: str
    user_id: str

    # New fields for decision agent & prerequisite analysis
    learning_intent: str           # "learning" or "content_only"
    detected_concepts: List[str]   # Concepts identified in user query by decision agent
    user_prerequisites: List[str]  # Concepts the user has MASTERED
    concept_prerequisites: List[str] # Prerequisites of the detected concept(s)
    missing_prerequisites: List[str] # Concept prereqs NOT covered by user's mastered
    concept_vectors: List[str]     # Retrieved doc chunks for main concept + gaps
