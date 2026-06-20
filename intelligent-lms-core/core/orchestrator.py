from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from core.agents import decision_agent, socratic_agent
from services.hybrid_rag import (
    hybrid_retrieve,
    get_user_mastered_concepts,
    get_concept_prerequisites,
    get_concept_vectors
)
from services.tutoring import tag_concepts_as_learning

def rag_node(state: AgentState) -> dict:
    """Retrieves context based on the decision agent's intent analysis."""
    if not state["messages"]:
        return {}

    last_message = state["messages"][-1].content
    learning_intent = state.get("learning_intent", "learning")
    courseid = state.get("courseid")
    user_id = state.get("user_id")
    detected_concepts = state.get("detected_concepts", [])

    # Always do basic retrieval (documents + graph concepts)
    retrieval_data = hybrid_retrieve(last_message, courseid)
    result = {
        "retrieved_docs": retrieval_data["documents"],
        "prerequisites": retrieval_data["prerequisites"],
        "matched_concepts": retrieval_data["matched_concepts"],
        "graph_nodes": retrieval_data["graph_nodes"]
    }

    if learning_intent == "learning" and user_id:
        # Full prerequisite analysis
        user_prereqs = get_user_mastered_concepts(user_id, courseid)

        # Use detected concepts from decision agent, fall back to matched from hybrid_retrieve
        concepts_to_check = detected_concepts or retrieval_data["matched_concepts"]
        concept_prereqs = get_concept_prerequisites(concepts_to_check, courseid)

        # Missing = concept prereqs not in user's mastered set
        user_prereq_set = {p.lower() for p in user_prereqs}
        missing = [p for p in concept_prereqs if p.lower() not in user_prereq_set]

        # Get concept vectors for main concepts + any missing prerequisites
        vector_targets = list(set(concepts_to_check + missing))
        concept_vectors = get_concept_vectors(vector_targets, courseid)

        # Tag all fetched concepts as LEARNING in Neo4j
        tag_concepts_as_learning(user_id, concepts_to_check, courseid)
        if concept_prereqs:
            tag_concepts_as_learning(user_id, concept_prereqs, courseid)

        result.update({
            "user_prerequisites": user_prereqs,
            "concept_prerequisites": concept_prereqs,
            "missing_prerequisites": missing,
            "concept_vectors": concept_vectors
        })

    return result

# Build the Graph
builder = StateGraph(AgentState)

builder.add_node("decision", decision_agent)
builder.add_node("rag", rag_node)
builder.add_node("socratic", socratic_agent)

builder.add_edge(START, "decision")
builder.add_edge("decision", "rag")
builder.add_edge("rag", "socratic")
builder.add_edge("socratic", END)

graph = builder.compile()
