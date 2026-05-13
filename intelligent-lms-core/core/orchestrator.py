from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from core.agents import socratic_agent, memory_agent
from services.hybrid_rag import hybrid_retrieve

def rag_node(state: AgentState) -> dict:
    """Retrieves context and prerequisites before agents run."""
    # Only retrieve if there's a new message
    if not state["messages"]:
        return {}
    
    last_message = state["messages"][-1].content
    retrieval_data = hybrid_retrieve(last_message, state.get("courseid"))
    
    return {
        "retrieved_docs": retrieval_data["documents"],
        "prerequisites": retrieval_data["prerequisites"],
        "matched_concepts": retrieval_data["matched_concepts"],
        "graph_nodes": retrieval_data["graph_nodes"]
    }

# Build the Graph
builder = StateGraph(AgentState)

builder.add_node("rag", rag_node)
builder.add_node("memory", memory_agent)
builder.add_node("socratic", socratic_agent)

# Define edges
builder.add_edge(START, "rag")
builder.add_edge("rag", "memory")
builder.add_edge("memory", "socratic")
builder.add_edge("socratic", END)

# Compile the graph
graph = builder.compile()
