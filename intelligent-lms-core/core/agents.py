from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from core.state import AgentState
from services.telemetry_ml import CognitiveState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class DecisionOutput(BaseModel):
    intent: str = Field(description="Either 'learning' if user wants to understand/learn/be taught, or 'content_only' if user just wants direct facts or the raw content")
    detected_concepts: List[str] = Field(description="List of concept names mentioned or implied in the user's query")

def decision_agent(state: AgentState) -> dict:
    """Analyzes the user's latest message to determine intent and extract concepts."""
    if not state["messages"]:
        return {"learning_intent": "content_only", "detected_concepts": []}

    last_message = state["messages"][-1].content
    structured_llm = llm.with_structured_output(DecisionOutput)

    prompt = (
        "You are analyzing a student's query in an educational LMS. "
        "Determine:\n"
        "1. intent: 'learning' if the student wants to understand, learn, be taught, get explained, "
        "or needs help grasping a concept. 'content_only' if the student wants direct facts, definitions, "
        "raw text, or is just asking for information without needing scaffolding.\n"
        "2. detected_concepts: extract the main educational concepts/subjects mentioned or implied.\n\n"
        f"Student query: {last_message}"
    )

    try:
        result: DecisionOutput = structured_llm.invoke(prompt)
        return {
            "learning_intent": result.intent,
            "detected_concepts": result.detected_concepts
        }
    except Exception as e:
        print(f"[Decision Agent] Error: {e}")
        return {"learning_intent": "learning", "detected_concepts": []}

def socratic_agent(state: AgentState) -> dict:
    """Generates a response based on learning_intent and available context."""
    learning_intent = state.get("learning_intent", "learning")
    mode = state.get("mode", "Auto")
    cognitive_state = state.get("current_load_state", CognitiveState.FOCUSED)

    detected_concepts = state.get("detected_concepts", [])
    user_prereqs = state.get("user_prerequisites", [])
    missing_prereqs = state.get("missing_prerequisites", [])
    concept_prereqs = state.get("concept_prerequisites", [])
    concept_vectors = state.get("concept_vectors", [])
    retrieved_docs = state.get("retrieved_docs", [])

    doc_context = "\n\n".join(retrieved_docs) if retrieved_docs else ""

    if learning_intent == "content_only":
        system_prompt = (
            "You are a helpful educational assistant. "
            "Provide a direct, clear, and comprehensive answer to the student's question. "
            "Use the provided reference context if relevant. Do not ask questions back."
        )
        if doc_context:
            system_prompt += f"\n\nReference context:\n{doc_context}"
    elif mode == "Normal":
        prereqs_str = ", ".join(concept_prereqs) if concept_prereqs else "general programming concepts"
        user_knows_str = ", ".join(user_prereqs) if user_prereqs else "nothing yet"
        gaps_str = ", ".join(missing_prereqs) if missing_prereqs else "none (user has the required background)"

        system_prompt = (
            "You are a knowledgeable tutor helping a student learn. "
            "The student is asking about these concepts: "
        )
        if detected_concepts:
            system_prompt += f"{', '.join(detected_concepts)}. "
        system_prompt += (
            f"\n\nWhat the student already knows: {user_knows_str}."
            f"\nPrerequisites of the topic: {prereqs_str}."
            f"\nPrerequisites the student is missing (explain these first): {gaps_str}."
            "\n\nProvide a thorough explanation of the main topic. Cover any missing "
            "prerequisite concepts first, then explain the main topic in detail. "
            "Do NOT ask the student questions back — just teach them clearly."
        )

        if concept_vectors:
            system_prompt += f"\n\nReference material for the concepts:\n{' '.join(concept_vectors[:3])}"

        if cognitive_state == CognitiveState.FRUSTRATED:
            system_prompt += (
                "\n[High-Scaffolding]: The student appears frustrated. "
                "Provide easier hints, be encouraging, break things down."
            )
        elif cognitive_state == CognitiveState.IDLE:
            system_prompt += (
                "\n[Engagement]: The student appears idle. "
                "Ask an engaging question to bring them back."
            )
    else:
        prereqs_str = ", ".join(concept_prereqs) if concept_prereqs else "general programming concepts"
        user_knows_str = ", ".join(user_prereqs) if user_prereqs else "nothing yet"
        gaps_str = ", ".join(missing_prereqs) if missing_prereqs else "none (user has the required background)"

        system_prompt = (
            "You are a Socratic tutor helping a student learn. "
            "The student is asking about these concepts: "
        )
        if detected_concepts:
            system_prompt += f"{', '.join(detected_concepts)}. "
        system_prompt += (
            f"\n\nWhat the student already knows: {user_knows_str}."
            f"\nPrerequisites of the topic: {prereqs_str}."
            f"\nPrerequisites the student is missing (explain these first): {gaps_str}."
            "\n\nFirst, explain any missing prerequisite concepts the student needs. "
            "Then guide them toward understanding the main topic. "
            "Use a Socratic approach: ask guiding questions, check understanding, "
            "and build on what they already know. "
            "DO NOT give the full direct answer immediately. "
        )

        if concept_vectors:
            system_prompt += f"\n\nReference material for the concepts:\n{' '.join(concept_vectors[:3])}"

        if cognitive_state == CognitiveState.FRUSTRATED:
            system_prompt += (
                "\n[High-Scaffolding]: The student appears frustrated. "
                "Provide easier hints, be encouraging, break things down."
            )
        elif cognitive_state == CognitiveState.IDLE:
            system_prompt += (
                "\n[Engagement]: The student appears idle. "
                "Ask an engaging question to bring them back."
            )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
