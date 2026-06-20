import time
from typing import Optional
from pydantic import BaseModel
from services.hybrid_rag import llm, vector_store, graph_db

class ChallengeGeneration(BaseModel):
    concept_name: str
    question_text: str
    difficulty: str

class AssessmentResult(BaseModel):
    score: float
    feedback: str
    is_passed: bool
    socratic_hint: str

def get_next_challenge(user_id: str, courseid: Optional[str] = None) -> Optional[ChallengeGeneration]:
    if not graph_db:
        print("Graph DB not configured.")
        return None
    if not courseid:
        print("Course ID missing for challenge generation.")
        return None

    # Find a concept the user hasn't mastered, where all its prerequisites (if any) ARE mastered.
    cypher_query = """
    MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    WHERE NOT EXISTS {
        MATCH (:User {id: $user_id})-[:MASTERED]->(c)
    }
    AND NOT EXISTS {
        MATCH (prereq:Concept)-[:PREREQUISITE_OF]->(c)
        WHERE NOT EXISTS {
            MATCH (:User {id: $user_id})-[:MASTERED]->(prereq)
        }
    }
    RETURN c.name AS concept_name, c.description AS description
    LIMIT 1
    """
    
    try:
        params = {"user_id": user_id, "courseid": courseid}
        results = graph_db.query(cypher_query, params)
    except Exception as e:
        print(f"Error querying Neo4j for next challenge: {e}")
        return None

    if not results:
        fallback_query = """
        MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
        WHERE NOT EXISTS {
            MATCH (:User {id: $user_id})-[:MASTERED]->(c)
        }
        WITH c
        MATCH (prereq:Concept)-[:PREREQUISITE_OF]->(c)
        WHERE NOT EXISTS {
            MATCH (:User {id: $user_id})-[:MASTERED]->(prereq)
        }
        RETURN prereq.name AS concept_name, prereq.description AS description
        LIMIT 1
        """
        try:
            fallback = graph_db.query(fallback_query, {"user_id": user_id, "courseid": courseid})
        except Exception as e:
            print(f"Error querying Neo4j for fallback challenge: {e}")
            return None

        if not fallback:
            return None

        concept_name = fallback[0]["concept_name"]
        tag_concepts_as_learning(user_id, [concept_name], courseid)
    else:
        concept_name = results[0]["concept_name"]
    
    # Fetch Ground Truth
    context = ""
    if vector_store:
        try:
            search_kwargs = {"k": 3}
            if courseid:
                search_kwargs["filter"] = {"courseid": courseid}
            docs = vector_store.similarity_search(concept_name, **search_kwargs)
            context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error querying Astra DB for context: {e}")
            
    # Generate question
    structured_llm = llm.with_structured_output(ChallengeGeneration)
    prompt = f"""
    Based on the following context, generate a short-answer question to test the student's understanding of the concept '{concept_name}'.
    Return the concept_name, the question_text, and a difficulty level (Easy/Medium/Hard).
    
    Context:
    {context}
    """
    
    try:
        challenge: ChallengeGeneration = structured_llm.invoke(prompt)
        challenge.concept_name = concept_name # enforce correctness
        return challenge
    except Exception as e:
        print(f"Error generating challenge with Gemini: {e}")
        return None

def update_user_progress(user_id: str, concept_name: str, score: float):
    if not graph_db:
        return
        
    if score >= 0.7:
        cypher = """
        MERGE (u:User {id: $user_id})
        MATCH (c:Concept) WHERE toLower(c.name) = toLower($concept_name)
        MERGE (u)-[r:MASTERED]->(c)
        SET r.score = $score, r.timestamp = timestamp(), r.attempts = coalesce(r.attempts, 0) + 1
        WITH u, c
        OPTIONAL MATCH (u)-[s:STRUGGLING_WITH]->(c)
        DELETE s
        """
    else:
        cypher = """
        MERGE (u:User {id: $user_id})
        MATCH (c:Concept) WHERE toLower(c.name) = toLower($concept_name)
        MERGE (u)-[s:STRUGGLING_WITH]->(c)
        SET s.score = $score, s.timestamp = timestamp(), s.attempts = coalesce(s.attempts, 0) + 1
        """
        
    try:
        graph_db.query(cypher, {"user_id": user_id, "concept_name": concept_name, "score": score})
    except Exception as e:
        print(f"Error updating user progress in Neo4j: {e}")

def tag_concepts_as_learning(user_id: str, concept_names: list, courseid: Optional[str] = None):
    """Marks a list of concepts as LEARNING in Neo4j for the given user.
    Skips concepts already MASTERED so it doesn't overwrite mastery progress."""
    if not graph_db or not concept_names:
        print(f"[Neo4j] tag_concepts_as_learning skipped: graph_db={bool(graph_db)}, concepts={concept_names}")
        return

    print(f"[Neo4j] Attempting to tag as LEARNING for '{user_id}': {concept_names}")
    # OPTIONAL MATCH is used instead of NOT EXISTS subquery for broader Neo4j version compatibility
    if courseid:
        cypher = """
        UNWIND $concepts AS concept_name
        MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
        WHERE toLower(c.name) = toLower(concept_name)
        MERGE (u:User {id: $user_id})
        WITH concept_name, u, c
        OPTIONAL MATCH (u)-[m:MASTERED]->(c)
        WITH concept_name, u, c, m WHERE m IS NULL
        MERGE (u)-[r:LEARNING]->(c)
        SET r.last_seen = timestamp()
        RETURN concept_name
        """
    else:
        cypher = """
        UNWIND $concepts AS concept_name
        MATCH (c:Concept) WHERE toLower(c.name) = toLower(concept_name)
        MERGE (u:User {id: $user_id})
        WITH concept_name, u, c
        OPTIONAL MATCH (u)-[m:MASTERED]->(c)
        WITH concept_name, u, c, m WHERE m IS NULL
        MERGE (u)-[r:LEARNING]->(c)
        SET r.last_seen = timestamp()
        RETURN concept_name
        """
    try:
        params = {"user_id": user_id, "concepts": concept_names}
        if courseid:
            params["courseid"] = courseid
        result = graph_db.query(cypher, params)
        tagged = [row["concept_name"] for row in result]
        untagged = [n for n in concept_names if n not in tagged]
        print(f"[Neo4j] Tagged {len(tagged)}/{len(concept_names)} as LEARNING: {tagged}")
        if untagged:
            print(f"[Neo4j] WARNING: {len(untagged)} concept(s) not tagged — not found in course graph or already MASTERED: {untagged}")
    except Exception as e:
        print(f"[Neo4j] ERROR tagging concepts as LEARNING: {e}")

def grade_answer(user_id: str, concept_name: str, question: str, student_answer: str, courseid: Optional[str] = None) -> Optional[AssessmentResult]:
    context = ""
    if vector_store:
        try:
            search_kwargs = {"k": 3}
            if courseid:
                search_kwargs["filter"] = {"courseid": courseid}
            docs = vector_store.similarity_search(concept_name, **search_kwargs)
            context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error querying Astra DB for context: {e}")
            
    structured_llm = llm.with_structured_output(AssessmentResult)
    prompt = f"""
    Evaluate the student's answer to the question based on the provided ground truth context.
    
    Concept: {concept_name}
    Question: {question}
    Student Answer: {student_answer}
    
    Ground Truth Context:
    {context}
    
    Return a score between 0.0 and 1.0 (where >= 0.7 is passing), brief feedback, whether they passed, and a socratic hint if they failed.
    """
    
    try:
        result: AssessmentResult = structured_llm.invoke(prompt)
        update_user_progress(user_id, concept_name, result.score)
        return result
    except Exception as e:
        print(f"Error grading answer with Gemini: {e}")
        return None
