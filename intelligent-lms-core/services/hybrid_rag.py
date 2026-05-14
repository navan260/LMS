import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

# Pydantic models for Knowledge Graph Extraction
class Concept(BaseModel):
    name: str = Field(description="Name of the concept (lowercase, snake_case or short phrase)")
    description: str = Field(description="Brief description of the concept")

class Relationship(BaseModel):
    source_concept: str
    target_concept: str
    relationship_type: str = Field(description="Must be PREREQUISITE_OF or RELATED_TO")

class KnowledgeGraphExtraction(BaseModel):
    concepts: List[Concept]
    relationships: List[Relationship]

# Initialize Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

# Initialize DataStax Astra DB Vector Store
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")

vector_store = None
if ASTRA_DB_API_ENDPOINT and ASTRA_DB_API_ENDPOINT != "...":
    try:
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="lms_documents",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        print("Connected to DataStax Astra DB.")
    except Exception as e:
        print(f"Error connecting to Astra DB: {e}")

# Initialize Neo4j Graph
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

graph_db = None
if NEO4J_URI and NEO4J_URI != "...":
    try:
        graph_db = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

def initialize_neo4j_schema():
    if graph_db:
        try:
            graph_db.query("CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            graph_db.query("CREATE CONSTRAINT course_id_unique IF NOT EXISTS FOR (c:Course) REQUIRE c.courseid IS UNIQUE")
            graph_db.query("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            print("Neo4j schema initialized.")
        except Exception as e:
            print(f"Error initializing Neo4j schema: {e}")

def ensure_user_course_nodes(
    user_id: str,
    username: str,
    email: str,
    fullname: str,
    courseid: str,
    courseshortname: str,
    coursefullname: str
):
    if not graph_db:
        return
    cypher = """
    MERGE (u:User {id: $user_id})
    SET u.username = $username,
        u.email = $email,
        u.fullname = $fullname
    MERGE (c:Course {courseid: $courseid})
    SET c.shortname = $courseshortname,
        c.fullname = $coursefullname
    """
    try:
        graph_db.query(
            cypher,
            {
                "user_id": user_id,
                "username": username,
                "email": email,
                "fullname": fullname,
                "courseid": courseid,
                "courseshortname": courseshortname,
                "coursefullname": coursefullname
            }
        )
    except Exception as e:
        print(f"Error ensuring user/course relationship in Neo4j: {e}")

def enroll_user_in_course(user_id: str, courseid: str):
    if not graph_db:
        return
    cypher = """
    MATCH (u:User {id: $user_id})
    MATCH (c:Course {courseid: $courseid})
    MERGE (u)-[:ENROLLED_IN]->(c)
    """
    try:
        graph_db.query(cypher, {"user_id": user_id, "courseid": courseid})
    except Exception as e:
        print(f"Error enrolling user in course: {e}")

def assign_coordinator_to_course(user_id: str, courseid: str):
    if not graph_db:
        return
    cypher = """
    MATCH (u:User {id: $user_id})
    MATCH (c:Course {courseid: $courseid})
    MERGE (u)-[:COORDINATOR_OF]->(c)
    """
    try:
        graph_db.query(cypher, {"user_id": user_id, "courseid": courseid})
    except Exception as e:
        print(f"Error assigning coordinator to course: {e}")

def is_enrolled_in_course(user_id: str, courseid: str) -> bool:
    if not graph_db:
        return False
    if not courseid:
        return False
    cypher = """
    MATCH (:User {id: $user_id})-[:ENROLLED_IN]->(:Course {courseid: $courseid})
    RETURN count(*) AS count
    """
    try:
        res = graph_db.query(cypher, {"user_id": user_id, "courseid": courseid})
        return bool(res and res[0].get("count", 0) > 0)
    except Exception as e:
        print(f"Error checking enrollment for course: {e}")
        return False

def is_coordinator_of_course(user_id: str, courseid: str) -> bool:
    if not graph_db:
        return False
    if not courseid:
        return False
    cypher = """
    MATCH (:User {id: $user_id})-[:COORDINATOR_OF]->(:Course {courseid: $courseid})
    RETURN count(*) AS count
    """
    try:
        res = graph_db.query(cypher, {"user_id": user_id, "courseid": courseid})
        return bool(res and res[0].get("count", 0) > 0)
    except Exception as e:
        print(f"Error checking coordinator for course: {e}")
        return False

def is_enrolled_any(user_id: str) -> bool:
    if not graph_db:
        return False
    cypher = """
    MATCH (:User {id: $user_id})-[:ENROLLED_IN]->(:Course)
    RETURN count(*) AS count
    """
    try:
        res = graph_db.query(cypher, {"user_id": user_id})
        return bool(res and res[0].get("count", 0) > 0)
    except Exception as e:
        print(f"Error checking enrollment for any course: {e}")
        return False

def get_enrolled_courses_with_counts(user_id: str) -> List[Dict[str, Any]]:
    if not graph_db:
        return []
    cypher = """
    MATCH (u:User {id: $user_id})-[:ENROLLED_IN]->(c:Course)
    OPTIONAL MATCH (:User)-[:ENROLLED_IN]->(c)
    RETURN c.courseid AS courseid,
           c.shortname AS courseshortname,
           c.fullname AS coursefullname,
           count(*) AS enrolled_count
    ORDER BY enrolled_count DESC
    """
    try:
        return graph_db.query(cypher, {"user_id": user_id})
    except Exception as e:
        print(f"Error fetching enrolled courses: {e}")
        return []

def get_coordinator_courses_with_counts(user_id: str) -> List[Dict[str, Any]]:
    if not graph_db:
        return []
    cypher = """
    MATCH (u:User {id: $user_id})-[:COORDINATOR_OF]->(c:Course)
    OPTIONAL MATCH (:User)-[:ENROLLED_IN]->(c)
    RETURN c.courseid AS courseid,
           c.shortname AS courseshortname,
           c.fullname AS coursefullname,
           count(*) AS enrolled_count
    ORDER BY enrolled_count DESC
    """
    try:
        return graph_db.query(cypher, {"user_id": user_id})
    except Exception as e:
        print(f"Error fetching coordinator courses: {e}")
        return []

# LLM for Extraction
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def ingest_document(text: str, source_name: str, course: Optional[Dict[str, str]] = None):
    """Processes uploaded text, stores chunks in Vector DB, and extracts graph to Neo4j."""
    
    # 1. Store in Vector DB (DataStax)
    if vector_store:
        # Simple chunking (for production, use Langchain's RecursiveCharacterTextSplitter)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        chunks = [c.strip() for c in chunks if c.strip()]
        
        if chunks:
            success_count = 0
            for c in chunks:
                try:
                    metadata = {"source": source_name}
                    if course:
                        metadata.update({
                            "courseid": course.get("courseid"),
                            "courseshortname": course.get("courseshortname"),
                            "coursefullname": course.get("coursefullname")
                        })
                    doc = Document(page_content=c, metadata=metadata)
                    vector_store.add_documents([doc])
                    success_count += 1
                except Exception as e:
                    print(f"Failed to add chunk to Vector Store. Error: {e}")
            print(f"Successfully added {success_count} out of {len(chunks)} chunks to Vector Store.")
        else:
            print("No valid text chunks to insert into Vector Store.")
    else:
        print("Vector store not configured. Skipping vector insertion.")

    # 2. Extract and Store Knowledge Graph (Neo4j)
    if graph_db:
        print("Extracting knowledge graph from text...")
        start_llm = time.time()
        structured_llm = llm.with_structured_output(KnowledgeGraphExtraction)
        prompt = f"Extract key programming/educational concepts and their prerequisite relationships from the following text:\n\n{text[:5000]}" # Limit text for extraction
        
        try:
            kg: KnowledgeGraphExtraction = structured_llm.invoke(prompt)
            llm_duration = time.time() - start_llm
            print(f"LLM Extraction took {llm_duration:.2f}s")
            
            start_neo4j = time.time()
            
            # Insert into Neo4j
            concepts_data = [{"name": c.name, "description": c.description} for c in kg.concepts]
            if concepts_data:
                graph_db.query(
                    "UNWIND $data AS item MERGE (c:Concept {name: toLower(item.name)}) SET c.description = item.description",
                    {"data": concepts_data}
                )

            courseid = None
            if course:
                courseid = course.get("courseid")
                if courseid:
                    graph_db.query(
                        "MERGE (c:Course {courseid: $courseid}) SET c.shortname = $shortname, c.fullname = $fullname",
                        {
                            "courseid": courseid,
                            "shortname": course.get("courseshortname"),
                            "fullname": course.get("coursefullname")
                        }
                    )
                    graph_db.query(
                        "MERGE (d:Document {source: $source, courseid: $courseid}) SET d.source = $source, d.courseid = $courseid",
                        {"source": source_name, "courseid": courseid}
                    )
                    graph_db.query(
                        "MATCH (d:Document {source: $source, courseid: $courseid}) MATCH (c:Course {courseid: $courseid}) MERGE (d)-[:FOR_COURSE]->(c)",
                        {"source": source_name, "courseid": courseid}
                    )
                    graph_db.query(
                        "UNWIND $data AS item MATCH (k:Concept {name: toLower(item.name)}) MATCH (c:Course {courseid: $courseid}) MERGE (k)-[:PART_OF]->(c)",
                        {"data": concepts_data, "courseid": courseid}
                    )
            
            prereq_data = [{"source": r.source_concept, "target": r.target_concept} for r in kg.relationships if r.relationship_type.upper() == "PREREQUISITE_OF"]
            related_data = [{"source": r.source_concept, "target": r.target_concept} for r in kg.relationships if r.relationship_type.upper() != "PREREQUISITE_OF"]

            if prereq_data:
                graph_db.query(
                    "UNWIND $data AS item MERGE (s:Concept {name: toLower(item.source)}) MERGE (t:Concept {name: toLower(item.target)}) MERGE (s)-[:PREREQUISITE_OF]->(t)",
                    {"data": prereq_data}
                )
            
            if related_data:
                graph_db.query(
                    "UNWIND $data AS item MERGE (s:Concept {name: toLower(item.source)}) MERGE (t:Concept {name: toLower(item.target)}) MERGE (s)-[:RELATED_TO]->(t)",
                    {"data": related_data}
                )
            
            neo4j_duration = time.time() - start_neo4j
            print(f"Neo4j Batch Ingest took {neo4j_duration:.2f}s")
            print(f"Extracted {len(kg.concepts)} concepts and {len(kg.relationships)} relationships to Neo4j.")
        except Exception as e:
            print(f"Error extracting/inserting graph: {e}")
    else:
        print("Neo4j not configured. Skipping graph extraction.")


def hybrid_retrieve(query: str, courseid: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves documents from DataStax and prerequisites from Neo4j."""
    
    retrieved_docs = []
    if vector_store:
        search_kwargs = {"k": 3}
        if courseid:
            search_kwargs["filter"] = {"courseid": courseid}
        results = vector_store.similarity_search(query, **search_kwargs)
        retrieved_docs = [r.page_content for r in results]
    else:
        retrieved_docs = ["(Vector RAG not configured)"]

    prerequisites = []
    graph_nodes = {}
    matched_concepts = []  # Must be initialised here — used in return regardless of graph_db state
    
    if graph_db:
        # Find concepts mentioned in query using word-level bidirectional matching.
        # This handles plurals/variants e.g. "loop" matches concept "loops", "variable" matches "variables".
        query_words = [w for w in query.lower().split() if len(w) > 2]
        print(f"[RAG] Query words for Neo4j matching: {query_words}")
        cypher_query = """
        MATCH (c:Concept)-[:PART_OF]->(course:Course {courseid: $courseid})
        WHERE ANY(word IN $query_words WHERE toLower(c.name) CONTAINS word OR word CONTAINS toLower(c.name))
        OPTIONAL MATCH (prereq:Concept)-[:PREREQUISITE_OF]->(c)
        RETURN c.name AS matched_concept, prereq.name AS prerequisite
        LIMIT 10
        """
        try:
            results = graph_db.query(cypher_query, {"query_words": query_words, "courseid": courseid})
            matched_concepts = list({r["matched_concept"] for r in results if r["matched_concept"]})
            prerequisites = list({r["prerequisite"] for r in results if r["prerequisite"]})
            print(f"[RAG] Matched concepts: {matched_concepts}, Prerequisites: {prerequisites}")
            
            # Also fetch all nodes for memory agent
            all_nodes_res = graph_db.query(
                "MATCH (c:Concept)-[:PART_OF]->(course:Course {courseid: $courseid}) RETURN c.name as name, c.description as desc LIMIT 20",
                {"courseid": courseid}
            )
            for record in all_nodes_res:
                graph_nodes[record["name"]] = {"name": record["name"], "description": record["desc"]}
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            
    else:
        prerequisites = ["(Graph RAG not configured)"]
        graph_nodes = {"variables": {"name": "Variables", "description": "Mock Data"}}

    return {
        "documents": retrieved_docs,
        "prerequisites": prerequisites,
        "matched_concepts": matched_concepts,
        "graph_nodes": graph_nodes
    }
