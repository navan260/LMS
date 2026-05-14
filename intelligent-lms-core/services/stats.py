from typing import Dict, Any, List, Optional

from services.hybrid_rag import graph_db, get_enrolled_courses_with_counts


def _safe_query(cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if not graph_db:
        return []
    try:
        return graph_db.query(cypher, params or {})
    except Exception as e:
        print(f"Error querying Neo4j stats: {e}")
        return []


def get_student_stats(user_id: str, courseid: Optional[str] = None) -> Dict[str, Any]:
    courses = get_enrolled_courses_with_counts(user_id)
    if not courses:
        return {"status": "not_enrolled", "message": "User is not enrolled in any course.", "courses": []}

    selected_course = courseid or courses[0].get("courseid")
    if courseid:
        enrolled_course_ids = {c.get("courseid") for c in courses if c.get("courseid")}
        if courseid not in enrolled_course_ids:
            return {"status": "not_enrolled", "message": "User is not enrolled in that course.", "courses": courses}

    counts_query = """
    MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    OPTIONAL MATCH (u:User {id: $user_id})-[r:MASTERED|LEARNING|STRUGGLING_WITH]->(c)
    WITH c, r
    RETURN
        sum(CASE WHEN type(r) = 'MASTERED' THEN 1 ELSE 0 END) AS mastered_count,
        sum(CASE WHEN type(r) = 'LEARNING' THEN 1 ELSE 0 END) AS learning_count,
        sum(CASE WHEN type(r) = 'STRUGGLING_WITH' THEN 1 ELSE 0 END) AS struggling_count,
        sum(CASE WHEN r IS NULL THEN 1 ELSE 0 END) AS not_started_count,
        count(*) AS total_concepts
    """
    counts_res = _safe_query(counts_query, {"user_id": user_id, "courseid": selected_course})
    counts = counts_res[0] if counts_res else {
        "mastered_count": 0,
        "learning_count": 0,
        "struggling_count": 0,
        "not_started_count": 0,
        "total_concepts": 0
    }

    top_mastered_query = """
    MATCH (u:User {id: $user_id})-[m:MASTERED]->(c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    RETURN c.name AS name, m.score AS score, m.attempts AS attempts, m.timestamp AS timestamp
    ORDER BY m.score DESC, m.attempts DESC, m.timestamp DESC
    LIMIT 5
    """
    top_mastered = _safe_query(top_mastered_query, {"user_id": user_id, "courseid": selected_course})

    learning_query = """
    MATCH (u:User {id: $user_id})-[r:LEARNING]->(c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    RETURN c.name AS name, r.last_seen AS last_seen
    ORDER BY r.last_seen DESC
    LIMIT 6
    """
    learning_concepts = _safe_query(learning_query, {"user_id": user_id, "courseid": selected_course})

    struggling_query = """
    MATCH (u:User {id: $user_id})-[r:STRUGGLING_WITH]->(c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    RETURN c.name AS name, r.score AS score, r.attempts AS attempts, r.timestamp AS timestamp
    ORDER BY r.score ASC, r.attempts DESC, r.timestamp DESC
    LIMIT 6
    """
    struggling_concepts = _safe_query(struggling_query, {"user_id": user_id, "courseid": selected_course})

    nodes_query = """
    MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    OPTIONAL MATCH (u:User {id: $user_id})-[r:MASTERED|LEARNING|STRUGGLING_WITH]->(c)
    WITH c, r
    RETURN c.name AS name,
    CASE
        WHEN type(r) = 'MASTERED' THEN 'MASTERED'
        WHEN type(r) = 'LEARNING' THEN 'LEARNING'
        WHEN type(r) = 'STRUGGLING_WITH' THEN 'STRUGGLING_WITH'
        ELSE 'NOT_STARTED'
    END AS status
    ORDER BY c.name
    LIMIT $limit
    """
    nodes = _safe_query(nodes_query, {"user_id": user_id, "courseid": selected_course, "limit": 40})
    node_names = [n.get("name", "").lower() for n in nodes if n.get("name")]

    edges = []
    if node_names:
        edges_query = """
        MATCH (a:Concept)-[:PREREQUISITE_OF]->(b:Concept)
        WHERE toLower(a.name) IN $names AND toLower(b.name) IN $names
        RETURN a.name AS source, b.name AS target
        LIMIT 60
        """
        edges = _safe_query(edges_query, {"names": node_names})

    return {
        "status": "ok",
        "courses": courses,
        "selected_courseid": selected_course,
        "concept_counts": counts,
        "top_mastered": top_mastered,
        "learning_concepts": learning_concepts,
        "struggling_concepts": struggling_concepts,
        "nodes": nodes,
        "edges": edges
    }


def get_admin_stats() -> Dict[str, Any]:
    total_courses_res = _safe_query("MATCH (c:Course) RETURN count(c) AS total_courses")
    total_courses = int(total_courses_res[0]["total_courses"]) if total_courses_res else 0

    total_students_res = _safe_query("MATCH (u:User) RETURN count(u) AS total_students")
    total_students = int(total_students_res[0]["total_students"]) if total_students_res else 0

    enrollment_query = """
    MATCH (c:Course)
    OPTIONAL MATCH (:User)-[:ENROLLED_IN]->(c)
    RETURN c.courseid AS courseid, c.shortname AS shortname, c.fullname AS fullname,
           count(*) AS enrolled_count
    ORDER BY enrolled_count DESC
    LIMIT 8
    """
    top_courses = _safe_query(enrollment_query)

    mastery_query = """
    MATCH (c:Course)
    OPTIONAL MATCH (c)<-[:PART_OF]-(concept:Concept)
    WITH c, count(DISTINCT concept) AS concept_count
    OPTIONAL MATCH (c)<-[:ENROLLED_IN]-(u:User)
    WITH c, concept_count, count(DISTINCT u) AS enrolled_count
    CALL {
        WITH c
        OPTIONAL MATCH (c)<-[:PART_OF]-(concept:Concept)<-[:MASTERED]-(:User)
        RETURN count(*) AS mastered_edges
    }
    RETURN c.courseid AS courseid,
           c.shortname AS shortname,
           c.fullname AS fullname,
           enrolled_count,
           concept_count,
           mastered_edges,
           CASE
             WHEN concept_count = 0 OR enrolled_count = 0 THEN 0
             ELSE toFloat(mastered_edges) / (concept_count * enrolled_count)
           END AS mastery_rate
    ORDER BY mastery_rate DESC
    LIMIT 8
    """
    mastery_rates = _safe_query(mastery_query)

    total_enrollments_res = _safe_query("MATCH (:User)-[e:ENROLLED_IN]->(:Course) RETURN count(e) AS total_enrollments")
    total_enrollments = int(total_enrollments_res[0]["total_enrollments"]) if total_enrollments_res else 0

    top_course = top_courses[0] if top_courses else None

    return {
        "status": "ok",
        "total_courses": total_courses,
        "total_students": total_students,
        "total_enrollments": total_enrollments,
        "top_course": top_course,
        "top_courses": top_courses,
        "mastery_rates": mastery_rates
    }


def get_coordinator_stats(courseid: str, top_k: int = 5) -> Dict[str, Any]:
    course_info = _safe_query(
        "MATCH (c:Course {courseid: $courseid}) RETURN c.courseid AS courseid, c.shortname AS shortname, c.fullname AS fullname",
        {"courseid": courseid}
    )
    if not course_info:
        return {"status": "not_found", "message": "Course not found."}

    documents_query = """
    MATCH (d:Document)-[:FOR_COURSE]->(c:Course {courseid: $courseid})
    RETURN d.source AS name
    ORDER BY d.source
    """
    documents = _safe_query(documents_query, {"courseid": courseid})

    concepts_query = """
    MATCH (c:Concept)-[:PART_OF]->(:Course {courseid: $courseid})
    RETURN c.name AS name
    ORDER BY c.name
    """
    concepts = _safe_query(concepts_query, {"courseid": courseid})

    enrollment_res = _safe_query(
        "MATCH (:User)-[:ENROLLED_IN]->(c:Course {courseid: $courseid}) RETURN count(*) AS enrolled_count",
        {"courseid": courseid}
    )
    enrolled_count = int(enrollment_res[0]["enrolled_count"]) if enrollment_res else 0

    mastered_all_query = """
    MATCH (c:Course {courseid: $courseid})<-[:PART_OF]-(concept:Concept)
    WITH c, count(concept) AS total_concepts
    MATCH (u:User)-[:ENROLLED_IN]->(c)
    OPTIONAL MATCH (u)-[:MASTERED]->(concept)
    WITH u, total_concepts, count(DISTINCT concept) AS mastered_count
    WHERE total_concepts > 0 AND mastered_count = total_concepts
    RETURN count(u) AS mastered_all_count
    """
    mastered_all_res = _safe_query(mastered_all_query, {"courseid": courseid})
    mastered_all_count = int(mastered_all_res[0]["mastered_all_count"]) if mastered_all_res else 0

    pending_count = max(enrolled_count - mastered_all_count, 0)

    top_mastered_query = """
    MATCH (c:Course {courseid: $courseid})<-[:PART_OF]-(concept:Concept)
    OPTIONAL MATCH (:User)-[:MASTERED]->(concept)
    RETURN concept.name AS name, count(*) AS mastered_count
    ORDER BY mastered_count DESC
    LIMIT $limit
    """
    top_mastered = _safe_query(top_mastered_query, {"courseid": courseid, "limit": top_k})

    struggling_query = """
    MATCH (c:Course {courseid: $courseid})<-[:PART_OF]-(concept:Concept)
    MATCH (:User)-[:STRUGGLING_WITH]->(concept)
    RETURN concept.name AS name, count(*) AS struggling_count
    ORDER BY struggling_count DESC
    LIMIT $limit
    """
    struggling_concepts = _safe_query(struggling_query, {"courseid": courseid, "limit": max(top_k, 8)})

    return {
        "status": "ok",
        "course": course_info[0],
        "documents": {
            "count": len(documents),
            "items": documents
        },
        "concepts": {
            "count": len(concepts),
            "items": concepts
        },
        "enrollment_count": enrolled_count,
        "mastered_all_count": mastered_all_count,
        "pending_count": pending_count,
        "top_mastered_concepts": top_mastered,
        "struggling_concepts": struggling_concepts
    }
