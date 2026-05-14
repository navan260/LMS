from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
import io
import PyPDF2
import os
import jwt

from dotenv import load_dotenv
load_dotenv()

from core.orchestrator import graph
from services.telemetry_ml import analyze_student_state, TelemetryData, CognitiveState
from langchain_core.messages import HumanMessage
from services.hybrid_rag import is_enrolled_in_course, is_enrolled_any

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Intelligent Agentic LMS")

JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_ISSUER = os.getenv("JWT_ISSUER", "moodle")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "intelligent-lms")

def _parse_origins(value: str) -> List[str]:
    origins = [o.strip() for o in value.split(",") if o.strip()]
    return origins if origins else ["*"]

allowed_origins = _parse_origins(os.getenv("MOODLE_ALLOWED_ORIGINS", "*"))

# Add CORS middleware to allow Moodle (or any other domain) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    from services.hybrid_rag import initialize_neo4j_schema
    initialize_neo4j_schema()

class AuthContext(BaseModel):
    user_id: str
    username: str
    email: str
    firstname: str
    lastname: str
    fullname: str
    courseid: str
    courseshortname: str
    coursefullname: str
    role: str = "student"

def get_auth_context(request: Request) -> AuthContext:
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.query_params.get("token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token.")
    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT secret not configured.")
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            audience=JWT_AUDIENCE,
            issuer=JWT_ISSUER
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    required_fields = [
        "userid", "username", "email", "firstname", "lastname", "fullname",
        "courseid", "courseshortname", "coursefullname"
    ]
    missing_fields = [f for f in required_fields if f not in payload]
    if missing_fields:
        raise HTTPException(status_code=401, detail=f"Missing claims: {', '.join(missing_fields)}")

    role = str(payload.get("role", "student")).strip().lower()

    return AuthContext(
        user_id=str(payload["userid"]),
        username=str(payload["username"]),
        email=str(payload["email"]),
        firstname=str(payload["firstname"]),
        lastname=str(payload["lastname"]),
        fullname=str(payload["fullname"]),
        courseid=str(payload["courseid"]),
        courseshortname=str(payload["courseshortname"]),
        coursefullname=str(payload["coursefullname"]),
        role=role
    )

def ensure_user_course_graph(auth: AuthContext):
    from services.hybrid_rag import ensure_user_course_nodes
    ensure_user_course_nodes(
        user_id=auth.user_id,
        username=auth.username,
        email=auth.email,
        fullname=auth.fullname,
        courseid=auth.courseid,
        courseshortname=auth.courseshortname,
        coursefullname=auth.coursefullname
    )

# Global variables to simulate session state
# In production, use a database and session IDs
chat_histories = {}

class ChatRequest(BaseModel):
    message: str
    mode: str = "Auto"
    telemetry: Optional[TelemetryData] = None

class ChatResponse(BaseModel):
    reply: str
    missing_nodes: List[str]
    cognitive_state: str
    status: str = "ok"

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_in_course(auth.user_id, auth.courseid):
        return ChatResponse(
            reply="You are not enrolled in this course yet. Please enroll to access course materials.",
            missing_nodes=[],
            cognitive_state=CognitiveState.FOCUSED.value,
            status="not_enrolled"
        )
    user_history = chat_histories.setdefault(auth.user_id, [])

    # Append user message
    user_history.append(HumanMessage(content=request.message))
    
    # Calculate state dynamically if telemetry is provided
    if request.telemetry:
        req_state = analyze_student_state(request.telemetry)
    else:
        req_state = CognitiveState.FOCUSED
        
    # Run the LangGraph
    initial_state = {
        "messages": user_history,
        "mode": request.mode,
        "current_load_state": req_state,
        "retrieved_docs": [],
        "prerequisites": [],
        "matched_concepts": [],
        "graph_nodes": {},
        "missing_nodes": [],
        "courseid": auth.courseid
    }
    
    result = graph.invoke(initial_state)
    
    # Extract the AI's response and missing nodes
    ai_message = result["messages"][-1].content
    missing = result.get("missing_nodes", [])
    
    # Tag only the directly matched concepts + their prerequisites as LEARNING in Neo4j
    matched = result.get("matched_concepts", [])
    prereqs = result.get("prerequisites", [])
    learned_concepts = list(set(matched + prereqs))  # deduplicate
    tag_concepts_as_learning(auth.user_id, learned_concepts, auth.courseid)
    
    # Save the AI message to history
    user_history.append(result["messages"][-1])
    
    return ChatResponse(
        reply=ai_message,
        missing_nodes=missing,
        cognitive_state=req_state.value,
        status="ok"
    )

from services.tutoring import get_next_challenge, grade_answer, tag_concepts_as_learning, ChallengeGeneration, AssessmentResult
from services.hybrid_rag import get_enrolled_courses_with_counts, get_coordinator_courses_with_counts, is_coordinator_of_course, assign_coordinator_to_course

class ChallengeRequest(BaseModel):
    user_id: Optional[str] = None
    courseid: Optional[str] = None

class NotEnrolledResponse(BaseModel):
    status: str
    message: str

class CourseSummary(BaseModel):
    courseid: str
    courseshortname: Optional[str] = None
    coursefullname: Optional[str] = None
    enrolled_count: int

class CourseListResponse(BaseModel):
    status: str = "ok"
    courses: List[CourseSummary]

@app.post("/generate-challenge", response_model=Optional[Union[ChallengeGeneration, NotEnrolledResponse]])
async def generate_challenge_endpoint(request: ChallengeRequest, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_any(auth.user_id):
        return NotEnrolledResponse(
            status="not_enrolled",
            message="You are not enrolled in any course yet. Please enroll to access challenges."
        )
    course_id = getattr(request, "courseid", None)
    if course_id and not is_enrolled_in_course(auth.user_id, course_id):
        return NotEnrolledResponse(
            status="not_enrolled",
            message="You are not enrolled in that course yet. Please enroll to access course challenges."
        )
    return get_next_challenge(auth.user_id, course_id)

class GradeRequest(BaseModel):
    user_id: Optional[str] = None
    courseid: Optional[str] = None
    concept_name: str
    question: str
    student_answer: str

@app.post("/grade", response_model=Optional[Union[AssessmentResult, NotEnrolledResponse]])
async def grade_endpoint(request: GradeRequest, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_any(auth.user_id):
        return NotEnrolledResponse(
            status="not_enrolled",
            message="You are not enrolled in any course yet. Please enroll to access grading."
        )
    course_id = request.courseid
    if course_id and not is_enrolled_in_course(auth.user_id, course_id):
        return NotEnrolledResponse(
            status="not_enrolled",
            message="You are not enrolled in that course yet. Please enroll to access grading."
        )
    return grade_answer(auth.user_id, request.concept_name, request.question, request.student_answer, course_id)

@app.post("/telemetry")
async def telemetry_endpoint(data: TelemetryData, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_in_course(auth.user_id, auth.courseid):
        return {"status": "not_enrolled", "message": "You are not enrolled in this course yet. Please enroll to access telemetry."}
    calculated_state = analyze_student_state(data)
    return {"status": "success", "cognitive_state": calculated_state.value}

@app.post("/register")
async def register_endpoint(auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    from services.hybrid_rag import enroll_user_in_course
    enroll_user_in_course(auth.user_id, auth.courseid)
    coordinator_roles = {"coordinator", "teacher", "instructor", "editingteacher"}
    if auth.role in coordinator_roles:
        assign_coordinator_to_course(auth.user_id, auth.courseid)
    return {"status": "success"}

@app.get("/courses", response_model=CourseListResponse)
async def courses_endpoint(auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_any(auth.user_id):
        return CourseListResponse(status="ok", courses=[])
    courses = get_enrolled_courses_with_counts(auth.user_id)
    summaries = [
        CourseSummary(
            courseid=str(c.get("courseid", "")),
            courseshortname=c.get("courseshortname"),
            coursefullname=c.get("coursefullname"),
            enrolled_count=int(c.get("enrolled_count", 0))
        )
        for c in courses
        if c.get("courseid")
    ]
    return CourseListResponse(status="ok", courses=summaries)

@app.get("/courses/coordinator", response_model=CourseListResponse)
async def coordinator_courses_endpoint(auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    courses = get_coordinator_courses_with_counts(auth.user_id)
    summaries = [
        CourseSummary(
            courseid=str(c.get("courseid", "")),
            courseshortname=c.get("courseshortname"),
            coursefullname=c.get("coursefullname"),
            enrolled_count=int(c.get("enrolled_count", 0))
        )
        for c in courses
        if c.get("courseid")
    ]
    return CourseListResponse(status="ok", courses=summaries)

from services.hybrid_rag import ingest_document
from fastapi.concurrency import run_in_threadpool
from services.stats import get_student_stats, get_admin_stats, get_coordinator_stats
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    if not is_enrolled_in_course(auth.user_id, auth.courseid):
        return {"status": "not_enrolled", "message": "You are not enrolled in this course yet. Please enroll to upload materials."}
    if not file.filename.endswith(('.pdf', '.txt')):
        return {"error": "Only PDF and TXT files are supported."}
    
    content = await file.read()
    text = ""
    
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    else:
        text = content.decode('utf-8')
        
    # Ingest document synchronously in a thread pool so it doesn't block other users
    await run_in_threadpool(
        ingest_document,
        text,
        file.filename,
        {
            "courseid": auth.courseid,
            "courseshortname": auth.courseshortname,
            "coursefullname": auth.coursefullname
        }
    )
    
    return {"status": "success", "message": f"Document '{file.filename}' successfully processed."}

# Serve the static UI
import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/embed")
async def embed():
    return FileResponse("static/chat.html")

@app.get("/chat-ui")
async def chat_ui():
    return FileResponse("static/chat.html")

@app.get("/telemetry-ui")
async def telemetry_ui():
    return FileResponse("static/telemetry.html")

@app.get("/challenge-ui")
async def challenge_ui():
    return FileResponse("static/challenge.html")

@app.get("/upload-ui")
async def upload_ui():
    return FileResponse("static/upload.html")

@app.get("/token-ui")
async def token_ui():
    return FileResponse("static/token.html")

@app.get("/stats/student")
async def student_stats(courseid: Optional[str] = None, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    return get_student_stats(auth.user_id, courseid)

@app.get("/stats/admin")
async def admin_stats(auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    return get_admin_stats()

@app.get("/stats/coordinator")
async def coordinator_stats(courseid: Optional[str] = None, top_k: int = 5, auth: AuthContext = Depends(get_auth_context)):
    ensure_user_course_graph(auth)
    selected_course = courseid or auth.courseid
    if not is_coordinator_of_course(auth.user_id, selected_course):
        raise HTTPException(status_code=403, detail="Not authorized for coordinator stats.")
    return get_coordinator_stats(selected_course, top_k)

@app.get("/stats-student-ui")
async def stats_student_ui():
    return FileResponse("static/stats-student.html")

@app.get("/stats-admin-ui")
async def stats_admin_ui():
    return FileResponse("static/stats-admin.html")

@app.get("/stats-coordinator-ui")
async def stats_coordinator_ui():
    return FileResponse("static/stats-coordinator.html")
