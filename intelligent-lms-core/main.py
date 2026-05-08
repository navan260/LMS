from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import PyPDF2

from dotenv import load_dotenv
load_dotenv()

from core.orchestrator import graph
from services.telemetry_ml import analyze_student_state, TelemetryData, CognitiveState
from langchain_core.messages import HumanMessage

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Intelligent Agentic LMS")

# Add CORS middleware to allow Moodle (or any other domain) to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact Moodle domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    from services.hybrid_rag import initialize_neo4j_schema
    initialize_neo4j_schema()

# Global variables to simulate session state
# In production, use a database and session IDs
chat_history = []

class ChatRequest(BaseModel):
    message: str
    mode: str = "Auto"
    telemetry: Optional[TelemetryData] = None

class ChatResponse(BaseModel):
    reply: str
    missing_nodes: List[str]
    cognitive_state: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global chat_history
    
    # Append user message
    chat_history.append(HumanMessage(content=request.message))
    
    # Calculate state dynamically if telemetry is provided
    if request.telemetry:
        req_state = analyze_student_state(request.telemetry)
    else:
        req_state = CognitiveState.FOCUSED
        
    # Run the LangGraph
    initial_state = {
        "messages": chat_history,
        "mode": request.mode,
        "current_load_state": req_state,
        "retrieved_docs": [],
        "prerequisites": [],
        "graph_nodes": {},
        "missing_nodes": []
    }
    
    result = graph.invoke(initial_state)
    
    # Extract the AI's response and missing nodes
    ai_message = result["messages"][-1].content
    missing = result.get("missing_nodes", [])
    
    # Save the AI message to history
    chat_history.append(result["messages"][-1])
    
    return ChatResponse(
        reply=ai_message,
        missing_nodes=missing,
        cognitive_state=req_state.value
    )

from services.tutoring import get_next_challenge, grade_answer, ChallengeGeneration, AssessmentResult

class ChallengeRequest(BaseModel):
    user_id: str = "DEV_GHOST_USER"

@app.post("/generate-challenge", response_model=Optional[ChallengeGeneration])
async def generate_challenge_endpoint(request: ChallengeRequest):
    return get_next_challenge(request.user_id)

class GradeRequest(BaseModel):
    user_id: str = "DEV_GHOST_USER"
    concept_name: str
    question: str
    student_answer: str

@app.post("/grade", response_model=Optional[AssessmentResult])
async def grade_endpoint(request: GradeRequest):
    return grade_answer(request.user_id, request.concept_name, request.question, request.student_answer)

@app.post("/telemetry")
async def telemetry_endpoint(data: TelemetryData):
    calculated_state = analyze_student_state(data)
    return {"status": "success", "cognitive_state": calculated_state.value}

from services.hybrid_rag import ingest_document
from fastapi.concurrency import run_in_threadpool
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
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
    await run_in_threadpool(ingest_document, text, file.filename)
    
    return {"status": "success", "message": f"Document '{file.filename}' successfully processed."}

# Serve the static UI
import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

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
