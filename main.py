"""
AURA v2 — Main Application
Run:   uvicorn main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from services.analyze_service import analyze_document
from models.schemas import ExtractionResult

app = FastAPI(
    title       = "AURA — Adaptive Understanding and Retaining Agent",
    description = (
        "Upload any PDF, DOCX, or PPTX document.\n\n"
        "AURA extracts the **complete text**, detects all **topics and subtopics**, "
        "extracts **keywords** per section, and classifies every topic as "
        "**High / Medium / Low** exam priority."
    ),
    version = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health():
    return {"status": "AURA v2 is running ✅"}


@app.post(
    "/api/v1/analyze",
    response_model = ExtractionResult,
    tags           = ["Analysis"],
    summary        = "Upload document → get topics, subtopics, keywords, and priority",
)
async def analyze_document(file: UploadFile = File(...)):
    """
    ### Full AURA Pipeline

    Upload a **PDF, DOCX, or PPTX** and receive:

    - ✅ Complete text extraction (entire document, no truncation)
    - ✅ All **topics** detected from headings and structure
    - ✅ All **subtopics** nested under each topic
    - ✅ **Keywords** extracted per topic and subtopic
    - ✅ Each topic classified as **High / Medium / Low** exam priority
    - ✅ Word count per section

    Results are sorted: **High priority topics first**.
    """
    return await analyze(file)