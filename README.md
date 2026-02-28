# AURA — Setup & Run (Under 10 Minutes)

## FILE STRUCTURE
```
AURA/
├── main.py                   ← Start here. All API routes.
├── requirements.txt          ← All dependencies
├── README.md
├── models/
│   ├── __init__.py
│   └── schemas.py            ← Data shapes (inputs/outputs)
├── utils/
│   ├── __init__.py
│   └── text_utils.py         ← Cleaning, chunking, heading detection
└── services/
    ├── __init__.py
    ├── doc_service.py        ← Feature #1 logic
    └── concept_service.py    ← Feature #2 logic
```

---

## STEP 1 — Create Virtual Environment
```bash
cd AURA
python -m venv venv
```

Activate it:
- Windows:  `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

---

## STEP 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

---

## STEP 3 — Run the Server
```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

---

## STEP 4 — Open API Docs (Your Demo UI)
```
http://localhost:8000/docs
```

This opens Swagger UI — a built-in interactive interface to test all endpoints.

---

## ENDPOINTS

| Method | URL | What it does |
|--------|-----|--------------|
| GET  | `/health` | Check server is running |
| POST | `/api/v1/document/upload` | Feature #1 — Extract & clean text |
| POST | `/api/v1/concepts/extract` | Feature #2 — Extract & prioritize concepts |
| POST | `/api/v1/analyze` | Both features combined (use for demo) |

---

## FOR YOUR DEMO — Use `/api/v1/analyze`

1. Open `http://localhost:8000/docs`
2. Click on `POST /api/v1/analyze`
3. Click **"Try it out"**
4. Click **"Choose File"** → upload any textbook PDF or DOCX
5. Click **"Execute"**
6. Show the response:
   - Summary (word count, chunk count, concept counts)
   - `high_priority` concepts → these are your exam topics
   - `medium_priority` and `low_priority` below

---

## SUPPORTED FILE TYPES
- ✅ PDF (must be text-based, not scanned)
- ✅ DOCX (Word documents)
- ✅ PPTX (PowerPoint presentations)