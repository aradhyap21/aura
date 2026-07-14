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

# AURA Dataset Collection

## Folder Structure

```
AURA_dataset/
├── requirements.txt
├── scrapers/
│   ├── openstax_scraper.py    ← scrapes OpenStax textbooks
│   ├── wikipedia_scraper.py   ← scrapes Wikipedia articles
│   └── formatter.py           ← cleans and formats into training data
└── data/                      ← created automatically
    ├── openstax_raw.json
    ├── wikipedia_raw.json
    ├── aura_dataset.jsonl     ← FINAL: ready for fine-tuning
    ├── aura_dataset.json      ← human-readable version
    └── stats.json             ← collection statistics
```

## Setup

```bash
pip install requests beautifulsoup4 wikipedia-api pandas tqdm nltk
```

## Run in Order

### Step 1 — Scrape OpenStax
```bash
python scrapers/openstax_scraper.py
```
- Scrapes 10 textbooks
- Extracts all chapter sections
- Saves to data/openstax_raw.json
- Takes about 20-30 minutes

### Step 2 — Scrape Wikipedia
```bash
python scrapers/wikipedia_scraper.py
```
- Scrapes 60+ educational topics
- Extracts all article sections
- Saves to data/wikipedia_raw.json
- Takes about 5-10 minutes

### Step 3 — Format into Training Data
```bash
python scrapers/formatter.py
```
- Cleans all raw data
- Removes duplicates
- Creates 3 training examples per topic:
  1. Simplified explanation task
  2. Exam bullet points task
  3. Real-world analogy task
- Saves final JSONL ready for fine-tuning

## Expected Output

| Source    | Raw Examples | After Cleaning |
|-----------|-------------|----------------|
| OpenStax  | ~1500       | ~800           |
| Wikipedia | ~800        | ~600           |
| **Total** | **~2300**   | **~4200 training examples** |

(3 tasks × ~1400 unique topics = ~4200 training examples)

## Dataset Format (Alpaca Format)

Each line in aura_dataset.jsonl looks like:
```json
{
  "instruction": "You are an expert tutor...",
  "input": "Topic: Machine Learning\n\nContent: ...",
  "output": "...",
  "task": "explanation",
  "topic": "Machine Learning"
}
```

## Next Step After Collection
Hand the aura_dataset.jsonl to the fine-tuning script
to train Mistral 7B or Phi-3 on Google Colab.