# AURA  — Adaptive Understanding and Retaining Agent

AURA is an AI-powered academic study assistant that transforms raw PDF documents
into structured, adaptive study material. Upload any textbook chapter, research
paper, or lecture notes and the system produces summaries, key notes, Bloom
taxonomy-aligned questions, multiple-choice quizzes, active recall sessions,
concept maps, and an interactive Q&A interface — all within a single application.

---

## The Problem

Students waste significant time manually re-reading dense documents to extract
what matters. Passive re-reading is one of the least effective study strategies.
What actually works — retrieval practice, spaced repetition, and elaborative
interrogation — requires generating the right questions and prompts from the
material, which is time-consuming to do by hand.

Existing tools either summarize blindly (no pedagogical structure) or ask generic
questions unrelated to the specific document. None adapt difficulty to the
student's actual performance.

---

## What AURA Solves

```
  Raw PDF / Document
         |
         v
  +------+----------+
  |  Text Extraction |   <- pdfplumber, clean-text pipeline
  +------+----------+
         |
         v
  +------+----------+
  |  Local ML Models |   <- BART (summarization), NLTK (chunking)
  +------+----------+
         |
         v
  +------+-------------------+
  |  NVIDIA NIM (Llama 3.1)  |   <- AI analysis, MCQs, recall, concept map
  +------+-------------------+
         |
         v
  +------+---------------------+
  |  Adaptive Study Interface  |   <- Bloom levels, scoring, remediation paths
  +----------------------------+
```

- Extracts and cleans text from academic PDFs automatically
- Summarizes using a fine-tuned BART model locally (no API required for this step)
- Augments with Llama 3.1 (via NVIDIA NIM) for richer analysis, MCQs, and Q&A
- Generates questions mapped to Bloom's Taxonomy levels (Remember through Evaluate)
- Adapts difficulty dynamically based on student answer scores
- Detects misconceptions (high confidence + wrong answer) and routes to remediation
- Builds concept dependency graphs to trace root causes of knowledge gaps
- Renders an interactive concept flowchart from document content

---

## Architecture

```
aura/
  app.py                  Main Streamlit application (all UI pages)
  main.py                 Entry point
  requirements.txt        Python dependencies

  -- Document Processing --
  pdf_extractor.py        Extract and clean raw text from PDF bytes
  summarizer.py           Load and run BART summarization model
  notes_generator.py      Convert summary to bullet-point key notes
  pipeline.py             Orchestration helpers

  -- Question and Assessment --
  question_generator.py   Base question generation utilities
  question_service.py     Bloom-taxonomy aligned question sets
  mcq_builder.py          Convert Q/A pairs into 4-option MCQs
  active_recall.py        Build and evaluate spaced recall prompts

  -- AI Layer --
  gemini_enhancer.py      NVIDIA NIM (Llama 3.1) calls: full analysis,
                          AI MCQs, diagram generation, smart Q&A, recall eval
  topic_enhancer.py       Per-topic deep-dive generation via API
  notes_service.py        Structured notes (summary, key points, definitions)

  -- Adaptive Engine --
  adaptive_engine.py      Bloom-level progression, confidence-weighted scoring,
                          misconception detection
  topic_graph.py          Prerequisite dependency graph, root-cause tracing,
                          cross-topic transfer prediction
  remediation_service.py  Generate targeted remediation paths for weak topics

  -- Models and Utils --
  models/
    schemas.py            Pydantic data shapes
  utils/
    text_utils.py         Cleaning, chunking, heading detection
  evaluate_models.py      ROUGE / BLEU evaluation against fine-tuned checkpoints
  train_bart.py           Fine-tune BART on AURA dataset
  train_t5.py             Fine-tune Flan-T5 on AURA dataset
```

---

## Application Pages

| Page | Description |
|------|-------------|
| Document Analysis | Upload PDF, run full pipeline, view extraction stats |
| Notes and Summary | BART summary, AI-enhanced summary, per-topic note generation |
| Questions and MCQs | Bloom-taxonomy Q&A, local MCQs, AI-generated MCQs with explanations |
| Active Recall | Timed recall prompts with local or AI-based scoring and feedback |
| Concept Map | Auto-generated Graphviz flowchart from document logic |
| Smart Q&A | Chat interface over document; strict mode or general knowledge mode |
| Topic Explorer | Per-topic deep-dive notes at beginner, intermediate, or advanced level |
| Performance | Score history, Bloom level progression, misconception flags |

---

## Requirements

- Python 3.10 or higher
- NVIDIA NIM API key (free tier available at https://integrate.api.nvidia.com)
- A CUDA-capable GPU is recommended for local BART inference but not required

---

## Setup

**1. Clone and enter the project directory**

```
git clone https://github.com/your-username/aura.git
cd aura
```

**2. Create and activate a virtual environment**

```
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**

```
pip install -r requirements.txt
```

**4. Configure the API key**

Open `gemini_enhancer.py` and replace the `api_key` value with your NVIDIA NIM key:

```python
_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="your-nvapi-key-here",
)
```

**5. Run the application**

```
streamlit run app.py
```

The application opens at `http://localhost:8501` by default.

---

## Usage

**Standard workflow:**

```
1. Open the application in the browser.
2. Navigate to "Document Analysis" in the sidebar.
3. Upload a PDF (text-based; scanned PDFs are not supported).
4. Click "Analyze Document".
5. Wait for the pipeline to complete (progress bar shown).
6. Navigate to any page in the sidebar to use the generated material.
```

**Pipeline steps on document upload:**

```
  [1] Extract text from PDF                (pdfplumber)
  [2] Clean and chunk text                 (NLTK, text_utils)
  [3] Summarize locally                    (BART model)
  [4] Generate key notes                   (from BART summary)
  [5] Generate Bloom questions             (question_service, Llama 3.1)
  [6] Build MCQs from Q/A pairs            (mcq_builder)
  [7] Build recall prompts                 (active_recall)
  [8] Run full AI analysis                 (Llama 3.1 via NVIDIA NIM)
  [9] Generate AI MCQs with explanations   (Llama 3.1)
  [10] Extend recall prompts from AI       (Llama 3.1)
```

---

## Adaptive Difficulty System

The adaptive engine adjusts question difficulty based on student performance
using Bloom's Taxonomy as the progression axis.

```
  Score >= 0.8  -->  Advance to next Bloom level
  Score  0.5    -->  Stay at current Bloom level
  Score < 0.5   -->  Drop to previous Bloom level

  Bloom levels: Remember -> Understand -> Apply -> Analyze -> Evaluate
```

**Confidence-weighted scoring:**

```
  Final Score = 0.7 * Correctness + 0.3 * (Confidence / 5)
```

Where `Correctness` is 1 if correct, 0 if wrong, and `Confidence` is a 1-5
self-reported rating. A wrong answer with high confidence (>= 4) is flagged as
a misconception and triggers a targeted remediation path.

---

## Concept Dependency Graph

`topic_graph.py` maintains a prerequisite graph for common academic topics.
When a student scores poorly on a topic, the system traces backwards through
the dependency chain to find the root knowledge gap:

```
  Failed topic: Red Black Tree
    Prerequisites: BST, Recursion
      BST requires: Recursion, Pointers
        Recursion requires: Functions, Stack
  Root cause identified: Stack (score 0.2)
  Remediation: study Stack -> Recursion -> BST -> Red Black Tree
```

---

## Supported File Types

| Format | Support |
|--------|---------|
| PDF (text-based) | Supported |
| DOCX | Not yet (planned) |
| PPTX | Not yet (planned) |
| Scanned PDFs | Not supported |

---

## Model Details

| Component | Model | Where it runs |
|-----------|-------|---------------|
| Summarization | BART (fine-tunable) | Local (CPU or GPU) |
| Question generation | Llama 3.1 8B Instruct | NVIDIA NIM API |
| MCQ generation | Llama 3.1 8B Instruct | NVIDIA NIM API |
| Answer evaluation | Llama 3.1 8B Instruct | NVIDIA NIM API |
| Concept map | Llama 3.1 8B Instruct + Graphviz | NVIDIA NIM API + local render |

---

## Running Tests

```
pytest test_summarizer.py
pytest test_mcq_builder.py
pytest test_question_generator.py
pytest test_active_recall.py
pytest test_pdf_extractor.py
pytest test_pipeline.py
```

All test modules use `hypothesis` for property-based testing in addition to
standard unit tests.

---

## Dataset Collection (Optional)

The `AURA_dataset/` directory contains scrapers to build a fine-tuning dataset
from OpenStax textbooks and Wikipedia articles in the Alpaca instruction format.

```
python AURA_dataset/scrapers/openstax_scraper.py   # ~20-30 min
python AURA_dataset/scrapers/wikipedia_scraper.py  # ~5-10 min
python AURA_dataset/scrapers/formatter.py          # produces aura_dataset.jsonl
```

Each training example has the structure:

```json
{
  "instruction": "You are an expert tutor...",
  "input": "Topic: Machine Learning\n\nContent: ...",
  "output": "...",
  "task": "explanation",
  "topic": "Machine Learning"
}
```

Use `train_bart.py` or `train_t5.py` to fine-tune locally or on Google Colab.

---

## License

This project is part of a B.Tech Minor Research Project.
See individual source files for attribution where applicable.
