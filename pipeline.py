"""
Pipeline: orchestrates all stages from PDF bytes to StudyMaterial.

Requirements: 3.3, 3.4, 5.6, 8.2, 9.1
"""

from pdf_extractor import extract_text, clean_text
from summarizer import load_summarizer, summarize
from notes_generator import generate_notes
from question_generator import load_question_generator, generate_questions
from mcq_builder import build_mcqs
from diagram_generator import generate_diagram
from active_recall import build_recall_prompts
from models import StudyMaterial


def run_pipeline(pdf_bytes: bytes) -> StudyMaterial:
    """Run the full study material pipeline on PDF bytes.

    Steps:
        1. Extract and clean text from PDF
        2. Summarize with BART
        3. Generate bullet-point notes
        4. Generate Bloom-tagged QA pairs with T5
        5. Build MCQs (skipped if fewer than 4 QA pairs)
        6. Generate Mermaid diagram
        7. Build active recall prompts

    Returns:
        StudyMaterial dataclass with all generated artifacts.
    """
    # Step 1: Extract
    raw = extract_text(pdf_bytes)
    clean = clean_text(raw)
    assert len(clean) > 0, "No text extracted from PDF"

    # Step 2: Summarize
    summarizer = load_summarizer()
    summary = summarize(clean, summarizer)
    assert len(summary) > 0

    # Step 3: Notes
    notes = generate_notes(summary)
    assert len(notes) >= 1

    # Step 4: Questions
    qg = load_question_generator()
    qa_pairs = generate_questions(notes, qg)
    assert all("question" in q and "answer" in q for q in qa_pairs)

    # Step 5: MCQs (req 5.6: skip if fewer than 4 QA pairs)
    mcqs = build_mcqs(qa_pairs) if len(qa_pairs) >= 4 else []

    # Step 6: Diagram
    diagram = generate_diagram(notes)

    # Step 7: Active Recall
    prompts = build_recall_prompts(notes, qa_pairs)

    return StudyMaterial(summary, notes, qa_pairs, mcqs, diagram, prompts)
