"""
AI enhancer using NVIDIA NIM API (Llama 3.1 8B).
"""

import re
from openai import OpenAI

_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

_SECTION_RE = re.compile(
    r"\*{0,2}(SUMMARY|NOTES|QUESTIONS|MCQ|DIAGRAM|ACTIVE RECALL)\*{0,2}\s*:?\s*\n",
    re.IGNORECASE,
)


def _call_model(prompt: str, max_tokens: int = 4096) -> str:
    completion = _CLIENT.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=max_tokens,
        stream=True,
    )
    result = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices[0].delta.content is not None:
            result.append(chunk.choices[0].delta.content)
    return "".join(result)


def _parse_response(text: str) -> dict:
    result = {"summary": "", "notes": "", "questions": "", "mcq": "", "diagram": "", "active_recall": ""}
    parts = _SECTION_RE.split(text)
    i = 1
    while i < len(parts) - 1:
        header = parts[i].upper().strip()
        content = parts[i + 1].strip()
        if header == "SUMMARY":
            result["summary"] = content
        elif header == "NOTES":
            result["notes"] = content
        elif "QUESTION" in header and "RECALL" not in header:
            result["questions"] = content
        elif header == "MCQ":
            result["mcq"] = content
        elif header == "DIAGRAM":
            result["diagram"] = content
        elif "RECALL" in header:
            result["active_recall"] = content
        i += 2
    return result


def parse_mcqs(mcq_text: str) -> list[dict]:
    """Parse MCQ text into list of dicts."""
    mcqs = []
    if not mcq_text:
        return mcqs
    blocks = re.split(r"\n(?=Q:)", mcq_text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q_match = re.search(r"Q:\s*(.+)", block)
        a_match = re.search(r"A\)\s*(.+)", block)
        b_match = re.search(r"B\)\s*(.+)", block)
        c_match = re.search(r"C\)\s*(.+)", block)
        d_match = re.search(r"D\)\s*(.+)", block)
        ans_match = re.search(r"ANSWER:\s*([ABCD])", block, re.IGNORECASE)
        exp_match = re.search(r"EXPLANATION:\s*(.+)", block, re.DOTALL)
        if not (q_match and a_match and b_match and c_match and d_match and ans_match):
            continue
        options = [m.group(1).strip() for m in [a_match, b_match, c_match, d_match]]
        correct_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ans_match.group(1).upper(), 0)
        mcqs.append({
            "question": q_match.group(1).strip(),
            "options": options,
            "correct_index": correct_index,
            "explanation": exp_match.group(1).strip() if exp_match else "",
        })
    return mcqs


def parse_recall_prompts(active_recall_text: str) -> list[dict]:
    """Parse ACTIVE RECALL section into list of {question, expected} dicts."""
    prompts = []
    if not active_recall_text:
        return prompts
    blocks = re.split(r"\n(?=\d+\.)", active_recall_text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        expected_match = re.search(r"Expected:\s*(.+)", block, re.DOTALL | re.IGNORECASE)
        expected = expected_match.group(1).strip() if expected_match else ""
        question_text = re.sub(r"Expected:.*", "", block, flags=re.DOTALL | re.IGNORECASE)
        question_text = re.sub(r"^\d+\.\s*", "", question_text).strip()
        if question_text:
            prompts.append({"question": question_text, "expected": expected})
    return prompts


def generate_full_analysis(raw_text: str) -> dict:
    """Analyze document and produce structured summary, notes, questions, diagram, recall."""
    text_input = raw_text[:12000]
    if len(raw_text) > 12000:
        text_input += "\n[... document continues ...]"

    prompt = f"""You are an expert academic study assistant. Analyze the document and generate five sections with these exact headers on their own line:

SUMMARY
NOTES
QUESTIONS
DIAGRAM
ACTIVE RECALL

SUMMARY: Comprehensive topic-wise summary. Bold heading per topic (**Topic**). 4-6 sentences per topic. Cover all major topics (minimum 6).

NOTES: Topic-wise bullet notes. Bold heading per topic. 6-10 bullets per topic. Include definitions, examples, key terms.

QUESTIONS: 15 exam questions. 5 recall, 5 application, 5 analysis. Numbered.

DIAGRAM: Generate a process-based flowchart from the document content.
Use EXACTLY this format:
START: <starting concept or input>
STEP 1: <first process step>
STEP 2: <second process step>
STEP 3: <third process step>
STEP 4: <fourth process step>
DECISION: <condition or evaluation point>
YES -> <outcome if yes>
NO -> <outcome if no>
END: <final outcome or result>
Keep 6-8 nodes total. Show actual logical flow, NOT just topic names.

ACTIVE RECALL: 10 deep recall questions. Each numbered, followed by:
Expected: [detailed model answer 2-4 sentences]

Document:
{text_input}
"""
    try:
        text = _call_model(prompt, max_tokens=4096)
        return _parse_response(text)
    except Exception as e:
        return {"summary": "", "notes": "", "questions": "", "mcq": "", "diagram": "", "active_recall": "", "error": str(e)}


def generate_mcqs(raw_text: str) -> list[dict]:
    """Dedicated call to generate 10 MCQs from the document."""
    text_input = raw_text[:8000]
    prompt = f"""You are an exam question writer. Generate exactly 10 multiple-choice questions from the document below.

Use EXACTLY this format for every question with no extra text between questions:

Q: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [A or B or C or D]
EXPLANATION: [one sentence]

Generate all 10 now.

Document:
{text_input}
"""
    try:
        text = _call_model(prompt, max_tokens=3000)
        return parse_mcqs(text)
    except Exception:
        return []


def evaluate_recall_with_ai(question: str, expected: str, user_answer: str) -> dict:
    """Evaluate a user's recall answer with AI feedback."""
    prompt = f"""You are an academic evaluator. Evaluate the student's answer.

Question: {question}
Expected Answer: {expected}
Student's Answer: {user_answer}

Respond in exactly this format:
SCORE: [0 to 10]
FEEDBACK: [2-3 sentences: what was correct, what was missing, how to improve]
"""
    try:
        text = _call_model(prompt, max_tokens=300)
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", text)
        feedback_match = re.search(r"FEEDBACK:\s*(.+)", text, re.DOTALL)
        score = float(score_match.group(1)) / 10.0 if score_match else 0.5
        score = max(0.0, min(1.0, score))
        feedback = feedback_match.group(1).strip() if feedback_match else text.strip()
        return {"score": score, "is_correct": score >= 0.5, "feedback": feedback, "expected": expected}
    except Exception as e:
        return {"score": 0.0, "is_correct": False, "feedback": str(e), "expected": expected}


def generate_with_gemini(notes: list[str]) -> dict:
    """Fallback: generate from notes only."""
    notes_text = "\n".join(f"- {n}" for n in notes[:15])
    prompt = f"""You are an AI study assistant. Generate three sections with these exact headers:

QUESTIONS
DIAGRAM
ACTIVE RECALL

QUESTIONS: 15 exam questions, numbered.
DIAGRAM: Concept A -> Concept B -> Concept C (6-8 concepts)
ACTIVE RECALL: 10 recall questions, each followed by Expected: [model answer]

Notes:
{notes_text}
"""
    try:
        text = _call_model(prompt, max_tokens=3000)
        return _parse_response(text)
    except Exception as e:
        return {"questions": "", "diagram": "", "active_recall": "", "error": str(e)}


def diagram_to_graphviz(diagram_text: str) -> str:
    """Convert structured flowchart text into Graphviz DOT with proper shapes."""
    if not diagram_text or not diagram_text.strip():
        return ""

    nodes = []   # list of (id, label, shape, fillcolor)
    edges = []   # list of (from_id, to_id, edge_label)
    node_id = 0

    def _clean(text: str) -> str:
        return text.strip().replace('"', "'")[:50]

    def _add_node(label: str, shape: str = "box", color: str = "#AED6F1") -> str:
        nonlocal node_id
        nid = f"n{node_id}"
        node_id += 1
        nodes.append((nid, _clean(label), shape, color))
        return nid

    prev_id = None
    decision_id = None

    for raw_line in diagram_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        upper = line.upper()

        if upper.startswith("START:"):
            label = line.split(":", 1)[1].strip()
            nid = _add_node(label, shape="ellipse", color="#82E0AA")
            if prev_id:
                edges.append((prev_id, nid, ""))
            prev_id = nid

        elif upper.startswith("END:"):
            label = line.split(":", 1)[1].strip()
            nid = _add_node(label, shape="ellipse", color="#F1948A")
            if prev_id:
                edges.append((prev_id, nid, ""))
            prev_id = nid

        elif upper.startswith("STEP"):
            label = line.split(":", 1)[1].strip() if ":" in line else line
            nid = _add_node(label, shape="box", color="#AED6F1")
            if prev_id:
                edges.append((prev_id, nid, ""))
            prev_id = nid

        elif upper.startswith("DECISION:"):
            label = line.split(":", 1)[1].strip()
            nid = _add_node(label, shape="diamond", color="#F9E79F")
            if prev_id:
                edges.append((prev_id, nid, ""))
            decision_id = nid
            prev_id = nid

        elif upper.startswith("YES") and "->" in line:
            label = line.split("->", 1)[1].strip()
            nid = _add_node(label, shape="box", color="#ABEBC6")
            if decision_id:
                edges.append((decision_id, nid, "Yes"))
            prev_id = nid

        elif upper.startswith("NO") and "->" in line:
            label = line.split("->", 1)[1].strip()
            nid = _add_node(label, shape="box", color="#F5B7B1")
            if decision_id:
                edges.append((decision_id, nid, "No"))

        elif "->" in line:
            # Fallback: legacy arrow-based format
            parts = line.split("->")
            for p in parts:
                p = p.strip().strip("*").strip()
                if p:
                    nid = _add_node(p)
                    if prev_id:
                        edges.append((prev_id, nid, ""))
                    prev_id = nid

    if not nodes:
        return ""

    dot = [
        "digraph G {",
        "    rankdir=TB;",
        "    bgcolor=transparent;",
        '    node [style=\"filled,rounded\" fontname=\"Helvetica\" fontsize=11 margin=\"0.2,0.1\"];',
        '    edge [color=\"#555555\" penwidth=1.5 fontname=\"Helvetica\" fontsize=9];',
    ]
    for nid, label, shape, color in nodes:
        dot.append(f'    {nid} [label="{label}" shape={shape} fillcolor="{color}"];')
    for from_id, to_id, elabel in edges:
        if elabel:
            dot.append(f'    {from_id} -> {to_id} [label="{elabel}"];')
        else:
            dot.append(f"    {from_id} -> {to_id};")
    dot.append("}")
    return "\n".join(dot)


def smart_answer(context: str, question: str, strict_mode: bool = False) -> str:
    """Hybrid question answering for documents."""
    mode_instructions = ""
    if strict_mode:
        mode_instructions = """
1. You MUST answer the question using ONLY the provided document.
2. If the answer is NOT present in the document, reply exactly with: "I'm sorry, I cannot answer this strictly from the document. Please turn off Strict Document Mode for a general explanation."
"""
    else:
        mode_instructions = """
1. If the answer is clearly present in the document:
   -> Start your response exactly with [DOC_BASED]
   -> Answer based on the document.
2. If the answer is NOT present:
   -> Start your response exactly with [GEN_KNOWLEDGE]
   -> Provide a clear general explanation.
   -> Include a simple real-world example.

Ensure your response is student-friendly and has this structure:
- Short Answer
- Explanation
- Example (if general knowledge)
"""

    prompt = f"""You are an intelligent academic assistant.
Given: Document content and a user question.

Instructions:
{mode_instructions}

Document:
{context[:12000]}

Question:
{question}
"""
    try:
        text = _call_model(prompt, max_tokens=1500)
        return text.strip()
    except Exception as e:
        return f"Error connecting to AI assistant: {e}"
