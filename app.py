"""
AURA 2 — Adaptive Understanding & Retaining Agent
Full Streamlit application with AI-powered study material generation.
"""

import streamlit as st
import time
import copy

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA 2 — AI Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #141233 30%, #1a1a3e 60%, #24243e 100%);
    color: #e0e0ff;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.97) 0%, rgba(20,18,51,0.97) 100%);
    border-right: 1px solid rgba(99,102,241,0.15);
}
[data-testid="stSidebar"] .stRadio label {
    color: #c4c4f0 !important;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 8px;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(99,102,241,0.15);
}

/* Glass cards */
.glass-card {
    background: rgba(30, 30, 72, 0.5);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s, box-shadow 0.2s;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.15);
}

/* Hero */
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0;
    animation: fadeInUp 0.8s ease-out;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: #a0a0d0;
    text-align: center;
    margin-top: 4px;
    margin-bottom: 32px;
    animation: fadeInUp 0.8s ease-out 0.15s both;
}

/* Section titles */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(167,139,250,0.3);
}

/* Stat card */
.stat-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #818cf8;
}
.stat-label {
    font-size: 0.85rem;
    color: #8888bb;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* MCQ option buttons */
.mcq-option {
    background: rgba(30,30,72,0.6);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 12px 16px;
    color: #c4c4f0;
    cursor: pointer;
    transition: all 0.2s;
    margin: 6px 0;
}
.mcq-option:hover {
    border-color: rgba(99,102,241,0.5);
    background: rgba(99,102,241,0.15);
}

/* Score badges */
.badge-correct { color: #4ade80; font-weight: 600; }
.badge-incorrect { color: #f87171; font-weight: 600; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(15,12,41,0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #a0a0d0;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #818cf8 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(30,30,72,0.4);
    border-radius: 10px;
    color: #c4c4f0 !important;
    font-weight: 600;
}

/* Info/success/warning boxes */
.stAlert { border-radius: 12px !important; }

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.loading-text {
    animation: pulse 1.5s infinite;
    color: #818cf8;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy imports (expensive) ─────────────────────────────────────────────────

def _import_pipeline_modules():
    from pdf_extractor import extract_text, clean_text
    from summarizer import load_summarizer, summarize
    from notes_generator import generate_notes
    from mcq_builder import build_mcqs
    from active_recall import build_recall_prompts, evaluate_answer
    return {
        "extract_text": extract_text,
        "clean_text": clean_text,
        "load_summarizer": load_summarizer,
        "summarize": summarize,
        "generate_notes": generate_notes,
        "build_mcqs": build_mcqs,
        "build_recall_prompts": build_recall_prompts,
        "evaluate_answer": evaluate_answer,
    }


def _import_ai_modules():
    from gemini_enhancer import (
        generate_full_analysis, generate_mcqs as ai_mcqs,
        parse_mcqs, parse_recall_prompts,
        evaluate_recall_with_ai, smart_answer,
        diagram_to_graphviz,
    )
    from topic_enhancer import process_document_with_topics
    from notes_service import generate_topic_notes
    from question_service import (
        generate_bloom_questions, generate_mcqs as topic_mcqs,
        generate_recall_questions,
    )
    from adaptive_engine import (
        next_bloom_level, bloom_level_name,
        confidence_weighted_score, is_misconception,
    )
    from remediation_service import generate_remediation_path
    from topic_graph import TopicGraph
    return {
        "generate_full_analysis": generate_full_analysis,
        "ai_mcqs": ai_mcqs,
        "parse_mcqs": parse_mcqs,
        "parse_recall_prompts": parse_recall_prompts,
        "evaluate_recall_with_ai": evaluate_recall_with_ai,
        "smart_answer": smart_answer,
        "diagram_to_graphviz": diagram_to_graphviz,
        "process_document_with_topics": process_document_with_topics,
        "generate_topic_notes": generate_topic_notes,
        "generate_bloom_questions": generate_bloom_questions,
        "topic_mcqs": topic_mcqs,
        "generate_recall_questions": generate_recall_questions,
        "next_bloom_level": next_bloom_level,
        "bloom_level_name": bloom_level_name,
        "confidence_weighted_score": confidence_weighted_score,
        "is_misconception": is_misconception,
        "generate_remediation_path": generate_remediation_path,
        "TopicGraph": TopicGraph,
    }


# ── Session State Defaults ───────────────────────────────────────────────────
# NOTE: Use copy.deepcopy when resetting to avoid shared mutable references.

DEFAULTS = {
    "raw_text": "",
    "clean_text_data": "",
    "analysis": None,
    "local_summary": "",
    "local_notes": [],
    "local_qa": [],
    "local_mcqs": [],
    "recall_prompts": [],
    "ai_mcqs_list": [],
    "processing_done": False,
    "active_page": "📄 Document Analysis",
    "bloom_level": 1,
    "topic_scores": {},
    "chat_history": [],
    "strict_mode": False,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = copy.deepcopy(default)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.8rem;text-align:left;">🧠 AURA 2</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8888bb;font-size:0.85rem;margin-top:-8px;">Adaptive Understanding & Retaining Agent</p>', unsafe_allow_html=True)
    st.markdown("---")

    pages = [
        "📄 Document Analysis",
        "📝 Notes & Summary",
        "❓ Questions & MCQs",
        "🎯 Active Recall",
        "🗺️ Concept Map",
        "💬 Smart Q&A",
        "📚 Topic Explorer",
        "📊 Performance",
    ]
    selected = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.active_page = selected

    if st.session_state.processing_done:
        st.markdown("---")
        st.success("✅ Document processed")
        stats_text = ""
        if st.session_state.local_notes:
            stats_text += f"**{len(st.session_state.local_notes)}** notes  \n"
        if st.session_state.local_qa:
            stats_text += f"**{len(st.session_state.local_qa)}** questions  \n"
        if st.session_state.local_mcqs:
            stats_text += f"**{len(st.session_state.local_mcqs)}** MCQs  \n"
        if st.session_state.ai_mcqs_list:
            stats_text += f"**{len(st.session_state.ai_mcqs_list)}** AI MCQs  \n"
        if st.session_state.recall_prompts:
            stats_text += f"**{len(st.session_state.recall_prompts)}** recall prompts"
        if stats_text:
            st.markdown(stats_text)


# ── Helper: Run full pipeline ────────────────────────────────────────────────

def run_full_pipeline(pdf_bytes: bytes):
    """Process uploaded PDF through both local ML models and AI API."""
    mods = _import_pipeline_modules()

    progress = st.progress(0, text="Extracting text from PDF...")
    raw = mods["extract_text"](pdf_bytes)
    clean = mods["clean_text"](raw)
    st.session_state.raw_text = raw
    st.session_state.clean_text_data = clean
    progress.progress(15, text="Text extracted ✓")
    time.sleep(0.3)

    # Local BART summarization
    progress.progress(20, text="Loading BART summarizer...")
    summarizer = mods["load_summarizer"]()
    progress.progress(30, text="Summarizing with BART...")
    summary = mods["summarize"](clean, summarizer)
    st.session_state.local_summary = summary
    progress.progress(40, text="Summary generated ✓")

    # Notes from summary
    progress.progress(45, text="Generating notes...")
    notes = mods["generate_notes"](summary)
    st.session_state.local_notes = notes
    progress.progress(50, text=f"{len(notes)} notes generated ✓")

    # Question generation with API (Bloom)
    progress.progress(55, text="Generating Bloom questions via API...")
    try:
        from question_service import generate_bloom_questions
        topic_text = summary[:1000] if summary else "Study topic"
        qa_pairs = []
        qa_pairs.extend(generate_bloom_questions(topic_text, bloom_level=1, count=5))
        qa_pairs.extend(generate_bloom_questions(topic_text, bloom_level=2, count=5))
        qa_pairs.extend(generate_bloom_questions(topic_text, bloom_level=3, count=5))
        st.session_state.local_qa = qa_pairs
        progress.progress(70, text=f"{len(qa_pairs)} questions generated ✓")
    except Exception as e:
        qa_pairs = []
        st.session_state.local_qa = []
        progress.progress(70, text="Question generation skipped/failed")

    # MCQs
    progress.progress(72, text="Building MCQs...")
    mcqs = mods["build_mcqs"](qa_pairs) if len(qa_pairs) >= 4 else []
    st.session_state.local_mcqs = mcqs
    progress.progress(75, text="MCQs ready ✓")

    # Recall prompts
    progress.progress(82, text="Building recall prompts...")
    prompts = mods["build_recall_prompts"](notes, qa_pairs)
    st.session_state.recall_prompts = prompts
    progress.progress(85, text="Recall prompts ready ✓")

    # AI-enhanced analysis
    progress.progress(87, text="Running AI analysis (NVIDIA NIM)...")
    try:
        ai_mods = _import_ai_modules()
        analysis = ai_mods["generate_full_analysis"](raw)
        st.session_state.analysis = analysis

        # AI MCQs
        progress.progress(92, text="Generating AI MCQs...")
        ai_mcqs_data = ai_mods["ai_mcqs"](raw)
        st.session_state.ai_mcqs_list = ai_mcqs_data

        # AI recall prompts
        if analysis.get("active_recall"):
            ai_recall = ai_mods["parse_recall_prompts"](analysis["active_recall"])
            if ai_recall:
                st.session_state.recall_prompts.extend(
                    [{"prompt": r["question"], "expected": r["expected"], "prompt_type": "qa"} for r in ai_recall]
                )

        progress.progress(97, text="AI analysis complete ✓")
    except Exception as e:
        st.warning(f"AI analysis partially failed: {e}")
        progress.progress(97, text="AI analysis skipped")

    st.session_state.processing_done = True
    progress.progress(100, text="✅ All processing complete!")
    time.sleep(0.5)
    progress.empty()


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

page = st.session_state.active_page

# ── 1. Document Analysis ─────────────────────────────────────────────────────
if page == "📄 Document Analysis":
    st.markdown('<h1 class="hero-title">🧠 AURA 2</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Upload a PDF and let AI transform it into structured study materials</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-value">📄</div><div class="stat-label">PDF → Text</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-value">🤖</div><div class="stat-label">BART + T5</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card"><div class="stat-value">🧬</div><div class="stat-label">Llama 3.1</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="stat-card"><div class="stat-value">🎯</div><div class="stat-label">Bloom Taxonomy</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    uploaded = st.session_state.get("uploaded_file")

    if uploaded:
        st.info(f"📄 **{uploaded['name']}** ({uploaded['size'] / 1024:.1f} KB)")

        if not st.session_state.processing_done:
            if st.button("🚀 Analyze Document", use_container_width=True):
                try:
                    pdf_bytes = uploaded['bytes']
                    run_full_pipeline(pdf_bytes)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Processing error: {e}")
        else:
            st.success("✅ Document has been processed! Navigate using the sidebar.")

            # Summary statistics
            st.markdown('<div class="section-title">📊 Processing Results</div>', unsafe_allow_html=True)
            cols = st.columns(5)
            items = [
                ("Notes", len(st.session_state.local_notes)),
                ("Questions", len(st.session_state.local_qa)),
                ("MCQs", len(st.session_state.local_mcqs) + len(st.session_state.ai_mcqs_list)),
                ("Recall", len(st.session_state.recall_prompts)),
                ("AI Sections", sum(1 for v in (st.session_state.analysis or {}).values() if v and v != "")),
            ]
            for col, (label, val) in zip(cols, items):
                with col:
                    st.markdown(f'<div class="stat-card"><div class="stat-value">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

            # Raw text preview
            with st.expander("📃 Extracted Text Preview"):
                st.text(st.session_state.raw_text[:3000] + ("..." if len(st.session_state.raw_text) > 3000 else ""))

            if st.button("🔄 Re-process Document"):
                for key in DEFAULTS:
                    st.session_state[key] = copy.deepcopy(DEFAULTS[key])
                if "uploaded_file" in st.session_state:
                    del st.session_state["uploaded_file"]
                if "recall_idx" in st.session_state:
                    del st.session_state["recall_idx"]
                if "recall_scores" in st.session_state:
                    del st.session_state["recall_scores"]
                st.rerun()
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding: 48px;">
            <div style="font-size: 4rem; margin-bottom: 16px;">📄</div>
            <h3 style="color: #a78bfa;">Upload a PDF to get started</h3>
            <p style="color: #8888bb;">Drag & drop your file below to begin</p>
        </div>
        """, unsafe_allow_html=True)

        direct_upload = st.file_uploader("Drop your PDF here", type=["pdf"], key="main_uploader")
        if direct_upload:
            st.session_state["uploaded_file"] = {
                "name": direct_upload.name,
                "size": direct_upload.size,
                "bytes": direct_upload.getvalue()
            }
            st.rerun()


# ── 2. Notes & Summary ──────────────────────────────────────────────────────
elif page == "📝 Notes & Summary":
    st.markdown('<div class="section-title">📝 Notes & Summary</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        tab1, tab2, tab3 = st.tabs(["🤖 Local Summary (BART)", "🧬 AI Summary", "📋 Structured Notes"])

        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### BART-Generated Summary")
            st.write(st.session_state.local_summary or "_No summary available._")
            st.markdown('</div>', unsafe_allow_html=True)

            # Download summary
            if st.session_state.local_summary:
                st.download_button(
                    "📥 Download Summary",
                    st.session_state.local_summary,
                    file_name="aura_summary.txt",
                    mime="text/plain",
                    key="dl_summary",
                )

            st.markdown("#### 📌 Key Notes")
            for i, note in enumerate(st.session_state.local_notes, 1):
                st.markdown(f"**{i}.** {note}")

            # Download notes
            if st.session_state.local_notes:
                notes_text = "\n".join(f"{i}. {n}" for i, n in enumerate(st.session_state.local_notes, 1))
                st.download_button(
                    "📥 Download Notes",
                    notes_text,
                    file_name="aura_notes.txt",
                    mime="text/plain",
                    key="dl_notes",
                )

        with tab2:
            analysis = st.session_state.analysis
            if analysis and analysis.get("summary"):
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### AI-Enhanced Summary")
                st.markdown(analysis["summary"])
                st.markdown('</div>', unsafe_allow_html=True)

                if analysis.get("notes"):
                    st.markdown("#### 🧬 AI-Generated Notes")
                    st.markdown(analysis["notes"])
            else:
                st.info("AI summary not available for this document.")

        with tab3:
            st.markdown("#### Generate detailed notes for any topic")
            topic_input = st.text_input("Enter a topic", placeholder="e.g., Binary Search Trees, Neural Networks...")
            level = st.selectbox("Difficulty Level", ["beginner", "intermediate", "advanced"])

            if st.button("📝 Generate Notes") and topic_input:
                with st.spinner("Generating structured notes..."):
                    try:
                        ai = _import_ai_modules()
                        notes_data = ai["generate_topic_notes"](topic_input, level)
                        if notes_data.get("error"):
                            st.error(f"Error: {notes_data['error']}")
                        else:
                            if notes_data.get("summary"):
                                st.markdown("##### 📖 Summary")
                                st.markdown(notes_data["summary"])
                            if notes_data.get("key_points"):
                                st.markdown("##### 🔑 Key Concepts")
                                st.markdown(notes_data["key_points"])
                            if notes_data.get("definitions"):
                                st.markdown("##### 📚 Definitions")
                                st.markdown(notes_data["definitions"])
                            if notes_data.get("examples"):
                                st.markdown("##### 💡 Examples")
                                st.markdown(notes_data["examples"])
                            if notes_data.get("revision_block"):
                                st.markdown("##### 🎯 Exam Revision")
                                st.markdown(notes_data["revision_block"])
                    except Exception as e:
                        st.error(f"Failed to generate notes: {e}")


# ── 3. Questions & MCQs ─────────────────────────────────────────────────────
elif page == "❓ Questions & MCQs":
    st.markdown('<div class="section-title">❓ Questions & MCQs</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        tab1, tab2, tab3 = st.tabs(["📝 Bloom Questions", "🎲 MCQs (Local)", "🧬 AI MCQs"])

        with tab1:
            qa_pairs = st.session_state.local_qa
            if qa_pairs:
                for i, qa in enumerate(qa_pairs, 1):
                    with st.expander(f"Q{i} [{qa.get('bloom_level', '?')}]: {qa['question']}"):
                        st.markdown(f"**Answer:** {qa['answer']}")
                        st.caption(f"Source: {qa.get('source_note', '')[:100]}")
            else:
                st.info("No questions generated. The document may be too short.")

            # AI questions
            analysis = st.session_state.analysis
            if analysis and analysis.get("questions"):
                st.markdown("---")
                st.markdown("#### 🧬 AI-Generated Exam Questions")
                st.markdown(analysis["questions"])

        with tab2:
            mcqs = st.session_state.local_mcqs
            if mcqs:
                if "mcq_answers" not in st.session_state:
                    st.session_state.mcq_answers = {}

                for i, mcq in enumerate(mcqs):
                    st.markdown(f"**Q{i+1}** [{mcq.get('bloom_level', '?')}]: {mcq['question']}")
                    key = f"mcq_local_{i}"
                    answer = st.radio(
                        f"Select answer for Q{i+1}:",
                        mcq['options'],
                        key=key,
                        label_visibility="collapsed",
                    )
                    if st.button(f"Check Answer Q{i+1}", key=f"check_local_{i}"):
                        selected_idx = mcq['options'].index(answer)
                        if selected_idx == mcq['correct_index']:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. Correct answer: **{mcq['options'][mcq['correct_index']]}**")
                    st.markdown("---")
            else:
                st.info("Not enough questions to build MCQs (need ≥ 4 QA pairs).")

        with tab3:
            ai_mcqs = st.session_state.ai_mcqs_list
            if ai_mcqs:
                for i, mcq in enumerate(ai_mcqs):
                    st.markdown(f"**Q{i+1}:** {mcq['question']}")
                    key = f"ai_mcq_{i}"
                    answer = st.radio(
                        f"Select answer:",
                        mcq['options'],
                        key=key,
                        label_visibility="collapsed",
                    )
                    if st.button(f"Check Answer", key=f"check_ai_{i}"):
                        selected_idx = mcq['options'].index(answer)
                        if selected_idx == mcq['correct_index']:
                            st.success("✅ Correct!")
                            if mcq.get('explanation'):
                                st.info(f"💡 {mcq['explanation']}")
                        else:
                            st.error(f"❌ Incorrect. Correct: **{mcq['options'][mcq['correct_index']]}**")
                            if mcq.get('explanation'):
                                st.info(f"💡 {mcq['explanation']}")
                    st.markdown("---")
            else:
                st.info("AI MCQs not available. Process a document first.")


# ── 4. Active Recall ─────────────────────────────────────────────────────────
elif page == "🎯 Active Recall":
    st.markdown('<div class="section-title">🎯 Active Recall</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        prompts = st.session_state.recall_prompts
        if not prompts:
            st.info("No recall prompts available.")
        else:
            use_ai = st.toggle("Use AI evaluation (more detailed feedback)", value=False)
            st.markdown(f"**{len(prompts)} recall prompts available**")
            st.markdown("---")

            if "recall_idx" not in st.session_state:
                st.session_state.recall_idx = 0
            if "recall_scores" not in st.session_state:
                st.session_state.recall_scores = []

            idx = st.session_state.recall_idx
            if idx < len(prompts):
                prompt = prompts[idx]
                st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**Prompt {idx + 1} / {len(prompts)}** ({prompt.get('prompt_type', 'qa').upper()})")
                st.markdown(f"### {prompt['prompt']}")
                st.markdown('</div>', unsafe_allow_html=True)

                user_answer = st.text_area("Your answer:", key=f"recall_ans_{idx}", height=100)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Submit Answer", use_container_width=True):
                        if user_answer.strip():
                            if use_ai:
                                with st.spinner("AI is evaluating your answer..."):
                                    try:
                                        ai = _import_ai_modules()
                                        result = ai["evaluate_recall_with_ai"](
                                            prompt["prompt"], prompt["expected"], user_answer
                                        )
                                    except Exception:
                                        mods = _import_pipeline_modules()
                                        result = mods["evaluate_answer"](user_answer, prompt["expected"])
                            else:
                                mods = _import_pipeline_modules()
                                result = mods["evaluate_answer"](user_answer, prompt["expected"])

                            st.session_state.recall_scores.append(result["score"])

                            if result.get("is_correct"):
                                st.success(f'✅ Score: {result["score"]:.0%}')
                            else:
                                st.error(f'❌ Score: {result["score"]:.0%}')

                            st.markdown(f"**Expected:** {result.get('expected', prompt['expected'])}")
                            if result.get("feedback"):
                                st.info(f"💡 {result['feedback']}")
                        else:
                            st.warning("Please enter an answer.")

                with col2:
                    if st.button("➡️ Next Prompt", use_container_width=True):
                        st.session_state.recall_idx = min(idx + 1, len(prompts) - 1)
                        st.rerun()

                # Progress
                if st.session_state.recall_scores:
                    avg = sum(st.session_state.recall_scores) / len(st.session_state.recall_scores)
                    st.progress(avg, text=f"Average score: {avg:.0%}")
            else:
                st.balloons()
                st.success("🎉 You've completed all recall prompts!")
                if st.session_state.recall_scores:
                    avg = sum(st.session_state.recall_scores) / len(st.session_state.recall_scores)
                    st.metric("Final Score", f"{avg:.0%}")

            if st.button("🔄 Reset Recall Session"):
                st.session_state.recall_idx = 0
                st.session_state.recall_scores = []
                st.rerun()


# ── 5. Concept Map ───────────────────────────────────────────────────────────
elif page == "🗺️ Concept Map":
    st.markdown('<div class="section-title">🗺️ Concept Map</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        st.markdown("#### 🧬 AI Flowchart")
        analysis = st.session_state.analysis
        if analysis and analysis.get("diagram"):
            try:
                ai = _import_ai_modules()
                dot_code = ai["diagram_to_graphviz"](analysis["diagram"])
                if dot_code:
                    st.graphviz_chart(dot_code)
                    with st.expander("View DOT Source"):
                        st.code(dot_code, language="dot")
                else:
                    st.markdown("#### Raw Diagram Text")
                    st.code(analysis["diagram"])
            except Exception as e:
                st.warning(f"Could not render AI diagram: {e}")
                st.code(analysis.get("diagram", ""))
        else:
            st.info("AI diagram not available.")


# ── 6. Smart Q&A ─────────────────────────────────────────────────────────────
elif page == "💬 Smart Q&A":
    st.markdown('<div class="section-title">💬 Smart Q&A — Ask Anything</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done:
        st.warning("⚠️ Please upload and process a document first.")
    else:
        st.session_state.strict_mode = st.toggle(
            "🔒 Strict Document Mode (answers only from the document)",
            value=st.session_state.strict_mode,
        )

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        question = st.chat_input("Ask a question about your document...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        ai = _import_ai_modules()
                        context = st.session_state.raw_text
                        answer = ai["smart_answer"](context, question, st.session_state.strict_mode)

                        # Format answer display
                        if answer.startswith("[DOC_BASED]"):
                            st.markdown("📄 *From document:*")
                            answer = answer.replace("[DOC_BASED]", "").strip()
                        elif answer.startswith("[GEN_KNOWLEDGE]"):
                            st.markdown("🌐 *General knowledge:*")
                            answer = answer.replace("[GEN_KNOWLEDGE]", "").strip()

                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err_msg = f"Error: {e}"
                        st.error(err_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()


# ── 7. Topic Explorer ────────────────────────────────────────────────────────
elif page == "📚 Topic Explorer":
    st.markdown('<div class="section-title">📚 Topic Explorer</div>', unsafe_allow_html=True)
    st.markdown("Generate study materials for **any topic** — no document needed.")

    topic = st.text_input("🔍 Enter topic", placeholder="e.g., Operating Systems, Transformers, DBMS...")

    if topic:
        tab1, tab2, tab3 = st.tabs(["📝 Notes", "❓ Bloom Questions", "🎲 MCQs"])

        with tab1:
            level = st.selectbox("Level", ["beginner", "intermediate", "advanced"], key="te_level")
            if st.button("📝 Generate Notes", key="te_notes_btn"):
                with st.spinner(f"Generating {level} notes for '{topic}'..."):
                    try:
                        ai = _import_ai_modules()
                        result = ai["generate_topic_notes"](topic, level)
                        if result.get("error"):
                            st.error(result["error"])
                        else:
                            for section, label in [
                                ("summary", "📖 Summary"),
                                ("key_points", "🔑 Key Concepts"),
                                ("definitions", "📚 Definitions"),
                                ("examples", "💡 Examples"),
                                ("revision_block", "🎯 Exam Revision"),
                            ]:
                                if result.get(section):
                                    st.markdown(f"##### {label}")
                                    st.markdown(result[section])
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        bloom_labels = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate"}

        with tab2:
            bloom = st.slider("Bloom Level", 1, 5, 2, format="%d", key="te_bloom")
            st.caption(f"Level: **{bloom_labels[bloom]}**")
            count = st.number_input("Number of questions", 3, 10, 5, key="te_q_count")

            if st.button("❓ Generate Questions", key="te_q_btn"):
                with st.spinner("Generating Bloom-taxonomy questions..."):
                    try:
                        ai = _import_ai_modules()
                        questions = ai["generate_bloom_questions"](topic, bloom, count)
                        if questions:
                            for i, q in enumerate(questions, 1):
                                with st.expander(f"Q{i}: {q['question']}"):
                                    st.markdown(f"**Answer:** {q.get('answer', 'N/A')}")
                        else:
                            st.warning("No questions generated. Try a different topic.")
                    except Exception as e:
                        st.error(f"Failed: {e}")

        with tab3:
            bloom_mcq = st.slider("Bloom Level", 1, 5, 2, format="%d", key="te_mcq_bloom")
            st.caption(f"Level: **{bloom_labels[bloom_mcq]}**")
            mcq_count = st.number_input("Number of MCQs", 3, 10, 5, key="te_mcq_count")

            if st.button("🎲 Generate MCQs", key="te_mcq_btn"):
                with st.spinner("Generating MCQs..."):
                    try:
                        ai = _import_ai_modules()
                        mcqs = ai["topic_mcqs"](topic, bloom_mcq, mcq_count)
                        if mcqs:
                            for i, mcq in enumerate(mcqs):
                                st.markdown(f"**Q{i+1}:** {mcq['question']}")
                                answer = st.radio(
                                    "Select:", mcq['options'],
                                    key=f"te_mcq_ans_{i}",
                                    label_visibility="collapsed",
                                )
                                if st.button("Check", key=f"te_mcq_check_{i}"):
                                    sel = mcq['options'].index(answer)
                                    if sel == mcq['correct_index']:
                                        st.success("✅ Correct!")
                                    else:
                                        st.error(f"❌ Correct: {mcq['options'][mcq['correct_index']]}")
                                    if mcq.get("explanation"):
                                        st.info(f"💡 {mcq['explanation']}")
                                st.markdown("---")
                        else:
                            st.warning("No MCQs generated. Try a different topic.")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ── 8. Performance ───────────────────────────────────────────────────────────
elif page == "📊 Performance":
    st.markdown('<div class="section-title">📊 Performance Dashboard</div>', unsafe_allow_html=True)

    if not st.session_state.processing_done and not st.session_state.get("recall_scores"):
        st.warning("⚠️ No data yet. Process a document and use active recall to see performance stats.")
    else:
        scores = st.session_state.get("recall_scores", [])

        col1, col2, col3 = st.columns(3)
        with col1:
            total_q = len(st.session_state.local_qa) + len(st.session_state.ai_mcqs_list)
            st.markdown(f'<div class="stat-card"><div class="stat-value">{total_q}</div><div class="stat-label">Total Questions</div></div>', unsafe_allow_html=True)
        with col2:
            attempted = len(scores)
            st.markdown(f'<div class="stat-card"><div class="stat-value">{attempted}</div><div class="stat-label">Recall Attempted</div></div>', unsafe_allow_html=True)
        with col3:
            avg = f"{sum(scores)/len(scores):.0%}" if scores else "—"
            st.markdown(f'<div class="stat-card"><div class="stat-value">{avg}</div><div class="stat-label">Avg Score</div></div>', unsafe_allow_html=True)

        if scores:
            st.markdown("---")
            st.markdown("#### 📈 Score Trend")
            import pandas as pd
            df = pd.DataFrame({"Attempt": range(1, len(scores)+1), "Score": scores})
            st.line_chart(df.set_index("Attempt"))

            # Bloom level progression
            ai = _import_ai_modules()
            current_bloom = st.session_state.bloom_level
            avg_score = sum(scores) / len(scores) if scores else 0
            new_bloom = ai["next_bloom_level"](avg_score, current_bloom)
            st.session_state.bloom_level = new_bloom

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🎓 Bloom Level")
                bloom_name = ai["bloom_level_name"](new_bloom)
                st.markdown(f'<div class="stat-card"><div class="stat-value">Level {new_bloom}</div><div class="stat-label">{bloom_name}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown("#### 🎯 Next Target")
                if new_bloom < 5:
                    next_name = ai["bloom_level_name"](new_bloom + 1)
                    st.markdown(f'<div class="stat-card"><div class="stat-value">Level {new_bloom+1}</div><div class="stat-label">{next_name}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="stat-card"><div class="stat-value">🏆</div><div class="stat-label">Maximum Level!</div></div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 40px;">
                <div style="font-size: 3rem;">📊</div>
                <p style="color: #8888bb; margin-top: 12px;">Complete some Active Recall prompts to see your performance stats here.</p>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#555;font-size:0.8rem;">'
    '🧠 AURA 2 — Adaptive Understanding & Retaining Agent &nbsp;|&nbsp; '
    'Built with BART, FLAN-T5, Llama 3.1 &nbsp;|&nbsp; '
    'v2.0.0</p>',
    unsafe_allow_html=True,
)
