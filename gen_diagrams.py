"""
Generate actual architecture diagrams for the AURA report.
Saves PNG files to c:\aura\2\diagrams\
"""
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DIAG_DIR = r"c:\aura\2\diagrams"
os.makedirs(DIAG_DIR, exist_ok=True)

W, H = 14, 9   # figure size inches

# ─── colour palette ───────────────────────────────────────────────────────────
C_DARK   = '#1a1a2e'
C_BLUE   = '#16213e'
C_ACCENT = '#0f3460'
C_PURPLE = '#533483'
C_TEAL   = '#2196f3'
C_GREEN  = '#4caf50'
C_ORANGE = '#ff9800'
C_RED    = '#f44336'
C_WHITE  = '#ffffff'
C_GRAY   = '#90a4ae'
C_LBLUE  = '#bbdefb'
C_LGREEN = '#c8e6c9'
C_LYELL  = '#fff9c4'


def arrow(ax, src, dst, col='#555', lw=1.5, style='->'):
    ax.annotate('', xy=dst, xytext=src,
                arrowprops=dict(arrowstyle=style, color=col,
                                lw=lw, connectionstyle='arc3,rad=0'))


def box(ax, x, y, w, h, text, fc='#1565c0', tc='white',
        fs=9, bold=False, radius=0.03):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f'round,pad={radius}',
                          linewidth=1.2, edgecolor='white',
                          facecolor=fc, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight=weight,
            zorder=4, wrap=True,
            multialignment='center')


def diamond(ax, x, y, w, h, text, fc='#f57f17', tc='black', fs=8):
    pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
    poly = plt.Polygon(pts, closed=True, fc=fc, ec='white', lw=1.2, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold', zorder=4,
            multialignment='center')


def save(fig, name):
    path = os.path.join(DIAG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved {path}')
    return path


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Use Case Diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

ax.text(7, 8.5, 'Use Case Diagram – AURA System',
        ha='center', va='center', fontsize=14, color=C_WHITE, fontweight='bold')

# system boundary
sys_rect = FancyBboxPatch((2, 0.5), 10, 7.5, boxstyle='round,pad=0.05',
                           linewidth=2, edgecolor=C_TEAL,
                           facecolor='#0d1b2a', zorder=1)
ax.add_patch(sys_rect)
ax.text(7, 7.7, '« System » AURA Platform', ha='center', va='center',
        fontsize=10, color=C_TEAL, style='italic')

# actors
for y_pos, label, col in [(6, 'Student\n(Primary)', C_GREEN),
                            (3, 'Educator\n(Secondary)', C_ORANGE),
                            (1, 'Administrator', C_RED)]:
    ax.text(0.8, y_pos, label, ha='center', va='center',
            fontsize=9, color=col, fontweight='bold')
    circle = plt.Circle((0.8, y_pos+0.5), 0.25, color=col, zorder=3)
    ax.add_patch(circle)
    # stick figure line
    ax.plot([0.8, 0.8], [y_pos-0.1, y_pos+0.25], color=col, lw=1.5)

# use cases
ucs = [
    (7, 6.8, 'Upload PDF Document'),
    (7, 5.9, 'Generate Study Material'),
    (7, 5.0, 'View AI Summary & Notes'),
    (7, 4.1, 'Take MCQ Quiz'),
    (7, 3.2, 'Practice Active Recall'),
    (7, 2.3, 'View Flowchart Diagram'),
    (7, 1.4, 'Ask AI (Smart Assistant)'),
    (10.5, 6.5, 'Manage API Keys\n(Admin)'),
    (10.5, 5.2, 'Configure Models\n(Admin)'),
]
for x, y, lbl in ucs:
    ell = mpatches.Ellipse((x, y), 3.2, 0.6, fc=C_ACCENT,
                            ec=C_TEAL, lw=1.2, zorder=3)
    ax.add_patch(ell)
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=8, color=C_WHITE, zorder=4)

# arrows from Student
for uc_y in [6.8, 5.9, 5.0, 4.1, 3.2, 2.3, 1.4]:
    ax.annotate('', xy=(5.35, uc_y), xytext=(1.1, 6),
                arrowprops=dict(arrowstyle='->', color=C_GREEN,
                                lw=0.8, connectionstyle='arc3,rad=0'))

# arrows from Educator
for uc_y in [6.8, 5.9, 5.0, 4.1]:
    ax.annotate('', xy=(5.35, uc_y), xytext=(1.1, 3),
                arrowprops=dict(arrowstyle='->', color=C_ORANGE,
                                lw=0.8, connectionstyle='arc3,rad=0'))

# arrows from Admin
for uc_y in [6.5, 5.2]:
    ax.annotate('', xy=(8.85, uc_y), xytext=(13.2, 1),
                arrowprops=dict(arrowstyle='->', color=C_RED,
                                lw=0.8, connectionstyle='arc3,rad=0'))

FIG1 = save(fig, 'fig3_1_use_case.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Data Flow Diagram (7-stage pipeline)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W, 6))
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)
ax.set_xlim(0, 14); ax.set_ylim(0, 6)
ax.axis('off')

ax.text(7, 5.6, 'Data Flow Diagram (Level 1) – AURA Seven-Stage Pipeline',
        ha='center', va='center', fontsize=13, color=C_WHITE, fontweight='bold')

stages = [
    (1.0, 3.0, 'PDF\nInput', '#546e7a', 'white'),
    (3.0, 3.0, 'PDF\nExtractor\n(pdfplumber)', '#1565c0', 'white'),
    (5.0, 3.0, 'BART\nSummarizer\n(GPU/CPU)', '#6a1b9a', 'white'),
    (7.0, 3.0, 'NLTK\nNotes\nGenerator', '#0277bd', 'white'),
    (9.0, 3.0, 'FLAN-T5\nQuestion\nGenerator', '#6a1b9a', 'white'),
    (11.0, 3.0, 'MCQ\nBuilder', '#1565c0', 'white'),
    (13.0, 3.0, 'Active\nRecall', '#0277bd', 'white'),
]

outputs = [
    (3.0, 1.2, 'Cleaned\ntext string', C_GREEN),
    (5.0, 1.2, 'Summary\nparagraphs', C_TEAL),
    (7.0, 1.2, 'Bullet\nnotes list', C_GREEN),
    (9.0, 1.2, 'QA pairs\n+ Bloom tags', C_TEAL),
    (11.0, 1.2, 'MCQ items\n(4 options)', C_GREEN),
    (13.0, 1.2, 'Recall\nprompts', C_TEAL),
]

for x, y, lbl, fc, tc in stages:
    box(ax, x, y, 1.6, 1.2, lbl, fc=fc, tc=tc, fs=7.5, bold=False)

# horizontal arrows between stages
for i in range(len(stages)-1):
    x1 = stages[i][0] + 0.8
    x2 = stages[i+1][0] - 0.8
    ax.annotate('', xy=(x2, 3.0), xytext=(x1, 3.0),
                arrowprops=dict(arrowstyle='->', color=C_TEAL, lw=1.5))

# downward output arrows
for x, y, lbl, col in outputs:
    ax.annotate('', xy=(x, y+0.35), xytext=(x, 2.4),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=7, color=col, fontweight='bold')

# Llama 3.1 parallel path
box(ax, 7, 5.0, 4.5, 0.7, '☁  Llama 3.1 (NVIDIA NIM API)  —  runs in parallel thread',
    fc='#4a148c', tc='white', fs=9, bold=True)
ax.annotate('', xy=(7, 4.6), xytext=(7, 4.655+0.7/2),
            arrowprops=dict(arrowstyle='->', color='#ce93d8', lw=1.5))
ax.text(7, 4.4, '↓ AI Summary / Notes / MCQ / Diagram / Recall eval / Smart QA',
        ha='center', va='center', fontsize=7.5, color='#ce93d8')

FIG2 = save(fig, 'fig3_2_dfd.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Component Diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

ax.text(7, 8.6, 'Component Diagram – AURA Module Architecture',
        ha='center', va='center', fontsize=13, color=C_WHITE, fontweight='bold')

# UI layer
for x, lbl in [(2.5, 'Sources Panel\n(Upload /\nGenerate)'),
               (7, 'Content Panel\n(7 Tabs:\nSummary, Notes,\nQuestions, MCQ,\nDiagram, Recall, AI)'),
               (11.5, 'Studio Panel\n(Feature status\nindicators)')]:
    box(ax, x, 7.2, 3.5, 1.6, lbl, fc='#1a237e', tc=C_WHITE, fs=8)

ax.text(7, 8.2, '« Presentation Layer » Streamlit 3-Panel UI',
        ha='center', va='center', fontsize=9, color=C_LBLUE, style='italic')
rect_ui = FancyBboxPatch((0.3, 6.3), 13.4, 1.8, boxstyle='round,pad=0.05',
                          linewidth=1.5, edgecolor=C_LBLUE, facecolor='none')
ax.add_patch(rect_ui)

# Processing layer
ax.text(7, 6.0, '« Processing Layer » ThreadPoolExecutor (3 threads)',
        ha='center', va='center', fontsize=9, color='#a5d6a7', style='italic')
rect_proc = FancyBboxPatch((0.3, 4.7), 13.4, 1.2, boxstyle='round,pad=0.05',
                            linewidth=1.5, edgecolor='#a5d6a7', facecolor='none')
ax.add_patch(rect_proc)

for x, lbl, fc in [(2.5, 'run_local_pipeline()\n(BART + T5)', '#1b5e20'),
                    (7, 'ai_full_analysis()\n(Llama 3.1 NIM API)', '#4a148c'),
                    (11.5, 'ai_generate_mcqs()\n(Llama 3.1 NIM API)', '#b71c1c')]:
    box(ax, x, 5.2, 3.8, 0.9, lbl, fc=fc, tc=C_WHITE, fs=8)

# Service layer
ax.text(7, 4.4, '« Service Layer » Domain Logic Modules',
        ha='center', va='center', fontsize=9, color='#ffe0b2', style='italic')
rect_svc = FancyBboxPatch((0.3, 3.1), 13.4, 1.2, boxstyle='round,pad=0.05',
                           linewidth=1.5, edgecolor='#ffe0b2', facecolor='none')
ax.add_patch(rect_svc)

for x, lbl in [(1.2, 'pdf_extractor\n.py'), (2.9, 'summarizer\n.py'),
               (4.6, 'notes_generator\n.py'), (6.3, 'question_generator\n.py'),
               (8.0, 'mcq_builder\n.py'), (9.7, 'gemini_enhancer\n.py'),
               (11.4, 'active_recall\n.py'), (13.1, 'topic_enhancer\n.py')]:
    box(ax, x, 3.65, 1.5, 0.9, lbl, fc='#e65100', tc=C_WHITE, fs=7)

# Data layer
ax.text(7, 2.8, '« Data Layer » Streamlit Session State + Dataclass Models',
        ha='center', va='center', fontsize=9, color='#b3e5fc', style='italic')
rect_data = FancyBboxPatch((0.3, 1.5), 13.4, 1.2, boxstyle='round,pad=0.05',
                            linewidth=1.5, edgecolor='#b3e5fc', facecolor='none')
ax.add_patch(rect_data)

for x, lbl in [(2.0, 'StudyMaterial\n(dataclass)'), (4.5, 'QAPair\n(dataclass)'),
               (7.0, 'MCQItem\n(dataclass)'), (9.5, 'RecallPrompt\n(dataclass)'),
               (12.0, 'Chat History\n(list[dict])')]:
    box(ax, x, 2.1, 2.6, 0.8, lbl, fc='#01579b', tc=C_WHITE, fs=8)

# arrows between layers
for bx in [2.5, 7, 11.5]:
    arrow(ax, (bx, 6.3), (bx, 5.65), col=C_GRAY)

for bx in [2.5, 7, 11.5]:
    arrow(ax, (bx, 4.75), (bx, 4.1), col=C_GRAY)

for bx in [2.5, 7, 11.5]:
    arrow(ax, (bx, 3.2), (bx, 2.5), col=C_GRAY)

FIG3 = save(fig, 'fig3_3_component.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Sequence Diagram
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

ax.text(7, 8.6, 'Sequence Diagram – Parallel Thread Execution on Document Generation',
        ha='center', va='center', fontsize=12, color=C_WHITE, fontweight='bold')

actors = [('User', 1.0), ('app.py\n(UI)', 3.0), ('Thread 1\nBART Pipeline', 5.5),
          ('Thread 2\nLlama 3.1\nAnalysis', 8.5), ('Thread 3\nLlama 3.1\nMCQ', 11.0),
          ('session_state\n(Cache)', 13.2)]

for lbl, x in actors:
    box(ax, x, 7.9, 1.7, 0.7, lbl, fc='#1565c0', tc=C_WHITE, fs=8, bold=True)
    ax.plot([x, x], [0.3, 7.55], color='#546e7a', lw=1, linestyle='--', zorder=1)

steps = [
    (7.5, 1.0, 3.0, 'upload PDF', C_GREEN),
    (7.1, 3.0, 5.5, 'submit(run_local_pipeline)', C_TEAL),
    (7.1, 3.0, 8.5, 'submit(ai_full_analysis)', C_PURPLE),
    (7.1, 3.0, 11.0, 'submit(ai_generate_mcqs)', C_ORANGE),
    (6.3, 5.5, 5.5, 'extract → BART → T5 → MCQ\n→ diagram → recall', C_TEAL),
    (5.5, 8.5, 8.5, 'Llama 3.1 API\n(summary + notes + questions)', C_PURPLE),
    (5.5, 11.0, 11.0, 'Llama 3.1 API\n(10 MCQs + explanations)', C_ORANGE),
    (4.5, 5.5, 13.2, 'store StudyMaterial', C_TEAL),
    (3.5, 8.5, 13.2, 'store AI Analysis JSON', C_PURPLE),
    (2.5, 11.0, 13.2, 'store MCQ list', C_ORANGE),
    (1.5, 3.0, 13.2, 'read all results', C_GREEN),
    (0.8, 3.0, 1.0, 'render 7 tabs ✓', C_GREEN),
]

for y, x1, x2, lbl, col in steps:
    xmin, xmax = min(x1, x2), max(x1, x2)
    if x1 == x2:
        rect = FancyBboxPatch((x1-0.6, y-0.25), 1.2, 0.5,
                               boxstyle='round,pad=0.02', fc='#263238',
                               ec=col, lw=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x1, y, lbl, ha='center', va='center',
                fontsize=6.5, color=col, zorder=3, multialignment='center')
    else:
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2))
        ax.text((x1+x2)/2, y+0.12, lbl, ha='center', va='bottom',
                fontsize=6.5, color=col)

FIG4 = save(fig, 'fig3_4_sequence.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Layered System Architecture
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(W, H))
fig.patch.set_facecolor(C_DARK)
ax.set_facecolor(C_DARK)
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')

ax.text(7, 8.6, 'Layered System Architecture – AURA Four-Layer Platform',
        ha='center', va='center', fontsize=13, color=C_WHITE, fontweight='bold')

layers = [
    (7.5, '1 — PRESENTATION LAYER', 'Streamlit 3-Panel UI  |  Sources Panel  |  Content Tabs (X7)  |  Studio Panel  |  Custom CSS Glassmorphism', '#1a237e', '#e3f2fd'),
    (5.8, '2 — PROCESSING LAYER', 'ThreadPoolExecutor (3 workers)  |  Thread 1: BART Pipeline  |  Thread 2: AI Full Analysis  |  Thread 3: MCQ Generation', '#1b5e20', '#e8f5e9'),
    (4.1, '3 — SERVICE LAYER', 'pdf_extractor.py  |  summarizer.py  |  notes_generator.py  |  question_generator.py  |  mcq_builder.py  |  gemini_enhancer.py  |  active_recall.py  |  topic_enhancer.py', '#e65100', '#fff3e0'),
    (2.4, '4 — DATA LAYER', 'st.session_state key-value store  |  StudyMaterial  |  QAPair  |  MCQItem  |  RecallPrompt  |  DiagramNode  |  ChatHistory  |  Validated Dataclasses', '#01579b', '#e1f5fe'),
]

for yc, title, content, fc, tc_light in layers:
    rect = FancyBboxPatch((0.3, yc-0.7), 13.4, 1.3,
                           boxstyle='round,pad=0.05',
                           linewidth=2, edgecolor='white', facecolor=fc, zorder=2)
    ax.add_patch(rect)
    ax.text(1.0, yc+0.25, title, ha='left', va='center',
            fontsize=10, color='white', fontweight='bold', zorder=3)
    ax.text(1.0, yc-0.2, content, ha='left', va='center',
            fontsize=7.5, color='#eceff1', zorder=3)

# GPU / API annotations
box(ax, 12.5, 5.8, 1.6, 0.5, 'NVIDIA\nRTX 4050\n(CUDA)', fc='#76ff03', tc='black', fs=7)
box(ax, 12.5, 4.1, 1.6, 0.5, 'NVIDIA\nNIM API\n(Cloud)', fc='#40c4ff', tc='black', fs=7)

# vertical arrows
for y1, y2 in [(7.3, 6.5), (5.5, 4.8), (3.8, 3.1)]:
    ax.annotate('', xy=(7, y2), xytext=(7, y1),
                arrowprops=dict(arrowstyle='<->', color='#aaa', lw=1.5))

FIG5 = save(fig, 'fig3_5_layered_arch.png')

print("All 5 diagrams generated successfully.")
print(f"DIAG_DIR: {DIAG_DIR}")
print([FIG1, FIG2, FIG3, FIG4, FIG5])
