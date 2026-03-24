"""
AURA — Explanation Engine
==========================
Loads your fine-tuned T5 model and generates:
- Simplified explanations
- Exam bullet points
- Real-world analogies

Used by analyze_service.py after topic extraction.

Usage:
    from services.explainer import AURAExplainer
    engine = AURAExplainer()
    result = engine.explain_topic("Virtualization", "Virtualization is a technology...")
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────
MODEL_PATH  = "models/aura_explainer"
BASE_MODEL  = "t5-small"
MAX_INPUT   = 512
MAX_OUTPUT  = 256
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTIONS = {
    "explanation": "You are an expert tutor. Explain this topic simply and clearly for a student who is studying for an exam.",
    "bullets"    : "You are an expert tutor. Generate exactly 5 concise exam bullet points covering the most important facts about this topic.",
    "analogy"    : "You are an expert tutor. Create a memorable real-world analogy that helps a student understand this topic intuitively.",
}


class AURAExplainer:
    """
    Wrapper around the fine-tuned T5 model.
    Lazy loads on first use — doesn't slow down server startup.
    """
    _instance = None

    def __init__(self):
        self.model     = None
        self.tokenizer = None
        self.loaded    = False

    def load(self):
        """Load model — called on first explanation request."""
        if self.loaded:
            return

        import os
        if not os.path.exists(MODEL_PATH):
            print(f"  ⚠ Model not found at {MODEL_PATH}. Run train.py first.")
            self.loaded = False
            return

        print(f"  Loading AURA explainer from {MODEL_PATH}...")
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

        # Load base model + LoRA weights
        base  = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
        try:
            self.model = PeftModel.from_pretrained(base, MODEL_PATH)
        except Exception:
            # Fallback: load directly if not PEFT
            self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.loaded = True
        print(f"  ✅ Explainer loaded on {DEVICE}")

    def _generate(self, instruction: str, input_text: str) -> str:
        """Run inference on one input."""
        if not self.loaded:
            return ""

        t5_input = f"instruction: {instruction} input: {input_text}"

        enc = self.tokenizer(
            t5_input,
            max_length     = MAX_INPUT,
            truncation     = True,
            return_tensors = 'pt',
        ).to(DEVICE)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids      = enc['input_ids'],
                attention_mask = enc['attention_mask'],
                max_new_tokens = MAX_OUTPUT,
                num_beams      = 4,         # beam search — better quality
                no_repeat_ngram_size = 3,   # prevents repetition
                early_stopping = True,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def explain_topic(self, topic: str, content: str) -> dict:
        """
        Given a topic name and its content, generate all 3 outputs.
        Returns dict with explanation, bullets, analogy.
        """
        self.load()

        if not self.loaded:
            return {
                "explanation": "Model not loaded. Run train.py first.",
                "bullets"    : "",
                "analogy"    : "",
            }

        base_input = f"Topic: {topic}\n\nContent: {content[:600]}"

        explanation = self._generate(INSTRUCTIONS['explanation'], base_input)
        bullets     = self._generate(INSTRUCTIONS['bullets'],     base_input)
        analogy     = self._generate(INSTRUCTIONS['analogy'],     base_input)

        return {
            "explanation": explanation,
            "bullets"    : bullets,
            "analogy"    : analogy,
        }


# ── Singleton instance ────────────────────────────────────────────
_engine = None

def get_engine() -> AURAExplainer:
    global _engine
    if _engine is None:
        _engine = AURAExplainer()
    return _engine