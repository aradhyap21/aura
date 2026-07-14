"""
Part B: Inference script using the fine-tuned BART model.
Loads from ./bart-finetuned and summarizes input text.

Run: python infer_bart.py
"""

import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Use fine-tuned model if available, fall back to base model
MODEL_PATH = "./bart-finetuned" if os.path.isdir("./bart-finetuned") else "facebook/bart-large-cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_PATH}")
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize(text: str) -> str:
    """Summarize text using the fine-tuned BART model."""
    chunks = chunk_text(text, chunk_size=1000)
    summaries = []

    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=40,
                num_beams=4,
                do_sample=False,
            )
        summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    return " ".join(summaries) if summaries else text


if __name__ == "__main__":
    sample = (
        "The mitochondria is the powerhouse of the cell. It produces ATP through "
        "cellular respiration. The process involves the electron transport chain and "
        "oxidative phosphorylation. Mitochondria have their own DNA and are thought "
        "to have originated from ancient bacteria through endosymbiosis."
    )
    print("\nInput text:")
    print(sample)
    print("\nSummary:")
    print(summarize(sample))
