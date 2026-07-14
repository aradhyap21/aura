import os
import torch
import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use fine-tuned model if available, otherwise fall back to base model
_MODEL_PATH = "./bart-finetuned" if os.path.isdir("./bart-finetuned") else "facebook/bart-large-cnn"


@st.cache_resource
def load_summarizer():
    """Load and cache the BART summarization model and tokenizer on GPU if available."""
    tokenizer = BartTokenizer.from_pretrained(_MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(_MODEL_PATH).to(DEVICE)
    return tokenizer, model


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into chunks of at most chunk_size characters."""
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def summarize(text: str, summarizer) -> str:
    """Summarize text by chunking and combining BART outputs."""
    tokenizer, model = summarizer
    chunks = chunk_text(text, chunk_size=1000)
    chunks = chunks[:8]  # cap at 8 chunks for speed
    summaries = []

    for chunk in chunks:
        if len(chunk.strip()) < 50:
            continue
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():  # saves memory, speeds up inference
            output_ids = model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=20,
                do_sample=False,
                num_beams=2,  # reduced from default 4 — ~2x faster
            )
        summary_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summaries.append(summary_text)

    if not summaries:
        return text

    return " ".join(summaries)
