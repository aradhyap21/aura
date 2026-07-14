"""Question Generator: T5-based exam question generation aligned to Bloom's Taxonomy."""

import os
import torch
import nltk
import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use fine-tuned model if available, otherwise fall back to base model
_MODEL_PATH = "./flan-t5-finetuned" if os.path.isdir("./flan-t5-finetuned") else "google/flan-t5-base"

# Download required NLTK data quietly
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


@st.cache_resource
def load_question_generator():
    """Load and cache the T5 question generation model and tokenizer on GPU if available."""
    tokenizer = T5Tokenizer.from_pretrained(_MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(_MODEL_PATH).to(DEVICE)
    return tokenizer, model


def extract_key_phrases(note: str) -> list[tuple[str, str]]:
    """Extract (answer_phrase, context_sentence) pairs from a note."""
    tokens = nltk.word_tokenize(note)
    tagged = nltk.pos_tag(tokens)

    grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
    chunk_parser = nltk.RegexpParser(grammar)
    tree = chunk_parser.parse(tagged)

    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(word for word, tag in subtree.leaves())
        if phrase.strip():
            phrases.append((phrase, note))

    return phrases


def generate_questions(
    notes: list[str],
    qg_pipeline,
    bloom_levels: list[str] = ["Remember", "Understand", "Apply", "Analyze"],
) -> list[dict]:
    """Generate questions for each note, tagged with a Bloom level."""
    tokenizer, model = qg_pipeline
    qa_pairs = []
    bloom_index = 0

    for note in notes[:10]:  # req 4.4: limit to first 10 notes
        phrases = extract_key_phrases(note)
        if not phrases:
            continue

        answer, context = phrases[0]
        t5_input = f"Generate a question whose answer is: {answer}. Context: {context}"

        inputs = tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        output_ids = model.generate(inputs["input_ids"], max_length=64)
        question_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # req 4.5: filter empty or non-question outputs
        if not question_text or "?" not in question_text:
            continue

        level = bloom_levels[bloom_index % len(bloom_levels)]
        bloom_index += 1

        qa_pairs.append({
            "question": question_text,
            "answer": answer,
            "bloom_level": level,
            "source_note": note,
        })

    return qa_pairs
