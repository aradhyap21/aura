"""
Evaluate fine-tuned BART and FLAN-T5 models.
Computes: Loss (perplexity), ROUGE-1/2/L, BLEU, F1 (token overlap)

Run: python evaluate_models.py
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BartTokenizer, BartForConditionalGeneration,
    T5Tokenizer, T5ForConditionalGeneration,
)
import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_SIZE = 100  # samples to evaluate on (fast)

rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    return {
        "rouge1": round(rouge_scores["rouge1"] * 100, 2),
        "rouge2": round(rouge_scores["rouge2"] * 100, 2),
        "rougeL": round(rouge_scores["rougeL"] * 100, 2),
        "bleu": round(bleu_score["score"], 2),
        "f1": round(np.mean(f1_scores) * 100, 2),
    }


def generate_summaries(texts, tokenizer, model, max_input=512, max_output=120):
    summaries = []
    model.eval()
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=max_input, truncation=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_length=max_output, min_length=20,
                num_beams=2, do_sample=False,
            )
        summaries.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return summaries


def generate_questions(inputs_text, tokenizer, model, max_input=512, max_output=64):
    questions = []
    model.eval()
    for text in inputs_text:
        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=max_input, truncation=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_length=max_output, num_beams=2, do_sample=False,
            )
        questions.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return questions


# ---------------------------------------------------------------------------
# Evaluate BART
# ---------------------------------------------------------------------------

def evaluate_bart():
    model_path = "./bart-finetuned" if os.path.isdir("./bart-finetuned") else "facebook/bart-large-cnn"
    print(f"\n{'='*60}")
    print(f"Evaluating BART from: {model_path}")
    print(f"{'='*60}")

    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

    dataset = load_dataset("cnn_dailymail", "3.0.0")
    eval_data = dataset["validation"].select(range(EVAL_SIZE))

    articles = eval_data["article"]
    references = eval_data["highlights"]

    print(f"Generating summaries for {EVAL_SIZE} samples...")
    predictions = generate_summaries(articles, tokenizer, model)

    metrics = compute_metrics(predictions, references)
    print("\nBART Results:")
    print(f"  ROUGE-1 : {metrics['rouge1']}")
    print(f"  ROUGE-2 : {metrics['rouge2']}")
    print(f"  ROUGE-L : {metrics['rougeL']}")
    print(f"  BLEU    : {metrics['bleu']}")
    print(f"  F1      : {metrics['f1']}")
    return metrics


# ---------------------------------------------------------------------------
# Evaluate FLAN-T5
# ---------------------------------------------------------------------------

def evaluate_t5():
    model_path = "./flan-t5-finetuned" if os.path.isdir("./flan-t5-finetuned") else "google/flan-t5-base"
    print(f"\n{'='*60}")
    print(f"Evaluating FLAN-T5 from: {model_path}")
    print(f"{'='*60}")

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(DEVICE)

    dataset = load_dataset("squad")
    eval_data = dataset["validation"].select(range(EVAL_SIZE))

    inputs_text = [
        f"answer: {a['text'][0]} context: {c}"
        for a, c in zip(eval_data["answers"], eval_data["context"])
    ]
    references = eval_data["question"]

    print(f"Generating questions for {EVAL_SIZE} samples...")
    predictions = generate_questions(inputs_text, tokenizer, model)

    metrics = compute_metrics(predictions, references)
    print("\nFLAN-T5 Results:")
    print(f"  ROUGE-1 : {metrics['rouge1']}")
    print(f"  ROUGE-2 : {metrics['rouge2']}")
    print(f"  ROUGE-L : {metrics['rougeL']}")
    print(f"  BLEU    : {metrics['bleu']}")
    print(f"  F1      : {metrics['f1']}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    bart_metrics = evaluate_bart()
    t5_metrics = evaluate_t5()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'BART':>10} {'FLAN-T5':>10}")
    print(f"{'-'*34}")
    for key in ["rouge1", "rouge2", "rougeL", "bleu", "f1"]:
        print(f"{key:<12} {bart_metrics[key]:>10} {t5_metrics[key]:>10}")
