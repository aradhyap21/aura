"""
Fine-tune BART on CNN/DailyMail (summarization) — improved run.
3000 train samples, 3 epochs, warmup scheduler, gradient accumulation.

Run: python train_bart.py
"""

import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "facebook/bart-large-cnn"
OUTPUT_DIR = "./bart-finetuned"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128
TRAIN_SIZE = 3000

print("Loading tokenizer and model...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"].select(range(TRAIN_SIZE))
print(f"Train: {len(train_data)} samples")


def preprocess(batch):
    inputs = tokenizer(
        batch["article"],
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["highlights"],
        max_length=MAX_TARGET_LEN,
        padding="max_length",
        truncation=True,
    )
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    return inputs


print("Preprocessing dataset...")
train_dataset = train_data.map(
    preprocess, batched=True,
    remove_columns=["article", "highlights", "id"],
)
train_dataset.set_format("torch")

use_fp16 = torch.cuda.is_available()
print(f"GPU: {torch.cuda.is_available()} | fp16: {use_fp16}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective batch = 8
    num_train_epochs=3,
    learning_rate=3e-5,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=use_fp16,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("\nStarting training...")
trainer.train()

print(f"\nSaving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")
