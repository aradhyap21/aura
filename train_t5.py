"""
Fine-tune FLAN-T5 on SQuAD for question generation — improved run.
2000 train samples, 3 epochs, warmup scheduler, gradient accumulation.

Run: python train_t5.py
"""

import torch
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "google/flan-t5-base"
OUTPUT_DIR = "./flan-t5-finetuned"
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 64
TRAIN_SIZE = 2000

print("Loading tokenizer and model...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

print("Loading SQuAD dataset...")
dataset = load_dataset("squad")
train_data = dataset["train"].select(range(TRAIN_SIZE))
print(f"Train: {len(train_data)} samples")


def preprocess(batch):
    inputs_text = [
        f"answer: {a['text'][0]} context: {c}"
        for a, c in zip(batch["answers"], batch["context"])
    ]
    inputs = tokenizer(
        inputs_text,
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=batch["question"],
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
    remove_columns=train_data.column_names,
)
train_dataset.set_format("torch")

use_fp16 = torch.cuda.is_available()
print(f"GPU: {torch.cuda.is_available()} | fp16: {use_fp16}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,   # effective batch = 8
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
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
