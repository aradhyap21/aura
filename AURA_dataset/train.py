"""
AURA — Fine-tune T5 on RTX 4050
=================================
Trains T5-small on your unified dataset using LoRA (PEFT).

Why T5-small:
- 60M parameters — fits easily on 6GB VRAM
- Text-to-text format — perfect for explanation generation
- Fast to train — 2-3 hours on RTX 4050
- Good quality for focused educational tasks

Install:
    pip install transformers peft datasets torch accelerate

Run:
    python train.py

Output:
    models/aura_explainer/   — your fine-tuned model
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────
CONFIG = {
    "model_name"    : "t5-small",          # base model
    "dataset_path"  : "data/aura_training_final.jsonl",
    "output_dir"    : "models/aura_explainer",
    "max_input_len" : 512,
    "max_output_len": 256,
    "batch_size"    : 8,                   # safe for 6GB VRAM
    "epochs"        : 3,
    "lr"            : 3e-4,
    "warmup_steps"  : 100,
    "save_every"    : 500,                 # save checkpoint every N steps
    "log_every"     : 50,
    "max_examples"  : 20000,               # cap for manageable training time
    "seed"          : 42,
}

# LoRA config — trains only 1% of parameters
LORA_CONFIG = LoraConfig(
    task_type    = TaskType.SEQ_2_SEQ_LM,
    r            = 16,       # rank
    lora_alpha   = 32,
    lora_dropout = 0.1,
    target_modules = ["q", "v"],  # T5 attention modules
)


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class AURADataset(Dataset):
    def __init__(self, path: str, tokenizer, config: dict):
        self.tokenizer  = tokenizer
        self.config     = config
        self.examples   = []

        print(f"Loading dataset from {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                    # Build T5 input: "instruction: {instruction} input: {input}"
                    instruction = ex.get('instruction', '')
                    inp         = ex.get('input', '')
                    output      = ex.get('output', '')

                    if not output or len(output.split()) < 5:
                        continue

                    t5_input = f"instruction: {instruction} input: {inp}"
                    self.examples.append({
                        'input' : t5_input,
                        'output': output,
                        'weight': 3.0 if ex.get('annotated') else 1.0,
                    })

                except Exception:
                    continue

        # Cap examples
        if len(self.examples) > config['max_examples']:
            # Keep all annotated examples
            annotated = [e for e in self.examples if e['weight'] > 1.0]
            rest      = [e for e in self.examples if e['weight'] == 1.0]
            rest      = rest[:config['max_examples'] - len(annotated)]
            self.examples = annotated + rest

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Tokenize input
        input_enc = self.tokenizer(
            ex['input'],
            max_length  = self.config['max_input_len'],
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt',
        )

        # Tokenize output (labels)
        output_enc = self.tokenizer(
            ex['output'],
            max_length  = self.config['max_output_len'],
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt',
        )

        labels = output_enc['input_ids'].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids'      : input_enc['input_ids'].squeeze(),
            'attention_mask' : input_enc['attention_mask'].squeeze(),
            'labels'         : labels,
        }


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train():
    torch.manual_seed(CONFIG['seed'])

    # ── Device ────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer and model ───────────────────────────────────
    print(f"\nLoading {CONFIG['model_name']}...")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG['model_name'])
    model     = T5ForConditionalGeneration.from_pretrained(CONFIG['model_name'])

    # ── Apply LoRA ────────────────────────────────────────────────
    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    model = model.to(device)

    # ── Dataset + DataLoader ───────────────────────────────────────
    dataset    = AURADataset(CONFIG['dataset_path'], tokenizer, CONFIG)
    dataloader = DataLoader(
        dataset,
        batch_size = CONFIG['batch_size'],
        shuffle    = True,
        num_workers= 0,  # Windows compatible
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = CONFIG['lr'],
        weight_decay = 0.01,
    )

    total_steps = len(dataloader) * CONFIG['epochs']
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = CONFIG['warmup_steps'],
        num_training_steps = total_steps,
    )

    # ── Output dir ────────────────────────────────────────────────
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────
    print(f"\nStarting training: {CONFIG['epochs']} epochs, "
          f"{len(dataloader)} steps/epoch\n")

    best_loss   = float('inf')
    global_step = 0
    train_log   = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss  = 0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for batch in pbar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels,
            )

            loss = outputs.loss
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss  += loss.item()
            epoch_steps += 1
            global_step += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log
            if global_step % CONFIG['log_every'] == 0:
                avg_loss = epoch_loss / epoch_steps
                train_log.append({
                    'step' : global_step,
                    'loss' : avg_loss,
                    'epoch': epoch + 1,
                })

            # Save checkpoint
            if global_step % CONFIG['save_every'] == 0:
                checkpoint_dir = os.path.join(
                    CONFIG['output_dir'], f"checkpoint-{global_step}"
                )
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"\n  Saved checkpoint at step {global_step}")

        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"\nEpoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(CONFIG['output_dir'])
            tokenizer.save_pretrained(CONFIG['output_dir'])
            print(f"  ✅ Best model saved (loss: {best_loss:.4f})")

    # ── Save training log ─────────────────────────────────────────
    log_path = os.path.join(CONFIG['output_dir'], 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'config'   : CONFIG,
            'best_loss': best_loss,
            'log'      : train_log,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best loss  : {best_loss:.4f}")
    print(f"  Model saved: {CONFIG['output_dir']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()