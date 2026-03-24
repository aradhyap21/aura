"""
AURA — Quick Model Test
Run this to see what your fine-tuned model generates.

python test_model.py
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel

MODEL_PATH = "models/aura_explainer"
BASE_MODEL = "t5-small"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

INSTRUCTIONS = {
    "explanation": "You are an expert tutor. Explain this topic simply and clearly for a student who is studying for an exam.",
    "bullets"    : "You are an expert tutor. Generate exactly 5 concise exam bullet points covering the most important facts about this topic.",
    "analogy"    : "You are an expert tutor. Create a memorable real-world analogy that helps a student understand this topic intuitively.",
}

# Test topics
TEST_TOPICS = [
    {
        "topic"  : "Virtualization",
        "content": "Virtualization is a technology that creates virtual versions of physical computing resources including servers storage and networks. It allows multiple virtual machines to run on a single physical machine maximizing resource utilization.",
    },
    {
        "topic"  : "Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
    },
    {
        "topic"  : "Photosynthesis",
        "content": "Photosynthesis is the process by which plants use sunlight water and carbon dioxide to produce oxygen and energy in the form of glucose. It takes place mainly in the chloroplasts using the green pigment chlorophyll.",
    },
]

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    base      = T5ForConditionalGeneration.from_pretrained(BASE_MODEL)
    try:
        model = PeftModel.from_pretrained(base, MODEL_PATH)
        print("  Loaded with LoRA weights")
    except Exception:
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
        print("  Loaded directly")
    model = model.to(DEVICE)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, instruction, input_text):
    t5_input = f"instruction: {instruction} input: {input_text}"
    enc = tokenizer(
        t5_input,
        max_length     = 512,
        truncation     = True,
        return_tensors = 'pt',
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids             = enc['input_ids'],
            attention_mask        = enc['attention_mask'],
            max_new_tokens        = 200,
            num_beams             = 4,
            no_repeat_ngram_size  = 3,
            early_stopping        = True,
            temperature           = 0.7,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def test():
    model, tokenizer = load_model()

    print(f"\n{'='*65}")
    print("  AURA Model Output Test")
    print(f"{'='*65}")

    for test in TEST_TOPICS:
        topic   = test['topic']
        content = test['content']
        base_input = f"Topic: {topic}\n\nContent: {content}"

        print(f"\n{'─'*65}")
        print(f"TOPIC: {topic}")
        print(f"{'─'*65}")

        for task, instruction in INSTRUCTIONS.items():
            output = generate(model, tokenizer, instruction, base_input)
            print(f"\n[{task.upper()}]")
            print(output)

    print(f"\n{'='*65}")
    print("Test complete. Check outputs above.")
    print("If outputs look good → model is working")
    print("If outputs look random → need more training")
    print(f"{'='*65}")

if __name__ == '__main__':
    test()