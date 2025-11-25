"""
Download and filter SQuAD dataset for GPT-2 124M SFT training
Filters for short, factual answers only
"""

import json
from datasets import load_dataset
from tqdm import tqdm

# Config
OUTPUT_FILE = "squad_qa_filtered.jsonl"
MAX_ANSWER_LENGTH = 80  # SQuAD answers can be long, filter aggressively
MAX_QUESTION_LENGTH = 150
TARGET_SAMPLES = 20000

print("📥 Downloading SQuAD v2 dataset...")
dataset = load_dataset("squad_v2", split="train")

print(f"📊 Total samples: {len(dataset):,}")
print(f"🔍 Filtering for short Q&A pairs...")

filtered_pairs = []

for item in tqdm(dataset):
    question = item.get("question", "").strip()
    
    # SQuAD format has answers as list of dicts
    answers = item.get("answers", {})
    answer_list = answers.get("text", [])
    
    # Skip if no answer (SQuAD v2 has unanswerable questions)
    if not answer_list:
        continue
    
    # Take first answer
    answer = answer_list[0].strip()
    
    # Filter criteria
    if not question or not answer:
        continue
    
    if len(question) > MAX_QUESTION_LENGTH:
        continue
    
    if len(answer) > MAX_ANSWER_LENGTH:
        continue
    
    # Skip if answer is multiple sentences (too complex)
    if answer.count('.') > 1:
        continue
    
    # Skip if answer has question marks (usually not simple factual)
    if '?' in answer:
        continue
    
    # Create clean Q&A pair
    qa_pair = {
        "question": question,
        "answer": answer
    }
    
    filtered_pairs.append(qa_pair)
    
    # Stop if we have enough
    if len(filtered_pairs) >= TARGET_SAMPLES:
        break

print(f"✅ Filtered samples: {len(filtered_pairs):,}")

# Save to JSONL
print(f"💾 Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for pair in filtered_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"✅ Done! Created {OUTPUT_FILE} with {len(filtered_pairs):,} Q&A pairs")
print(f"\n📊 Statistics:")
print(f"   - Questions: {len(filtered_pairs):,}")
print(f"   - Avg question length: {sum(len(p['question']) for p in filtered_pairs) / len(filtered_pairs):.1f} chars")
print(f"   - Avg answer length: {sum(len(p['answer']) for p in filtered_pairs) / len(filtered_pairs):.1f} chars")

# Show samples
print(f"\n🔮 Sample Q&A pairs:")
for i, pair in enumerate(filtered_pairs[:5]):
    print(f"\n[{i+1}] Q: {pair['question']}")
    print(f"    A: {pair['answer']}")

