# generate_infinite.py
import argparse
import sys
import torch
from torch.nn import functional as F
from v2 import BigramLanguageModel, decode, encode, device, block_size, vocab_size

# ---------------------------------------------------------------------
# 🧠 Command-line arguments
parser = argparse.ArgumentParser(description="Generate streaming Shakespeare text using a trained model.")
parser.add_argument("--model_path", type=str, default="shakespeare_model.pth", help="Path to the trained model weights (.pth file)")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random, lower = more deterministic)")
parser.add_argument("--seed_text", type=str, default="", help="Optional text prompt to start generation")
parser.add_argument("--max_tokens", type=int, default=None, help="Maximum total tokens to generate (default = infinite)")
parser.add_argument("--output", type=str, default=None, help="If provided, generated text will also be saved to this file")
args = parser.parse_args()

# ---------------------------------------------------------------------
# ⚙️ Enable fast matmul on RTX GPUs
torch.set_float32_matmul_precision("high")

# ⚙️ Load model
model = BigramLanguageModel()
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()
model = torch.compile(model)  # JIT compile for faster inference

print(f"✅ Model loaded from {args.model_path}")
print(f"🧠 Starting generation (temperature={args.temperature})...\n")

# ---------------------------------------------------------------------
# 📝 Encode the seed or start from blank
if args.seed_text:
    context = torch.tensor([encode(args.seed_text)], dtype=torch.long, device=device)
    sys.stdout.write(args.seed_text)
    sys.stdout.flush()
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

# ---------------------------------------------------------------------
# ⚡ Streaming text generation
with torch.no_grad():
    if args.output:
        f = open(args.output, "w", encoding="utf-8")
    else:
        f = None

    total_generated = 0

    while True:
        # Crop context to block_size for speed
        idx_cond = context[:, -block_size:]
        logits, _ = model(idx_cond)

        # Sample next token
        logits = logits[:, -1, :] / args.temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)

        # Decode and print only the new character
        new_char = decode([idx_next.item()])
        sys.stdout.write(new_char)
        sys.stdout.flush()
        if f:
            f.write(new_char)
            f.flush()

        total_generated += 1
        if args.max_tokens and total_generated >= args.max_tokens:
            break

    if f:
        f.close()

print("\n✅ Generation complete!")
