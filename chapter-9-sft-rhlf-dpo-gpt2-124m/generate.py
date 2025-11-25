import torch
import tiktoken
from train_gpt2 import GPT, GPTConfig
import os

# ----------- CONFIG ------------
CHECKPOINT_PATH = "checkpoints/model_09535.pt"   # <-- change to your actual final step
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 80
TOP_K = 50
SEED = 42
# --------------------------------

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

def load_model(ckpt_path, device):
    print(f"🔄 Loading checkpoint: {ckpt_path} ...")

    checkpoint = torch.load(ckpt_path, map_location=device)
    config: GPTConfig = checkpoint["config"]
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()
    model.half()

    print("✅ Model loaded.")
    return model


def generate(model, prompt):
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    # tokenize input
    ids = enc.encode(prompt)
    x = torch.tensor(ids, dtype=torch.long)[None, :].repeat(NUM_SAMPLES, 1).to(DEVICE)

    # autoregressive sampling loop
    for _ in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

            topk_probs, topk_idx = torch.topk(probs, k=TOP_K, dim=-1)
            sample_ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_idx, -1, sample_ix)

        x = torch.cat((x, next_token), dim=1)

    outputs = []
    for i in range(NUM_SAMPLES):
        decoded = enc.decode(x[i].tolist())
        outputs.append(decoded)

    return outputs


if __name__ == "__main__":
    model = load_model(CHECKPOINT_PATH, DEVICE)

    prompt = "Hello, I am a language model trained from scratch,"
    print(f"\n📝 Prompt: {prompt}\n")

    samples = generate(model, prompt)

    print("🔮 Generated samples:\n")
    for i, text in enumerate(samples):
        print(f"--- Sample {i+1} ---")
        print(text)
        print()
