import torch
from torch.nn import functional as F
import tiktoken

# import the model code from train script
from train_gpt2 import GPT, GPTConfig

# -------------------------------------------------------------
#                CONFIGURATION
# -------------------------------------------------------------
checkpoint_path = "gpt2_only_weights.pt"   # <-- your saved checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"

max_new_tokens = 100    # number of tokens to generate
top_k = 50              # restrict sampling to top-k predictions

print(f"Using device: {device}")

# -------------------------------------------------------------
#                LOAD TOKENIZER
# -------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")  # same tokenizer used during training


# -------------------------------------------------------------
#                LOAD MODEL
# -------------------------------------------------------------
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,   # you trained with vocab_size=50304
    n_layer=12,
    n_head=12,
    n_embd=768
)

model = GPT(config)
model.to(device)

# Load weights
state_dict = torch.load(checkpoint_path, map_location=device)

# ✅ strip "_orig_mod." keys created by torch.compile
fixed_state_dict = {}
for k, v in state_dict.items():
    new_k = k.replace("_orig_mod.", "")
    fixed_state_dict[new_k] = v

model.load_state_dict(fixed_state_dict)
model.eval()

print("✅ Model loaded successfully!")


# -------------------------------------------------------------
#                GENERATE FUNCTION
# -------------------------------------------------------------
@torch.no_grad()
def generate(prompt: str):
    """ Generate text autoregressively from a starting prompt """
    model.eval()

    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        logits, _ = model(x)

        # last token logits
        logits = logits[:, -1, :]

        # convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # apply top-k sampling
        topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
        next_token = torch.multinomial(topk_probs, num_samples=1)

        # map back to real vocab ids
        next_token = torch.gather(topk_indices, -1, next_token)

        # append
        x = torch.cat((x, next_token), dim=1)

    return enc.decode(x[0].tolist())


# -------------------------------------------------------------
#                TEST GENERATION
# -------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Once upon a time in Adelaide,"
    output = generate(prompt)
    print("\n=== Generated Text ===\n")
    print(output)
