import os, json, math, random, copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import amp
import tiktoken

from train_gpt2 import GPT, GPTConfig

# -------------------
# Config
# -------------------
DATA_PATH = "dpo_gpt2_small.jsonl"
SFT_CKPT_PATH = "sft_checkpoints/sft_best.pt" # Assumes you ran sft_train.py first
OUT_DIR = "dpo_checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DPO Hyperparams
BATCH_SIZE = 2          # Small batch size for DPO (pairs)
GRAD_ACCUM_STEPS = 16   # Accumulate gradients
LR = 1e-5               # Lower LR for DPO
BETA = 0.1              # DPO Beta parameter
MAX_STEPS = 1000
EVAL_EVERY = 100
SAVE_EVERY = 500
SEQ_LEN = 1024

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token

# -------------------
# 1. Load Data
# -------------------
def format_dpo_pair(ex):
    # Handle both "prompt" and "instruction" keys
    prompt_text = ex.get("prompt") or ex.get("instruction")
    chosen_text = ex["chosen"]
    rejected_text = ex["rejected"]
    
    # Format matches SFT: ### Instruction: ... ### Response: ...
    # We need to separate prompt and response for masking
    
    prompt_formatted = (
        "### Instruction:\n"
        f"{prompt_text}\n\n"
        "### Response:\n"
    )
    
    chosen_full = prompt_formatted + chosen_text + enc.decode([EOS_ID])
    rejected_full = prompt_formatted + rejected_text + enc.decode([EOS_ID])
    
    return prompt_formatted, chosen_full, rejected_full

def load_dpo_data():
    with open(DATA_PATH, "r") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data

# -------------------
# 2. Dataset & Collate
# -------------------
class DPODataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, chosen, rejected = format_dpo_pair(self.data[idx])
        
        # Tokenize
        prompt_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        chosen_ids = enc.encode(chosen, allowed_special={"<|endoftext|>"})
        rejected_ids = enc.encode(rejected, allowed_special={"<|endoftext|>"})
        
        # Create masks (1 for response, 0 for prompt/padding)
        # We only want to calculate loss on the response part
        
        def get_ids_and_mask(full_ids, prompt_len):
            ids = torch.tensor(full_ids, dtype=torch.long)
            mask = torch.zeros_like(ids)
            mask[prompt_len:] = 1 # Mask out prompt
            return ids, mask
            
        c_ids, c_mask = get_ids_and_mask(chosen_ids, len(prompt_ids))
        r_ids, r_mask = get_ids_and_mask(rejected_ids, len(prompt_ids))
        
        return c_ids, c_mask, r_ids, r_mask

def collate_fn(batch):
    # Pad to longest in batch
    max_len = 0
    for item in batch:
        max_len = max(max_len, len(item[0]), len(item[2]))
    
    max_len = min(max_len, SEQ_LEN) # Cap at SEQ_LEN
    
    c_ids_batch = []
    c_mask_batch = []
    r_ids_batch = []
    r_mask_batch = []
    
    for c_ids, c_mask, r_ids, r_mask in batch:
        # Pad chosen
        pad_len = max_len - len(c_ids)
        if pad_len > 0:
            c_ids = torch.cat([c_ids, torch.tensor([EOS_ID] * pad_len)])
            c_mask = torch.cat([c_mask, torch.zeros(pad_len)])
        else:
            c_ids = c_ids[:max_len]
            c_mask = c_mask[:max_len]
            
        # Pad rejected
        pad_len = max_len - len(r_ids)
        if pad_len > 0:
            r_ids = torch.cat([r_ids, torch.tensor([EOS_ID] * pad_len)])
            r_mask = torch.cat([r_mask, torch.zeros(pad_len)])
        else:
            r_ids = r_ids[:max_len]
            r_mask = r_mask[:max_len]
            
        c_ids_batch.append(c_ids)
        c_mask_batch.append(c_mask)
        r_ids_batch.append(r_ids)
        r_mask_batch.append(r_mask)
        
    return (
        torch.stack(c_ids_batch),
        torch.stack(c_mask_batch),
        torch.stack(r_ids_batch),
        torch.stack(r_mask_batch)
    )

# -------------------
# 3. DPO Loss
# -------------------
def get_batch_logps(model, input_ids, attention_mask):
    # input_ids: [B, T]
    # attention_mask: [B, T] (1 for response, 0 for prompt/pad)
    
    logits, _ = model(input_ids) # [B, T, V]
    
    # Shift logits and labels for next-token prediction
    # logits[:, :-1] predicts input_ids[:, 1:]
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:]
    
    # Gather log probs of the actual tokens
    log_probs = F.log_softmax(logits, dim=-1)
    
    # gather_log_probs
    # labels.unsqueeze(-1) -> [B, T-1, 1]
    per_token_logps = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out prompt and padding
    return (per_token_logps * mask).sum(-1)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    
    losses = -F.logsigmoid(BETA * logits)
    rewards = BETA * (pi_logratios - ref_logratios).detach()
    
    return losses.mean(), rewards.mean()

# -------------------
# 4. Main Training
# -------------------
def train_dpo():
    if not os.path.exists(SFT_CKPT_PATH):
        print(f"❌ SFT checkpoint not found at {SFT_CKPT_PATH}. Please run sft_train.py first!")
        return

    # Load Data
    raw_data = load_dpo_data()
    dataset = DPODataset(raw_data)
    
    # Load Models
    print("🔄 Loading models...")
    ckpt = torch.load(SFT_CKPT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    # Policy Model (Trainable)
    policy_model = GPT(config)
    policy_model.load_state_dict(ckpt["model"])
    policy_model.to(DEVICE)
    policy_model.train()
    
    # Reference Model (Frozen)
    ref_model = GPT(config)
    ref_model.load_state_dict(ckpt["model"])
    ref_model.to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
        
    print("✅ Models loaded.")
    
    optimizer = policy_model.configure_optimizers(weight_decay=0.1, learning_rate=LR, device_type=DEVICE)
    scaler = amp.GradScaler(device="cuda")
    
    # Training Loop
    print("🚀 Starting DPO Training...")
    
    step = 0
    pbar = tqdm(range(MAX_STEPS))
    
    while step < MAX_STEPS:
        # Simple random batch sampling
        batch_indices = random.sample(range(len(dataset)), BATCH_SIZE)
        batch = [dataset[i] for i in batch_indices]
        c_ids, c_mask, r_ids, r_mask = collate_fn(batch)
        
        c_ids, c_mask = c_ids.to(DEVICE), c_mask.to(DEVICE)
        r_ids, r_mask = r_ids.to(DEVICE), r_mask.to(DEVICE)
        
        # Forward pass
        with amp.autocast("cuda"):
            # Policy Logps
            policy_chosen_logps = get_batch_logps(policy_model, c_ids, c_mask)
            policy_rejected_logps = get_batch_logps(policy_model, r_ids, r_mask)
            
            # Reference Logps (no grad)
            with torch.no_grad():
                ref_chosen_logps = get_batch_logps(ref_model, c_ids, c_mask)
                ref_rejected_logps = get_batch_logps(ref_model, r_ids, r_mask)
            
            loss, reward = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps
            )
            
            loss = loss / GRAD_ACCUM_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            pbar.update(GRAD_ACCUM_STEPS)
            pbar.set_description(f"Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | Reward: {reward.item():.4f}")
            
            if step % SAVE_EVERY == 0 and step > 0:
                ckpt_path = os.path.join(OUT_DIR, f"dpo_{step:05d}.pt")
                torch.save({
                    "model": policy_model.state_dict(),
                    "config": config,
                    "step": step
                }, ckpt_path)
                
        step += 1
        
    # Final Save
    torch.save({
        "model": policy_model.state_dict(),
        "config": config,
        "step": MAX_STEPS
    }, os.path.join(OUT_DIR, "dpo_final.pt"))
    print("🏁 DPO Training Complete!")

if __name__ == "__main__":
    train_dpo()
