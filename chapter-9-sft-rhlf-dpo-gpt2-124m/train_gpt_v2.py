 # ON ULTRA-FINEWEB EDU DATASET 5B
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import os
import math
from torch import amp
from torch.backends.cuda import sdp_kernel
sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)



# GOOGLE CLOUD UPLOAD SUPPORT
from google.cloud import storage

# Weights & Biases logging
import wandb
os.environ['WANDB_API_KEY'] = 'bc775a93d32c104af12794bd595056eb315f300f'
WANDB_KEY = os.getenv("WANDB_API_KEY")

wandb.login(key=WANDB_KEY)





# path to your JSON in kaggle/input
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gpt2-bucket-key.json"
os.environ["BUCKET_NAME"] = "gpt2-ultrafineweb"
bucket_name = "gpt2-ultrafineweb"
print("Service Account Key Loaded")

client = storage.Client()
bucket = client.bucket(bucket_name)

print("Auth OK, bucket exists:", bucket)


def upload_to_gcs(local_path, dest_blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_blob_name)
        blob.upload_from_filename(local_path)
        print(f"Uploaded to GCS: gs://{bucket_name}/{dest_blob_name}")
    except Exception as e:
        print(f"Upload failed: {e}")

def download_latest_checkpoint(bucket_name, prefix="checkpoints", local_dir="checkpoints"):
    """
    Download only the latest checkpoint (.pt) file from GCS into local_dir.
    Skips download if same file already exists locally with the same size.
    Returns: local file path or None if not found.
    """
    os.makedirs(local_dir, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # list blobs inside the prefix folder
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".pt")]

    if not blobs:
        print("❌ No checkpoints found in GCS.")
        return None

    # sort based on filename (model_00001.pt → latest is last)
    blobs.sort(key=lambda b: b.name)

    latest_blob = blobs[-1]
    filename = latest_blob.name.split("/")[-1]
    local_path = os.path.join(local_dir, filename)

    # If file already exists AND size matches, skip
    if os.path.exists(local_path) and os.path.getsize(local_path) == latest_blob.size:
        print(f"✅ Latest checkpoint already exists locally: {filename}")
        return local_path

    print(f"⬇️ Downloading latest checkpoint from GCS:\n  {latest_blob.name} → {local_path}")
    latest_blob.download_to_filename(local_path)

    print(f"✅ Download complete: {local_path}")
    return local_path

#--------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) #bias ON
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        #calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # attention (materializes the large (T,T) matrix for all the queries and keys)

        #flash attention v1 as T4 supports v1 only

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention (inbuilt-pytorch version)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y



class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= 2 * (self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # inside class GPT(nn.Module):

    def resize_embeddings(self, new_vocab_size: int):
      """Resize token embeddings and lm_head to new vocab size."""
      old_weight = self.transformer.wte.weight.data
      old_vocab_size, embed_dim = old_weight.shape

      new_embed = torch.zeros(new_vocab_size, embed_dim, device=old_weight.device)
      new_embed[:min(old_vocab_size, new_vocab_size)] = old_weight[:min(old_vocab_size, new_vocab_size)]

      self.transformer.wte = torch.nn.Embedding(new_vocab_size, embed_dim)
      self.transformer.wte.weight.data = new_embed

      self.lm_head = torch.nn.Linear(embed_dim, new_vocab_size, bias=False)
      self.lm_head.weight = self.transformer.wte.weight


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # if master_process:
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        # if master_process:
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt



class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        #get the shard filenames
        data_root = "edu_ultrafineweb5B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0 , f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # SHAKESPEAR DATASET:
        # # at init load tokens form disk and store them in memory
        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(tokens)} tokens")
        # print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        #state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # outputs
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """Load checkpoint and restore training state"""
    print(f"Loading checkpoint from {checkpoint_path}")

    import torch.serialization
    torch.serialization.add_safe_globals([GPTConfig])

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

    # Restore RNG states
    torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint['cuda_rng_state'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    start_step = checkpoint['step'] + 1
    print(f"Resumed from step {checkpoint['step']}, starting at step {start_step}")
    return start_step




@torch.no_grad()
def evaluate_val_loss(model, val_loader, batches=20):
    """Run on a few batches of val set and compute loss + perplexity."""
    model.eval()

    losses = []
    for _ in range(batches):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with amp.autocast("cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)

    model.train()
    return avg_loss, ppl

if __name__ == "__main__":
    pass
# # -----------------------------------------------------------------------------

#     # simple launch:
#     # python train_gpt2.py
#     # DDP launch for e.g. 8 GPUs:
#     # torchrun --standalone --nproc_per_node=8 train_gpt2.py

#     # attempt to autodetect the device

#     os.environ.setdefault("NCCL_P2P_DISABLE", "1")
#     os.environ.setdefault("NCCL_IB_DISABLE", "1")
#     os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

#     import time
#     torch.backends.cuda.matmul.allow_tf32 = True  # no effect on T4, safe to enable
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")


#     # run the training loop
#     from torch.distributed import init_process_group, destroy_process_group
#     from torch.nn.parallel import DistributedDataParallel as DDP
#     import torch.distributed as dist



#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = "mps"
#     print(f"using device: {device}")
#     # device = "cpu" #OVERRIDE
#     max_length = 30
#     num_return_sequences = 5

#     log_dir = "checkpoints"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, f"log.txt")

#     # set up DDP (distributed data parallel).
#     # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
#     ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
#     if ddp:
#         # use of DDP atm demands CUDA, we set the device appropriately according to rank
#         assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
#         init_process_group(backend='nccl')
#         ddp_rank = int(os.environ['RANK'])
#         ddp_local_rank = int(os.environ['LOCAL_RANK'])
#         ddp_world_size = int(os.environ['WORLD_SIZE'])
#         device = f'cuda:{ddp_local_rank}'
#         torch.cuda.set_device(device)
#         master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
#     else:
#         # vanilla, non-DDP run
#         ddp_rank = 0
#         ddp_local_rank = 0
#         ddp_world_size = 1
#         master_process = True
#         # attempt to autodetect device
#         device = "cpu"
#         if torch.cuda.is_available():
#             device = "cuda"
#         elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#             device = "mps"
#         print(f"using device: {device}")

#     device_type = "cuda" if device.startswith("cuda") else "cpu"

#     torch.manual_seed(1337)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(1337)

#     total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
#     B = 8 # micro batch size
#     T = 1024 # sequence length
#     assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T"
#     grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
#     if master_process:
#         print(f"total desired batch size: {total_batch_size}")
#         print(f"calculated gradient accumulation steps: {grad_accum_steps}")

#         wandb.init(
#             project="gpt2-ultra-fineweb",
#             name="kaggle-ddp-training",
#             config={
#                 "model": "GPT2_124M_from_scratch",
#                 "batch_size": 8,
#                 "seq_len": 1024,
#                 "grad_accum_steps": grad_accum_steps,
#                 "dataset": "UltraFineWeb-5B",
#                 "precision": "fp16 + GradScaler",
#             }
#         )


#     train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

#     val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

#     # create model
#     model = GPT(GPTConfig(vocab_size=50304))
#     # model = GPT.from_pretrained("gpt2")
#     model.to(device)
#     #compile the model
#     # model = torch.compile(model)
#     if ddp:
#         model = DDP(model, device_ids=[ddp_local_rank])
#     raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

#     max_lr = 6e-4
#     min_lr = max_lr * 0.1
#     warmup_steps = 715
#     max_steps = 9536 # as 5B tokens (5e9 / 2**19)
#     def get_lr(it):
#         # 1) linear warmup for warmup_iters steps
#         if it < warmup_steps:
#             return max_lr * (it+1) / warmup_steps
#         # 2) if it > lr_decay_iters, return min learning rate
#         if it >= max_steps:
#             return min_lr
#         # 3) in between, use consine decay down to min learning rate
#         decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
#         assert 0 <= decay_ratio <= 1
#         coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges at 1 and goes to 0
#         return min_lr + coeff * (max_lr - min_lr)

#     #optimize!

#     scaler = amp.GradScaler(device="cuda")  # NEW: enable mixed precision

#     optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)
#     # optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

#     start_step = 0

#     if master_process:  # only rank0 scans filesystem

#         latest_gcs_ckpt = None
#         ckpts = [f for f in os.listdir(log_dir) if f.endswith(".pt")]
#         if not ckpts:
#             print("No local checkpoint found — checking Google Cloud...")
#             latest_gcs_ckpt = download_latest_checkpoint(bucket_name="gpt2-ultrafineweb")

#             if latest_gcs_ckpt:
#                 print(f"Loaded checkpoint from GCS: {latest_gcs_ckpt}")
#                 ckpts = [latest_gcs_ckpt]   # treat as local checkpoint now

#         if ckpts:
#             ckpts.sort()  # ensures last checkpoint is latest
#             latest_ckpt = ckpts[-1]
#             if not latest_ckpt.startswith(log_dir):
#                 latest_ckpt = os.path.join(log_dir, latest_ckpt)

#             print(f"▶ Found checkpoint: {latest_ckpt}")
#             start_step = load_checkpoint(latest_ckpt, raw_model, optimizer, scaler)
#             ckpt = torch.load(latest_ckpt, map_location="cpu")
#             if "data_state" in ckpt:
#                 train_loader.current_shard = ckpt["data_state"]["current_shard"]
#                 train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
#                 train_loader.current_position = ckpt["data_state"]["current_position"]

#             avg_step_time = ckpt.get("avg_step_time", None)
#         else:
#             print("▶ No checkpoint found — starting fresh.")

#     # Sync start_step across all GPUs if running DDP
#     if ddp:
#         start_step_tensor = torch.tensor([start_step], device=device)
#         dist.broadcast(start_step_tensor, src=0)
#         start_step = int(start_step_tensor.item())

#     if master_process:
#         print(f"Training will start at step = {start_step} / {max_steps}")


#     for step in range(start_step, max_steps):

#         t0 = time.time()
#         optimizer.zero_grad()
#         loss_accum = 0.0

#         #determine and set the learning rate for this iteration
#         lr = get_lr(step)
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

#         for micro_step in range(grad_accum_steps):
#             x, y = train_loader.next_batch()
#             x, y = x.to(device), y.to(device)

#             # FP16 forward + loss (autocast enables mixed precision on T4)
#             with amp.autocast("cuda"):
#                 logits, loss = model(x, y)

#             # we have to scale the loss to account for gradient accumulation,
#             # because the gradients just add on each successive backward().
#             # addition of gradients corresponds to a SUM in the objective, but
#             # instead of a SUM we want MEAN. Scale the loss here so it comes out right.
#             loss = loss / grad_accum_steps # scale the loss
#             loss_accum += loss.item()

#             if ddp:
#                 model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
#             # backward pass (scaled for safe gradients in FP16)
#             scaler.scale(loss).backward()
#         if ddp:
#             loss_accum_tensor = torch.tensor(loss_accum, dtype=torch.float32, device=device)
#             dist.all_reduce(loss_accum_tensor, op=dist.ReduceOp.AVG)
#             loss_accum = loss_accum_tensor.item()
#         #unscale before clipping
#         scaler.unscale_(optimizer)
#         norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)


#         # update weights
#         scaler.step(optimizer)
#         scaler.update()

#         torch.cuda.synchronize() # wait for GPU to finish work
#         t1 = time.time()

#         dt = (t1 - t0) # time difference
#         ms = dt * 1000 #ms
#         tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
#         tokens_per_sec = tokens_processed / dt
#         if step == start_step:
#             avg_step_time = dt
#         else:
#             avg_step_time = avg_step_time * 0.9 + dt * 0.1  # EMA smoothing

#         steps_left = max_steps - step - 1
#         remaining_sec = steps_left * avg_step_time

#         hrs = int(remaining_sec // 3600)
#         mins = int((remaining_sec % 3600) //#scrollTo=qHmihQqmdyAl 60)
#         if master_process:

#             wandb.log({
#                 "train/loss": loss_accum,
#                 "train/lr": lr,
#                 "train/grad_norm": norm,
#                 "train/tokens_per_sec": tokens_per_sec,
#                 "train/step_time_ms": ms,
#                 "step": step,
#             })

#             print(
#                 f" step {step:4d} | loss: {loss_accum:.6f} | lr: {lr:.4e} | "
#                 f"norm: {norm:.4f} | dt: {ms:.2f}ms | tok/sec: {tokens_per_sec:.2f} | "
#                 f"ETA: {hrs:02d}h:{mins:02d}m"
#             )

#             # --- VALIDATION CHECK ---
#             if step % 200 == 0:   # adjust frequency as needed
#                 val_loss, val_ppl = evaluate_val_loss(raw_model, val_loader)
#                 wandb.log({
#                     "val/loss": val_loss,
#                     "val/perplexity": val_ppl,
#                 })
#                 print(f" VAL | loss: {val_loss:.6f} | ppl: {val_ppl:.2f}")


#             if step % 500 == 0:  # change to whatever frequency you want
#                     raw_model.eval()  # disable dropout/layernorm noise

#                     prompt = "Hello, I'm training a GPT model,"
#                     enc = tiktoken.get_encoding("gpt2")
#                     tokens = enc.encode(prompt)
#                     tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1).to(device)

#                     max_gen_len = 40  # maximum generated length

#                     with torch.no_grad():
#                         for _ in range(max_gen_len - tokens.shape[1]):
#                             logits, _ = raw_model(tokens)
#                             logits = logits[:, -1, :]
#                             probs = F.softmax(logits, dim=-1)
#                             topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#                             ix = torch.multinomial(topk_probs, 1)
#                             xcol = torch.gather(topk_indices, -1, ix)
#                             tokens = torch.cat((tokens, xcol), dim=1)

#                     print("\n🔮 SAMPLE GENERATION (step", step, ")")
#                     for i in range(num_return_sequences):
#                         decoded = enc.decode(tokens[i].tolist())
#                         print(f"   > {decoded}")
#                         wandb.log({f"sample_{i}": decoded})

#                     raw_model.train()  # switch model back to train mode

#             # Save checkpoint every 10 steps (adjust as needed)
#             # if step == max_steps - 1:  #for final
#             # if step > 0 and (step % 200 == 0 or step == max_steps - 1): #for after every 10 epochs
#             if step > 0 and (step % 500 == 0 or step == max_steps - 1): #for after every 10 epochs
#                 checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
#                 checkpoint = {
#                     'model': raw_model.state_dict(),
#                     'config': raw_model.config,
#                     'step': step,
#                     'train_loss': loss_accum,
#                     'optimizer': optimizer.state_dict(),
#                     'scaler': scaler.state_dict(),  # save scaler state for mixed precision
#                     'rng_state': torch.get_rng_state(),
#                     'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
#                     'data_state': {
#                         'current_shard': train_loader.current_shard,
#                         'current_position': train_loader.current_position,
#                     },
#                     'val_loss': val_loss if 'val_loss' in locals() else None,
#                     'val_perplexity': val_ppl if 'val_ppl' in locals() else None,

#                 }
#                 checkpoint['avg_step_time'] = avg_step_time if 'avg_step_time' in locals() else None
#                 torch.save(checkpoint, checkpoint_path)
#                 upload_to_gcs(checkpoint_path, f"checkpoints/model_{step:05d}.pt")
#                 wandb.save(checkpoint_path)


#                 print(f"Checkpoint saved at step {step}: {checkpoint_path}")
#                 # After torch.save(...)
#                 keep = 3
#                 ckpts = sorted([f for f in os.listdir(log_dir) if f.endswith(".pt")])
#                 while len(ckpts) > keep:
#                     old = os.path.join(log_dir, ckpts.pop(0))
#                     try:
#                         os.remove(old)
#                         print(f"Removed old checkpoint: {old}")
#                     except Exception as e:
#                         print(f"Warn: could not remove {old}: {e}")


#     if ddp:
#         destroy_process_group()

#     import sys; sys.exit(0)


#     #prefix tokens
#     import tiktoken
#     enc = tiktoken.get_encoding("gpt2")
#     tokens = enc.encode("Hello, I'm a language model,")
#     tokens = torch.tensor(tokens, dtype=torch.long) #(8, )
#     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)
#     x = tokens.to(device)


#     # generate! right now x is (B, T) where B = 5, T = 8
#     # set the seed is 42 (my fav what is life :P)
#     torch.manual_seed(42)
#     torch.cuda.manual_seed(42)
#     while x.size(1) < max_length:
#         # forward the model to get the logits
#         with torch.no_grad():
#             logits = model(x) # (B, T, vocab_size)
#             # take the logits at the last position
#             logits = logits[:, -1, :] # (B, vocab_size)
#             # get the probabilities
#             probs = F.softmax(logits, dim=-1)
#             # do top-k sampling of 50 (huggingface pipeline default)
#             # topk_probs here becomes (5, 50), topk_indices is  (5, 50)
#             topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
#             # select a token from the top-k probabilities
#             ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
#             # gather the corresponding indices
#             xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
#             # append to the sequnce
#             x = torch.cat((x, xcol), dim=1)

#     # print the generated text
#     for i in range(num_return_sequences):
#         tokens = x[i, :max_length].tolist()
#         decoded = enc.decode(tokens)
#         print(">", decoded)

