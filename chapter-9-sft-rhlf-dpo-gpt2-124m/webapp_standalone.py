#!/usr/bin/env python3
"""
Standalone Gradio Web App for GPT-2 Q&A Model
No GCS/wandb dependencies - just model inference!
"""
import sys
import os

# Prevent GCS/wandb initialization
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_SILENT'] = 'true'

import torch
import tiktoken
import torch.nn.functional as F
import gradio as gr

# Import model only (not the whole training script)
sys.path.insert(0, os.path.dirname(__file__))

# Minimal imports
from dataclasses import dataclass
import torch.nn as nn
import math

# ============================================================
# Minimal GPT Model Definition (copied to avoid import issues)
# ============================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ============================================================
# Model Loading
# ============================================================

MODEL_CKPT = "qa-sft_best.pt"
BASE_CKPT = "checkpoints/model_09535.pt"

print("Loading GPT-2 Q&A Model...")

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token

def load_model_minimal():
    """Load model without triggering GCS/wandb"""
    print(f"Loading checkpoint: {MODEL_CKPT}")
    
    # Load SFT checkpoint
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)
    
    # Load config from base if needed
    if "config" not in ckpt:
        base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
        config = base_ckpt["config"]
    else:
        config = ckpt["config"]
    
    # Initialize model
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    step = ckpt.get("step", "unknown")
    val_loss = ckpt.get("val_loss", "unknown")
    
    print(f"Model loaded successfully! Step: {step}, Val Loss: {val_loss}, Device: {device}")
    
    return model, device

# Load model
model, device = load_model_minimal()

# ============================================================
# Generation Function
# ============================================================

@torch.no_grad()
def generate_answer(question, temperature=0.6, top_k=15, max_length=100):
    """Generate answer for a question"""
    if not question or not question.strip():
        return "Please enter a question!"
    
    prompt = f"Q: {question.strip()}\nA:"
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, :].to(device)
    
    for _ in range(max_length):
        logits, _ = model(tokens)
        logits = logits[0, -1, :]
        logits = logits / max(temperature, 0.01)
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == EOS_ID or next_token.item() == enc.encode('\n')[0]:
            break
        
        tokens = torch.cat([tokens, next_token[None, :]], dim=1)
    
    full_text = enc.decode(tokens[0].tolist())
    
    if "A:" in full_text:
        answer = full_text.split("A:", 1)[1].strip()
        answer = answer.replace("<|endoftext|>", "").strip()
        return answer if answer else "I don't know."
    
    return full_text

def qa_interface(question, temperature, top_k):
    """Gradio interface"""
    try:
        return generate_answer(question, temperature, top_k)
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# Gradio Interface
# ============================================================

with gr.Blocks(title="GPT-2 Q&A") as demo:
    gr.Markdown("# GPT-2 Q&A Assistant\n\nAsk me anything!")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the capital of Australia?",
                lines=2
            )
            
            with gr.Row():
                temperature_slider = gr.Slider(0.1, 1.0, value=0.6, step=0.1, label="Temperature")
                topk_slider = gr.Slider(5, 50, value=15, step=5, label="Top-K")
            
            submit_btn = gr.Button("Ask Question", variant="primary")
            
        answer_output = gr.Textbox(label="Answer", lines=5, interactive=False)
    
    gr.Examples(
        examples=[
            ["What is the capital of Australia?", 0.6, 15],
            ["What does 'arvo' mean?", 0.6, 15],
            ["What is RAM?", 0.6, 15],
            ["What is overfitting?", 0.6, 15],
        ],
        inputs=[question_input, temperature_slider, topk_slider],
        outputs=answer_output,
        fn=qa_interface,
    )
    
    submit_btn.click(
        fn=qa_interface,
        inputs=[question_input, temperature_slider, topk_slider],
        outputs=answer_output
    )
    
    # Footer with links
    gr.Markdown("""
    ---
    
    Website: [devshubh.me](https://devshubh.me)  
    LinkedIn: [linkedin.com/in/shubharthaksangharsha](https://linkedin.com/in/shubharthaksangharsha/)  
    GitHub: [github.com/shubharthaksangharsha/karpathy](https://github.com/shubharthaksangharsha/karpathy)
    
    <div style="text-align: center; margin-top: 20px; color: #888; font-size: 0.9em;">
        Built by Shubharthak Sangharsha | Powered by GPT-2 124M
    </div>
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Web App")
    print("="*60)
    demo.launch(
        share=True,
        server_port=7860,
        show_error=True,
    )


