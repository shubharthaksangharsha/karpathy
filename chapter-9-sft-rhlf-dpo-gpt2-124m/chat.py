import torch
import tiktoken
import torch.nn.functional as F
from train_gpt2_v2 import GPT, GPTConfig

# ✅ Use the BEST checkpoint (from step 120, before overfitting)
MODEL_CKPT = "qa-sft_best.pt"  # Step 120 checkpoint
BASE_CKPT = "checkpoints/model_09535.pt"  # For config

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token

def load_model():
    print(f"🔄 Loading model from {MODEL_CKPT} ...")
    
    # Load SFT checkpoint
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=False)
    
    # If config not in checkpoint, load from base model
    if "config" not in ckpt:
        print("   Loading config from base checkpoint...")
        base_ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
        config = base_ckpt["config"]
    else:
        config = ckpt["config"]
    
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    step = ckpt.get("step", "unknown")
    val_loss = ckpt.get("val_loss", "unknown")
    print(f"✅ Model loaded!")
    print(f"   Step: {step}")
    print(f"   Val loss: {val_loss}")
    return model, device

@torch.no_grad()
def generate_answer(model, question, device, max_new=100, temperature=0.6, top_k=15):
    """
    Generate Q&A with AGGRESSIVE stopping for short answers.
    
    Lower temperature + lower top_k = more focused, factual answers.
    """
    # Match training format EXACTLY
    prompt = f"Q: {question}\nA:"
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, :].to(device)
    
    generated_tokens = []
    
    for _ in range(max_new):
        # Forward pass
        logits, _ = model(tokens)
        logits = logits[0, -1, :]  # Last token logits
        
        # Apply temperature (lower = more confident)
        logits = logits / temperature
        
        # Top-k filtering (lower = less random)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop at EOS
        if next_token.item() == EOS_ID:
            break
        
        generated_tokens.append(next_token.item())
        tokens = torch.cat([tokens, next_token[None, :]], dim=1)
        
        # AGGRESSIVE STOP: Stop at newline (end of answer)
        if next_token.item() == enc.encode('\n')[0]:
            break
    
    # Decode only the answer part (strip prompt)
    full_text = enc.decode(tokens[0].tolist())
    
    # Extract answer after "A:"
    if "A:" in full_text:
        answer = full_text.split("A:", 1)[1].strip()
        # Clean up any <|endoftext|> markers
        answer = answer.replace("<|endoftext|>", "").strip()
        return answer
    
    return full_text

def run_quick_test(model, device):
    """Quick quality test with common questions"""
    print("\n" + "="*60)
    print("🧪 QUICK QUALITY TEST")
    print("="*60 + "\n")
    
    test_questions = [
        "What is the capital of Australia?",
        "What does 'arvo' mean?",
        "What is a koala's main food?",
        "What is RAM?",
        "What is overfitting?",
    ]
    
    for i, q in enumerate(test_questions, 1):
        answer = generate_answer(model, q, device, temperature=0.5, top_k=10)
        print(f"[{i}] Q: {q}")
        print(f"    A: {answer}\n")
    
    print("="*60 + "\n")

def chat():
    model, device = load_model()
    
    # Run quick test first
    run_quick_test(model, device)
    
    print("🟢 Interactive Q&A mode. Type your question or 'exit' to quit.")
    print("💡 Tip: Ask short, factual questions for best results.\n")
    
    while True:
        user_input = input("🧑 You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Generate answer with optimized settings
        answer = generate_answer(
            model, 
            user_input, 
            device,
            temperature=0.6,  # Balanced creativity/accuracy
            top_k=15          # Focus on likely tokens
        )
        
        print(f"🤖 Bot: {answer}\n")

if __name__ == "__main__":
    chat()
