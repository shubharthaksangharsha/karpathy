#!/usr/bin/env python3
"""
Gradio Web App for GPT-2 Q&A Model
Creates a shareable link for testing!
"""
import torch
import tiktoken
import torch.nn.functional as F
from train_gpt2_v2 import GPT, GPTConfig
import gradio as gr

# Configuration
MODEL_CKPT = "qa-sft_best.pt"
BASE_CKPT = "checkpoints/model_09535.pt"

print("🚀 Loading GPT-2 Q&A Model...")

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.eot_token

# Load model
def load_model():
    """Load the trained Q&A model"""
    print(f"📂 Loading checkpoint: {MODEL_CKPT}")
    
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
    
    print(f"✅ Model loaded!")
    print(f"   Step: {step}")
    print(f"   Val Loss: {val_loss}")
    print(f"   Device: {device}")
    
    return model, device

# Load model globally
model, device = load_model()

@torch.no_grad()
def generate_answer(question, temperature=0.6, top_k=15, max_length=100):
    """
    Generate answer for a given question
    
    Args:
        question: User's question
        temperature: Sampling temperature (0.1-1.0)
        top_k: Top-k sampling (5-50)
        max_length: Maximum tokens to generate
    """
    if not question or not question.strip():
        return "Please enter a question!"
    
    # Match training format
    prompt = f"Q: {question.strip()}\nA:"
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)[None, :].to(device)
    
    generated_tokens = []
    
    for _ in range(max_length):
        # Forward pass
        logits, _ = model(tokens)
        logits = logits[0, -1, :]
        
        # Apply temperature
        logits = logits / max(temperature, 0.01)
        
        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')
        
        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop at EOS or newline
        if next_token.item() == EOS_ID or next_token.item() == enc.encode('\n')[0]:
            break
        
        generated_tokens.append(next_token.item())
        tokens = torch.cat([tokens, next_token[None, :]], dim=1)
    
    # Decode answer
    full_text = enc.decode(tokens[0].tolist())
    
    # Extract answer after "A:"
    if "A:" in full_text:
        answer = full_text.split("A:", 1)[1].strip()
        answer = answer.replace("<|endoftext|>", "").strip()
        return answer if answer else "I don't know."
    
    return full_text

def qa_interface(question, temperature, top_k):
    """Gradio interface function"""
    try:
        answer = generate_answer(question, temperature, top_k)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="GPT-2 Q&A Assistant") as demo:
    gr.Markdown("""
    # 🤖 GPT-2 Q&A Assistant
    
    **Model**: GPT-2 124M fine-tuned on Q&A dataset
    **Training**: Step 120, Val Loss 1.91
    
    Ask me questions about:
    - 🇦🇺 Australian facts
    - 💻 Technology & computers  
    - 🧠 Machine learning concepts
    - 📚 General knowledge
    
    *Note: This is a small model (124M params), so expect ~30-40% accuracy on factual questions.*
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What is the capital of Australia?",
                lines=2
            )
            
            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    label="Temperature (lower = more focused)"
                )
                topk_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=15,
                    step=5,
                    label="Top-K (lower = less random)"
                )
            
            submit_btn = gr.Button("Ask Question 🚀", variant="primary")
            clear_btn = gr.Button("Clear")
            
        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="Answer",
                lines=5,
                interactive=False
            )
    
    # Example questions
    gr.Markdown("### 💡 Try these example questions:")
    gr.Examples(
        examples=[
            ["What is the capital of Australia?", 0.6, 15],
            ["What does 'arvo' mean?", 0.6, 15],
            ["What is a koala's main food?", 0.6, 15],
            ["What is RAM?", 0.6, 15],
            ["What is overfitting?", 0.6, 15],
            ["What is a neural network?", 0.6, 15],
            ["What is GPU?", 0.6, 15],
            ["How many hours in a day?", 0.5, 10],
        ],
        inputs=[question_input, temperature_slider, topk_slider],
        outputs=answer_output,
        fn=qa_interface,
    )
    
    # Button actions
    submit_btn.click(
        fn=qa_interface,
        inputs=[question_input, temperature_slider, topk_slider],
        outputs=answer_output
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[question_input, answer_output]
    )
    
    gr.Markdown("""
    ---
    ### 📊 Model Info:
    - **Architecture**: GPT-2 124M parameters
    - **Training**: Supervised Fine-Tuning on 1,336 Q&A pairs
    - **Best Checkpoint**: Step 120
    - **Validation Loss**: 1.9147
    - **Expected Accuracy**: 30-40% on diverse factual questions
    
    ### 🎯 Tips for Better Results:
    - Ask short, simple questions
    - Use clear, direct language
    - Lower temperature (0.3-0.5) for factual questions
    - Higher temperature (0.7-0.9) for creative answers
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio Web App")
    print("="*60)
    print("\n📝 The app will create two URLs:")
    print("   1. Local URL: http://127.0.0.1:7860 (for you)")
    print("   2. Public URL: https://xxxxx.gradio.live (shareable!)")
    print("\n💡 Share the public URL with others to test the model!")
    print("   Public link expires after 72 hours\n")
    print("="*60 + "\n")
    
    # Launch with share=True to get public URL
    demo.launch(
        share=True,          # ✅ Creates shareable public link!
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )

