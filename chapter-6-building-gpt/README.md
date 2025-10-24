# Chapter 6: Building GPT

## Overview
This chapter implements a complete GPT (Generative Pre-trained Transformer) from scratch. It covers self-attention, multi-head attention, and the full transformer architecture that revolutionized natural language processing.

## 📁 Files
- `gpt-dev.ipynb` - Main GPT development notebook
- `my_first_nano_gpt_train.ipynb` - Training notebook for nano GPT
- `bigram.py` - Bigram model implementation
- `infinite_shakespare.py` - Infinite Shakespeare generation
- `v2.py` - Version 2 implementation
- `shakespeare_model.pth` - Trained model weights
- `input.txt` - Shakespeare text dataset
- `infinite_shakespeare.txt` - Generated text output

## 🎯 Learning Objectives
- Understand the transformer architecture and its components
- Implement self-attention and multi-head attention from scratch
- Learn about positional encoding and its role in transformers
- Understand the relationship between transformers and RNNs
- Build a complete language model using the GPT architecture

## 🔧 Key Concepts

### Self-Attention
- **Query, Key, Value**: The fundamental components of attention
- **Attention Scores**: How much each position attends to every other position
- **Scaled Dot-Product Attention**: The mathematical formulation of attention

### Multi-Head Attention
- **Parallel Attention**: Multiple attention heads processing simultaneously
- **Different Representations**: Each head can focus on different aspects
- **Concatenation**: Combining outputs from multiple heads

### Transformer Architecture
- **Encoder-Decoder**: Original transformer design
- **Decoder-Only**: GPT-style architecture
- **Layer Normalization**: Stabilizing training in transformers
- **Residual Connections**: Improving gradient flow

## 🚀 Implementation Highlights

### Self-Attention Implementation
```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out
```

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

### Feed-Forward Network
```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

### Transformer Block
```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

## 📊 Model Architecture

### Hyperparameters
- **Batch Size**: 64
- **Block Size**: 256 (context length)
- **Embedding Dimension**: 384
- **Number of Heads**: 6
- **Number of Layers**: 6
- **Dropout**: 0.2
- **Learning Rate**: 3e-4

### Training Results
- **Training Loss**: 1.769
- **Validation Loss**: 1.993
- **Parameters**: ~10M total parameters
- **Training Time**: 5000 iterations

## 🎓 Key Takeaways

1. **Self-Attention**: The core mechanism that allows transformers to process sequences
2. **Multi-Head Attention**: Parallel processing of different attention patterns
3. **Positional Encoding**: How transformers understand sequence order
4. **Layer Normalization**: Essential for training deep transformer networks
5. **Residual Connections**: Improving gradient flow in deep networks

## 🔧 Technical Details

### Attention Mechanism
- **Scaled Dot-Product**: Prevents attention scores from becoming too large
- **Causal Masking**: Ensures autoregressive property (no future information)
- **Dropout**: Regularization during training

### Positional Encoding
```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
pos_emb = self.position_embedding_table(torch.arange(T, device=device))
x = tok_emb + pos_emb
```

### Training Dynamics
- **AdamW Optimizer**: Better weight decay than Adam
- **Learning Rate Scheduling**: Step decay for better convergence
- **Gradient Clipping**: Preventing exploding gradients

## 📚 Prerequisites
- Chapter 5 (RNNs and Hierarchical Processing)
- Understanding of attention mechanisms
- Linear algebra (matrix operations)
- Basic knowledge of neural networks

## 🛠️ Dependencies
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## 🎯 Next Steps
This chapter prepares you for:
- Understanding modern language models
- Working with pre-trained transformers
- Implementing more advanced architectures
- Understanding the evolution from RNNs to Transformers

## 💡 Key Insights
- **Transformers vs RNNs**: Transformers can process sequences in parallel
- **Attention is All You Need**: The title of the original transformer paper
- **Scalability**: Transformers scale better than RNNs
- **Modern NLP**: Most state-of-the-art models are based on transformers

## 🔗 Generated Text Example
The trained model generates Shakespeare-like text:
```
BUCKINGHAM:
Good part, as ends me, to come.
But my Carspidel. O Lord God, that 'tis word
After me worth interr'd true in this doverty men;
Upon every my slaving with side,
And his swellips sleet legs in him!
```
