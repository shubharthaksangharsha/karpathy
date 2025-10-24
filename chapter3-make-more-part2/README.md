# Chapter 3: Make More Part 2 - MLP

## Overview
This chapter extends the bigram model to a Multi-Layer Perceptron (MLP) with embeddings and hidden layers. It introduces neural networks as a more powerful approach to language modeling compared to simple count-based methods.

## 📁 Files
- `building-make-more-mlp.ipynb` - Main implementation notebook
- `names.txt` - Dataset of names for training

## 🎯 Learning Objectives
- Understand the transition from count-based to neural network approaches
- Learn about character embeddings and their role in neural language models
- Implement a multi-layer perceptron for character prediction
- Understand the relationship between neural networks and traditional n-gram models
- Learn about training neural networks with gradient descent

## 🔧 Key Concepts

### Character Embeddings
- Each character is represented as a dense vector
- Embeddings are learned during training
- Allow the model to capture similarities between characters

### Multi-Layer Perceptron (MLP)
- Input layer: concatenated character embeddings
- Hidden layer: non-linear transformation
- Output layer: probability distribution over next character

### Neural Network Architecture
```
Input: [char1, char2, char3] (context window)
↓
Embeddings: [emb1, emb2, emb3]
↓
Concatenate: [emb1, emb2, emb3]
↓
Hidden Layer: tanh(W1 * input + b1)
↓
Output Layer: W2 * hidden + b2
↓
Softmax: probabilities over vocabulary
```

## 🚀 Implementation Highlights

### Dataset Construction
```python
def build_dataset(words):
    block_size = 3  # context length
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)
```

### Neural Network Parameters
```python
# Embedding layer
C = torch.randn((27, 10))  # 27 chars, 10-dim embeddings

# Hidden layer
W1 = torch.randn((30, 200))  # 30 inputs (3*10), 200 hidden
b1 = torch.randn(200)

# Output layer
W2 = torch.randn((200, 27))  # 200 hidden, 27 outputs
b2 = torch.randn(27)
```

### Forward Pass
```python
# Forward pass
emb = C[Xb]  # (B, 3, 10)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (B, 200)
logits = h @ W2 + b2  # (B, 27)
loss = F.cross_entropy(logits, Yb)
```

## 📊 Model Performance

### Training Results
- **Parameters**: ~12K total parameters
- **Context Length**: 3 characters
- **Embedding Dimension**: 10
- **Hidden Units**: 200
- **Training**: 50,000 iterations with learning rate 0.01

### Generated Names
The model learns to generate more realistic names:
- `mora.`
- `mayah.`
- `seel.`
- `ndheyah.`
- `reisha.`

## 🎓 Key Takeaways

1. **Neural Networks vs Count-Based**: Neural networks can learn more complex patterns than simple counting
2. **Embeddings**: Dense vector representations capture character similarities
3. **Context Windows**: Using multiple previous characters improves prediction
4. **Training Dynamics**: Understanding how neural networks learn through gradient descent
5. **Model Evaluation**: Using cross-entropy loss for training and evaluation

## 🔗 Relationship to Bigram Model

### Similarities
- Both predict next character given context
- Both use probability distributions
- Both can be evaluated with NLL

### Differences
- Neural networks learn distributed representations
- Can capture non-linear relationships
- More parameters but more expressive
- Requires training rather than counting

## 📚 Prerequisites
- Chapter 2 (Bigram Model)
- Basic understanding of neural networks
- Linear algebra (matrix operations)
- Calculus (gradients and optimization)

## 🛠️ Dependencies
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## 🎯 Next Steps
This MLP model prepares you for:
- Deeper neural networks
- Batch normalization
- More sophisticated architectures
- Understanding training dynamics

## 💡 Key Insights
- Neural networks can learn the same patterns as count-based models but more flexibly
- Embeddings provide a powerful way to represent discrete tokens
- The relationship between neural networks and traditional methods is important to understand
