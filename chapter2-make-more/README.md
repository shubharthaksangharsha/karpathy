# Chapter 2: Make More - Bigram Language Model

## Overview
This chapter introduces the fundamentals of language modeling by building a character-level bigram model to generate names. It's the first step toward understanding how neural networks can learn patterns in text data.

## 📁 Files
- `Building-make-more.ipynb` - Main implementation notebook
- `names.txt` - Dataset of names for training

## 🎯 Learning Objectives
- Understand what language modeling is and why it's important
- Learn about character-level vs word-level modeling
- Implement a bigram model from scratch
- Understand probability distributions and sampling
- Learn about model evaluation using negative log-likelihood

## 🔧 Key Concepts

### Bigram Model
A bigram model predicts the next character given the previous character:
- P(next_char | current_char)
- Simple but surprisingly effective for character-level tasks
- Forms the foundation for more complex language models

### Character-Level Processing
- Text is treated as a sequence of characters
- Special tokens: `<S>` (start) and `<E>` (end)
- Vocabulary consists of all unique characters in the dataset

### Probability Estimation
- Count-based approach: P(a|b) = count(b,a) / count(b)
- Add-one smoothing to handle unseen bigrams
- Convert counts to probabilities

## 🚀 Implementation Highlights

### Data Preprocessing
```python
# Build vocabulary
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0  # Special end token
itos = {i:s for s,i in stoi.items()}
```

### Bigram Counting
```python
# Count all bigrams
N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = stoi[ch1], stoi[ch2]
        N[ix1, ix2] += 1
```

### Probability Calculation
```python
# Convert counts to probabilities
P = (N + 1).float()  # Add-one smoothing
P /= P.sum(1, keepdim=True)
```

## 📊 Model Performance

### Training Data
- 32,033 names in the dataset
- Character vocabulary size: 27 (26 letters + special token)
- Bigram matrix: 27×27 transition probabilities

### Evaluation
- Negative Log-Likelihood (NLL) as loss function
- Lower NLL indicates better model performance
- Model learns character transition patterns

## 🎓 Key Takeaways

1. **Language Modeling**: The task of predicting the next token given previous context
2. **Character-Level Processing**: Working with individual characters rather than words
3. **Probability Distributions**: Understanding how models represent uncertainty
4. **Model Evaluation**: Using NLL to measure model quality
5. **Sampling**: Generating text by sampling from learned distributions

## 🔗 Next Steps
This bigram model sets the foundation for:
- More complex n-gram models
- Neural network-based language models
- Understanding the relationship between count-based and neural approaches

## 📚 Prerequisites
- Basic Python programming
- Understanding of probability and statistics
- Basic linear algebra (matrix operations)

## 🛠️ Dependencies
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## 🎯 Example Output
The model learns to generate names like:
- `mora.`
- `mayah.`
- `seel.`
- `ndheyah.`

While not perfect, it demonstrates the model's ability to learn character-level patterns in names.
