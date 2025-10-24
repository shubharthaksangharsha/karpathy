# Chapter 5: Building WaveNet - RNNs

## Overview
This chapter introduces Recurrent Neural Networks (RNNs) and hierarchical processing for sequence modeling. It builds upon the previous MLP approach by adding temporal dependencies and more sophisticated architectures.

## 📁 Files
- `building-makemore-part5-rnns.ipynb` - Main implementation notebook
- `convolutional_example.py` - Convolutional layer examples
- `convolutional_implementation.py` - Convolutional layer implementation
- `names.txt` - Dataset of names for training

## 🎯 Learning Objectives
- Understand the limitations of feedforward networks for sequence modeling
- Learn about Recurrent Neural Networks (RNNs) and their applications
- Implement hierarchical processing for better sequence understanding
- Understand the relationship between RNNs and convolutions
- Learn about context windows and temporal dependencies

## 🔧 Key Concepts

### Recurrent Neural Networks (RNNs)
- **Temporal Dependencies**: Networks that can remember previous states
- **Sequential Processing**: Processing sequences one element at a time
- **Hidden State**: Maintaining information across time steps

### Hierarchical Processing
- **Multi-level Features**: Learning features at different time scales
- **Context Aggregation**: Combining information from different time steps
- **Efficient Processing**: Reducing computational complexity through hierarchy

### Context Windows
- **Longer Context**: Using more previous characters for prediction
- **Block Size**: Increasing from 3 to 8 characters
- **Better Predictions**: More context leads to better language modeling

## 🚀 Implementation Highlights

### Hierarchical Network Architecture
```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    FlattenConsecutive(2), Linear(n_embd * 2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
```

### FlattenConsecutive Layer
```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n
    
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        return x
```

### Dataset with Longer Context
```python
block_size = 8  # Increased context length
def build_dataset(words):
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

## 📊 Model Performance

### Architecture Comparison
- **Original MLP**: 3-char context, 200 hidden, 12K params → Train: 2.058, Val: 2.105
- **Extended Context**: 8-char context, 22K params → Train: 1.918, Val: 2.027
- **Hierarchical**: 8-char context, 22K params → Train: 1.941, Val: 2.029
- **Scaled Up**: 8-char context, 76K params → Train: 1.769, Val: 1.993

### Generated Names
The hierarchical model generates more coherent names:
- `aiyanah.`
- `giusopf.`
- `lorron.`
- `roger.`
- `rhyitte.`
- `christell.`

## 🎓 Key Takeaways

1. **Temporal Dependencies**: RNNs can capture long-range dependencies in sequences
2. **Hierarchical Processing**: Multi-level feature extraction improves performance
3. **Context Windows**: Longer context leads to better language modeling
4. **Architecture Design**: Careful design of network architecture is crucial
5. **Parameter Efficiency**: Hierarchical approaches can be more parameter-efficient

## 🔧 Technical Details

### FlattenConsecutive Benefits
- **Reduced Sequence Length**: Combines consecutive time steps
- **Increased Feature Dimension**: More information per time step
- **Computational Efficiency**: Fewer operations for longer sequences

### Batch Normalization in RNNs
- **Training Stability**: Helps with gradient flow in recurrent networks
- **Faster Convergence**: Reduces training time
- **Better Generalization**: Improves model performance

## 📚 Prerequisites
- Chapter 4 (Deep MLPs and BatchNorm)
- Understanding of sequence modeling
- Basic knowledge of RNNs
- Linear algebra (matrix operations)

## 🛠️ Dependencies
- Python 3.6+
- PyTorch
- NumPy
- Matplotlib

## 🎯 Next Steps
This chapter prepares you for:
- More advanced RNN architectures (LSTM, GRU)
- Transformer architectures
- Understanding the evolution from RNNs to Transformers
- Modern sequence modeling approaches

## 💡 Key Insights
- **RNNs vs MLPs**: RNNs can capture temporal dependencies that MLPs cannot
- **Hierarchical Processing**: Multi-level feature extraction is powerful
- **Context Matters**: Longer context windows improve language modeling
- **Architecture Evolution**: Understanding the progression from simple to complex models

## 🔗 Connection to WaveNet
This chapter introduces concepts that are fundamental to WaveNet:
- **Dilated Convolutions**: Efficient processing of long sequences
- **Hierarchical Features**: Multi-scale feature extraction
- **Causal Convolutions**: Maintaining temporal order
- **Residual Connections**: Improving gradient flow in deep networks
