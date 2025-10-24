# Neural Networks: Zero to Hero

A comprehensive collection of machine learning and neural network implementations following Andrej Karpathy's "Neural Networks: Zero to Hero" course. This repository contains hands-on implementations from basic neural networks to advanced transformer architectures.

## 📚 Table of Contents

### Chapter 1: Building Micrograd
**Directory:** `chapter1-building-micrograd/`
- **Notebook:** `mgrad.ipynb`
- **Description:** Implementation of a tiny autograd engine (micrograd) from scratch. Learn the fundamentals of automatic differentiation, computational graphs, and backpropagation.
- **Key Concepts:** Value class, computational graphs, gradient computation, chain rule
- **Files:** `mgrad.ipynb`

### Chapter 2: Make More - Bigram Language Model
**Directory:** `chapter2-make-more/`
- **Notebook:** `Building-make-more.ipynb`
- **Description:** Building a character-level bigram language model to generate names. Introduction to language modeling and probability distributions.
- **Key Concepts:** Bigram models, character-level prediction, probability distributions, text generation
- **Files:** `Building-make-more.ipynb`, `names.txt`

### Chapter 3: Make More Part 2 - MLP
**Directory:** `chapter3-make-more-part2/`
- **Notebook:** `building-make-more-mlp.ipynb`
- **Description:** Extending the bigram model to a Multi-Layer Perceptron (MLP) with embeddings and hidden layers.
- **Key Concepts:** Neural networks, embeddings, hidden layers, cross-entropy loss, optimization
- **Files:** `building-make-more-mlp.ipynb`, `names.txt`

### Chapter 4: Make More Part 3 - Activations, Gradients & BatchNorm
**Directory:** `chapter4-make-more-activations-gradients-batchnorm/`
- **Notebook:** `building-make-more-mlp-2.ipynb`
- **Description:** Deep dive into neural network internals: activation functions, gradient flow, batch normalization, and training dynamics.
- **Key Concepts:** Tanh activations, gradient analysis, batch normalization, training stability, deeper networks
- **Files:** `building-make-more-mlp-2.ipynb`, `names.txt`

### Chapter 5: Building WaveNet - RNNs
**Directory:** `chapter-5-building-wavenet/`
- **Notebook:** `building-makemore-part5-rnns.ipynb`
- **Description:** Introduction to Recurrent Neural Networks (RNNs) and hierarchical processing for sequence modeling.
- **Key Concepts:** RNNs, hierarchical processing, sequence modeling, context windows
- **Files:** `building-makemore-part5-rnns.ipynb`, `convolutional_example.py`, `convolutional_implementation.py`, `names.txt`

### Chapter 6: Building GPT
**Directory:** `chapter-6-building-gpt/`
- **Notebooks:** `gpt-dev.ipynb`, `my_first_nano_gpt_train.ipynb`
- **Description:** Building a complete GPT (Generative Pre-trained Transformer) from scratch. Implementation of self-attention, multi-head attention, and transformer architecture.
- **Key Concepts:** Self-attention, multi-head attention, transformers, GPT architecture, positional encoding
- **Files:** `gpt-dev.ipynb`, `my_first_nano_gpt_train.ipynb`, `bigram.py`, `infinite_shakespare.py`, `v2.py`, `shakespeare_model.pth`, `input.txt`, `infinite_shakespeare.txt`

### Chapter 7: Building Tokenization
**Directory:** `chapter-7-building-tokenization/`
- **Notebook:** `building-tokenization.ipynb`
- **Description:** Understanding and implementing tokenization techniques for language models, including BPE (Byte Pair Encoding).
- **Key Concepts:** Tokenization, BPE, vocabulary building, text preprocessing
- **Files:** `building-tokenization.ipynb`, `encoder.json`, `tok400.model`, `tok400.vocab`, `vocab.bpe`, `toy.txt`, `resources/`

### Chapter 8: Building GPT-2 124M
**Directory:** `chapter-8-building-gpt2-124m/`
- **Notebook:** `play.ipynb`
- **Description:** Scaling up to GPT-2 124M parameters. Working with pre-trained models and understanding large-scale transformer architectures.
- **Key Concepts:** Large-scale transformers, pre-trained models, model scaling, Hugging Face transformers
- **Files:** `play.ipynb`, `train_gpt2v1.py`

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Jupyter Notebook
- NumPy
- Matplotlib
- Graphviz (for computational graphs)

### Installation
```bash
pip install torch torchvision torchaudio
pip install jupyter matplotlib graphviz
pip install transformers  # for Chapter 8
```

### Running the Notebooks
1. Clone this repository
2. Navigate to any chapter directory
3. Open the Jupyter notebook
4. Run the cells sequentially

## 📖 Learning Path

1. **Start with Chapter 1** - Understand the fundamentals of automatic differentiation
2. **Progress through Chapters 2-4** - Build increasingly complex language models
3. **Explore RNNs in Chapter 5** - Learn about recurrent architectures
4. **Master Transformers in Chapter 6** - Implement GPT from scratch
5. **Understand Tokenization in Chapter 7** - Learn text preprocessing
6. **Scale Up in Chapter 8** - Work with large pre-trained models

## 🎯 Key Learning Outcomes

- **Automatic Differentiation**: Understanding how gradients flow through computational graphs
- **Language Modeling**: Building models that can generate coherent text
- **Neural Network Architecture**: From simple MLPs to complex transformers
- **Training Dynamics**: Understanding optimization, batch normalization, and training stability
- **Modern NLP**: Working with state-of-the-art transformer architectures

## 📁 Repository Structure

```
kapathy/
├── chapter1-building-micrograd/
├── chapter2-make-more/
├── chapter3-make-more-part2/
├── chapter4-make-more-activations-gradients-batchnorm/
├── chapter-5-building-wavenet/
├── chapter-6-building-gpt/
├── chapter-7-building-tokenization/
└── chapter-8-building-gpt2-124m/
```

## 🤝 Contributing

This repository follows the structure and content from Andrej Karpathy's "Neural Networks: Zero to Hero" course. Feel free to experiment with the code and extend the implementations!

## 📚 Additional Resources

- [Neural Networks: Zero to Hero - YouTube Playlist](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=SH5UXoaWnx7mwTaW)
- [Karpathy's Blog](https://karpathy.github.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 📄 License

This educational content is based on Andrej Karpathy's course materials and is intended for learning purposes.
