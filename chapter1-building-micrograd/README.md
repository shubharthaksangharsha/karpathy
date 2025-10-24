# Chapter 1: Building Micrograd

## Overview
This chapter implements a tiny autograd engine called "micrograd" from scratch. It's the foundation for understanding how automatic differentiation works in modern deep learning frameworks like PyTorch.

## 📁 Files
- `mgrad.ipynb` - Main implementation notebook

## 🎯 Learning Objectives
- Understand computational graphs and how they represent mathematical expressions
- Implement automatic differentiation from first principles
- Learn how gradients flow backward through a computational graph
- Understand the chain rule and its application in neural networks

## 🔧 Key Concepts

### Value Class
The core of micrograd is the `Value` class that represents a scalar value in a computational graph:
- Stores data (the actual value)
- Tracks gradients
- Maintains references to child nodes
- Records the operation that created it

### Computational Graph
- Each `Value` object is a node in the graph
- Edges represent operations between values
- The graph enables automatic gradient computation

### Backpropagation
- Forward pass: compute the output value
- Backward pass: compute gradients using the chain rule
- Gradients flow backward through the graph

## 🚀 Implementation Highlights

### Basic Operations
```python
class Value:
    def __add__(self, other):
        # Addition operation
    def __mul__(self, other):
        # Multiplication operation
    def __pow__(self, other):
        # Power operation
```

### Visualization
The notebook includes functions to visualize computational graphs using Graphviz, showing:
- Node values and gradients
- Operation types
- Graph structure

## 📊 Example Usage

```python
# Create values
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Build computation
e = a * b
d = e + c
f = Value(-2.0, label='f')
L = d * f

# Forward pass
print(f"Loss: {L.data}")

# Backward pass
L.grad = 1.0
L.backward()

# Check gradients
print(f"dL/da: {a.grad}")
print(f"dL/db: {b.grad}")
```

## 🎓 Key Takeaways

1. **Automatic Differentiation**: The engine automatically computes gradients for any differentiable expression
2. **Chain Rule**: Gradients are computed using the chain rule, flowing backward through the graph
3. **Computational Efficiency**: The graph structure allows efficient gradient computation
4. **Foundation for Deep Learning**: This is the core mechanism behind PyTorch's autograd system

## 🔗 Next Steps
After understanding micrograd, you'll be ready to:
- Build neural networks using this autograd engine
- Understand how PyTorch's autograd works
- Implement more complex operations and activation functions

## 📚 Prerequisites
- Basic Python programming
- Understanding of derivatives and the chain rule
- Basic linear algebra concepts

## 🛠️ Dependencies
- Python 3.6+
- NumPy
- Matplotlib
- Graphviz (for visualization)
