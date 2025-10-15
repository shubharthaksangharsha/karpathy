"""
Practical example of applying convolutions to the makemore model
This shows how to modify the existing notebook to use convolutions
"""

import torch
import torch.nn.functional as F

# Add this Conv1d class to your notebook
class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Kaiming initialization
        fan_in = in_channels * kernel_size
        self.weight = torch.randn(out_channels, in_channels, kernel_size) / fan_in**0.5
        self.bias = torch.zeros(out_channels) if bias else None
    
    def __call__(self, x):
        # x: (batch_size, in_channels, sequence_length)
        self.out = F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

# Modified BatchNorm1d for conv1d
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 2)  # For conv1d: normalize over batch and sequence
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

# Reshape layer to convert from (B, T, C) to (B, C, T) for conv1d
class ReshapeForConv:
    def __call__(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        self.out = x.transpose(1, 2)
        return self.out
    
    def parameters(self):
        return []

# Global average pooling
class GlobalAvgPool:
    def __call__(self, x):
        # x: (batch, channels, seq_len)
        self.out = x.mean(dim=2)  # Average over sequence length
        return self.out
    
    def parameters(self):
        return []

# Keep your existing classes: Linear, Tanh, Embedding, Sequential

# CONVOLUTIONAL MODEL - Replace your current model with this:
def create_conv_model(vocab_size, n_embd=24, n_hidden=128):
    model = Sequential([
        Embedding(vocab_size, n_embd),           # (B, T, n_embd)
        ReshapeForConv(),                        # (B, n_embd, T)
        
        # First conv: captures 2-grams
        Conv1d(n_embd, n_hidden, kernel_size=2, padding=1),  # (B, n_hidden, T)
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Second conv: captures 3-grams  
        Conv1d(n_hidden, n_hidden, kernel_size=3, padding=1),  # (B, n_hidden, T)
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Third conv: captures 4-grams
        Conv1d(n_hidden, n_hidden, kernel_size=4, padding=1),  # (B, n_hidden, T)
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Global pooling + final linear
        GlobalAvgPool(),                         # (B, n_hidden)
        Linear(n_hidden, vocab_size),           # (B, vocab_size)
    ])
    return model

# ALTERNATIVE: Multi-scale approach (processes multiple n-gram sizes in parallel)
class MultiScaleConv:
    def __init__(self, in_channels, out_channels):
        self.conv1 = Conv1d(in_channels, out_channels//4, kernel_size=1, padding=0)  # 1-grams
        self.conv2 = Conv1d(in_channels, out_channels//4, kernel_size=2, padding=1)  # 2-grams  
        self.conv3 = Conv1d(in_channels, out_channels//4, kernel_size=3, padding=1)  # 3-grams
        self.conv4 = Conv1d(in_channels, out_channels//4, kernel_size=4, padding=1)  # 4-grams
    
    def __call__(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x) 
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        self.out = torch.cat([out1, out2, out3, out4], dim=1)
        return self.out
    
    def parameters(self):
        return (self.conv1.parameters() + self.conv2.parameters() + 
               self.conv3.parameters() + self.conv4.parameters())

def create_multiscale_conv_model(vocab_size, n_embd=24, n_hidden=128):
    model = Sequential([
        Embedding(vocab_size, n_embd),
        ReshapeForConv(),
        
        MultiScaleConv(n_embd, n_hidden),
        BatchNorm1d(n_hidden),
        Tanh(),
        
        MultiScaleConv(n_hidden, n_hidden),
        BatchNorm1d(n_hidden),
        Tanh(),
        
        GlobalAvgPool(),
        Linear(n_hidden, vocab_size),
    ])
    return model

# USAGE IN YOUR NOTEBOOK:
# Replace this line in your notebook:
# model = Sequential([...])  # your current hierarchical model

# With this:
# model = create_conv_model(vocab_size, n_embd=24, n_hidden=128)
# OR
# model = create_multiscale_conv_model(vocab_size, n_embd=24, n_hidden=128)

# The rest of your training loop remains exactly the same!
