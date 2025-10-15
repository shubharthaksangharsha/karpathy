import torch
import torch.nn.functional as F

# Convolutional implementation for the makemore model

class Conv1d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights using Kaiming initialization
        fan_in = in_channels * kernel_size
        self.weight = torch.randn(out_channels, in_channels, kernel_size) / fan_in**0.5
        self.bias = torch.zeros(out_channels) if bias else None
    
    def __call__(self, x):
        # x shape: (batch_size, in_channels, sequence_length)
        # Apply convolution
        self.out = F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

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
                dim = (0, 2)  # For conv1d: (batch, channels, sequence)
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

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
    
    def parameters(self):
        return [self.weight]

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Convolutional model architecture
def create_conv_model(vocab_size, n_embd=24, n_hidden=128):
    """
    Create a convolutional model that replaces the hierarchical approach
    """
    model = Sequential([
        Embedding(vocab_size, n_embd),
        # Reshape for conv1d: (batch, seq_len, embd) -> (batch, embd, seq_len)
        # We'll handle this reshape in the forward pass
        
        # First conv layer: kernel_size=2, captures bigrams
        Conv1d(n_embd, n_hidden, kernel_size=2, padding=1),  # padding=1 to maintain sequence length
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Second conv layer: kernel_size=3, captures trigrams
        Conv1d(n_hidden, n_hidden, kernel_size=3, padding=1),
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Third conv layer: kernel_size=4, captures 4-grams
        Conv1d(n_hidden, n_hidden, kernel_size=4, padding=1),
        BatchNorm1d(n_hidden),
        Tanh(),
        
        # Global average pooling to get a single vector per sequence
        # Then linear layer to vocab_size
    ])
    
    return model

# Alternative: Multi-scale convolutional approach
def create_multiscale_conv_model(vocab_size, n_embd=24, n_hidden=128):
    """
    Multi-scale convolutional model that processes different n-gram sizes in parallel
    """
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
            # Concatenate along channel dimension
            self.out = torch.cat([out1, out2, out3, out4], dim=1)
            return self.out
        
        def parameters(self):
            return (self.conv1.parameters() + self.conv2.parameters() + 
                   self.conv3.parameters() + self.conv4.parameters())
    
    class GlobalAvgPool:
        def __call__(self, x):
            # x: (batch, channels, seq_len)
            self.out = x.mean(dim=2)  # Average over sequence length
            return self.out
        
        def parameters(self):
            return []
    
    class Linear:
        def __init__(self, fan_in, fan_out, bias=True):
            self.weight = torch.randn((fan_in, fan_out)) / fan_in**0.5
            self.bias = torch.zeros(fan_out) if bias else None
        
        def __call__(self, x):
            self.out = x @ self.weight
            if self.bias is not None:
                self.out += self.bias
            return self.out
        
        def parameters(self):
            return [self.weight] + ([] if self.bias is None else [self.bias])
    
    model = Sequential([
        Embedding(vocab_size, n_embd),
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

# Example usage and comparison
if __name__ == "__main__":
    # This would be used with the same training loop as in the notebook
    print("Convolutional model architectures defined!")
    print("Key advantages:")
    print("1. Parameter sharing across positions")
    print("2. Translation invariance")
    print("3. Efficient parallel processing")
    print("4. Natural n-gram modeling")
