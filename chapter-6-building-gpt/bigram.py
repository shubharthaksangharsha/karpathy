import torch 
import torch.nn as nn 
from torch.nn import functional as F 

#hyperparameters 
batch_size = 32 #how many independent sequences will be process in parallel?
block_size = 8  #what is the maximum context length for predictions?
max_iters = 3000 
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
eval_iters = 200 
#---------------

torch.manual_seed(1337)

# Download the text file (assuming you're running this in an environment like a Jupyter notebook or Colab)
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Read the entire text file into a single string
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# ----------------------------------------------------------------------

# Get all the unique characters that occur in the text
chars = sorted(list(set(text)))

# Determine the size of the vocabulary (number of unique characters)
vocab_size = len(chars)

# Create a mapping from characters to integers (stoi: string-to-integer)
# Example: {'\n': 0, ' ': 1, '!': 2, ...}
stoi = {ch: i for i, ch in enumerate(chars)}

# Create a mapping from integers back to characters (itos: integer-to-string)
# This is the inverse of stoi. Example: {0: '\n', 1: ' ', 2: '!', ...}
itos = {i: ch for i, ch in enumerate(chars)}

# Define an encoder function: takes a string and outputs a list of integers
# by looking up the integer representation for each character.
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers

# Define a decoder function: takes a list of integers and outputs a string
# by joining the corresponding characters together.
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# ----------------------------------------------------------------------

# Convert the entire text dataset into a PyTorch tensor of integers
import torch # This line is implied or needs to be added if not present

data = torch.tensor(encode(text), dtype=torch.long)

# Determine the split point (90% for training, 10% for validation)
n = int(0.9*len(data)) # 90% first will be train, rest val

# Create the training dataset (first 90% of the integer-encoded data)
train_data = data[:n]

# Create the validation dataset (the remaining 10%)
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y

    # Selects the appropriate dataset (train_data or val_data) based on the 'split' argument
    data = train_data if split == 'train' else val_data

    # Generates a set of random starting indices (ix) for the chunks of text.
    # block_size is the length of the sequence (e.g., 8 characters), batch_size is how many sequences to get.
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stacks the input sequences (x).
    # For each index 'i' in 'ix', it takes a slice of 'data' of length 'block_size'.
    x = torch.stack([data[i:i+block_size] for i in ix])

    # Stacks the target sequences (y).
    # The target for a given input sequence is the same sequence shifted one position to the right.
    # This means the model is trained to predict the character at position i+1 based on the character at position i.
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # Moves the input and target tensors to the specified device (e.g., 'cuda' for GPU, or 'cpu').
    x, y = x.to(device), y.to(device)

    return x, y

# ----------------------------------------------------------------------

# Decorator that tells PyTorch not to calculate gradients for this function.
# This saves memory and computation time as we are only evaluating, not training.
@torch.no_grad()
def estimate_loss():
    out = {}
    
    # Sets the model to evaluation mode. This disables features like Dropout and BatchNorm updates.
    model.eval()

    # Loop over both the training and validation splits
    for split in ['train', 'val']:
        
        # Initializes a tensor to store the losses for a specified number of evaluation iterations.
        losses = torch.zeros(eval_iters)
        
        # Runs the evaluation loop
        for k in range(eval_iters):
            # Gets a batch of data for the current split
            X, Y = get_batch(split)
            
            # Performs a forward pass through the model
            logits, loss = model(X, Y)
            
            # Stores the loss value
            losses[k] = loss.item()
            
        # Calculates the average loss across all evaluation iterations for the current split
        out[split] = losses.mean().item()
    
    # Sets the model back to training mode (enabling features like Dropout/BatchNorm)
    model.train()
    
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off the logits for the next token from the lookup table 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx) #(B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current target 
        for _ in range(max_new_tokens):
            #get the predictions 
            logits, loss = self(idx) 
            #focus only on the last time step 
            logits = logits[:, -1, :] #becomes (B, C) 
            #apply softmax to get probabilities 
            probs = F.softmax(logits, dim=1) # (B, C)
            #sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) 
            #append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) 
        return idx



model = BigramLanguageModel(vocab_size)
m = model.to(device)

#create a PyTorch Optimizer 
optmizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):


    #every once in while evaluate the loss on trian and val sets 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    
    #sample a batch of data 
    xb, yb = get_batch('train')

    #evaluate the loss 
    logits, loss = model(xb, yb)
    optmizer.zero_grad(set_to_none=True)
    loss.backward()
    optmizer.step()

#generate from the model 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

                             