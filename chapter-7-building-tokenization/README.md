# Chapter 7: Building Tokenization

## Overview
This chapter focuses on understanding and implementing tokenization techniques for language models, particularly Byte Pair Encoding (BPE). Tokenization is a crucial preprocessing step that significantly impacts model performance.

## 📁 Files
- `building-tokenization.ipynb` - Main implementation notebook
- `encoder.json` - BPE encoder mappings
- `tok400.model` - Trained BPE model
- `tok400.vocab` - BPE vocabulary
- `vocab.bpe` - BPE merge rules
- `toy.txt` - Sample text for tokenization
- `resources/` - Additional resources and assets

## 🎯 Learning Objectives
- Understand the importance of tokenization in language modeling
- Learn about different tokenization approaches (character, word, subword)
- Implement Byte Pair Encoding (BPE) from scratch
- Understand the trade-offs between different tokenization methods
- Learn how tokenization affects model performance and vocabulary size

## 🔧 Key Concepts

### Tokenization Approaches
- **Character-level**: Each character is a token (simple but inefficient)
- **Word-level**: Each word is a token (efficient but limited vocabulary)
- **Subword-level**: Balance between character and word-level approaches

### Byte Pair Encoding (BPE)
- **Iterative Merging**: Start with characters, merge frequent pairs
- **Vocabulary Growth**: Gradually build a vocabulary of subword units
- **Handling OOV**: Can represent any text using learned subwords

### BPE Algorithm
1. **Initialize**: Start with character vocabulary
2. **Count Pairs**: Count all adjacent symbol pairs
3. **Merge**: Replace most frequent pair with new symbol
4. **Repeat**: Continue until desired vocabulary size

## 🚀 Implementation Highlights

### BPE Training
```python
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
```

### BPE Encoding
```python
def encode(self, text):
    tokens = list(text)
    while True:
        pairs = get_pairs(tokens)
        if not pairs:
            break
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
        if bigram not in self.bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(tokens):
            try:
                j = tokens.index(first, i)
                new_word.extend(tokens[i:j])
                i = j
            except ValueError:
                new_word.extend(tokens[i:])
                break
            
            if tokens[i] == first and i < len(tokens)-1 and tokens[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(tokens[i])
                i += 1
        tokens = new_word
    return tokens
```

### Vocabulary Analysis
```python
def analyze_vocab(vocab):
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Average token length: {sum(len(token) for token in vocab) / len(vocab):.2f}")
    print(f"Most frequent tokens:")
    for token, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {token}: {freq}")
```

## 📊 Tokenization Comparison

### Character-level Tokenization
- **Vocabulary Size**: 27 (26 letters + special token)
- **Sequence Length**: Very long (one token per character)
- **Efficiency**: Low (many tokens needed)

### Word-level Tokenization
- **Vocabulary Size**: ~50,000 (typical English vocabulary)
- **Sequence Length**: Short (one token per word)
- **Efficiency**: High for common words, fails for rare words

### BPE Tokenization
- **Vocabulary Size**: 400 (configurable)
- **Sequence Length**: Moderate (balance between character and word)
- **Efficiency**: High for common patterns, handles rare words

## 🎓 Key Takeaways

1. **Tokenization Impact**: Choice of tokenization significantly affects model performance
2. **Vocabulary Size**: Balance between efficiency and coverage
3. **Subword Units**: BPE provides a good compromise between character and word-level
4. **Handling OOV**: Subword tokenization can represent any text
5. **Preprocessing**: Tokenization is a crucial preprocessing step

## 🔧 Technical Details

### BPE Training Process
1. **Initialize**: Start with character vocabulary
2. **Count Pairs**: Count all adjacent symbol pairs in training data
3. **Merge**: Replace most frequent pair with new symbol
4. **Update**: Update vocabulary with new symbol
5. **Repeat**: Continue until desired vocabulary size

### BPE Encoding Process
1. **Split**: Split text into characters
2. **Find Pairs**: Find all adjacent symbol pairs
3. **Apply Rules**: Apply BPE merge rules in order
4. **Output**: Return final token sequence

## 📚 Prerequisites
- Chapter 6 (GPT Implementation)
- Understanding of text preprocessing
- Basic knowledge of algorithms
- Understanding of vocabulary and tokenization

## 🛠️ Dependencies
- Python 3.6+
- Collections module
- Regular expressions
- Basic text processing libraries

## 🎯 Next Steps
This chapter prepares you for:
- Working with pre-trained language models
- Understanding modern tokenization schemes
- Implementing custom tokenizers
- Working with different languages and scripts

## 💡 Key Insights
- **Tokenization Matters**: The choice of tokenization significantly impacts model performance
- **BPE Benefits**: Byte Pair Encoding provides a good balance between efficiency and coverage
- **Vocabulary Design**: Careful design of vocabulary is crucial for model success
- **Preprocessing Pipeline**: Tokenization is a critical part of the preprocessing pipeline

## 🔗 Connection to Modern Models
- **GPT Models**: Use BPE tokenization
- **BERT Models**: Use WordPiece tokenization
- **T5 Models**: Use SentencePiece tokenization
- **Modern LLMs**: All use sophisticated subword tokenization
