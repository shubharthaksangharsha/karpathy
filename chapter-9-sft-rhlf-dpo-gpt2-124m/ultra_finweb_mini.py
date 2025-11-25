"""
Ultra-FineWeb Mini (10B tokens) — STREAMING + RESUME + VAL SHARD

Output structure (same as Karpathy / FineWeb-Edu):
edu_ultrafineweb10B/
  edufineweb_val_000000.npy
  edufineweb_train_000001.npy
  ...
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import time

# ============================================================
# CONFIG
# ============================================================

local_dir = "edu_ultrafineweb10B"
TARGET_TOKENS = int(1e10)      # 10B tokens
shard_size = int(1e8)          # 100M tokens per shard (100M × 100 = 10B)
os.makedirs(local_dir, exist_ok=True)

# ============================================================
# RESUME SUPPORT — detect existing shards
# ============================================================

existing_shards = sorted([f for f in os.listdir(local_dir) if f.endswith(".npy")])

if existing_shards:
    last = existing_shards[-1]
    shard_index = int(last.split("_")[-1].split(".")[0])
    print(f"▶️ Resume mode: Found {len(existing_shards)} shards, starting from index {shard_index + 1}")
else:
    shard_index = 0
    print("🆕 Starting fresh...")

# ============================================================
# LOAD DATASET (Streaming, no full 3TB download)
# ============================================================

fw = load_dataset(
    "openbmb/Ultra-FineWeb",
    split="train",
    streaming=True,
)

# ============================================================
# TOKENIZER (GPT-2 tiktoken)
# ============================================================

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    return np.array(tokens, dtype=np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# ============================================================
# MAIN LOOP
# ============================================================

start_time = time.time()
total_tokens_written = shard_index * shard_size
nprocs = max(1, os.cpu_count() // 2)

with mp.Pool(nprocs) as pool:
    token_count = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):

        if total_tokens_written >= TARGET_TOKENS:
            break

        # store token chunk
        remaining_space = shard_size - token_count

        if len(tokens) <= remaining_space:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            total_tokens_written += len(tokens)

            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens",
                                    desc=f"Shard {shard_index:06d} ({'val' if shard_index==0 else 'train'})")

            progress_bar.update(len(tokens))

        else:
            # fill remaining part
            all_tokens_np[token_count:] = tokens[:remaining_space]
            total_tokens_written += remaining_space

            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np)

            # new shard
            shard_index += 1
            progress_bar = None

            leftover = len(tokens) - remaining_space
            all_tokens_np[0:leftover] = tokens[remaining_space:]
            token_count = leftover

        # logging speed
        elapsed = time.time() - start_time
        toks_per_sec = total_tokens_written / elapsed
        print(f"\r⏱️  {total_tokens_written:,} tokens ({toks_per_sec:,.0f} tok/s)", end="")

    # write last shard
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print("\n✅ Ultra-FineWeb-Mini ready!")
print(f"📊 Total tokens written: {total_tokens_written:,}")
print(f"📁 Output directory: {local_dir}")

