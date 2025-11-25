import os
import json
import csv
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from train_gpt2 import GPT, GPTConfig
import tiktoken

# --- output directory ---
LOG_DIR = "eval_logs"
os.makedirs(LOG_DIR, exist_ok=True)


# -----------------------------
# --- Load checkpoint/model ---
# -----------------------------
def load_model(checkpoint_path, device="cuda"):
    print(f"\n🔄 Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)     # <--- no weights_only
    config: GPTConfig = ckpt["config"]

    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    model.half()

    print("✅ Model ready.")
    return model



# -----------------------------
# --- Tokenizer (GPT-2) ---
# -----------------------------
enc = tiktoken.get_encoding("gpt2")

def tokens_from(text, device="cuda"):
    ids = enc.encode(text)
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)



# -----------------------------
# --- Evaluate: LAMBADA ---
# -----------------------------
def eval_lambada(model, device="cuda"):
    print("\n📘 Running LAMBADA...")

    correct = 0
    total = 0
    out_file = f"{LOG_DIR}/lambada_log.csv"

    with open(out_file, "w", newline="") as f, \
        open("lambada_test.jsonl") as datafile:

        writer = csv.writer(f)
        writer.writerow(["context", "target", "predicted"])

        for line in tqdm(datafile, total=5153):
            sample = json.loads(line)
            text = sample["text"].strip().split()
            context, target = " ".join(text[:-1]), text[-1]

            toks = tokens_from(context, device)
            logits, _ = model(toks)
            last = logits[0, -1]
            pred_id = last.argmax().item()
            pred_word = enc.decode([pred_id]).strip()

            correct += int(pred_word == target)
            total += 1
            writer.writerow([context, target, pred_word])

    acc = correct / total
    print(f"🎯 LAMBADA accuracy: {acc:.4f}")
    return acc



# -----------------------------
# --- Evaluate: HellaSwag ---
# -----------------------------
from hellaswag import iterate_examples, render_example

def eval_hellaswag(model, device="cuda"):
    print("\n🧠 Running HellaSwag...")

    correct = 0
    total = 0
    out_file = f"{LOG_DIR}/hellaswag_log.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "label", "pred"])

        for example in iterate_examples("val"):
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            logits, _ = model(tokens)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_tokens = tokens[..., 1:].contiguous()

            losses = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_tokens.view(-1),
                reduction="none"
            ).view(tokens.size(0), -1)

            mask = mask[..., 1:]
            masked = losses * mask
            avg_loss = masked.sum(dim=1) / mask.sum(dim=1)
            pred = avg_loss.argmin().item()

            correct += int(pred == label)
            total += 1

            writer.writerow([example["ctx"], label, pred])

    acc = correct / total
    print(f"🎯 HellaSwag accuracy: {acc:.4f}")
    return acc



# -----------------------------
# --- Evaluate: PIQA ---
# -----------------------------
def eval_piqa(model, device="cuda"):
    print("\n🪑 Running PIQA...")

    dataset = load_dataset("piqa", split="validation")
    correct = 0
    total = 0
    out_file = f"{LOG_DIR}/piqa_log.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "label", "pred"])

        for sample in tqdm(dataset):

            a, b = sample["sol1"], sample["sol2"]
            ctx = sample["goal"]

            rows = [ctx + " " + a, ctx + " " + b]
            toks = [tokens_from(r, device) for r in rows]

            losses = []
            for ids in toks:
                logits, _ = model(ids)
                shift = logits[0, :-1]
                target = ids[0, 1:]
                losses.append(F.cross_entropy(shift, target, reduction="mean").item())

            pred = int(losses[1] < losses[0])
            correct += int(pred == sample["label"])
            total += 1

            writer.writerow([ctx, sample["label"], pred])

    acc = correct / total
    print(f"🎯 PIQA accuracy: {acc:.4f}")
    return acc



# -----------------------------
# --- Evaluate: WinoGrande ---
# -----------------------------
def eval_winogrande(model, device="cuda"):
    print("\n👥 Running WinoGrande...")

    dataset = load_dataset("winogrande", "winogrande_xs", split="validation")
    correct = 0
    total = 0
    out_file = f"{LOG_DIR}/winogrande_log.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label", "pred"])

        for sample in tqdm(dataset):

            s = sample["sentence"]
            option1 = s.replace("_", sample["option1"])
            option2 = s.replace("_", sample["option2"])

            toks1 = tokens_from(option1, device)
            toks2 = tokens_from(option2, device)

            logits1, _ = model(toks1)
            logits2, _ = model(toks2)

            loss1 = F.cross_entropy(logits1[0, :-1], toks1[0, 1:], reduction="mean")
            loss2 = F.cross_entropy(logits2[0, :-1], toks2[0, 1:], reduction="mean")
            pred = int(loss2 < loss1)

            correct += int(pred == sample["answer"])
            total += 1

            writer.writerow([s, sample["answer"], pred])

    acc = correct / total
    print(f"🎯 WinoGrande accuracy: {acc:.4f}")
    return acc



# -----------------------------
# --- ARC (Easy + Challenge) ---
# -----------------------------
def eval_arc(model, dataset_name, device="cuda"):
    print(f"\n🧪 Running ARC {dataset_name}...")

    dataset = load_dataset("ai2_arc", dataset_name, split="validation")
    correct = 0
    total = 0
    out_file = f"{LOG_DIR}/arc_{dataset_name}_log.csv"

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "label", "pred"])

        for sample in tqdm(dataset):
            question = sample["question"]
            options = sample["choices"]["text"]
            answer = sample["answerKey"]
            label = sample["choices"]["label"].index(answer)

            # compute loss for each answer option
            losses = []
            for opt in options:
                toks = tokens_from(question + "\n" + opt, device)
                logits, _ = model(toks)
                loss = F.cross_entropy(logits[0, :-1], toks[0, 1:], reduction="mean")
                losses.append(loss.item())

            pred = int(min(range(len(losses)), key=lambda i: losses[i]))
            correct += int(pred == label)
            total += 1

            writer.writerow([question, label, pred])

    acc = correct / total
    print(f"🎯 ARC-{dataset_name} accuracy: {acc:.4f}")
    return acc



# -----------------------------
# --- RUN ALL ---
# -----------------------------
if __name__ == "__main__":

    checkpoint = "checkpoints/model_09535.pt"
    device = "cuda"

    model = load_model(checkpoint, device)

    eval_lambada(model, device)
    eval_hellaswag(model, device)
    #eval_piqa(model, device)
    #eval_winogrande(model, device)
    #eval_arc(model, "ARC-Easy", device)
    #eval_arc(model, "ARC-Challenge", device)

    print("\n\n✅ All evaluations done. Logs saved in:", LOG_DIR)
5
