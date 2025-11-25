#!/bin/bash

# Train GPT-2 124M with Large Combined Dataset (41k samples)

echo "============================================================"
echo "🚀 Starting Q&A SFT Training - LARGE DATASET (41k samples)"
echo "============================================================"
echo ""

# Check if combined dataset exists
if [ ! -f "combined_qa_dataset.jsonl" ]; then
    echo "❌ ERROR: combined_qa_dataset.jsonl not found!"
    echo ""
    echo "Please run first:"
    echo "  bash setup_large_dataset.sh"
    echo ""
    exit 1
fi

# Delete old token files (forces recreation with new dataset)
if [ -f "qa_train_tokens.pt" ] || [ -f "qa_val_tokens.pt" ]; then
    echo "🗑️  Removing old token files..."
    rm -f qa_train_tokens.pt qa_val_tokens.pt
    echo "✅ Old token files removed (will be recreated)"
    echo ""
fi

# Count samples in dataset
SAMPLE_COUNT=$(wc -l < combined_qa_dataset.jsonl)
echo "📊 Dataset: combined_qa_dataset.jsonl"
echo "📊 Samples: $SAMPLE_COUNT Q&A pairs"
echo ""

# Check GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "📊 Detected GPUs: $NUM_GPUS"
echo ""

# Ask for confirmation
echo "⚙️  Training config:"
echo "   - Batch size: 16 x 8 = 128"
echo "   - Max steps: 2,500"
echo "   - Learning rate: 1e-5"
echo "   - Expected time: 6-8 hours (single GPU)"
echo ""
read -p "Start training? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "============================================================"
echo "🔥 STARTING TRAINING"
echo "============================================================"
echo ""

# Run training
if [ $NUM_GPUS -gt 1 ]; then
    echo "🔥 Running with DDP on $NUM_GPUS GPUs"
    torchrun --standalone --nproc_per_node=$NUM_GPUS sft-qa-large.py
else
    echo "🔥 Running on single GPU"
    python sft-qa-large.py
fi

echo ""
echo "============================================================"
echo "✅ Training complete!"
echo "============================================================"
echo ""
echo "📊 Check results:"
echo "   - Best model: qa_sft_checkpoints/qa-sft_best.pt"
echo "   - W&B dashboard: https://wandb.ai"
echo ""





