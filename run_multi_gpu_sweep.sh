#!/bin/bash

# Multi-GPU Hyperparameter Sweep Runner
# Uses all 4 available NVIDIA TITAN V GPUs for parallel training

# Default parameters
NUM_TRIALS=20
NUM_EPOCHS=100
BATCH_SIZE=128
NUM_GPUS=4
SEED=42
OUTPUT_DIR="outputs/sweep_distributed_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_trials N] [--num_epochs N] [--batch_size N] [--num_gpus N] [--seed N] [--output_dir DIR]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Multi-GPU Hyperparameter Sweep"
echo "=========================================="
echo "Trials:     $NUM_TRIALS"
echo "Epochs:     $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "GPUs:       $NUM_GPUS"
echo "Seed:       $SEED"
echo "Output:     $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if graphs exist
if [ ! -d "virtual_graphs/data/all_graphs/raw_graphs" ]; then
    echo "ERROR: Graph data not found!"
    echo "Please run: python graph_motif_generator.py"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA may not be installed."
    exit 1
fi

echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Run the sweep
echo "Starting hyperparameter sweep..."
echo "Estimated time: ~$((NUM_TRIALS * 5 / NUM_GPUS)) minutes"
echo ""

python hyperparameter_sweep_distributed.py \
    --num_trials "$NUM_TRIALS" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_gpus "$NUM_GPUS" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Sweep completed successfully!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  Best params:  cat $OUTPUT_DIR/best_params.json"
    echo "  Study info:   cat $OUTPUT_DIR/study_info.json"
    echo "  All trials:   cat $OUTPUT_DIR/trials.csv"
else
    echo ""
    echo "=========================================="
    echo "✗ Sweep failed!"
    echo "=========================================="
    exit 1
fi
