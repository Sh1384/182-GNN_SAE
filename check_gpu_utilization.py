"""
GPU Utilization Diagnostic Script

Checks current GPU status and provides recommendations.
"""

import subprocess
import torch

def parse_nvidia_smi():
    """Parse nvidia-smi output to get GPU info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 6:
                gpus.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4]),
                    'power': float(parts[5])
                })
        return gpus
    except Exception as e:
        print(f"Error parsing nvidia-smi: {e}")
        return []


def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    print("=" * 80)
    print("PYTORCH CUDA STATUS")
    print("=" * 80)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
    print()


def check_gpu_status():
    """Check and display GPU status."""
    print("=" * 80)
    print("CURRENT GPU STATUS (from nvidia-smi)")
    print("=" * 80)

    gpus = parse_nvidia_smi()

    if not gpus:
        print("Could not read GPU information")
        return

    # Print table header
    print(f"{'GPU':<5} {'Name':<20} {'Memory':<15} {'Utilization':<15} {'Power':<10}")
    print("-" * 80)

    idle_gpus = []
    busy_gpus = []

    for gpu in gpus:
        memory_str = f"{gpu['memory_used']}/{gpu['memory_total']} MB"
        util_str = f"{gpu['utilization']}%"
        power_str = f"{gpu['power']:.1f} W"

        print(f"{gpu['id']:<5} {gpu['name']:<20} {memory_str:<15} {util_str:<15} {power_str:<10}")

        if gpu['utilization'] < 10:
            idle_gpus.append(gpu['id'])
        else:
            busy_gpus.append(gpu['id'])

    print()

    # Analysis
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if len(idle_gpus) == len(gpus):
        print("⚠ ALL GPUs are IDLE (< 10% utilization)")
        print()
        print("Possible issues:")
        print("  1. Training not started yet")
        print("  2. Data loading is the bottleneck (CPU-bound)")
        print("  3. Batch size is too small")
        print("  4. Model is too small for the GPU")
        print()
        print("Recommendations:")
        print("  - Use the multi-GPU sweep scripts to utilize all GPUs")
        print("  - Increase batch size (try 256 or 512)")
        print("  - Increase num_workers in DataLoader")
        print("  - Use larger models (increase hidden_dim)")

    elif busy_gpus:
        print(f"✓ {len(busy_gpus)} GPU(s) actively training: {busy_gpus}")
        if idle_gpus:
            print(f"⚠ {len(idle_gpus)} GPU(s) idle and available: {idle_gpus}")
            print()
            print("Recommendation:")
            print(f"  - Run multi-GPU sweep to use all {len(gpus)} GPUs")
            print(f"  - Potential speedup: {len(gpus)}x faster")

    print()

    # Check for processes
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
             '--format=csv,noheader'],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            print("=" * 80)
            print("ACTIVE PROCESSES ON GPU")
            print("=" * 80)
            print(f"{'PID':<10} {'Process':<40} {'Memory':<10}")
            print("-" * 80)
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    print(f"{parts[0]:<10} {parts[1]:<40} {parts[2]:<10}")
            print()
    except:
        pass


def main():
    print()
    check_pytorch_cuda()
    check_gpu_status()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("To use all GPUs for hyperparameter sweep:")
    print()
    print("  1. Stop current training (if running in notebook)")
    print()
    print("  2. Run multi-GPU sweep:")
    print("     ./run_multi_gpu_sweep.sh")
    print()
    print("  3. Monitor GPU usage in another terminal:")
    print("     watch -n 1 nvidia-smi")
    print()
    print("Expected result:")
    print("  - All 4 GPUs showing 30-90% utilization")
    print("  - Memory usage: 400-800 MB per GPU")
    print("  - Power: 100-200W per GPU")
    print("  - 4x faster completion time")
    print()


if __name__ == "__main__":
    main()
