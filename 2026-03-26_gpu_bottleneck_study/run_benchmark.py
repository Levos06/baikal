import subprocess
import os
import time

TESTS = [
    ("Pure I/O", "test_1_pure_io.py"),
    ("Preprocessing (Features)", "test_2_preprocessing.py"),
    ("Worker Scaling", "test_3_worker_scaling.py"),
    ("Synthetic GPU Max", "test_4_synthetic_gpu.py"),
    ("Batch Size Scaling", "test_5_batch_scaling.py")
]

LOG_DIR = "2026-03-26_gpu_bottleneck_study/results"
os.makedirs(LOG_DIR, exist_ok=True)

def run_benchmark():
    print("="*50)
    print("GPU BOTTLENECK INVESTIGATION ORCHESTRATOR")
    print("Target: CUDA_VISIBLE_DEVICES=0 (RTX 4090)")
    print("="*50)
    
    final_report = []
    
    for name, script in TESTS:
        print(f"\n>>> Running {name}...")
        log_file = os.path.join(LOG_DIR, script.replace(".py", ".log"))
        
        # Run with GPU 0
        cmd = f"export CUDA_VISIBLE_DEVICES=0 && python3 2026-03-26_gpu_bottleneck_study/{script}"
        
        start = time.time()
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        duration = time.time() - start
        
        with open(log_file, "w") as f:
            f.write(process.stdout)
            if process.stderr:
                f.write("\nERRORS:\n")
                f.write(process.stderr)
        
        print(f"    Finished in {duration:.1f}s. Results in {log_file}")
        
        # Extract last lines for quick summary
        summary = [line for line in process.stdout.split('\n') if line.strip()][-5:]
        final_report.append(f"\n--- {name} ---\n" + "\n".join(summary))

    report_path = os.path.join(LOG_DIR, "FINAL_REPORT.txt")
    with open(report_path, "w") as f:
        f.write("=== GPU BOTTLENECK STUDY FINAL REPORT ===\n")
        f.write("\n".join(final_report))
    
    print("\n" + "="*50)
    print(f"BENCHMARK COMPLETE. Summary saved to {report_path}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
