import numpy as np
import time
import sys
import platform
import subprocess
from grainvdb import GrainVDB
import pprint

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def run_technical_audit():
    # Fixed Seed
    np.random.seed(42)
    
    N = 1_000_000
    DIM = 128
    K = 10
    TRIALS = 10
    
    print(f"--- GrainVDB Benchmark ---")
    print(f"Commit: {get_git_revision_short_hash()}")
    print(f"OS/Arch: {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}")
    
    # Correctly display BLAS info for newer NumPy
    try:
        blas_info = np.show_config(mode="dicts")
        # Just grab the first Build Dependency as a proxy for backend
        print("BLAS Backend Check:")
        pprint.pprint(blas_info.get("Build Dependencies", {}).get("blas", "Unknown"))
    except:
        print("BLAS Config: (Could not parse)")
        
    print(f"Params: N={N:,} | D={DIM} | K={K} | Trials={TRIALS}")
    
    print("\n[1] Data Generation (Random Normal, Fixed Seed)")
    db_raw = np.random.randn(N, DIM).astype(np.float32)
    q_raw = np.random.randn(DIM).astype(np.float32)
    
    # Pre-normalize for isolation
    db_norm = db_raw / (np.linalg.norm(db_raw, axis=1, keepdims=True) + 1e-9)
    q_norm = q_raw / (np.linalg.norm(q_raw) + 1e-9)

    print("\n[2] CPU Baseline")
    print("    Method: np.dot + np.argpartition")
    
    cpu_times = []
    _ = np.argpartition(np.dot(db_norm, q_norm), -K)[-K:] # Warmup

    for _ in range(TRIALS):
        t0 = time.perf_counter()
        # Native DOT + Argpartition
        sims = np.dot(db_norm, q_norm)
        top_k_idx = np.argpartition(sims, -K)[-K:]
        # Sorting for stability in comparison
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
        cpu_times.append((time.perf_counter() - t0) * 1000)
    
    print(f"    Latency: {np.median(cpu_times):.2f} ms (p50) | {np.percentile(cpu_times, 95):.2f} ms (p95)")

    print("\n[3] Native Core (GrainVDB)")
    try:
        vdb = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"    FATAL: {e}")
        return

    vdb.add_vectors(db_raw)
    
    vdb.query(q_raw, k=K) # Warmup

    vdb_times = []
    for _ in range(TRIALS):
        # Latency includes Python overhead + GPU execution + CPU selection
        _, _, lat = vdb.query(q_raw, k=K)
        vdb_times.append(lat)
        
    print(f"    Latency: {np.median(vdb_times):.2f} ms (p50) | {np.percentile(vdb_times, 95):.2f} ms (p95)")

    print("\n[4] Correctness (Recall@K)")
    indices, scores, _ = vdb.query(q_raw, k=K)
    
    cpu_set = set(top_k_idx)
    gpu_set = set(indices)
    overlap = len(cpu_set.intersection(gpu_set))
    recall = overlap / K
    
    print(f"    Recall: {recall:.2f} ({overlap}/{K})")
    print(f"    CPU Top-1: {top_k_idx[0]} | GPU Top-1: {indices[0]}")
    
    if recall < 0.9:
        print("    WARNING: Low recall. Check precision/normalization.")
    else:
        print("    PASS: High overlap with CPU baseline.")

    speedup = np.median(cpu_times) / np.median(vdb_times)
    print(f"\n✨ Speedup: {speedup:.1f}x (p50)")

if __name__ == "__main__":
    run_technical_audit()
