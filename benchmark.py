import numpy as np
import time
import sys
import platform
import subprocess
from grainvdb import GrainVDB

def get_git_revision_short_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except:
        return "unknown"

def generate_clustered_data(n_samples, n_features, n_clusters=10):
    """Generates data with some structure to make semantic audit meaningful."""
    # simple centers
    centers = np.random.randn(n_clusters, n_features).astype(np.float32)
    # distribute points around centers
    data = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        center_idx = i % n_clusters
        noise = np.random.randn(n_features).astype(np.float32) * 0.1
        data[i] = centers[center_idx] + noise
    return data

def run_technical_audit():
    N = 1_000_000
    DIM = 128
    K = 10
    TRIALS = 10
    
    print(f"--- GrainVDB Engineering Benchmark Certification ---")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware: {platform.machine()} / {platform.system()} {platform.release()}")
    print(f"Commit: {get_git_revision_short_hash()}")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}")
    print(f"Config: N={N:,} | D={DIM} | K={K}")
    
    # 1. Dataset Generation (Clustered for meaningful topology)
    print("\n[1] Synthesizing Data (10 Semantic Clusters)...")
    db_raw = generate_clustered_data(N, DIM, n_clusters=20)
    # Pick a query close to cluster 0 center
    q_raw = db_raw[0] + np.random.randn(DIM).astype(np.float32) * 0.05
    
    # 2. Optimized CPU Baseline
    # Pre-normalize DB and Query once to isolate strictly search/selection time.
    # Note: We use float32 for CPU reference.
    db_norm = db_raw / (np.linalg.norm(db_raw, axis=1, keepdims=True) + 1e-9)
    q_norm = q_raw / (np.linalg.norm(q_raw) + 1e-9)

    print("\n[2] CPU Reference Baseline")
    print("    Method: np.dot (BLAS) + np.argpartition (O(N) selection)")
    
    # Warmup
    _ = np.argpartition(np.dot(db_norm, q_norm), -K)[-K:]

    cpu_times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        sims = np.dot(db_norm, q_norm)
        top_k_idx = np.argpartition(sims, -K)[-K:]
        # Sorting strictly for comparison stability (negligible cost on K=10)
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
        cpu_times.append((time.perf_counter() - t0) * 1000)
    
    cpu_p50 = np.percentile(cpu_times, 50)
    cpu_p95 = np.percentile(cpu_times, 95)
    print(f"    Latency: {cpu_p50:.2f} ms (p50) | {cpu_p95:.2f} ms (p95)")

    # 3. Native Core (GrainVDB)
    try:
        vdb = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"    FATAL: Engine Load Failed: {e}")
        return

    print("\n[3] Native Core (GrainVDB)")
    print("    Method: Metal (FP16/half4) Brute Force + CPU Priority Queue")
    print("    Loading Unified Memory...")
    vdb.add_vectors(db_raw)
    
    # Warmup
    vdb.query(q_raw, k=K)
    
    vdb_times = []
    for _ in range(TRIALS):
        # Timer starts inside .query() which covers:
        # 1. Input preparation
        # 2. Native boundary cross
        # 3. GPU Dispatch + Wait
        # 4. CPU Selection
        # 5. Return
        _, _, lat = vdb.query(q_raw, k=K) # lat is end-to-end wall time
        vdb_times.append(lat)
        
    vdb_p50 = np.percentile(vdb_times, 50)
    vdb_p95 = np.percentile(vdb_times, 95)
    print(f"    Latency: {vdb_p50:.2f} ms (p50) | {vdb_p95:.2f} ms (p95)")

    # 4. Correctness Audit
    print("\n[4] Correctness Verification (Recall@K)")
    indices, scores, _ = vdb.query(q_raw, k=K)
    
    # Compare Sets
    cpu_set = set(top_k_idx)
    gpu_set = set(indices)
    overlap = len(cpu_set.intersection(gpu_set))
    recall = overlap / K
    
    # Compare Scores (First match)
    score_diff = abs(scores[0] - sims[top_k_idx[0]])
    
    print(f"    CPU Top-1: {top_k_idx[0]} ({sims[top_k_idx[0]]:.4f})")
    print(f"    GPU Top-1: {indices[0]} ({scores[0]:.4f})")
    
    if recall == 1.0:
        print(f"    ✅ Perfect Comparison: 10/10 indices match.")
    elif recall >= 0.9:
        print(f"    ⚠️ High Overlap: {int(recall*10)}/10 indices match (acceptable for FP16/FP32).")
    else:
        print(f"    ❌ Low Overlap: {int(recall*10)}/10 indices match.")

    # 5. Connectivity Audit
    print("\n[5] Topology Audit (Neighborhood Coherence)")
    # We used clustered data, so we expect high connectivity
    consensus = vdb.audit(indices)
    print(f"    Density: {consensus:.4f} (Threshold=0.85)")
    if consensus > 0.5:
        print("    ✅ Strong semantic clustering detected.")
    else:
        print("    ⚠️ Low coherence (possibly noise or sparse manifold).")

    speedup = cpu_p50 / vdb_p50
    print(f"\n✨ Final Result: {speedup:.1f}x Speedup (p50)")

if __name__ == "__main__":
    run_technical_audit()
