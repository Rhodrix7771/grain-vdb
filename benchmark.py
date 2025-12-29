import numpy as np
import time
from grainvdb import GrainVDB

def run_technical_audit():
    N = 1000000 # 1,000,000 Vectors
    DIM = 128
    K = 10
    
    print(f"--- GrainVDB Engineering Benchmark ---")
    print(f"Manifold Size: {N:,} vectors | Complexity: {DIM} dimensions")
    
    # 1. Synthesize Data
    print("Generating random float32 vectors...")
    db_raw = np.random.randn(N, DIM).astype(np.float32)
    query_raw = np.random.randn(DIM).astype(np.float32)
    
    # 2. Establish CPU Baseline (NumPy Optimized)
    # We pre-normalize to ensure the benchmark measures strictly the search/resolution performance.
    db_norm = db_raw / (np.linalg.norm(db_raw, axis=1, keepdims=True) + 1e-9)
    query_norm = query_raw / (np.linalg.norm(query_raw) + 1e-9)

    print("\n[CPU Baseline] Using np.dot + np.argpartition...")
    t0 = time.perf_counter()
    sims = np.dot(db_norm, query_norm)
    top_k_indices = np.argpartition(sims, -K)[-K:]
    top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"End-to-End CPU Wall-Time: {cpu_ms:.2f} ms")

    # 3. Establish Native Baseline (GrainVDB)
    try:
        vdb = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"Native Init Failure: {e}")
        return

    print("\n[Native Core] Loading vectors into Unified Memory Buffer...")
    vdb.add_vectors(db_raw)
    
    # Warmup
    vdb.query(query_raw, k=K)
    
    print("Executing query measurement loops (n=10)...")
    wall_latencies = []
    
    for _ in range(10):
        t1 = time.perf_counter()
        indices, scores, _ = vdb.query(query_raw, k=K)
        t2 = time.perf_counter()
        wall_latencies.append((t2 - t1) * 1000)

    avg_wall_ms = np.mean(wall_latencies)
    print(f"End-to-End Bridge Wall-Time: {avg_wall_ms:.2f} ms")
    print(f"System Throughput: {1000 / avg_wall_ms:.1f} queries/sec")

    # 4. Correctness Verification
    print("\n[Verification] Comparing results...")
    if indices[0] == top_k_indices[0]:
        print("✅ SUCCESS: Native index matches CPU reference.")
        print(f"   Top-1 Score: {scores[0]:.4f}")
    else:
        print(f"❌ MISMATCH: Check normalization or float precision.")
        print(f"   Native Score: {scores[0]:.4f}")
        print(f"   CPU Score:    {sims[top_k_indices[0]]:.4f}")

    # 5. Audit Metric
    connectivity = vdb.audit(indices)
    print(f"\n[Audit] Neighborhood Connectivity: {connectivity:.4f}")
    
    print(f"\n✨ Effective Speedup: {cpu_ms / avg_wall_ms:.1f}x vs optimized CPU.")

if __name__ == "__main__":
    run_technical_audit()
