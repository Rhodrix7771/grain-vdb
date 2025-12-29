import numpy as np
import time
from grainvdb.engine import GrainVDB

def run_benchmark():
    N = 1000000 # 1 Million Vectors
    DIM = 128
    K = 10
    
    print(f"🚀 GrainVDB Industrial Benchmark")
    print(f"Generating {N:,} random vectors (float32)...")
    db_vectors = np.random.randn(N, DIM).astype(np.float32)
    query_vec = np.random.randn(DIM).astype(np.float32)
    
    # Pre-normalize database and query for fair comparison
    # In production, you'd store normalized vectors
    db_cpu = db_vectors / (np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9)
    q_cpu = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    # 1. CPU Baseline (Optimized NumPy)
    print("\n--- CPU Baseline (NumPy Partition) ---")
    start = time.perf_counter()
    # brute force cosine similarity
    sims = np.dot(db_cpu, q_cpu)
    # Use partition for O(N) top-k selection (not a full sort)
    top_k_idx = np.argpartition(sims, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]] # Sort only the top 10
    cpu_time = (time.perf_counter() - start) * 1000
    print(f"Latency: {cpu_time:.2f} ms")

    # 2. GrainVDB (Native Metal)
    try:
        engine = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"Skipping Metal Benchmark: {e}")
        return

    print("\n--- GrainVDB Native Core (Metal) ---")
    print("Uploading 1M vectors to Unified Memory...")
    engine.add_vectors(db_vectors)
    
    # Warmup the GPU
    engine.query(query_vec, k=K)
    
    # Benchmark
    loops = 10
    latencies = []
    for _ in range(loops):
        _, _, ms = engine.query(query_vec, k=K)
        latencies.append(ms)
    
    avg_metal_time = np.mean(latencies)
    print(f"Latency: {avg_metal_time:.2f} ms (internal GPU timing)")
    print(f"Throughput: {1000 / avg_metal_time:.1f} req/s")
    
    print(f"\n✨ Performance Gain: {cpu_time / avg_metal_time:.1f}x vs optimized CPU partition")

    # Results
    indices, scores, _ = engine.query(query_vec, k=K)
    print(f"Top 1 index: {indices[0]}, Score: {scores[0]:.4f}")
    
    # Audit
    connectivity = engine.audit_consistency(indices)
    print(f"Neighborhood Consistency (Gluing Energy): {connectivity:.4f}")

if __name__ == "__main__":
    run_benchmark()
