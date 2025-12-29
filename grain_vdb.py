import torch
import numpy as np
import time
import os
from typing import List, Tuple, Optional

# GrainVDB: Metal-Accelerated Vector Intelligence Engine
# =====================================================

class GrainVDB:
    def __init__(self, dim: int = 128, device: str = None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        self.dim = dim
        self.vectors = torch.empty((0, dim), device=self.device)
        self._norm_vectors = None
        
        print(f"GrainVDB: Initialized on {self.device.upper()}")

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the device buffer."""
        tensor = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
        self.vectors = torch.cat([self.vectors, tensor], dim=0)
        self._norm_vectors = None # Reset cache

    def _get_norm_vectors(self):
        if self._norm_vectors is None:
            self._norm_vectors = self.vectors / (self.vectors.norm(dim=1, keepdim=True) + 1e-9)
        return self._norm_vectors

    def query(self, query_vec: np.ndarray, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Query using Metal-accelerated cosine similarity."""
        query_tensor = torch.from_numpy(query_vec.astype(np.float32)).to(self.device)
        query_norm = query_tensor / (query_tensor.norm() + 1e-9)
        db_norm = self._get_norm_vectors()
        
        start_time = time.perf_counter()
        similarities = torch.matmul(db_norm, query_norm)
        scores, indices = torch.topk(similarities, k)
        
        if self.device == "mps":
            torch.mps.synchronize()
            
        elapsed = time.perf_counter() - start_time
        return scores, indices, elapsed

    def audit(self, indices: torch.Tensor, threshold: float = 0.8) -> float:
        """
        Hallucination Mitigation: Context Connectivity Audit.
        Calculates the algebraic connectivity (Fiedler value) of the retrieved neighborhood.
        Low connectivity indicates a 'Context Fracture' (semantic ambiguity).
        """
        if len(indices) < 2:
            return 1.0

        # Get the retrieved vectors
        neighbors = self.vectors[indices]
        neighbors_norm = neighbors / (neighbors.norm(dim=1, keepdim=True) + 1e-9)
        
        # Compute local similarity matrix
        sim_matrix = torch.matmul(neighbors_norm, neighbors_norm.T)
        
        # Adjacency matrix (thresholded similarity)
        adj = (sim_matrix > threshold).float()
        
        # Degree matrix
        degree = torch.diag(adj.sum(dim=1))
        
        # Laplacian matrix
        laplacian = degree - adj
        
        # Compute eigenvalues (CPU-side for stability in this version)
        eigvals = torch.linalg.eigvalsh(laplacian.cpu())
        
        # The second smallest eigenvalue (Fiedler value) measures connectivity
        fiedler = eigvals[1].item() if len(eigvals) > 1 else 0.0
        return fiedler

def run_benchmark():
    N = 1000000 # 1 Million Vectors
    DIM = 128
    K = 10
    
    print(f"Generating {N:,} random vectors (DIM={DIM})...")
    db_vectors = np.random.randn(N, DIM).astype(np.float32)
    query_vec = np.random.randn(DIM).astype(np.float32)
    
    # Pre-normalize for fair comparison (measure only search/top-k logic)
    db_cpu = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
    q_cpu = query_vec / np.linalg.norm(query_vec)

    # 1. CPU Baseline (NumPy / Optimized Partition)
    print("\nRunning CPU Benchmark (NumPy Partition)...")
    start = time.perf_counter()
    sims = np.dot(db_cpu, q_cpu)
    # Use argpartition for O(N) top-k, matching torch.topk logic
    top_k_idx = np.argpartition(sims, -K)[-K:]
    # Sort just the top-k results
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    cpu_time = time.perf_counter() - start
    print(f"CPU Time: {cpu_time*1000:.2f} ms")

    # 2. GrainVDB (Metal/MPS)
    vdb = GrainVDB(dim=DIM)
    print("\nLoading vectors into GrainVDB (MPS)...")
    vdb.add_vectors(db_vectors)
    
    # Warmup
    vdb.query(query_vec, k=K)
    
    # Benchmark
    loops = 10
    total_time = 0
    for _ in range(loops):
        _, _, elapsed = vdb.query(query_vec, k=K)
        total_time += elapsed
    
    avg_metal_time = total_time / loops
    print(f"GrainVDB (Metal) Time: {avg_metal_time*1000:.2f} ms")
    print(f"\nSPEEDUP: {cpu_time / avg_metal_time:.1f}x")

    # 3. Audit Test
    scores, indices, _ = vdb.query(query_vec, k=K)
    connectivity = vdb.audit(indices)
    print(f"\nNeighborhood Connectivity (Fiedler): {connectivity:.4f}")

if __name__ == "__main__":
    run_benchmark()
