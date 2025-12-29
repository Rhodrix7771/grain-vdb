# GrainVDB 🌾
### Metal-Core Vector Intelligence for Apple Silicon
**Hardware-Accelerated Similarity Search & Neighborhood Connectivity Audits**

GrainVDB is a high-performance local engine for vector search, optimized for the **Unified Memory Architecture** of Apple Silicon. It utilizes the Metal Performance Shaders (MPS) via PyTorch for GPU-accelerated similarity resolution and implements a graph-theoretic audit layer to mitigate RAG hallucinations.

---

## 📊 Benchmarks (1 Million x 128D Vectors)
| Metric | CPU (NumPy Optimized) | **GrainVDB (Metal/MPS)** |
|--------|----------------------|-----------------------|
| Query Latency (k=10) | ~240 ms | **~28 ms** |
| Throughput | 4.1 req/s | **35.7 req/s** |

**Hardware**: MacBook M2 (Unified Memory).
**Methodology**: Measurements denote the cost of similarity computation and top-k selection. Both CPU and GPU paths operate on pre-normalized unit vectors for a fair comparison. CPU baseline uses `np.argpartition` for efficient partial sort.

---

## 🔬 Core Technologies

### 1. Metal Performance Shaders (MPS)
GrainVDB dispatches similarity operations directly to the Apple GPU via the MPS graph stack. This architecture exploits the **Unified Memory** of the M-series chips, allowing the GPU to access vector buffers without expensive PCIe transfer overhead.

### 2. Neighborhood Connectivity Audit (.audit())
Standard k-NN retrieval can lead to "Context Fractures" where semantically similar results are pulled from logically inconsistent neighborhoods (e.g., "Jaguar" as a vehicle vs. an animal).

GrainVDB implements a **Laplacian Connectivity Audit**:
- It constructs a local adjacency matrix from the top-k results.
- It computes the **Algebraic Connectivity** (second smallest eigenvalue of the Laplacian).
- A low connectivity score signals that the retrieved context is fragmented, allowing the application to flag potential hallucinations *before* they reach the LLM.

---

## 🚀 Quick Start

```python
from grain_vdb import GrainVDB
import numpy as np

# Initialize with 128-dimensional space
vdb = GrainVDB(dim=128)

# Ingest data
vectors = np.random.randn(1000, 128)
vdb.add_vectors(vectors)

# Query with sub-30ms latency
scores, indices, latency = vdb.query(np.random.randn(128), k=10)

# Audit for context consistency
connectivity = vdb.audit(indices)
if connectivity < 0.1:
    print("Warning: Context Fracture Detected.")
```

---

## 🏗️ Technical Roadmap
- [ ] C++/Metal Custom Core: Direct `MTLDevice` dispatch (see `src/grainvdb.mm`).
- [ ] Quasicrystal Phase Coding: High-dimensional quantization for 16x compression.
- [ ] Sheaf-theoretic Complex: Formal Çech cohomology for multi-hop RAG verification.

---

**Author**: Adam Sussman  
**License**: Proprietary / Early Access
