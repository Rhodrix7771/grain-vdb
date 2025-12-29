# GrainVDB 🌾
### Native Metal Vector Engine for Apple Silicon
**High-performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built specifically for macOS and Apple Silicon. It bypasses the overhead of standard frameworks by using a direct Objective-C++ bridge to Metal Performance Shaders, enabling efficient brute-force search on massive vector manifolds.

---

## 📊 Industrial Benchmark (1 Million Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| Query Latency (k=10) | ~18 ms | **~4.9 ms** |
| Throughput | 55.5 req/s | **204.1 req/s** |

**Hardware**: MacBook M2 (Unified Memory).
**Methodology**: Measurements denote end-to-end query latency for similarity computation and top-k selection on pre-normalized unit vectors.
- **CPU Baseline**: Uses `np.argpartition` for efficient partial sort (O(N) complexity).
- **GrainVDB**: Accelerates the similarity discovery (Matrix-Vector multiplication) via custom Metal kernels. The top-k selection is performed on the CPU using a priority queue over the shared unified memory buffer.

---

## 🔬 Core Technologies

### 1. Unified Memory Bridge
GrainVDB utilizes Apple Silicon's Unified Memory Architecture. By mapping host-side Python buffers directly into the GPU's address space using `MTLBuffer(options: .storageModeShared)`, we eliminate expensive memory copies. The GPU performs the heavy dot-product calculations directly on the source data.

### 2. Custom Metal Kernels
The engine dispatches custom Metal Shading Language (MSL) kernels optimized for the M-series GPU. These kernels perform vectorized `half4` operations, maximizing throughput for high-dimensional similarity resolution.

### 3. Context Consistency Audit
To identify "Semantic Fractures" (where top results come from logically inconsistent clusters), GrainVDB includes a topological audit layer. It calculates the **Neighborhood Connectivity** (average pairwise similarity above a threshold) to help identify potential RAG hallucinations.

---

## 🚀 Getting Started

### 1. Build the Native Core
GrainVDB requires a native build to link against your local Metal frameworks:
```bash
chmod +x build.sh
./build.sh
```

### 2. Run the Benchmark
```bash
python3 main.py
```

### 3. Basic Implementation
```python
from grainvdb.engine import GrainVDB
import numpy as np

# Initialize with 128-dimensional vectors
vdb = GrainVDB(dim=128)

# Add 1 million vectors (Normalized internally)
vectors = np.random.randn(1000000, 128).astype(np.float32)
vdb.add_vectors(vectors)

# Query in ~5ms
indices, scores, latency_ms = vdb.query(np.random.randn(128), k=10)

# Check context consistency
score = vdb.audit_consistency(indices)
```

---

## 🏗️ Engineering Roadmap
- [ ] **Quantized Storage**: INT8 and INT4 storage for 10M+ vector scales.
- [ ] **GPU-Side Selection**: Implementing Bitonic Sort or Heap-select on-GPU to furtherize reduce CPU overhead.
- [ ] **Sheaf-theoretic Graph RAG**: Formal Çech cohomology for multi-hop manifold verification.

---

**Author**: Adam Sussman  
**License**: MIT
