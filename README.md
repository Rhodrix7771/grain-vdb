# GrainVDB 🌾
### Native Metal-Accelerated Vector Engine for Apple Silicon
**High-Performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built specifically for macOS and Apple Silicon. It utilizes a direct Objective-C++/Metal bridge to bypass the overhead of standard frameworks, enabling efficient brute-force similarity search on massive vector manifolds using hardware-accelerated SIMD.

---

## 📊 Performance Certification
**Hardware**: MacBook M2 (Unified Memory) | **Dataset**: 1 Million x 128D (Float32) | **OS**: macOS Sequioa
**Reference**: `benchmark.py` (Fixed Seed)

| Metric | CPU (NumPy + Accelerate) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| **Latency (p50)** | ~19 ms | **~5-8 ms** |
| **Speedup** | 1.0x | **2.2x - 4.0x** |

**Methodology**:
- **Wall-Time**: Measured at the Python boundary via `time.perf_counter()`. Includes bridge overhead, GPU execution, sync, and CPU selection.
- **CPU Baseline**: `np.dot` (Accelerate BLAS) + `np.argpartition` (O(N) selection) on pre-normalized vectors.
- **Native Implementation**: `half4` vectorized Metal kernel (FP16) on Unified Memory + `std::priority_queue` Top-K.

---

## 🔬 Core Architecture

### 1. Unified Memory Mapping
GrainVDB exploits the shared memory architecture of Apple M-series chips. By mapping host-resident Python/NumPy buffers directly into the GPU's address space using `storageModeShared` MTLBuffers, the engine minimizes the need for explicit data copying, maintaining a single addressable manifold.

### 2. Custom Metal Kernels
Similarity resolution is performed by vectorized `half4` kernels written in Metal Shading Language (MSL). These kernels are designed to maximize the memory bandwidth of the M-series SOC while performing low-precision (FP16) dot-product accumulation.

### 3. Semantic Audit Layer
The engine includes a built-in topological audit (`vdb.audit()`). It calculates the **Neighborhood Connectivity** (density of pairwise similarities) among retrieved results.
- **Function**: `density = (connected_pairs) / (total_possible_pairs)`
- **Utility**: Detects "Semantic Fractures" (low coherence) which often correlate with RAG hallucinations. 
- **Verification**: On random noise (default benchmark), density approaches `0.00`. On coherent clusters, density approaches `1.00`.

---

## 🚀 Getting Started

### 1. Build Native Core
GrainVDB requires a local build to link against your system's Metal frameworks.
```bash
chmod +x build.sh
./build.sh
```

### 2. Run Validation Benchmark
Verify performance and Recall@K on your machine.
```bash
python3 benchmark.py
```

### 3. Usage Example
```python
from grainvdb import GrainVDB
import numpy as np

# Initialize for 128-dimensional vectors
vdb = GrainVDB(dim=128)

# Add 1 million vectors (Normalized internally)
data = np.random.randn(1000000, 128).astype(np.float32)
vdb.add_vectors(data)

# High-speed search
indices, scores, latency_ms = vdb.query(np.random.randn(128), k=10)

# Topology Audit
density = vdb.audit(indices)
```

**Author**: Adam Sussman  
**License**: MIT
