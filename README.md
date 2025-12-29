# GrainVDB 🌾
### Native Metal Vector Engine for Apple Silicon
**High-performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built specifically for macOS and Apple Silicon. It bypasses the overhead of standard databases by using a unified memory bridge between C++ and Metal Performance Shaders, enabling lightning-fast brute-force search on massive vector manifolds.

---

## 📊 Industrial Benchmark (1 Million Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| Query Latency (k=10) | ~16.5 ms | **~7.1 ms** |
| Throughput | 60.6 req/s | **140.8 req/s** |

**Hardware**: MacBook M2 (Unified Memory).
**Methodology**: Measurements denote end-to-end query latency for similarity computation and top-k selection on pre-normalized unit vectors.
- **CPU Baseline**: Uses `np.argpartition` for efficient partial sort (O(N) complexity).
- **GrainVDB**: Direct Metal dispatch via custom kernels with shared memory concurrency.

---

## 🔬 Core Technologies

### 1. Unified Memory Bridge
GrainVDB bypasses high-level AI frameworks like PyTorch for its core resolution logic. It utilizes a direct Objective-C++ bridge that maps Python buffers into the GPU's memory space. This avoids expensive memory copies and allows the GPU to process 1M+ vectors directly from the host RAM.

### 2. Custom Metal Kernels
The similarity discovery is performed by custom Metal Shading Language (MSL) kernels optimized for the M-series GPU architecture. These kernels exploit high-throughput FP16/FP32 dot-product operations and perform GPU-side selection to minimize host communication.

---

## 🚀 Getting Started

### 1. Build the Native Core
GrainVDB requires a native build to link against your macOS Metal frameworks:
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

# Query in ~7ms
indices, scores, latency_ms = vdb.query(np.random.randn(128), k=10)
```

---

## 🏗️ Engineering Roadmap
- [ ] **Quantized Storage**: INT8 and INT4 storage for 10M+ vector scales.
- [ ] **Neighborhood Consistency Audits**: Algebraic connectivity layers for hallucination detection.
- [ ] **Swift & Rust SDKs**: Direct C-API bindings for native mac-app integration.

---

**Author**: Adam Sussman  
**License**: MIT / Proprietary Alpha
