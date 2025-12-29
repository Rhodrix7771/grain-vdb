import ctypes
import numpy as np
import os
from typing import Tuple

class GrainVDB:
    """
    GrainVDB: Native Metal-Accelerated Vector Engine for Apple Silicon.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.lib = None
        self.ctx = None
        self._load_native_library()
        
    def _load_native_library(self):
        # Determine path to shared library
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_name = "libgrainvdb.dylib"
        lib_path = os.path.join(base_path, lib_name)
        
        if not os.path.exists(lib_path):
            # Try to build if missing? No, user should run build.sh.
            raise FileNotFoundError(f"Native binary {lib_name} not found. Run ./build.sh first.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # 1. gv1_ctx_create
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        
        # 2. gv1_data_feed
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
        
        # 3. gv1_manifold_fold
        self.lib.gv1_manifold_fold.restype = ctypes.c_float
        self.lib.gv1_manifold_fold.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_uint32, 
            ctypes.POINTER(ctypes.c_uint64), 
            ctypes.POINTER(ctypes.c_float)
        ]
        
        # 4. gv1_topology_audit
        self.lib.gv1_topology_audit.restype = ctypes.c_float
        self.lib.gv1_topology_audit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        
        # 5. gv1_ctx_destroy
        self.lib.gv1_ctx_destroy.argtypes = [ctypes.c_void_p]
        
        # Initialize context
        metallib_path = os.path.join(base_path, "grainvdb/gv_kernel.metallib")
        if not os.path.exists(metallib_path):
             # Fallback check
             metallib_path = "grainvdb/gv_kernel.metallib"
             
        self.ctx = self.lib.gv1_ctx_create(self.dim, metallib_path.encode('utf-8'))
        if not self.ctx:
            raise RuntimeError(f"Failed to initialize GrainVDB context using {metallib_path}")

    def add_vectors(self, vectors: np.ndarray):
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Dim mismatch. Expected {self.dim}, got {vectors.shape[1]}")
        
        # Input normalization for unit-vector cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data), False)

    def query(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        probe = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        probe = np.ascontiguousarray(probe, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        results_idx = np.zeros(k, dtype=np.uint64)
        results_scores = np.zeros(k, dtype=np.float32)
        idx_ptr = results_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        score_ptr = results_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        latency_ms = self.lib.gv1_manifold_fold(self.ctx, p_ptr, k, idx_ptr, score_ptr)
        return results_idx, results_scores, latency_ms

    def audit_consistency(self, indices: np.ndarray) -> float:
        idx_ptr = indices.astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.gv1_topology_audit(self.ctx, idx_ptr, len(indices))

    def __del__(self):
        if self.ctx and self.lib:
            self.lib.gv1_ctx_destroy(self.ctx)
