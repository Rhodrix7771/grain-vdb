import ctypes
import numpy as np
import os
import time
import subprocess
from typing import Tuple, List

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
        # Determine paths
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_path = os.path.join(base_path, "libgrainvdb.dylib")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"Native binary not found at {lib_path}. "
                "Please run 'python setup.py build_ext --inplace' or the provided build script."
            )
            
        self.lib = ctypes.CDLL(lib_path)
        
        # Define C-API signatures
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32]
        
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
        
        self.lib.gv1_manifold_fold.restype = ctypes.c_float
        self.lib.gv1_manifold_fold.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_uint32, 
            ctypes.POINTER(ctypes.c_uint64), 
            ctypes.POINTER(ctypes.c_float)
        ]
        
        self.lib.gv1_topology_audit.restype = ctypes.c_float
        self.lib.gv1_topology_audit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        
        self.lib.gv1_ctx_destroy.argtypes = [ctypes.c_void_p]
        
        self.ctx = self.lib.gv1_ctx_create(self.dim)
        if not self.ctx:
            raise RuntimeError("Failed to initialize GrainVDB Native Context (check Metal permissions).")

    def add_vectors(self, vectors: np.ndarray):
        """
        Upload vectors to the GPU's unified memory buffer.
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {vectors.shape[1]}")
            
        # Normalize for cosine similarity consistency
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        
        # Ensure contiguous memory for C-API
        data = np.ascontiguousarray(vectors, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data), False)

    def query(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Execute sub-millisecond similarity search using custom Metal kernels.
        """
        probe = np.ascontiguousarray(query_vec, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        results_idx = np.zeros(k, dtype=np.uint64)
        results_scores = np.zeros(k, dtype=np.float32)
        
        idx_ptr = results_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        score_ptr = results_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Returns latency measured inside the C++ driver
        latency_ms = self.lib.gv1_manifold_fold(self.ctx, p_ptr, k, idx_ptr, score_ptr)
        
        return results_idx, results_scores, latency_ms

    def audit_consistency(self, indices: np.ndarray) -> float:
        """
        Computes the average pair-wise similarity (Gluing Energy) of retrieved results.
        High values indicate a highly consistent semantic neighborhood.
        """
        idx_ptr = indices.astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.gv1_topology_audit(self.ctx, idx_ptr, len(indices))

    def __del__(self):
        if self.ctx and self.lib:
            self.lib.gv1_ctx_destroy(self.ctx)
