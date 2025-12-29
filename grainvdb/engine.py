import ctypes
import numpy as np
import os
from typing import Tuple

class GrainVDB:
    """
    Python interface to the Native GrainVDB Metal Core.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.lib = None
        self.ctx = None
        self._init_native()
        
    def _init_native(self):
        # Resolve dynamic library path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_path = os.path.join(root, "libgrainvdb.dylib")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native core missing. Run ./build.sh first.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # C-API Definitions
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        
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
        
        # Load metallib relative to project root
        metallib = os.path.join(root, "grainvdb/gv_kernel.metallib")
        self.ctx = self.lib.gv1_ctx_create(self.dim, metallib.encode('utf-8'))
        
        if not self.ctx:
            raise RuntimeError("Backend initialization failed.")

    def add_vectors(self, vectors: np.ndarray):
        """
        Loads vectors into GPU memory. Pre-normalization applied for cosine similarity.
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Dim mismatch. Expected {self.dim}")
            
        # Normalization (Cosine Similarity Requirement)
        v_f32 = vectors.astype(np.float32)
        norms = np.linalg.norm(v_f32, axis=1, keepdims=True)
        v_norm = v_f32 / (norms + 1e-9)
        
        data = np.ascontiguousarray(v_norm, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data))

    def query(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Execute sub-ms similarity discovery using the native bridge.
        """
        # Normalize probe
        q_f32 = query_vec.astype(np.float32)
        q_norm = q_f32 / (np.linalg.norm(q_f32) + 1e-9)
        
        probe = np.ascontiguousarray(q_norm, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        indices = np.zeros(k, dtype=np.uint64)
        scores = np.zeros(k, dtype=np.float32)
        
        idx_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        score_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # The returned value is the internal C++ performance metric (ms)
        internal_ms = self.lib.gv1_manifold_fold(self.ctx, p_ptr, k, idx_ptr, score_ptr)
        
        return indices, scores, internal_ms

    def audit(self, indices: np.ndarray) -> float:
        """
        Verify topological neighborhood consistency.
        """
        ptr = indices.astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.gv1_topology_audit(self.ctx, ptr, len(indices))

    def __del__(self):
        if self.ctx and self.lib:
            self.lib.gv1_ctx_destroy(self.ctx)
