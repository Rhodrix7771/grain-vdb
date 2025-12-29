#include <metal_stdlib>
using namespace metal;

// ⚡ Hardware-Accelerated Kernel
// Uses vectorized 4-way SIMD for maximum throughput.
kernel void gv_k_m1_h(
    device const half4* probe [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* mag [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    
    // Each thread processes one manifold entry
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    for (uint i = 0; i < v_rank; i++) {
        half4 p = probe[i];
        half4 m = manifold[offset + i];
        dot_val += (float)dot(p, m);
    }
    
    mag[id] = dot_val;
}
