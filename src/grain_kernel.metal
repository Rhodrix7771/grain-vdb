#include <metal_stdlib>
using namespace metal;

// ⚡ Hardware-Optimized Kernel (Vectorized 4-way SIMD)
// Using half4 for 4x floating-point throughput per instruction cycle.
kernel void gv_k_m1_h(
    device const half4* probe [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* mag [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    
    // Each thread processes one manifold entry
    // Rank is divided by 4 because we use half4
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    for (uint i = 0; i < v_rank; i++) {
        // half4 * half4 performs 4 multiplications in parallel
        // .x, .y, .z, .w are summed into the running dot product
        half4 p = probe[i];
        half4 m = manifold[offset + i];
        dot_val += (float)dot(p, m);
    }
    
    mag[id] = dot_val;
}
