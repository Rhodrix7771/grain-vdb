/**
 * GrainVDB Native Driver
 * ----------------------
 * Verified C++/Metal Bridge for Apple Silicon.
 * Licensed under the MIT License.
 */

#include "gv_core.h"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <vector>

typedef uint16_t gv_half_t;

// Half-precision float conversion (FP32 -> FP16)
static gv_half_t f32_to_f16(float f) {
  uint32_t i = *((uint32_t *)&f);
  int s = (i >> 16) & 0x00008000;
  int e = ((i >> 23) & 0x000000ff) - (127 - 15);
  int m = i & 0x007fffff;
  if (e <= 0) {
    if (e < -10)
      return s;
    m = (m | 0x00800000) >> (1 - e);
    return s | (m >> 13);
  } else if (e == 0xff - (127 - 15)) {
    return (m == 0) ? (s | 0x7c00) : (s | 0x7c00 | (m >> 13));
  } else {
    if (e > 30)
      return s | 0x7c00;
    return s | (e << 10) | (m >> 13);
  }
}

// Half-precision float conversion (FP16 -> FP32)
static float f16_to_f32(gv_half_t h) {
  uint32_t s = (h & 0x8000) << 16;
  uint32_t e = (h & 0x7c00) >> 10;
  uint32_t m = (h & 0x03ff) << 13;
  if (e == 0x1f) {
    e = 0xff;
  } else if (e == 0) {
    if (m != 0) {
      while (!(m & 0x00800000)) {
        m <<= 1;
        e--;
      }
      e++;
    }
  } else {
    e = e + (127 - 15);
  }
  uint32_t res = s | (e << 23) | m;
  return *((float *)&res);
}

struct gv1_state_t {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> p_state;
  uint32_t rnk;
  id<MTLBuffer> m_buf;
  uint32_t e_cnt;
  id<MTLLibrary> lib;
};

gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path) {
  gv1_state_t *state = new gv1_state_t();
  state->dev = MTLCreateSystemDefaultDevice();
  if (!state->dev)
    return nullptr;
  state->q = [state->dev newCommandQueue];
  state->rnk = rank;
  state->e_cnt = 0;
  state->m_buf = nil;

  NSError *err = nil;
  NSString *path = [NSString stringWithUTF8String:library_path];
  state->lib = [state->dev newLibraryWithFile:path error:&err];

  if (!state->lib) {
    std::cerr << "GrainVDB Error: Failed to load Metal library from "
              << library_path << std::endl;
    delete state;
    return nullptr;
  }

  id<MTLFunction> fn = [state->lib newFunctionWithName:@"gv_k_m1_h"];
  state->p_state = [state->dev newComputePipelineStateWithFunction:fn
                                                             error:&err];
  if (!state->p_state) {
    delete state;
    return nullptr;
  }

  return state;
}

void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count) {
  uint32_t total_elements = count * state->rnk;
  uint32_t b_len = total_elements * sizeof(gv_half_t);

  state->m_buf = [state->dev newBufferWithLength:b_len
                                         options:MTLResourceStorageModeShared];
  gv_half_t *dst = (gv_half_t *)[state->m_buf contents];

  for (uint32_t i = 0; i < total_elements; i++) {
    dst[i] = f32_to_f16(buffer[i]);
  }
  state->e_cnt = count;
}

float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag) {
  if (state->e_cnt == 0 || !state->m_buf)
    return 0.0f;

  // 1. Prepare Query (Shared Memory)
  id<MTLBuffer> p_buf =
      [state->dev newBufferWithLength:state->rnk * sizeof(gv_half_t)
                              options:MTLResourceStorageModeShared];
  gv_half_t *qp = (gv_half_t *)[p_buf contents];
  for (uint32_t i = 0; i < state->rnk; i++)
    qp[i] = f32_to_f16(probe[i]);

  // 2. Prepare Scores Buffer
  id<MTLBuffer> s_buf =
      [state->dev newBufferWithLength:state->e_cnt * sizeof(float)
                              options:MTLResourceStorageModeShared];

  // Start Timing at dispatch
  auto t_start = std::chrono::high_resolution_clock::now();

  // 3. Dispatch GPU Work
  id<MTLCommandBuffer> c_buf = [state->q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [c_buf computeCommandEncoder];
  [enc setComputePipelineState:state->p_state];
  [enc setBuffer:p_buf offset:0 atIndex:0];
  [enc setBuffer:state->m_buf offset:0 atIndex:1];
  [enc setBuffer:s_buf offset:0 atIndex:2];
  [enc setBytes:&state->rnk length:sizeof(uint32_t) atIndex:3];

  MTLSize g_sz = MTLSizeMake(state->e_cnt, 1, 1);
  NSUInteger max_t = state->p_state.maxTotalThreadsPerThreadgroup;
  MTLSize t_sz = MTLSizeMake(std::min((uint32_t)max_t, state->e_cnt), 1, 1);

  [enc dispatchThreads:g_sz threadsPerThreadgroup:t_sz];
  [enc endEncoding];

  [c_buf commit];
  [c_buf waitUntilCompleted];

  // 4. CPU Selection (Priority Queue)
  // Note: We use the priority queue to find the top results from the score
  // buffer.
  float *scores = (float *)[s_buf contents];
  typedef std::pair<float, uint64_t> Res;
  std::priority_queue<Res, std::vector<Res>, std::greater<Res>> pq;

  for (uint64_t i = 0; i < state->e_cnt; i++) {
    if (pq.size() < top) {
      pq.push({scores[i], i});
    } else if (scores[i] > pq.top().first) {
      pq.pop();
      pq.push({scores[i], i});
    }
  }

  // Capture results
  for (int i = (int)top - 1; i >= 0; i--) {
    result_mag[i] = pq.top().first;
    result_map[i] = pq.top().second;
    pq.pop();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  // Return explicit Milliseconds
  return (float)(std::chrono::duration_cast<std::chrono::microseconds>(t_end -
                                                                       t_start)
                     .count()) /
         1000.0f;
}

float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count) {
  if (count < 2 || !state->m_buf)
    return 1.0f;

  gv_half_t *m_ptr = (gv_half_t *)[state->m_buf contents];
  std::vector<float *> neighbors;
  for (uint32_t i = 0; i < count; i++) {
    float *v = new float[state->rnk];
    uint64_t offset = map[i] * state->rnk;
    for (uint32_t k = 0; k < state->rnk; k++)
      v[k] = f16_to_f32(m_ptr[offset + k]);
    neighbors.push_back(v);
  }

  int connections = 0;
  const float threshold = 0.85f;
  for (uint32_t i = 0; i < count; i++) {
    for (uint32_t j = i + 1; j < count; j++) {
      float dot = 0.0f;
      for (uint32_t k = 0; k < state->rnk; k++)
        dot += neighbors[i][k] * neighbors[j][k];
      if (dot > threshold)
        connections++;
    }
  }

  // Cleanup
  for (float *v : neighbors)
    delete[] v;

  int total_pairs = (count * (count - 1)) / 2;
  return (float)connections / (float)total_pairs;
}

void gv1_ctx_destroy(gv1_state_t *state) {
  if (state)
    delete state;
}
