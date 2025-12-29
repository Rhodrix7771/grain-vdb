/**
 * GrainVDB Core Interface
 * -----------------------
 * Licensed under the MIT License.
 */

#ifndef GV_CORE_H
#define GV_CORE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gv1_state_t gv1_state_t;

/**
 * @brief Initialize the GrainVDB context using a specified Metal library.
 * @param rank Dimensionality of the vectors.
 * @param library_path Absolute or relative path to the .metallib file.
 */
gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path);

/**
 * @brief Load float32 vectors into the GPU's shared memory.
 * @param buffer Pointer to float32 data.
 * @param count Number of vectors.
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count);

/**
 * @brief Resolve similarity search on the GPU.
 * @return kernel_latency_ms Wall-time for GPU dispatch and synchronization in
 * ms.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Topology/Consistency Audit.
 * Calculates neighborhood density as a connectivity heuristic.
 */
float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count);

/**
 * @brief Free resources.
 */
void gv1_ctx_destroy(gv1_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // GV_CORE_H
