/**
 * @file gv_core.h
 * @brief GrainVDB Core API: Native Metal Engine
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
 * @brief Allocate system state for rank-R manifold.
 * @param library_path Path to the compiled .metallib file.
 */
gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path);

/**
 * @brief Ingest signal data into the primary manifold.
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count,
                   bool fold);

/**
 * @brief Resolve manifold interference for a given probe.
 * @return latency_ms Measured query time in milliseconds.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Verify topological neighborhood consistency.
 */
float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count);

/**
 * @brief Release system state.
 */
void gv1_ctx_destroy(gv1_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // GV_CORE_H
