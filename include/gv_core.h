/**
 * @file gv_core.h
 * @brief Proprietary Core API - System State Machine
 *
 * PROPRIETARY AND CONFIDENTIAL.
 * Copyright (c) 2025 AuraWifi / Adam Sussman.
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
 */
gv1_state_t *gv1_ctx_create(uint32_t rank);

/**
 * @brief Ingest signal data into the primary manifold.
 * @param fold If true, applies phase folding (Middle-Out).
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count,
                   bool fold);

/**
 * @brief Resolve manifold interference for a given probe.
 * @return latency_ms Performance metric.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Verify topological gluing (Sheaf) across results.
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
