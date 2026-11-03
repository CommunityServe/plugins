#ifndef S41_ANE_BRIDGE_H
#define S41_ANE_BRIDGE_H

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The only interface to the private AppleNeuralEngine framework. Everything
 * above this file speaks MIL text, weight blobs, and IOSurfaces; everything
 * below it is objc_msgSend into _ANEInMemoryModel and friends.
 *
 * Binding rule: request inputs bind to MIL function parameters in
 * ALPHABETICAL name order, not declaration order. Callers must name their
 * parameters so the sort order matches the surface indices (i0_*, i1_*, or
 * x0..x7). Same-shape mismatches swap data silently. */

typedef struct s41_ane_model s41_ane_model;

int s41_ane_bridge_available(void);

/* 16KB-aligned, host-mapped, zero-fill on create is the caller's job. */
IOSurfaceRef s41_ane_bridge_surface(size_t bytes);

/* Compile (or restore from the compile cache) and load one MIL program.
 * Takes ownership of the malloc'd weights blob. The input surfaces bind to
 * indices 0..input_count-1 and `output` to output index 0.
 *
 * `procedure_count` is how many `procedureNNN` functions the MIL declares
 * (1 for a plain `main` program): one _ANERequest is built per procedure
 * and dispatched by index, all sharing the same surfaces.
 *
 * `ane_instance` pins the program to one Neural Engine die: 0 keeps the
 * empty options dict (no affinity — the driver schedules it), 1..4 pass
 * kANEFAneInstanceHint together with the single-ANE procedure-variant hint.
 * BOTH keys are required — the scheduler only honours an instance hint for
 * the single-ANE variant. */
s41_ane_model *s41_ane_model_create(const char *name, const char *mil,
                                  void *weights, size_t weight_bytes,
                                  IOSurfaceRef *inputs, uint32_t input_count,
                                  IOSurfaceRef output,
                                  uint32_t procedure_count, int ane_instance,
                                  char *error, size_t error_size);
void s41_ane_model_free(s41_ane_model *model);

int s41_ane_model_eval(s41_ane_model *model, uint32_t procedure, char *error,
                       size_t error_size);

/* Residency rotation: unload releases the wired compiled net but keeps the
 * handle and request; reload re-wires it from the compile cache. */
int s41_ane_model_unload(s41_ane_model *model, char *error, size_t error_size);
int s41_ane_model_reload(s41_ane_model *model, char *error, size_t error_size);

/* Load cost of the create: a fresh compile or a cached load. */
double s41_ane_model_compile_seconds(const s41_ane_model *model);
bool s41_ane_model_cache_hit(const s41_ane_model *model);

/* Compiled models persist in ~/.mlx-serve/ane-cache/entries/<content-hash>
 * (the hash covers the MIL text AND the weights, so same-shape
 * different-weight models never collide; MLX_SERVE_ANE_CACHE_DIR overrides
 * the root). Only the compiled artifacts are kept; the `data` file embeds
 * the constants. */

/* Names the (seam + shape, share) the next compiles belong to: entries of
 * the same group at a different variant are pruned on the cold-compile
 * path, so a share sweep replaces its predecessors. Empty group = untagged. */
void s41_ane_cache_lineage(const char *group, const char *variant);
/* The variant of the most recently used entry tagged with `group` (empty
 * when none), so a later build can reuse the share that set was built at. */
void s41_ane_cache_variant(const char *group, char *out, int out_len);
/* Apply the cache's caps now (as every cold store does): other variants of
 * the current group first, then the oldest entries outside the current set. */
void s41_ane_cache_prune(void);
/* After a build: drop the current group's entries it did not restore or store. */
void s41_ane_cache_sweep(void);

/* ── programs (the ANE LM engine) ──
 * One MIL program of several functions ("procedures") over one weight blob, loaded as `instances`
 * model objects of the same compiled program: one weight copy, and each instance its own queue (two
 * instances evaluate at once on two ANE clusters; one instance's requests run in turn). Inputs and
 * outputs bind by the names in the compiled model's attributes, in the order `s41_ane_program_io`
 * lists them; their surfaces follow its strides (rows padded to 64 bytes). */
typedef struct s41_ane_program s41_ane_program;
typedef struct s41_ane_request s41_ane_request;

typedef struct {
    char name[128];
    uint32_t dtype; /* 0 fp16, 1 fp32, 2 int8, 3 uint8 */
    uint32_t batches, channels, depth, height, width;
    uint64_t batch_stride, depth_stride, plane_stride, row_stride;
} s41_ane_io;

/* Takes ownership of the malloc'd weights blob (freed once compiled). Loads it `instances` times;
 * `ane_hints` (NULL, or one per instance; 0 = none) pins each load to an ANE instance: two loads
 * evaluated at once without hints can land on one cluster. */
s41_ane_program *s41_ane_program_create(const char *name, const char *mil,
                                        void *weights, size_t weight_bytes,
                                        uint32_t instances, const uint32_t *ane_hints,
                                        char *error, size_t error_size);
void s41_ane_program_free(s41_ane_program *program);
/* The procedure index of the MIL function `function`; -1 when absent. */
int s41_ane_program_procedure(s41_ane_program *program, const char *function);
uint32_t s41_ane_program_io_count(s41_ane_program *program, uint32_t procedure, int output);
int s41_ane_program_io(s41_ane_program *program, uint32_t procedure, int output,
                       uint32_t index, s41_ane_io *out);
s41_ane_request *s41_ane_request_create(s41_ane_program *program, uint32_t instance,
                                        uint32_t procedure, IOSurfaceRef *inputs,
                                        uint32_t input_count, IOSurfaceRef *outputs,
                                        uint32_t output_count, char *error,
                                        size_t error_size);
/* 1 on success. */
int s41_ane_request_eval(s41_ane_request *request, char *error, size_t error_size);
void s41_ane_request_free(s41_ane_request *request);
double s41_ane_program_compile_seconds(const s41_ane_program *program);
bool s41_ane_program_cache_hit(const s41_ane_program *program);

#endif
