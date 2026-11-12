// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE (src/fused_router.zig): a phase 1 partial read device-coherently by
// the threadgroup that arrives last (an atomic load: the f32 bits phase 1's atomic store left).
static inline float pload(device float* p, size_t i) {
  return as_type<float>(atomic_load_explicit(reinterpret_cast<device atomic_uint*>(p + i), memory_order_relaxed));
}
