// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE1 (src/fused_router.zig composes it; a std.fmt template: doubled braces
// are one): a one-row router call in one launch at GLM-5.3-Flash's dims (K 4096, N 288). The stock route at one row
// casts x to f32, runs MLX's gemv (gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0: 18 threadgroups of four
// simdgroups, four outputs a simdgroup) over the router's f32 copy, then the stock router text. Here each of the 18
// threadgroups runs MLX's own GEMVKernel::run at those template arguments (the vendored gemv.h with its operands' types
// and its output's store edited: x read as stored, bf16, and the raw router as stored, each widened to f32 as the cast
// and the copy widen them, exactly; each logit stored device-coherently, an atomic store of the same f32 bits), so the
// same loads, multiply-adds and lane reductions in the same order give the same logits. Every threadgroup fences its
// stores at device scope and counts its arrival on the call's own counter (a fresh zeroed array a call); the one that
// counts last runs the stock router text (sigmoid_bias) over the 288 logits, its read device-coherent (`pload`) and its
// score loop over this launch's 128 threads, the same statements per expert; the other 17 return.
threadgroup float ro1_tgp[1];
GEMVKernel<float, 4, 1, 1, 32, 4, 4, false>::run(w, x, logits, logits, ro1_in, ro1_out, ro1_ld, ro1_alpha, ro1_beta, ro1_bias_stride, ro1_tgp, threadgroup_position_in_grid, thread_position_in_threadgroup, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
constexpr uint RO1_NTG = 18u;
threadgroup uint ro1_last;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup_barrier(mem_flags::mem_device);
if (thread_position_in_threadgroup.x == 0u) {{
  device atomic_uint* ro1_arrivals = reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(counter));
  ro1_last = (atomic_fetch_add_explicit(ro1_arrivals, 1u, memory_order_relaxed) == RO1_NTG - 1u) ? 1u : 0u;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (ro1_last == 0u) {{
  return;
}}
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
{s}
