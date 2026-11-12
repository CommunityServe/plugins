// turboexl3_glm53f TURBOEXL3_GLM53F_ROUTER_ONE (src/fused_router.zig): a one-launch call's arrival, after phase 1
// (router_p1.metal, its chain's partial stored device-coherently). Every thread fences its device stores at device
// scope, the threadgroup meets, and its first thread counts the group's arrival on the call's own counter (`counter`: a
// fresh zeroed array a call, so no two calls share one). The group that counts last finds the other NTG - 1 counted
// before it, so every partial stored: it fences again and runs phase 2; every other group returns. Nothing waits on
// another group, and only the order of the work moves: phase 2 reads the partials the two-launch call's second launch
// reads.
constexpr uint NTG = uint(NB * (K / PS));
threadgroup uint one_last;
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
threadgroup_barrier(mem_flags::mem_device);
if (tid == 0) {
  device atomic_uint* arrivals = reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(counter));
  one_last = (atomic_fetch_add_explicit(arrivals, 1u, memory_order_relaxed) == NTG - 1u) ? 1u : 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (one_last == 0u) {
  return;
}
atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
