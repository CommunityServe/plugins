//! Kernel routes over a pinned registry `R` (G5; mlx-stream's is `exl3_kernels`): the machinery every consumer's
//! routes share. A route is built once (static inputs, per-M launch tables, the weights it binds checked
//! against the kernel signature and refused by name), then called with the per-call arrays only.
//! A call launches exactly what the Python lane's kernel call launches (inputs in the kernel's
//! order, the lane's grid / threadgroup / template) and returns the same outputs.
//!
//! mlx-stream's consumers: the EXL3 routed-expert quant (`exl3_quant`, C2) and the V4.1 trunk
//! (`dsv41_kernel_routes`). Both run on the one kernel set the load context owns
//! (`KernelSet(R).Set`).
//!
//! Routes are generic over the model's graph backend `G`, which provides `T`, `shapeOf`
//! (a value with `slice()`), `dtypeOf`, `hostArray`, `keep`, `release`, `reshape`, `astype`
//! and one launch:
//!     launch(g: *G, k: Kernel, inputs: []const G.T, cfg: *const LaunchConfig, out: []G.T) !void
//! (a real backend: `Bound.apply`, the outputs owned by the backend's scope; a trace backend:
//! nodes of `cfg`'s output shapes and dtypes).

const std = @import("std");
const mlx = @import("sdk").mlx;
