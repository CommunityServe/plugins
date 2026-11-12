//! Conformance (docs/plugins.md): the checks a plugin's kinds run against the SDK's contracts, and the
//! fakes the host's own tests drive. Every check declares its lane:
//! - `cpu`: `zig build conformance` with no device; the lane fails if a Metal device was created.
//! - `gpu_small`: fixture shapes on a Mac with a GPU.
//! - `window`: a plugin's own full-size checks, never in the suite.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx_host").mlx;
const peek = @import("peek.zig");
const arch = @import("arch.zig");
const spec = @import("spec.zig");
const bill = @import("memory_bill.zig");

// ── The CPU lane's device check ──

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

// ── Claims (cpu) ──

// ── Bills, fills and admission (cpu) ──

// ── Fixture files (cpu) ──

// ── Fakes for the host's tests (cpu) ──

const testing = std.testing;
