//! copied_from: src/sdk/quant/resident.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The bytes a quant format's resident matrices are owed and do not hold yet: an EXL3 trunk pack's (turboexl3's
//! `resident_pending`, which qwen4_exp's trunk formats hand over when turboexl3 claims the trunk). The GPU ceiling holds
//! them aside until they are kept (`host/registry/memory_ceiling.zig`); for every other pack nothing is set and they are
//! 0. MLX-free, so the host reaches it without the formats' kernels.
/// A struct: `Shared` hands its storage out as `*anyopaque`, which a function pointer does not coerce to.
pub const Pending = struct { f: ?*const fn () callconv(.c) u64 = null };
/// Process-wide under `-Dunits` (`sdk/util/shared.zig`).
pub const pending = @import("../util/shared.zig").Shared(Pending, "quant_resident_pending", .{});
