//! The KV seam (R/mlx-stream/kv-seam-20261002.md): the lane storage a module-owned arch builds its decode state
//! from, generic over the arch's op backend `G` (`T`, `shapeOf`, `dtypeOf`, `keep`, `release`, `zeros`,
//! `hostArray`, `sliceUpdateDyn`, `slice`, `concat`). The lanes hold rows; they never compute attention. The route
//! is picked once at state construction and never changes during a request; a bounded lane is allocated once at
//! its cap and an append past it is refused by name before anything is written. No per-forward mark: `trim` is the
//! only in-request rewind.

const std = @import("std");

/// The widest array rank a lane slices (the backends' shape arrays hold at most this many dims).
pub const max_dims = 8;

/// Which backing the lanes use (Python precedence KV_BOUNDED > WINDOW_RING >
/// KV_CHUNK_GROW > plain), picked once when the sequence state is built.
pub const Route = enum {
    /// `_grow` concatenate stores: the stock path, every lever off.
    full_history,
    /// W73 `_GrowBuffer` lanes (geometric capacity from 256).
    chunk_grow,
    /// W80 `_WindowRing` window; compress / index `_GrowBuffer`s sized to
    /// `max_kv` when set, else geometric; the frontier a plain store.
    window_ring,
    /// W107: the ring plus compress / index / frontier preallocated to
    /// `max_kv` (geometric when unset), appends past the cap refused.
    bounded,
};

pub const Geometry = struct {
    route: Route = .full_history,
    /// The widest verify block one forward appends (DSpark depth + 1 fits).
    max_verify: u32 = 8,
    slack: u32 = 8,
    /// In-place appends between two ring compactions.
    headroom: u32 = 64,
    /// Preallocation capacity (`--max-kv`); null keeps geometric growth.
    max_kv: ?u32 = null,
};

/// A request's lane bound (the arch's constants): the positions a request that declared no budget may still
/// generate, and the rows one forward may append past the last committed position (a verify block).
pub const Bound = struct { headroom: u64, scratch: u32 };

/// One bounded lane of a request's plan: the rows of its cap and the bytes of one row.
pub const LanePlan = struct { rows: u64, row_bytes: u64 };
/// One ring of a request's plan: its window and the bytes of one ring row (over every array that row spans).
pub const RingPlan = struct { window: u64, row_bytes: u64 };
/// A request's KV as its arch describes it: the bounded lanes, each allocated once at its cap, and the rings.
pub const Plan = struct { lanes: []const LanePlan, rings: []const RingPlan };
pub const Phase = enum { prompt, decode };

pub const Error = error{ BoundedLaneFull, RingRollbackTooDeep, TrimPastStart };
