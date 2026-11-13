//! glm5_next's prompt levers (the prefill audit's exact levers 1-5): construction-time routes set by the
//! pack's `expert_streaming.settings` block, read once, refused by name. The defaults are batch 27's measured bundle
//! (g22-pw at 190/217, lane on, draft_depth 3 + draft_gate 0.5: prefill 1,010 -> 1,073, +6.18 %, exact, ids identical):
//! `exl3_wave_rows` 2048, `prompt_route_cut` on, `prompt_static_order` sampled, `ttft_tail` deferred; g22-exl3's routes
//! (1024, off, off, off: the control) stay selectable by name.
//!   - `exl3_wave_rows` (the EXL3 family's DIG-X prompt waves): a wave's rows at most, default 2,048 (the control 1,024,
//!     MiMo's projection tile), 128 .. 4,096 in steps of the GEMMs' 128-row M tile. Exact: each M tile's rows accumulate their own K in the
//!     same order whatever wave holds them (a row's output does not depend on its wave). The bill's `expert_workspace`
//!     prices two waves in flight at the setting (bill.zig `workspaceBytes`); a value off the grid refuses the load.
//!   - `prompt_route_cut` (`off` | `on`): the wide call's rows bucketed by group once (the shared scheduler's
//!     `Wide.route_cut`: each group's bank passes over its own rows, the distinct experts and their counts in one pass)
//!     and, on the EXL3 family, each prompt call's rows ordered by slot with a stable counting sort (the block sort's
//!     order exactly). Host planning only: the same plans and waves, byte for byte.
//!   - `prompt_static_order` (`off` | `sampled`): P1's predictor (graph.zig `predictor`) routes at most `sampled_spans`
//!     evenly spaced spans of a prompt call (every span of a call of fewer): its counts only rank the read-ahead seed,
//!     never a row's experts. Token-identical.
//!   - `ttft_tail` (`off` | `deferred`): the prompt pass returns its logits before the MTP lane's catch-up and the phase
//!     change (the prompt scratch freed, every layer grown to its decode rows); both run at the request's first decode
//!     call, in the same order, on the same arrays, before its forward. Exact; the bill's phases are unchanged (the
//!     prompt phase's arrays live until the phase change either way).

const std = @import("std");

pub const RouteCut = enum { off, on };
pub const StaticOrder = enum { off, sampled };
pub const TtftTail = enum { off, deferred };

/// The DIG-X prompt wave's rows: the default (batch 27's measured bundle), the control's (g22-exl3's: MiMo's projection
/// tile; by name), the bounds and the grid (the GEMMs' M tile). 2,048 rows bill as 1,024 (bill.zig `tileBytes`: both
/// inside NVFP4's tile).
pub const default_wave_rows: u32 = 2048;
pub const control_wave_rows: u32 = 1024;
pub const min_wave_rows: u32 = 128;
pub const max_wave_rows: u32 = 4096;
pub const wave_rows_step: u32 = 128;

/// The keys this file reads (settings.zig `unknownSetting` admits them beside `setting_keys`).
pub const keys = [_][]const u8{ "exl3_wave_rows", "prompt_route_cut", "prompt_static_order", "ttft_tail" };

pub const Settings = struct {
    wave_rows: u32 = default_wave_rows,
    route_cut: RouteCut = .on,
    static_order: StaticOrder = .sampled,
    ttft_tail: TtftTail = .deferred,
    /// The settings named `exl3_wave_rows` (on the NVFP4 family the default is inert; a named size other than the
    /// control's refuses).
    wave_rows_named: bool = false,
    /// A key given a value it does not take (refused by name in `check`).
    malformed: struct { wave_rows: bool = false, route_cut: bool = false, static_order: bool = false, ttft_tail: bool = false } = .{},

};

const testing = std.testing;
