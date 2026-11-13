//! `MIMO_DIAG` (PROFILE builds only): diagnostic routes that are never a served or headline route.
//! `MIMO_DIAG=missfree`: in the decode phase every routed expert that is not resident is served from a resident slot of
//! its layer (a fixed substitute, `MissFree.sub`), so no decode route plans a load: no SSD read, no pre-read, no
//! speculation, no admission or eviction (the resident set the prompt pass left stays as it was). Outputs are garbage
//! by design; the cell's timing is the decode cycle's (or MTP round's) compute floor. The prompt pass is unchanged.
//! The env is read once, at the source's construction (`Source.armDiag`). In a served build `compiled` is false: the
//! stream source's field is `void` and every branch on it is compiled out. Test builds compile the route (never the
//! env read) so the CPU tests drive it directly.
const std = @import("std");
const builtin = @import("builtin");
const profile = @import("profile.zig");
const expert_policy = @import("../sdk_ext.zig").expert.policy;

pub const compiled = profile.enabled or builtin.is_test;

pub const Mode = enum { off, missfree };

/// The env's value: unset, empty or "off" = off; "missfree"; anything else refused (the construction fails once).
pub fn parse(value: ?[]const u8) error{MimoDiag}!Mode {
    const v = value orelse return .off;
    if (v.len == 0 or std.mem.eql(u8, v, "off")) return .off;
    if (std.mem.eql(u8, v, "missfree")) return .missfree;
    return error.MimoDiag;
}

/// MiMo's widest geometry (47 routed layers x 256 experts) with room for one more layer.
pub const max_layers = 48;
pub const max_experts = 256;

/// Per routed layer (relative), each expert's substitute: itself when resident, else the resident expert
/// `residents[e % n]` (the layer's residents in expert order), so a layer's misses spread over distinct resident rows
/// instead of all reading one. Rebuilt at every decode boundary (`fill` per layer after `Stream.grow`); constant through
/// the decode phase since a missfree route admits nothing.
pub const MissFree = struct {
    map: [max_layers][max_experts]u16 = undefined,
    /// Residents per layer at the last fill (the receipt's proof that every layer had a slot to serve from).
    residents: [max_layers]u16 = @splat(0),
    n_layers: u32 = 0,
    n_experts: u32 = 0,
    /// Routed ids substituted (misses removed) since the last fill, and routes taken (profile line, not a hot-path probe
    /// of a served route: this route exists only in this diagnostic).
    substituted: u64 = 0,
    routes: u64 = 0,

    pub fn sub(self: *const MissFree, layer: u32, e: u16) u16 {
        return self.map[layer][e];
    }
};
