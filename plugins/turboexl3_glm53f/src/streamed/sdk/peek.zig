//! What the registry routes on: a plugin's claim and the descriptions it claims from (docs/plugins.md: one
//! question per hook, asked in registry order). Pure host: nothing here reads a weight or touches MLX.

const std = @import("std");

/// How strongly a plugin claims what it is shown: the registry takes the highest, ties in registry order.
pub const Priority = enum(u8) {
    /// served through a generic path (MLX's own gather kernels)
    generic = 1,
    /// the plugin's own format, every tensor and layer covered by its code
    native = 2,
};

/// Why a claim declined or a hook refused, for the one log line the caller writes.
pub const Diag = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diag) []const u8 {
        return self.buf[0..self.len];
    }

    /// Replaces the message (truncated to the buffer).
    pub fn set(self: *Diag, comptime fmt: []const u8, args: anytype) void {
        self.len = if (std.fmt.bufPrint(&self.buf, fmt, args)) |m| m.len else |_| self.buf.len;
    }
};

/// A model as discovery sees it before any weight loads: its directory and config.json, built once per model by
/// the host. The `arch`, `source`, `engine` and `expert_source` kinds claim from it.
pub const ConfigPeek = struct {
    model_dir: []const u8,
    /// config.json as read: an arch's own parse reads the whole document.
    text: []const u8,
    /// config.json's top-level object.
    root: std.json.ObjectMap,

    /// Parses `text` (kept by reference) into `arena`.
    pub fn parse(arena: std.mem.Allocator, model_dir: []const u8, text: []const u8) error{ ConfigNotJson, ConfigNotObject, OutOfMemory }!ConfigPeek {
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.ConfigNotJson,
        };
        if (v != .object) return error.ConfigNotObject;
        return .{ .model_dir = model_dir, .text = text, .root = v.object };
    }

};

/// One per-expert tensor of a weight group, as the group's description names it.
pub const Segment = struct { name: []const u8, dtype: []const u8, shape: []const u64 };

/// One layer of a weight group: its bits per weight (EXL3's K; a quantization_config's bits) and its per-expert
/// tensors.
pub const LayerPeek = struct { bits: u32, segments: []const Segment };

/// A routed-expert weight group's description at load: what the `quant` kind claims from, once per weight group.
/// `quantization` is opaque to the host: each quant reads its own fields (EXL3: the bank manifest's `quantization`
/// object; the gather quant: the group's quantization_config, `null` for dense experts).
pub const GroupPeek = struct {
    quantization: std.json.Value,
    hidden: u64,
    inter: u64,
    n_experts: u64,
    n_layers: u64,
    layers: []const LayerPeek,
};

const testing = std.testing;
