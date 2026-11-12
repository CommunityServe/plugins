//! The `source` and `engine` kinds' registry tables (docs/plugins.md). Each carries the routing question; a kind's
//! full interface lands with its first consumer. A quant or an expert source is an arch's internal, bound at comptime.

const std = @import("std");
const peek = @import("peek.zig");
const check = @import("check.zig");

/// Opens a non-HF container: claims a model path before any file is read as a model directory.
pub const Source = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

    pub fn of(comptime T: type) Source {
        comptime {
            const w = "source " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
        }
        return .{ .name = T.name, .claims = T.claims };
    }
};

/// A whole engine behind an opaque session (ds4, llama.cpp): it gets none of the host's stack below HTTP.
pub const Engine = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

    pub fn of(comptime T: type) Engine {
        comptime {
            const w = "engine " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
        }
        return .{ .name = T.name, .claims = T.claims };
    }
};

const testing = std.testing;

fn claimsModel(p: *const peek.ConfigPeek) ?peek.Priority {
    return if (p.modelType() != null) .generic else null;
}
