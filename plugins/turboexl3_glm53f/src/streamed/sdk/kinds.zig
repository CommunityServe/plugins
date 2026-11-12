//! The `source` and `engine` kinds' registry tables (docs/plugins.md). Each carries the routing question; a kind's
//! full interface lands with its first consumer. A quant or an expert source is an arch's internal, bound at comptime.

const std = @import("std");
const peek = @import("peek.zig");
const check = @import("check.zig");

/// Opens a non-HF container: claims a model path before any file is read as a model directory.
pub const Source = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

};

/// A whole engine behind an opaque session (ds4, llama.cpp): it gets none of the host's stack below HTTP.
pub const Engine = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

};

const testing = std.testing;
