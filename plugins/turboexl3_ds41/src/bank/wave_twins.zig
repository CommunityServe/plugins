//! The scratch waves' write-into twins: a DIG-X wave launch's stock text rewritten so its outputs land
//! in buffers the caller binds as inputs (scratch_waves.zig's regions), derived at construction from the registry's
//! pinned stock texts (the quant's `IntoTwins`), never stored.
//!
//! MLX declares a custom kernel's outputs itself, as `device T*` parameters named in the call, so a stock text writes
//! its outputs by name and never declares them. The twin binds each output `o` as an input `o_dst` (MLX declares it
//! `const device T*`, a row-contiguous region view) and declares `device T* o = (device T*) o_dst;` itself before the
//! stock statements, so every store the stock text makes lands in the region; the stock statements are unchanged. Its
//! one MLX output is a `[1]` uint32 `carrier` the grid's first thread writes (a reader binding it is ordered after
//! every write of the launch); `dep`, one more input it never reads, is the launch before's carrier. The caller binds
//! whole regions (evaluated arrays: a lazy view of a larger buffer would reach a row-contiguous kernel through MLX's
//! `contiguous`, which copies a view whose buffer passes its bytes by more than 16 KiB, and the writes would land in
//! the copy). A dep twin (`checkDep`) is the stock text unchanged with one more input, `dep`: its outputs stay MLX's
//! own.
//!
//! The derivation refuses by name rather than guess: the `#pragma METAL fp contract` line where the twin's spec says
//! (the first statement, only comments before it, the prologue right after it; or none at all, the prologue first),
//! every output named by the stock text, and none of the names the twin adds (`<out>_dst`, `carrier`, `dep`) nor an
//! output's `_strides` / `_shape` (a twin output has neither) in the text or its header. Std only: tested without MLX.

const std = @import("std");

pub const Error = error{
    /// The stock text's `#pragma METAL fp contract` line is not where the twin's spec puts it.
    TwinPragmaAnchor,
    /// The stock text never names an output (nothing would land in its region).
    TwinOutputUnused,
    /// The stock text or its header names a name the twin adds, or an output's strides or shape.
    TwinNameTaken,
    /// An output's dtype has no Metal type name here.
    TwinOutputType,
};

/// One output of the stock launch: its name and Metal type (`metalType`).
pub const Out = struct { name: []const u8, mtype: []const u8 };

/// Where the stock text keeps `#pragma METAL fp contract(...)`: its first statement (only comment or blank lines before
/// it; the prologue goes right after it, under the pragma), or nowhere (the prologue goes first).
pub const Pragma = enum { first, none };

pub const pragma_key = "#pragma METAL fp contract";

/// MLX's Metal type name of an output dtype (mlx `get_type_string`), by the dtype's tag name; null: not a twin output.
pub fn metalType(dtype: []const u8) ?[]const u8 {
    const map = [_][2][]const u8{ .{ "float32", "float" }, .{ "float16", "float16_t" }, .{ "bfloat16", "bfloat16_t" }, .{ "uint32", "uint32_t" }, .{ "int32", "int32_t" } };
    for (map) |m| if (std.mem.eql(u8, m[0], dtype)) return m[1];
    return null;
}

fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// `text` names identifier `a ++ b` (not as part of a longer identifier).
fn hasWord(text: []const u8, a: []const u8, b: []const u8) bool {
    const n = a.len + b.len;
    if (n == 0 or text.len < n) return false;
    var i: usize = 0;
    while (i + n <= text.len) : (i += 1) {
        if (!std.mem.eql(u8, text[i..][0..a.len], a) or !std.mem.eql(u8, text[i + a.len ..][0..b.len], b)) continue;
        if (i > 0 and isIdent(text[i - 1])) continue;
        if (i + n < text.len and isIdent(text[i + n])) continue;
        return true;
    }
    return false;
}

/// Only blank lines and `//` comment lines in `text`.
fn onlyComments(text: []const u8) bool {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const l = std.mem.trim(u8, line, " \t\r");
        if (l.len > 0 and !std.mem.startsWith(u8, l, "//")) return false;
    }
    return true;
}

/// Every anchor checked: the byte offset of `stock` the prologue goes at.
pub fn anchor(stock: []const u8, header: []const u8, outs: []const Out, pragma: Pragma) Error!usize {
    for (outs) |o| {
        if (!hasWord(stock, o.name, "")) return error.TwinOutputUnused;
        for ([_][]const u8{ "_dst", "_strides", "_shape" }) |suffix| {
            if (hasWord(stock, o.name, suffix) or hasWord(header, o.name, suffix)) return error.TwinNameTaken;
        }
    }
    for ([_][]const u8{ "carrier", "dep" }) |w| {
        if (hasWord(stock, w, "") or hasWord(header, w, "")) return error.TwinNameTaken;
    }
    const at = std.mem.indexOf(u8, stock, pragma_key);
    switch (pragma) {
        .none => return if (at == null) 0 else error.TwinPragmaAnchor,
        .first => {
            const i = at orelse return error.TwinPragmaAnchor;
            if (!onlyComments(stock[0..i])) return error.TwinPragmaAnchor;
            const eol = std.mem.indexOfScalarPos(u8, stock, i, '\n') orelse return error.TwinPragmaAnchor;
            // one pragma only: a second one later would put stock statements outside the twin's reading of the first
            if (std.mem.indexOfPos(u8, stock, eol, pragma_key) != null) return error.TwinPragmaAnchor;
            return eol + 1;
        },
    }
}

/// A dep twin's anchor: neither the stock text nor its header names `dep` (the one input the twin adds).
pub fn checkDep(stock: []const u8, header: []const u8) Error!void {
    if (hasWord(stock, "dep", "") or hasWord(header, "dep", "")) return error.TwinNameTaken;
}

/// The twin's text (the caller frees it): the stock text up to the anchor; each output declared from its `_dst` input
/// and the carrier word; the rest of the stock text.
pub fn derive(a: std.mem.Allocator, stock: []const u8, header: []const u8, outs: []const Out, pragma: Pragma) (Error || error{OutOfMemory})![:0]u8 {
    const at = try anchor(stock, header, outs, pragma);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, stock[0..at]);
    for (outs) |o| try out.print(a, "    device {s}* {s} = (device {s}*) {s}_dst;\n", .{ o.mtype, o.name, o.mtype, o.name });
    try out.appendSlice(a, "    if (thread_position_in_grid.x == 0u && thread_position_in_grid.y == 0u && thread_position_in_grid.z == 0u) carrier[0] = 1u;\n");
    try out.appendSlice(a, stock[at..]);
    return out.toOwnedSliceSentinel(a, 0);
}
