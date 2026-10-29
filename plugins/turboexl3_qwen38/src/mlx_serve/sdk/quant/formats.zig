//! copied_from: src/sdk/quant/formats.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The built-in quant formats' tables, in `block.builtins` order: the index `block.claimant` returns names the table
//! here. Discovery reads `block.zig` (MLX-free); the loader reads this list.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const common = @import("mlx_serve").abi.common;
const block = @import("block.zig");
const native = @import("native.zig");
const hadamard = @import("hadamard/format.zig");
const exl3 = @import("exl3/format.zig");
const gguf = @import("gguf/format.zig");

pub const tables = [_]*const abi.QuantTable{ &native.affine, &native.fp8, &native.mxfp8, &native.nvfp4, &hadamard.table, &exl3.table, &gguf.table };

/// The table serving `block_json` of `config_json`; null when no built-in claims it.
pub fn claimant(block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    const i = try block.claimant(block_json, config_json) orelse return null;
    return tables[i];
}

/// `claimant` with a model's own tables `extra` weighed against the built-ins (qwen4_exp's trunk formats: turboexl3's
/// table, linked in): an extra table named as a built-in replaces it; otherwise one claiming above every built-in wins, a
/// built-in wins a tie, and extra tables tied at the top leave the block to the built-ins.
pub fn claimantAmong(extra: []const *const abi.QuantTable, block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    var got: [tables.len]abi.Claim = undefined;
    var top: u8 = 0;
    for (block.builtins, &got) |b, *c| {
        c.* = if (replaced(b.name, extra)) .no else b.claims(.of(block_json), .of(config_json));
        top = @max(top, @intFromEnum(c.*));
    }
    if (extra.len > 0 and extra.len <= max_extra) {
        var ranks: [max_extra]abi.Claim = undefined;
        for (extra, ranks[0..extra.len]) |t, *c| c.* = t.claims(.of(block_json), .of(config_json));
        if (common.pick(ranks[0..extra.len]) catch null) |i| {
            if (@intFromEnum(ranks[i]) > top) return extra[i];
        }
    }
    const i = try common.pick(&got) orelse return null;
    return tables[i];
}

/// The extra tables weighed at most.
const max_extra = 64;

fn replaced(name: []const u8, extra: []const *const abi.QuantTable) bool {
    for (extra) |t| if (std.mem.eql(u8, t.name.slice(), name)) return true;
    return false;
}

// ── tests ──

const testing = std.testing;
const sdk = @import("mlx_serve").sdk;
const conformance = @import("../testing/conformance.zig");
const fixture = @import("mlx_serve").sdk.fixture;

fn claimsAtMarker(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
    return if (std.mem.indexOf(u8, block_json.slice(), "\"exl3\"") != null) .by_marker else .no;
}

fn claimsAtType(block_json: abi.Str, _: abi.Str) callconv(.c) abi.Claim {
    return if (std.mem.indexOf(u8, block_json.slice(), "\"exl3\"") != null) .by_type else .no;
}

/// An extra table as the claimant reads it: its name and its claim.
fn extraTable(comptime name: []const u8, claims: block.ClaimsFn) abi.QuantTable {
    return .{ .name = abi.Str.of(name), .claims = claims, .validate = undefined, .bill = undefined, .open = undefined, .close = undefined, .load_tensor = undefined, .load_bank = undefined, .release = undefined, .matmul = undefined, .moe = undefined };
}

/// A built-in format as the conformance suite reads a quant plugin.
fn Builtin(comptime i: usize, comptime block_json: []const u8, comptime config_json: []const u8) type {
    return struct {
        pub const plugin = sdk.Plugin{ .name = block.builtins[i].name, .kind = .quant };
        pub const table = tables[i].*;
        pub fn conformanceFixture(_: std.mem.Allocator) !fixture.Fixture {
            return .{ .files = &.{.{ .name = "config.json", .bytes = config_json }}, .block = block_json };
        }
    };
}

test {
    _ = block;
    _ = native;
    _ = hadamard;
    _ = exl3;
    _ = gguf;
    _ = @import("../weights/resident.zig");
    _ = @import("../weights/streamed.zig");
    _ = @import("fp8_block128.zig");
}
