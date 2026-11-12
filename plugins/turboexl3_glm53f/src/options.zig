//! turboexl3_glm53f's switches as `Manifest.options`: every `TURBOEXL3_GLM53F_*` variable the plugin reads at its open,
//! its kind (`bool` "0" or "1", `number` a plain decimal, `text` up to 1 KiB), the plugin's own value when none is set
//! and one line of help. Each runs inside this plugin (the bank, its stream, turboexl3's schedules); the default is the
//! frozen exact configuration's (2026-11-12, greedy text c3c346222307) for the switches it keeps. A switch whose
//! mechanism is a seam of the kit's stream (src/streamed/glm5n/experts.zig `seams`) defaults off on a kit whose stream
//! lacks it, and is refused at open when set there. PF_INT8 (not exact) and DEMAND_HELP (rejected in its A/B) are off
//! unset. `defaultOf` answers an unset variable (`routed_t3.processEnv`, plugin.zig's `envSpan`). MLX-free.
const std = @import("std");
const kit = @import("mlx_serve");
/// The bank's switches whose mechanism is a seam of the kit's stream, and whether this kit's stream has each
/// (src/streamed/glm5n/experts.zig `seams`): without its seam a switch defaults off and is refused when set.
const seams = @import("mlx_stream").glm5n_source.seams;

/// An option's kind, as the kit's `abi.dylib.OptionKind` names it.
pub const Kind = enum { bool, number, text };
/// One switch: its variable, kind, the plugin's own value when none is set and one line of help.
pub const Decl = struct { name: []const u8, kind: Kind, default: []const u8, help: []const u8 };

pub const all = [_]Decl{
    .{ .name = "TURBOEXL3_GLM53F_LAYER_BILL", .kind = .bool, .default = "1", .help = "Bill the slot rows at each routed layer's own row size (the pack's v2 manifest), not the widest" },
    .{ .name = "TURBOEXL3_GLM53F_GROW_UNFILLED", .kind = .bool, .default = "1", .help = "Take the decode grow's new slot rows as MLX buffers without the zero fill" },
    .{ .name = "TURBOEXL3_GLM53F_PROMPT_KEEP", .kind = .bool, .default = if (seams.prompt_keep) "1" else "0", .help = "No prompt-phase admission evicts a resident expert (needs the kit stream's prompt_keep seam)" },
    .{ .name = "TURBOEXL3_GLM53F_SPEC_PICK1", .kind = .bool, .default = "1", .help = "The decode lookahead reads ahead one record a layer call, its most likely (off: 2); the read pool unchanged" },
    .{ .name = "TURBOEXL3_GLM53F_DEMAND_HELP", .kind = .number, .default = "0", .help = "0 off (rejected in its A/B); 2 decode's read pool puts its demand first and a waiting demand read takes its claimed record's untaken pieces (the kit stream's demandFirst(2))" },
    .{ .name = "TURBOEXL3_GLM53F_ADOPT_PIECES", .kind = .bool, .default = if (seams.adopt_pieces) "1" else "0", .help = "A demand read served from a lookahead record copies its bytes in pieces across the read pool's workers (needs the kit stream's adopt_pieces seam)" },
    .{ .name = "TURBOEXL3_GLM53F_SEED_DECODE", .kind = .bool, .default = "1", .help = "Seed the residents from the pack's decode-only usage record" },
    .{ .name = "TURBOEXL3_GLM53F_KEEP_WARM", .kind = .bool, .default = "1", .help = "Keep the GPU warm through the expert stream's waits (one small FMA chain on its own queue)" },
    .{ .name = "TURBOEXL3_GLM53F_PF_ROWS", .kind = .bool, .default = "1", .help = "A prompt call and a cold-row chunk run turboexl3's moeClampedRows" },
    .{ .name = "TURBOEXL3_GLM53F_DEC_BF16", .kind = .bool, .default = "1", .help = "A decode wave's banked down finish stores bf16" },
    .{ .name = "TURBOEXL3_GLM53F_PF_FEED", .kind = .bool, .default = "1", .help = "A prompt call returns with its last chunks in flight; one wait covers its group" },
    .{ .name = "TURBOEXL3_GLM53F_PF_ALIGN", .kind = .bool, .default = "1", .help = "A prompt call's chunks end where no expert's 128-row run is split" },
    .{ .name = "TURBOEXL3_GLM53F_BANKED", .kind = .text, .default = "g4m3.t4.t4", .help = "The banked decode's schedule: g4m3.t4.t4" },
    .{ .name = "TURBOEXL3_GLM53F_DEC", .kind = .text, .default = "g1", .help = "The dense decode GEMV: g1" },
    .{ .name = "TURBOEXL3_GLM53F_PF_OPTS", .kind = .text, .default = "bk128", .help = "The routed prefill's options: bk128" },
    .{ .name = "TURBOEXL3_GLM53F_PG2", .kind = .text, .default = "50115151111", .help = "The trunk's prefill tile codes on turboexl3's pg2, one a linear (11)" },
    .{ .name = "TURBOEXL3_GLM53F_PF_STAGE", .kind = .bool, .default = "1", .help = "A prompt call routes its layer's misses before its routing barrier" },
    .{ .name = "TURBOEXL3_GLM53F_PF_DRAIN_LAG", .kind = .bool, .default = "1", .help = "With PF_FEED: a prompt MoE group's drain waits for the next group's first chunk" },
    .{ .name = "TURBOEXL3_GLM53F_PF_INT8", .kind = .number, .default = "0", .help = "Not exact: 0 off; 1 the routed prefill's gate|up and down GEMMs on int8, a scale each activation 128-block; 2 one scale a row; 3 1 and the trunk's KDA qkv and o_proj at prefill rows on int8; 4 1 and every trunk linear at prefill rows on int8" },
    // the verify's launch trims (src/kda_trim.zig, src/cached_scalars.zig, src/host_zero.zig)
    // the MTP head's router in ROUTER's plan (src/fused_router.zig)
    // the MTP draft's step 0 on SMALLOP's texts (src/draft_s0.zig)
    // a verify linear's producer writing its prepared input, DEC-ONE's GEMV alone after it (src/dec_prep.zig)
};

/// The plugin's own value of switch `name` when the environment sets none (`all`'s default), or null for a name it does
/// not declare (the device tests' gates).
pub fn defaultOf(name: []const u8) ?[]const u8 {
    for (all) |d| if (std.mem.eql(u8, d.name, name)) return d.default;
    return null;
}

/// `all` as options of the kit's type `O` (`sdk.Plugin.options`' element: name, kind, default, help).
pub fn forKit(comptime O: type) [all.len]O {
    var out: [all.len]O = undefined;
    for (all, &out) |d, *o| o.* = .{ .name = d.name, .kind = switch (d.kind) {
        .bool => .bool,
        .number => .number,
        .text => .text,
    }, .default = d.default, .help = d.help };
    return out;
}

test "turboexl3_glm53f stream options: every switch declared once with a line of help, each name the kit's option name rule and each default a value of its kind, where the kit has the rules" {
    const t = std.testing;
    for (all, 0..) |d, i| {
        try t.expect(std.mem.startsWith(u8, d.name, "TURBOEXL3_GLM53F_") and d.name.len <= 128 and d.help.len > 0);
        for (all[0..i]) |e| try t.expect(!std.mem.eql(u8, d.name, e.name));
    }
    // mlx-serve #1131's own checks (`abi.dylib.optionNameOk`, `valueOk`; a 26.11.2 kit has neither)
    if (comptime @hasDecl(kit.abi, "dylib") and @hasDecl(kit.abi.dylib, "optionNameOk") and @hasDecl(kit.abi.dylib, "valueOk")) {
        const dy = kit.abi.dylib;
        for (all) |d| {
            try t.expect(dy.optionNameOk("turboexl3_glm53f", d.name));
            const k: dy.OptionKind = switch (d.kind) {
                .bool => .bool,
                .number => .number,
                .text => .text,
            };
            try t.expect(dy.valueOk(k, d.default));
        }
    }
}

/// The switches whose mechanism is a seam of the kit's stream, by name, and whether this kit's stream has it.
fn seamOf(name: []const u8) ?bool {
    const S = struct { n: []const u8, on: bool };
    const list = [_]S{
        .{ .n = "TURBOEXL3_GLM53F_PROMPT_KEEP", .on = seams.prompt_keep },
        .{ .n = "TURBOEXL3_GLM53F_ADOPT_PIECES", .on = seams.adopt_pieces },
    };
    for (list) |x| if (std.mem.eql(u8, x.n, name)) return x.on;
    return null;
}

/// The switches the served configuration keeps off ("0", or "" for a text switch): PF_INT8 (not exact; a server-start
/// option, 1 to 4), LOGITS_DUMP (a check, never served), and PF_RUN_TRIM and DEMAND_HELP (exact, but their A/Bs
/// rejected them; DEMAND_HELP on the kit stream's `demandFirst(2)` decoded 3.1% slower in g53g2h1).
const off = [_][]const u8{
    "TURBOEXL3_GLM53F_PF_INT8",
    "TURBOEXL3_GLM53F_DEMAND_HELP",
};
fn isOff(name: []const u8) bool {
    for (off) |o| if (std.mem.eql(u8, o, name)) return true;
    return false;
}

test "turboexl3_glm53f stream options: every default is the frozen exact configuration's: each bool on, each number and text switch its accepted value, a switch whose seam the kit's stream lacks off, the served configuration's off ones off, and defaultOf answers it" {
    const t = std.testing;
    for (off) |o| try t.expect(defaultOf(o) != null);
    for (all) |d| {
        if (isOff(d.name)) {
            try t.expectEqualStrings(if (d.kind == .text) "" else "0", d.default);
            try t.expectEqualStrings(d.default, defaultOf(d.name).?);
            continue;
        }
        if (seamOf(d.name)) |on| if (!on) {
            try t.expectEqualStrings("0", d.default);
            try t.expectEqualStrings(d.default, defaultOf(d.name).?);
            continue;
        };
        switch (d.kind) {
            .bool => try t.expectEqualStrings("1", d.default),
            .number, .text => try t.expect(d.default.len > 0 and !std.mem.eql(u8, d.default, "0")),
        }
        try t.expectEqualStrings(d.default, defaultOf(d.name).?);
    }
    try t.expectEqual(@as(usize, 19), all.len);
    try t.expectEqualStrings("g4m3.t4.t4", defaultOf("TURBOEXL3_GLM53F_BANKED").?);
    try t.expectEqualStrings("g1", defaultOf("TURBOEXL3_GLM53F_DEC").?);
    try t.expectEqualStrings("bk128", defaultOf("TURBOEXL3_GLM53F_PF_OPTS").?);
    try t.expectEqualStrings("50115151111", defaultOf("TURBOEXL3_GLM53F_PG2").?);
    try t.expectEqualStrings("0", defaultOf("TURBOEXL3_GLM53F_DEMAND_HELP").?);
    try t.expectEqualStrings("0", defaultOf("TURBOEXL3_GLM53F_PF_INT8").?);
    // the model's levers are mlx-serve's glm5_next's to carry, not this plugin's
    try t.expect(defaultOf("TURBOEXL3_GLM53F_DSA_HOLD") == null);
    try t.expect(defaultOf("TURBOEXL3_GLM53F_ADOPT_ZC") == null);
    try t.expect(defaultOf("TURBOEXL3_GLM53F_KDA_GPU") == null);
}
