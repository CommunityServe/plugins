//! turboexl3_ling's served constants that were switches. Since the strip (2026-11-09) every switch the plugin served
//! is built in: the stack S1-S48 (CANDIDATES.md) runs unconditionally, and EMBED_ROWS (S51, src/ling_embed.zig) has no
//! option either. Since the port to mlx-serve 26.11.3's kit, its span fields and last-row flag are too (config.zig
//! stops a build on a kit without them), so the plugin always declares the whole-prompt prefill and the last-row
//! logits. The one variable the plugin reads is LAYER_MAJOR's off value, its one Manifest option, for the parity
//! instrument only (`layer_major_option`); no served configuration sets it (the tests read their own `MLX_SERVE_LING_*`
//! variables: GPU_TESTS, PACK, RESTORE_*). What is left here:
//! - LAYER_MAJOR (`layerMajor`): the single-pass prompt over the host's spans, unless the parity instrument turns it
//!   off; MEM_REFIT, LM_DECODE_ROWS, PROMPT_AHEAD and the W_hat sharing ride on it.
//! - PF_DFIN's whole-call rule at its served cutoff and its counts (the `PF_DFIN calls:` line plugin.zig prints).
//! - The SSD root's layout namespace of the served MLA cache.
const std = @import("std");
const sdk = @import("mlx_serve").sdk;

/// LAYER_MAJOR: the prompt runs layer by layer over the host's spans, unless the parity instrument turns the pass off.
/// The first call reads the off value (`layerMajorOff`); config.zig's parse calls it first.
pub fn layerMajor() bool {
    return !layerMajorOff();
}

/// LAYER_MAJOR's off switch, for the decode-width parity instrument only (wave2-parity-plan section 8: every prompt
/// position a one-row forward of the host's chunk loop at `--prefill-chunk 1`, for a text-changing candidate's bar): the
/// plugin's one Manifest option (plugin.zig `plugin.options`), which mlx-serve draws in Models > Plugins, sets in the
/// environment before the plugin loads (over the shell's, where plugins.json saves a value) and reports in /props. At 0
/// the layer-major pass is off, so the plugin declares no whole-prompt prefill and the host runs its chunk loop. Unset,
/// at its default 1, or any other value: the pass runs, as served. No served configuration sets it.
pub const layer_major_option: sdk.plugin.Option = .{
    .name = layer_major_var,
    .kind = .bool,
    .default = "1",
    .help = "Parity instrument only: 0 turns the layer-major prompt pass off, so mlx-serve's chunk loop runs every prompt. Leave it at 1.",
};

/// The option's variable (a C string for getenv).
const layer_major_var = "TURBOEXL3_LING_LAYER_MAJOR";

/// The option's value, read once, at the first call.
var layer_major_off: ?bool = null;

fn layerMajorOff() bool {
    if (layer_major_off) |cached| return cached;
    const raw: ?[]const u8 = if (std.c.getenv(layer_major_var)) |p| std.mem.span(p) else null;
    const off = offValue(raw);
    layer_major_off = off;
    if (off) std.log.info("[turboexl3_ling] LAYER_MAJOR off (" ++ layer_major_var ++ "=0, the parity instrument only): the host's chunk loop runs every prompt", .{});
    return off;
}

/// The variable's value rule: only "0" turns the pass off.
fn offValue(raw: ?[]const u8) bool {
    const v = raw orelse return false;
    return std.mem.eql(u8, v, "0");
}

/// Tests only: LAYER_MAJOR's off value (null: read the environment again at the next call), for a test of what the
/// parse serves at either value (config.zig).
pub fn setLayerMajorOffForTests(off: ?bool) void {
    layer_major_off = off;
}

/// The SSD root's layout namespace, which config.zig `served` gives the host (and the restore check's part A records
/// for part B): the served MLA cache, the rope-once layout (src/ling_mla_rope1.zig: each MLA layer caches the roped key
/// once a token and kv_b's rows) with the MTP head's latent cache (src/ling_mtp_latent.zig: the head's layer caches its
/// normed latent and roped shared key). An entry persisted under another layout never restores into this one: the disk
/// tier checks the quant config, not the head shapes, so the layout has its own root.
pub fn layoutNamespace() [:0]const u8 {
    return "ling-mla-rope1-mtp-head-latent-v1";
}

/// PF_DFIN's whole-call rule (src/ling_math.zig `chunks()`): a wide routed prompt call of `rows` rows (one layer's routed
/// call over whole spans under LAYER_MAJOR, or a chunk-loop span) takes turboexl3's fused finish when its mean rows an
/// expert, rows * topk / experts, is at least `call_min`, in integers rows * topk >= call_min * experts; a `call_min` of 0
/// takes it at every call. Under it the call keeps the served chain. The host decides from `rows` alone (no read-back of
/// the routes), once a call: every chunk of the call takes the same side.
pub fn dfinCallTakes(rows: u64, topk: u32, experts: u32, call_min: u32) bool {
    return call_min == 0 or rows * topk >= @as(u64, call_min) * experts;
}

/// PF_DFIN's served cutoff: 80 rows an expert, the first step of turboexl3's TIME ladder (ling-v1-routed, rows an
/// expert) where the plain fused finish was at or under the served chain at both n 40 and 32 (at 8 to 32 rows an expert
/// it was slower). Main kept it after the cutoff records (2026-11-09: CALL_MIN 0 was no faster at 1,120, 2,042 and
/// 4,094 tokens).
pub const dfin_call_min: u16 = 80;

/// The two sides of a routed call PF_DFIN chooses for (src/ling_math.zig): a wide prompt call (`chunks()`, each of its
/// chunks on the side its rows R take) or a decode-width call (`down()`: a cold-row chunk or a decode wave's group, at
/// most 48 pairs, each by the same rule from its own rows)
pub const DfinSide = enum { wide, decode_width };

/// Where the counts line is printed (plugin.zig): after a prompt forward (its rows), or at a decode step (every 64
/// decode steps, the `[stream-stats]` cadence, and decode's last when the next prompt begins)
pub const DfinAt = union(enum) { prompt: u32, decode: u32 };

/// PF_DFIN's counts: the turboexl3 routed calls it chose an ending for, by side and ending (0 the served chain, 1 the
/// fused finish). src/ling_math.zig counts each call as it is issued. Host counters only: no launch, no byte changes.
pub const DfinCalls = struct {
    calls: [2][2]u64 = .{ .{ 0, 0 }, .{ 0, 0 } },

    /// A call on `side` at PF_DFIN's choice `fused` (the fused finish, or the served chain)
    pub fn count(c: *DfinCalls, side: DfinSide, fused: bool) void {
        c.calls[@intFromEnum(side)][@intFromBool(fused)] += 1;
    }

    /// The counts line into `buf`: `[turboexl3_ling] PF_DFIN calls:`, where (`prompt rows=R` or `decode steps=N`), the
    /// calls since load by side and ending (`wide_fused`, `wide_served`, `decode_width_fused`, `decode_width_served`),
    /// then the rule both sides took at `call_min` rows an expert
    pub fn line(c: *const DfinCalls, buf: []u8, at: DfinAt, call_min: u16) []const u8 {
        const w = c.calls[@intFromEnum(DfinSide.wide)];
        const d = c.calls[@intFromEnum(DfinSide.decode_width)];
        var where_buf: [32]u8 = undefined;
        const where: []const u8 = switch (at) {
            .prompt => |r| std.fmt.bufPrint(&where_buf, "prompt rows={d}", .{r}) catch "prompt",
            .decode => |n| std.fmt.bufPrint(&where_buf, "decode steps={d}", .{n}) catch "decode",
        };
        return std.fmt.bufPrint(buf, "[turboexl3_ling] PF_DFIN calls: {s} wide_fused={d} wide_served={d} decode_width_fused={d} decode_width_served={d} (turboexl3 routed calls since load; each side by the whole-call rule from its call's rows, the fused finish at {d} rows an expert and up)", .{ where, w[1], w[0], d[1], d[0], call_min }) catch "[turboexl3_ling] PF_DFIN calls: the line does not fit its buffer";
    }
};

// ── tests ──

const testing = std.testing;

test "LAYER_MAJOR's parity-instrument option: only 0 turns the pass off; unset or at the option's default, the pass runs" {
    try testing.expect(!offValue(null));
    try testing.expect(!offValue(layer_major_option.default));
    try testing.expect(offValue("0"));
    for ([_][]const u8{ "1", "", "false", "00", "off" }) |v| try testing.expect(!offValue(v));
    const saved_off = layer_major_off;
    defer layer_major_off = saved_off;
    layer_major_off = true;
    try testing.expect(!layerMajor());
    layer_major_off = false;
    try testing.expect(layerMajor());
}

test "RQ is built in: no Manifest option, environment read or test hook is left for it here" {
    inline for (.{ "rq_option", "rq_var", "rq_on", "rq", "onAtOne", "setRqForTests" }) |name| {
        try testing.expect(!@hasDecl(@This(), name));
    }
}

test "KDA_NORM_PREP is built in: no Manifest option, environment read or test hook is left for it here" {
    inline for (.{ "kda_norm_prep_option", "kda_norm_prep_var", "kda_norm_prep_on", "kdaNormPrep", "oneTurnsOn", "setKdaNormPrepForTests" }) |name| {
        try testing.expect(!@hasDecl(@This(), name));
    }
}

test "KDA_GATE_EPI is built in: no Manifest option, environment read or test hook is left for it here" {
    inline for (.{ "kda_gate_epi_option", "kda_gate_epi_var", "kda_gate_epi_on", "kdaGateEpi", "onValue", "setKdaGateEpiForTests" }) |name| {
        try testing.expect(!@hasDecl(@This(), name));
    }
}

test "EMBED_ROWS is built in: no Manifest option, environment read or test hook is left for it here" {
    inline for (.{ "embed_rows_option", "embed_rows_var", "embed_rows_on", "embedRows", "isOne", "setEmbedRowsForTests" }) |name| {
        try testing.expect(!@hasDecl(@This(), name));
    }
}

test "the served layout's namespace is the rope-once cache's with the MTP head's latent" {
    try testing.expectEqualStrings("ling-mla-rope1-mtp-head-latent-v1", layoutNamespace());
}

test "PF_DFIN's whole-call rule at Ling's top-8 over 512 experts: 5,119 rows keeps the served chain, 5,120 and 16,384 take the fused finish, 1,120 keeps the served chain; call_min 0 takes it at every call" {
    const n = dfin_call_min;
    try testing.expectEqual(@as(u16, 80), n);
    // either side of a mean of 80 rows an expert (40,952 and 40,960 routed rows against 80 * 512 = 40,960)
    try testing.expect(!dfinCallTakes(5119, 8, 512, n));
    try testing.expect(dfinCallTakes(5120, 8, 512, n));
    // a layer-major call (16,384 rows: 256 an expert; with its tail, 18,432) and a short call (1,120 rows: 17.5)
    try testing.expect(dfinCallTakes(16384, 8, 512, n));
    try testing.expect(dfinCallTakes(16384 + 2048, 8, 512, n));
    try testing.expect(!dfinCallTakes(1120, 8, 512, n));
    // the rule's own boundary at another top-k and expert count: 80 * 256 / 4 = 5,120 rows
    try testing.expect(!dfinCallTakes(5119, 4, 256, n) and dfinCallTakes(5120, 4, 256, n));
    // 0: every call, whatever its rows
    for ([_]u64{ 1, 17, 1120, 5119, 5120, 16384 }) |r| try testing.expect(dfinCallTakes(r, 8, 512, 0));
}

test "PF_DFIN's counts (DfinCalls): each call counted on its side by its ending; the line the A/B greps, `] PF_DFIN calls:`, gives where, the four counts and the cutoff" {
    var c: DfinCalls = .{};
    c.count(.wide, true);
    c.count(.wide, true);
    c.count(.wide, false);
    c.count(.decode_width, false);
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_ling] PF_DFIN calls: prompt rows=18380 wide_fused=2 wide_served=1 decode_width_fused=0 decode_width_served=1 (turboexl3 routed calls since load; each side by the whole-call rule from its call's rows, the fused finish at 80 rows an expert and up)", c.line(&buf, .{ .prompt = 18380 }, dfin_call_min));
    // the counts keep growing (since load), and a decode line names its step
    c.count(.decode_width, true);
    const later = c.line(&buf, .{ .decode = 64 }, dfin_call_min);
    try testing.expect(std.mem.startsWith(u8, later, "[turboexl3_ling] PF_DFIN calls: decode steps=64 wide_fused=2 wide_served=1 decode_width_fused=1 decode_width_served=1 ("));
    // a buffer too small for the line gives a line that says so, never a part of one
    var small: [16]u8 = undefined;
    try testing.expectEqualStrings("[turboexl3_ling] PF_DFIN calls: the line does not fit its buffer", c.line(&small, .{ .prompt = 1 }, 80));
}
