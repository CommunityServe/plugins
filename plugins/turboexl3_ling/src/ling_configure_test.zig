//! turboexl3_ling's served turboexl3 Config, pinned field by field (the strip review's finding 3 and its follow-up a):
//! src/ling_math.zig `configure` writes literals, and the plugin relies on turboexl3's defaults for every other field.
//! Before this test no check read either back. A field dropped from `configure` would leave its served path off with the
//! same words (for `mla_rope_once`, every wide trunk prompt call would rebuild the expanded key, slower, with a transient
//! the bill does not hold), and a default turboexl3 changed would change the served launches unseen (for `pf_t128`,
//! ling-strip2 11/11 removed the path its off side took). The test runs `configure`, reads turboexl3's Config back,
//! requires each written field and each relied-on default at its served value, then changes each field alone and
//! requires the check to name it (the red check). It needs no GPU, but it links turboexl3, so it lives in the GPU test
//! binary and runs, as the others do, only with MLX_SERVE_LING_GPU_TESTS=1.
const std = @import("std");
const t3 = @import("turboexl3");
const ling_math = @import("ling_math.zig");

/// The fields `configure` writes, in its order
const written = [_][]const u8{ "banked_group", "banked_m3", "dec_one_row_table", "banked_variant_gu", "banked_variant_dn", "mla_latent_stage", "mla_latent_nax", "mla_rope_once", "pf_dfin", "router_rows" };
/// The fields the plugin leaves at turboexl3's defaults and relies on, in Config's order: the served launches
/// (decode_form 9, shared_spt 4, fused_fin_fold, pf_alay 2, pf_t128, dec_one, dense_lean, dense_lean5), the routed
/// prefill on NAX (prefill, nax), and the diagnostics off (slot_stats, pf_capture_dir)
const relied = [_][]const u8{ "decode_form", "shared_spt", "fused_fin_fold", "pf_alay", "pf_t128", "prefill", "nax", "dec_one", "dense_lean", "dense_lean5", "slot_stats", "pf_capture_dir" };

/// The first field of `c` that is not the served value, else null: first the written fields (banked decode in groups of
/// 4 with the M = 3 path, the one-row table, banked variant t4 for gate|up and down, the MTP head's latent attention
/// staged and on NAX, rope-once, PF_DFIN and the router's verify rows), then the relied-on defaults.
fn servedMismatch(c: t3.config.Config) ?[]const u8 {
    const t4 = t3.banked.groupVariantOf("t4") orelse return "banked_variant (no t4 in this turboexl3)";
    if (c.banked_group != 4) return "banked_group";
    if (!c.banked_m3) return "banked_m3";
    if (!c.dec_one_row_table) return "dec_one_row_table";
    if (c.banked_variant_gu != t4) return "banked_variant_gu";
    if (c.banked_variant_dn != t4) return "banked_variant_dn";
    if (!c.mla_latent_stage) return "mla_latent_stage";
    if (!c.mla_latent_nax) return "mla_latent_nax";
    if (!c.mla_rope_once) return "mla_rope_once";
    if (!c.pf_dfin) return "pf_dfin";
    if (!c.router_rows) return "router_rows";
    if (c.decode_form != 9) return "decode_form";
    if (c.shared_spt != 4) return "shared_spt";
    if (!c.fused_fin_fold) return "fused_fin_fold";
    if (c.pf_alay != 2) return "pf_alay";
    if (!c.pf_t128) return "pf_t128";
    if (!c.prefill) return "prefill";
    if (!c.nax) return "nax";
    if (!c.dec_one) return "dec_one";
    if (!c.dense_lean) return "dense_lean";
    if (!c.dense_lean5) return "dense_lean5";
    if (c.slot_stats) return "slot_stats";
    if (c.pf_capture_dir != null) return "pf_capture_dir";
    return null;
}

test "turboexl3_ling configure: every field it writes and every default it relies on reads back at the served value, and each one changed alone is named" {
    const gate = std.c.getenv("MLX_SERVE_LING_GPU_TESTS") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, std.mem.span(gate), "1")) return error.SkipZigTest;
    const saved = t3.config.get().*;
    defer t3.configure(saved) catch {};
    try ling_math.configure();
    const c = t3.config.get().*;
    var failed: usize = 0;
    if (servedMismatch(c)) |f| {
        std.debug.print("TURBOEXL3_LING_CONFIGURE {s} is not its served value FAIL\n", .{f});
        failed += 1;
    } else std.debug.print("TURBOEXL3_LING_CONFIGURE all {d} written fields and {d} relied-on defaults at their served values PASS\n", .{ written.len, relied.len });
    // the red check: each pinned field changed alone must be the one the check names
    inline for (written ++ relied) |name| {
        var bad = c;
        switch (@typeInfo(@TypeOf(@field(bad, name)))) {
            .bool => @field(bad, name) = !@field(bad, name),
            .optional => @field(bad, name) = "/tmp/turboexl3_ling-configure-red",
            else => @field(bad, name) +%= 1,
        }
        const caught = if (servedMismatch(bad)) |f| std.mem.eql(u8, f, name) else false;
        std.debug.print("TURBOEXL3_LING_CONFIGURE_RED {s} flipped: {s}\n", .{ name, if (caught) "named PASS" else "not named, MISSED" });
        failed += @intFromBool(!caught);
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}
