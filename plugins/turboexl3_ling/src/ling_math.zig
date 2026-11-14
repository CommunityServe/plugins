//! Ling-3.1-flash's routed math settings on turboexl3, for the bank (src/ling_bank.zig): turboexl3's configuration as
//! this plugin serves it (`configure`), PF_DFIN's whole-call choice over the model's routing (`dfinTakes`), and the
//! prompt calls' padding (`gemm_min_rows`: a call of 16 pairs or fewer is padded to 17 so every prompt pair takes
//! turboexl3's prefill GEMM, whose words do not depend on how the bank's residency split the call).

const t3 = @import("turboexl3");
const switches = @import("ling_switches.zig");

pub const Routing = struct { topk: u32 = 0, experts: u32 = 0 };
pub var routing: Routing = .{};

/// PF_DFIN's whole-call choice for a call of `rows` rows: whether it takes the fused finish (src/ling_switches.zig
/// `dfinCallTakes` at `switches.dfin_call_min` rows an expert, over the model's routing). Before `routing` is set (no
/// routed call comes then) the rule's integers are 0 and the call takes the fused finish, the ending turboexl3's
/// `Config.pf_dfin` gives a call that names none.
pub fn dfinTakes(rows: usize) bool {
    return switches.dfinCallTakes(@intCast(rows), routing.topk, routing.experts, switches.dfin_call_min);
}

/// The fewest pairs a prompt call takes to `moeLing`: turboexl3 runs 16 rows or fewer on its decode kernels and more on
/// its prefill GEMM, which agree within tolerance, not byte for byte; a smaller call is padded to this many.
pub const gemm_min_rows = 17;

comptime {
    if (gemm_min_rows != t3.kernels.DECODE_ROWS_MAX + 1) @compileError("gemm_min_rows must be turboexl3's DECODE_ROWS_MAX + 1");
}

/// At load, before any turboexl3 launch is built (plugin.zig `open`, and again at the bank's open, src/ling_bank.zig):
/// turboexl3's configuration for the calls this plugin makes (the trunk's linears, the routed experts), each field the
/// served stack sets; every other field keeps turboexl3's default.
pub fn configure() error{InvalidTurboexl3Config}!void {
    var c = t3.config.get().*;
    // The banked decode on turboexl3's grouped texts, as turboexl3_ds41 serves it: up to four row-slots of one expert a
    // weight decode, with the M = 3 member path.
    c.banked_group = 4;
    c.banked_m3 = true;
    // ONE_ROW_TABLE: a one-row trunk linear on its shape's entry of turboexl3's `dense.one_row_table` (the M5 Max's
    // fastest one-row launch for each Ling-3.1-flash trunk shape: output tiles a threadgroup, and DEC-ONE's prepare inside
    // the launch or in its own); the shapes it does not name keep the rest. Launch shape only: the same bytes.
    c.dec_one_row_table = true;
    // BANKED_T4: the grouped banked GEMVs (gate|up and down) on variant t4 of turboexl3's `banked.group_variants`, another
    // schedule of the same loads, decodes and sums, so the same bytes. It runs on the grouped texts, `banked_group` 4
    // above. turboexl3 prints `[turboexl3] banked variant engaged: ...` at the first routed call of each GEMV on it.
    const t4 = t3.banked.groupVariantOf("t4") orelse return error.InvalidTurboexl3Config;
    c.banked_variant_gu = t4;
    c.banked_variant_dn = t4;
    // PF_DFIN: the routed prompt chain's finish as the down GEMM's epilogue, one launch writing y, at every call that
    // takes it (`dfinTakes`); every other call keeps the served chain.
    c.pf_dfin = true;
    try t3.configure(c);
}
