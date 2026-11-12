//! turboexl3_ling's exact-verify MLA rows (VERIFY_MLA_ROWS, S4): under the exact verify (`ling_exl3.exact_rows`), a
//! verify's MLA attention rows over the expanded key (the copied plugins/bailing_hybrid/mla.zig `verifyAttn`) run in one
//! launch per kind of row (plugins/bailing_hybrid/verify_rows.zig: MLX's vector-SDPA statements for each row over its
//! own keys) instead of one MLX SDPA call per row (`rowAttn`), each row's words its one-row call's. Byte-identical
//! (src/ling_verify_mla_test.zig holds it to `rowAttn`). The first verify that takes it logs one line, and one whose
//! shapes it does not serve logs once and runs `rowAttn`.
const std = @import("std");

var engaged_said = false;
var declined_said = false;

/// The first verify the batched rows took: its rows and its groups (kinds of launch)
pub fn engaged(rows: c_int, groups: u32, two_pass: bool) void {
    if (engaged_said) return;
    engaged_said = true;
    std.log.info("[turboexl3_ling] exact verify MLA rows: {d} rows in {d} group(s) of MLX's {s} vector kernels, each row its one-row call's words", .{ rows, groups, if (two_pass) "two-pass" else "one-pass" });
}

/// A verify the batched rows did not serve (its shapes or dtypes): rowAttn runs it
pub fn declined(rows: c_int) void {
    if (declined_said) return;
    declined_said = true;
    std.log.info("[turboexl3_ling] exact verify MLA rows: a verify of {d} rows the batched rows do not serve; rowAttn runs it", .{rows});
}
