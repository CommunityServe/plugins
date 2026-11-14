//! PF_DFIN's whole-call rule, MLX-free: a wide routed prompt call takes turboexl3's fused finish (the down GEMM's
//! epilogue) only at a mean of at least `dfin_call_min` rows an expert, every other call the served chain, the same
//! bytes either way (src/ling_math.zig `dfinTakes`).
const std = @import("std");

/// PF_DFIN's whole-call rule (src/ling_bank.zig `promptCall`): a wide routed prompt call of `rows` rows (one layer's
/// routed call over whole spans in mlx-serve's whole-prompt pass, or a chunk-loop span) takes turboexl3's fused finish
/// when its mean rows an expert, rows * topk / experts, is at least `call_min`, in integers rows * topk >= call_min *
/// experts; a `call_min` of 0 takes it at every call. Under it the call keeps the served chain. The host decides from
/// `rows` alone (no read-back of the routes), once a call: every rows-path call of the call takes the same side.
pub fn dfinCallTakes(rows: u64, topk: u32, experts: u32, call_min: u32) bool {
    return call_min == 0 or rows * topk >= @as(u64, call_min) * experts;
}

/// PF_DFIN's served cutoff: 80 rows an expert, the first step of turboexl3's TIME ladder (ling-v1-routed, rows an
/// expert) where the plain fused finish was at or under the served chain at both n 40 and 32 (at 8 to 32 rows an expert
/// it was slower). Main kept it after the cutoff records (2026-11-09: CALL_MIN 0 was no faster at 1,120, 2,042 and
/// 4,094 tokens).
pub const dfin_call_min: u16 = 80;

// ── tests ──

const testing = std.testing;

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
