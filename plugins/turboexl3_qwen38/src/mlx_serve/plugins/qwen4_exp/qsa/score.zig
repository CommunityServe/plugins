//! copied_from: src/plugins/qwen4_exp/qsa/score.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! qwen4_exp's QSA score sheet upkeep: the block dump (`QWEN4_DUMP_QSA_BLOCKS`), and the bank, bill and fallback tests
//! of the trunk that feeds the SDK's score kernel.

const std = @import("std");
const env = @import("../../../env.zig");
const crt = @import("../../../sdk/util/crt.zig");
const mlx = @import("../../../mlx/ffi.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const history = @import("history.zig");
const score = @import("../../../sdk/blocks/attention/sparse_block/score.zig");
const select = @import("../../../sdk/blocks/attention/sparse_block/select.zig");
const Blocks = @import("../../../sdk/blocks/attention/sparse_block/params.zig").Blocks;
const testing = std.testing;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Transformer = @import("../trunk.zig").Trunk;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const verify_qmm_nax_vars = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig");
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT = score.QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
const qsaPooledRowContiguous = score.qsaPooledRowContiguous;
const qsaPrefillTransientBytes = score.qsaPrefillTransientBytes;
const qsaScoreFused = score.qsaScoreFused;
const qsaScoreFusedActive = score.qsaScoreFusedActive;
const qsaScoreFusedActiveFor = score.qsaScoreFusedActiveFor;
const qsaScoreFusedArm = score.qsaScoreFusedArm;
const qsaScoreFusedEligibleFrom = score.qsaScoreFusedEligibleFrom;
const qsaScoreRowsPerChunk = score.qsaScoreRowsPerChunk;
const qsaScoreSheet = score.qsaScoreSheet;
const qsaScoreSheetComposed = score.qsaScoreSheetComposed;
const qsaSelectTopBlocks = select.qsaSelectTopBlocks;

/// Test seam for the dump path; null reads `QWEN4_DUMP_QSA_BLOCKS`.
pub var qsa_dump_blocks_path_override: ?[]const u8 = null;
var qsa_dump_blocks_cached: ?[]const u8 = null;
var qsa_dump_blocks_read: bool = false;

fn qsaDumpBlocksPath() ?[]const u8 {
    if (qsa_dump_blocks_path_override) |p| {
        if (p.len == 0 or p[0] == '0') return null;
        return p;
    }
    if (qsa_dump_blocks_read) return qsa_dump_blocks_cached;
    qsa_dump_blocks_read = true;
    const raw = env.get(.QWEN4_DUMP_QSA_BLOCKS) orelse return null;
    if (!env.onValue(raw)) return null;
    const s = std.mem.sliceTo(raw, 0);
    qsa_dump_blocks_cached = s;
    return s;
}

pub fn qsaDumpQsaBlocks(layer: u32, row0: c_int, rows: c_int, kb: c_int, blocks: mlx.mlx_array) void {
    const path = qsaDumpBlocksPath() orelse return;
    if (blocks.ctx == null) return;
    if (mlx.mlx_array_eval(blocks) != 0) return;
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    const s = mlx.gpuStream();
    if (mlx.mlx_contiguous(&contig, blocks, false, s) != 0) return;
    if (mlx.mlx_array_eval(contig) != 0) return;
    const data = mlx.mlx_array_data_int32(contig) orelse return;
    const n = mlx.mlx_array_size(contig);
    var path_z: [1024]u8 = undefined;
    if (path.len == 0 or path.len >= path_z.len) return;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;
    const fd = crt.openWrite(@ptrCast(&path_z), false);
    if (fd < 0) return;
    defer crt.close(fd);
    var hdr: [128]u8 = undefined;
    const header = std.fmt.bufPrint(&hdr, "layer={d} row0={d} rows={d} kb={d}\n", .{ layer, row0, rows, kb }) catch return;
    _ = crt.write(fd, header);
    var buf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const piece = if (i == 0)
            std.fmt.bufPrint(&buf, "{d}", .{data[i]}) catch return
        else
            std.fmt.bufPrint(&buf, " {d}", .{data[i]}) catch return;
        _ = crt.write(fd, piece);
    }
    _ = crt.write(fd, "\n");
}

/// The block geometry of the dump test.
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };
