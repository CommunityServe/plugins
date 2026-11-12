//! EMBED_ROWS (S51, built in, with no option): the trunk keeps no resident embedding table, and its embedding (the
//! copied trunk.zig `Model.embedding`, through ling-trunk.patch) reads a forward's rows from the pack's own tensor
//! (src/ling_embed_rows.zig), as stored: the words the resident table's `take` gave. Ling-3.1-flash's table is
//! 157,184 x 4,096 bf16, 1,287,651,328 bytes (1,228 MiB), which the bank's slot rows take: plugin.zig `openBank` reads
//! MLX's memory in use after the load, and the table is not in it.
//! - The load (plugin.zig `load`) opens the table's shard before the trunk loads (`open`). The trunk then keeps the
//!   shards' tensor unevaluated (ling-trunk.patch `evalTrunk`, `keepsLazy`), so MLX never reads it. That tensor is what
//!   the trunk lends the engine's drafters (`SpecHooks.trunk_weight(.embed)`, while the MTP head is served). Without a
//!   DFlash drafter, mlx-serve 66aad949 reads it there by pointer, dtype and shape only
//!   (src/engines/mlx/plugin/driver.zig `adoptTrunkWeights`, src/sdk/quant/ternary.zig `firstRowExactMiss`,
//!   src/engines/mlx/spec/mtp_nax_profile.zig), and the MTP head embeds through the trunk (src/ling_mtp.zig `step`). A
//!   DFlash drafter would take rows from it (src/engines/mlx/spec/dflash.zig, `rawEmbedding`), so the parse clears
//!   `Caps.dflash_capture` (config.zig `served`). mlx-serve then declines any drafter at load, names it in the log
//!   and runs without it. `checkLent` logs it by name if it is ever evaluated.
//! - The ids are read on the host. A prompt's are the engine's host tokens. A decode-phase forward's (a verify's drafts,
//!   an MTP head step's previous draft) are still being computed on the GPU, so the read waits for them first: the cost
//!   a resident table's gather on the GPU does not have.
//! - The rows: each distinct id read once, the reads in parallel, the repeats copied (src/ling_embed_rows.zig), into a
//!   host buffer, then one MLX array of bf16 [B, L, cols] made from it.
const std = @import("std");
const mlx = @import("mlx_serve").mlx;
const er = @import("ling_embed_rows.zig");
const ran = @import("ling_ran.zig");
const log = @import("mlx_serve/sdk/util/log.zig");

const gpa = std.heap.c_allocator;

/// The table's shard while a load serves (`open` to `close`).
var served: ?er.Source = null;
/// Whether `checkLent`'s line has printed this load.
var lent_said = false;

/// The table `key` of the pack in `dir` (`vocab` x `cols` bf16) opened for row reads, before the trunk loads; every
/// refusal named (error.LingEmbedRows*).
pub fn open(dir: []const u8, key: []const u8, vocab: u32, cols: u32) !void {
    close();
    served = try er.Source.open(gpa, dir, key, vocab, cols);
}

/// The shard closed (the model's unload, or a load that failed after `open`).
pub fn close() void {
    if (served) |*p| p.close(gpa);
    served = null;
    lent_said = false;
}

/// The open table (null before a load's `open` and after `close`), and its shard's name ("" then).
pub fn servedTable() ?er.Table {
    return if (served) |p| p.table else null;
}
pub fn servedShard() []const u8 {
    return if (served) |p| p.shard else "";
}

/// Whether `a` is the trunk's table handle `w`, which the load leaves unevaluated (ling-trunk.patch `evalTrunk`).
pub fn keepsLazy(a: mlx.mlx_array, w: mlx.mlx_array) bool {
    const x = a.ctx orelse return false;
    const y = w.ctx orelse return false;
    return x == y;
}

/// The trunk's embedding of `ids` (`[B, L]`: int32, uint32, or another integer type cast to int32): `[B, L, cols]` bf16
/// read from the pack. With no table open it is refused by name (error.LingEmbedRowsNotOpen): the trunk holds no
/// resident table to take from. Marks EMBED_ROWS on the run-path line.
pub fn rowsOf(s: mlx.mlx_stream, ids: mlx.mlx_array, cols: u32) !mlx.mlx_array {
    const src = if (served) |*p| p else return error.LingEmbedRowsNotOpen;
    if (cols != src.table.cols) return error.LingEmbedRowsShape;
    const sh = mlx.getShape(ids);
    if (sh.len != 2 or sh[0] < 1 or sh[1] < 1) return error.LingEmbedRowsIdsShape;
    const nb: usize = @intCast(sh[0]);
    const nl: usize = @intCast(sh[1]);
    const host_ids = try hostIds(s, ids, nb, nl);
    defer gpa.free(host_ids);
    const n_bytes = host_ids.len * src.table.row_bytes;
    // A prompt's rows (more than 1 MiB) in pages of their own, given back to the system at once.
    const ba: std.mem.Allocator = if (n_bytes > 1 << 20) std.heap.page_allocator else gpa;
    const buf = try ba.alloc(u8, n_bytes);
    defer ba.free(buf);
    _ = try src.read(gpa, host_ids, buf);
    const shape = [_]c_int{ sh[0], sh[1], @intCast(cols) };
    const out = mlx.mlx_array_new_data(buf.ptr, &shape, 3, .bfloat16);
    if (out.ctx == null) return error.LingEmbedRowsArray;
    ran.mark(.EMBED_ROWS);
    return out;
}

/// `ids` `[nb, nl]` on the host, row-major. `ids` is evaluated first: a decode-phase forward's are still being computed
/// on the GPU, so this waits for them. A negative id is refused by name (one past the table: `Source.read`).
fn hostIds(s: mlx.mlx_stream, ids: mlx.mlx_array, nb: usize, nl: usize) ![]u32 {
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    const dt = mlx.mlx_array_dtype(ids);
    if (dt == .int32 or dt == .uint32) {
        try mlx.check(mlx.mlx_array_set(&x, ids));
    } else try mlx.check(mlx.mlx_astype(&x, ids, .int32, s));
    try mlx.check(mlx.mlx_array_eval(x));
    const st = mlx.mlx_array_strides(x);
    const out = try gpa.alloc(u32, nb * nl);
    errdefer gpa.free(out);
    switch (mlx.mlx_array_dtype(x)) {
        .int32 => {
            const p = mlx.mlx_array_data_int32(x) orelse return error.LingEmbedRowsIds;
            for (0..nb) |i| for (0..nl) |j| {
                const v = p[i * st[0] + j * st[1]];
                if (v < 0) return error.LingEmbedRowsTokenPastTable;
                out[i * nl + j] = @intCast(v);
            };
        },
        .uint32 => {
            const p = mlx.mlx_array_data_uint32(x) orelse return error.LingEmbedRowsIds;
            for (0..nb) |i| for (0..nl) |j| {
                out[i * nl + j] = p[i * st[0] + j * st[1]];
            };
        },
        else => return error.LingEmbedRowsIds,
    }
    return out;
}

/// The table the trunk lends (`lent`, its unevaluated tensor: `trunk_weight(.embed)`) must stay unevaluated: evaluated,
/// it is 1,228 MiB resident that the bill does not hold. Logs that once a load, by name.
pub fn checkLent(lent: mlx.mlx_array) void {
    const t = servedTable() orelse return;
    if (lent_said or lent.ctx == null) return;
    var avail = false;
    _ = mlx._mlx_array_is_available(&avail, lent);
    if (!avail) return;
    lent_said = true;
    log.err("[turboexl3_ling] EMBED_ROWS: the embedding table the trunk lends was evaluated: {d} MiB resident that the bill does not hold\n", .{t.bytes() >> 20});
}
