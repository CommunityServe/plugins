//! `TURBOEXL3_GLM53F_DSA_TAILS=1` (opt-in, exact): a decode or verify forward reads its DSA layers' indexer tail rows
//! before anything else, in a command buffer of their own, so that each layer's indexer cache append reuses the cache's
//! buffer instead of copying it whole. Through patches/turboexl3-model.patch.
//!
//! Why the stock forward copies: a DSA layer's pooled-keys launch (glm5_next `withPooledKeys`) needs the up to
//! kpool - 1 = 3 indexer rows just before the forward, and table_forward `dsaAttn` slices them from the layer's cache
//! view right before it appends the forward's rows. MLX's metal `eval` keeps every input's buffer alive until its command
//! buffer completes (the completion handler holds each input's data), so when the append's SliceUpdate is encoded the
//! cache buffer's data is held twice, `array::is_donatable` is false, and SliceUpdate's `copy_gpu` copies the whole
//! buffer into a new one before it writes the rows: its capacity times 512 bytes, every DSA layer, every verify and
//! every MTP draft step. The latent (K) append reads nothing before it and is donated.
//!
//! Here `begin` (at the start of the trunk forward and of the MTP head's step) copies each DSA layer's tail rows out of
//! its cache view into a buffer of their own (one launch: the trunk's 11 layers, or the head's one; not MLX's `copy`,
//! which shares its input's buffer) and commits them at once (an async eval); the layer's pooled-keys launch then reads
//! its rows from that copy (`take`). Each row is the cache row the stock path slices, so every value is the same and so
//! are the outputs. A copy found inside a cache's buffer would hold it as the stock read does: the forward then declines
//! (its layers read their views as before) and says so once. The cache buffer is donated whenever that small command
//! buffer has completed by the time the layer's append is encoded (the forward's graph is built in between); otherwise
//! MLX copies as before. Nothing is kept across forwards: `end` drops the copy. The trunk's first DSA layer (layer 3) is
//! committed early in a verify (READ_AHEAD3 commits its attention output), so its append may still copy when the GPU is
//! busy with the step before; the later layers are encoded well after the copy has run.
//!
//! Each served cache's buffer is counted at the next served forward's start (`summary`, the request's summary line):
//! the same buffer as at this forward's start, the append reused it (donated); another, it copied it (or the cache
//! grew). Only a forward that continues the last one counts (its first position inside the last one's rows).
//!
//! GLM-5.3-Flash's dims only: indexer rows 256 wide (key | gate, 2 x 128), pools of 4, the trunk's 11 DSA layers or
//! the head's one, one slot, B 1, 1 to 16 rows (a decode, verify or draft window; a prompt keeps the stock path).
const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const log = @import("mlx_serve/sdk/util/log.zig");
const kda_prompt = @import("kda_prompt.zig");
const Array = mlx.mlx_array;

/// `TURBOEXL3_GLM53F_DSA_TAILS=1`, set by the plugin at load, before any forward.
pub var on: bool = false;
/// Forwards whose tail rows were read first, and DSA layers given their rows from there, since the load.
pub var runs: u64 = 0;
pub var served: u64 = 0;
/// Each cache's served appends found donated and copied since the load: the trunk's DSA layers in order, then the
/// head's.
pub var donated: [keys]u64 = @splat(0);
pub var copied: [keys]u64 = @splat(0);
/// The switch's load line (the A/B step reads "<switch>=1: ").
pub const line = "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_TAILS=1: a decode or verify forward reads its DSA layers' indexer tail rows (the 3 before it) first, in a command buffer of their own, so each layer's indexer cache append reuses its buffer instead of copying it whole (the same rows: the same bits)\n";

/// GLM-5.3-Flash's indexer rows (key | gate) and pools.
pub const row_w: c_int = 256;
pub const kpool: c_int = 4;
/// The trunk's DSA layers (layers 3, 7, ..., 43) and the MTP head's.
pub const trunk_layers: usize = 11;
pub const head_layers: usize = 1;
/// The rows a decode, verify or draft window has at most (glm5_next `HC_PRE_MAX_ROWS`): a prompt keeps the stock path.
pub const max_rows: c_int = 16;
/// The caches counted: the trunk's DSA layers', then the head's.
pub const keys = trunk_layers + head_layers;

/// One cache's buffer at a served forward's start: its address (the view's first row), the forward's first position
/// and rows, its layer, and whether the forward served it (`take`: its append ran after the early read).
pub const Track = struct { base: usize = 0, offset: u32 = 0, rows: c_int = 0, layer: u32 = 0, armed: bool = false };
var track: [keys]Track = @splat(.{});

/// At a served forward's start (cache buffer at `base`, first position `offset`): whether the last served forward's
/// append to this cache reused its buffer (true), copied it (false), or does not count (null: the last forward did not
/// serve it or this one does not continue it). MLX-free.
pub fn appendKept(t: Track, base: usize, offset: u32) ?bool {
    if (!t.armed or t.rows < 1 or offset <= t.offset or offset > t.offset + @as(u32, @intCast(t.rows))) return null;
    return base == t.base;
}

/// The rows `begin` reads before a forward of `rows` rows at `offset` over `n_layers` DSA layers (B `batch`, `slots`
/// slots): the tail's length (the stock `n_prev`), 0 where the switch does not serve. MLX-free.
pub fn tailRows(batch: c_int, rows: c_int, slots: usize, offset: u32, n_layers: usize) c_int {
    if (batch != 1 or rows < 1 or rows > max_rows or slots != 1) return 0;
    if (n_layers != trunk_layers and n_layers != head_layers) return 0;
    return @intCast(@min(offset, @as(u32, @intCast(kpool - 1))));
}

/// The gather of `n` layers' tails: input t<l> (layer l's tail [1, 1, R, 256]) to out [n, R, 256] in a buffer of its
/// own, one thread an element (the trunk's 11, the head's 1).
fn Gather(comptime n: usize) type {
    return struct {
        const name = std.fmt.comptimePrint("turboexl3_glm53f_dsa_tails{d}", .{n});
        const names = blk: {
            var a: [n][*:0]const u8 = undefined;
            for (0..n) |k| a[k] = std.fmt.comptimePrint("t{d}", .{k});
            break :blk a;
        };
        const source: [:0]const u8 = blk: {
            var src: []const u8 =
                \\const uint c = thread_position_in_grid.x;
                \\const uint r = thread_position_in_grid.y;
                \\const uint l = thread_position_in_grid.z;
                \\const uint R = threads_per_grid.y;
                \\const size_t i = size_t(r) * ROW_W + c;
                \\T v;
                \\switch (l) {
                \\
            ;
            for (0..n) |k| src = src ++ std.fmt.comptimePrint("case {d}: v = t{d}[i]; break;\n", .{ k, k });
            src = src ++ "default: return;\n}\nout[(size_t(l) * R + r) * ROW_W + c] = v;\n";
            const z = src ++ "\x00";
            break :blk z[0..src.len :0];
        };
        var kernel: ?mlx.mlx_fast_metal_kernel = null;

        /// The tails [1, 1, R, 256] each, in one launch: [n, R, 256], a new array.
        fn run(s: mlx.mlx_stream, tails: []const Array, rows: c_int, dt: mlx.mlx_dtype) !Array {
            const k = try kda_prompt.kernelOf(&kernel, name, &names, &.{"out"}, source, "");
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(n), rows, row_w }, 3, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, row_w, rows, @intCast(n)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, row_w, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROW_W", row_w));
            var o: [1]Array = undefined;
            try kda_prompt.applyKernel(s, k, cfg, tails, &o);
            return o[0];
        }
    };
}
const trunk_gather = Gather(trunk_layers);
const head_gather = Gather(head_layers);
/// Whether a forward found its tail copy inside a cache's buffer (declined; said once).
var shared_said = false;
pub var shared_declined: u64 = 0;

/// One forward's rows: `rows[i]` [1, R, 256] is layer `layers[i]`'s tail, a view of `gathered`.
const Table = struct {
    active: bool = false,
    cache: ?*anyopaque = null,
    offset: u32 = 0,
    n_prev: c_int = 0,
    n: usize = 0,
    layers: [trunk_layers]u32 = undefined,
    rows: [trunk_layers]Array = undefined,
    gathered: Array = .{ .ctx = null },
    first_key: usize = 0,
    dt: mlx.mlx_dtype = .bfloat16,
};
var table: Table = .{};
var engaged = [2]bool{ false, false };

/// Reads the tail rows of `layers` (DSA layers of slot 0's cache in `sl`, glm5_next `Slots`) before a forward of the
/// `ids_shape` [B, S] ids, `batched` a batched tick (declined), and commits them; `take` then hands them to each layer.
/// Any shape it does not serve leaves no rows (each layer reads its cache as before).
pub fn begin(s: mlx.mlx_stream, sl: anytype, batched: bool, ids_shape: []const c_int, layers: []const u32) !void {
    end();
    if (!on or batched or !mlx.streamIsGpu(s) or ids_shape.len != 2) return;
    if (sl.caches.len != 1 or sl.positions.len != 1) return;
    const offset = sl.positions[0];
    const n_prev = tailRows(ids_shape[0], ids_shape[1], sl.caches.len, offset, layers.len);
    if (n_prev == 0) return;
    // each layer's tail, as `dsaAttn` slices it: the view's last n_prev rows
    var tails: [trunk_layers]Array = undefined;
    var made: usize = 0;
    defer for (tails[0..made]) |t| {
        _ = mlx.mlx_array_free(t);
    };
    var dt: ?mlx.mlx_dtype = null;
    var view_t: [trunk_layers]c_int = undefined;
    for (layers) |layer| {
        var v = try sl.view(0, layer, s);
        defer v.deinit();
        if (v.v.ctx == null) return;
        const sh = mlx.getShape(v.v);
        if (sh.len != 4 or sh[0] != 1 or sh[1] != 1 or sh[2] < n_prev or sh[3] != row_w) return;
        const d = mlx.mlx_array_dtype(v.v);
        if (d != .bfloat16 and d != .float16) return;
        if (dt) |x| if (x != d) return;
        dt = d;
        var t = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice(&t, v.v, &[_]c_int{ 0, 0, sh[2] - n_prev, 0 }, 4, &[_]c_int{ 1, 1, sh[2], row_w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s)) catch |e| {
            _ = mlx.mlx_array_free(t);
            return e;
        };
        tails[made] = t;
        view_t[made] = sh[2];
        made += 1;
    }
    // the copy, in a buffer of its own: one launch (the trunk's 11 layers or the head's one)
    const g = if (layers.len == 1) try head_gather.run(s, tails[0..made], n_prev, dt.?) else try trunk_gather.run(s, tails[0..made], n_prev, dt.?);
    var g_kept = false;
    defer if (!g_kept) {
        _ = mlx.mlx_array_free(g);
    };
    // committed now: its command buffer runs while the forward is built
    {
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{g}, 1);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }
    // each cache's buffer now (the tail's address less its rows before it): the copy must lie outside every one, and
    // the donation count compares it with the last served forward's
    const first_key: usize = if (layers.len == 1) trunk_layers else 0;
    const g_at = dataAddr(g, dt.?);
    var bases: [trunk_layers]?usize = @splat(null);
    for (tails[0..made], view_t[0..made], 0..) |tail, vt, i| {
        const p = dataAddr(tail, dt.?) orelse continue;
        const base = p -% @as(usize, @intCast(vt - n_prev)) * @as(usize, @intCast(row_w)) * 2;
        bases[i] = base;
        if (g_at) |ga| if (ga >= base and ga < base + @as(usize, @intCast(vt)) * @as(usize, @intCast(row_w)) * 2) {
            shared_declined += 1;
            if (!shared_said) {
                shared_said = true;
                log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_TAILS declined: the tail copy lies inside a DSA cache's buffer (it would hold it as the stock read does); its layers read their views\n", .{});
            }
            return;
        };
    }
    for (bases[0..made], layers, 0..) |b, layer, i| {
        const base = b orelse continue;
        const k = first_key + i;
        if (appendKept(track[k], base, offset)) |kept| {
            if (kept) donated[k] += 1 else copied[k] += 1;
        }
        track[k] = .{ .base = base, .offset = offset, .rows = ids_shape[1], .layer = layer };
    }
    var tb: Table = .{ .active = true, .cache = sl.caches[0], .offset = offset, .n_prev = n_prev, .n = 0, .gathered = g, .first_key = first_key, .dt = dt.? };
    errdefer {
        for (tb.rows[0..tb.n]) |r| _ = mlx.mlx_array_free(r);
    }
    for (layers, 0..) |layer, i| {
        var r = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(r);
        var one = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one);
        const li: c_int = @intCast(i);
        try mlx.check(mlx.mlx_slice(&one, g, &[_]c_int{ li, 0, 0 }, 3, &[_]c_int{ li + 1, n_prev, row_w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_reshape(&r, one, &[_]c_int{ 1, n_prev, row_w }, 3, s));
        tb.layers[i] = layer;
        tb.rows[i] = r;
        tb.n += 1;
    }
    table = tb;
    g_kept = true;
    runs += 1;
    const which: usize = if (layers.len == 1) 1 else 0;
    if (!engaged[which]) {
        engaged[which] = true;
        log.info("[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_TAILS engaged: a {d}-row {s} forward's {d} DSA layer(s), {d} tail rows each, read first in one launch\n", .{ ids_shape[1], if (layers.len == 1) "head" else "trunk", layers.len, n_prev });
    }
}

/// Layer `layer`'s tail rows [1, n_prev, 256] read by this forward's `begin` (borrowed: valid until `end`), or null
/// (the layer reads its cache view as before): another slot, cache, position or row count than `begin` read for.
pub fn take(sl: anytype, slot: usize, layer: u32, n_prev: c_int) ?Array {
    if (!table.active or slot != 0 or n_prev != table.n_prev) return null;
    if (sl.caches.len <= slot or sl.positions.len <= slot) return null;
    if (sl.caches[slot] != table.cache or sl.positions[slot] != table.offset) return null;
    for (table.layers[0..table.n], table.rows[0..table.n], 0..) |l, r, i| if (l == layer) {
        served += 1;
        track[table.first_key + i].armed = true;
        return r;
    };
    return null;
}

/// The address of this forward's tail copy (its data set once `begin` committed it), or null without one: the device
/// check holds it apart from the caches' buffers.
pub fn gatheredAddr() ?usize {
    if (!table.active or table.gathered.ctx == null) return null;
    return dataAddr(table.gathered, table.dt);
}

/// The address of `a`'s first element (bf16 or f16, its data set), or null.
fn dataAddr(a: Array, dt: mlx.mlx_dtype) ?usize {
    return switch (dt) {
        .bfloat16 => if (mlx.mlx_array_data_bfloat16(a)) |p| @intFromPtr(p) else null,
        .float16 => if (mlx.mlx_array_data_float16(a)) |p| @intFromPtr(p) else null,
        else => null,
    };
}

/// The request summary's line: the served forwards and layers, and each cache's appends found donated and copied since
/// the load (`L<layer>=<donated>/<copied>`, the head's last).
pub fn summary(buf: []u8) []const u8 {
    var d: u64 = 0;
    var c: u64 = 0;
    for (donated, copied) |x, y| {
        d += x;
        c += y;
    }
    var n = (std.fmt.bufPrint(buf, "[turboexl3_glm53f] TURBOEXL3_GLM53F_DSA_TAILS forwards={d} layers_served={d} appends donated={d} copied={d}:", .{ runs, served, d, c }) catch return buf[0..0]).len;
    for (0..keys) |k| {
        const part = if (k < trunk_layers)
            std.fmt.bufPrint(buf[n..], " L{d}={d}/{d}", .{ track[k].layer, donated[k], copied[k] })
        else
            std.fmt.bufPrint(buf[n..], " head={d}/{d}", .{ donated[k], copied[k] });
        n += (part catch return buf[0..n]).len;
    }
    if (n < buf.len) {
        buf[n] = '\n';
        n += 1;
    }
    return buf[0..n];
}

/// Drops what `begin` read (the forward's end; also the start of the next `begin`).
pub fn end() void {
    if (!table.active) return;
    for (table.rows[0..table.n]) |r| _ = mlx.mlx_array_free(r);
    if (table.gathered.ctx != null) _ = mlx.mlx_array_free(table.gathered);
    table = .{};
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_DSA_TAILS reads the stock tail's 3 rows (fewer at the cache's start) for a decode, verify or draft window of one slot over the trunk's 11 DSA layers or the head's one, and nothing for a prompt, a batch or other layer counts" {
    const t = std.testing;
    // a verify window of 1 to 16 rows deep in the context: 3 rows
    for (1..17) |r| try t.expectEqual(@as(c_int, 3), tailRows(1, @intCast(r), 1, 16390, trunk_layers));
    try t.expectEqual(@as(c_int, 3), tailRows(1, 1, 1, 16390, head_layers));
    // the cache's first rows: as many as it holds, none at its start
    try t.expectEqual(@as(c_int, 2), tailRows(1, 3, 1, 2, trunk_layers));
    try t.expectEqual(@as(c_int, 1), tailRows(1, 1, 1, 1, head_layers));
    try t.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 0, trunk_layers));
    // a prompt chunk, a batch, two slots and other layer counts: none
    try t.expectEqual(@as(c_int, 0), tailRows(1, 17, 1, 16390, trunk_layers));
    try t.expectEqual(@as(c_int, 0), tailRows(1, 4096, 1, 16390, trunk_layers));
    try t.expectEqual(@as(c_int, 0), tailRows(2, 3, 1, 16390, trunk_layers));
    try t.expectEqual(@as(c_int, 0), tailRows(1, 3, 2, 16390, trunk_layers));
    try t.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 16390, 10));
    try t.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 16390, 0));
}

test "turboexl3_glm53f stream: TURBOEXL3_GLM53F_DSA_TAILS counts a served append as donated when the next served forward that continues it finds the same buffer, copied when another, and not at all after an unserved forward or a jump" {
    const t = std.testing;
    const last: Track = .{ .base = 0x1000, .offset = 16390, .rows = 3, .layer = 3, .armed = true };
    // the next verify starts 1 to 3 rows on (any accepted count): the same buffer is a donated append, another a copy
    for ([_]u32{ 16391, 16392, 16393 }) |o| {
        try t.expectEqual(@as(?bool, true), appendKept(last, 0x1000, o));
        try t.expectEqual(@as(?bool, false), appendKept(last, 0x2000, o));
    }
    // not continued: the same position (a re-run), behind it, past its rows (a new request or a prompt in between)
    for ([_]u32{ 16390, 100, 16394, 32768 }) |o| try t.expectEqual(@as(?bool, null), appendKept(last, 0x1000, o));
    // the last forward did not serve this cache (its append was the stock one), or recorded no rows
    var unserved = last;
    unserved.armed = false;
    try t.expectEqual(@as(?bool, null), appendKept(unserved, 0x1000, 16391));
    var none = last;
    none.rows = 0;
    try t.expectEqual(@as(?bool, null), appendKept(none, 0x1000, 16391));
}
