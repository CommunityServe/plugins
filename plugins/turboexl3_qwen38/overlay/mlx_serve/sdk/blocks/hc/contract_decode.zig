//! copied_from: src/sdk/blocks/hc/contract_decode.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The EXL3 trunk read's (`hc_decode2.readTrunk`) multi-row and decode3 launches in the kernel contract; `read`'s one-row
//! launches are `contract.zig`'s. A row's bytes are the one-row launches'
//! (`hc_read_nd`, `hc_read_u2`), so each registration replaces one of them: gate 3 holds it to that launch's bytes on the
//! same inputs. `contract.zig`'s `Nd` and `Up` make the inputs and the float64 reference, and run the one-row launch on
//! a rig of their own as the baseline.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const hcc = @import("contract.zig");
const hd2 = @import("hc_decode2.zig");

const spec = kc.spec;
const metal = kc.metal;
const Shape = spec.Shape;
const HC = hcc.HC;
const HID = hcc.HID;
const K = hcc.K;
const R = hcc.R;
const PR = R + HC;

// M1/M2: no offline compile is recorded for these launches yet (`offline.zig`); 1024 stands until one is.

/// Threadgroup bytes: `xs[RB * H]` (bf16) at the most rows, `npart[RB * 8]` and `tgi[8 * HC]` (f32).
const ndm_launch: metal.Launch = .{
    .name = "mlxserve_hc_read_ndm",
    .inputs = &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q", "dw_s", "dw_b" },
    .outputs = &.{ "xn_out", "parts_out", "xs_out" },
    .source = hd2.NDM_SOURCE,
    .header = hd2.PACK8_HEADER,
    .tg_bytes = @intCast(2 * hd2.ND_RB_MAX * HID + 4 * 8 * hd2.ND_RB_MAX + 4 * 8 * HC),
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes: `acts[RB * R]` (bf16) at the most rows.
const u2m_launch: metal.Launch = .{
    .name = "mlxserve_hc_read_u2m",
    .inputs = &.{ "xn_in", "parts_in", "uw_q", "uw_s", "uw_b" },
    .outputs = &.{ "mixed_out", "inj_out" },
    .source = hd2.U2M_SOURCE,
    .header = hd2.PACK8_HEADER,
    .tg_bytes = @intCast(2 * hd2.UP_RB_MAX * R),
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes at the most rows a block holds: `xs4[RBD * H / 8]` and `xsm4[RBD * H / 64]` (16-byte vectors),
/// `npart[RBD * 8]` and `tgi[8 * HC]` (f32).
const d3n_launch: metal.Launch = .{
    .name = "mlxserve_hc_read_d3n",
    .inputs = &.{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in", "dw_q" },
    .outputs = &.{ "xn_out", "parts_out", "xs_out" },
    .source = hd2.D3N_SOURCE,
    .header = hd2.PACK8_HEADER,
    .tg_bytes = @intCast(16 * hd2.D3_RBD * @divExact(HID, 8) + 16 * hd2.D3_RBD * @divExact(HID, 64) + 4 * 8 * hd2.D3_RBD + 4 * 8 * HC),
    .m1 = 1024,
    .m2 = 1024,
};

/// Threadgroup bytes: `act2[RBU * R / 2]` (f32 pairs) at the most rows a block holds (a lane per row and stream).
const u3_launch: metal.Launch = .{
    .name = "mlxserve_hc_read_u3",
    .inputs = &.{ "xn_in", "parts_in", "uw_q" },
    .outputs = &.{ "mixed_out", "inj_out" },
    .source = hd2.U3_SOURCE,
    .header = hd2.PACK8_HEADER,
    .tg_bytes = 8 * @divExact(32, HC) * @divExact(R, 2),
    .m1 = 1024,
    .m2 = 1024,
};

/// The norm + down stage at 2..16 rows: the multi-row launch (`ndm`) or decode3's (`d3n`). `nd` holds the inputs and the
/// reference (the one-row launch's formula, row by row) and runs the one-row launch on its own rig for `replaced`.
const Down = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    nd: hcc.Nd,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, one_row: *const metal.Launch, cfg: hcc.Cfg, part: hcc.NdPart) Down {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .nd = hcc.Nd.init(a, one_row, cfg, part) };
    }

    pub fn deinit(self: *Down) void {
        self.rig.deinit();
        self.nd.deinit();
    }

    fn decode3(self: *const Down) bool {
        return self.launch == &d3n_launch;
    }

    /// The wrapper's configs (`hc_decode2.readTrunk`): rows on grid z in blocks, the template arguments in its order.
    pub fn geometry(self: *const Down, shape: Shape) ?metal.Geometry {
        const c = self.nd.cfg;
        if (shape.rows < 2 or shape.rows > 16) return null;
        const rows: c_int = @intCast(shape.rows);
        const inj: c_int = @intFromBool(c.inj);
        if (self.decode3() and !hd2.decode3Fits(rows, HC, HID, R, c.bits, .bfloat16)) return null;
        const blk = if (self.decode3()) hd2.d3Blocks(rows, hd2.D3_RBD) else hd2.multiBlocks(rows, hd2.ND_RB_MAX);
        const groups: c_int = if (self.decode3()) @divExact(R, hd2.D3_SGD * hd2.D3_DRP) else @divExact(R, 8);
        var g = metal.Geometry{ .grid = .{ 256, (groups + inj) * HC, blk[1] }, .tg = .{ 256, 1, 1 } };
        g.out(&.{rows * K}, .bfloat16);
        g.out(&.{rows * HC * PR}, .float32);
        g.out(&.{if (c.wr) rows * K else 1}, .bfloat16);
        g.unread = if (self.nd.part == .partials) 0b101 else if (c.wr) 0b010 else 0b110;
        g.dtype("T", .bfloat16);
        g.int("HC", HC);
        g.int("H", HID);
        g.int("R", R);
        if (self.decode3()) {
            g.int("INJ", inj);
            g.int("WR", @intFromBool(c.wr));
            g.int("DRP", hd2.D3_DRP);
            g.int("RBD", blk[0]);
            g.int("ROWS", rows);
            g.int("LF", hd2.D3_LF);
        } else {
            g.int("GS", c.gs);
            g.int("BITS", c.bits);
            g.int("INJ", inj);
            g.int("WR", @intFromBool(c.wr));
            g.int("RPS", 1);
            g.int("ROWS_TG", 8);
            g.int("RB", blk[0]);
            g.int("ROWS", rows);
        }
        return g;
    }

    pub fn prepare(self: *Down, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        // `Nd` makes the inputs and binds them on this rig, lent for the call; its own rig keeps the baseline launch.
        std.mem.swap(metal.Rig, &self.rig, &self.nd.rig);
        defer std.mem.swap(metal.Rig, &self.rig, &self.nd.rig);
        return self.nd.prepare(shape, fill, seed);
    }

    /// decode3 reads no scales or biases: its inputs are the first seven.
    fn args(self: *const Down) []const mlx.mlx_array {
        return self.nd.args[0..if (self.decode3()) 7 else 9];
    }

    pub fn run(self: *Down) !void {
        try self.rig.launch(self.launch, self.geometry(self.nd.shape).?, self.args());
    }

    pub fn count(self: *const Down) usize {
        return self.nd.count();
    }

    pub fn reference(self: *const Down, want: []f64, mag: []f64) !void {
        try self.nd.reference(want, mag);
    }

    pub fn simulate(self: *const Down, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.nd.simulate(acc, order, out);
    }

    /// The one-row launch's bytes on the same inputs.
    pub fn replaced(self: *Down) ![]const u8 {
        try self.nd.run();
        return self.nd.rig.outputBytes();
    }
};

/// The up/mix stage at 2..16 rows: the multi-row launch (`u2m`) or decode3's (`u3`). `up` holds the inputs and the
/// reference and runs the one-row launch on its own rig for `replaced`.
const Mix = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    up: hcc.Up,

    fn init(a: std.mem.Allocator, launch: *const metal.Launch, one_row: *const metal.Launch, cfg: hcc.Cfg) Mix {
        return .{ .rig = metal.Rig.init(a), .launch = launch, .up = hcc.Up.init(a, one_row, cfg, false) };
    }

    pub fn deinit(self: *Mix) void {
        self.rig.deinit();
        self.up.deinit();
    }

    fn decode3(self: *const Mix) bool {
        return self.launch == &u3_launch;
    }

    pub fn geometry(self: *const Mix, shape: Shape) ?metal.Geometry {
        const c = self.up.cfg;
        if (shape.rows < 2 or shape.rows > 16) return null;
        const rows: c_int = @intCast(shape.rows);
        if (self.decode3() and !hd2.decode3Fits(rows, HC, HID, R, c.bits, .bfloat16)) return null;
        const cols: c_int = if (self.decode3()) hd2.D3_SGU else 16;
        const blk = if (self.decode3()) hd2.d3Blocks(rows, @divExact(32, HC)) else hd2.multiBlocks(rows, hd2.UP_RB_MAX);
        var g = metal.Geometry{ .grid = .{ 32 * cols, @divExact(HID, cols), blk[1] }, .tg = .{ 32 * cols, 1, 1 } };
        g.out(&.{rows * HID}, .bfloat16);
        g.out(&.{rows * HC}, .bfloat16);
        if (!c.inj) g.unread = 0b10;
        g.dtype("T", .bfloat16);
        g.int("HC", HC);
        g.int("H", HID);
        g.int("R", R);
        if (self.decode3()) {
            g.int("INJ", @intFromBool(c.inj));
            g.int("SGU", cols);
            g.int("RBU", blk[0]);
            g.int("ROWS", rows);
            g.int("WH", hd2.D3_WH);
            g.int("LF", hd2.D3_LF);
        } else {
            g.int("GS", c.gs);
            g.int("BITS", c.bits);
            g.int("INJ", @intFromBool(c.inj));
            g.int("COLS", cols);
            g.int("RB", blk[0]);
            g.int("ROWS", rows);
        }
        return g;
    }

    pub fn prepare(self: *Mix, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        // As `Down.prepare`: `Up` binds on this rig, lent for the call.
        std.mem.swap(metal.Rig, &self.rig, &self.up.rig);
        defer std.mem.swap(metal.Rig, &self.rig, &self.up.rig);
        return self.up.prepare(shape, fill, seed);
    }

    /// decode3 reads no scales or biases: its inputs are the first three.
    fn args(self: *const Mix) []const mlx.mlx_array {
        return self.up.args[0..if (self.decode3()) 3 else 5];
    }

    pub fn run(self: *Mix) !void {
        try self.rig.launch(self.launch, self.geometry(self.up.shape).?, self.args());
    }

    pub fn count(self: *const Mix) usize {
        return self.up.count();
    }

    pub fn reference(self: *const Mix, want: []f64, mag: []f64) !void {
        try self.up.reference(want, mag);
    }

    pub fn simulate(self: *const Mix, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.up.simulate(acc, order, out);
    }

    /// The one-row launch's bytes on the same inputs.
    pub fn replaced(self: *Mix) ![]const u8 {
        try self.up.run();
        return self.up.rig.outputBytes();
    }
};

/// Two rows, a partial block of the multi-row launches (four rows a block), the widest read.
const rows_served: []const Shape = &.{ .{ .rows = 2, .n = R, .k = K }, .{ .rows = 5, .n = R, .k = K }, .{ .rows = 16, .n = R, .k = K } };

fn rowsSpec(comptime name: []const u8, out: spec.Elem, sums: spec.Sums, comptime replaces: []const u8) spec.Spec {
    var s = hcc.decodeSpec(name, out, rows_served, sums);
    s.replaces = replaces;
    return s;
}

const exact: spec.Sums = .{};
const up_sums: spec.Sums = .{ .f16 = 100, .bf16 = 100 };
/// The dense reads' configuration: bf16 weights, the group the wrapper is handed for them.
const dense: hcc.Cfg = .{ .bits = 16, .gs = 16 };

pub fn register(set: *kc.Set) !void {
    const a = set.a;
    const by_nd = "the one-row norm + down launch (hc_read_nd)";
    const by_u2 = "the one-row up + mix launch (hc_read_u2)";
    try set.add(Down, .init(a, &ndm_launch, &hcc.nd8, .{ .bits = 8 }, .streams), rowsSpec("hc_read_ndm 8-bit (normed + written streams)", .bf16, hcc.norm_sums, by_nd), null);
    try set.add(Down, .init(a, &ndm_launch, &hcc.nd8, .{ .bits = 8 }, .partials), rowsSpec("hc_read_ndm 8-bit (down + inject partials)", .f32, exact, by_nd), null);
    try set.add(Down, .init(a, &ndm_launch, &hcc.nd4, .{ .bits = 4 }, .partials), rowsSpec("hc_read_ndm 4-bit (down + inject partials)", .f32, exact, by_nd), null);
    try set.add(Down, .init(a, &ndm_launch, &hcc.nd16, dense, .partials), rowsSpec("hc_read_ndm bf16 weights (down + inject partials)", .f32, exact, by_nd), null);
    try set.add(Mix, .init(a, &u2m_launch, &hcc.up4, .{ .bits = 4 }), rowsSpec("hc_read_u2m 4-bit", .bf16, up_sums, by_u2), null);
    try set.add(Mix, .init(a, &u2m_launch, &hcc.up16, dense), rowsSpec("hc_read_u2m bf16 weights", .bf16, up_sums, by_u2), null);
    try set.add(Down, .init(a, &d3n_launch, &hcc.nd16, dense, .streams), rowsSpec("hc_read_d3n bf16 weights (normed + written streams)", .bf16, hcc.norm_sums, by_nd), null);
    try set.add(Down, .init(a, &d3n_launch, &hcc.nd16, dense, .partials), rowsSpec("hc_read_d3n bf16 weights (down + inject partials)", .f32, exact, by_nd), null);
    try set.add(Down, .init(a, &d3n_launch, &hcc.nd16, .{ .bits = 16, .gs = 16, .inj = false, .wr = false }, .streams), rowsSpec("hc_read_d3n bf16 weights (no inject, no pending write)", .bf16, hcc.norm_sums, by_nd), null);
    try set.add(Mix, .init(a, &u3_launch, &hcc.up16, dense), rowsSpec("hc_read_u3 bf16 weights", .bf16, up_sums, by_u2), null);
    try set.add(Mix, .init(a, &u3_launch, &hcc.up16, .{ .bits = 16, .gs = 16, .inj = false }), rowsSpec("hc_read_u3 bf16 weights (no inject)", .bf16, up_sums, by_u2), null);
}

// ── tests ──

/// The declared launches chained (norm + down, then up + mix) equal the shipped trunk read's (`readTrunk`) outputs at
/// `rows`: the multi-row pair, or decode3's (the read takes it on dense weights unless `decode3_override` says otherwise).
fn drift(a: std.mem.Allocator, down: *Down, mix: *Mix, rows: usize) !void {
    const shape: Shape = .{ .rows = rows, .n = R, .k = K };
    try std.testing.expect(try down.prepare(shape, .random, 31));
    var geo = down.geometry(shape).?;
    geo.unread = 0;
    try down.rig.launch(down.launch, geo, down.args());
    try std.testing.expect(try mix.prepare(shape, .random, 32));
    // The up launch reads the declared down launch's outputs and the up test's weights.
    mix.up.args[0] = down.rig.output(0);
    mix.up.args[1] = down.rig.output(1);
    try mix.run();
    const ours = try a.dupe(u8, mix.rig.outputBytes());
    defer a.free(ours);
    const c = down.nd.cfg;
    const x = down.nd.args;
    const u = mix.up.args;
    const is_dense = c.bits == 16;
    const none = mlx.mlx_array{ .ctx = null };
    const multi0 = hd2.multi_served;
    const d3_0 = hd2.d3_served;
    const out = (try hd2.readTrunk(down.rig.stream(), x[0], x[1], x[6], if (is_dense) none else x[7], if (is_dense) none else x[8], u[2], if (is_dense) none else u[3], if (is_dense) none else u[4], if (c.inj) x[2] else none, x[3], @intCast(rows), HC, HID, c.bits, @intCast(c.gs), if (c.wr) .{ .out = x[4], .inj = x[5] } else null)) orelse return error.Declined;
    defer for ([_]mlx.mlx_array{ out.mixed, out.inj, out.stream }) |o| if (o.ctx != null) {
        _ = mlx.mlx_array_free(o);
    };
    // The read took the pair declared here.
    if (down.decode3()) try std.testing.expectEqual(d3_0 + 1, hd2.d3_served) else try std.testing.expectEqual(multi0 + 1, hd2.multi_served);
    const theirs = if (c.inj) try mix.rig.bytesOf(&.{ out.mixed, out.inj }) else try mix.rig.bytesOf(&.{out.mixed});
    try std.testing.expectEqualSlices(u8, theirs, ours);
}

test "kernel contract hc decode: the multi-row and decode3 launches chained are the shipped read's, byte for byte" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    // 4-bit: the multi-row pair; dense bf16: decode3's, then the multi-row pair with decode3 held off.
    {
        var down = Down.init(a, &ndm_launch, &hcc.nd4, .{ .bits = 4 }, .streams);
        defer down.deinit();
        var mix = Mix.init(a, &u2m_launch, &hcc.up4, .{ .bits = 4 });
        defer mix.deinit();
        try drift(a, &down, &mix, 5);
    }
    {
        var down = Down.init(a, &d3n_launch, &hcc.nd16, dense, .streams);
        defer down.deinit();
        var mix = Mix.init(a, &u3_launch, &hcc.up16, dense);
        defer mix.deinit();
        try drift(a, &down, &mix, 5);
    }
    {
        hd2.decode3_override = false;
        defer hd2.decode3_override = null;
        var down = Down.init(a, &ndm_launch, &hcc.nd16, dense, .streams);
        defer down.deinit();
        var mix = Mix.init(a, &u2m_launch, &hcc.up16, dense);
        defer mix.deinit();
        try drift(a, &down, &mix, 16);
    }
}
