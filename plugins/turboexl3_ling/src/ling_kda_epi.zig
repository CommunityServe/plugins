//! turboexl3_ling's KDA step epilogue (exact): a KDA layer's decode step and exact verify launch also writes what the
//! launches after it made (sdk/blocks/kda/ling.zig `decodeStepEpi` and `verifyRowsEpi`, the epilogue
//! kernels/kda/ling_kda_epi.metal): a verify's window inputs, kept for a partial accept's rollback (the concatenate's
//! copy), and o_proj's DEC-ONE prepared rows of y, their block sums and its GEMV's zeroed arrival counters
//! (kernels/kda/ling_kda_oprep.metal: turboexl3's prep_sc statements). o_proj then runs from them (turboexl3's
//! `dense.Dense.linearFromPrepared`: DEC-ONE's GEMV alone at 1 to 4 rows, the chain's GEMV and finish at 5 to 8), which
//! is `linear`'s bytes of y whichever prepare `linear` takes there. So a verify's KDA layer runs two launches fewer (the
//! concatenate, o_proj's prepare) and a decode step one, every word the same: this plugin's src/ling_kda_test.zig
//! (KDA_EPILOGUE) holds the window, the state, the state after every row and the window's inputs to the served launches
//! word for word at 1 to 8 rows, and turboexl3's models/ling/ling_test.zig ("gpu ling kda epilogue o_proj",
//! TURBOEXL3_LING_KDA_EPI_OPROJ) holds `linearFromPrepared`'s words to `linear`'s at 1 to 4 rows. Where o_proj is not
//! EXL3, turboexl3 takes neither path at the call's rows or
//! the launch declines, the layer keeps its path. Each step that takes it marks KDA_EPILOGUE on the run-path line
//! (src/ling_ran.zig).
const mlx = @import("mlx_serve").mlx;
const t3 = @import("turboexl3");
const ling_kda_k = @import("mlx_serve/sdk/blocks/kda/ling.zig");
const ran = @import("ling_ran.zig");

fn free(a: mlx.mlx_array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

/// What the layer keeps from a step with the epilogue: o_proj's output [1, S, out] (what `m.mm(&la.out, y)` returns
/// for y [1, S, hv dv]), the next window and state, and a verify's state after every row and window inputs (empty at
/// decode). The caller owns each.
pub const Out = struct {
    out: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    state: mlx.mlx_array,
    state_seq: mlx.mlx_array = .{ .ctx = null },
    conv_input: mlx.mlx_array = .{ .ctx = null },
};

/// o_proj at the call's rows: its Dense (`linear`'s own, by shape and y's dtype), its operands and the counters its
/// GEMV reads
const Plan = struct { d: *t3.dense.Dense, p: t3.Proj, ncnt: c_int };

/// o_proj's plan at `rows` rows over y `width` wide in `xd`; null where o_proj is not EXL3, its input is not y's width,
/// or turboexl3 takes neither DEC-ONE nor the chain from prepared rows there
fn planFor(o: anytype, rows: usize, width: c_int, xd: mlx.mlx_dtype) !?Plan {
    if (!o.exl3) return null;
    const p = t3.Proj{ .trellis = o.w, .suh = o.s, .svh = o.b };
    const d = t3.denseOf(p, @intCast(width), xd, .mul1) catch |e| switch (e) {
        error.BadExl3Shape, error.BadExl3Dtype => return null,
        else => return e,
    };
    // DEC-ONE's GEMV reads its counters; the chain's reads none (8 stand in: MLX binds a smaller input in constant space)
    const ncnt: u32 = if (d.decTakes(rows)) try d.decCounters(rows) else if (d.preparedTakes(rows)) 8 else return null;
    return .{ .d = d, .p = p, .ncnt = @intCast(@max(ncnt, 8)) };
}

/// The step's rows, or null outside a [1, S, C] call
fn rowsOf(in: ling_kda_k.DecodeIn) ?usize {
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[1] < 1) return null;
    return @intCast(qsh[1]);
}

/// The layer's decode step with the epilogue, then o_proj from its prepared rows; null where any piece declines (the
/// caller runs `decodeStep` and o_proj as before; nothing was launched)
pub fn decodeStep(s: mlx.mlx_stream, in: ling_kda_k.DecodeIn, o: anytype) !?Out {
    const rows = rowsOf(in) orelse return null;
    const plan = (try planFor(o, rows, in.hv * in.dv, mlx.mlx_array_dtype(in.qkv))) orelse return null;
    const r = (try ling_kda_k.decodeStepEpi(s, in, .{ .suh = o.s, .ncnt = plan.ncnt })) orelse return null;
    return try finish(s, plan, r, rows);
}

/// An exact verify's rows with the epilogue, then o_proj from their prepared rows; null as `decodeStep`
pub fn verifyRows(s: mlx.mlx_stream, in: ling_kda_k.DecodeIn, o: anytype) !?Out {
    const rows = rowsOf(in) orelse return null;
    const plan = (try planFor(o, rows, in.hv * in.dv, mlx.mlx_array_dtype(in.qkv))) orelse return null;
    const r = (try ling_kda_k.verifyRowsEpi(s, in, .{ .suh = o.s, .ncnt = plan.ncnt })) orelse return null;
    return try finish(s, plan, r, rows);
}

/// o_proj's GEMV over the step's prepared rows, reshaped as `linear` shapes y's [1, S, out]; the step's outputs the
/// layer keeps handed over, the rest freed
fn finish(s: mlx.mlx_stream, plan: Plan, r: ling_kda_k.EpiOut, rows: usize) !Out {
    defer for ([_]mlx.mlx_array{ r.y, r.xp, r.xs, r.cnt }) |a| free(a);
    errdefer for ([_]mlx.mlx_array{ r.conv_state, r.state, r.state_seq, r.conv_input }) |a| free(a);
    const y2 = try plan.d.linearFromPrepared(s, r.xp, r.xs, r.cnt, plan.p.trellis, plan.p.svh, rows);
    defer free(y2);
    var out = mlx.mlx_array_new();
    errdefer free(out);
    try mlx.check(mlx.mlx_reshape(&out, y2, &[_]c_int{ 1, @intCast(rows), mlx.getShape(y2)[1] }, 3, s));
    ran.mark(.KDA_EPILOGUE);
    return .{ .out = out, .conv_state = r.conv_state, .state = r.state, .state_seq = r.state_seq, .conv_input = r.conv_input };
}
