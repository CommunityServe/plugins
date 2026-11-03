//! The shared expert of a wide prefill chunk on the Neural Engine (`SUSHI_DS41_ANE_SHARED`, a prototype, off by
//! default): one int8 SwiGLU procedure per covered layer in one bank per unit (src/ane, a renamed copy of mlx-serve's
//! lib/ane). With two units each die takes half the chunk's rows, so a layer's program takes half the time and both
//! dies run at once. The GPU runs the routed experts meanwhile; the model adds this output before the next layer.
//!
//! I/O is fp16 in channel-major planes: unit u reads `[hidden][rows]` of rows `u * rows ..`, and writes `[hidden][rows]`.

const std = @import("std");
const log = @import("../log.zig");

const Mlp = opaque {};
const Bank = opaque {};
extern fn s41_ane_available() c_int;
extern fn s41_ane_bank_create() ?*Bank;
extern fn s41_ane_bank_free(b: ?*Bank) void;
extern fn s41_ane_bank_add_mlp(b: ?*Bank, hidden: u32, ffn: u32, rows: u32, gate_q: [*]const i8, gate_s: [*]const f32, up_q: [*]const i8, up_s: [*]const f32, down_q: [*]const i8, down_s: [*]const f32, limit: f32, err: [*]u8, err_size: usize) c_int;
extern fn s41_ane_bank_finish(b: ?*Bank, name: [*:0]const u8, ane_instance: c_int, input_plane: ?*anyopaque, output_plane: ?*anyopaque, err: [*]u8, err_size: usize) ?*Mlp;
extern fn s41_ane_mlp_free(m: ?*Mlp) void;
extern fn s41_ane_mlp_input(m: ?*Mlp) ?[*]f16;
extern fn s41_ane_mlp_output(m: ?*Mlp) ?[*]f16;
extern fn s41_ane_mlp_eval(m: ?*Mlp, procedure: u32, err: [*]u8, err_size: usize) c_int;
extern fn s41_ane_mlp_compile_seconds(m: ?*const Mlp) f64;
extern fn s41_ane_mlp_cache_hit(m: ?*const Mlp) c_int;

pub const max_units = 2;

/// Monotonic nanoseconds.
pub fn nowNs() u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    return @intCast(std.Io.Timestamp.now(io, .boot).nanoseconds);
}
pub const max_layers = 64;

pub fn available() bool {
    return s41_ane_available() != 0;
}

/// Per-output-row int8 of a row-major `[rows, cols]` f32 weight: `q` `[rows, cols]`, `s` `[rows]` (w ~ q * s).
pub fn quantizeRows(w: []const f32, rows: usize, cols: usize, q: []i8, s: []f32) void {
    for (0..rows) |r| {
        const row = w[r * cols ..][0..cols];
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        const scale: f32 = if (amax > 0) amax / 127.0 else 1.0;
        s[r] = scale;
        const inv = 1.0 / scale;
        for (row, q[r * cols ..][0..cols]) |v, *o| o.* = @intFromFloat(std.math.clamp(@round(v * inv), -127, 127));
    }
}

pub const Programs = struct {
    units: [max_units]?*Mlp = @splat(null),
    banks: [max_units]?*Bank = @splat(null),
    n_units: usize,
    /// Rows a unit's program takes; a chunk is `n_units * rows` rows (zero-padded).
    rows: u32,
    hidden: u32,
    ffn: u32,
    /// The bank procedure of each layer, -1 where the layer has none.
    proc: [max_layers]i32 = @splat(-1),

    pub fn init(n_units: usize, chunk_rows: u32, hidden: u32, ffn: u32) !Programs {
        if (n_units == 0 or n_units > max_units or chunk_rows % (32 * @as(u32, @intCast(n_units))) != 0) return error.AneSharedShape;
        var p: Programs = .{ .n_units = n_units, .rows = chunk_rows / @as(u32, @intCast(n_units)), .hidden = hidden, .ffn = ffn };
        for (0..n_units) |u| p.banks[u] = s41_ane_bank_create() orelse return error.AneSharedBank;
        return p;
    }

    pub fn deinit(self: *Programs) void {
        for (&self.banks) |*b| if (b.*) |x| {
            s41_ane_bank_free(x);
            b.* = null;
        };
        for (&self.units) |*u| if (u.*) |x| {
            s41_ane_mlp_free(x);
            u.* = null;
        };
    }

    /// Adds layer `li`'s program to every unit's bank: `gate`/`up` `[ffn, hidden]`, `down` `[hidden, ffn]`, int8 rows.
    pub fn addLayer(self: *Programs, li: usize, gq: []const i8, gs: []const f32, uq: []const i8, us: []const f32, dq: []const i8, ds: []const f32, limit: f32) !void {
        var err: [256]u8 = @splat(0);
        var idx: c_int = -1;
        for (0..self.n_units) |u| {
            const got = s41_ane_bank_add_mlp(self.banks[u], self.hidden, self.ffn, self.rows, gq.ptr, gs.ptr, uq.ptr, us.ptr, dq.ptr, ds.ptr, limit, &err, err.len);
            if (got < 0) {
                log.err("[sushi_ds41] ane shared: layer {d} program: {s}\n", .{ li, std.mem.sliceTo(&err, 0) });
                return error.AneSharedBank;
            }
            if (idx >= 0 and got != idx) return error.AneSharedBank;
            idx = got;
        }
        self.proc[li] = idx;
    }

    /// Compiles (or restores from the compile cache) and loads each unit's bank, pinned to die u + 1.
    pub fn finish(self: *Programs) !void {
        var err: [512]u8 = @splat(0);
        for (0..self.n_units) |u| {
            var name: [64]u8 = undefined;
            const txt = try std.fmt.bufPrint(name[0 .. name.len - 1], "ds41_shared_r{d}_u{d}of{d}", .{ self.rows, u, self.n_units });
            name[txt.len] = 0;
            const nm: [*:0]const u8 = @ptrCast(&name);
            // A single unit gets no instance hint (the runtime picks a die); two pin one die each.
            const inst: c_int = if (self.n_units == 1) 0 else @intCast(u + 1);
            const b = self.banks[u];
            self.banks[u] = null;
            self.units[u] = s41_ane_bank_finish(b, nm, inst, null, null, &err, err.len) orelse {
                log.err("[sushi_ds41] ane shared: compile of unit {d}: {s}\n", .{ u, std.mem.sliceTo(&err, 0) });
                return error.AneSharedCompile;
            };
            log.info("[sushi_ds41] ane shared: unit {d} (die hint {d}) {d} rows, compile {d:.2} s, cache {s}\n", .{ u, inst, self.rows, s41_ane_mlp_compile_seconds(self.units[u]), if (s41_ane_mlp_cache_hit(self.units[u]) != 0) "hit" else "miss" });
        }
    }

    pub fn slabLen(self: *const Programs) usize {
        return @as(usize, self.hidden) * self.rows;
    }

    pub fn input(self: *const Programs, u: usize) [*]f16 {
        return s41_ane_mlp_input(self.units[u]).?;
    }

    pub fn output(self: *const Programs, u: usize) [*]const f16 {
        return s41_ane_mlp_output(self.units[u]).?;
    }

    const Job = struct { p: *const Programs, u: usize, proc: u32, x: ?[*]const f16, ok: bool = false, ns: u64 = 0, copy_ns: u64 = 0 };

    fn runUnit(j: *Job) void {
        const t0 = nowNs();
        const n = j.p.slabLen();
        const dst = s41_ane_mlp_input(j.p.units[j.u]).?;
        // A null input means the GPU already wrote the plane (`input`).
        if (j.x) |x| @memcpy(dst[0..n], x[j.u * n ..][0..n]);
        j.copy_ns = nowNs() - t0;
        var err: [256]u8 = @splat(0);
        // eval returns 1 on success.
        j.ok = s41_ane_mlp_eval(j.p.units[j.u], j.proc, &err, err.len) == 1;
        if (!j.ok) log.warn("[sushi_ds41] ane shared: eval unit {d}: {s}\n", .{ j.u, std.mem.sliceTo(&err, 0) });
        j.ns = nowNs() - t0;
    }

    pub const Ran = struct { ok: bool, unit_ns: [max_units]u64, copy_ns: u64 = 0 };

    /// Layer `li` over `x`: `n_units` channel-major slabs of `slabLen()` fp16 each, back to back. Unit 0 runs on the
    /// calling thread, unit 1 on its own; both copy their slab in and evaluate. The outputs are in `output(u)`.
    pub fn run(self: *const Programs, li: usize, x: [*]const f16) Ran {
        var jobs: [max_units]Job = undefined;
        const proc: u32 = @intCast(self.proc[li]);
        for (0..self.n_units) |u| jobs[u] = .{ .p = self, .u = u, .proc = proc, .x = x };
        var th: ?std.Thread = null;
        if (self.n_units > 1) th = std.Thread.spawn(.{}, runUnit, .{&jobs[1]}) catch null;
        runUnit(&jobs[0]);
        if (self.n_units > 1) {
            if (th) |t| t.join() else runUnit(&jobs[1]);
        }
        var r: Ran = .{ .ok = true, .unit_ns = @splat(0) };
        for (0..self.n_units) |u| {
            r.ok = r.ok and jobs[u].ok;
            r.unit_ns[u] = jobs[u].ns;
            r.copy_ns = @max(r.copy_ns, jobs[u].copy_ns);
        }
        return r;
    }

    /// `run` with every unit on its own thread, returned at once: the caller queues GPU work, then `join`s.
    pub const Pending = struct {
        jobs: [max_units]Job = undefined,
        th: [max_units]?std.Thread = @splat(null),
        n: usize = 0,

        pub fn join(self: *Pending) Ran {
            for (0..self.n) |u| {
                if (self.th[u]) |t| t.join() else runUnit(&self.jobs[u]);
            }
            var r: Ran = .{ .ok = true, .unit_ns = @splat(0) };
            for (0..self.n) |u| {
                r.ok = r.ok and self.jobs[u].ok;
                r.unit_ns[u] = self.jobs[u].ns;
                r.copy_ns = @max(r.copy_ns, self.jobs[u].copy_ns);
            }
            return r;
        }
    };

    /// Starts layer `li` over `x` on every unit; `p` must stay put until `p.join()`.
    pub fn start(self: *const Programs, p: *Pending, li: usize, x: ?[*]const f16) void {
        const proc: u32 = @intCast(self.proc[li]);
        p.n = self.n_units;
        for (0..self.n_units) |u| {
            p.jobs[u] = .{ .p = self, .u = u, .proc = proc, .x = x };
            // A unit whose thread did not start runs in `join`.
            p.th[u] = std.Thread.spawn(.{}, runUnit, .{&p.jobs[u]}) catch null;
        }
    }

    /// One eval of procedure 0 on every unit with whatever the planes hold: wakes the dies.
    pub fn warm(self: *const Programs) void {
        var err: [256]u8 = @splat(0);
        for (0..self.n_units) |u| _ = s41_ane_mlp_eval(self.units[u], 0, &err, err.len);
    }
};
