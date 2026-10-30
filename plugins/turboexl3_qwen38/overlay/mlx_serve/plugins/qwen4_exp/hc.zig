//! copied_from: src/plugins/qwen4_exp/hc.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Hyper-connection streams: the compiled read/write callbacks (`compileQwen4Hc`), the stand-ins, `HcRead`, and the
//! grouped RMS norm over the `hc` streams.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ple = @import("ple.zig");
const applyClosure = @import("../../engines/mlx/compiled.zig").applyClosure;
const closureIn = @import("../../engines/mlx/compiled.zig").closureIn;
const closureOut = @import("../../engines/mlx/compiled.zig").closureOut;
const scalarOf = @import("../../engines/mlx/load/weights.zig").scalarOf;
const Trunk = @import("trunk.zig").Trunk;
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const HC_FUSED_MAX_ROWS = @import("hc_fused.zig").HC_FUSED_MAX_ROWS;
const HcFusedOut = @import("hc_fused.zig").HcFusedOut;
const HcPending = @import("hc_fused.zig").HcPending;
const HcWeights = @import("weights.zig").HcWeights;
const PleWeights = @import("weights.zig").PleWeights;

/// qwen4_exp: compile the four hyper-connection elementwise tails. An EXL3 trunk pack's load also says whether its rows
/// take the prefill kernels.
pub fn compileQwen4Hc(self: *Trunk) void {
    const specs = .{
        .{ &self.compiled_hc_silu, &hcSiluCallback, true },
        .{ &self.compiled_hc_mix, &hcMixCallback, false },
        .{ &self.compiled_hc_inj, &hcInjCallback, true },
        .{ &self.compiled_hc_write, &hcWriteCallback, true },
    };
    var n: usize = 0;
    inline for (specs) |sp| {
        const raw = mlx.mlx_closure_new_func_payload(sp[1], @ptrCast(self), null);
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw, sp[2]);
        _ = mlx.mlx_closure_free(raw);
        if (rc == 0 and compiled.ctx != null) {
            sp[0].* = compiled;
            n += 1;
        }
    }
    if (n > 0) log.info("[qwen4] hyper-connection tails compiled ({d}/4 kernels)\n", .{n});
    if (self.config.trunk_format and self.moe_layers.len > 0) if (self.moe_layers[0].hc_attn) |*w| @import("load.zig").sayHcPrefillAtLoad(w);
}

pub fn hcSiluCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const x = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    const y = self.silu(x) catch return -1;
    return closureOut(res, y);
}

pub fn hcMixCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const up4 = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(up4);
    const n4 = closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(n4);
    var mix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mix);
    if (mlx.mlx_sigmoid(&mix, up4, self.s) != 0) return -1;
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    if (mlx.mlx_multiply(&prod, mix, n4, self.s) != 0) return -1;
    var mixed = mlx.mlx_array_new();
    if (mlx.mlx_mean_axis(&mixed, prod, 2, false, self.s) != 0) return -1;
    return closureOut(res, mixed);
}

pub fn hcInjCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const x = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    if (mlx.mlx_sigmoid(&sig, x, self.s) != 0) return -1;
    const two = scalarOf(2.0, mlx.mlx_array_dtype(x), self.s) catch return -1;
    defer _ = mlx.mlx_array_free(two);
    var g = mlx.mlx_array_new();
    if (mlx.mlx_multiply(&g, sig, two, self.s) != 0) return -1;
    return closureOut(res, g);
}

pub fn hcWriteCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const stream4 = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(stream4);
    const out4 = closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(out4);
    const inj = closureIn(input, 2) orelse return -1;
    defer _ = mlx.mlx_array_free(inj);
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    if (mlx.mlx_multiply(&add4, out4, inj, self.s) != 0) return -1;
    var next = mlx.mlx_array_new();
    if (mlx.mlx_add(&next, stream4, add4, self.s) != 0) return -1;
    return closureOut(res, next);
}

// ── MoE forward pass (Qwen3.5 + Gemma 4) ──

// ── qwen4_exp (Qwen3.8-Flash-Next) ──

pub const HcRead = struct {
    mixed: mlx.mlx_array, // [B, S, hidden]
    inj: mlx.mlx_array, // [B, S, hc, 1] (null-ctx on the mixer)
    pub fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.mixed);
        if (self.inj.ctx != null) _ = mlx.mlx_array_free(self.inj);
    }
};

// ── tests ──

const testing = std.testing;
