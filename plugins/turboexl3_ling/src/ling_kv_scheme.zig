//! KV_SCHEME_DENSE_COPIES: a warning, once a load, the first time an MLA append's view shows a quantized KV cache
//! (`--kv-quant`, a model's `kv_quant` setting or a request's `kv_quant`). It refuses nothing and changes no byte; on a
//! dense cache it costs one branch on the view's own `bits` (`atView`). MLX-free.
//!
//! Why a warning and not a refusal, on mlx-serve 66aad949:
//! - Each request's cache takes the request's `kv_quant`, else the model's setting, else `--kv-quant`
//!   (src/engines/mlx/scheduler.zig:515-516).
//! - A quantized cache's append hands back dense dequantized K and V, which the view owns, beside the stored triples
//!   (src/sdk/kv/cache.zig:435-452); the view's `bits` is the triples' width, 0 on a dense cache
//!   (src/sdk/kv/cache_ops.zig:85-101). The copied trunk reads only the dense K and V: each call, one rope-once MLA
//!   layer's bf16 K and V of every cached row (src/bill.zig `ropeOnceLayer`), which no term of the bill holds.
//! - The load is never told the scheme: the host's one call of `bill` passes no `Shape.kv_bits`
//!   (src/engines/mlx/plugin/driver.zig:177-187), and the load settings carry `kv_quant` only to a plugin that owns
//!   its decode state (driver.zig:138-141), which this one does not.
//! - So this view, mid-forward, is the first sight of the scheme. A forward refused there would reach the client as
//!   HTTP 500 "generation failed", and mlx-serve's own rule gates a KV scheme at load instead
//!   (src/engines/mlx/AGENTS.md:115). A refusal by name needs a host change.

const std = @import("std");
const log = @import("mlx_serve/sdk/util/log.zig");

/// The name the warning goes by.
pub const name = "KV_SCHEME_DENSE_COPIES";

/// What the warning prices: one MLA layer's dense K/V copies at `tokens` cached tokens.
pub const Plan = struct { tokens: u64 = 0, layer_bytes: u64 = 0 };

/// Whether this load has warned (plugin.zig `load` clears it).
pub var warned = false;
/// This load's price (plugin.zig `load`: the bill's context, one rope-once layer's bf16 K and V at it).
pub var load_plan: Plan = .{};
var line_buf: [512]u8 = undefined;

/// The served call at the rope-once append's view (the copied mla.zig `rope1Attn`, ling-mla.patch). `warning`, inlined
/// with it, branches once on `bits`, a field of the view in hand: a dense cache's view (0) costs that one branch. The
/// text and the line come only at a quantized cache's first view of the load.
pub inline fn atView(bits: u8) void {
    if (warning(&line_buf, &warned, bits, &load_plan)) |w| log.warn("[turboexl3_ling] {s}\n", .{w});
}

/// The decision at an MLA append's view: the warning's text (into `buf`) the first time a load sees a quantized cache
/// (`bits` 8 or 4), else null: a dense cache's view (`bits` 0), or any view once the load has warned (`said`).
pub inline fn warning(buf: []u8, said: *bool, bits: u8, plan: *const Plan) ?[]const u8 {
    if (bits == 0 or said.*) return null;
    said.* = true;
    return std.fmt.bufPrint(buf, warning_fmt, .{ bits, plan.layer_bytes >> 20, plan.tokens }) catch warning_short;
}

/// The warning: the scheme's bits, one layer's dense copies in MiB, and the cached tokens they are priced at.
const warning_fmt = name ++ ": the KV cache is quantized at {d} bits (--kv-quant, the model's kv_quant setting " ++
    "or a request's kv_quant). Each MLA layer reads the dense dequantized K/V copies mlx-serve makes at every " ++
    "append: {d} MiB a layer a call at {d} cached tokens (calculated), which no term of this plugin's bill holds. " ++
    "Refusing the scheme needs a host change; keep --kv-quant off and send no kv_quant.";
/// The warning when `buf` cannot hold it.
const warning_short = name ++ ": the KV cache is quantized; keep --kv-quant off and send no kv_quant.";

// ── tests ──

const testing = std.testing;

/// Ling-3.1-flash's copies at the bill's 32,768 tokens: kv_b's rows (64 heads x (128 + 128)) and the 64-wide roped key,
/// bf16, a token.
const ling_plan: Plan = .{ .tokens = 32768, .layer_bytes = 32768 * (64 * (128 + 128) + 64) * 2 };

test "turboexl3_ling KV_SCHEME_DENSE_COPIES: a dense cache's view (bits 0) gives no warning, however often it is seen" {
    var buf: [512]u8 = undefined;
    var said = false;
    for (0..3) |_| try testing.expect(warning(&buf, &said, 0, &ling_plan) == null);
    // a dense view uses up nothing: the load's first quantized view still warns
    try testing.expect(!said);
}

test "turboexl3_ling KV_SCHEME_DENSE_COPIES: an 8-bit and a 4-bit cache's view each give one warning, which names its bits and the dense copies' 1028 MiB a layer at 32768 tokens" {
    var buf: [512]u8 = undefined;
    for ([_]u8{ 8, 4 }, [_][]const u8{ "quantized at 8 bits", "quantized at 4 bits" }) |bits, named| {
        // each a load of its own
        var said = false;
        const w = warning(&buf, &said, bits, &ling_plan) orelse return error.TestExpectedWarning;
        try testing.expect(std.mem.startsWith(u8, w, name ++ ": "));
        try testing.expect(std.mem.indexOf(u8, w, named) != null);
        try testing.expect(std.mem.indexOf(u8, w, "1028 MiB a layer a call at 32768 cached tokens") != null);
        try testing.expect(said);
    }
}

test "turboexl3_ling KV_SCHEME_DENSE_COPIES: once a load has warned, no later view repeats it, whatever its bits" {
    var buf: [512]u8 = undefined;
    var said = false;
    try testing.expect(warning(&buf, &said, 8, &ling_plan) != null);
    for ([_]u8{ 8, 4, 0, 8 }) |bits| try testing.expect(warning(&buf, &said, bits, &ling_plan) == null);
}
