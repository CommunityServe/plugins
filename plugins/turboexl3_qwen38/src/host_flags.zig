//! The server's own launch flags this plugin follows, read off the process's arguments at load (`_NSGetArgv`), and the
//! model's `kv_quant` in the server's per-model settings: no load setting carries them. `known` has one line per flag,
//! and each value parses by the server's own rules, so the plugin takes exactly what the server takes. The server exits
//! on a value it refuses; one that still reaches the plugin refuses the load by name.

const std = @import("std");
const depth_bounds = @import("mlx_serve/engines/mlx/spec/mtp_depth_bounds.zig");
const mtp_mod = @import("mlx_serve/sdk/spec/mtp/mtp.zig");
const kv_quant = @import("mlx_serve/sdk/blocks/attention/kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const log = @import("mlx_serve/sdk/util/log.zig");
const attn_core = @import("direct/attn_core.zig");

/// `--kv-attn-mode`: how a request reads an affine cache below the QSA budget.
pub const KvAttnMode = enum { dense, fused, auto };

/// What the flags say. A depth or KV flag the process does not name is null; every other field then holds the server's
/// default.
pub const Flags = struct {
    mtp_min_depth: ?u32 = null,
    mtp_max_depth: ?u32 = null,
    /// `--max-mtp-ctx`: no round drafts past this many context tokens; 0 = no ceiling.
    max_mtp_ctx: u32 = 0,
    /// `--mtp-history-window`: a long prefill's head history covers its last this many positions; 0 = all of them.
    mtp_history_window: usize = 0,
    /// `--warmup-eager` / `--no-warmup-eager`: the server's startup load warms (`warmsAt`).
    warmup_eager: bool = true,
    /// `--model`: the model the server loads at its start, the one load the warm-up flags reach.
    model: ?[]const u8 = null,
    /// `--int8-qk`: the dense prefill attention on int8 Q and K.
    int8_qk: bool = false,
    kv_quant: ?KVQuantConfig = null,
    kv_attn_mode: ?KvAttnMode = null,
    /// `--mtp-head-kv-quant`: the MTP head's KV takes the trunk's scheme (else dense).
    mtp_head_kv_quant: bool = false,

    /// The draft-depth range as the server's configure sets it from `--mtp-min-depth` and `--mtp-max-depth`, checked
    /// as its main checks it; null when neither is named.
    pub fn depthBounds(f: Flags) error{MinAboveMax}!?depth_bounds.Bounds {
        if (f.mtp_min_depth == null and f.mtp_max_depth == null) return null;
        const b: depth_bounds.Bounds = .{ .min = f.mtp_min_depth orelse 1, .max = f.mtp_max_depth orelse 0 };
        try depth_bounds.validate(b);
        return b;
    }
};

/// The flags this plugin reads, one line each: the server's spelling, the `Flags` field it sets, and how its value
/// parses (`value`), or, for a switch, which takes no value, what naming it sets (`set`). A flag named twice keeps its
/// last value, as the server's parse does.
const known = .{
    .{ .name = "--mtp-min-depth", .field = "mtp_min_depth", .value = depth },
    .{ .name = "--mtp-max-depth", .field = "mtp_max_depth", .value = depth },
    .{ .name = "--mtp-depth", .field = "mtp_max_depth", .value = oldDepth },
    .{ .name = "--max-mtp-ctx", .field = "max_mtp_ctx", .value = tokens },
    .{ .name = "--mtp-history-window", .field = "mtp_history_window", .value = positions },
    .{ .name = "--warmup-eager", .field = "warmup_eager", .set = true },
    .{ .name = "--no-warmup-eager", .field = "warmup_eager", .set = false },
    .{ .name = "--model", .field = "model", .value = modelDir },
    .{ .name = "--int8-qk", .field = "int8_qk", .set = true },
    .{ .name = "--kv-quant", .field = "kv_quant", .value = kvQuant },
    .{ .name = "--kv-attn-mode", .field = "kv_attn_mode", .value = kvAttnMode },
    .{ .name = "--mtp-head-kv-quant", .field = "mtp_head_kv_quant", .set = true },
};

/// `--kv-quant`: off (or 0), else affine group quantization at 4 or 8 bits.
fn kvQuant(text: []const u8) error{NotAKvQuant}!KVQuantConfig {
    if (std.mem.eql(u8, text, "off") or std.mem.eql(u8, text, "0")) return KVQuantConfig.dense;
    if (std.mem.eql(u8, text, "4")) return KVQuantConfig.affine(4);
    if (std.mem.eql(u8, text, "8")) return KVQuantConfig.affine(8);
    return error.NotAKvQuant;
}

fn kvAttnMode(text: []const u8) error{NotAKvAttnMode}!KvAttnMode {
    return std.meta.stringToEnum(KvAttnMode, text) orelse error.NotAKvAttnMode;
}

/// A draft depth: an integer in 1..MAX_DEPTH.
fn depth(text: []const u8) error{NotADepth}!u32 {
    return depth_bounds.parseDepth(text, mtp_mod.MAX_DEPTH);
}

/// `--mtp-max-depth`'s deprecated spelling `--mtp-depth`: any integer, clamped into 1..MAX_DEPTH.
fn oldDepth(text: []const u8) error{NotADepth}!u32 {
    return depth_bounds.parseClamped(text, mtp_mod.MAX_DEPTH);
}

/// `--max-mtp-ctx`'s token count, as the server's main reads it.
fn tokens(text: []const u8) std.fmt.ParseIntError!u32 {
    return std.fmt.parseInt(u32, text, 10);
}

/// `--mtp-history-window`'s position count, as the server's main reads it.
fn positions(text: []const u8) std.fmt.ParseIntError!usize {
    return std.fmt.parseInt(usize, text, 10);
}

/// `--model`'s directory, as the server's main takes it: the text itself, which its startup load hands the plugin.
fn modelDir(text: []const u8) error{}![]const u8 {
    return text;
}

/// A flag whose value the server refuses, and that value.
pub const Bad = struct { flag: []const u8 = "", value: []const u8 = "" };

/// The flags in `args` (`argv[0]` first). A flag's value is the argument after it, as the server reads it; a switch
/// takes none.
pub fn parse(args: []const [*:0]const u8, bad: *Bad) !Flags {
    var out: Flags = .{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.span(args[i]);
        inline for (known) |k| if (std.mem.eql(u8, arg, k.name)) {
            if (@hasField(@TypeOf(k), "set")) {
                @field(out, k.field) = k.set;
            } else {
                i += 1;
                const text: []const u8 = if (i < args.len) std.mem.span(args[i]) else "";
                @field(out, k.field) = k.value(text) catch |e| {
                    bad.* = .{ .flag = k.name, .value = text };
                    return e;
                };
            }
            break;
        };
    }
    return out;
}

extern "c" fn _NSGetArgc() *const c_int;
extern "c" fn _NSGetArgv() *const [*]const [*:0]const u8;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// The flags as `apply` read them at load.
pub var flags: Flags = .{};

/// The KV scheme a load of the model at `model_dir` takes, as the server resolves it with no per-request field: the
/// model's `kv_quant` in `~/.mlx-serve/model-settings.json`, else `--kv-quant`, else dense.
pub fn kvQuantFor(model_dir: []const u8) KVQuantConfig {
    return modelKvQuant(model_dir) orelse flags.kv_quant orelse KVQuantConfig.dense;
}

/// The model's `kv_quant` as the server reads its settings file: the entry whose path, trailing slashes trimmed, is the
/// model's. A missing or malformed file, or a value the server takes as unset, gives none.
fn modelKvQuant(model_dir: []const u8) ?KVQuantConfig {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/.mlx-serve/model-settings.json", .{std.mem.span(getenv("HOME") orelse "/tmp")}) catch return null;
    const io = std.Io.Threaded.global_single_threaded.io();
    const body = std.Io.Dir.cwd().readFileAlloc(io, path, std.heap.c_allocator, .limited(1 << 20)) catch return null;
    defer std.heap.c_allocator.free(body);
    const parsed = std.json.parseFromSlice(std.json.Value, std.heap.c_allocator, body, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    var it = parsed.value.object.iterator();
    while (it.next()) |e| if (std.mem.eql(u8, trimSlash(e.key_ptr.*), trimSlash(model_dir))) {
        if (e.value_ptr.* != .object) return null;
        return KVQuantConfig.fromJsonValue(e.value_ptr.object.get("kv_quant") orelse return null);
    };
    return null;
}

fn trimSlash(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 1 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    return p;
}

/// Whether a request of `prompt_len` tokens on a cache of `scheme` reads it fused below the QSA budget, as the server
/// resolves it with no per-request field: `--kv-attn-mode`, whose `auto` (the default) reads an affine cache fused from
/// 8192 prompt tokens.
pub fn kvAttnFused(scheme: kv_quant.Scheme, prompt_len: usize) bool {
    return switch (flags.kv_attn_mode orelse .auto) {
        .dense => false,
        .fused => true,
        .auto => scheme == .affine and prompt_len >= 8192,
    };
}

/// What the server's flags set in this plugin: the MTP round policy's depth range (`depth_bounds.active`), with one line
/// when a depth flag is named, and the dense prefill attention's int8 Q/K (`attn_core.int8_qk`), with one line when
/// `--int8-qk` is; the rest are kept in `flags` and returned for the load, with one line for each that changes what the
/// plugin does. A value the server refuses fails here by name, with a line naming the flag.
pub fn apply() !Flags {
    var bad: Bad = .{};
    const f = parse(_NSGetArgv().*[0..@intCast(_NSGetArgc().*)], &bad) catch |e| {
        log.err("[turboexl3_qwen38] {s} '{s}' is not a value the server takes ({s})\n", .{ bad.flag, bad.value, @errorName(e) });
        return e;
    };
    const bounds = f.depthBounds() catch |e| {
        log.err("[turboexl3_qwen38] --mtp-min-depth {d} is above --mtp-max-depth {d}\n", .{ f.mtp_min_depth.?, f.mtp_max_depth.? });
        return e;
    };
    flags = f;
    attn_core.int8_qk = f.int8_qk;
    if (f.int8_qk) log.info("[turboexl3_qwen38] --int8-qk: a dense prefill block of 64 rows or more attends on int8 Q and K (lossy)\n", .{});
    if (bounds) |b| {
        depth_bounds.active = b;
        if (b.pinned()) |n| {
            log.info("[turboexl3_qwen38] MTP depth pinned to {d}\n", .{n});
        } else if (b.max != 0) {
            log.info("[turboexl3_qwen38] MTP depth bounded {d}..{d}\n", .{ b.min, b.max });
        } else {
            log.info("[turboexl3_qwen38] MTP depth bounded {d}..its adaptive cap\n", .{b.min});
        }
    }
    if (f.max_mtp_ctx != 0) log.info("[turboexl3_qwen38] MTP stops drafting past {d} context tokens\n", .{f.max_mtp_ctx});
    if (f.mtp_history_window != 0) log.info("[turboexl3_qwen38] MTP head history: the last {d} positions of a prefill past {d}\n", .{ f.mtp_history_window, mtp_mod.HISTORY_WINDOW_THRESHOLD });
    return f;
}

/// Whether the server's eager warm-up reaches this plugin's load of `model_dir`, as the server decides it: its startup
/// load (this process's first, of the model `--model` names) warms unless `--no-warmup-eager`, the one load its main
/// hands the flag; every other load (on demand, a headless server's) warms.
pub fn warmsAt(f: Flags, model_dir: []const u8) bool {
    defer loaded = true;
    const startup = !loaded and f.model != null and std.mem.eql(u8, f.model.?, model_dir);
    return f.warmup_eager or !startup;
}

/// This process has loaded the plugin before (`warmsAt`): any later load is one the server makes on demand.
var loaded = false;

test "host flags: the depth, KV and --int8-qk flags take what the server takes" {
    const testing = std.testing;
    var bad: Bad = .{};
    // None named: no bounds, no int8 Q/K. `serve` comes first, as the server's subcommand does.
    const plain = try parse(&.{ "mlx-serve", "serve", "--model", "/m", "--no-mtp" }, &bad);
    try testing.expect(try plain.depthBounds() == null);
    try testing.expect(!plain.int8_qk);
    // `--int8-qk` is a switch: it takes no value, so the flag after it is still read.
    const q8 = try parse(&.{ "mlx-serve", "serve", "--int8-qk", "--mtp-max-depth", "3" }, &bad);
    try testing.expect(q8.int8_qk);
    try testing.expectEqual(@as(?u32, 3), q8.mtp_max_depth);
    // `--kv-quant` takes off, 0, 4 or 8, `--kv-attn-mode` dense, fused or auto; `--mtp-head-kv-quant` is a switch.
    const kv = try parse(&.{ "mlx-serve", "--kv-quant", "8", "--mtp-head-kv-quant", "--kv-attn-mode", "dense" }, &bad);
    try testing.expectEqual(KVQuantConfig.affine(8), kv.kv_quant.?);
    try testing.expect(kv.mtp_head_kv_quant and kv.kv_attn_mode.? == .dense);
    try testing.expectEqual(KVQuantConfig.dense, (try parse(&.{ "mlx-serve", "--kv-quant", "0" }, &bad)).kv_quant.?);
    try testing.expectError(error.NotAKvQuant, parse(&.{ "mlx-serve", "--kv-quant", "16" }, &bad));
    try testing.expectError(error.NotAKvAttnMode, parse(&.{ "mlx-serve", "--kv-attn-mode", "packed" }, &bad));
    // Equal flags pin a depth.
    const pin = try (try parse(&.{ "mlx-serve", "--mtp-min-depth", "4", "--mtp-max-depth", "4" }, &bad)).depthBounds();
    try testing.expectEqual(@as(?u32, 4), pin.?.pinned());
    // `--mtp-depth` is the max, any integer clamped into 1..MAX_DEPTH, and the flag named last wins.
    const old = try (try parse(&.{ "mlx-serve", "--mtp-max-depth", "3", "--mtp-depth", "12" }, &bad)).depthBounds();
    try testing.expectEqual(depth_bounds.Bounds{ .max = mtp_mod.MAX_DEPTH }, old.?);
    // What the server refuses: a depth outside 1..MAX_DEPTH, a flag without a value, a min above the max.
    try testing.expectError(error.NotADepth, parse(&.{ "mlx-serve", "--mtp-min-depth", "9" }, &bad));
    try testing.expectEqualStrings("--mtp-min-depth", bad.flag);
    try testing.expectError(error.NotADepth, parse(&.{ "mlx-serve", "--mtp-max-depth" }, &bad));
    try testing.expectError(error.MinAboveMax, (try parse(&.{ "mlx-serve", "--mtp-min-depth", "5", "--mtp-max-depth", "4" }, &bad)).depthBounds());
}

test "host flags: the context ceiling, the history window and the warm-up take what the server takes" {
    const testing = std.testing;
    var bad: Bad = .{};
    // None named: the server's defaults.
    try testing.expectEqual(Flags{}, try parse(&.{ "mlx-serve", "serve", "--port", "8080" }, &bad));
    // A warm-up flag takes no value, so the flag after it still parses; the one named last wins.
    const f = try parse(&.{ "mlx-serve", "--no-warmup-eager", "--max-mtp-ctx", "32768", "--mtp-history-window", "8192" }, &bad);
    try testing.expectEqual(Flags{ .max_mtp_ctx = 32768, .mtp_history_window = 8192, .warmup_eager = false }, f);
    try testing.expect((try parse(&.{ "mlx-serve", "--no-warmup-eager", "--warmup-eager" }, &bad)).warmup_eager);
    try testing.expectEqualStrings("/m/k3", (try parse(&.{ "mlx-serve", "--model", "/m/k3", "--no-warmup-eager" }, &bad)).model.?);
    // What the server refuses: a count that is not one, one past its type, a flag without a value.
    try testing.expectError(error.InvalidCharacter, parse(&.{ "mlx-serve", "--max-mtp-ctx", "32k" }, &bad));
    try testing.expectEqualStrings("--max-mtp-ctx", bad.flag);
    try testing.expectError(error.Overflow, parse(&.{ "mlx-serve", "--max-mtp-ctx", "4294967296" }, &bad));
    try testing.expectError(error.InvalidCharacter, parse(&.{ "mlx-serve", "--mtp-history-window" }, &bad));
    try testing.expectEqualStrings("--mtp-history-window", bad.flag);
}
