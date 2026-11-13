//! The streamed bank's slot rows from the bytes the host leaves it (`LoadCtx.stream_room` past the residents, the
//! `BankBudget` mlx-serve's glm5_next hands `bank_rows` and `open_bank`), and the bank's switches. MLX-free.

const std = @import("std");

/// The streamer's transient rows (`mimo/stream.zig` `transient_rows`: wide_depth 2 x 48 route ids), which the prompt
/// phase leaves beside the slot rows; plugin.zig checks it against the streamer at compile time.
pub const streamer_transient_rows: u32 = 96;

pub const Plan = struct { prompt: u32, decode: u32 };

/// `rowsFor` with the decode phase's own free bytes (`decode_free`; `free` itself gives `rowsFor`'s rows exactly).
pub fn rowsForPhases(free: u64, decode_free: u64, record: u64, n_moe: u32, transient: u32, top_k: u32, n_experts: u32) Plan {
    const row = @max(record * n_moe, 1);
    const lo: u64 = @max(top_k, 1);
    const decode = std.math.clamp(decode_free / row, lo, n_experts);
    const prompt = std.math.clamp((free -| @as(u64, transient) * record) / row, lo, decode);
    return .{ .prompt = @intCast(prompt), .decode = @intCast(decode) };
}

/// The streamer's decode window 0 (`mimo/stream.zig` `decode_lane_route_ids`: the decode lane's 4 rows x top-8), which the
/// phase change allocates per geometry once the prompt's transient rows are freed; plugin.zig checks it against the
/// streamer at compile time.
pub const streamer_window_rows: u32 = 32;

/// The bank's per-layer split of a plan (`Glm5n.planRows`: each layer's prompt and decode rows, as the bank's open makes
/// them).
pub const SplitFn = *const fn (plan: Plan, prompt: []u32, decode: []u32) anyerror!void;

/// The per-layer bill (`TURBOEXL3_GLM53F_LAYER_BILL`): the rows of `layer_bytes` (each routed layer's slot-row bytes,
/// `bank_layers.read`) that fit `free` in the prompt phase and `decode_free` in the decode phase, where `rowsFor` counts
/// the widest row in every layer. The prompt phase holds its uniform rows and the `transient` rows of each of the bank's
/// geometries (the widest layer's and, on a two-geometry bank, the other's); the decode phase holds the split's rows
/// (`split` into `prompt_rows` / `decode_rows`, its bytes summed layer by layer) and `window` rows of each geometry. At
/// least `top_k` rows a layer, at most `n_experts`, and the prompt rows never above the decode rows.
pub fn rowsByLayer(free: u64, decode_free: u64, layer_bytes: []const u64, transient: u32, window: u32, top_k: u32, n_experts: u32, split: SplitFn, prompt_rows: []u32, decode_rows: []u32) !Plan {
    var widest: u64 = 0;
    var sum: u64 = 0;
    for (layer_bytes) |b| {
        widest = @max(widest, b);
        sum +|= b;
    }
    var alt: u64 = 0;
    for (layer_bytes) |b| if (b != widest) {
        alt = @max(alt, b);
    };
    const geo_row = widest + alt;
    const lo: u32 = @max(top_k, 1);
    const row = @max(sum, 1);
    const p_room = free -| @as(u64, transient) * geo_row;
    const d_room = decode_free -| @as(u64, window) * geo_row;
    var d: u32 = @intCast(std.math.clamp(d_room / row, lo, n_experts));
    var p: u32 = @min(d, @as(u32, @intCast(std.math.clamp(p_room / row, lo, n_experts))));
    while (true) {
        try split(.{ .prompt = p, .decode = d }, prompt_rows, decode_rows);
        var bytes: u64 = 0;
        for (decode_rows, layer_bytes) |r, b| bytes +|= @as(u64, r) * b;
        if (bytes <= d_room or d == lo) break;
        d -= 1;
        p = @min(p, d);
    }
    return .{ .prompt = p, .decode = d };
}

/// The bank's bytes each phase as a plan bills them (what the budget pays for): the slot rows and the streamer's
/// transient rows (prompt) or decode window rows (decode). A `bank-fit.json` base is a phase's measured peak less these.
pub const Billed = struct { prompt: u64, decode: u64 };

/// `rowsForPhases`' bill of `p`: every layer at the widest row, the transient rows in the prompt phase.
pub fn billedUniform(p: Plan, record: u64, n_moe: u32, transient: u32) Billed {
    const row = record * n_moe;
    return .{ .prompt = @as(u64, p.prompt) * row + @as(u64, transient) * record, .decode = @as(u64, p.decode) * row };
}

/// `rowsByLayer`'s bill of `p` and its split's `decode_rows`: each layer's own row bytes, with `transient` (prompt) or
/// `window` (decode) rows of each geometry.
pub fn billedByLayer(p: Plan, decode_rows: []const u32, layer_bytes: []const u64, transient: u32, window: u32) Billed {
    var widest: u64 = 0;
    var sum: u64 = 0;
    for (layer_bytes) |b| {
        widest = @max(widest, b);
        sum +|= b;
    }
    var alt: u64 = 0;
    for (layer_bytes) |b| if (b != widest) {
        alt = @max(alt, b);
    };
    var decode: u64 = @as(u64, window) * (widest + alt);
    for (decode_rows, layer_bytes) |r, b| decode +|= @as(u64, r) * b;
    return .{ .prompt = @as(u64, p.prompt) * sum + @as(u64, transient) * (widest + alt), .decode = decode };
}

/// The decode lookahead's records read ahead a layer call (`mimo/stream_source.zig` `lookahead.budget`); plugin.zig
/// checks it against the streamer at compile time.
pub const streamer_spec_budget: u32 = 2;

/// The read pool's workers (`mimo/stream.zig` `io_workers`) and each worker's staging buffer (`glm5n/settings.zig`
/// `reader_staging_bytes`); plugin.zig checks both against the streamer at compile time.
pub const streamer_io_workers: u32 = 4;
pub const streamer_reader_staging_bytes: u64 = 16 << 20;

/// The bank's switches, read at its open (`parseSwitch`).
pub const Switches = struct {
    /// `TURBOEXL3_GLM53F_LAYER_BILL=1`: the slot rows from each layer's own row bytes (`rowsByLayer`).
    layer_bill: bool = false,
    /// `TURBOEXL3_GLM53F_GROW_UNFILLED=1`: the decode grow's new rows without the zero fill (the stream's
    /// `GrowFill.unfilled`); no row count changes.
    grow_unfilled: bool = false,
    /// `TURBOEXL3_GLM53F_PROMPT_KEEP=1`: no prompt-phase admission evicts a resident (the stream's
    /// `Options.prompt_keep`); no row count changes.
    prompt_keep: bool = false,
    /// `TURBOEXL3_GLM53F_SPEC_PICK1=1`: the decode lookahead reads ahead one record a layer call, the selector's first
    /// non-resident prediction, not 2 (the stream's `Lookahead.picks` 1); the read pool keeps its slots, threads and
    /// bytes, so no row count changes.
    spec_pick1: bool = false,
    /// `TURBOEXL3_GLM53F_DEMAND_HELP=2` (`parseDemandHelp`): the read pool's demand-first mode in decode (the kit
    /// stream's `demandFirst(2)`, the read pool's `q3ld_demand_first`, set at each decode phase and back to 0 at each
    /// prompt phase): an unclaimed lookahead record reads only while no demand job is queued or running, in pieces, and
    /// a demand range that finds its record claimed and still in flight reads its untaken pieces itself; reads only; no
    /// row count changes. 0: off.
    demand_first: u32 = 0,
    /// `TURBOEXL3_GLM53F_ADOPT_PIECES=1`: a range adopted from a lookahead record is copied into its rows in as many
    /// pieces as the read pool has workers, the idle workers taking some (the stream's `Lookahead.adopt_pieces`, the
    /// read pool's `q3ld_adopt_pieces`): the same bytes into the same rows; no row count changes.
    adopt_pieces: bool = false,
    /// `TURBOEXL3_GLM53F_SEED_DECODE=1`: the resident seed reads the pack's `expert-usage-decode-v1.json` in place of
    /// `expert-usage-v1.json` (`glm5n_source.seedDecision`); only which experts the prompt rows start with moves.
    seed_decode: bool = false,
    /// `TURBOEXL3_GLM53F_KEEP_WARM=1`: the keep-warm engine on the stream's waits (`gpu_warm` at `.waits`, glm5n_source.zig
    /// `keep_warm`): its own Metal queue keeps the GPU busy while a gated wait or a read-ahead host
    /// wait is pending; nothing a model array reads changes. Its bill term (`glm5n_source.keep_warm_bill_bytes`, 1 MiB) comes
    /// off the bytes the rows are sized from.
    keep_warm: bool = false,
};

/// A switch's value: unset, empty or `0` off; `1` on; anything else refused by name.
pub fn parseSwitch(v: ?[]const u8) error{Glm53fSwitchValue}!bool {
    const s = v orelse return false;
    if (s.len == 0 or std.mem.eql(u8, s, "0")) return false;
    if (std.mem.eql(u8, s, "1")) return true;
    return error.Glm53fSwitchValue;
}

/// `TURBOEXL3_GLM53F_DEMAND_HELP`'s value: unset, empty or `0` off; `2` the read pool's demand first with the demand's
/// help (`Switches.demand_first`); anything else, `1` among it, refused by name.
pub fn parseDemandHelp(v: ?[]const u8) error{Glm53fSwitchValue}!u2 {
    if (v) |s| if (std.mem.eql(u8, s, "2")) return 2;
    if (try parseSwitch(v)) return error.Glm53fSwitchValue;
    return 0;
}


// ── tests ──

const testing = std.testing;

fn sc310Bytes() [42]u64 {
    var b: [42]u64 = @splat(9_474_048);
    b[0] = 12_619_776;
    return b;
}

/// A split that keeps every layer at the plan's rows (the bank's own split is `Glm5n.planRows`, tested plugin-rooted).
fn uniformSplit(plan: Plan, prompt: []u32, decode: []u32) anyerror!void {
    @memset(prompt, plan.prompt);
    @memset(decode, plan.decode);
}

/// g53smd's bank open (run 1): the plugin's own room at the 102,400 MiB ceiling and the 2 GiB margin, 28,123 MiB in use.
const g53smd_free: u64 = (102_400 << 20) - 21_368_258_560 - (2 << 30) - (28_123 << 20);

test "turboexl3_glm53f bank rows: the widest-row bill keeps top_k rows, at most every expert, the prompt within decode" {
    const record: u64 = 12_619_776;
    // g53smd's rows (its load line: 100 prompt / 102 decode)
    try testing.expectEqual(Plan{ .prompt = 100, .decode = 102 }, rowsForPhases(g53smd_free, g53smd_free, record, 42, streamer_transient_rows, 8, 288));
    try testing.expectEqual(Plan{ .prompt = 8, .decode = 8 }, rowsForPhases(0, 0, record, 42, streamer_transient_rows, 8, 288));
    try testing.expectEqual(Plan{ .prompt = 288, .decode = 288 }, rowsForPhases(1 << 50, 1 << 50, record, 42, streamer_transient_rows, 8, 288));
}

test "turboexl3_glm53f bank rows: the per-layer bill fits each phase's bytes with the K4 layer's and both geometries' transient rows counted" {
    const bytes = sc310Bytes();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    const geo: u64 = 12_619_776 + 9_474_048;
    const sum: u64 = 12_619_776 + 41 * 9_474_048;
    const plan = try rowsByLayer(g53smd_free, g53smd_free, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 130, .decode = 133 }, plan);
    try testing.expect(@as(u64, plan.prompt) * sum + 96 * geo <= g53smd_free);
    try testing.expect(@as(u64, plan.decode + 1) * sum + 32 * geo > g53smd_free);
    try testing.expectEqual(Plan{ .prompt = 8, .decode = 8 }, try rowsByLayer(0, 0, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr));
    // the plan's bill: each phase within its bytes, one more decode row a layer past them
    _ = try rowsByLayer(g53smd_free, g53smd_free, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    const billed = billedByLayer(plan, &dr, &bytes, streamer_transient_rows, streamer_window_rows);
    try testing.expectEqual(@as(u64, 130) * sum + 96 * geo, billed.prompt);
    try testing.expectEqual(@as(u64, 133) * sum + 32 * geo, billed.decode);
    try testing.expect(billed.decode <= g53smd_free and billed.decode + sum > g53smd_free);
    const small = try rowsByLayer(60 << 30, 30 << 30, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    try testing.expect(small.prompt <= small.decode);
}

test "turboexl3_glm53f bank rows: the widest-row bill is its rows at the widest row, the transient rows in the prompt phase" {
    const record: u64 = 12_619_776;
    const b = billedUniform(.{ .prompt = 100, .decode = 102 }, record, 42, streamer_transient_rows);
    try testing.expectEqual(@as(u64, 100 * 42 + 96) * record, b.prompt);
    try testing.expectEqual(@as(u64, 102 * 42) * record, b.decode);
    try testing.expect(b.prompt <= g53smd_free and b.decode <= g53smd_free);
}

test "turboexl3_glm53f bank rows: a switch is on at 1, off unset, empty or 0, and any other value is refused by name" {
    try testing.expect(!try parseSwitch(null));
    try testing.expect(!try parseSwitch("0"));
    try testing.expect(try parseSwitch("1"));
    try testing.expectError(error.Glm53fSwitchValue, parseSwitch("2"));
    try testing.expectEqual(@as(u2, 2), try parseDemandHelp("2"));
    try testing.expectError(error.Glm53fSwitchValue, parseDemandHelp("1"));
}
