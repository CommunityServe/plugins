//! GLM-5.3-Flash's routed experts from its EXL3 bank (`experts.bin` + `expert-manifest-v2.json`) for the glm5_next
//! trunk: the streamed stack of record (`Exl3Source`: MiMo's stream and read schedule, GLM-shape EXL3 texts, the
//! in-place prompt combine, the decode hit pass) at its default routes, behind `Routed`. A call returns its experts'
//! weighted sum; the trunk adds the shared expert. Slot rows per phase: the prompt rows during a prompt pass, the
//! decode rows after `phase(decode)`; the pack's usage profile seeds the prompt rows at load.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("sdk_ext.zig");
const experts = @import("glm5n/experts.zig");
const settings = @import("glm5n/settings.zig");
const prompt_levers = @import("glm5n/prompt_levers.zig");
const gpu_warm = @import("mimo/gpu_warm.zig");
const expert_usage = sdk_ext.expert.usage;
const exl3_bank = @import("glm5n/exl3_bank.zig");
const expert_bank = @import("expert/bank.zig");
const mimo_stream = @import("mimo/stream.zig");

const T = mlx.mlx_array;
pub const Gate = experts.Gate;

/// The bank's records: one expert of one routed layer, `record_bytes` each.
pub const Geometry = struct { record_bytes: u64, first_moe_layer: u32, num_moe_layers: u32, num_experts: u32 };

/// A call of at most this many rows runs the decode lane; a wider one runs in the prompt phase.
pub const decode_lane_rows: u32 = experts.decode_lane_rows;

/// Rows a wide prompt call borrows past the slot rows (its in-flight windows).
pub const transient_rows: u32 = mimo_stream.transient_rows;

/// The bank in `dir` read from its manifests (no slot memory allocated).
pub fn geometryOf(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var diag: sdk.Diag = .{};
    var b = exl3_bank.Bank.open(a, io, dir, exl3_bank.flash_first_moe_layer, exl3_bank.flash, &diag) catch |e| {
        if (diag.len > 0) std.log.err("glm5_next bank: {s}", .{diag.message()});
        return e;
    };
    defer b.deinit();
    const g = b.geometry;
    return .{ .record_bytes = g.record_bytes, .first_moe_layer = g.first_moe_layer, .num_moe_layers = g.num_moe_layers, .num_experts = g.num_experts };
}
/// turboexl3_glm53f: the stream's construction checks `Source.open` runs (`experts.checkStreamOptions`) over the bank in
/// `dir` at `slot_rows`: its manifests and experts.bin opened as the load opens them, no record read, no slot memory,
/// read pool, event or GPU (CPU only: the plugin's test over a pack's own bank). Returns whether the served options turn
/// prompt lever 1's relocation off (the kit's stream refuses it on a bank of two geometries).
pub fn checkStreamOptions(a: std.mem.Allocator, io: std.Io, dir: []const u8, slot_rows: sdk.Rows) !bool {
    var diag: sdk.Diag = .{};
    var b = exl3_bank.Bank.open(a, io, dir, exl3_bank.flash_first_moe_layer, exl3_bank.flash, &diag) catch |e| {
        if (diag.len > 0) std.log.err("glm5_next bank: {s}", .{diag.message()});
        return e;
    };
    defer b.deinit();
    var sb = try exl3_bank.SDKBank.init(a, &b);
    defer sb.deinit();
    return experts.checkStreamOptions(a, &sb, slot_rows);
}
/// turboexl3_glm53f: the same checks over a bank of GLM-5.3-Flash's dims (42 routed layers of 288 experts) whose bank
/// layer L is at `ns[L]` halfwords a tile, built in memory (no file).
pub fn checkStreamOptionsN(a: std.mem.Allocator, ns: []const u32, slot_rows: sdk.Rows) !bool {
    const layers = try a.alloc(exl3_bank.Layer, ns.len);
    defer a.free(layers);
    const sb = try bankN(ns, layers);
    return experts.checkStreamOptions(a, &sb, slot_rows);
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_GROW_UNFILLED` for the next `Source.open` (`experts.grow_unfilled`).
pub fn setGrowUnfilled(on: bool) void {
    experts.grow_unfilled = on;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_PF_STAGE` for the next `Source.open` (`experts.stage_misses`).
pub fn setStageMisses(on: bool) void {
    experts.stage_misses = on;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_PF_DRAIN_LAG` for the next `Source.open` (`experts.drain_lag`).
pub fn setDrainLag(on: bool) void {
    experts.drain_lag = on;
}
/// turboexl3_glm53f: `experts.servedFill` over the in-memory bank of layout `ns` (as `checkStreamOptionsN`).
pub fn servedFillN(a: std.mem.Allocator, ns: []const u32, slot_rows: sdk.Rows) !experts.ServedFill {
    const layers = try a.alloc(exl3_bank.Layer, ns.len);
    defer a.free(layers);
    const sb = try bankN(ns, layers);
    return experts.servedFill(a, &sb, slot_rows);
}
/// turboexl3_glm53f: the kit's residency policy (`LayerPolicy`, `Plan`), pure CPU, for the plugin's tests.
pub const policy = sdk_ext.expert.policy;
/// turboexl3_glm53f: the bank's switches whose mechanism is a seam of the kit's stream, and whether this kit has it
/// (`experts.seams`); the first one set without its seam (`experts.missingSeam`).
pub const seams = experts.seams;
pub const missingSeam = experts.missingSeam;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_PROMPT_KEEP` for the next `Source.open` (`experts.prompt_keep`).
pub fn setPromptKeep(on: bool) void {
    experts.prompt_keep = on;
}
/// turboexl3_glm53f: the in-memory bank of layout `ns` over `layers` (`ns.len` of them, the caller's), as
/// `checkStreamOptionsN` builds it.
fn bankN(ns: []const u32, layers: []exl3_bank.Layer) !exl3_bank.SDKBank {
    const f = exl3_bank.flash;
    if (ns.len != f.n_layers or layers.len != ns.len) return error.UnsupportedTopology;
    const total = expert_bank.layerTableN(ns, f.hidden, f.inter, f.n_experts, layers) orelse return error.InvalidGeometry;
    var widest: u64 = 0;
    for (layers) |l| widest = @max(widest, l.record_bytes);
    return .{ .layers = layers, .n_experts = @intCast(f.n_experts), .first_moe_layer = exl3_bank.flash_first_moe_layer, .record_bytes = widest, .sidecar = .{ .fd = -1, .size = total } };
}
/// turboexl3_glm53f: the decode lookahead's records read ahead a layer call, the served default, for the plugin's bill
/// (src/stream_room.zig checks it at compile time).
pub const spec_budget: u32 = experts.B.lookahead.budget;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_SPEC_PICK1` for the next `Source.open` (`experts.spec_pick1`).
pub fn setSpecPick1(on: bool) void {
    experts.spec_pick1 = on;
}
/// turboexl3_glm53f: `experts.servedLookahead` over the in-memory bank of layout `ns`.
pub fn servedLookaheadN(a: std.mem.Allocator, ns: []const u32, slot_rows: sdk.Rows) !experts.ServedLookahead {
    const layers = try a.alloc(exl3_bank.Layer, ns.len);
    defer a.free(layers);
    const sb = try bankN(ns, layers);
    return experts.servedLookahead(a, &sb, slot_rows);
}
/// turboexl3_glm53f: the read pool's workers (the copy's) and each worker's staging bytes, for the plugin's bill
/// (src/stream_room.zig checks them at compile time).
pub const io_workers_default: u32 = mimo_stream.io_workers;
pub const reader_staging_bytes: u64 = settings.reader_staging_bytes;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_DEMAND_SPLIT` for the next `Source.open` (`experts.demand_split`).
pub fn setDemandSplit(on: bool) void {
    experts.demand_split = on;
}
/// turboexl3_glm53f: the pieces of a decode call's lone gate/up span at a read pool's workers, for the plugin's load line.
pub const demandSplitPieces = experts.demandSplitPieces;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_DEMAND_HELP`, the decode phase's demand-first mode (0 off, or 2) for the next
/// `Source.open` (`experts.demand_first`).
pub fn setDemandFirst(mode: u32) void {
    experts.demand_first = mode;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_ADOPT_PIECES` for the next `Source.open` (`experts.adopt_pieces`).
pub fn setAdoptPieces(on: bool) void {
    experts.adopt_pieces = on;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_SHARED_HOIST` (off unless the plugin sets it before the bank opens): the source's
/// `Routed` carries `source_hoist` (streamed.zig `glm5nRouted`), so the trunk's MoE builds a decode call's shared expert
/// before the call and the call commits it behind its hit wave when it misses (`Source.rowsHoist`).
pub var shared_hoist: bool = false;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_SHARED_HOIST` for the next `Routed` (`shared_hoist`).
pub fn setSharedHoist(on: bool) void {
    shared_hoist = on;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_READ_AHEAD3` (off unless the plugin sets it before the bank opens): the source's
/// `Routed` carries `ahead_score_fn` and `ahead_issue_fn` (streamed.zig `glm5nRouted`), so a decode window's first routed
/// layer, which no routed call before it predicts, has its likely records read ahead from the last dense layer's MLP
/// input (`Source.aheadScore`, `Source.aheadIssue`): speculative reads only, the routes and the outputs unchanged.
pub var read_ahead3: bool = false;
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_READ_AHEAD3` for the next `Routed` (`read_ahead3`).
pub fn setReadAhead3(on: bool) void {
    read_ahead3 = on;
}
/// turboexl3_glm53f: READ_AHEAD3's rows a decode window has at most (the lookahead's score rows).
pub const ahead_max_rows: usize = 8;
/// turboexl3_glm53f: the model layer of the bank's first routed layer, the one READ_AHEAD3 reads ahead (the plugin's line).
pub const first_moe_layer: u32 = exl3_bank.flash_first_moe_layer;
/// turboexl3_glm53f: the rows the stream allocates for component `c` of a slot bank of `rows` rows at `n` halfwords a
/// tile (`StreamOf.allocRows`: every slot array one VM page at least, the EXL3 bank's floor), for the plugin's MLX-free
/// tests of what its math hands turboexl3.
pub fn slotArrayRows(n: u32, c: usize, rows: u32) !u32 {
    const f = exl3_bank.flash;
    const l = expert_bank.layerSegmentsN(n, f.hidden, f.inter) orelse return error.InvalidGeometry;
    return experts.B.S.allocRows(&l, c, rows);
}
/// turboexl3_glm53f: the per-layer prompt and decode rows `Source.open` plans from `slot_rows` (the default split).
pub fn planRows(slot_rows: sdk.Rows, prompt: []u32, decode: []u32) !void {
    try settings.layerRows(settings.defaultLayerRows(), slot_rows, @intCast(exl3_bank.flash.n_experts), prompt, decode);
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_SEED_DECODE` (off unless the plugin sets it before the bank opens): the resident
/// seed reads the pack's decode-weighted profile (`seed_decode_file`, a usage record of decode routes only) in place of
/// `expert-usage-v1.json`, the seed named, so a pack without the file is refused by name (`ExpertUsageMissing`). Only
/// which experts the prompt rows start with moves.
pub var seed_decode: bool = false;
pub const seed_decode_file = "expert-usage-decode-v1.json";
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_SEED_DECODE` for the next `Source.open` (`seed_decode`).
pub fn setSeedDecode(on: bool) void {
    seed_decode = on;
}
/// turboexl3_glm53f: `TURBOEXL3_GLM53F_KEEP_WARM`: the next `Source.open` installs the keep-warm engine (`gpu_warm`'s
/// `.waits`: its own queue, kernel and buffer, from this plugin's Metal shim, src/csrc/glm53f_gpu_warm.m) on the stream's
/// waits; off, nothing is installed.
pub var keep_warm: bool = false;
pub fn setKeepWarm(on: bool) void {
    keep_warm = on;
}
/// turboexl3_glm53f: the keep-warm's bill term (`gpu_warm.bill_bytes`), billed by the plugin where it is on.
pub const keep_warm_bill_bytes: u64 = gpu_warm.bill_bytes;
/// turboexl3_glm53f: the pack's usage profile the resident seed reads (`seed_decode`).
pub fn seedFile() []const u8 {
    return if (seed_decode) seed_decode_file else expert_usage.file_name;
}
/// turboexl3_glm53f: the resident seed's decision `Source.open` makes over the pack in `dir` (MLX-free): the default
/// (`expert-usage-v1.json` when the pack has it, else none) or, with `seed_decode`, the decode-weighted profile, named.
pub fn seedDecision(a: std.mem.Allocator, io: std.Io, dir: []const u8, layers: u32, n_experts: u32) !expert_usage.Decision {
    var s: expert_usage.Settings = .{};
    if (seed_decode) s.seed = .usage;
    return decideFile(a, io, &s, dir, seedFile(), "glm5_next", layers, n_experts);
}
/// turboexl3_glm53f: the kit's `expert.usage.decide` over the pack's `file` (a file name in `dir`) in place of
/// `expert-usage-v1.json` (the seed's decode-weighted profile, `seed_decode`): the same refusals by name, the profile
/// parsed by the kit's `expert.usage.parse`.
fn decideFile(a: std.mem.Allocator, io: std.Io, s: *const expert_usage.Settings, dir: []const u8, file: []const u8, comptime arch: []const u8, layers: u32, n_experts: u32) !expert_usage.Decision {
    if (s.seed == .none) return .{ .profile = null, .reason = .setting_none };
    const path = try std.fs.path.join(a, &.{ dir, file });
    defer a.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(expert_usage.max_file_bytes)) catch |e| switch (e) {
        error.FileNotFound => {
            if (s.seed == .usage) {
                sdk.log.warn(arch ++ ": load refused: resident_seed is usage, but {s} is absent (ExpertUsageMissing)\n", .{path});
                return error.ExpertUsageMissing;
            }
            return .{ .profile = null, .reason = .no_file };
        },
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            sdk.log.warn(arch ++ ": load refused: {s} is not readable: {s} (ExpertUsageMalformed)\n", .{ path, @errorName(e) });
            return error.ExpertUsageMalformed;
        },
    };
    defer a.free(text);
    const p = expert_usage.parse(a, text, arch, layers, n_experts) catch |e| {
        sdk.log.warn(arch ++ ": load refused: {s}: {s} (the file must be the v1 form of {s} with {d} layers x {d} experts)\n", .{ path, @errorName(e), arch, layers, n_experts });
        return e;
    };
    return .{ .profile = p, .reason = .usage };
}

pub const Source = struct {
    a: std.mem.Allocator,
    inner: *experts.Exl3Source,
    /// turboexl3_glm53f: SHARED_HOIST handed a decode call its hoist (its engaged line prints at the first).
    hoist_engaged: bool = false,
    /// turboexl3_glm53f: READ_AHEAD3's scores between `aheadScore` and `aheadIssue` (a kept array), null between windows.
    ahead_scores: ?T = null,
    /// turboexl3_glm53f: READ_AHEAD3's records read ahead so far (its engaged line prints at the first).
    ahead_records: u64 = 0,

    /// The bank in `dir`, streamed on `s` at `rows` slot rows per routed layer (the calibrated per-layer split of the
    /// decode rows). `gates`: every model layer's router as stored (bf16 `[experts, hidden]`) and its correction bias,
    /// the dense layers' empty (borrowed for the source's life).
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, gates: []const Gate, n_moe: u32, n_experts: u32, slot_rows: sdk.Rows) !*Source {
        const prompt_rows = try a.alloc(u32, n_moe);
        defer a.free(prompt_rows);
        const decode_rows = try a.alloc(u32, n_moe);
        defer a.free(decode_rows);
        try settings.layerRows(settings.defaultLayerRows(), slot_rows, n_experts, prompt_rows, decode_rows);
        // The stack of record's routes: the decode hit pass, the BF16 predictor, the keep-warm waits on a GPU stream, no
        // rate tally, the EXL3 forms, the prompt levers, direct demand reads, the prompt residents kept.
        // turboexl3_glm53f: the keep-warm queue only with TURBOEXL3_GLM53F_KEEP_WARM (its Metal shim this plugin's own,
        // src/csrc/glm53f_gpu_warm.m: the server's csrc/mimo_gpu_warm.inc is not in the host table).
        const warm: gpu_warm.Route = if (keep_warm) .waits else .off;
        const inner = try experts.Exl3Source.init(a, io, dir, prompt_rows, decode_rows, gates, s, .hitpass, .bf16, warm, .{ .ratio = settings.default_tally_ratio }, .{}, prompt_levers.Settings{}, .direct, .keep);
        errdefer inner.deinit();
        try inner.ex.bindMergeGateWaits();
        _ = try inner.prebuildExperts();
        // turboexl3_glm53f: the seed's profile file (`seedDecision`, `TURBOEXL3_GLM53F_SEED_DECODE`).
        var d = try seedDecision(a, io, dir, n_moe, n_experts);
        defer if (d.profile) |*p| p.deinit();
        if (d.profile) |*p| _ = try inner.seedResidents(p);
        const self = try a.create(Source);
        self.* = .{ .a = a, .inner = inner };
        return self;
    }

    pub fn deinit(self: *Source) void {
        self.inner.deinit();
        self.a.destroy(self);
    }

    /// Model layer `layer`'s routed sum `[n, hidden]` for the normed MoE input `x` `[n, hidden]`, the routes `idx`
    /// `[n, k]` and their weights `wts`.
    pub fn rows(self: *Source, layer: u32, x: T, idx: T, wts: T) !T {
        var g = &self.inner.g;
        const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
        const w = if (g.dtypeOf(wts) == .float32) wts else try g.astype(wts, .float32);
        return self.inner.forward(layer, xb, idx, w, &.{});
    }
    /// turboexl3_glm53f: READ_AHEAD3: the first routed layer's predictor scored on a decode window's last dense layer's
    /// MLP input `x` and committed (an async eval: no host wait); a prompt call, a wider window or a missing gate scores
    /// nothing. `aheadIssue` reads the scores back.
    pub fn aheadScore(self: *Source, x: T) !void {
        if (!self.inner.decoding()) return;
        const g = &self.inner.g;
        if (self.ahead_scores) |old| {
            g.release(old);
            self.ahead_scores = null;
        }
        const sh = g.shapeOf(x);
        const hidden = sh.d[sh.n - 1];
        var n_rows: usize = 1;
        for (sh.d[0 .. sh.n - 1]) |d| n_rows *= @intCast(d);
        if (n_rows == 0 or n_rows > ahead_max_rows) return;
        const m = g.mark();
        defer g.resetTo(m);
        const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
        const xf = try g.reshape(xb, &.{ @intCast(n_rows), hidden });
        const sc = (try self.inner.ex.firstScores(g, xf, first_moe_layer)) orelse return;
        const kept = g.keep(sc);
        g.asyncEval(&.{kept}) catch |e| {
            g.release(kept);
            return e;
        };
        self.ahead_scores = kept;
    }
    /// turboexl3_glm53f: READ_AHEAD3: the scores `aheadScore` committed read back and the first routed layer's likely
    /// records read ahead before its route (`experts.speculateAhead` on the kit's stream: tagged for that route, whose
    /// demand reads claim them and whose own step settles the rest). A no-op with no scores or outside the decode phase.
    pub fn aheadIssue(self: *Source) !void {
        const sc = self.ahead_scores orelse return;
        self.ahead_scores = null;
        const g = &self.inner.g;
        defer g.release(sc);
        if (!self.inner.decoding()) return;
        var buf: [ahead_max_rows * 512]f32 = undefined;
        const n: usize = @intCast(mlx.mlx_array_size(sc));
        if (n == 0 or n > buf.len) return;
        const scores = try g.hostF32(sc, buf[0..n]);
        const picked = try experts.speculateAhead(self.inner.sdk_stream, 0, scores);
        if (self.ahead_records == 0 and picked > 0) std.log.scoped(.turboexl3_glm53f).info("[turboexl3_glm53f] TURBOEXL3_GLM53F_READ_AHEAD3 engaged: {d} records of the first routed layer read ahead from the last dense layer's MLP input", .{picked});
        self.ahead_records += picked;
    }
    /// turboexl3_glm53f: SHARED_HOIST: `rows` with `hoist` (the layer's shared expert over the same rows), which a
    /// decode call commits behind its hit wave and ahead of its miss waves when it misses; a wide call does not take it.
    pub fn rowsHoist(self: *Source, layer: u32, x: T, idx: T, wts: T, hoist: T) !T {
        var g = &self.inner.g;
        const xb = if (g.dtypeOf(x) == .bfloat16) x else try g.astype(x, .bfloat16);
        const nt: u32 = @intCast(g.shapeOf(xb).d[0]);
        const lane = self.inner.decodeLaneAt(nt);
        // turboexl3_glm53f: TURBOEXL3_GLM53F_MOE_COMBINE1 (src/moe_combine1.zig): a decode call's combine reads the
        // router's bf16 weights and widens them itself (exact), so their f32 cast is left out
        const w = if (g.dtypeOf(wts) == .float32 or (lane and @import("mlx_host").moe_combine1.on and g.dtypeOf(wts) == .bfloat16)) wts else try g.astype(wts, .float32);
        if (lane and !self.hoist_engaged) {
            self.hoist_engaged = true;
            std.log.scoped(.turboexl3_glm53f).info("[turboexl3_glm53f] TURBOEXL3_GLM53F_SHARED_HOIST engaged: a decode call of {d} rows (model layer {d}) took its shared expert as its hoist", .{ nt, layer });
        }
        return self.inner.forward(layer, xb, idx, w, if (lane) &.{hoist} else &.{});
    }

    /// A prompt call's predicted routes of `layer` (every row's ids): its experts, hottest first, read while its
    /// attention runs.
    pub fn readAhead(self: *Source, layer: u32, ids: []const u16) !void {
        if (self.inner.decoding()) return;
        var counts: [512]u32 = @splat(0);
        const n = self.inner.bank.geometry.num_experts;
        for (ids) |e| counts[e] += 1;
        var ranked: [512]u16 = undefined;
        try self.inner.readAheadSeed(layer, sdk_ext.expert.policy.rankHottest(counts[0..n], ranked[0..n]));
    }

    /// A prompt pass starts the prompt phase (a new request's after a decode); a decode call grows the decode rows.
    pub fn setPhase(self: *Source, prompt: bool) !void {
        if (prompt) {
            if (self.inner.decoding()) try self.inner.beginPrefill();
        } else if (!self.inner.decoding()) try self.inner.beginDecode();
    }

    /// After the eval that consumed the forward's calls: the released routes settled.
    pub fn settle(self: *Source) !void {
        try self.inner.flush();
    }
};
