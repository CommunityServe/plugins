//! The streamed bank's slot rows: the prefill bill the host's stream room reads, and the rows that room leaves.
//!
//! The host hands a streamed plugin `LoadCtx.stream_room`, the GPU ceiling less one prompt's bill at 16K + 1K and the
//! wired margin (mlx-serve's `src/engines/mlx/plugin/driver.zig` `streamRoomAt`). For its built-in glm5_next the
//! engine prices that prompt with `glm5PrefillMemoryNeeded` (`src/engines/mlx/admission/glm5_bill.zig`), because the
//! legacy parse marks the pack `expert_bank`. An outside plugin's engine config is its own facts
//! (`plugin_config.adapted`), with no `expert_bank`, so the host prices the prompt through the plugin's
//! `Bill.prefill_bytes` and hands a room of 0 when the bill has none. The copied built-in bill has none: the room was
//! 0, every layer held top_k (8) slot rows and every decode token read its experts from the SSD (g53smb, 2026-11-06:
//! 4.8 tok/s). `prefillNeed` is `glm5PrefillMemoryNeeded` over the copied Config at the host's defaults, for the
//! plugin's `prefill_bytes`.
//!
//! mlx-serve 26.11.1's driver (the kit this plugin builds on, the box's G_KITBM server) has no plugin branch: its
//! `streamRoom` prices only a built-in GLM pack (`expert_bank`) and hands any other plugin 0. So the plugin sizes its bank
//! from its own room when the host gives none (`roomFor`): the host's `staticGpuMemoryCeiling` (`gpuCeiling`) less the
//! same prompt bill and the host's wired margin (`LoadCtx.wired_margin_bytes`). A host that gives a room is used as is.
//!
//! MLX-free: the tests run in the kit's test step (config.zig imports this file) and the plugin-rooted test.

const std = @import("std");
const Config = @import("mlx_serve/plugins/glm5_next/config.zig").Config;

/// The engine's terms `glm5PrefillMemoryNeeded` reads from outside the config, at mlx-serve bc6c5f2c's defaults: the
/// SSM checkpoint stride and cap (`Launch.ssm_checkpoint_stride` 256, `ssm_checkpoint_max` 16), the snapshot backoff
/// (`generate/prefill_chunk.zig` `SSM_SNAPSHOT_BACKOFF`), the runtime floor (`sdk/bill/prefill.zig`
/// `PREFILL_RUNTIME_FLOOR_BYTES`) and the indexer's bytes a key pool (`GLM5_INDEXER_BYTES_PER_POOL`).
pub const checkpoint_stride: u64 = 256;
pub const checkpoint_max: u64 = 16;
pub const snapshot_backoff: u64 = 30;
pub const runtime_floor: u64 = 512 * 1024 * 1024;
pub const indexer_bytes_per_pool: u64 = 12;
/// The host's stream room's prompt (`streamRoom`): 16K + 1K tokens at the 16384-token wide chunk
/// (`generate.zig` `WIDE_PREFILL_CHUNK`), 16-bit KV, max_tokens 1.
pub const room_seq: u64 = 16_384 + 1_024;
pub const room_chunk: u64 = 16_384;
pub const room_output: u64 = 1;
/// The streamer's transient rows (`mimo/stream.zig` `transient_rows`: wide_depth 2 x 48 route ids), which the prompt
/// phase leaves beside the slot rows; plugin.zig checks it against the streamer at compile time.
pub const streamer_transient_rows: u32 = 96;

/// `sdk/bill/prefill.zig` `kvBytesPerTokenAtBits`.
fn kvAtBits(dense: u64, kv_bits: u64) u64 {
    if (kv_bits >= 16) return dense;
    return dense * (2 * kv_bits + 1) / 32;
}

/// `bills.zig` `prefillFfnWidth`.
fn ffnWidth(c: *const Config) u64 {
    const per_tok: u64 = @max(@as(u64, c.num_experts_per_tok), 1);
    const moe_w: u64 = @as(u64, c.moe_intermediate_size) * per_tok + c.shared_expert_intermediate_size;
    const dense_w: u64 = if (c.intermediate_size_declared) c.intermediate_size else 0;
    const w = @max(dense_w, moe_w);
    return if (w > 0) w else c.intermediate_size;
}

/// The bill's MLP envelope (`terms`' 3 x mlp: a chunk's working set) a forward row: TURBOEXL3_GLM53F_PF_REUSE's raise
/// of MLX's cache limit through a KDA run (src/pf_reuse.zig).
pub fn envelopeRowBytes(c: *const Config) u64 {
    return 3 * 8 * @max(@as(u64, c.hidden_size), ffnWidth(c)) * 2;
}

/// `bills.zig` `retainedSsmCheckpointBytes` at the default stride and cap: the stride-aligned captures in
/// (warm, seq] plus the end-of-prompt snapshot, the stride the prefill chunk at least.
fn retainedCheckpoints(per_cp: u64, seq: u64, warm: u64, chunk: u64) u64 {
    if (per_cp == 0) return 0;
    const stride = @max(checkpoint_stride, chunk);
    const span = seq -| @min(warm, seq);
    var n: u64 = seq / stride -| @min(warm, seq) / stride;
    if (span > snapshot_backoff) n += 1;
    return @min(n, checkpoint_max) * per_cp;
}

/// The terms of `glm5PrefillMemoryNeeded` before its sum: what stays (`resident`: the prompt's KV, the retained and the
/// live SSM checkpoints), one chunk's working set (`scratch`) and a generation's growth (`growth`).
const Terms = struct { resident: u64, scratch: u64, growth: u64 };

/// One prompt's bill: `glm5PrefillMemoryNeeded(config, seq, room_output, kv_bits, chunk, .{ .matched_tokens = warm })`
/// for an EXL3 pack (no affine weights to dequantize, `dq_weights` 0) with no sliding window or QSA ring
/// (`slotFixedKvBytes` 0) and no warm-row credit (a donated prefix's KV is not taken off). `nax`: MLX's NAX gate
/// (`verifyQmmNaxAvailable`), where the indexer composes no score sheets and sparse attention gathers no latent rows.
pub fn prefillNeed(c: *const Config, seq: u64, chunk: u64, kv_bits: u64, warm: u64, nax: bool) u64 {
    const t = terms(c, seq, chunk, kv_bits, warm, nax);
    const gross = t.resident +| @max(t.scratch, t.growth) +| runtime_floor;
    return gross * 5 / 4;
}

/// The host's hot prefix cache at its default cap (the server's `Hot prefix cache: ENABLED (capacity=32,
/// mem-cap=2048.0 MB)`): it keeps the request's decode-end checkpoint beside the decode, and the prompt bill does not
/// name it.
pub const hot_cache_bytes: u64 = 2048 << 20;

/// The decode phase's bill at the prompt bill's cell (`TURBOEXL3_GLM53F_DECODE_ROOM`): what the prompt leaves (its KV and
/// checkpoints), a generation's growth and the runtime floor at the prompt bill's 5/4, plus the host's hot prefix cache.
/// No chunk's working set: the phase change frees the prompt's scratch before decode's rows grow.
pub fn decodeNeed(c: *const Config, seq: u64, chunk: u64, kv_bits: u64, warm: u64, nax: bool) u64 {
    const t = terms(c, seq, chunk, kv_bits, warm, nax);
    return (t.resident +| t.growth +| runtime_floor) * 5 / 4 +| hot_cache_bytes;
}

fn terms(c: *const Config, seq: u64, chunk: u64, kv_bits: u64, warm: u64, nax: bool) Terms {
    const end = seq +| room_output;
    const attn: u64 = c.attnLayerCount();
    const kv_tok = kvAtBits(attn * (@as(u64, c.dsa_kv_lora_rank) + 2 * @as(u64, c.dsa_index_head_dim)) * 2, kv_bits);
    const view_tok: u64 = if (kv_bits < 16) (@as(u64, c.dsa_kv_lora_rank) + 4 * @as(u64, c.dsa_index_head_dim)) * 2 else 0;
    const fwd = @min(chunk, @max(seq -| warm, 1));
    const per_cp = c.ssmCheckpointBytes(4);
    const resident = seq *| kv_tok +| retainedCheckpoints(per_cp, seq, warm, chunk) +| 4 *| per_cp;
    const mlp: u64 = 8 * fwd * @max(@as(u64, c.hidden_size), ffnWidth(c)) * 2;
    // GLM-5 on NAX: the sorted expert gather and the per-core KDA recurrence hold no per-token stream
    // (`prefillStreamBytesPerTokenOn`); off NAX the plugin refuses to load, so this is the only case.
    const envelope = 3 * mlp;
    const pools = seq / @max(@as(u64, c.dsa_index_kpool), 1);
    const indexed = nax and c.dsa_index_head_dim == 128;
    const indexer = fwd *| pools *| (indexer_bytes_per_pool + if (indexed) @as(u64, 0) else 12 * @as(u64, c.dsa_index_heads));
    const sparse = nax and c.dsa_kv_lora_rank == 512 and c.num_attention_heads % 32 == 0;
    const selected = @min(seq, @as(u64, c.dsa_index_topk) +| c.dsa_index_kpool -| 1);
    const gather = if (sparse) 0 else fwd *| selected *| @as(u64, c.dsa_kv_lora_rank) *| 2;
    const layer_kv = seq *| kv_tok / @max(attn, 1);
    const scratch = envelope +| indexer +| gather +| seq *| view_tok +| layer_kv;
    const growth = room_output *| kv_tok +| end *| kv_tok +| end *| view_tok;
    return .{ .resident = resident, .scratch = scratch, .growth = growth };
}

/// The host's stream room from a plugin's bill (`streamRoomAt`'s plugin branch, mlx-serve after 26.11.1): the ceiling
/// less the prompt's bill at the room's cell and the wired margin; 0 when the bill has no `prefill_bytes` (and always 0
/// on 26.11.1's driver).
pub fn hostRoom(prefill: ?*const fn (seq: u64, chunk: u64) u64, ceiling: u64, margin: u64) u64 {
    const f = prefill orelse return 0;
    return ceiling -| f(room_seq, room_chunk) -| margin;
}

/// The host's `staticGpuMemoryCeiling` (`src/host/registry/memory_ceiling.zig` `getGpuWorkingSetLimit`):
/// `MLX_SERVE_GPU_CEILING_MB` (`env_mb`: a positive integer of MB; anything else is ignored) stands in for the
/// machine's working set; else Metal's `max_recommended_working_set_size` (`max_rec`); else 75% of the RAM
/// (`memsize`, hw.memsize), 8 GiB when that reads 0.
pub fn gpuCeiling(env_mb: ?[]const u8, max_rec: u64, memsize: u64) u64 {
    if (env_mb) |raw| {
        const mb = std.fmt.parseInt(u64, raw, 10) catch 0;
        if (mb > 0) return mb *| (1 << 20);
    }
    if (max_rec > 0) return max_rec;
    if (memsize == 0) return 8 << 30;
    return memsize * 75 / 100;
}

/// Where the bank's room came from: the host's `LoadCtx.stream_room`, the plugin's own, or `TURBOEXL3_GLM53F_BILL_FIT`'s
/// fixed plan (`fitPlan`, its room the ceiling).
pub const RoomSource = enum { host, own, fit };

/// The room the bank is sized from: the host's `LoadCtx.stream_room` when it gives one, else the plugin's own: the
/// ceiling less one prompt's bill at the room's cell (`prompt_need`) and the wired margin.
pub fn roomFor(host_room: u64, ceiling: u64, prompt_need: u64, margin: u64) struct { bytes: u64, source: RoomSource } {
    if (host_room > 0) return .{ .bytes = host_room, .source = .host };
    return .{ .bytes = ceiling -| prompt_need -| margin, .source = .own };
}

/// A streamed pack's slot rows per routed layer (the built-in's `streamRows`): `free` bytes (the room less the memory
/// in use), `record` bytes a row (the widest layer's) in each of `n_moe` layers; the prompt phase also leaves the
/// bank's `transient` rows. At least `top_k` rows, at most `n_experts`.
pub fn rowsFor(free: u64, record: u64, n_moe: u32, transient: u32, top_k: u32, n_experts: u32) Plan {
    return rowsForPhases(free, free, record, n_moe, transient, top_k, n_experts);
}

/// The slot rows a load hands the bank: uniform prompt and decode rows per routed layer (the bank's own per-layer split
/// spreads the decode rows past the prompt rows at the phase change).
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

/// The decode lookahead's records read ahead a layer call (`mimo/stream_source.zig` `lookahead.budget`); plugin.zig
/// checks it against the streamer at compile time.
pub const streamer_spec_budget: u32 = 2;

/// The read pool's workers (`mimo/stream.zig` `io_workers`) and each worker's staging buffer (`glm5n/settings.zig`
/// `reader_staging_bytes`); plugin.zig checks both against the streamer at compile time.
pub const streamer_io_workers: u32 = 4;
pub const streamer_reader_staging_bytes: u64 = 16 << 20;

/// The read pool the kit's stream maps when the bank opens (the kit's `expert.io.poolBytes` over the pool `Stream.init`
/// starts), from the stream's own sizes: `mimo/stream.zig`'s 4 workers, each with a staging buffer of `glm5n/settings.zig`
/// `reader_staging_bytes` (16 MiB; both the constants above, checked against the bank at compile time); the early read
/// schedule's ticket ring (`io_tickets_early`, 2,048) at 89 B a ticket (the kit's 10 result words, a log word and a
/// published flag); and 2 x `budget` speculative slots, each the widest record page-rounded plus two pages
/// (`expert_io.slotBytes`). Host memory the default bill leaves out; the plugin-rooted test checks it against the kit's
/// `poolBytes` on every SC layout.
pub const streamer_pool_workers: u64 = streamer_io_workers;
pub const streamer_staging_bytes: u64 = streamer_reader_staging_bytes;
pub const streamer_pool_tickets: u64 = 2048;
pub const pool_ticket_bytes: u64 = 10 * 8 + 8 + 1;
pub fn readPoolBytes(record: u64, page: u64, budget: u32) u64 {
    return poolBytesAt(record, page, budget, pool_ticket_bytes);
}
/// The read pool the fit's evidence ran (sc_3.10's peaks, `fit_*` below): the copied pool before the kit's, its ticket 8
/// result words (73 B); its bytes are in the measured peaks, so the bases take them off as they were.
const evidence_ticket_bytes: u64 = 8 * 8 + 8 + 1;
fn poolBytesAt(record: u64, page: u64, budget: u32, ticket: u64) u64 {
    const slot = std.mem.alignForward(u64, record, page) + 2 * page;
    return streamer_pool_workers * streamer_staging_bytes + streamer_pool_tickets * ticket + 2 * @as(u64, budget) * slot;
}

/// The largest context `contextLimit` takes (16M tokens).
pub const max_context = 1 << 24;

/// The server's context limit from the load settings (`LoadCtx.settings_json`: mlx-serve writes `"ctx_size":<n>` when
/// the model's context is set, by `--ctx-size` or the model's `ctx_size`; the driver's `loadSettings`): a positive
/// integer up to `max_context`, else null (an auto context, which the plugin cannot read).
pub fn contextLimit(a: std.mem.Allocator, settings_json: []const u8) ?u64 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), settings_json, .{}) catch return null;
    if (doc != .object) return null;
    const v = doc.object.get("ctx_size") orelse return null;
    if (v != .integer or v.integer <= 0 or v.integer > max_context) return null;
    const n: u64 = @intCast(v.integer);
    return n;
}

/// `TURBOEXL3_GLM53F_BILL_FIT`: the slot rows fixed for a box, its ceiling and the server's context limit, from the
/// measured peak of the box's non-file memory (wired + anonymous + compressor: the account the GPU window's guard trips
/// on), never from the memory available at load. The evidence (ai-server: M5 Max, 128 GiB; sc_3.10; the standard
/// 16,387 + 1,024-token request at MTP depth 2, the usage seed on; GB = 1e9):
/// - g53fzr3 (2026-11-07), LAYER_BILL + DECODE_ROOM at 167/212 and 168/213 rows: the guard's accounted peak 110.0 GiB,
///   the step's footprint 96.1 GiB at it, in the decode phase (`fit_guard_peak`, `fit_step_footprint`; the 212 rows);
/// - g53lb1 (2026-11-07), LAYER_BILL alone at 157/161 rows: the server's footprint peak 95.5 GB, in the prompt phase
///   (`fit_prompt_footprint`).
/// Each server was started for its run, so its hot prefix cache held at most the pre-warm's entry. The fit is
/// sc_3.10's (its trunk is in the bases): `fit_record` is its widest record, and a load of another pack is refused.
pub const fit_record: u64 = 12_619_776;
const fit_k3_record: u64 = 9_474_048;
pub const fit_guard_peak: u64 = 110 << 30;
pub const fit_step_footprint: u64 = 961 * (1 << 30) / 10;
const fit_decode_rows: u64 = 212;
const fit_prompt_footprint: u64 = 95_500_000_000;
const fit_prompt_rows: u64 = 157;
/// sc_3.10's row in all 42 routed layers (layer 3 at K4, 41 at K3), its two geometries' row (a transient or window
/// row of each), and its K4 layer's share of the bank's decode split (`glm5n/settings.zig` `default_layer_offsets[0]`:
/// 63 rows more there, as many fewer in K3 layers).
const fit_row: u64 = fit_record + 41 * fit_k3_record;
const fit_geo_row: u64 = fit_record + fit_k3_record;
const fit_split_extra: u64 = 63 * (fit_record - fit_k3_record);
/// The evidence's read pool: the default lookahead budget and the default 4 workers at the M5's 16 KiB page, its
/// tickets the copied pool's (`evidence_ticket_bytes`).
const fit_pool: u64 = poolBytesAt(fit_record, 16 << 10, streamer_spec_budget, evidence_ticket_bytes);
/// The box's non-file memory outside the server at the evidence's peak (the guard's account less the step's
/// footprint): what the fit allows the rest of the box at load.
pub const fit_rest: u64 = fit_guard_peak - fit_step_footprint;
/// The decode phase's base: the guard's peak less the bank (the 212 decode rows on the split, the window rows of both
/// geometries, the read pool). The rest of the box, the server's trunk and runtime, and the request's decode memory.
pub const fit_base_decode: u64 = fit_guard_peak - (fit_decode_rows * fit_row + fit_split_extra) - @as(u64, streamer_window_rows) * fit_geo_row - fit_pool;
/// The prompt phase's base: the rest of the box and the server's footprint at its prompt peak less its bank there (the
/// 157 prompt rows, the transient rows of both geometries, the read pool).
pub const fit_base_prompt: u64 = fit_rest + (fit_prompt_footprint - fit_prompt_rows * fit_row - @as(u64, streamer_transient_rows) * fit_geo_row - fit_pool);

/// `TURBOEXL3_GLM53F_FIT_INUSE`'s margin over the box's non-file memory in use outside the server at load (the
/// purgeable and reusable pages left out, `box_memory.inUseNonFileBytes`): 1 GiB, about three times what the GPU
/// window's guard account grew past the window's start and the server's footprint in the 59-entry control's windows
/// (0.22-0.36 GiB: g53rah8, g53dfs9, g53sab2m), for that memory's growth between the load and the peak.
pub const fit_inuse_margin: u64 = 1 << 30;

/// `TURBOEXL3_GLM53F_FIT_INUSE`: the fit's allowance for the box's memory outside the server, `in_use` (its non-file
/// memory in use at load less the server's footprint) and the margin. plugin.zig passes it to `fitPlan` as `allow`
/// with `in_use` as the rest, so nothing is over it and each phase takes `fit_rest - allow` more bytes than the fixed
/// allowance gives (fewer when the allowance is above `fit_rest`).
pub fn fitInUseAllowance(in_use: u64) u64 {
    return in_use +| fit_inuse_margin;
}

/// `TURBOEXL3_GLM53F_BILL_FIT`'s plan: the bytes each phase's slot rows may take (the prompt's with its transient rows,
/// the decode's with its window rows), so the box's non-file memory peaks at `limit` (the server's GPU ceiling) at
/// most for any request up to `context` tokens. Each phase takes its measured base, the bills' growth from the
/// evidence's cell (the room's 16K + 1K) to the context (`prefillNeed` / `decodeNeed`: the engine's exact KV,
/// checkpoint and indexer terms at its 5/4; none below the cell), the host's hot prefix cache at its cap (the
/// evidence's held a request's entry at most), the read pool (`pool`), and the box's non-file memory outside the
/// server at load (`rest`) past the fit's allowance for it (`allow`, the fit's own `fit_rest` as plugin.zig passes it;
/// `excess`, taken off both phases: rows capped, never an undercount). Each base holds `allow` for that memory
/// (`base_prompt`, `base_decode`). Without `decode_room` the decode rows take the prompt's bytes.
pub const FitPlan = struct { prompt: u64, decode: u64, grow_prompt: u64, grow_decode: u64, excess: u64, base_prompt: u64, base_decode: u64, freed: u64 = 0 };
pub fn fitPlan(c: *const Config, limit: u64, context: u64, nax: bool, pool: u64, rest: u64, decode_room: bool, allow: u64) FitPlan {
    return fitPlanFreed(c, limit, context, nax, pool, rest, decode_room, allow, 0);
}

/// turboexl3_glm53f: TURBOEXL3_GLM53F_TRUNK_F16_FREE: `fitPlan` with `freed` bytes of the server's trunk released at load
/// (src/f16_free.zig: the f16 sources of its bf16 copies, which the evidence's load held in both phases): each phase's
/// base (with the allowance `allow`) taken less them, so both phases' rows take them. No new measured fit: the bases'
/// other terms are the evidence's as they were.
pub fn fitPlanFreed(c: *const Config, limit: u64, context: u64, nax: bool, pool: u64, rest: u64, decode_room: bool, allow: u64, freed: u64) FitPlan {
    const seq = @max(context, room_seq);
    const grow_prompt = prefillNeed(c, seq, room_chunk, 16, 0, nax) -| prefillNeed(c, room_seq, room_chunk, 16, 0, nax);
    const grow_decode = decodeNeed(c, seq, room_chunk, 16, 0, nax) -| decodeNeed(c, room_seq, room_chunk, 16, 0, nax);
    const base_prompt = fit_base_prompt - fit_rest + allow;
    const base_decode = fit_base_decode - fit_rest + allow;
    const excess = rest -| allow;
    const prompt = limit -| (base_prompt -| freed) -| grow_prompt -| hot_cache_bytes -| pool -| excess;
    const decode = if (decode_room) limit -| (base_decode -| freed) -| grow_decode -| hot_cache_bytes -| pool -| excess else prompt;
    return .{ .prompt = prompt, .decode = decode, .grow_prompt = grow_prompt, .grow_decode = grow_decode, .excess = excess, .base_prompt = base_prompt, .base_decode = base_decode, .freed = freed };
}

/// The plugin's opt-in bank switches, read at load (`parseSwitch`).
pub const Switches = struct {
    /// `TURBOEXL3_GLM53F_LAYER_BILL=1`: the slot rows from each layer's own row bytes (`rowsByLayer`).
    layer_bill: bool = false,
    /// `TURBOEXL3_GLM53F_DECODE_ROOM=1`: the decode phase's rows also in the prompt bill less the decode bill
    /// (`decodeNeed`), which the phase change frees.
    decode_room: bool = false,
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
    /// `TURBOEXL3_GLM53F_DEMAND_SPLIT=1`: a decode call that reads one record reads its gate/up span in pieces, one per
    /// read pool worker but one (the stream's `Options.demand_split`); no row count changes.
    demand_split: bool = false,
    /// `TURBOEXL3_GLM53F_SHARED_HOIST=1`: a decode call's shared expert built before the routed call and committed
    /// behind its hit wave, ahead of its miss waves, when it misses (the copy's `Routed.source_hoist`, glm5n_source.zig
    /// `shared_hoist`): the same ops and the same add; no row count changes.
    shared_hoist: bool = false,
    /// `TURBOEXL3_GLM53F_READ_AHEAD3=1`: a decode window's first routed layer, which no routed call before it predicts,
    /// has its likely records read ahead from the last dense layer's MLP input (the copy's `Routed.ahead_score_fn` and
    /// `ahead_issue_fn`, glm5n_source.zig `read_ahead3`): speculative reads only; no row count changes.
    read_ahead3: bool = false,
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
    /// `TURBOEXL3_GLM53F_BILL_FIT=1`: the rows fixed by the measured peak (`fitPlan`) for any request to the server's
    /// context limit, the read pool billed (`readPoolBytes`), not sized from the memory available at load.
    bill_fit: bool = false,
    /// `TURBOEXL3_GLM53F_FIT_INUSE=1` (with `bill_fit`): the fit's allowance for the box's memory outside the server is
    /// what is in use there at load and `fit_inuse_margin` (`fitInUseAllowance`), in place of the evidence's fixed
    /// `fit_rest`; only the rows move.
    fit_inuse: bool = false,
    /// `TURBOEXL3_GLM53F_SEED_DECODE=1`: the resident seed reads the pack's `expert-usage-decode-v1.json` in place of
    /// `expert-usage-v1.json` (the copy's `Glm5n.seedDecision`); only which experts the prompt rows start with moves.
    seed_decode: bool = false,
    /// `TURBOEXL3_GLM53F_KEEP_WARM=1`: the keep-warm engine on the stream's waits (the copy's `gpu_warm` at `.waits`,
    /// glm5n_source.zig `keep_warm`): its own Metal queue keeps the GPU busy while a gated wait or a read-ahead host
    /// wait is pending; nothing a model array reads changes. Its bill term (`Glm5n.keep_warm_bill_bytes`, 1 MiB) comes
    /// off the bytes the rows are sized from (with `TURBOEXL3_GLM53F_BILL_FIT`, in its read pool).
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

/// mlx-serve's own GLM-5.3-Flash bill fixture (`admission/bills_test.zig` `glm5TestConfig`) in the copied Config:
/// 45 layers (11 DSA, 34 KDA), hidden 4096, 64 heads, the DSA latent 512, indexer 32 heads of 128 over pools of 4 at
/// top 2048, KDA 64 x 128 heads at conv 4, 288 experts of 2048 at top-8, a 2048 shared expert, the dense MLP 12288
/// (src/row_plan_test.zig's plans read it too).
pub fn glmFixture() Config {
    var c: Config = .{};
    c.model_type = "glm5_next";
    c.num_hidden_layers = 45;
    c.full_attention_interval = 4;
    c.num_attention_heads = 64;
    c.hidden_size = 4096;
    c.dsa_kv_lora_rank = 512;
    c.dsa_index_topk = 2048;
    c.dsa_index_kpool = 4;
    c.dsa_index_head_dim = 128;
    c.dsa_index_heads = 32;
    c.linear_num_key_heads = 64;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 64;
    c.linear_value_head_dim = 128;
    c.linear_conv_kernel_dim = 4;
    c.num_experts = 288;
    c.num_experts_per_tok = 8;
    c.moe_intermediate_size = 2048;
    c.shared_expert_intermediate_size = 2048;
    c.intermediate_size = 12288;
    c.intermediate_size_declared = true;
    return c;
}

test "turboexl3_glm53f stream room: the prefill bill is the engine's glm5PrefillMemoryNeeded on its own fixture" {
    const c = glmFixture();
    const kv_tok: u64 = 16_896; // 11 DSA layers x (512 + 2 x 128) x 2
    try testing.expectEqual(@as(u32, 11), c.attnLayerCount());
    // the engine's checkpoint ("34 KDA layers x (f32 state 64x128x128 + conv 3 x (2x64x128 + 64x128) x 2)")
    try testing.expectEqual(@as(u64, 147_619_840), c.ssmCheckpointBytes(4));
    // the engine's marginal test: 44,000 more prompt tokens cost their KV, a 1/11 share for the one DSA layer that
    // grows and 11,000 indexer pools at 12 B (NAX), at chunk 4096
    const a = prefillNeed(&c, 100_100, 4096, 16, 0, true);
    const b = prefillNeed(&c, 144_100, 4096, 16, 0, true);
    try testing.expectEqual((44_000 * kv_tok + 4_000 * kv_tok + 4096 * 11_000 * 12) * 5 / 4, b - a);
    // the stream room's cell (16K + 1K at the 16384-token wide chunk, 16-bit KV): the prompt KV 294,125,568 B, two
    // retained checkpoints and the live state's four (1,180 MB), the MLP envelope 3 x 8 x 16384 x 18432 x 2, the
    // indexer 16384 x 4352 pools x 12, one DSA layer's grow copy, the runtime floor, x 5/4
    try testing.expectEqual(@as(u64, 21_368_258_560), prefillNeed(&c, 17_408, 16_384, 16, 0, true));
    // off NAX the indexer composes its score sheets and sparse attention gathers latent rows: a larger bill
    try testing.expect(prefillNeed(&c, 17_408, 16_384, 16, 0, false) > prefillNeed(&c, 17_408, 16_384, 16, 0, true));
}

test "turboexl3_glm53f stream room: TURBOEXL3_GLM53F_PF_REUSE's raise a forward row is the bill's MLP envelope a row (GLM-5.3-Flash's fixture: 884,736 B)" {
    const c = glmFixture();
    try testing.expectEqual(@as(u64, 884_736), envelopeRowBytes(&c));
    // the scratch term's growth a chunk row at a fixed prompt: the envelope and the indexer's pools at 12 B (NAX)
    const seq: u64 = 17_408;
    const a = terms(&c, seq, 8192, 16, 0, true).scratch;
    const b = terms(&c, seq, 16_384, 16, 0, true).scratch;
    try testing.expectEqual((16_384 - 8192) * (envelopeRowBytes(&c) + seq / 4 * indexer_bytes_per_pool), b - a);
}

fn fixtureNeed(seq: u64, chunk: u64) u64 {
    const c = glmFixture();
    return prefillNeed(&c, seq, chunk, 16, 0, true);
}

test "turboexl3_glm53f stream room: with the prefill bill the host's room leaves the bank its rows; without it every layer holds top_k" {
    const GiB: u64 = 1 << 30;
    // g53smb: MLX_SERVE_GPU_CEILING_MB=102400, --wired-margin-gib 2; the sc_3.10 pack's widest record (its K4 layer
    // 3) in 42 routed layers of 288 experts at top-8
    const ceiling: u64 = 102_400 << 20;
    const margin: u64 = 2 * GiB;
    const record: u64 = 12_619_776;
    const transient = streamer_transient_rows;
    // the copied built-in bill (no prefill_bytes): the host hands a room of 0, so 8 rows a layer (g53smb's line)
    const none = hostRoom(null, ceiling, margin);
    try testing.expectEqual(@as(u64, 0), none);
    const starved = rowsFor(none -| 30 * GiB, record, 42, transient, 8, 288);
    try testing.expectEqual(@as(u32, 8), starved.decode);
    try testing.expectEqual(@as(u32, 8), starved.prompt);
    // the plugin's prefill bill: the ceiling less 19.9 GiB and the margin
    const room = hostRoom(&fixtureNeed, ceiling, margin);
    try testing.expectEqual(ceiling - 21_368_258_560 - margin, room);
    // beside 30 GiB in use (the residents and the box), 97 decode rows a layer
    const rows = rowsFor(room - 30 * GiB, record, 42, transient, 8, 288);
    try testing.expectEqual(@as(u32, 97), rows.decode);
    try testing.expect(rows.prompt > 8 and rows.prompt <= rows.decode);
    // every expert resident at most
    try testing.expectEqual(@as(u32, 288), rowsFor(1 << 50, record, 42, transient, 8, 288).decode);
}

test "turboexl3_glm53f stream room: on a host that hands an outside plugin no room (mlx-serve 26.11.1), the plugin's own room gives the bank its rows, within the window's guard" {
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    // the host's ceiling: the env's MB, else Metal's working set, else 75% of the RAM
    try testing.expectEqual(102_400 * MiB, gpuCeiling("102400", 115 * GiB, 128 * GiB));
    try testing.expectEqual(115 * GiB, gpuCeiling(null, 115 * GiB, 128 * GiB));
    try testing.expectEqual(115 * GiB, gpuCeiling("0", 115 * GiB, 128 * GiB));
    try testing.expectEqual(115 * GiB, gpuCeiling("102400 MB", 115 * GiB, 128 * GiB));
    try testing.expectEqual(96 * GiB, gpuCeiling(null, 0, 128 * GiB));
    try testing.expectEqual(8 * GiB, gpuCeiling(null, 0, 0));
    // g53dbg's probe (G_KITBM's server, MLX_SERVE_GPU_CEILING_MB=102400, --wired-margin-gib 2): the host's room 0;
    // at the bank's open, total 131072 MiB and available 103937 MiB
    const ceiling = gpuCeiling("102400", 0, 128 * GiB);
    const margin = 2 * GiB;
    const need = fixtureNeed(room_seq, room_chunk);
    try testing.expectEqual(@as(u64, 21_368_258_560), need);
    const own = roomFor(0, ceiling, need, margin);
    try testing.expectEqual(RoomSource.own, own.source);
    try testing.expectEqual(ceiling - need - margin, own.bytes);
    // a host that gives a room keeps it
    const given = roomFor(70 * GiB, ceiling, need, margin);
    try testing.expectEqual(RoomSource.host, given.source);
    try testing.expectEqual(70 * GiB, given.bytes);
    const used = 131_072 * MiB - 103_937 * MiB;
    const free = own.bytes -| used;
    const record: u64 = 12_619_776;
    const rows = rowsFor(free, record, 42, streamer_transient_rows, 8, 288);
    try testing.expectEqual(@as(u32, 104), rows.decode);
    try testing.expectEqual(@as(u32, 102), rows.prompt);
    // the bank at its decode rows billed at the widest record (layer 3's) in all 42 layers, beside what was in use and
    // one prompt's bill: within the ceiling less the margin by construction, so within the window's 112 GiB guard
    // (g53smb's) with 14 GiB to spare
    const bank = @as(u64, rows.decode) * 42 * record;
    try testing.expect(bank + used + need <= ceiling - margin);
    try testing.expect(bank + used + need + 14 * GiB <= 112 * GiB);
}

/// The sc_3.10 pack's slot-row bytes (`bank_layers.read`): its layer 3 at K4, layers 4-44 at K3.
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

test "turboexl3_glm53f stream room: the decode bill is the prompt bill's KV, checkpoints, growth and floor at 5/4, plus the host's hot cache" {
    const c = glmFixture();
    // the room's cell: the prompt KV 294,125,568 B and six checkpoints (two retained, the live state's four) stay; the
    // growth is the 17,409 positions' KV and the room's one output row; the floor; then 2 GiB of hot cache
    const resident: u64 = 17_408 * 16_896 + 6 * 147_619_840;
    const growth: u64 = 16_896 + 17_409 * 16_896;
    try testing.expectEqual((resident + growth + runtime_floor) * 5 / 4 + hot_cache_bytes, decodeNeed(&c, room_seq, room_chunk, 16, 0, true));
    try testing.expectEqual(@as(u64, 4_661_077_248), decodeNeed(&c, room_seq, room_chunk, 16, 0, true));
    // the prompt's bill is the larger by its chunk's working set: what the phase change frees for decode's rows
    try testing.expectEqual(@as(u64, 16_707_181_312), prefillNeed(&c, room_seq, room_chunk, 16, 0, true) - decodeNeed(&c, room_seq, room_chunk, 16, 0, true));
}

test "turboexl3_glm53f stream room: the two-phase rows are rowsFor's when the decode phase has the prompt's free bytes" {
    const record: u64 = 12_619_776;
    var free: u64 = 0;
    while (free < 90 << 30) : (free += 1_234_567_891) {
        const a = rowsFor(free, record, 42, streamer_transient_rows, 8, 288);
        const b = rowsForPhases(free, free, record, 42, streamer_transient_rows, 8, 288);
        try testing.expectEqual(a, b);
    }
    // g53smd's rows (its load line: 100 prompt / 102 decode)
    try testing.expectEqual(Plan{ .prompt = 100, .decode = 102 }, rowsFor(g53smd_free, record, 42, streamer_transient_rows, 8, 288));
    // the decode room at the widest row: 134 decode rows, the prompt's unchanged
    const c = glmFixture();
    const extra = prefillNeed(&c, room_seq, room_chunk, 16, 0, true) - decodeNeed(&c, room_seq, room_chunk, 16, 0, true);
    try testing.expectEqual(Plan{ .prompt = 100, .decode = 134 }, rowsForPhases(g53smd_free, g53smd_free + extra, record, 42, streamer_transient_rows, 8, 288));
}

test "turboexl3_glm53f stream room: the per-layer bill fits each phase's bytes with the K4 layer's and both geometries' transient rows counted" {
    const bytes = sc310Bytes();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    // the transient (96) and window (32) rows of each geometry: 12,619,776 + 9,474,048 B a row
    const geo: u64 = 12_619_776 + 9_474_048;
    const sum: u64 = 12_619_776 + 41 * 9_474_048;
    const plan = try rowsByLayer(g53smd_free, g53smd_free, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 130, .decode = 133 }, plan);
    try testing.expect(@as(u64, plan.prompt) * sum + 96 * geo <= g53smd_free);
    try testing.expect(@as(u64, plan.decode) * sum + 32 * geo <= g53smd_free);
    try testing.expect(@as(u64, plan.decode + 1) * sum + 32 * geo > g53smd_free);
    // the widest bill's rows at the same free bytes are fewer by a third in every phase
    const widest = rowsFor(g53smd_free, 12_619_776, 42, streamer_transient_rows, 8, 288);
    try testing.expect(widest.prompt + 30 <= plan.prompt and widest.decode + 31 <= plan.decode);
    // with the decode room: 175 rows a layer at the uniform split
    const c = glmFixture();
    const extra = prefillNeed(&c, room_seq, room_chunk, 16, 0, true) - decodeNeed(&c, room_seq, room_chunk, 16, 0, true);
    try testing.expectEqual(Plan{ .prompt = 130, .decode = 175 }, try rowsByLayer(g53smd_free, g53smd_free + extra, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr));
}

test "turboexl3_glm53f stream room: the per-layer bill keeps top_k rows on a starved box, at most every expert, and the prompt rows within the decode rows" {
    const bytes = sc310Bytes();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    try testing.expectEqual(Plan{ .prompt = 8, .decode = 8 }, try rowsByLayer(0, 0, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr));
    try testing.expectEqual(Plan{ .prompt = 288, .decode = 288 }, try rowsByLayer(1 << 50, 1 << 50, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr));
    // a decode phase with less room than the prompt's (a decode bill above the prompt's): the prompt rows follow it down
    const small = try rowsByLayer(60 << 30, 30 << 30, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    try testing.expect(small.prompt <= small.decode);
}

test "turboexl3_glm53f stream room: TURBOEXL3_GLM53F_BILL_FIT's bases are the measured peaks less their banks, and its plan holds each phase's peak within the ceiling for any request to the context limit" {
    const c = glmFixture();
    // the evidence's read pool (4 x 16 MiB of staging, 2,048 tickets of 73 B, 4 speculative slots of 12,664,832 B), and
    // the kit's (its tickets 89 B: 10 result words), which the plan bills
    try testing.expectEqual(@as(u64, 117_917_696), fit_pool);
    const pool = readPoolBytes(fit_record, 16 << 10, streamer_spec_budget);
    try testing.expectEqual(@as(u64, 117_950_464), pool);
    // the decode base: the guard's 110.0 GiB less 212 decode rows on the split, both geometries' window rows, the pool
    try testing.expectEqual(@as(u64, 32_064_681_984), fit_base_decode);
    try testing.expectEqual(fit_guard_peak, fit_base_decode + 212 * 401_055_744 + 63 * 3_145_728 + 32 * 22_093_824 + fit_pool);
    // the rest of the box at that peak (110.0 less 96.1 GiB), and the prompt base: that rest and the server's 95.5 GB
    // less 157 prompt rows, both geometries' 96 transient rows and the pool
    try testing.expectEqual(@as(u64, 14_925_011_354), fit_rest);
    try testing.expectEqual(@as(u64, 45_220_334_746), fit_base_prompt);
    try testing.expectEqual(fit_base_prompt, fit_rest + 95_500_000_000 - 157 * 401_055_744 - 96 * 22_093_824 - fit_pool);
    // ai-server's ceiling (112,640 MiB) and --ctx-size 32768: the bills' growth past the evidence's 16K + 1K, then
    // each phase's bytes
    const limit: u64 = 112_640 << 20;
    const plan = fitPlan(&c, limit, 32_768, true, pool, 0, true, fit_rest);
    try testing.expectEqual(@as(u64, 1_482_137_600), plan.grow_prompt);
    try testing.expectEqual(@as(u64, 833_331_200), plan.grow_decode);
    try testing.expectEqual(@as(u64, 69_143_726_950 - 32_768), plan.prompt);
    try testing.expectEqual(@as(u64, 82_948_186_112 - 32_768), plan.decode);
    // the per-layer rows at the uniform split (the bank's own split, 167 / 204: src/row_plan_test.zig)
    const bytes = sc310Bytes();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    const rows = try rowsByLayer(plan.prompt, plan.decode, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr);
    try testing.expectEqual(Plan{ .prompt = 167, .decode = 205 }, rows);
    // each phase's peak at those rows within the ceiling: the base, the growth, the hot cache, the pool, the rows
    try testing.expect(fit_base_prompt + plan.grow_prompt + hot_cache_bytes + pool + 167 * 401_055_744 + 96 * 22_093_824 <= limit);
    try testing.expect(fit_base_decode + plan.grow_decode + hot_cache_bytes + pool + 205 * 401_055_744 + 32 * 22_093_824 <= limit);
    // no growth at or below the evidence's cell; a longer context leaves less in both phases
    const cell = fitPlan(&c, limit, 8_192, true, pool, 0, true, fit_rest);
    try testing.expect(cell.grow_prompt == 0 and cell.grow_decode == 0 and cell.prompt == plan.prompt + plan.grow_prompt);
    const long = fitPlan(&c, limit, 65_536, true, pool, 0, true, fit_rest);
    try testing.expect(long.prompt < plan.prompt and long.decode < plan.decode);
    // without the decode room the decode rows take the prompt's bytes
    try testing.expectEqual(plan.prompt, fitPlan(&c, limit, 32_768, true, pool, 0, false, fit_rest).decode);
}

test "turboexl3_glm53f stream room: TURBOEXL3_GLM53F_BILL_FIT's rows do not follow the memory available at load; non-file memory outside the server past the fit's comes off both phases" {
    const c = glmFixture();
    const limit: u64 = 112_640 << 20;
    const pool = readPoolBytes(fit_record, 16 << 10, streamer_spec_budget);
    const plan = fitPlan(&c, limit, 32_768, true, pool, 0, true, fit_rest);
    // the rest of the box at load up to the fit's: the same plan (the load's own memory is the server's, not read)
    try testing.expectEqual(plan, fitPlan(&c, limit, 32_768, true, pool, fit_rest, true, fit_rest));
    // 4 GiB more: 4 GiB less in each phase, the rows capped
    const over = fitPlan(&c, limit, 32_768, true, pool, fit_rest + (4 << 30), true, fit_rest);
    try testing.expectEqual(@as(u64, 4 << 30), over.excess);
    try testing.expectEqual(plan.prompt - (4 << 30), over.prompt);
    try testing.expectEqual(plan.decode - (4 << 30), over.decode);
    // TURBOEXL3_GLM53F_TRUNK_F16_FREE: the trunk's freed f16 sources in both phases; none freed is the plan as it was
    try testing.expectEqual(plan, fitPlanFreed(&c, limit, 32_768, true, pool, 0, true, fit_rest, 0));
    const freed: u64 = 650 << 20;
    const more = fitPlanFreed(&c, limit, 32_768, true, pool, 0, true, fit_rest, freed);
    try testing.expectEqual(plan.prompt + freed, more.prompt);
    try testing.expectEqual(plan.decode + freed, more.decode);
    try testing.expectEqual(freed, more.freed);
    // ... and with more non-file memory outside the server, the excess still comes off both
    const both = fitPlanFreed(&c, limit, 32_768, true, pool, fit_rest + (1 << 30), true, fit_rest, freed);
    try testing.expectEqual(plan.decode + freed - (1 << 30), both.decode);
}

test "turboexl3_glm53f stream room: TURBOEXL3_GLM53F_FIT_INUSE allows the memory outside the server in use at load and 1 GiB, so each phase takes what the fixed allowance leaves over that" {
    const c = glmFixture();
    const limit: u64 = 112_640 << 20;
    const pool = readPoolBytes(fit_record, 16 << 10, streamer_spec_budget);
    const fixed = fitPlan(&c, limit, 32_768, true, pool, 0, true, fit_rest);
    const bytes = sc310Bytes();
    var pr: [42]u32 = undefined;
    var dr: [42]u32 = undefined;
    // 10, 12 and 13 GiB in use outside the server: the allowance 1 GiB over it, nothing over the allowance, each phase
    // the fixed plan's bytes plus fit_rest less the allowance, and the rows at the uniform split (the fixed plan's
    // 167 / 205 at fit_rest: more below it, fewer above it)
    const cases = [_]struct { gib: u64, rows: Plan }{
        .{ .gib = 10, .rows = .{ .prompt = 174, .decode = 212 } },
        .{ .gib = 12, .rows = .{ .prompt = 169, .decode = 207 } },
        .{ .gib = 13, .rows = .{ .prompt = 166, .decode = 204 } },
    };
    for (cases) |k| {
        const in_use = k.gib << 30;
        const allow = fitInUseAllowance(in_use);
        try testing.expectEqual(in_use + (1 << 30), allow);
        const p = fitPlan(&c, limit, 32_768, true, pool, in_use, true, allow);
        try testing.expectEqual(@as(u64, 0), p.excess);
        try testing.expectEqual(@as(i128, fixed.prompt) + fit_rest - allow, p.prompt);
        try testing.expectEqual(@as(i128, fixed.decode) + fit_rest - allow, p.decode);
        try testing.expectEqual(k.rows, try rowsByLayer(p.prompt, p.decode, &bytes, streamer_transient_rows, streamer_window_rows, 8, 288, &uniformSplit, &pr, &dr));
    }
    // the allowance at the fixed allowance's own value gives the fixed plan
    const at = fit_rest - fit_inuse_margin;
    try testing.expectEqual(fixed, fitPlan(&c, limit, 32_768, true, pool, at, true, fitInUseAllowance(at)));
}

test "turboexl3_glm53f stream room: the server's context limit is the load settings' ctx_size, a positive integer, else none" {
    const a = testing.allocator;
    try testing.expectEqual(@as(?u64, 32_768), contextLimit(a, "{\"ctx_size\":32768}"));
    try testing.expectEqual(@as(?u64, 32_768), contextLimit(a, "{\"mtp\":true,\"stream_experts\":201,\"ctx_size\":32768}"));
    for ([_][]const u8{ "{}", "{\"mtp\":true}", "{\"ctx_size\":0}", "{\"ctx_size\":-1}", "{\"ctx_size\":\"32768\"}", "{\"ctx_size\":3.5}", "{\"ctx_size\":99999999999}", "[]", "", "{" }) |t|
        try testing.expectEqual(@as(?u64, null), contextLimit(a, t));
}

test "turboexl3_glm53f stream room: a switch is on at 1, off unset, empty or 0, and any other value is refused by name" {
    try testing.expect(!try parseSwitch(null));
    try testing.expect(!try parseSwitch(""));
    try testing.expect(!try parseSwitch("0"));
    try testing.expect(try parseSwitch("1"));
    try testing.expectError(error.Glm53fSwitchValue, parseSwitch("on"));
    try testing.expectError(error.Glm53fSwitchValue, parseSwitch("2"));
    // TURBOEXL3_GLM53F_DEMAND_HELP takes 0 or 2, and refuses 1
    for ([_]?[]const u8{ null, "", "0" }) |v| try testing.expectEqual(@as(u2, 0), try parseDemandHelp(v));
    try testing.expectEqual(@as(u2, 2), try parseDemandHelp("2"));
    for ([_][]const u8{ "1", "3", "02", "2 ", "on", "-1" }) |v| try testing.expectError(error.Glm53fSwitchValue, parseDemandHelp(v));
}
