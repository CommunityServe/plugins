//! The plugin's memory beside the expert bank: what a prompt pass and a decode step keep free of the bank's slot rows.
//! MLX-free.
//!
//! The host hands the bank the GPU ceiling less `prompt` and its wired margin (`LoadCtx.stream_room`, from
//! `Bill.prefill_bytes`). Both reserves hold the KV of the served context's longest whole prompt (C1: `contextOf`, the
//! load's `--ctx-size`; src/ling_rq.zig `promptCapacityMax`), a pass's working set past its KV and `free_ram`, so a
//! request at the full context still leaves the box free memory. Fitted on
//! our M5 Max (2026-11-06, the standard test: 18,380 + 1,024 tokens, 8 GiB prompt / 2 GiB decode reserves): the
//! prompt pass peaked 7.5-8.3 GiB past the trunk and the bank's rows with 6.3 GiB of KV, and the decode steps 4.2 GiB
//! past them; the box's free RAM fell to 0.15-2.6 GB.

const std = @import("std");

/// The context the bill holds when the load names none (no `--ctx-size`): config.zig clamps an auto-sized context to
/// it, so mlx-serve never serves past it. The README's serve command names `--ctx-size 32768`.
pub const default_context: u64 = 32768;

/// The longest context the bill takes from a load, 2^24 tokens: past any context a Mac's memory holds a plan for.
pub const max_context: u64 = 1 << 24;

/// C1: the context the bill holds: the load's `ctx_size` (`LoadCtx.settings_json`; mlx-serve 66aad949 and 9a173773
/// write the manual context there, src/engines/mlx/plugin/driver.zig `loadSettings`: `--ctx-size`, else the server's
/// `max_context_size`), else `default_context` (no `ctx_size`, or 0: an auto-sized context, which config.zig clamps to
/// it). Settings that do not parse, or a `ctx_size` that is not a whole number from 0 to `max_context`, are refused by
/// name: the plugin could not bill the context mlx-serve serves. (`Shape.context` is not the served context: 9a173773
/// prices the stream room at a fixed 16,384 + 1,024 shape, driver.zig `streamRoomAt`.)
pub fn contextOf(settings_json: []const u8) error{LingCtxSizeUnreadable}!u64 {
    if (settings_json.len == 0) return default_context;
    var mem: [16 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&mem);
    const doc = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), settings_json, .{}) catch return error.LingCtxSizeUnreadable;
    if (doc != .object) return error.LingCtxSizeUnreadable;
    const v = doc.object.get("ctx_size") orelse return default_context;
    if (v != .integer or v.integer < 0 or v.integer > @as(i64, @intCast(max_context))) return error.LingCtxSizeUnreadable;
    if (v.integer == 0) return default_context;
    const served: u64 = @intCast(v.integer);
    return served;
}

/// C1: the bank's room at the served context. mlx-serve prices its stream room before `load` (9a173773
/// src/engines/mlx/plugin/driver.zig `streamRoomAt`: the GPU ceiling less `Bill.prefill_bytes` and the wired margin),
/// when the plugin has no served context yet: plugin.zig `prefillBytes` answers at `default_context` (`priced`). `load`
/// reads the served context (`contextOf`) and moves the room by the difference to that context's reserve (`served`):
/// `room` + `priced` is the ceiling less the margin. Null when the host gave no room, or the served context's reserve
/// takes it all (the bank's open refuses that by name).
pub fn roomAt(room: u64, priced: u64, served: u64) ?u64 {
    if (room == 0) return null;
    const top = room + priced;
    return if (served >= top) null else top - served;
}

/// A prompt pass's working set past its KV (measured: 1.2-2.0 GiB at a 4096-token chunk).
pub const prompt_work: u64 = 2 << 30;
/// A decode step's working set past its KV (bounded by the prompt pass's; the bank's decode rows take no more room).
pub const decode_work: u64 = 2 << 30;
/// Free RAM a request at the full context leaves the box past the host's wired margin.
pub const free_ram: u64 = 4 << 30;
/// MEM_REFIT (the bill before C2; C2's `passWork` folds its pass part into the fitted W, and plugin.zig `promptOnlyBytes`
/// keeps its server part): a layer-major prompt pass's peak past that bill's terms, and the server's own memory past
/// MLX's counters, both measured. Fitted on our M5 Max (2026-11-10, the 2.60
/// pack) at 31,652 prompt tokens (`--ctx-size 32768`, max_tokens 1,024) on memfit v3 (records/bin/memfit.py), whose terms
/// are the trunk, the bank's rows and its prompt windows (the host stream's transient scratch, MLX memory until the phase
/// change), the KV mlx-serve allocated (32,000 tokens), `prompt_work` and the layer-major term. The runs: the context
/// ladder at c8d54a83 (8 runs, MLX prompt peak 98,807 MiB on 202 rows, 987 MiB past the terms;
/// records/ladder-c8d54a83), and R8b's verification at 901a8e4a (a pre-warm and 4 runs at 100,091 to 100,092 MiB on
/// 206 rows, 988 to 989 MiB past; records/ladder-rf). The pass part is the worst remainder, 989 MiB. When a prompt
/// pass peaks below the warm-up's decode peak, its `[mem] a prompt pass ends` line shows that decode peak, not its own.
/// mlx-serve grows its KV lazily, and an append a span would take it 25,600 -> 32,000 -> 40,192 tokens, past this bill's
/// 32,768-token term inside the pass, for a prompt over 32,000 tokens. D2 (src/ling_rq.zig `d2_from`) appends each MLA
/// layer's spans past 24,576 in one call, so no whole prompt up to the term grows it past the term: RQ serves whole
/// prompts up to 32,768 tokens and refuses by name a prompt forward whose KV would pass the term. RQ bills each request's
/// KV at the capacity mlx-serve allocates for it, in place of that term.
/// The server's peak footprint (time -l) held at most 544 MiB past its run's MLX peak and MLX's 8,192 MiB buffer pool
/// (2026-11-08); the read pool's staging (host memory) is inside it, and the host's wired margin holds both. The extra
/// is their sum, 1,533 MiB, and `free_ram` stays the box's reserve. A constant, never a function of a request's length.
pub const mem_refit_extra: u64 = mem_refit_pass + mem_refit_server;
/// MEM_REFIT's part a prompt pass uses: its wide calls' peak past this bill's other terms, idle once the pass's forward
/// has returned, except about 269 MiB (calculated) that is the host's resident draft-rerank head (`echo_entry_bytes`).
pub const mem_refit_pass: u64 = 989 << 20;
/// MEM_REFIT's other part: the server's own memory past MLX's counters, which no step can use.
pub const mem_refit_server: u64 = 544 << 20;

/// Bytes a prompt pass keeps free of the bank (`Bill.prefill_bytes`) with the KV of `kv_tokens` tokens (C1: the served
/// context's longest whole prompt's, src/ling_rq.zig `promptCapacityMax`).
pub fn prompt(kv_per_token: u64, kv_tokens: u64) u64 {
    return kv_per_token * kv_tokens + prompt_work + free_ram;
}

/// Bytes a decode step keeps free of the bank, with the KV of `kv_tokens` tokens.
pub fn decode(kv_per_token: u64, kv_tokens: u64) u64 {
    return kv_per_token * kv_tokens + decode_work + free_ram;
}

/// The bank's slot-row budgets for `room` (the host's stream room, which already leaves `prompt` and `prompt_only` out)
/// with `in_use` bytes resident: decode takes what its smaller reserve leaves, and the bytes a prompt pass alone holds
/// (`prompt_only`: the layer-major pass's routed working set and checkpoints, `Bill.prefill_bytes`'s rest) back. Both
/// reserves hold the same KV term, so it cancels here.
pub const Budget = struct { prompt: u64, decode: u64 };
pub fn budget(kv_per_token: u64, prompt_only: u64, room: u64, in_use: u64) Budget {
    const p = room -| in_use;
    return .{ .prompt = p, .decode = p + (prompt(kv_per_token, default_context) + prompt_only -| decode(kv_per_token, default_context)) };
}

/// The latent cache's KV a token: `attn_layers` MLA layers' normed latent and roped shared key (`headLatentKv`).
pub fn latentKv(attn_layers: u64, kv_lora: u64, rope: u64) u64 {
    return attn_layers * (kv_lora + rope) * 2;
}

/// One MLA layer's KV a token on the rope-once cache, the served MLA cache (src/ling_mla_rope1.zig), bf16: kv_b's rows
/// (each head's nope keys and values) and the roped key once, `heads` x (nope + v) + rope words, against the expanded
/// cache's `heads` x (nope + rope + v).
pub fn ropeOnceLayer(heads: u64, nope: u64, rope: u64, v: u64) u64 {
    return (heads * (nope + v) + rope) * 2;
}

/// The rope-once cache's fallback a cached token (src/ling_mla_rope1.zig `expand`): a prompt call neither rope-once path
/// takes rebuilds the expanded key, `heads` x (nope + rope) bf16, alive for the call's attention alone. The values stay
/// views of the cached rows: at one sequence MLX 0.32 reads them in place on both of its paths (the vector kernels'
/// `kv_copy_unless` keeps a key or value of batch 1; the composed path's matmul, `check_transpose`, keeps an operand
/// whose last stride is 1, at its own row stride), so no copy of them is billed. Every prompt call of 17 rows or more
/// takes the rope-once flash launch, so only a call of 16 rows or fewer rebuilds the key, when the prompt pass's working
/// set past its KV is idle: the spike at the full context sits inside `prompt_work` (the test pins it), so the bill holds
/// no term for it.
pub fn ropeOnceSpike(heads: u64, nope: u64, rope: u64) u64 {
    return heads * (nope + rope) * 2;
}

/// The KV a token with the MTP head's MLA layer on its latent cache (src/ling_mtp_latent.zig): the trunk's MLA layers'
/// KV (`trunk_per_layer` each: as served, the rope-once cache's, `ropeOnceLayer`) and the head's latent and shared rope
/// key. The head's prompt calls only append to its cache, so no prompt-chunk expansion is billed for it.
pub fn headLatentKv(trunk_layers: u64, trunk_per_layer: u64, kv_lora: u64, rope: u64) u64 {
    return trunk_layers * trunk_per_layer + latentKv(1, kv_lora, rope);
}

/// A layer-major prompt call's routed working set past an engine chunk's: each extra row's `top_k` pairs' expert
/// outputs and their weighted copies, f32 (the call's `call_rows` against the chunk's `chunk_rows`).
pub fn layerMajorWork(call_rows: u64, chunk_rows: u64, top_k: u64, hidden: u64) u64 {
    return (call_rows -| chunk_rows) * top_k * hidden * 4 * 2;
}

// ── C2: the layer-major pass's working set at the served context ──

/// C2: what a layer-major prompt pass of a whole prompt holds past its KV, W, as memfit v3 reads it: the MLX prompt peak
/// less the trunk, the bank's prompt rows, its prompt windows and the KV mlx-serve allocated (at the trunk's capacity, a
/// token's bytes for both caches). The model's terms, each a function of the prompt's length:
/// - `prompt_work`;
/// - the widest routed call's rows past an engine chunk's (`layerMajorWork`);
/// - the checkpoints alive (src/ling_rq.zig `checkpointsOf`: a pass holds its own until every layer has passed them);
/// - the KV growth's copy: one MLA layer's old buffer at the last growth (`layer_kv` a token);
/// - D2's group: kv_b's rows and their joined copy, `group` a token, for the rows from 24,576 on;
/// - a fitted part, `pass_fit_rate` a token less `pass_fit_offset`.
/// The fit (2026-11-11) is exact at the two measured lengths and over every other measurement:
/// - 16,377 tokens (the 01ca9769 ladder's 8 runs, MLX prompt peak at most 92,887 MiB on 214 rows): W 3,717.5 MiB;
/// - 31,652 tokens under D2 (ac0d03c1's 32K check, the worst of 4 runs, 99,465 MiB on 214 rows): W 5,869.2 MiB;
/// - over at 18,380 (f03b37e9's A/B, at most 94,857 MiB) by 53 MiB, at f03b37e9's own 32K check (99,422) by 43, and at
///   D2's safety legs (32,000 to 32,765 tokens, at most 99,666 MiB) by 19 to 141 MiB.
/// Each peak is read as its logged MiB plus one, and the in-use bytes at their logged MiB, so W is an upper bound. Past
/// 31,652 tokens W is extrapolated by the terms (0.14 MiB a token); the 32K and 64K kill-mode checks measure it.
/// Under 16,377 tokens W is 16,377's: a shorter prompt's pass has narrower calls, fewer checkpoints and a smaller copy.
/// The routed calls are the copied table_forward.zig `moeSpans`' (C4: src/ling_segments.zig `routedCalls`): over the
/// prompt's spans (src/ling_rq.zig `Spans`), a call takes the next span while its rows stay at or under `call_rows`, or
/// while the rows left after it are fewer than `tail_rows`.
/// `segment_rows` 0 runs the whole prompt in one pass (this build). C4's segments (src/ling_segments.zig: whole calls, at
/// most `segment_rows` rows a segment unless fewer than `segment_tail_rows` rows would remain after it, each segment
/// through every layer before the next) take each segment's terms at once: its widest call; its D2 group (its rows past 24,576); its growth's copy (the spans before 24,576 each append
/// alone; from it, two or more append once at the segment's end); its own checkpoints beside the host's (at most
/// `host_checkpoints`); the fitted part on its rows less a hidden row's; and the MTP history's hidden rows (`hidden` x 2
/// B a token, written once into one array, src/ling_lm_pass.zig `HiddenRows`): the segment's own in the first segment,
/// the whole prompt's from the second on (the array holds every row from the first segment's end). One segment is the
/// whole pass's W.
pub const Pass = struct {
    top_k: u64,
    hidden: u64,
    /// One checkpoint's bytes: every KDA layer's bf16 state and conv window (the sdk's `ssmCheckpointBytes(2)`).
    checkpoint: u64,
    /// One MLA layer's KV a token (`ropeOnceLayer`).
    layer_kv: u64,
    /// D2's group a token: heads x (nope + v) x 2 B, and its joined copy.
    group: u64,
    /// The routed calls' rows (plugin.zig `layer_major_rows`, `layer_major_tail_rows`).
    call_rows: u64,
    tail_rows: u64,
    /// C4's segment bound in rows; 0: one pass.
    segment_rows: u64 = 0,
    /// C4's segment tail: a segment is not cut while fewer rows than this would remain after it (so a segment holds up
    /// to `segment_rows` + this - 1 rows).
    segment_tail_rows: u64 = 0,
};
pub const pass_fit_from: u64 = 16377;
pub const pass_fit_rate: u64 = 51_528;
pub const pass_fit_offset: u64 = 3_282_562_824;
/// The checkpoints the host keeps at most (mlx-serve's `ssm_checkpoint_max`, 16): what C4's earlier segments leave.
pub const host_checkpoints: u64 = 16;

/// The routed calls over a whole prompt's spans (`Pass`): `next` gives each call's end.
const PassCalls = struct {
    spans: ling_rq.Spans,
    /// The next span's end.
    ahead: ?u64,
    /// The next call's first position.
    at: u64 = 0,
    n: u64,
    call_rows: u64,
    tail_rows: u64,

    fn init(n: u64, call_rows: u64, tail_rows: u64) PassCalls {
        var s = ling_rq.Spans.init(n);
        const first = s.next();
        return .{ .spans = s, .ahead = first, .n = n, .call_rows = call_rows, .tail_rows = tail_rows };
    }

    fn next(c: *PassCalls) ?u64 {
        var end = c.ahead orelse return null;
        c.ahead = c.spans.next();
        const left = c.n - c.at;
        while (c.ahead) |e| : (c.ahead = c.spans.next()) {
            if (e - c.at > c.call_rows and left - (end - c.at) >= c.tail_rows) break;
            end = e;
        }
        c.at = end;
        return end;
    }
};

/// One run of the pass model over a whole prompt of `n` tokens: W (`passWork`), the KV capacity mlx-serve allocates for
/// the prompt as the pass appends it (one pass: src/ling_rq.zig `wholePromptCapacity`; the test holds the two equal),
/// and its segments: their count, and each one's end (a position) in `seg_ends` while it has room (C4's tie test holds
/// them to src/ling_segments.zig's, which the pass runs).
const PassRun = struct { w: u64, cap: u64, segs: usize };

fn passRun(p: Pass, n: u64, seg_ends: ?[]u64) PassRun {
    const hidden_row = p.hidden * 2;
    const d2 = ling_rq.d2_from;
    var calls = PassCalls.init(n, p.call_rows, p.tail_rows);
    var call_end = calls.next();
    var spans = ling_rq.Spans.init(n);
    var span_end = spans.next();
    var best: u64 = 0;
    var cap: u64 = 0;
    var filed: u64 = 0;
    var lo: u64 = 0;
    var segs: usize = 0;
    while (call_end) |first_call| {
        // the segment: whole calls while its rows stay at or under `segment_rows` (0: every call) or fewer than
        // `segment_tail_rows` rows would remain after it, at least one
        var hi = first_call;
        var widest = hi - lo;
        call_end = calls.next();
        while (call_end) |e| : (call_end = calls.next()) {
            if (p.segment_rows != 0 and e - lo > p.segment_rows and n - hi >= p.segment_tail_rows) break;
            widest = @max(widest, e - hi);
            hi = e;
        }
        // the segment's spans, those ending in (lo, hi]: how many, and the first that starts at D2's position or later
        var look = spans;
        var e_look = span_end;
        var count: u64 = 0;
        var first_d2: ?u64 = null;
        var s_look = lo;
        while (e_look) |e| : (e_look = look.next()) {
            if (e > hi) break;
            if (first_d2 == null and s_look >= d2) first_d2 = count;
            count += 1;
            s_look = e;
        }
        const g = if (first_d2) |f| (if (count - f >= 2) f else count) else count;
        // its appends, its growth's copy (the last growth's old buffer) and its own checkpoints
        var old: u64 = 0;
        var own: u64 = 0;
        var i: u64 = 0;
        while (i < count) : (i += 1) {
            const e = span_end.?;
            if (i < g or i + 1 == count) {
                const c = ling_rq.nextCapacity(cap, e);
                if (c != cap) old = cap;
                cap = c;
            }
            if (e != n) own += 1;
            span_end = spans.next();
        }
        const alive = @min(filed, host_checkpoints) + own;
        const hidden_rows = if (lo == 0) hi else n;
        const w = (prompt_work + layerMajorWork(widest, 4096, p.top_k, p.hidden) + alive * p.checkpoint + (hi -| @max(lo, d2)) * p.group +
            old * p.layer_kv + (pass_fit_rate -| hidden_row) * (hi - lo) + hidden_row * hidden_rows) -| pass_fit_offset;
        best = @max(best, w);
        filed += own;
        if (seg_ends) |se| {
            if (segs < se.len) se[segs] = hi;
        }
        segs += 1;
        lo = hi;
    }
    return .{ .w = best, .cap = cap, .segs = segs };
}

/// C2: W for a whole prompt of `n_in` tokens from position 0 (`Pass`).
pub fn passWork(p: Pass, n_in: u64) u64 {
    return passRun(p, @max(n_in, pass_fit_from), null).w;
}

/// C2: the W the bill holds for a served context: the most over every whole prompt up to it. W is not monotone in the
/// prompt's length (the widest routed call narrows past each tail-merge point; a prompt whose backoff span ends just
/// under a 256-token step keeps a smaller copy than one a token shorter), so each length from the fit's floor is read:
/// about 151,000 at the 128K point's context, milliseconds at load.
pub fn passWorkBill(p: Pass, context: u64) u64 {
    var most = passWork(p, pass_fit_from);
    var n: u64 = pass_fit_from + 1;
    while (n <= context) : (n += 1) most = @max(most, passWork(p, n));
    return most;
}

/// The MTP draft readout's bytes (`[vocab, hidden]`, ling_mtp.zig): bf16 at 16 bits, else MLX affine at `bits` with a
/// bf16 scale and bias a group of `group` along a row. It is resident when the bank opens, so the bank's rows hold
/// what it leaves.
pub fn draftReadout(vocab: u64, hidden: u64, bits: u64, group: u64) u64 {
    if (bits == 16) return vocab * hidden * 2;
    return vocab * hidden * bits / 8 + 2 * vocab * (hidden / group) * 2;
}

/// EMBED_ROWS (src/ling_embed.zig): the bytes the bank's slot rows take from the embedding table, `vocab` x `hidden`
/// bf16, which the trunk never makes resident: each forward reads its rows from the pack. `openBank` reads MLX's memory
/// in use when the bank opens, and the table is not in it, so both phases' slot rows take its bytes. No reserve of this
/// bill moves: a forward's rows array is the size of the resident table's `take` output it replaced, made at the
/// forward's start (a prompt pass's, before its routed calls' peak), and its host buffer is given back at once.
pub fn embedRowsFreed(vocab: u64, hidden: u64) u64 {
    return vocab * hidden * 2;
}

/// A prompt pass asked for every row's logits (`Logits.all_rows`: the engine asks it only for prompt log-probabilities,
/// since the plugin declares `Caps.prompt_last_row`): every span's logits and their joined copy, bf16, both alive at the
/// engine's eval (the mem-zhc A/B's arm A measured it). The bill holds no term for it: the engine's step over those
/// rows holds more, and `echoRefusalAt` refuses a prompt whose step would not fit.
pub fn headRowsJoined(rows: u64, vocab: u64) u64 {
    return 2 * rows * vocab * 2;
}

/// The engine's prompt log-probability step at its peak, in bytes for each row and vocabulary entry (mlx-serve 66aad949
/// sampling/logprobs.zig `rowLogprobs`). A whole-prompt prefill runs the step over every row of its one forward
/// (generate/init.zig, after the forward). The chunk loop runs it one chunk at a time, 256 rows for an echo request
/// (prefill_interleave.zig `PROMPT_LOGPROBS_CHUNK`).
/// Measured, rounded up: 31.6 B. That is test 3 on our M5 Max, 2026-11-09, at R7b 9700aa6
/// (records/ladder-r7/ladder-r7echo.234907). Before a 4,171-token echo request MLX held 82,174 MiB active, and at its peak
/// 103,094 MiB. Less the request's 1,182 MiB of KV, the step took 19,738 MiB over 4,171 x 157,184 entries. The re-run at
/// R7c 0e1f5c0 (records/ladder-r7/ladder-r7echo) measured 30.48 B at 2,940 rows, from /props before and after.
/// The host's own account (rowLogprobs' comment) is 22 B: the bf16 logits it holds through the step and five 4-byte
/// arrays. MLX 0.32.5 (c69303b0) holds six. ArgPartition allocates its output (mlx/backend/metal/sort.cpp:348) beside the
/// f32 input and the multi-block sort's four scratch arrays (sort.cpp:142-146), so the account is 26 B. 2 B more are this
/// plugin's. `spanMajorOr` evaluates each span's logits and returns their join unevaluated. MLX releases an op's inputs
/// only when its command buffer completes (mlx/backend/metal/eval.cpp:47-71), so the spans' copy is alive through the
/// engine's step. The other 2.5 to 3.6 B are not explained.
/// The room the step is checked against (`EchoPlan`, `echoIdle`) counts the layer-major term (4,797 MiB at the served
/// plan) since R7e: the pass's arrays are idle at the step (main, 2026-11-10, the strict fit; the refusal test at
/// 0e1f5c0 served 2,940 tokens with MLX's peak 5,411 MiB under the plan's line, room that the term fits). Since R7e-c
/// the room leaves out the host's one resident draft-rerank head (`rerankHead`, 281,673,728 B): a coarse 3-bit copy of
/// the readout at group size 64 (mlx-serve 66aad949 src/sdk/spec/mtp/mtp.zig:742, built from
/// src/engines/mlx/plugin/head.zig:304 after the bank opens; it replaces the draft-only copy, mtp.zig:628-633). The head
/// is not idle at the step: it stays resident from load on, and the measured part of MEM_REFIT holds it, so counting it
/// as idle overstated the room. The left-out layer-major term covered that while the room did not count the term; with
/// the term counted, the bill holds at its own 32 B only without the head.
pub const echo_entry_bytes: u64 = 32;

/// The room a prompt log-probability step may fill. The plan holds KV for `kv_tokens` tokens of `kv_per_token` bytes
/// each (C1: the served context's longest whole prompt's, src/ling_rq.zig `promptCapacityMax`), and a forward's KV fills
/// part of it. `idle` is the bill's terms counted as idle from the end of a prompt forward until decode's first forward
/// (`echoIdle`; `echo_entry_bytes` says what is not idle). At that forward (ling_source.zig `setPhase`) the decode rows
/// take the layer-major term and the server's part back (C2), and decode's own working set replaces `prompt_work`. The
/// engine's step runs before it. `vocab` is the head's width.
pub const EchoPlan = struct { kv_tokens: u64, kv_per_token: u64, vocab: u64, idle: u64 };

/// The plan's room once a forward's KV reaches `end` tokens (as served, mlx-serve's capacity for its positions): the KV
/// the plan holds past them, and `idle`. KV past the plan's takes its bytes out of `idle`.
fn echoRoom(plan: EchoPlan, end: u64) u64 {
    return (plan.kv_tokens * plan.kv_per_token + plan.idle) -| end * plan.kv_per_token;
}

/// The plan's idle terms (plugin.zig `openBank`): `prompt_work`, and with the layer-major pass its working set past it at
/// the served context (C2: `passWorkBill` less `prompt_work`, plugin.zig `layerMajorBytes`, where MEM_REFIT's pass part
/// is folded in; 0 without the pass). `resident_bytes` is memory resident through the step that the pass's fit holds
/// (the host's rerank head, `rerankHead`).
pub fn echoIdle(layer_major: bool, pass_bytes: u64, resident_bytes: u64) u64 {
    return (prompt_work + (if (layer_major) pass_bytes else 0)) -| resident_bytes;
}

/// The host's one resident draft-rerank head (`echo_entry_bytes`): a coarse 3-bit copy of the `vocab` x `hidden` readout
/// at group size 64 (mlx-serve 66aad949 src/sdk/spec/mtp/mtp.zig `buildRerankCoarse` at DRAFT_HEAD_BITS 3).
pub fn rerankHead(vocab: u64, hidden: u64) u64 {
    return draftReadout(vocab, hidden, 3, 64);
}

/// Rule 7's limit: the longest whole prompt (every shorter too) whose step fits the room at mlx-serve's KV for it.
pub fn echoRowsMax(plan: EchoPlan) u64 {
    return echoRowsMaxHost(plan);
}

/// Rule 7 at a forward's positions (as served, `echoRefusalAt` at mlx-serve's KV capacity). A prefill forward that asks
/// every row's logits (prompt log-probabilities) is refused by name when its step would not fit the plan's room at its
/// last position: the reason is written into `buf`. A last-row prompt, a verify and a decode step are never refused.
pub fn echoRefusal(buf: []u8, plan: EchoPlan, prefill: bool, all_rows: bool, first: u64, rows: u64) ?[]const u8 {
    if (!prefill or !all_rows) return null;
    const need = rows * plan.vocab * echo_entry_bytes;
    const room = echoRoom(plan, first + rows);
    if (need <= room) return null;
    return echoWhy(buf, plan, first, rows, need, room);
}

/// Ling-3.1-flash's KV a token as mlx-serve's bailing_hybrid bills it: 9 MLA layers (one in each 8-layer group) x 64
/// heads x (192 + 128) x 2 bytes.
const ling_kv: u64 = 9 * 64 * (192 + 128) * 2;

test "Ling's reserves hold the full context's KV, the measured working set and the free RAM" {
    const kv32k: u64 = ling_kv * 32768;
    try std.testing.expectEqual(@as(u64, 12_079_595_520), kv32k);
    try std.testing.expectEqual(kv32k + (6 << 30), prompt(ling_kv, default_context));
    try std.testing.expect(decode(ling_kv, default_context) <= prompt(ling_kv, default_context));
}

test "the standard test's load: the bank gives up the reserve the old 8 GiB bill did not hold" {
    // our M5 Max, 2.60: a 96 GiB room under the old 8 GiB reserve (the static ceiling less it and the 8 GiB margin),
    // 23,783 MiB resident.
    const ceiling_less_margin: u64 = (98304 << 20) + (8 << 30);
    const in_use: u64 = 23783 << 20;
    const b = budget(ling_kv, 0, ceiling_less_margin - prompt(ling_kv, default_context), in_use);
    const old = (98304 << 20) - in_use;
    try std.testing.expectEqual(old - (prompt(ling_kv, default_context) - (8 << 30)), b.prompt);
    // The prompt peak at the full context sits `free_ram` under the margin: trunk + rows + KV + working set.
    try std.testing.expectEqual(ceiling_less_margin - free_ram, in_use + b.prompt + ling_kv * default_context + prompt_work);
    try std.testing.expect(b.decode >= b.prompt);
    // A room the trunk already fills leaves the bank nothing (the bank refuses that by name).
    try std.testing.expectEqual(@as(u64, 0), budget(ling_kv, 0, 1 << 30, 2 << 30).prompt);
}

test "a prompt-only reserve (the layer-major pass's) comes out of the prompt rows only: decode takes it back" {
    const room: u64 = 100 << 30;
    const in_use: u64 = 20 << 30;
    const lm: u64 = 5 << 30;
    const without = budget(ling_kv, 0, room, in_use);
    // The host's room already leaves the prompt-only bytes out (Bill.prefill_bytes holds them).
    const with = budget(ling_kv, lm, room - lm, in_use);
    try std.testing.expectEqual(without.prompt - lm, with.prompt);
    try std.testing.expectEqual(without.decode, with.decode);
}

test "the MTP head alone on the latent cache: 1,244 MiB less KV at the full context, 3.9 rows: 4 more prompt and decode rows at the zhl A/B's bank line" {
    // our M5 Max, 2.60, the control (2026-11-07, e10 kept): room 82,754 MiB, 16,844 MiB in use, the layer-major
    // pass's 4,797 MiB back to the decode rows: prompt 65,909 MiB, decode 70,707 MiB (201 / 219 rows of 336,427,008 B).
    // 9 MLA layers and the NextN head's, expanded, against the 9 expanded and the head's latent.
    const row: u64 = 336_427_008;
    const layer: u64 = 64 * (192 + 128) * 2;
    const ex_kv: u64 = 10 * layer;
    const hl = headLatentKv(9, layer, 512, 64);
    try std.testing.expectEqual(@as(u64, 369_792), hl);
    try std.testing.expectEqual(@as(u64, 1244 << 20), (ex_kv - hl) * default_context);
    const lm: u64 = 4797 << 20;
    const in_use: u64 = 16844 << 20;
    const room_e: u64 = 82754 << 20;
    // The GPU ceiling less the host's wired margin; the host's room leaves `prefillBytes` out of it.
    const top = room_e + prompt(ex_kv, default_context) + lm;
    const e = budget(ex_kv, lm, room_e, in_use);
    const h = budget(hl, lm, top - (prompt(hl, default_context) + lm), in_use);
    // Both phases' rows take the whole difference.
    try std.testing.expectEqual(e.prompt + (prompt(ex_kv, default_context) - prompt(hl, default_context)), h.prompt);
    try std.testing.expectEqual(e.decode + (decode(ex_kv, default_context) - decode(hl, default_context)), h.decode);
    // 1,244 MiB is 3.88 rows. A phase's slot rows are its budget's whole rows, so the bank gains 3 or 4 as the budget's
    // fraction of a row falls; here both phases sit at about .4 of a row and each gains 4 (the zhl A/B, our M5 Max
    // 2026-11-07: 201 prompt / 219 decode rows without the head's latent cache, 205 / 223 with it).
    try std.testing.expectEqual(@as(u64, 3), (h.prompt - e.prompt) / row);
    try std.testing.expectEqual(@as(u64, 3), (h.decode - e.decode) / row);
    try std.testing.expectEqual(@as(u64, 4), h.prompt / row - e.prompt / row);
    try std.testing.expectEqual(@as(u64, 4), h.decode / row - e.decode / row);
}

test "the rope-once cache: 2,268 MiB less KV at the full context over the 9 MLA layers, 7.07 rows, the head's latent cache as served; both phases' rows take the difference" {
    // Ling-3.1-flash's MLA layer, 64 heads x (nope 128, rope 64, v 128): 40,960 B a token expanded, 32,896 rope-once.
    const row: u64 = 336_427_008;
    const layer: u64 = 64 * (192 + 128) * 2;
    const r1 = ropeOnceLayer(64, 128, 64, 128);
    try std.testing.expectEqual(@as(u64, 32_896), r1);
    try std.testing.expectEqual(@as(u64, 8_064), layer - r1);
    // The served KV (the 9 MLA layers expanded, the NextN head's latent) against the rope-once trunk and the same head.
    const hl = headLatentKv(9, layer, 512, 64);
    const hr = headLatentKv(9, r1, 512, 64);
    try std.testing.expectEqual(@as(u64, 297_216), hr);
    try std.testing.expectEqual(@as(u64, 2268 << 20), (hl - hr) * default_context);
    // The zhl A/B's bank line (our M5 Max, 2026-11-07), the head's latent on: both phases' rows take the whole
    // difference, 7.07 rows of 336,427,008 B.
    const lm: u64 = 4797 << 20;
    const in_use: u64 = 16844 << 20;
    const top: u64 = (82754 << 20) + prompt(10 * layer, default_context) + lm;
    const e = budget(hl, lm, top - (prompt(hl, default_context) + lm), in_use);
    const r = budget(hr, lm, top - (prompt(hr, default_context) + lm), in_use);
    try std.testing.expectEqual(e.prompt + (prompt(hl, default_context) - prompt(hr, default_context)), r.prompt);
    try std.testing.expectEqual(e.decode + (decode(hl, default_context) - decode(hr, default_context)), r.decode);
    try std.testing.expectEqual(@as(u64, 7), (r.prompt - e.prompt) / row);
    try std.testing.expectEqual(@as(u64, 7), (r.decode - e.decode) / row);
}

test "the rope-once fallback's spike: the rebuilt key, 768 MiB at the full context, inside the prompt pass's idle working set at a call of 16 rows or fewer, so the bill holds no term for it" {
    const mib: u64 = 1 << 20;
    const spike = ropeOnceSpike(64, 128, 64);
    // 24 KiB a cached token: 805,306,368 B (768 MiB) at 32,768 tokens.
    try std.testing.expectEqual(@as(u64, 24_576), spike);
    try std.testing.expectEqual(@as(u64, 805_306_368), spike * default_context);
    try std.testing.expectEqual(@as(u64, 768) * mib, spike * default_context);
    // With the rope-once flash launch on the wide calls, only a call of 16 rows or fewer rebuilds the key: the prompt
    // pass's working set past its KV (prompt_work, fitted at 4,096-row chunks) is idle then, and the spike fits inside it.
    // MEM_REFIT's pass part no longer holds it as well: of its 989 MiB, about 269 MiB is the host's resident 3-bit rerank
    // head (`echo_entry_bytes`), and the idle rest is smaller than the spike.
    try std.testing.expect(spike * default_context <= prompt_work);
    try std.testing.expect(spike * default_context > mem_refit_pass - draftReadout(157_184, 4096, 3, 64));
}

test "the 4-bit draft readout: its 882 MiB go to both phases' rows, at least 2 more rows of each at today's bank line" {
    // Ling-3.1-flash's head: 157,184 x 4,096. The control (our M5 Max, 2.60, 2026-11-07): room 82,754 MiB, 16,844 MiB
    // held when the bank opens, the dense readout among it (201 prompt / 219 decode rows of 336,427,008 B).
    const vocab: u64 = 157184;
    const hidden: u64 = 4096;
    const dense = draftReadout(vocab, hidden, 16, 0);
    const q4 = draftReadout(vocab, hidden, 4, 64);
    try std.testing.expectEqual(@as(u64, 1_287_651_328), dense);
    try std.testing.expectEqual(@as(u64, 321_912_832 + 2 * 20_119_552), q4);
    const freed = dense - q4;
    try std.testing.expectEqual(@as(u64, 882), freed >> 20);
    const row: u64 = 336_427_008;
    const ex_kv: u64 = 10 * 64 * (192 + 128) * 2;
    const lm: u64 = 4797 << 20;
    const room: u64 = 82754 << 20;
    const in_use: u64 = 16844 << 20;
    const off = budget(ex_kv, lm, room, in_use);
    const on = budget(ex_kv, lm, room, in_use - freed);
    try std.testing.expectEqual(off.prompt + freed, on.prompt);
    try std.testing.expectEqual(off.decode + freed, on.decode);
    try std.testing.expectEqual(@as(u64, 2), freed / row);
}

test "EMBED_ROWS: the embedding table's 1,228 MiB are not in the bank's in-use reading, so at R8c's bank line the prompt rows are 214 and the decode rows 236" {
    const mib: u64 = 1 << 20;
    const row: u64 = 336_427_008;
    // Ling-3.1-flash's table: 157,184 x 4,096 bf16, 3.83 rows of every routed layer's slot
    const freed = embedRowsFreed(157_184, 4096);
    try std.testing.expectEqual(@as(u64, 1_287_651_328), freed);
    try std.testing.expectEqual(@as(u64, 1228), freed >> 20);
    try std.testing.expectEqual(@as(u64, 3), freed / row);
    // R8c's bank line (MEM_REFIT's test), read with the table resident: room 84,733 MiB, 15,962 MiB in use. The table
    // comes out of the in-use reading, nothing else, so both phases' budgets take its bytes.
    const kv = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64);
    const lm: u64 = 4797 * mib;
    const room: u64 = 84733 * mib;
    const r8c_in_use: u64 = 15962 * mib;
    const r8c = budget(kv, lm + mem_refit_extra, room, r8c_in_use);
    const served = budget(kv, lm + mem_refit_extra, room, r8c_in_use - freed);
    try std.testing.expectEqual(r8c.prompt + freed, served.prompt);
    try std.testing.expectEqual(r8c.decode + freed, served.decode);
    // The rows at R8c's logged budgets (prompt 66,153 MiB at c8d54a83 plus the 2,617 MiB the re-fit gave back; decode
    // 75,101 MiB) with the table's bytes, less the bank's windows and staging as MEM_REFIT's test counts them, at
    // either end of the MiB the log rounds to: 214 / 236 (the prompt budget's 0.46 of a row left over takes a fourth
    // row).
    const record: u64 = row / 68;
    const window: u64 = 48 * record;
    const staging: u64 = 16 * (std.mem.alignForward(u64, record, 16384) + 2 * 16384) + mib + 4 * (9 * mib);
    const r8c_prompt: u64 = 66153 * mib + ((3606 << 20) + mem_refit_server - mem_refit_extra);
    try std.testing.expectEqual(@as(u64, 68770) * mib, r8c_prompt);
    for ([_]u64{ 0, mib - 1 }) |frac| {
        try std.testing.expectEqual(@as(u64, 214), (r8c_prompt + frac + freed - (5 * window + staging)) / row);
        try std.testing.expectEqual(@as(u64, 236), (75101 * mib + frac + freed - (window + staging)) / row);
    }
}

test "a layer-major call bills its routed rows past an engine chunk's; a call no wider bills nothing" {
    try std.testing.expectEqual(@as(u64, (18432 - 4096) * 8 * 4096 * 8), layerMajorWork(16384 + 2048, 4096, 8, 4096));
    try std.testing.expectEqual(@as(u64, 0), layerMajorWork(4096, 4096, 8, 4096));
}

test "the all-rows head (prompt log-probabilities): every span's logits and their joined copy, 20,602,421,248 B at the full context, which the bill does not hold" {
    const vocab: u64 = 157184;
    try std.testing.expectEqual(@as(u64, 20_602_421_248), headRowsJoined(default_context, vocab));
    // The standard test's prompt (18,380 rows): 11,020 MiB.
    try std.testing.expectEqual(@as(u64, 11020), headRowsJoined(18380, vocab) >> 20);
    // Past the prompt pass's only term beside the bank (MEM_REFIT's extra) at either length.
    try std.testing.expect(headRowsJoined(18380, vocab) > mem_refit_extra);
}

test "MEM_REFIT: every control prompt peak at 31,652 tokens, the bank's windows counted, fits the re-fit's bill; the prompt rows grow from 202 to 210, decode keeps its 233, and at 32,000 prompt tokens the peak stays under the bill's line; with D2 no whole prompt up to the bill's 32,768-token context passes its KV term" {
    // our M5 Max, the 2.60 pack, 31,652-token prompts (mlx-serve allocated KV for 32,000), every measured control run:
    // the context ladder at c8d54a83 (PORTR7C, 2026-11-10: 4 greedy, 4 at the card's sampling; records/ladder-c8d54a83)
    // on 202 prompt rows, and R8b's verification at 901a8e4a (a pre-warm and 4 greedy runs; records/ladder-rf) on 206.
    // The ladder's bank line: room 82,116 MiB, 15,962 MiB in use, prompt 66,153 MiB / decode 75,101 MiB, 202 / 233 rows
    // of 336,427,008 B. Its KV a token: the rope-once trunk and the MTP head's latent cache.
    const mib: u64 = 1 << 20;
    const row: u64 = 336_427_008;
    const kv = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64);
    const lm: u64 = 4797 * mib;
    const in_use: u64 = 15962 * mib;
    // c8d54a83's MEM_REFIT extra: the pass part fitted on 2026-11-08 at 18,380 tokens, 3,606 MiB, and the server's 544.
    const extra_c: u64 = (3606 << 20) + mem_refit_server;
    const room_c: u64 = 82116 * mib;
    // The host's ceiling less its wired margin: the room it handed the bank, with the bill's prompt reserve, the
    // layer-major term and c8d54a83's extra put back. The re-fit's room is that line less the re-fit's reserves.
    const top = room_c + prompt(kv, default_context) + lm + extra_c;
    const room = top - (prompt(kv, default_context) + lm + mem_refit_extra);
    try std.testing.expectEqual(@as(u64, 84733), room >> 20);
    const c = budget(kv, lm + extra_c, room_c, in_use);
    const r = budget(kv, lm + mem_refit_extra, room, in_use);
    // The prompt pass's alone: the prompt rows gain the 2,617 MiB the re-fit gives back, and decode keeps its budget.
    try std.testing.expectEqual(c.prompt + (extra_c - mem_refit_extra), r.prompt);
    try std.testing.expectEqual(c.decode, r.decode);
    try std.testing.expectEqual(@as(u64, 75101), r.decode >> 20);
    // The bank's windows and staging (ling_source.zig `billBank`): 68 routed layers of 4,947,456 B a record, 5 prompt
    // windows (1 at decode) of 48 records, and staging for 16 lookahead records at 16 KiB pages plus 1 MiB, and 4
    // workers' 9 MiB past the pool's default 4. The logged budgets are whole MiB; the rows hold at either end.
    const record: u64 = row / 68;
    try std.testing.expectEqual(row, 68 * record);
    const window: u64 = 48 * record;
    const staging: u64 = 16 * (std.mem.alignForward(u64, record, 16384) + 2 * 16384) + mib + 4 * (9 * mib);
    try std.testing.expectEqual(@as(u64, 118_489_088), staging);
    for ([_]u64{ 0, mib - 1 }) |frac| {
        try std.testing.expectEqual(@as(u64, 202), (66153 * mib + frac - (5 * window + staging)) / row);
        try std.testing.expectEqual(@as(u64, 210), (66153 * mib + frac + (extra_c - mem_refit_extra) - (5 * window + staging)) / row);
        try std.testing.expectEqual(@as(u64, 233), (75101 * mib + frac - (window + staging)) / row);
    }
    // The fit, memfit v3 (records/bin/memfit.py): the terms are the trunk, the bank's rows, its prompt windows (the host
    // stream's transient scratch, MLX memory from `Stream.init` until the phase change; the staging is the read pool's
    // host memory, inside the server's 544 MiB), the KV mlx-serve allocated, prompt_work and the layer-major term, rounded
    // to the MiB. The ladder's 8 runs peaked at 98,807 MiB on 202 rows, 987 MiB past 97,820. R8b's verification (the
    // pre-warm, then runs 1-4) peaked at 100,092, 100,091, 100,092, 100,091 and 100,091 MiB on 206 rows, 989, 988, 989,
    // 988 and 988 MiB past 99,103. The pass part is the worst, 989 MiB.
    const terms = in_use + 202 * row + 5 * window + kv * 32000 + prompt_work + lm;
    try std.testing.expectEqual(@as(u64, 97820), (terms + mib / 2) >> 20);
    try std.testing.expect(terms + mem_refit_pass >= 98807 * mib);
    const terms_rf = terms + 4 * row;
    try std.testing.expectEqual(@as(u64, 99103), (terms_rf + mib / 2) >> 20);
    try std.testing.expectEqual(@as(u64, 989) * mib, mem_refit_pass);
    try std.testing.expect(terms_rf + mem_refit_pass >= 100092 * mib);
    // At 32,000 prompt tokens (RQ's served maximum before D2, src/ling_rq.zig) mlx-serve's KV is the fit's own 32,000
    // tokens; an append a span grows it one token later to 40,192, past this bill's 32,768-token term. D2 appends each
    // MLA layer's spans from 24,576 on at once, so one token more takes it to 32,256, and no whole prompt up to the term
    // passes it: RQ's served maximum is the term itself (the long-prompt legs measure the peak there). With the
    // remainder still rising at its rate from 18,380 to 31,652 tokens (smoke Q8's -2,484 MiB to the ladder's +987: 91 MiB
    // more at 32,000), R8b's worst peak on 4 more rows and the server's 544 MiB stay 388 MiB under the bill's line.
    try std.testing.expectEqual(top - free_ram, in_use + r.prompt + kv * default_context + prompt_work + lm + mem_refit_extra);
    try std.testing.expectEqual(@as(u64, 32000), ling_rq.wholePromptCapacity(32000));
    var ends: [16]u64 = undefined;
    try std.testing.expectEqual(@as(u64, 40192), ling_rq.replay(0, ling_rq.spanEnds(&ends, 32001)).cap);
    try std.testing.expectEqual(@as(u64, 32256), ling_rq.wholePromptCapacity(32001));
    try std.testing.expectEqual(default_context, ling_rq.promptCapacityMax(default_context));
    const rise = (987 + 2484) * mib * (32000 - 31652) / (31652 - 18380);
    try std.testing.expectEqual(@as(u64, 91), rise >> 20);
    const peak_max = 100092 * mib + 4 * row + rise;
    try std.testing.expectEqual(@as(u64, 388), (top - free_ram - (peak_max + mem_refit_server)) >> 20);
}

// Ling-3.1-flash's pass geometry (plugin.zig `passOf`): top 8 of 4,096-wide rows; a checkpoint of the 63 KDA layers'
// bf16 state and conv window (the sdk's `ssmCheckpointBytes`: 64 x 128 x 128 x 2 B of state and 3 x 24,576 x 2 B of
// window a layer); one rope-once MLA layer's KV a token; D2's group (64 heads x (128 + 128) x 2 B, and its joined copy);
// routed calls of 16,384 rows, a tail under 2,048 joining the call before it. `ling_pass_c4` is C4's: segments of at
// most 32,768 rows, not cut while fewer than 8,192 rows would remain after one.
const ling_pass: Pass = .{ .top_k = 8, .hidden = 4096, .checkpoint = 63 * (64 * 128 * 128 * 2 + 3 * (2 * 64 * 128 + 64 * 128) * 2), .layer_kv = ropeOnceLayer(64, 128, 64, 128), .group = 64 * (128 + 128) * 2 * 2, .call_rows = 16384, .tail_rows = 2048 };
const ling_pass_c4: Pass = .{ .top_k = 8, .hidden = 4096, .checkpoint = 63 * (64 * 128 * 128 * 2 + 3 * (2 * 64 * 128 + 64 * 128) * 2), .layer_kv = ropeOnceLayer(64, 128, 64, 128), .group = 64 * (128 + 128) * 2 * 2, .call_rows = 16384, .tail_rows = 2048, .segment_rows = 32768, .segment_tail_rows = 8192 };

// Rule 7's plan as served (plugin.zig `openBank`): the bill's context, the KV a token (the rope-once trunk and the MTP
// head's latent cache), Ling-3.1-flash's 157,184-word head, and the idle terms: prompt_work and the layer-major pass's
// working set past it at the context (C2: `passWorkBill`, MEM_REFIT's pass part folded in), less the host's rerank head
// (281,673,728 B), which is resident through the step. Made at run time: `passWorkBill` reads every whole prompt from
// 16,377 tokens up to the context.
fn echoToday() EchoPlan {
    return .{
        .kv_tokens = default_context,
        .kv_per_token = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64),
        .vocab = 157_184,
        .idle = echoIdle(true, passWorkBill(ling_pass, default_context) - prompt_work, rerankHead(157_184, 4096)),
    };
}
// The same plan at TURBOEXL3_LING_LAYER_MAJOR=0: no layer-major pass, so no term past prompt_work.
fn echoChunked() EchoPlan {
    return .{ .kv_tokens = default_context, .kv_per_token = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64), .vocab = 157_184, .idle = echoIdle(false, 0, rerankHead(157_184, 4096)) };
}

test "rule 7 at the served plan (32,768 tokens, 297,216 B of KV a token, 157,184 words, C2's pass working set at the context): mlx-serve allocates 3,072 tokens of KV for a 2,948- or 2,949-token whole prompt, so a 2,948-token prompt's log-probabilities are served and a 2,949-token prompt's are refused by name" {
    var buf: [320]u8 = undefined;
    const echo = echoToday();
    const pass_bytes = passWorkBill(ling_pass, default_context) -| prompt_work;
    try std.testing.expectEqual(@as(u64, 297_216), echo.kv_per_token);
    try std.testing.expectEqual(@as(u64, 4_137_423_096), pass_bytes);
    try std.testing.expectEqual(@as(u64, 281_673_728), rerankHead(157_184, 4096));
    try std.testing.expectEqual(@as(u64, 2948), echoRowsMax(echo));
    try std.testing.expect(echoRefusalAt(&buf, echo, true, true, 0, 2948, ling_rq.wholePromptCapacity(2948)) == null);
    const why = echoRefusalAt(&buf, echo, true, true, 0, 2949, ling_rq.wholePromptCapacity(2949)) orelse return error.TestExpectedRefusal;
    try std.testing.expect(std.mem.startsWith(u8, why, "turboexl3_ling: prompt log-probabilities"));
    try std.testing.expect(std.mem.indexOf(u8, why, "up to 2948 tokens") != null);
    // The room at mlx-serve's 3,072 tokens of KV (the KV of 32,768 - 3,072 tokens, prompt_work and the pass's working set
    // past it, less the rerank head) holds 2,948 rows' step and not 2,949 rows'.
    try std.testing.expectEqual(@as(u64, 3072), ling_rq.wholePromptCapacity(2948));
    try std.testing.expectEqual(@as(u64, 3072), ling_rq.wholePromptCapacity(2949));
    const room: u64 = (default_context - 3072) * echo.kv_per_token + prompt_work + pass_bytes - rerankHead(157_184, 4096);
    try std.testing.expectEqual(@as(u64, 14_829_359_352), room);
    try std.testing.expect(2948 * echo.vocab * echo_entry_bytes <= room and 2949 * echo.vocab * echo_entry_bytes > room);
}

test "rule 7: a last-row (or no-logits) prompt forward, a verify and a decode step are never refused, at any rows and position up to the context" {
    var buf: [320]u8 = undefined;
    const echo = echoToday();
    for ([_]u64{ 1, 2, 4, 8, 16, 31, 32, 256, 1578, 3318, 4096, 16384, 32768 }) |rows| {
        var first: u64 = 0;
        while (first + rows <= default_context) : (first += 1) {
            try std.testing.expect(echoRefusal(&buf, echo, true, false, first, rows) == null);
            try std.testing.expect(echoRefusal(&buf, echo, false, true, first, rows) == null);
        }
    }
}

test "rule 7: an all-rows prompt forward of 1,193 rows or fewer is never refused at any position up to the context (the warm-up's 8 and 32 rows, the chunk loop's 256-row echo chunks and 31-row tail), and one of 1,194 rows ending at the context is" {
    var buf: [320]u8 = undefined;
    const echo = echoToday();
    for ([_]u64{ 1, 8, 31, 32, 256, 1193 }) |rows| {
        var first: u64 = 0;
        while (first + rows <= default_context) : (first += 1) try std.testing.expect(echoRefusal(&buf, echo, true, true, first, rows) == null);
    }
    try std.testing.expect(echoRefusal(&buf, echo, true, true, default_context - 1194, 1194) != null);
    // LAYER_MAJOR=0: no whole-prompt prefill, so the chunk loop asks at most 256 rows at a time: every 256-row chunk is
    // served at every position (a last chunk that merges a tail, up to 767 rows, is served up to position 26,065).
    const chunked = echoChunked();
    var at: u64 = 0;
    while (at + 256 <= default_context) : (at += 1) try std.testing.expect(echoRefusal(&buf, chunked, true, true, at, 256) == null);
}

test "rule 7: the parity request's prompt (1,146 tokens) takes its log-probabilities as one whole prompt, with or without the layer-major pass's term" {
    var buf: [320]u8 = undefined;
    try std.testing.expect(echoRefusalAt(&buf, echoToday(), true, true, 0, 1146, ling_rq.wholePromptCapacity(1146)) == null);
    try std.testing.expect(echoRefusalAt(&buf, echoChunked(), true, true, 0, 1146, ling_rq.wholePromptCapacity(1146)) == null);
}

test "rule 7: the plan's idle room counts the layer-major pass's working set past prompt_work with the pass (R7e; C2: MEM_REFIT's pass part folded into it), and none of it without it; the resident rerank head comes off it either way (R7e-c)" {
    const head = rerankHead(157_184, 4096);
    const pass_bytes = passWorkBill(ling_pass, default_context) -| prompt_work;
    try std.testing.expectEqual(prompt_work + pass_bytes - head, echoIdle(true, pass_bytes, head));
    try std.testing.expectEqual(prompt_work - head, echoIdle(false, 0, head));
}

test "rule 7: the limit follows the plan's KV and room: a plan holding 16,384 tokens of KV gives 2,040 tokens, and the room without the prompt-only terms gives 1,800" {
    var buf: [320]u8 = undefined;
    const echo = echoToday();
    const half: EchoPlan = .{ .kv_tokens = 16384, .kv_per_token = echo.kv_per_token, .vocab = echo.vocab, .idle = echo.idle };
    try std.testing.expectEqual(@as(u64, 2040), echoRowsMax(half));
    try std.testing.expect(echoRefusalAt(&buf, half, true, true, 0, 2040, ling_rq.wholePromptCapacity(2040)) == null);
    try std.testing.expect(echoRefusalAt(&buf, half, true, true, 0, 2041, ling_rq.wholePromptCapacity(2041)) != null);
    const strict: EchoPlan = .{ .kv_tokens = echo.kv_tokens, .kv_per_token = echo.kv_per_token, .vocab = echo.vocab, .idle = 0 };
    try std.testing.expectEqual(@as(u64, 1800), echoRowsMax(strict));
}

// ── RQ (src/ling_rq.zig): rule 7 at the KV mlx-serve allocates ──

const ling_rq = @import("ling_rq.zig");
/// C4's segments, which the pass runs (C4's tie test holds the pass model's to them).
const ling_segments = @import("ling_segments.zig");

/// Rule 7's refusal as served (plugin.zig `forward`): `echoRefusal`, with the plan's room at the forward's end counting
/// the KV mlx-serve allocates for it, `kv_end` tokens (RQ: its capacity schedule replayed over the forward's spans), in
/// place of the forward's last position. `echoRefusal` is this at `first + rows`.
pub fn echoRefusalAt(buf: []u8, plan: EchoPlan, prefill: bool, all_rows: bool, first: u64, rows: u64, kv_end: u64) ?[]const u8 {
    if (!prefill or !all_rows) return null;
    const need = rows * plan.vocab * echo_entry_bytes;
    const room = echoRoom(plan, kv_end);
    if (need <= room) return null;
    return echoWhy(buf, plan, first, rows, need, room);
}

/// The longest whole prompt whose step fits the room its positions' KV leaves: `echoRowsMaxHost`'s bound.
fn echoRowsAtPositions(plan: EchoPlan) u64 {
    return (plan.kv_tokens * plan.kv_per_token + plan.idle) / (plan.vocab * echo_entry_bytes + plan.kv_per_token);
}

/// Rule 7's limit (`echoRowsMax`): the KV at a whole prompt's end is mlx-serve's capacity for it (src/ling_rq.zig
/// `wholePromptCapacity`), which moves in steps, so the limit is the length before the first whose step does not fit.
/// The capacity holds the prompt's positions, so it is never past the limit at them.
fn echoRowsMaxHost(plan: EchoPlan) u64 {
    const at_positions = echoRowsAtPositions(plan);
    var n: u64 = 1;
    while (n <= at_positions) : (n += 1) {
        if (n * plan.vocab * echo_entry_bytes > echoRoom(plan, ling_rq.wholePromptCapacity(n))) return n - 1;
    }
    return at_positions;
}

/// Rule 7's refusal text (`echoRefusal`, `echoRefusalAt`), its limit `echoRowsMax`'s.
fn echoWhy(buf: []u8, plan: EchoPlan, first: u64, rows: u64, need: u64, room: u64) []const u8 {
    return std.fmt.bufPrint(buf, "turboexl3_ling: prompt log-probabilities for a {d}-row prompt forward from position {d} need {d} MiB, past the {d} MiB this plan leaves at that length; it serves them for whole prompts up to {d} tokens", .{ rows, first, need >> 20, room >> 20, echoRowsMax(plan) }) catch "turboexl3_ling: prompt log-probabilities past the plan's room";
}

// ── C1's tests: the served context ──

test "C1: the context the bill holds is the load's ctx_size, as mlx-serve 66aad949 and 9a173773 write it, else the default 32,768 (no ctx_size, or 0: an auto-sized context, which config.zig clamps to it); settings that do not parse, or a ctx_size that is not a whole number up to 2^24, are refused by name" {
    // driver.zig `loadSettings`' forms: the manual context after the lossy, MTP and streaming settings, the KV precision
    // after it
    try std.testing.expectEqual(@as(u64, 49975), try contextOf("{\"ctx_size\":49975}"));
    try std.testing.expectEqual(@as(u64, 83694), try contextOf("{\"lm_head_mxfp8\":false,\"mtp\":false,\"ctx_size\":83694}"));
    try std.testing.expectEqual(@as(u64, 150999), try contextOf("{\"stream_experts\":true,\"ctx_size\":150999}"));
    try std.testing.expectEqual(@as(u64, 17442), try contextOf("{\"ctx_size\":17442,\"kv_quant\":\"bf16\"}"));
    try std.testing.expectEqual(default_context, try contextOf(""));
    try std.testing.expectEqual(default_context, try contextOf("{}"));
    try std.testing.expectEqual(default_context, try contextOf("{\"lm_head_mxfp8\":false}"));
    try std.testing.expectEqual(default_context, try contextOf("{\"ctx_size\":0}"));
    try std.testing.expectEqual(max_context, try contextOf("{\"ctx_size\":16777216}"));
    for ([_][]const u8{ "{\"ctx_size\":-1}", "{\"ctx_size\":16777217}", "{\"ctx_size\":4.9e4}", "{\"ctx_size\":\"49975\"}", "{\"ctx_size\":null}", "[49975]", "{ctx_size:1}", "{\"ctx_size\":99999999999999999999999}" }) |bad|
        try std.testing.expectError(error.LingCtxSizeUnreadable, contextOf(bad));
}

test "C1: both reserves hold the KV of the served context's longest whole prompt: 32,768 tokens at the default context, as before, 21,248 at the sweep's 1K point (ctx 17,442) and 50,176 at its 32K point (ctx 49,975)" {
    const kv = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64);
    try std.testing.expectEqual(kv * 32768 + prompt_work + free_ram, prompt(kv, ling_rq.promptCapacityMax(default_context)));
    try std.testing.expectEqual(kv * 21248 + prompt_work + free_ram, prompt(kv, ling_rq.promptCapacityMax(17442)));
    try std.testing.expectEqual(kv * 50176 + decode_work + free_ram, decode(kv, ling_rq.promptCapacityMax(49975)));
}

test "C1: the bank's room at the served context: mlx-serve prices its room at the reserve plugin.zig answers before the load, the default context's, and the load moves it by the difference to the served context's reserve; the port's 84,733 MiB at the default context, 3,848,159,232 B (3,669.9 MiB) more at the sweep's 1K point (11,520 fewer tokens of KV, 3 fewer checkpoints), none where the served reserve takes the ceiling less the margin" {
    const mib: u64 = 1 << 20;
    const kv = headLatentKv(9, ropeOnceLayer(64, 128, 64, 128), 512, 64);
    const cp: u64 = 63 * (64 * 128 * 128 * 2 + 3 * (2 * 64 * 128 + 64 * 128) * 2);
    // plugin.zig `prefillAt` before C2: the KV, prompt_work and free RAM, the layer-major term at the context's
    // checkpoints, and MEM_REFIT's extra
    const R = struct {
        fn at(k: u64, c: u64, ctx: u64) u64 {
            return prompt(k, ling_rq.promptCapacityMax(ctx)) + layerMajorWork(16384 + 2048, 4096, 8, 4096) + ling_rq.checkpointsOf(ctx) * c + mem_refit_extra;
        }
    };
    const host_room: u64 = 84733 * mib;
    const priced = R.at(kv, cp, default_context);
    try std.testing.expectEqual(host_room, roomAt(host_room, priced, priced).?);
    const at1k = R.at(kv, cp, 17442);
    try std.testing.expectEqual(@as(u64, 11520 * 297_216 + 3 * cp), priced - at1k);
    try std.testing.expectEqual(@as(u64, 3_848_159_232), priced - at1k);
    try std.testing.expectEqual(host_room + (priced - at1k), roomAt(host_room, priced, at1k).?);
    const at32k = R.at(kv, cp, 49975);
    try std.testing.expectEqual(host_room - (at32k - priced), roomAt(host_room, priced, at32k).?);
    try std.testing.expect(roomAt(host_room, priced, host_room + priced) == null);
    try std.testing.expect(roomAt(0, priced, priced) == null);
}

// ── C2's and C5's tests: the pass's working set, the load it gives, and RQ's plans against it ──

/// W as memfit v3 reads it (`Pass`) from a whole prompt of `n` tokens whose MLX prompt peak logged `peak_mib` (read as one
/// MiB more) on the port's bank line: 14,734 MiB in use, 214 prompt rows of 336,427,008 B, 5 prompt windows of 48
/// records, and the KV mlx-serve allocated (`wholePromptCapacity`) at 297,216 B a token.
fn wMeasured(peak_mib: u64, n: u64) u64 {
    const row: u64 = 336_427_008;
    return ((peak_mib + 1) << 20) -| ((14734 << 20) + 214 * row + 5 * 48 * (row / 68) + ling_rq.wholePromptCapacity(n) * 297_216);
}

/// RQ's terms at a load on our M5 Max as plugin.zig `openBank` makes them (C1, C2, C5), at a served context of `ctx`
/// tokens with the pass's bill `w` (`passWorkBill` at the context): the host's ceiling less its wired margin, 104 GiB
/// (the port's room, 84,733 MiB, under f03b37e9's reserve at 32,768 tokens), less the context's reserve; 14,734 MiB in
/// use and `uf` B past it (the MiB the bank line rounds to); the bank's rows at the budgets.
fn lingTerms(ctx: u64, w: u64, uf: u64) ling_rq.Terms {
    const row: u64 = 336_427_008;
    const kv: u64 = 297_216;
    const kv_tokens = ling_rq.promptCapacityMax(ctx);
    const prompt_only = (w -| prompt_work) + mem_refit_server;
    const b = budget(kv, prompt_only, (104 << 30) - (prompt(kv, kv_tokens) + prompt_only), (14734 << 20) + uf);
    const bank: ling_rq.Bank = .{ .row_bytes = row, .prompt_windows = 5 * 48 * (row / 68), .decode_windows = 48 * (row / 68), .staging = 118_489_088, .n_experts = 512 };
    const phases: ling_rq.Budget = .{ .prompt = b.prompt, .decode = b.decode };
    return .{ .bank = bank, .budget = phases, .context = ctx, .kv_tokens = kv_tokens, .kv = .{ .layer = 32_896, .layers = 9, .head = 1_152 }, .rows = ling_rq.rowsOf(bank, phases) catch unreachable, .pass_bill = w, .resident = rerankHead(157_184, 4096), .free_ram = free_ram };
}

/// The 32K check's request (31,652 tokens): its KV capacity after its spans, 32,000 (the last growth's old capacity
/// 25,600), and the MTP head's 31,744 after its history.
const check32k: ling_rq.Slot = .{ .id = 1, .p = 31652, .trunk_cap = 32000, .head_cap = 31744, .prompt = .{ .cap = 32000, .old = 25600 } };

test "C2: the pass's working set W at a whole prompt's length is the fit of memfit v3's peaks: exact at 16,377 tokens (the 01ca9769 ladder, 92,887 MiB) and 2,840 B over at 31,652 (ac0d03c1's 32K check under D2, 99,465 MiB); over every other measured peak, by at most 141 MiB; under 16,377 tokens the floor's" {
    try std.testing.expectEqual(@as(u64, 3_898_052_608), wMeasured(92887, 16377));
    try std.testing.expectEqual(@as(u64, 3_898_052_608), passWork(ling_pass, 16377));
    try std.testing.expectEqual(@as(u64, 6_154_260_480), wMeasured(99465, 31652));
    try std.testing.expectEqual(@as(u64, 6_154_263_320), passWork(ling_pass, 31652));
    // f03b37e9's A/B (18,380 tokens, at most 94,857 MiB) and its own 32K check (99,422), and D2's safety legs (32,000 to
    // 32,765 tokens)
    for ([_][2]u64{ .{ 18380, 94857 }, .{ 31652, 99422 }, .{ 32000, 99485 }, .{ 32198, 99496 }, .{ 32504, 99599 }, .{ 32689, 99658 }, .{ 32765, 99666 } }) |m| {
        const w = passWork(ling_pass, m[0]);
        try std.testing.expect(w > wMeasured(m[1], m[0]) and w - wMeasured(m[1], m[0]) <= 141 << 20);
    }
    try std.testing.expectEqual(passWork(ling_pass, 16377), passWork(ling_pass, 1058));
}

test "C2: the pass's routed calls are the copied moeSpans' (a span joins a call while the call stays at or under 16,384 rows, or while fewer than 2,048 rows are left after it): W drops 536,557,240 B from 18,431 tokens (one call) to 18,432 (16,384 rows, then 2,048), and a 33,268-token prompt's merged last chunk starts a call of its own; one pass's KV replay is RQ's at every length" {
    try std.testing.expectEqual(@as(u64, 4_818_487_216), passWork(ling_pass, 18431));
    try std.testing.expectEqual(@as(u64, 4_281_929_976), passWork(ling_pass, 18432));
    try std.testing.expectEqual(@as(u64, 536_557_240), passWork(ling_pass, 18431) -| passWork(ling_pass, 18432));
    // 33,268 tokens: spans of 4,096 rows up to 28,672, the last chunk with its 469-row tail (4,565 rows), then the
    // backoff's 31: calls of 16,384, 12,288 and 4,596 rows (by rows alone the second would take 16,884)
    var calls = PassCalls.init(33268, 16384, 2048);
    try std.testing.expectEqual(@as(?u64, 16384), calls.next());
    try std.testing.expectEqual(@as(?u64, 28672), calls.next());
    try std.testing.expectEqual(@as(?u64, 33268), calls.next());
    try std.testing.expectEqual(@as(?u64, null), calls.next());
    try std.testing.expectEqual(@as(u64, 6_343_438_744), passWork(ling_pass, 33268));
    var n: u64 = 1;
    while (n <= 50_000) : (n += 1) try std.testing.expectEqual(ling_rq.wholePromptCapacity(n), passRun(ling_pass, n, null).cap);
}

test "C2: the bill holds W at its most over every whole prompt up to the served context: 4,508,265,608 B at the sweep's 1K point (ctx 17,442), 6,284,906,744 at the default 32,768, 9,222,023,024 at the 32K point (49,975), 14,549,885,416 at the 64K point (83,694), 24,587,880,368 at the 128K point (150,999); with C4's segments the same up to 32,768 tokens, then 7,526,598,576, 10,772,727,728 and 14,596,753,328" {
    try std.testing.expectEqual(@as(u64, 4_508_265_608), passWorkBill(ling_pass, 17442));
    try std.testing.expectEqual(@as(u64, 6_284_906_744), passWorkBill(ling_pass, default_context));
    try std.testing.expectEqual(@as(u64, 6_284_906_744), passWorkBill(ling_pass_c4, default_context));
    try std.testing.expectEqual(@as(u64, 9_222_023_024), passWorkBill(ling_pass, 49975));
    try std.testing.expectEqual(@as(u64, 7_526_598_576), passWorkBill(ling_pass_c4, 49975));
    try std.testing.expectEqual(@as(u64, 14_549_885_416), passWorkBill(ling_pass, 83694));
    try std.testing.expectEqual(@as(u64, 10_772_727_728), passWorkBill(ling_pass_c4, 83694));
    try std.testing.expectEqual(@as(u64, 24_587_880_368), passWorkBill(ling_pass, 150999));
    try std.testing.expectEqual(@as(u64, 14_596_753_328), passWorkBill(ling_pass_c4, 150999));
    // W at the sweep's prompts past 32,768 tokens, one pass and C4's: 33,591 (C4: 16,384 + 17,207 rows), 67,310
    // (32,768 + 16,384 + 18,158) and 134,615 (3 x 32,768 + 36,311: the tail rule keeps the last 3,543 rows in the
    // segment before them)
    try std.testing.expectEqual(@as(u64, 6_738_405_232), passWork(ling_pass, 33591));
    try std.testing.expectEqual(@as(u64, 6_028_388_208), passWork(ling_pass_c4, 33591));
    try std.testing.expectEqual(@as(u64, 12_066_267_624), passWork(ling_pass, 67310));
    try std.testing.expectEqual(@as(u64, 9_100_370_408), passWork(ling_pass_c4, 67310));
    try std.testing.expectEqual(@as(u64, 21_742_781_552), passWork(ling_pass, 134615));
    try std.testing.expectEqual(@as(u64, 13_911_229_552), passWork(ling_pass_c4, 134615));
    // C4's segment tail: a 40,959-token prompt is one segment, the whole pass; at 40,960 the last 8,192 rows are a
    // segment of their own
    try std.testing.expectEqual(passWork(ling_pass, 40959), passWork(ling_pass_c4, 40959));
    try std.testing.expectEqual(@as(u64, 6_284_906_744), passWork(ling_pass_c4, 40960));
    for ([_]u64{ 16377, 18380, 25118, 31652, 32768, 33310 }) |k| try std.testing.expectEqual(passWork(ling_pass, k), passWork(ling_pass_c4, k));
    // C4's KV replay is one pass's at every length up to the 128K point's context and past it (134,615 tokens: 134,656,
    // the last growth from 98,304): each segment past 24,576 appends once, and no segment is left under 8,192 rows
    try std.testing.expectEqual(@as(u64, 134_656), passRun(ling_pass_c4, 134615, null).cap);
    var len: u64 = 1;
    while (len <= 160_000) : (len += 1) try std.testing.expectEqual(ling_rq.wholePromptCapacity(len), passRun(ling_pass_c4, len, null).cap);
}

test "C4 tie: the bill's pass model cuts every whole prompt from 2 to 150,999 tokens into the segments the pass runs (src/ling_segments.zig `Plan` over the host's spans, as plugin.zig `spanMajorOr` runs it), at plugin.zig `passOf`'s served geometry: 33,184 and 40,959 tokens one segment, 40,960 two (32,768 + 8,192), 134,615 four (3 x 32,768 + 36,311)" {
    // plugin.zig `passOf` with the layer-major pass: the routed calls' rows and the segments' bound and tail from the
    // constants the pass runs at; ling_pass_c4 (the C2 and C5 tests' model) is that geometry
    var served = ling_pass_c4;
    served.call_rows = ling_segments.routed_rows;
    served.tail_rows = ling_segments.tail_rows;
    served.segment_rows = ling_segments.segment_rows;
    served.segment_tail_rows = ling_segments.segment_tail_rows;
    try std.testing.expectEqual(ling_pass_c4, served);
    var model: [64]u64 = undefined;
    var ends: [64]u64 = undefined;
    var rows: [64]u32 = undefined;
    var n: u64 = 2;
    while (n <= 150_999) : (n += 1) {
        const run = passRun(served, n, model[0..]);
        // the pass's segments over the host's spans (src/ling_rq.zig `Spans`), each one's end as a position
        const e = ling_rq.spanEnds(&ends, n);
        for (e, 0..) |x, i| rows[i] = @intCast(x - (if (i == 0) 0 else e[i - 1]));
        var plan: ling_segments.Plan = undefined;
        try plan.init(rows[0..e.len], ling_segments.segment_rows);
        var k: usize = 0;
        var at: u64 = 0;
        while (plan.next()) |sg| : (k += 1) {
            at += sg.rows;
            if (k >= run.segs or model[k] != at) {
                std.debug.print("C4 tie: at {d} tokens the pass's segment {d} ends at {d}, the bill's model's at {d} ({d} segments)\n", .{ n, k, at, if (k < run.segs) model[k] else 0, run.segs });
                return error.TestUnexpectedResult;
            }
        }
        try std.testing.expectEqual(k, run.segs);
    }
    // the sweep's points past 32,768 tokens and the tail rule's edge
    const C = struct { n: u64, segs: []const u64 };
    for ([_]C{
        .{ .n = 33184, .segs = &.{33184} },
        .{ .n = 33591, .segs = &.{ 16384, 33591 } },
        .{ .n = 40959, .segs = &.{40959} },
        .{ .n = 40960, .segs = &.{ 32768, 40960 } },
        .{ .n = 67310, .segs = &.{ 32768, 49152, 67310 } },
        .{ .n = 134615, .segs = &.{ 32768, 65536, 98304, 134615 } },
    }) |c| {
        const run = passRun(served, c.n, model[0..]);
        try std.testing.expectEqualSlices(u64, c.segs, model[0..run.segs]);
    }
}

test "C2: the load at the served context: the bank's rows at the bill's KV and W terms, at either end of the in-use MiB: 220 prompt / 236 decode at --ctx-size 32,768 (the port's bill: 214 / 236), 235 / 247 at the sweep's 1K point, 219 / 236 at its 16K point, 195 / 221 at its 32K point with one pass and 200 / 221 with C4's segments; the decode budget holds at any W" {
    // The port's bill (f03b37e9: the layer-major term at 9 checkpoints, 5,030,789,120 B, and MEM_REFIT's extra) leaves
    // its logged room.
    try std.testing.expectEqual(@as(u64, 84733), ((104 << 30) - (prompt(297_216, default_context) + 5_030_789_120 + mem_refit_extra)) >> 20);
    const w_port = prompt_work + 5_030_789_120 + mem_refit_pass;
    const w32 = passWorkBill(ling_pass, default_context);
    const w1k = passWorkBill(ling_pass, 17442);
    const w16k = passWorkBill(ling_pass, 33184);
    const w32k = passWorkBill(ling_pass, 49975);
    const w32k_c4 = passWorkBill(ling_pass_c4, 49975);
    try std.testing.expectEqual(w16k, passWorkBill(ling_pass_c4, 33184));
    for ([_]u64{ 278_529, (1 << 20) - 1 }) |uf| {
        const port = lingTerms(default_context, w_port, uf);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 214, .decode = 236 }, port.rows);
        const served = lingTerms(default_context, w32, uf);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 220, .decode = 236 }, served.rows);
        try std.testing.expectEqual(port.budget.decode, served.budget.decode);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 235, .decode = 247 }, lingTerms(17442, w1k, uf).rows);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 219, .decode = 236 }, lingTerms(33184, w16k, uf).rows);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 195, .decode = 221 }, lingTerms(49975, w32k, uf).rows);
        try std.testing.expectEqual(ling_rq.Rows{ .prompt = 200, .decode = 221 }, lingTerms(49975, w32k_c4, uf).rows);
    }
}

test "C5: the 32K check is served: RQ's prompt plan gives the bill's KV and pass terms back and takes the request's own, so its 31,652-token prompt at --ctx-size 32,768 plans 221 rows against the base's 220; the plan before C5 (the growth's copy in its KV and the rerank head in O, beside the bill's pass term, which holds both) planned 217, under the base" {
    var buf: [320]u8 = undefined;
    const w = passWorkBill(ling_pass, default_context);
    try std.testing.expectEqual(@as(u64, 32000), ling_rq.wholePromptCapacity(31652));
    for ([_]u64{ 278_529, (1 << 20) - 1 }) |uf| {
        const t = lingTerms(default_context, w, uf);
        try std.testing.expectEqual(@as(u32, 220), t.rows.prompt);
        // O at its start: the rerank head (the check's RQ lines: O=269 MiB), at either end of the MiB
        for ([_]u64{ 269 << 20, (270 << 20) - 1 }) |o| {
            const pl = ling_rq.planPrompt(t, &check32k, o, passWork(ling_pass, 31652));
            try std.testing.expectEqual(@as(u64, 221), pl.held);
            try std.testing.expectEqual(ling_rq.OverVerdict.fits, ling_rq.overVerdict(t, pl));
            // before C5: both caches at their capacities with the growth's copy, and O whole
            const before = ling_rq.rowsAt(t, .prompt, 32000 * t.kv.trunk() + 31744 * t.kv.head + 25600 * t.kv.layer, o);
            try std.testing.expectEqual(@as(u64, 217), before.held);
            try std.testing.expect(before.over);
        }
        // its KV is within the plan's: RQ's prompt refusal passes it
        try std.testing.expect(ling_rq.promptRefusal(&buf, t, 0, 31652, 32000) == null);
    }
}

test "C5: a prompt plan under the base is refused by name, before any work: the 32K check's prompt with 1 GiB more in O plans 217 rows against the base's 220" {
    var buf: [320]u8 = undefined;
    const t = lingTerms(default_context, passWorkBill(ling_pass, default_context), 278_529);
    const pl = ling_rq.planPrompt(t, &check32k, (269 << 20) + (1 << 30), passWork(ling_pass, 31652));
    try std.testing.expectEqual(@as(u64, 217), pl.held);
    try std.testing.expect(pl.over);
    try std.testing.expectEqual(ling_rq.OverVerdict.refuse, ling_rq.overVerdict(t, pl));
    const why = ling_rq.overRefusal(&buf, t, &check32k, pl);
    try std.testing.expect(std.mem.startsWith(u8, why, ling_rq.over_refusal_name ++ ": slot 1 P=31652: the prompt plan holds 217 rows, 3 under the bank's base of 220"));
}

test "C5: a decode or trim plan under the base by at most the rows free RAM covers (12 rows of 336,427,008 B in 4 GiB) keeps the bank at its base with a named warning: the 32K check's decode plan (221 rows at its measured O, 1,730 MiB) with 13 rows more in O plans 208, 12 under the base's 220" {
    const row: u64 = 336_427_008;
    const o: u64 = 1730 << 20;
    for ([_]u64{ 278_529, (1 << 20) - 1 }) |uf| {
        const t = lingTerms(default_context, passWorkBill(ling_pass, default_context), uf);
        try std.testing.expectEqual(@as(u64, 12), ling_rq.coverRows(t));
        const at = ling_rq.planDecode(t, &check32k, .decode, 31652, 1, o);
        try std.testing.expectEqual(@as(u64, 221), at.held);
        try std.testing.expectEqual(ling_rq.OverVerdict.fits, ling_rq.overVerdict(t, at));
        const dec = ling_rq.planDecode(t, &check32k, .decode, 31652, 1, o + 13 * row);
        try std.testing.expectEqual(@as(u64, 208), dec.held);
        try std.testing.expectEqual(ling_rq.OverVerdict.warn, ling_rq.overVerdict(t, dec));
        try std.testing.expectEqual(ling_rq.OverVerdict.warn, ling_rq.overVerdict(t, ling_rq.planDecode(t, &check32k, .trim, 31652, 1, o + 13 * row)));
    }
}

test "C5: a decode or trim plan under the base by more than the rows free RAM covers is refused by name: the 32K check's decode plan with 14 rows more in O plans 207, 13 under the base's 220" {
    var buf: [320]u8 = undefined;
    const row: u64 = 336_427_008;
    const o: u64 = (1730 << 20) + 14 * row;
    for ([_]u64{ 278_529, (1 << 20) - 1 }) |uf| {
        const t = lingTerms(default_context, passWorkBill(ling_pass, default_context), uf);
        const dec = ling_rq.planDecode(t, &check32k, .decode, 31652, 1, o);
        try std.testing.expectEqual(@as(u64, 207), dec.held);
        try std.testing.expectEqual(ling_rq.OverVerdict.refuse, ling_rq.overVerdict(t, dec));
        const why = ling_rq.overRefusal(&buf, t, &check32k, dec);
        try std.testing.expect(std.mem.startsWith(u8, why, ling_rq.over_refusal_name ++ ": slot 1 P=31652: the decode plan holds 207 rows, 13 under the bank's base of 220 (free RAM covers 12)"));
        try std.testing.expectEqual(ling_rq.OverVerdict.refuse, ling_rq.overVerdict(t, ling_rq.planDecode(t, &check32k, .trim, 31652, 1, o)));
    }
}

test "C5 at the sweep's points: each prompt plan (O the rerank head) holds more rows than its base, one pass and C4's: 1K 254 / 235, 16K 244 / 219, 32K 217 / 195 (C4 219 / 200), 64K 172 / 150 (181 / 161), 128K 84 / 61 (107 / 90); the standard test's 18,380 tokens at 32,768: 235 / 220" {
    const Point = struct { ctx: u64, p: u64, pass: Pass, held: u64, base: u32 };
    for ([_]Point{
        .{ .ctx = 17442, .p = 1058, .pass = ling_pass, .held = 254, .base = 235 },
        .{ .ctx = 33184, .p = 16800, .pass = ling_pass, .held = 244, .base = 219 },
        .{ .ctx = 33184, .p = 16800, .pass = ling_pass_c4, .held = 244, .base = 219 },
        .{ .ctx = 49975, .p = 33591, .pass = ling_pass, .held = 217, .base = 195 },
        .{ .ctx = 49975, .p = 33591, .pass = ling_pass_c4, .held = 219, .base = 200 },
        .{ .ctx = 83694, .p = 67310, .pass = ling_pass, .held = 172, .base = 150 },
        .{ .ctx = 83694, .p = 67310, .pass = ling_pass_c4, .held = 181, .base = 161 },
        .{ .ctx = 150999, .p = 134615, .pass = ling_pass, .held = 84, .base = 61 },
        .{ .ctx = 150999, .p = 134615, .pass = ling_pass_c4, .held = 107, .base = 90 },
        .{ .ctx = default_context, .p = 18380, .pass = ling_pass, .held = 235, .base = 220 },
    }) |pt| {
        const w = passWorkBill(pt.pass, pt.ctx);
        const s: ling_rq.Slot = .{ .id = 1, .p = pt.p, .prompt = .{ .cap = passRun(pt.pass, pt.p, null).cap } };
        for ([_]u64{ 278_529, (1 << 20) - 1 }) |uf| {
            const t = lingTerms(pt.ctx, w, uf);
            try std.testing.expectEqual(pt.base, t.rows.prompt);
            for ([_]u64{ 269 << 20, (270 << 20) - 1 }) |o| {
                const pl = ling_rq.planPrompt(t, &s, o, passWork(pt.pass, pt.p));
                try std.testing.expectEqual(pt.held, pl.held);
                try std.testing.expectEqual(ling_rq.OverVerdict.fits, ling_rq.overVerdict(t, pl));
            }
        }
    }
}
