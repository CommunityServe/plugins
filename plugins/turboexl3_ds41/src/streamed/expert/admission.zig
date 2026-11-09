//! Memory admission of the expert streamer: how many slot rows each layer may
//! hold in prefill and decode, and the prefill-boundary bound, from the live
//! baseline, before anything is allocated. A port of the DSV4.1 Python stack's
//! admission as the tier runs it: packed_admission.retarget over the static
//! envelope (resolve_admission at base 0) with the phase-memory proof,
//! native_projection_memory.projection_fixed_bounds and the retirement
//! credits; the composite's members as charges on every post-prefill phase; the
//! 109.5 GB peak fill (AUTO control + filled run); the ROWSX slack credit with
//! the gate restore of its EXL3 port. Pure: it reads nothing and allocates
//! nothing. Every refusal is a named error, before any allocation.

const std = @import("std");
const host_env = @import("sdk").env;

pub const box_ceiling_bytes: u64 = 110_000_000_000;
pub const wired_ceiling_bytes: u64 = 100 << 30;
const gib: u64 = 1 << 30;
const mib: u64 = 1 << 20;
const n_layers: u64 = 40;
const transient_slots: u64 = 48;
/// The mxfp4 predecessor's records: prefill (weights + raw scales), decode weights, packed scale bank.
const raw_record: u64 = 18_800_640;
const weights_record: u64 = 17_694_720;
const packed_scales: u64 = 3_086_136_060;
const prefill_engine_remainder: u64 = 89_686_016;
const transform_reserve: u64 = 5 * gib;
const physical_headroom: u64 = gib;
const wired_headroom: u64 = gib;
/// native_projection_memory.
const projection_bytes: u64 = 67_108_864;
const packed_projection_bytes: u64 = 34_603_008;
const restored_projection_bytes: u64 = 40 * projection_bytes - 3 * projection_bytes - packed_projection_bytes;
pub const base_pipeline_bytes: u64 = 6 * projection_bytes;
/// native_retirement_memory.
const embedding_bytes: u64 = 1_323_827_200;
const tail_credit_bytes: u64 = 1_006_632_960 - 125_829_120;
const tail_rows_proved: u32 = 2048;
/// q3_peakfill_memory: the fill's target window is [box - 1 GiB (the unfilled search ceiling), 109.5 GB].
pub const max_target_bytes: u64 = 109_500_000_000;

/// The static envelope: the mxfp4 predecessor's admission at base 0 (the
/// native proof) and the phase-memory proof's constants. A calibration input:
/// `dsv41_pass2` is the Python stack's, re-derived by its own functions.
pub const Envelope = struct {
    /// The box the proof admits against (decimal); the allocator limit below is its own.
    box_bytes: u64 = box_ceiling_bytes,
    predecessor_decode_rows: u32,
    predecessor_prefill_rows: u32,
    decode_cache_bytes: u64,
    allocator_limit_at_base0: u64,
    host_reserve_bytes: u64,
    embedding_credit_bytes: u64,
    projection_credit_bytes: u64,
    steady_active_bytes: u64,
    prefill_active_bytes: u64,
    transition_start_bytes: u64,
    prefill_cache_bytes: u64,
    post_prefill_reserve_bytes: u64,
    projection_seed_fixed_bytes: u64,
    retirement_owners_proved: bool,
    /// Resident draft-head bytes the envelope's measured stack did not hold:
    /// the stack of record's compact DSpark head (run_full.py
    /// `MTP_PRUNED_BYTES`, its `text_only_resident_discount`). A process whose
    /// head leaves out fewer bytes is charged the difference on every phase.
    draft_pruned_bytes: u64 = 0,

    pub const dsv41_pass2: Envelope = .{
        .predecessor_decode_rows = 112,
        .predecessor_prefill_rows = 84,
        .decode_cache_bytes = 268_435_456,
        .allocator_limit_at_base0 = 108_561_226_752,
        .host_reserve_bytes = 1_438_773_248,
        .embedding_credit_bytes = 1_323_827_200,
        .projection_credit_bytes = 2_448_424_960,
        .steady_active_bytes = 97_762_795_628,
        .prefill_active_bytes = 89_965_493_340,
        .transition_start_bytes = 79_683_400_948,
        .prefill_cache_bytes = 1_160_216_040,
        .post_prefill_reserve_bytes = 1_301_547_008,
        .projection_seed_fixed_bytes = 17_536_326_524,
        .retirement_owners_proved = true,
        // 201 of the head's 3 x 128 experts x 18,800,640 B (the 09-13 trace selection, 93 / 58 / 32 kept).
        .draft_pruned_bytes = 3_778_928_640,
    };
};

pub const Allocation = enum { uniform, prefill_excess };
/// Aligned I/O staging of the read pool: 36 MiB for gate/up-first reads, else 32 MiB.
pub const IoLayout = enum { segments, record, gate_up };

/// The box an admission fits: the physical ceiling of the process's box, the GPU's wired ceiling, the
/// modeled-peak target's upper end and the rows a layer may hold. Null in `Inputs`: the envelope's own box (the
/// 110 GB / 100 GiB calibration, fills to 109.5 GB, 160 rows).
pub const Ceiling = struct {
    box_bytes: u64,
    wired_bytes: u64,
    max_target_bytes: u64,
    max_rows: u32,

    /// A box whose ceiling is the GPU's working set (the wired limit): the admitted modeled peak lands within
    /// `stop_bytes` of it; the fill's window ends 0.5 GB under the box, as the envelope's does.
    pub fn ofWorkingSet(working_set_bytes: u64, stop_bytes: u64, max_rows: u32) Ceiling {
        const target = working_set_bytes - stop_bytes;
        return .{ .box_bytes = target + (box_ceiling_bytes - max_target_bytes), .wired_bytes = working_set_bytes, .max_target_bytes = target, .max_rows = max_rows };
    }
};

pub const PeakFill = struct {
    /// Decimal-GB modeled-peak target, in [box - 1 GiB, the ceiling's max target]; null: that max.
    target_bytes: ?u64 = null,
    /// The named evidence credit (wide allowance + compiler reserve + tensor margin).
    evidence_credit_bytes: u64 = 2_737_047_552,
};

/// The ROWSX slack credit (EXL3 port of record): its evidence and the stack
/// geometry it was measured on.
pub const Rowsx = struct {
    credit_bytes: u64 = 1_611_661_312,
    mlx_side_credit_bytes: u64 = 1_044_381_696,
    decode_fixed_bytes: u64 = 14_491_882_096,
    prime_fixed_bytes: u64 = 15_201_501_308,
    /// host reserve + decode cache on the evidence stack
    non_mlx_bytes: u64 = 1_886_515_200 + 268_435_456,
    target_bytes: u64 = max_target_bytes,
};

pub const Inputs = struct {
    /// Non-file baseline and wired bytes at construction.
    baseline_bytes: u64,
    wired_bytes: u64,
    /// One expert record = one slot (EXL3 3.0: 13,315,584).
    record_bytes: u64,
    /// Forced decode rows (84..the ceiling's rows), or null for the largest that fits.
    fixed_rows: ?u32 = null,
    /// The largest decode rows searched (null: the ceiling's rows); an oracle launch searches up to its rows.
    search_ceiling: ?u32 = null,
    /// The box (null: the envelope's own).
    ceiling: ?Ceiling = null,
    /// The oracle's prefill capacity, which the admitted prefill must equal.
    matched_prefill_rows: ?u32 = null,
    allocation: Allocation = .prefill_excess,
    io_layout: IoLayout = .gate_up,
    embedding_rows: bool = true,
    tail_rows: ?u32 = tail_rows_proved,
    /// Active bytes every post-prefill phase carries beyond the proof's own
    /// owners: the base pipeline reserve plus every member charge except the
    /// lookahead staging (below).
    phase_reserve_bytes: u64,
    /// The lookahead pool's speculative staging charge; paid out of the ROWSX
    /// credit when that is on, else charged like the other members.
    lookahead_staging_bytes: u64 = 0,
    /// The wide lane's read-ahead windows: transient rows past the first 48,
    /// resident from construction, charged in every phase, the prefill included.
    wide_window_bytes: u64 = 0,
    /// Host bytes beyond the envelope's (pipeline host, comparison reserve).
    host_reserve_bytes: u64,
    /// The prefill members' charge on the prefill cache allowance.
    prefill_charge_bytes: u64 = 0,
    peak_fill: ?PeakFill = .{},
    rowsx: ?Rowsx = null,
    /// Resident draft-head bytes this process leaves out (0: the full head;
    /// a subset head: its pruned experts' bytes). Null: the envelope's own
    /// head. The admission charges `envelope.draft_pruned_bytes` minus this
    /// on every phase, the prefill included.
    draft_pruned_bytes: ?u64 = null,
};

pub const Phase = enum { growth, seed, prime, decode };
pub const Phases = struct {
    growth: u64,
    seed: u64,
    prime: u64,
    decode: u64,

    fn max(p: Phases) u64 {
        return @max(@max(p.growth, p.seed), @max(p.prime, p.decode));
    }

    /// The binding phase (the first maximum in growth, seed, prime, decode order).
    fn binding(p: Phases) Phase {
        const m = p.max();
        return if (p.growth == m) .growth else if (p.seed == m) .seed else if (p.prime == m) .prime else .decode;
    }
};

/// One admission (packed_admission.retarget's result, the fields a caller uses).
pub const Admission = struct {
    /// The whole admission for one construction: rows, bounds, and the
    /// peak-fill / ROWSX records, or the refusal by name.
    pub const plan = planAdmission;

    decode_rows: u32,
    /// Prefill rows in the predecessor's record (the engine budget's unit).
    prefill_rows: u32,
    /// Expert records the prefill bank holds per layer at that budget.
    prefill_capacity: u32,
    prefill_max_rows: u32,
    search_ceiling: u32,
    /// The MLX active the prefill-boundary retirement and the growth start may reach.
    transition_start_bytes: u64,
    retirement_entry_bytes: u64,
    phases: Phases,
    steady_bytes: u64,
    resize_bytes: u64,
    prefill_active_bytes: u64,
    prefill_physical_bytes: u64,
    prefill_cache_bytes: u64,
    physical_bound_bytes: u64,
    active_bound_bytes: u64,
    host_reserve_bytes: u64,
    allocator_limit_bytes: u64,
    decode_cache_bytes: u64,
    final_bank_bytes: u64,
    prefill_bank_bound_bytes: u64,
    embedding_credit_bytes: u64,
    tail_credit_bytes: u64,
    baseline_bytes: u64,
    wired_bytes: u64,
};

/// q3_peakfill_memory's receipt summary of one admission.
pub const Summary = struct {
    decode_rows: u32,
    prefill_rows: u32,
    prefill_max_rows: u32,
    final_bank_bytes: u64,
    phases: Phases,
    binding: Phase,
    binding_active_bytes: u64,
    binding_physical_bytes: u64,
    prefill_active_bytes: u64,
    prefill_physical_bytes: u64,
    physical_bound_bytes: u64,
    allocator_limit_bytes: u64,
    active_plus_cache_bytes: u64,
    wired_plus_active_plus_cache_bytes: u64,

    fn of(a: Admission) Summary {
        const m = a.phases.max();
        return .{
            .decode_rows = a.decode_rows,
            .prefill_rows = a.prefill_rows,
            .prefill_max_rows = a.prefill_max_rows,
            .final_bank_bytes = a.final_bank_bytes,
            .phases = a.phases,
            .binding = a.phases.binding(),
            .binding_active_bytes = m,
            .binding_physical_bytes = a.baseline_bytes + a.host_reserve_bytes + m + a.decode_cache_bytes,
            .prefill_active_bytes = a.prefill_active_bytes,
            .prefill_physical_bytes = a.prefill_physical_bytes,
            .physical_bound_bytes = a.physical_bound_bytes,
            .allocator_limit_bytes = a.allocator_limit_bytes,
            .active_plus_cache_bytes = m + a.decode_cache_bytes,
            .wired_plus_active_plus_cache_bytes = a.wired_bytes + m + a.decode_cache_bytes,
        };
    }
};

pub const PeakFillRecord = struct {
    target_bytes: u64,
    control: Summary,
    filled: Summary,
    total_credit_bytes: u64,
    modeled_peak_bytes: u64,
};

pub const RowsxRecord = struct {
    uncredited_rows: u32,
    credited_rows: u32,
    added_rows: u32,
    added_bytes: u64,
    credit_bytes: u64,
    mlx_side_credit_bytes: u64,
    modeled_peak_uncredited_bytes: u64,
    modeled_peak_credited_bytes: u64,
    target_bytes: u64,
    allocator_margin_bytes: i64,
    wired_margin_bytes: i64,
    box_margin_bytes: i64,
    baseline_bytes: u64,
    /// The prefill-boundary bound the credit would have lowered, restored.
    restored_gate_bytes: u64,
};

pub const Plan = struct {
    admission: Admission,
    peak_fill: ?PeakFillRecord = null,
    rowsx: ?RowsxRecord = null,
};

pub const Error = error{
    InvalidMatchedRows,
    InvalidFixedRows,
    FixedRowsAboveCeiling,
    TailRowsNotProved,
    RetirementProofMissing,
    ProjectionGeometry,
    ProjectionEnvelopeTooSmall,
    PhaseCreditExceedsEnvelope,
    RetirementCreditExceedsOwners,
    PrefillDoesNotFit,
    NoDecodeCapacity,
    FixedRowsDoNotFit,
    NoCausalPrefill,
    MatchedPrefillDoesNotFit,
    FixedRowsIncompatibleWithPrefill,
    InvalidTarget,
    PeakFillMovedPrefill,
    PeakFillReducedRows,
    PeakFillMovedFixedRows,
    PeakFillOverTarget,
    PeakFillOverAllocator,
    PeakFillOverWired,
    CreditWithFixedRows,
    CreditGeometry,
    CreditMovedPrefill,
    CreditAddsNoRow,
    CreditOverTarget,
    CreditOverAllocator,
    CreditOverWired,
    CreditReachedGate,
};

fn bankBytes(rows: u64, record: u64) u64 {
    return (n_layers * rows + transient_slots) * record;
}

/// The lookahead pool's staging charge (q3_lookahead4_candidate.charge_bytes):
/// `slots` staging slots of the page-rounded record plus two pages, plus 1 MiB of tables.
pub fn lookaheadCharge(record_bytes: u64, slots: u64, page: u64) u64 {
    return slots * (std.mem.alignForward(u64, record_bytes, page) + 2 * page) + mib;
}

/// One admission at `credit` bytes off every post-prefill phase, `fixed` rows
/// forced; `filled`: the peak fill's run, whose credited phases must stay positive.
fn retarget(env: Envelope, in: Inputs, credit: u64, fixed: ?u32, filled: bool) Error!Admission {
    const base: i64 = @intCast(in.baseline_bytes);
    const wired: i64 = @intCast(in.wired_bytes);
    const record: i64 = @intCast(in.record_bytes);
    const io_staging: u64 = if (in.io_layout == .gate_up) 36 * mib else 32 * mib;
    const io_host = io_staging + mib;
    // native_retirement_memory.credits (scheduled projections).
    if (in.tail_rows) |t| if (t != tail_rows_proved) return error.TailRowsNotProved;
    if ((in.embedding_rows or in.tail_rows != null) and !env.retirement_owners_proved) return error.RetirementProofMissing;
    const retired_embedding: i64 = if (in.embedding_rows) embedding_bytes else 0;
    const retired_tail: i64 = if (in.tail_rows != null) tail_credit_bytes else 0;
    const box = boxOf(env, in);
    if (in.matched_prefill_rows) |m| if (m < 16 or m > box.max_rows) return error.InvalidMatchedRows;
    var ceiling = in.search_ceiling orelse box.max_rows;
    if (fixed) |f| {
        if (f < 84 or f > box.max_rows) return error.InvalidFixedRows;
        if (f > ceiling) return error.FixedRowsAboveCeiling;
        ceiling = f;
    }
    const post_reserve: i64 = @intCast(env.post_prefill_reserve_bytes);
    const cache: i64 = @intCast(env.decode_cache_bytes);
    const allocator_limit: i64 = @as(i64, @intCast(env.allocator_limit_at_base0 + box.box_bytes)) - @as(i64, @intCast(env.box_bytes)) - base;
    const host: i64 = @intCast(env.host_reserve_bytes + io_host + in.host_reserve_bytes);
    const restored: i64 = @intCast(env.embedding_credit_bytes + env.projection_credit_bytes);
    const pred_prefill: i64 = env.predecessor_prefill_rows;
    const source_prefill_bank: i64 = @intCast(bankBytes(env.predecessor_prefill_rows, raw_record));
    const steady_fixed: i64 = @as(i64, @intCast(env.steady_active_bytes)) - @as(i64, @intCast(bankBytes(env.predecessor_decode_rows, weights_record))) - packed_scales + restored;
    const transition_fixed: i64 = @as(i64, @intCast(env.transition_start_bytes + env.embedding_credit_bytes)) - source_prefill_bank;

    // projection_fixed_bounds at the proof's pipeline reserve; every member
    // adds its bytes to every phase, every credit takes them off.
    if (env.projection_credit_bytes != restored_projection_bytes or in.phase_reserve_bytes < base_pipeline_bytes)
        return error.ProjectionGeometry;
    const inherited = steady_fixed - @as(i64, restored_projection_bytes) - 3 * @as(i64, projection_bytes);
    if (inherited < 0 or steady_fixed < 0 or transition_fixed < 0) return error.ProjectionEnvelopeTooSmall;
    const seed_fixed: i64 = @intCast(env.projection_seed_fixed_bytes);
    const group: i64 = projection_bytes;
    // The draft head's residents beyond (or short of) the envelope's.
    const head_extra: i64 = if (in.draft_pruned_bytes) |p| @as(i64, @intCast(env.draft_pruned_bytes)) - @as(i64, @intCast(p)) else 0;
    const wide: i64 = @intCast(in.wide_window_bytes);
    const charge: i64 = @as(i64, @intCast(in.phase_reserve_bytes - base_pipeline_bytes)) +
        (if (in.rowsx == null) @as(i64, @intCast(in.lookahead_staging_bytes)) else 0) - @as(i64, @intCast(credit)) + head_extra + wide;
    var ph = [4]i64{
        transition_fixed + group,
        seed_fixed + group,
        seed_fixed + projection_bytes + group,
        inherited + 5 * @as(i64, projection_bytes) + group,
    };
    for (&ph) |*v| v.* += charge;
    if (filled) for (ph) |v| if (v <= 0) return error.PhaseCreditExceedsEnvelope;
    for (&ph, 0..) |*v, i| v.* -= retired_embedding + (if (i != 3) retired_tail else 0);
    for (ph) |v| if (v < 0) return error.RetirementCreditExceedsOwners;
    const fixed_peak = @max(@max(ph[0], ph[1]), @max(ph[2], ph[3]));
    const prefill_cache: i64 = @intCast(env.prefill_cache_bytes + in.prefill_charge_bytes);

    const Fit = struct {
        box: i64,
        base: i64,
        host: i64,
        wired: i64,
        wired_ceiling: i64,
        allocator_limit: i64,
        fn ok(f: @This(), active: i64, cache_bytes: i64) bool {
            return f.base + f.host + active + cache_bytes + @as(i64, physical_headroom) <= f.box and
                active + cache_bytes <= f.allocator_limit and
                f.wired + active + cache_bytes + @as(i64, wired_headroom) <= f.wired_ceiling;
        }
    };
    const fit: Fit = .{ .box = @intCast(box.box_bytes), .base = base, .host = host, .wired = wired, .wired_ceiling = @intCast(box.wired_bytes), .allocator_limit = allocator_limit };
    const entry_fixed = transition_fixed + projection_bytes - retired_tail + post_reserve + head_extra + wide;
    const prefill_base: i64 = @as(i64, @intCast(env.prefill_active_bytes)) + prefill_engine_remainder + transform_reserve + head_extra + wide;
    const raw_band: i64 = n_layers * raw_record;

    // The prefill: the largest predecessor-record row budget whose active and
    // un-retired entry both fit.
    var prefill_rows: i64 = -1;
    var prefill_active: i64 = 0;
    var prefill_physical: i64 = 0;
    var rows: i64 = pred_prefill;
    while (rows > 15) : (rows -= 1) {
        prefill_active = prefill_base - (pred_prefill - rows) * raw_band;
        prefill_physical = base + host + prefill_active + prefill_cache;
        const old_bank: i64 = @as(i64, @intCast(bankBytes(@intCast(rows), raw_record))) + prefill_engine_remainder;
        if (fit.ok(prefill_active, prefill_cache) and fit.ok(entry_fixed + old_bank, prefill_cache)) {
            prefill_rows = rows;
            break;
        }
    }
    if (prefill_rows < 0) return error.PrefillDoesNotFit;

    // The decode rows: the largest capacity whose binding phase fits.
    var chosen: i64 = -1;
    var cap: i64 = ceiling;
    while (cap >= pred_prefill) : (cap -= 1) {
        const active = fixed_peak + @as(i64, @intCast(bankBytes(@intCast(cap), @intCast(record)))) + post_reserve;
        if (fit.ok(active, cache)) {
            chosen = cap;
            break;
        }
    }
    if (chosen < 0) return error.NoDecodeCapacity;
    if (fixed) |f| if (chosen != f) return error.FixedRowsDoNotFit;

    // The causal allocator's room: eight average extension rows past the prefill bank.
    var prefill_max: i64 = chosen - (if (in.allocation == .prefill_excess) @as(i64, 8) else 0);
    if (in.matched_prefill_rows) |m| prefill_max = @min(prefill_max, m);
    const capacityOf = struct {
        fn f(r: i64, rec: i64) i64 {
            const bound: i64 = @as(i64, @intCast(bankBytes(@intCast(r), raw_record))) + prefill_engine_remainder;
            return @divFloor(bound - @as(i64, transient_slots) * rec, @as(i64, n_layers) * rec);
        }
    }.f;
    if (in.allocation == .prefill_excess or in.matched_prefill_rows != null) {
        var r = prefill_rows;
        var found = false;
        while (r > 15) : (r -= 1) {
            if (capacityOf(r, record) <= prefill_max) {
                prefill_rows = r;
                found = true;
                break;
            }
        }
        if (!found) return error.NoCausalPrefill;
        prefill_active = prefill_base - (pred_prefill - prefill_rows) * raw_band;
        prefill_physical = base + host + prefill_active + prefill_cache;
    }
    if (in.matched_prefill_rows) |m| if (capacityOf(prefill_rows, record) != m) return error.MatchedPrefillDoesNotFit;
    if (fixed != null and capacityOf(prefill_rows, record) > prefill_max) return error.FixedRowsIncompatibleWithPrefill;

    const final_bank: i64 = @intCast(bankBytes(@intCast(chosen), @intCast(record)));
    const phases: Phases = .{
        .growth = @intCast(ph[0] + final_bank + post_reserve),
        .seed = @intCast(ph[1] + final_bank + post_reserve),
        .prime = @intCast(ph[2] + final_bank + post_reserve),
        .decode = @intCast(ph[3] + final_bank + post_reserve),
    };
    const steady: i64 = ph[3] + final_bank + post_reserve;
    const resize: i64 = fixed_peak + final_bank + post_reserve;
    const prefill_bank_bound: i64 = @as(i64, @intCast(bankBytes(@intCast(prefill_rows), raw_record))) + prefill_engine_remainder;
    const entry = entry_fixed + prefill_bank_bound;
    const transition_start = ph[0] + prefill_bank_bound + post_reserve;
    const physical_bound = @max(@max(prefill_physical, base + host + entry + prefill_cache), base + host + @max(steady, resize) + cache);
    return .{
        .decode_rows = @intCast(chosen),
        .prefill_rows = @intCast(prefill_rows),
        .prefill_capacity = @intCast(capacityOf(prefill_rows, record)),
        .prefill_max_rows = @intCast(prefill_max),
        .search_ceiling = ceiling,
        .transition_start_bytes = @intCast(transition_start),
        .retirement_entry_bytes = @intCast(entry),
        .phases = phases,
        .steady_bytes = @intCast(steady),
        .resize_bytes = @intCast(resize),
        .prefill_active_bytes = @intCast(prefill_active),
        .prefill_physical_bytes = @intCast(prefill_physical),
        .prefill_cache_bytes = @intCast(prefill_cache),
        .physical_bound_bytes = @intCast(physical_bound),
        .active_bound_bytes = @intCast(@max(@max(prefill_active, entry), @max(steady, resize))),
        .host_reserve_bytes = @intCast(host),
        .allocator_limit_bytes = @intCast(allocator_limit),
        .decode_cache_bytes = env.decode_cache_bytes,
        .final_bank_bytes = @intCast(final_bank),
        .prefill_bank_bound_bytes = @intCast(prefill_bank_bound),
        .embedding_credit_bytes = @intCast(retired_embedding),
        .tail_credit_bytes = @intCast(retired_tail),
        .baseline_bytes = in.baseline_bytes,
        .wired_bytes = in.wired_bytes,
    };
}

/// The peak fill: the AUTO control, then the filled run at `in.fixed_rows`
/// with its credit off every post-prefill phase; the fill may move nothing
/// but the decode rows and must stay under its target.
fn peakFilled(env: Envelope, in: Inputs, pf: PeakFill, extra_credit: u64) Error!struct { admission: Admission, record: PeakFillRecord } {
    const box = boxOf(env, in);
    const target = pf.target_bytes orelse box.max_target_bytes;
    const unfilled = box.box_bytes - physical_headroom;
    if (target < unfilled or target > box.max_target_bytes) return error.InvalidTarget;
    const spend = target - unfilled;
    const credit = pf.evidence_credit_bytes + spend;
    const control = try retarget(env, in, extra_credit, null, false);
    const filled = try retarget(env, in, extra_credit + credit, in.fixed_rows, true);
    if (filled.prefill_rows != control.prefill_rows or filled.prefill_active_bytes != control.prefill_active_bytes or
        filled.prefill_physical_bytes != control.prefill_physical_bytes or filled.retirement_entry_bytes != control.retirement_entry_bytes or
        filled.host_reserve_bytes != control.host_reserve_bytes) return error.PeakFillMovedPrefill;
    if (in.fixed_rows) |f| {
        if (filled.decode_rows != f) return error.PeakFillMovedFixedRows;
    } else if (filled.decode_rows < control.decode_rows) return error.PeakFillReducedRows;
    const active = filled.phases.max();
    const modeled = @max(filled.physical_bound_bytes, filled.baseline_bytes + filled.host_reserve_bytes + active + filled.decode_cache_bytes + spend);
    if (modeled > target) return error.PeakFillOverTarget;
    if (active + filled.decode_cache_bytes > filled.allocator_limit_bytes) return error.PeakFillOverAllocator;
    if (filled.wired_bytes + active + filled.decode_cache_bytes + gib > box.wired_bytes) return error.PeakFillOverWired;
    return .{ .admission = filled, .record = .{ .target_bytes = target, .control = Summary.of(control), .filled = Summary.of(filled), .total_credit_bytes = credit, .modeled_peak_bytes = modeled } };
}

/// The box an admission fits: the caller's ceiling, else the envelope's own.
fn boxOf(env: Envelope, in: Inputs) Ceiling {
    return in.ceiling orelse .{ .box_bytes = env.box_bytes, .wired_bytes = wired_ceiling_bytes, .max_target_bytes = max_target_bytes, .max_rows = 160 };
}

fn planAdmission(env: Envelope, in: Inputs) Error!Plan {
    const pf = in.peak_fill orelse {
        if (in.rowsx != null) return error.CreditGeometry;
        return .{ .admission = try retarget(env, in, 0, in.fixed_rows, false) };
    };
    const rx = in.rowsx orelse {
        const r = try peakFilled(env, in, pf, 0);
        return .{ .admission = r.admission, .peak_fill = r.record };
    };
    // The lane pays the lookahead staging out of its credit (whole MiB).
    const staging = std.mem.alignForward(u64, in.lookahead_staging_bytes, mib);
    if (staging >= rx.credit_bytes) return error.CreditGeometry;
    const credit = rx.credit_bytes - staging;
    const ref = try peakFilled(env, in, pf, 0);
    if (in.fixed_rows != null) return error.CreditWithFixedRows;
    const r = ref.admission;
    // The credit's evidence was measured in the envelope's own box.
    if (in.ceiling != null) return error.CreditGeometry;
    const spend = ref.record.target_bytes - (env.box_bytes - physical_headroom);
    if (r.phases.decode - r.final_bank_bytes != rx.decode_fixed_bytes or r.phases.prime - r.final_bank_bytes != rx.prime_fixed_bytes or
        r.host_reserve_bytes + r.decode_cache_bytes != rx.non_mlx_bytes or ref.record.target_bytes != rx.target_bytes or env.box_bytes != box_ceiling_bytes)
        return error.CreditGeometry;
    const cred = try peakFilled(env, in, pf, credit);
    const c = cred.admission;
    if (c.prefill_rows != r.prefill_rows or c.prefill_active_bytes != r.prefill_active_bytes or c.prefill_physical_bytes != r.prefill_physical_bytes or
        c.retirement_entry_bytes != r.retirement_entry_bytes or c.host_reserve_bytes != r.host_reserve_bytes or c.decode_cache_bytes != r.decode_cache_bytes or
        c.baseline_bytes != r.baseline_bytes or c.allocator_limit_bytes != r.allocator_limit_bytes or c.wired_bytes != r.wired_bytes)
        return error.CreditMovedPrefill;
    if (c.decode_rows <= r.decode_rows) return error.CreditAddsNoRow;
    const added = c.decode_rows - r.decode_rows;
    const true_active = rx.prime_fixed_bytes + c.final_bank_bytes;
    const modeled_true = c.baseline_bytes + c.host_reserve_bytes + true_active + c.decode_cache_bytes + spend;
    if (modeled_true - credit > rx.target_bytes) return error.CreditOverTarget;
    const mlx_side = true_active - rx.mlx_side_credit_bytes + c.decode_cache_bytes;
    if (mlx_side > c.allocator_limit_bytes) return error.CreditOverAllocator;
    if (c.wired_bytes + mlx_side + gib > wired_ceiling_bytes) return error.CreditOverWired;
    // The credit lowers the prefill-boundary bound only through the growth
    // phase, at an unchanged prefill bank: give that bound back.
    if ((r.phases.growth - r.final_bank_bytes) - (c.phases.growth - c.final_bank_bytes) != credit or
        r.transition_start_bytes - c.transition_start_bytes != credit or r.prefill_bank_bound_bytes != c.prefill_bank_bound_bytes)
        return error.CreditReachedGate;
    var admitted = c;
    admitted.transition_start_bytes = r.transition_start_bytes;
    return .{ .admission = admitted, .peak_fill = cred.record, .rowsx = .{
        .uncredited_rows = r.decode_rows,
        .credited_rows = c.decode_rows,
        .added_rows = added,
        .added_bytes = added * n_layers * in.record_bytes,
        .credit_bytes = credit,
        .mlx_side_credit_bytes = rx.mlx_side_credit_bytes,
        .modeled_peak_uncredited_bytes = modeled_true,
        .modeled_peak_credited_bytes = modeled_true - credit,
        .target_bytes = rx.target_bytes,
        .allocator_margin_bytes = @as(i64, @intCast(c.allocator_limit_bytes)) - @as(i64, @intCast(mlx_side)),
        .wired_margin_bytes = @as(i64, @intCast(wired_ceiling_bytes)) - @as(i64, @intCast(c.wired_bytes + mlx_side + gib)),
        .box_margin_bytes = @as(i64, @intCast(env.box_bytes)) - @as(i64, @intCast(modeled_true - credit)),
        .baseline_bytes = c.baseline_bytes,
        .restored_gate_bytes = r.transition_start_bytes,
    } };
}
