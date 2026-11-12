//! G4, phase-specific admission: an arch bills its own named terms per phase; the host fills two row
//! counts and admits both phases before any allocation, and checks the constructed module once against the bill.
//! Pure host: no MLX, no device query, no global, so bills and fills run in the CPU lane and the preflight never
//! touches the device.

const std = @import("std");
const peek = @import("peek.zig");

pub const MemoryBill = struct {
    pub const Phase = enum { prompt, decode };
    /// One named term, decimal bytes per phase (0 where absent). Rows-free by construction: the persistent slot rows
    /// are `per_row` times the fill's rows. Every term is billed, a measured one at its declared bound.
    pub const Term = struct {
        name: []const u8,
        bytes: [2]u64,
        /// Held by the constructed module before any request (the construction check's terms); a phase's
        /// transients (waves, KV, the MLX cache, posted gathers) are not.
        at_construction: bool,
        /// Not derivable from headers: `bytes` is a declared bound, measured once at construction (`checkMeasured`).
        measured: bool = false,
        /// What the constructed module holds of the term when it is not its prompt bytes (a measured term whose
        /// construction bound is tighter than its phases'); null: the prompt bytes.
        construction: ?u64 = null,
        /// Its bytes follow the slot rows (`MemoryBill.row_terms`): `bytes` is the value at the bill's own rows, and
        /// every total counts the term through `row_terms` at the rows it is asked for.
        with_rows: bool = false,

        /// The term's bytes at construction: `construction`, else the prompt phase's.
        pub fn atConstruction(t: Term) u64 {
            return t.construction orelse t.bytes[@backingInt(Phase.prompt)];
        }
    };

    terms: []const Term = &.{},
    /// One slot row on every routed layer (an expert_source's); 0 for a model without one.
    per_row: u64 = 0,
    /// The terms flagged `with_rows`, at any row count; null for a bill linear in its rows.
    row_terms: ?RowTerms = null,
    /// The slot rows' bytes at each row count `0 .. slot_table.len - 1` where they are not `rows * per_row`: a bank
    /// whose stream layers each take their share of a fill unit, rounded down (GLM-5.3's EXL3 bank). Empty: linear.
    /// The bill owns it (`free`); `per_row` stays the fill's first linear step.
    slot_table: []const u64 = &.{},

    /// Terms not linear in the slot rows (the kernel's page tables over the wired bytes): `at(&data, phase, rows, slots)` is
    /// their bytes at `rows` per layer, `slots` the slot rows' bytes there (`slots(rows)`). Subadditive in the rows, so the
    /// linear fill bounds the rows from above. `data` is the plugin's own (the inputs its formula reads), so a bill stays
    /// a value.
    pub const RowTerms = struct {
        data: [4]u64,
        at: *const fn (data: *const [4]u64, phase: Phase, rows: u32, slots: u64) u64,
    };

    /// The phase's terms, the baseline, the slot rows and the row-following terms apart.
    pub fn fixed(b: MemoryBill, phase: Phase) u64 {
        var n: u64 = 0;
        for (b.terms) |t| {
            if (!t.with_rows) n += t.bytes[@backingInt(phase)];
        }
        return n;
    }

    /// The slot rows' bytes at `rows` per layer: the table's (`slot_table`), else `rows * per_row`.
    pub fn slots(b: MemoryBill, rows: u32) u64 {
        return if (b.slot_table.len == 0) rows * b.per_row else b.slot_table[rows];
    }

    /// The phase's total over the box baseline at `rows` slot rows per layer.
    pub fn total(b: MemoryBill, phase: Phase, baseline: u64, rows: u32) u64 {
        const slot_bytes = b.slots(rows);
        const linear = baseline + b.fixed(phase) + slot_bytes;
        const r = b.row_terms orelse return linear;
        return linear + r.at(&r.data, phase, rows, slot_bytes);
    }

    /// What the process may hold above the baseline: the larger phase (a phase change frees before it grows, so
    /// there is no transition term).
    pub fn processBound(b: MemoryBill, rows: Rows) u64 {
        return @max(b.total(.prompt, 0, rows.prompt), b.total(.decode, 0, rows.decode));
    }

    /// What the constructed module holds before any request: the construction terms at their prompt bytes and the
    /// prompt rows.
    pub fn constructionBytes(b: MemoryBill, prompt_rows: u32) u64 {
        var n: u64 = b.slots(prompt_rows);
        for (b.terms) |t| {
            if (t.at_construction) n += t.atConstruction();
        }
        return n;
    }

    /// Frees an allocated bill's terms.
    pub fn free(b: MemoryBill, gpa: std.mem.Allocator) void {
        gpa.free(b.terms);
        if (b.slot_table.len > 0) gpa.free(b.slot_table);
    }
};

/// Slot rows per routed layer, per phase: prompt <= decode <= the layer's experts.
pub const Rows = struct { prompt: u32, decode: u32 };

/// What every kind's `bill` hook receives. Pure host: the plugin's own parsed config and the routes it installs
/// (resolved once, through the same resolver its install reads), the request shape, and the ceiling and the stop as
/// arguments. The box baseline never enters the terms: the fill and the admission take it.
pub const BillRequest = struct {
    peek: *const peek.ConfigPeek,
    /// The plugin's own parsed config (the object its install reads).
    cfg: *const anyopaque,
    /// The plugin's own resolved routes (its type; the host never reads them).
    routes: *const anyopaque,
    prompt_tokens: u64,
    /// The request's generation cap. Each plugin bills its own KV reservation from it, by the same rule its prompt
    /// forward reserves with.
    max_tokens: u64,
    /// The GPU memory ceiling every plan fits under.
    ceiling: u64,
    /// What the admission keeps free under the ceiling.
    stop: u64,

    pub fn target(r: BillRequest) u64 {
        return r.ceiling -| r.stop;
    }
};

/// The fill: the most decode rows, then the most prompt rows (prompt <= decode <= `n_experts`), each phase's total
/// within `target`. Refused by name below `min_rows`, and for a bill without slot rows (nothing to fill).
pub fn fill(b: MemoryBill, baseline: u64, target: u64, n_experts: u32, min_rows: u32) error{ NoSlotRows, NativeBillDoesNotFit }!Rows {
    if (b.per_row == 0) return error.NoSlotRows;
    if (b.slot_table.len > 0 and b.slot_table.len <= n_experts) return error.NativeBillDoesNotFit;
    const most = struct {
        // The linear rows (the row-following terms left out) bound it from above; step down while the total is over. A
        // slot table's rounded-down shares can fit past the linear rows: step up while the next row count fits.
        fn f(mb: MemoryBill, phase: MemoryBill.Phase, base: u64, t: u64, cap: u32) u64 {
            const fixed = base + mb.fixed(phase);
            var r: u64 = @min(if (fixed >= t) 0 else (t - fixed) / mb.per_row, cap);
            while (r > 0 and mb.total(phase, base, @intCast(r)) > t) r -= 1;
            if (mb.slot_table.len > 0) while (r < cap and mb.total(phase, base, @intCast(r + 1)) <= t) {
                r += 1;
            };
            return r;
        }
    }.f;
    const decode = @min(most(b, .decode, baseline, target, n_experts), n_experts);
    const prompt = @min(most(b, .prompt, baseline, target, n_experts), decode);
    if (prompt < min_rows) return error.NativeBillDoesNotFit;
    return .{ .prompt = @intCast(prompt), .decode = @intCast(decode) };
}

/// Both phases within `target` at `rows`, once, before any slot bank or resident is allocated (forced rows are
/// checked here; the fill guarantees its own).
pub fn admit(b: MemoryBill, baseline: u64, rows: Rows, target: u64) error{ PromptOverTarget, DecodeOverTarget }!void {
    if (b.total(.prompt, baseline, rows.prompt) > target) return error.PromptOverTarget;
    if (b.total(.decode, baseline, rows.decode) > target) return error.DecodeOverTarget;
}

/// Once, at construction: the bill was taken at the rows the module built.
pub fn checkRows(billed: Rows, built: Rows) error{BillRowsMismatch}!void {
    if (billed.prompt != built.prompt or billed.decode != built.decode) return error.BillRowsMismatch;
}

/// Once, at construction: the footprint after the install within `tolerance` of the construction bytes.
pub fn checkConstruction(construction_bytes: u64, footprint: u64, tolerance: u64) error{ConstructionOverBill}!void {
    if (footprint > construction_bytes + tolerance) return error.ConstructionOverBill;
}

/// Once, at construction: a measured term's one measurement within its declared bound at construction
/// (`Term.atConstruction`). A term the plugin derives from headers has no measurement to check.
pub fn checkMeasured(t: MemoryBill.Term, measured: u64) error{ ConstructionOverBill, TermNotMeasured }!void {
    if (!t.measured) return error.TermNotMeasured;
    if (measured > t.atConstruction()) return error.ConstructionOverBill;
}

const testing = std.testing;

const gb: u64 = 1_000_000_000;

fn fixtureBill() MemoryBill {
    const terms = comptime [_]MemoryBill.Term{
        .{ .name = "residents", .bytes = .{ 60 * gb, 59 * gb }, .at_construction = true },
        .{ .name = "waves", .bytes = .{ 14 * gb, 2 * gb }, .at_construction = false },
        .{ .name = "kv", .bytes = .{ 1 * gb, 1 * gb }, .at_construction = false },
    };
    return .{ .terms = &terms, .per_row = gb / 4 };
}
