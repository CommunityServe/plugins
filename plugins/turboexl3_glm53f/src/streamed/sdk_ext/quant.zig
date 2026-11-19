//! C2, the routed-expert `quant`: the routed-expert half of PLG's `quant` kind (weight load + the
//! MoE matmul). Given the slot ids and the bank, an MoE layer asks the quant that claimed the
//! weight group for its call's expert matmuls.
//!
//! The arch keeps the routing (gate, top-k, weights), the combine, the shared expert and the
//! choice between decode and prefill width (made on the logical M, a genuine runtime value). C1
//! `expert_source` keeps which slot holds which expert, the bank arrays, the wait gates and the
//! growth.
//!
//! The load path asks every quant `claims(peek)` once per weight group and takes the highest
//! claim. The peek is the group's description: C1's manifest reader for a streamed bank
//! (`expert_bank.peek`), the checkpoint's quantization_config for resident experts. The claimed
//! quant is `accept`ed once per backend (built, and self-checked where it brings kernels), and
//! every entry point of its `Accepted(G)` is prebound there: nothing is looked up or revalidated
//! per call. Clients:
//!   - `exl3_quant`: the streamed EXL3 bank's pinned Metal texts.
//!
//! A quant `Q` declares:
//!   name: []const u8
//!   Arrays(T): type                                   one projection's per-slot arrays
//!   claims(*const BankPeek, ?*Diag) ?Priority         at load, once per weight group
//!   Accepted(G): type                                 the quant accepted on backend G
//!   accept(G, Allocator, *G, Context, Spec, *Diag) !*Accepted(G)   once, before the weights bind
//! and its `Accepted(G)`:
//!   max_decode_rows: u32                              the widest call gateUp / down serve
//!   checkBank(*const, *G, BankArrays(Arrays(G.T)), *Diag) !void    per bank bind and per grow
//!   gateUp(*const, *G, x, slot_ids, gate, up) !G.T   the activation of the gate and up rows
//!   down(*const, *G, h, slot_ids, down) !G.T
//!   prefill(*, *G, layer, x, PrefillRows, BankArrays(..)) !G.T     any width: the call's down rows
//!   finishPrefill(*, *G) !void                        the prefill boundary: drains what is in flight
//!   deinit(*, *G) void

const std = @import("std");
const mlx = @import("sdk").mlx;
const pk = @import("sdk");

const Dtype = mlx.mlx_dtype;

pub const Diag = pk.Diag;

/// How strongly a quant claims a weight group: the load path asks every quant and takes the highest.
pub const Priority = pk.Priority;

/// A weight group's per-expert tensor, its layers, and its description at load (the SDK's `GroupPeek`).
pub const Segment = pk.Segment;
pub const LayerPeek = pk.LayerPeek;
pub const BankPeek = pk.GroupPeek;

/// The routed experts' activation between gate / up and down. It is the arch's: a quant whose
/// texts fuse it declares the one they fuse and refuses the rest at accept.
pub const Activation = union(enum) {
    /// silu(gate) * up
    swiglu,
    /// silu(min(gate, limit)) * clip(up, -limit, limit) (DeepSeek-V4.1's swiglu_limit)
    swiglu_clamped: f64,
};

/// What the arch asks of the quant at accept.
pub const Spec = struct {
    hidden: u32,
    inter: u32,
    top_k: u32,
    n_layers: u32,
    act: Activation,
    /// the MoE input's dtype
    input: Dtype,
};

/// One prefill call's routed rows: `slot[i]` is routed row i's bank slot; row i reads x row i,
/// or x row `act_row[i]` when given (the chunk's tokens: routed row / top_k).
pub const PrefillRows = struct { slot: []const u32, act_row: ?[]const u32 = null };

/// A layer bank's three projections, each one `Q.Arrays(T)`.
pub fn BankArrays(comptime A: type) type {
    return struct { gate: A, up: A, down: A };
}

pub const Proj = enum { gate, up, down };

/// Refusals of accept (construction time; never a per-call branch).
pub const Refusal = error{
    /// the arch's activation is not the one the quant's texts fuse
    ActivationNotFused,
    /// dims the quant's kernels are not compiled for
    DimsNotImplemented,
    /// an MoE input dtype the quant's kernels do not read
    InputDtype,
    /// decode rows (8 tokens x top_k) wider than the quant's decode tables
    TopKTooWide,
    /// a quant with pinned texts accepted without the load context's kernel set
    NoKernelSet,
    /// a quant whose parameters come from the group's description accepted without it
    NoWeightDescription,
    /// a description the quant does not serve (its claims would have declined it)
    NotClaimed,
    /// bank arrays that are not the kernels' signature
    BankArrays,
};

pub fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// A declined claim, the first mismatch in `why`.
pub fn decline(why: ?*Diag, comptime fmt: []const u8, args: anytype) ?Priority {
    if (why) |d| d.set(fmt, args);
    return null;
}

/// The string field `name` of a JSON object (null when absent or not a string).
pub fn str(v: std.json.Value, name: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .string) f.string else null;
}

/// The integer field `name` of a JSON object (null when absent or not an integer).
pub fn int(v: std.json.Value, name: []const u8) ?i64 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .integer) f.integer else null;
}

/// The object field `name` of a JSON object (`.null` when absent).
pub fn obj(v: std.json.Value, name: []const u8) std.json.Value {
    if (v != .object) return .null;
    return v.object.get(name) orelse .null;
}
