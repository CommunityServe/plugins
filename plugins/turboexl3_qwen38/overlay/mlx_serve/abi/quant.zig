//! copied_from: src/abi/quant.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The quant plugin table (`QuantTable`): a format claims a config's quantization block, validates and bills it,
//! loads its tensors and banks from a weight source (`Source`), runs its matmul and MoE dispatch, registers its
//! kernels. The built-in formats (`src/sdk/quant/`) implement the same table.
const Host = @import("mlx_serve").abi.host.Host;
const std = @import("std");
const kit_quant = @import("mlx_serve").abi.quant;
const common = @import("mlx_serve").abi.common;
pub const Span = kit_quant.Span;
pub const Str = kit_quant.Str;
pub const Verdict = kit_quant.Verdict;
pub const Array = kit_quant.Array;
pub const Stream = kit_quant.Stream;
pub const Claim = kit_quant.Claim;
pub const PackDir = kit_quant.PackDir;
pub const version: u32 = 1;
/// Where a format reads a pack's tensors: the resident source (`sdk/weights/resident.zig`) or the streamed one.
pub const Source = kit_quant.Source;
/// A loaded weight in the format's own layout. It owns every array in it; `QuantTable.release` frees them.
pub const Weight = kit_quant.Weight;
pub const LoadCtx = kit_quant.LoadCtx;
/// A quant plugin's table, exported as `exportName(name, version)`. `claims`, `validate` and `bill` never reach MLX.
/// A load runs `open`, then any number of `load_tensor`/`load_bank`, then `close`; the weights outlive `close`.
pub const QuantTable = extern struct {
    size: u32 = @sizeOf(QuantTable),
    version: u32 = version,
    name: Str,
    /// Claims one quantization block of `config_json`; a marker beside the block outranks the block's own mode.
    claims: *const fn (block_json: Str, config_json: Str) callconv(.c) Claim,
    /// Refuses by name a block the format cannot serve (an unknown mode, a width MLX has no kernels for).
    validate: *const fn (block_json: Str, config_json: Str, dir: *const PackDir) callconv(.c) Verdict,
    /// Resident bytes of the pack once loaded, given the `stored` bytes of the shards it loads.
    bill: *const fn (block_json: Str, model_dir: Str, stored: u64) callconv(.c) u64,
    open: *const fn (ctx: *LoadCtx) callconv(.c) Verdict,
    close: *const fn (ctx: *LoadCtx) callconv(.c) void,
    /// The projection `<name>.*`; `in_dim` (the activation's inner dim, 0 = unknown) pins its quant params exactly.
    load_tensor: *const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict,
    /// Routed experts stacked under `<name>.*` (mlx-lm's `switch_mlp` layout), one bank.
    load_bank: *const fn (ctx: *const LoadCtx, name: Str, in_dim: u32, out: *Weight) callconv(.c) Verdict,
    release: *const fn (w: *Weight) callconv(.c) void,
    /// `x @ W^T`; `out` receives a new array.
    matmul: *const fn (w: *const Weight, x: Array, s: Stream, out: *Array) callconv(.c) Verdict,
    /// Routed experts: `x` through the bank rows `rhs` selects (`lhs` the gathered input rows, or empty; `sorted` when
    /// `rhs` is sorted). `out` receives a new array.
    moe: *const fn (bank: *const Weight, x: Array, lhs: Array, rhs: Array, sorted: bool, s: Stream, out: *Array) callconv(.c) Verdict,
    /// Registers the format's kernels with the kernel contract (P2 step 6).
    register_kernels: ?*const fn (host: ?*const Host) callconv(.c) Verdict = null,
    /// Routed SwiGLU experts in one dispatch, for a format whose kernels fuse the three projections (EXL3): `x`
    /// [B, S, D] through the banks' experts `inds` [B, S, K], weighted by `scores`. Null: compose `moe` per projection.
    swiglu: ?*const fn (gate: *const Weight, up: *const Weight, down: *const Weight, x: Array, inds: Array, scores: Array, opts: *const Swiglu, s: Stream, out: *Array) callconv(.c) Verdict = null,
    // ── The GDN layer's tail: one trunk linear in parts around the host's own launches (qwen4_exp `trunk/gdn_tail.zig`).
    // The host's fused tail repeats a serving format's prepare and finish, so a format serves these only if, per 128-column
    // block, its prepare is xp = f16(H(f16(x) * scales) / sqrt(128)) and its finish y = f16(H(f16(z)) / sqrt(128) *
    // biases) then the output dtype: H the 128-wide Hadamard butterfly turboexl3's kernels splice (`sources.had128`), in its
    // order; `scales` f16 [in], `biases` f16 [out] (the `Weight`'s arrays). Null or a refusal: the host runs `matmul`. ──
    /// `matmul` with the work after x's prepare (on `s`) recorded on `opts.side`; with `opts.inner` it stops before the
    /// finish and `out` is the inner product f16 [rows, out] that `linear_finish` takes.
    linear_split: ?*const fn (w: *const Weight, x: Array, opts: *const LinearParts, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// The finish of `opts.parts`: one inner product, or a call's row pieces in row order; y [rows, out] in `opts.dtype`.
    linear_finish: ?*const fn (w: *const Weight, opts: *const LinearParts, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// `matmul` of rows the host prepared (`xp` f16 [rows, in]): y [rows, out] in `opts.dtype`.
    linear_prepared: ?*const fn (w: *const Weight, xp: Array, opts: *const LinearParts, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// The weight in the form a call's row pieces share (`opts.rows`: each piece's rows in row order, `opts.dtype` the
    /// call's), made on `s`; refused before any launch where the call or a piece leaves the path the pieces split.
    linear_pieces: ?*const fn (w: *const Weight, opts: *const LinearParts, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// One piece's inner product against `opts.held` (what `linear_pieces` made): of x [..., in], or with `opts.prepared`
    /// of rows the host prepared.
    linear_piece: ?*const fn (w: *const Weight, x: Array, opts: *const LinearParts, s: Stream, out: *Array) callconv(.c) Verdict = null,
    // A gated attention layer's linears and the shared expert in fused launches, each the bytes of the host's chain it
    // replaces. Null, or a refusal: the host runs that chain.
    /// q (each head [query | gate]), k and v of `x` [1, S, D] at prefill rows, the q / k RMSNorm and partial RoPE in
    /// the finish: `out` q, k, v [1, heads, S, head_dim] (q and k normed and rotated), gate [1, S, heads, head_dim].
    attn_in: ?*const fn (q: *const Weight, k: *const Weight, v: *const Weight, x: Array, opts: *const AttnIn, s: Stream, out: *AttnInOut) callconv(.c) Verdict = null,
    /// `attn_in`'s prepare with a fourth linear reading the same `x`: `out[0..4]` its four prepared copies f16 [rows,
    /// D] (q's, k's and v's for `AttnIn.prep`, the fourth's for `linear_prepared_idx`).
    attn_prep: ?*const fn (q: *const Weight, k: *const Weight, v: *const Weight, w4: *const Weight, x: Array, s: Stream, out: [*]Array) callconv(.c) Verdict = null,
    /// `matmul` of `x` from its rows `attn_prep` prepared (`xp`): y [..., out] in x's shape and dtype.
    linear_prepared_idx: ?*const fn (w: *const Weight, xp: Array, x: Array, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// o_proj over T(attn * sigmoid(gate)) for the attention core's `attn` [1, heads, S, head_dim], the gate read in
    /// place (`Gate`): y [1, S, out].
    gated_out: ?*const fn (o: *const Weight, attn: Array, gate: *const Gate, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// `gated_out` at decode rows over the host's fused sparse-attention core's split partials (pacc f32 [heads, S,
    /// splits, 256], pml f32 [heads, S, splits, 2]), merged in o_proj's prepare with the core's merge arithmetic.
    merged_gated_out: ?*const fn (o: *const Weight, pacc: Array, pml: Array, gate: *const Gate, s: Stream, out: *Array) callconv(.c) Verdict = null,
    /// q_proj (each head [query | gate]) of `x` [1, S, D] at decode rows as two contiguous planes [1, S, heads,
    /// head_dim]: `out[0]` the queries, `out[1]` the gate.
    q_planes: ?*const fn (q: *const Weight, x: Array, opts: *const QPlanes, s: Stream, out: [*]Array) callconv(.c) Verdict = null,
    /// The sparse-attention indexer projection of `x` [1, S, D] at decode rows with its finish's epilogue (`Indexer`):
    /// `out[0]` qk [1, S, out], `out[1]` the index queries normed and rotated [1, heads, S, 128], `out[2]` the pooled
    /// key the call completes [1, 1, 128] (the last two empty unless asked).
    indexer: ?*const fn (idx: *const Weight, x: Array, opts: *const Indexer, s: Stream, out: [*]Array) callconv(.c) Verdict = null,
    /// `swiglu` with a gated shared expert at decode / verify rows: y = routed + sigmoid(logit) * shared(x). With
    /// `SharedSwiglu.logits` the routing (the host's softmax router) runs in the launches and `inds` / `scores` are
    /// empty.
    swiglu_shared: ?*const fn (gate: *const Weight, up: *const Weight, down: *const Weight, x: Array, inds: Array, scores: Array, opts: *const SharedSwiglu, s: Stream, out: *Array) callconv(.c) Verdict = null,
    // ── Resident matrices (memory for speed: an EXL3 trunk's `resident` mode) ──
    /// Keeps each of `ws`' decoded matrices from its first prefill call on, when they fit in `room` bytes (0: no
    /// bound); `bytes` receives what they take. Refused `too_large` past `room`. An empty `ws` drops every kept matrix.
    resident: ?*const fn (ws: Span(Weight), room: u64, s: Stream, bytes: *u64) callconv(.c) Verdict = null,
    /// The bytes `resident` admitted and does not hold yet, which the GPU ceiling holds aside.
    resident_pending: ?*const fn () callconv(.c) u64 = null,
    /// The resident matrices out of the format's basis, in `act` (an mlx-c `mlx_dtype`): `ws` in groups of `sizes[g]`
    /// weights that read one input, each group joined in its order; `took[g]` whether group g took.
    unrotate: ?*const fn (ws: Span(Weight), sizes: Span(u8), act: i32, s: Stream, took: [*]bool) callconv(.c) Verdict = null,
    /// `x` times the joined matrix of the un-rotated group whose members are `ws`, in order: [..., the sum of their
    /// output widths] in x's dtype. Refused where the format holds them otherwise and at decode widths.
    linear_joined: ?*const fn (ws: Span(Weight), x: Array, s: Stream, out: *Array) callconv(.c) Verdict = null,
};
pub const Swiglu = kit_quant.Swiglu;
/// The GDN tail entries' options; each entry reads the fields its doc names.
pub const LinearParts = extern struct {
    size: u32 = @sizeOf(LinearParts),
    /// mlx-c's `mlx_dtype` of the call's activations: the output's.
    dtype: i32 = 0,
    /// `linear_split`: the stream for the work after the prepare (empty: `s`).
    side: Stream = .{},
    /// `linear_split`: stop at the inner product.
    inner: bool = false,
    /// `linear_piece`: x holds rows the host prepared.
    prepared: bool = false,
    /// `linear_piece`: `linear_pieces`' output.
    held: Array = .{},
    /// `linear_pieces`: each piece's row count, in row order.
    rows: Span(u32) = .{},
    /// `linear_finish`: the inner products, in row order.
    parts: Span(Array) = .{},
};
pub const AttnIn = extern struct {
    size: u32 = @sizeOf(AttnIn),
    heads: u32,
    kv_heads: u32,
    /// The rotated leading dims of each head.
    rope_dims: u32,
    /// The per-head RMSNorm weights [head_dim] in x's dtype; `eps` a 0-dim f32.
    q_norm: Array,
    k_norm: Array,
    eps: Array,
    /// f32 [S * rope_dims]: row p holds cos then sin at the chunk's position p.
    angles: Array,
    /// q's, k's and v's copies from `attn_prep`; empty: the entry prepares x itself.
    prep: [3]Array = @splat(.{}),
    /// The un-rotated trunk's joined product of x (`linear_joined`), holding q, k and v at columns `std_offs`: the
    /// epilogue alone runs over it. Empty: the entry projects x.
    std: Array = .{},
    std_offs: [3]u32 = @splat(0),
};
/// New arrays, the caller's.
pub const AttnInOut = extern struct { q: Array = .{}, gate: Array = .{}, k: Array = .{}, v: Array = .{} };
/// Where the gate lies: element (row, head h, d) at base[row * row_stride + h * head_stride + off + d].
pub const Gate = extern struct {
    size: u32 = @sizeOf(Gate),
    base: Array,
    row_stride: u32,
    head_stride: u32,
    off: u32,
    /// MLX's sigmoid of every 16-bit pattern [65536], in the gate's dtype.
    sigtab: Array,
};
pub const QPlanes = extern struct {
    size: u32 = @sizeOf(QPlanes),
    heads: u32,
    head_dim: u32,
};
pub const Indexer = extern struct {
    size: u32 = @sizeOf(Indexer),
    /// Index query heads, each one 128-block of the projection; the raw key's block follows them.
    heads: u32,
    rope_dims: u32,
    /// Rows per pooled key.
    ratio: u32,
    /// Normalize and rotate the query heads (`q_norm`, `angles` f32 [S * rope_dims]).
    q_rope: bool,
    /// Pool the key block the call completes: the ring's last `ln` rows (`ring` [1, ln, 128]) then the call's first
    /// rows, normed (`k_norm`) and rotated by the block's cos / sin row (`pcos`, `psin` [1, rope_dims]).
    pool: bool,
    ln: u32,
    eps: f32,
    q_norm: Array,
    angles: Array = .{},
    k_norm: Array,
    ring: Array = .{},
    pcos: Array = .{},
    psin: Array = .{},
};
pub const SharedSwiglu = extern struct {
    size: u32 = @sizeOf(SharedSwiglu),
    /// The shared expert's linears, as the trunk's format loaded them.
    gate: *const Weight,
    up: *const Weight,
    down: *const Weight,
    /// MLX's sigmoid of every 16-bit pattern [65536], in x's dtype.
    sigtab: Array,
    /// The shared gate's logit per row; empty with `gate_vec` (its weight [D]: the logit computed in the launches).
    logit: Array = .{},
    gate_vec: Array = .{},
    /// The router's logits [B, S, E] in x's dtype, and the experts per token.
    logits: Array = .{},
    k: u32 = 0,
};
// ── tests ──
const testing = std.testing;
