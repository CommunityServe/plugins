//! The pinned Metal kernel registry of the DeepSeek-V4.1 EXL3 lanes: the Python tier's
//! kernel texts, exported byte for byte into kernels/exl3/ with a manifest of their
//! signatures, launch geometry and self-checks. `Registry.init` checks the manifest against
//! `manifest_sha256` and every text against the manifest, once, and refuses by name;
//! `Registry.bind` builds the mlx fast kernels on a GPU stream, once. A bound kernel runs
//! with the geometry fixed here: nothing is selected or re-validated per call.

const std = @import("std");
const mlx = @import("sdk").mlx;
const bo = @import("../streamed/build_flags.zig");
const kset = @import("../streamed/sdk_ext/kernel_set.zig");
/// The registry's self-check plan (`sdk_ext.kernels.KernelSet(R)` runs it at each consumer's accept).
pub const selfcheck = @import("selfcheck.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
const Allocator = std.mem.Allocator;

/// sha256 of kernels/exl3/manifest.json: pins the manifest, which pins every text.
pub const manifest_sha256 = "921f7a69c4b356b31e78f8b3d6ef743caae7cfa7bf7aae4cfd541f0e6d919d81";

/// G7: the decode-timers build's launch probe (`sdk_ext/kernel_set.zig`: its build flag, keys and observer).
pub const launch_observed = kset.launch_observed;
pub const LaunchKey = kset.LaunchKey;
pub const Observer = kset.Observer;
pub const format = "mlx-serve-exl3-kernels-v1";
const dir = "kernels/exl3/";

/// The bank these texts decode: EXL3 codebook mul1, K = 2 or 3 per layer (ascending).
pub const bank_codebook = "mul1";
pub const bank_multiplier: u64 = 0x83DCD12D;
pub const bank_ks = [_]u32{ 2, 3 };

/// Every kernel of record; the tag is the kernel's MLX name and its file name. A variant
/// `<base>__<variant>` (appended) is a registered text at other template values / input
/// dtypes: the base's file, sha256, header and MLX name, its own recorded template, signature,
/// launch and self-check (key: text sha256, template values, input dtypes).
pub const Kernel = enum {
    dsv41_exl3_mul1h_k3_2304,
    dsv41_exl3_mul1h_k3_5120,
    q3_exl3_prep_in_rin,
    q3_exl3_prep_gu_epi,
    q3_exl3_prep_din_rin,
    q3_moeprep_dpost,
    q3rc_gate_part,
    q3rc_router_tail,
    q3rc_premix_part,
    q3rc_premix_fin,
    q3dk_sinkhorn16_hc4_it20,
    q3rc_mxfp8_fma,
    q3ht_combine,
    q3ht_collapse_norm,
    q3ht_combine_collapse_norm,
    q3ht_mixfin,
    q3_prefill_fused_exl3x3_mul1lut_k3_bf16,
    q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3,
    q3_prefill_dig_gemm_2304x5120_xmul1hk3,
    q3_prefill_dig_rot_take2_5120,
    q3_prefill_dig_rot_roundx_2304,
    q3_prefill_dig2_swiglu_2304_x,
    q3_prefill_dig_rot_widen2_2304,
    q3_prefill_dig_rot_widen1_5120,
    dsv41_prefill_dig_rot_widen1_5120_obf16,
    q3_exl3_dig_decmat_5120x2304_mul1hk3,
    q3_exl3_dig_decmat_2304x5120_mul1hk3,
    q3_exl3_dig_decmat_5120x2304_mul1k3,
    q3_exl3_dig_decmat_2304x5120_mul1k3,
    mtplx_dsv4_sinkhorn_hc4_it20,
    mtplx_dsv41_fp_rmsnorm_tg128_d1280,
    mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64,
    mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd,
    mtplx_dsv41_fp_rope_h64_hd512_rd64_inv,
    // DRAFTRC (record 3): the draft's template variants and the f32-x rcproj FMA text
    q3rc_gate_part__n128,
    q3rc_router_tail__n128_top3,
    q3ht_combine__f32,
    q3ht_collapse_norm__f32,
    q3ht_combine_collapse_norm__f32,
    q3ht_combine_collapse_norm__f32_rbf16,
    q3drc_mxfp8_fma_f32x,
    q3rc_mxfp8_fma__draft,
    // decode batch 2 (09-29): the texts the RC tiers of record still run beside the RC routes
    dsv41_woa_decode_transpose_32,
    mtplx_dsv41_index_topk_select,
    q3_attnfuse_softmax,
    q3_attnfuse_softmax__ls128,
    dsv41_mxfp8_m1rows,
    dsv41_head_m1rows,
    dsv41_smallm_all,
    dsv41_smallm_all__bf16,
    // prefill batch 2 (09-29): the prefill-rows texts the P line still runs (ATTNHALF idxscore /
    // corevec / ropefuse at every (TQ, TKV) the lane warms, ATTN hcnorm at both stream dtypes,
    // SMALLK combine)
    q3_ph_index_score,
    q3_ph_qkvec_win,
    q3_ph_qkvec_win__kvf32,
    q3_ph_qkvec_win__qf32,
    q3_ph_qkvec_win__qf32_kvf32,
    q3_ph_qkvec_cmp,
    q3_ph_qkvec_cmp__kvbf16,
    q3_ph_qkvec_cmp__qf32,
    q3_ph_pvvec_win,
    q3_ph_pvvec_win__kvf32,
    q3_ph_pvvec_cmp,
    q3_ph_pvvec_cmp__kvbf16,
    q3_ph_qkrope_win,
    q3_ph_qkrope_win__kvf32,
    q3_ph_qkrope_win__qf32,
    q3_ph_qkrope_win__qf32_kvf32,
    q3_ph_qkrope_cmp,
    q3_ph_qkrope_cmp__kvbf16,
    q3_ph_qkrope_cmp__qf32,
    q3_ph_pvrope_win,
    q3_ph_pvrope_win__kvf32,
    q3_ph_pvrope_cmp,
    q3_ph_pvrope_cmp__kvbf16,
    q3pf_hc_mix_rsqrt,
    q3pf_hc_pre_norm,
    q3pf_hc_mix_rsqrt__f32,
    q3pf_hc_pre_norm__f32,
    q3sk_combine,
    q3sk_combine__rbf16,
    // JOINLESS (09-30, port-exported: R/mlx-serve-kernels/tools/export_joinless.py): SMALLK's combine reading the
    // routed rows from the fused call outputs through a (source, row) table
    q3jl_combine,
    // The take2 retune (09-30, mlx-serve native; its lane pin is its own text): q3_prefill_dig_rot_take2_5120's values
    // with both outputs per simdgroup from one act load (the same butterfly pairs in the same bit order)
    dsv41_prefill_dig_take2v_5120,
    // The DIG-X GEMMs at a 128-row M tile (09-30, mlx-serve native): 8 simdgroups, each running the 64-row text's
    // 32 x 32 share, so one decoded B stage feeds 128 rows (twin check: the 64-row text's words)
    dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128,
    dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128,
    // The 128-row down GEMM at BN 128 with rot_widen1 as its epilogue (09-30, mlx-serve native): z stays in the
    // threadgroup (fused check: the words of the 128-row down text, then q3_prefill_dig_rot_widen1_5120)
    dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1,
    // The 128-row gate|up GEMM with the mul1h codebook through a threadgroup table (09-30, mlx-serve native; header
    // dig_mul1h_k3_lut): the same halves from a 1,024-entry table (twin check: the 128-row text's words)
    dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut,
    // Routed decode forms (10-02, kbench v9: exact on 19 of 19 census shapes): the down projection's pair form (the
    // Python forms module's text: a slot's rows paired, one decode for both; twin: mul1h_5120's words) and gate + up
    // in one launch (a threadgroup-uniform half; twin: mul1h_2304's words on each half)
    dsv41_exl3_pair_k3_5120,
    dsv41_exl3_guone_k3_2304,
    // The prompt-width HC combine in one pass (10-02, mlx-serve native; header hcpost_tf32): the compiled HcPost
    // region's words (the NAX f32 GEMM's TF32 truncation, products flushed below 2^-126, pairwise sum, mul-then-add
    // tail); the f32 residual and layer 0's bf16 one (checked against the region at model construction)
    dsv41_hcpost_tf32,
    dsv41_hcpost_tf32__rbf16,
    dsv41_hcpost_tf32_bf16,
    // The banked hit wave (10-02, kbench v6d / v9b: exact): each routed decode text over three banks (base, ext,
    // transient) in one launch, a row's bank from its packed slot (bank << 24 | row) (twin: the stock text on bank 0)
    dsv41_exl3_b3_mul1h_k3_2304,
    dsv41_exl3_b3_mul1h_k3_5120,
    dsv41_exl3_b3_prep_in_rin,
    dsv41_exl3_b3_prep_gu_epi,
    dsv41_exl3_b3_prep_din_rin,
    dsv41_exl3_b3_moeprep_dpost,
    dsv41_exl3_b3_pair_k3_5120,
    dsv41_exl3_b3_guone_k3_2304,
    dsv41_jl_combine_bf16,
    // The K = 2 twins (10-20): layers whose EXL3 code tiles are 16 u32 words (2 bits per
    // weight) decode and prefill through these; a layer's K reaches the dispatch as its code array's tile width
    dsv41_exl3_mul1h_k2_2304,
    dsv41_exl3_mul1h_k2_5120,
    dsv41_exl3_pair_k2_5120,
    dsv41_exl3_guone_k2_2304,
    dsv41_exl3_b3_mul1h_k2_2304,
    dsv41_exl3_b3_mul1h_k2_5120,
    dsv41_exl3_b3_pair_k2_5120,
    dsv41_exl3_b3_guone_k2_2304,
    dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128,
    dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128,
    dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1,
    q3_exl3_dig_decmat_5120x2304_mul1hk2,
    q3_exl3_dig_decmat_2304x5120_mul1hk2,
};

/// The text a tag runs: its own, or a variant's base (the part before "__").
pub fn baseName(comptime name: []const u8) []const u8 {
    return if (std.mem.indexOf(u8, name, "__")) |i| name[0..i] else name;
}

/// Header texts shared by several kernels (file header_<tag>.metal).
pub const Header = enum { dig2_x, dig_mul1_k3, dig_mul1h_k3, hctape, rcproj, router_tail, woa_e4m3, index_topk, attnfuse, mxfp8_m1rows, attnfuse_s2, attnhalf_idx, pf_hc, smallk, joinless, dig_mul1h_k3_lut, hcpost_tf32, dig_mul1h_k2 };

pub const n_kernels = std.meta.fieldNames(Kernel).len;
pub const n_headers = std.meta.fieldNames(Header).len;

/// The manifest and every text, as `Registry.init` reads them. `addenda`: the port's own text
/// appended to a lane header at bind ("" for a header without one; manifest `port_addenda`).
pub const Texts = struct {
    manifest: []const u8,
    sources: [n_kernels][:0]const u8,
    headers: [n_headers][:0]const u8,
    addenda: [n_headers][:0]const u8,
};

/// The lane headers the port appends its own text to (file port_<tag>.metal): pf_hc's q3pf_ld
/// twin for `const constant` inputs (MLX binds an input of fewer than 8 elements as constant).
pub const ported = [_]Header{.pf_hc};

pub const embedded: Texts = .{
    .manifest = @embedFile(dir ++ "manifest.json"),
    .sources = embedAll(Kernel, ""),
    .headers = embedAll(Header, "header_"),
    .addenda = embedAddenda(),
};

fn embedAddenda() [n_headers][:0]const u8 {
    var out: [n_headers][:0]const u8 = @splat("");
    for (ported) |h| out[@backingInt(h)] = @embedFile(dir ++ "port_" ++ @tagName(h) ++ ".metal");
    return out;
}

fn embedAll(comptime E: type, comptime prefix: []const u8) [std.meta.fieldNames(E).len][:0]const u8 {
    @setEvalBranchQuota(100_000);
    const names = std.meta.fieldNames(E);
    var out: [names.len][:0]const u8 = undefined;
    inline for (names, 0..) |name, i| out[i] = @embedFile(dir ++ prefix ++ comptime baseName(name) ++ ".metal");
    return out;
}

pub const Refusal = error{
    ManifestNotPinned,
    ManifestSyntax,
    ManifestFormat,
    BankNotImplemented,
    UnknownKernel,
    MissingKernel,
    DuplicateKernel,
    UnknownHeader,
    MissingHeader,
    TextSha256Mismatch,
    LanePinMismatch,
    SchemaInvalid,
    MathModeNotSafe,
    GeometryInvalid,
    VariantInvalid,
    NotGpuStream,
    KernelCreateFailed,
};

/// Why the registry refused, for the one log line the caller writes.
pub const Diag = @import("sdk").Diag;

const refuse = @import("../streamed/expert/bank_file.zig").refuse;

// ── Signature and geometry ──

/// Runtime sizes a launch depends on (the site shape supplies gn / k4 / k32 / gk). keys: the
/// attention's key count; ncomp / topk / width / allfin: the index top-k's compressed count, k,
/// output width and its k >= n flag (0-d int32 scalars of the call); ring / store / kc: the
/// prefill attention's window-store rows, compressed-store rows and compressed selection width.
pub const Var = enum { rows, cap, m_tokens, experts, tgs, a_rows, gn, k4, k32, gk, seq, keys, ncomp, topk, width, allfin, ring, store, kc, src };
pub const Vars = std.enums.EnumArray(Var, u64);

/// One extent: m x ceil(value(v) / div) + add, or the constant m (+ add); at most `max` when
/// set. div / add describe the prefill attention: its key tiles (256 x ceil(k / 128)) and its
/// key count (the 128 window keys + the compressed selection width).
pub const Dim = struct {
    m: u32,
    v: ?Var = null,
    max: ?u32 = null,
    div: u32 = 1,
    add: u32 = 0,

};

/// How the self-check fills an input: `rows` inputs are sliced by rows, `bank` inputs are
/// slot banks indexed by ids, `static` inputs are the lane's own constants.
pub const Role = enum { rows, shared, bank, static, table, scalar };
pub const DomainKind = enum { normal, uniform, index, bits, zeros, values, range, signed_pow2, @"var", wave_table, slots, join_table };
pub const Domain = struct {
    kind: DomainKind,
    scale: f64 = 0,
    lo: f64 = 0,
    hi: f64 = 0,
    of: ?Var = null,
    used: ?Var = null,
    ints: []const i64 = &.{},
    floats: []const f64 = &.{},
    /// wave_table: N tiles x operands per M tile (the GEMMs' first-threadgroup column).
    tiles: u32 = 0,
    /// wave_table: the GEMM's M tile rows (64, or 128 for the 128-row texts).
    bm: u32 = 64,
};

pub const Arg = struct {
    name: [:0]const u8,
    dtype: mlx.mlx_dtype,
    shape: []const Dim,
    role: Role,
    domain: Domain,
    row_axis: u8,
};

pub const TemplateValue = kset.TemplateValue;
pub const TemplateArg = kset.TemplateArg;

/// An explicit launch of a plan kernel at (site, rows): rcproj per site and M, sinkhorn per n.
pub const Plan = struct {
    site: []const u8,
    rows: u32,
    grid: [3]u32,
    threadgroup: [3]u32,
    template: []const TemplateArg,
    output_shapes: []const []const u32,
};

/// A launch rule; `threadgroup_rule`, when set, replaces the fixed threadgroup (the stock K3's min(n, 256)).
pub const Rule = struct { grid: [3]Dim, threadgroup: [3]u32, threadgroup_rule: ?[3]Dim = null };
pub const Launch = union(enum) { rule: Rule, plans: []const Plan };

/// A plan kernel's weight site (rcproj): N outputs per group, K inputs, G groups, strides.
pub const Site = struct { name: []const u8, N: u32, K: u32, G: u32, XS: u32, XG: u32, YS: u32, YG: u32 };

pub const Check = enum { compile, row_invariance, decode_table, golden_tiles, mlx_chain, f64, layout_guard, composition, join_equiv, twin, fused };

/// The lane's own launch at sample sizes, captured by the extractor (the geometry's witness).
pub const Sample = struct {
    site: ?[]const u8,
    vars: Vars,
    grid: [3]u32,
    threadgroup: [3]u32,
    output_shapes: []const []const u32,
    output_dtypes: []const mlx.mlx_dtype,
    template: []const TemplateArg,
};

pub const Entry = struct {
    kernel: Kernel,
    /// the MLX kernel name (a variant's base tag)
    mlx_name: [:0]const u8,
    /// a variant's base kernel (null for a text of record)
    variant_of: ?Kernel = null,
    /// a variant that is its base's instantiation at another launch rule (the fused softmax's ls 128)
    launch_variant: bool = false,
    /// sha256 of the text (the variant key's first part)
    text_sha256: [32]u8,
    family: []const u8,
    phase: []const u8,
    source: [:0]const u8,
    header: ?Header,
    inputs: []const Arg,
    outputs: []const Arg,
    ensure_row_contiguous: bool,
    template: []const TemplateArg,
    launch: Launch,
    bounds: std.enums.EnumArray(Var, ?[2]u64),
    sites: []const Site,
    checks: std.EnumSet(Check),
    rows_max: u32,
    samples: []const Sample,

};

pub const max_outputs = 4;
/// Inputs per kernel: JOINLESS binds 24 sources + 3 (MLX: at most 31 buffers with the outputs).
pub const max_inputs = 27;
pub const max_rank = 4;

/// Everything `Bound.apply` hands mlx-c for one launch.
pub const LaunchConfig = struct {
    grid: [3]u32,
    threadgroup: [3]u32,
    template: []const TemplateArg,
    n_out: usize,
    out_ranks: [max_outputs]usize = @splat(0),
    out_shapes: [max_outputs][max_rank]c_int = @splat(@splat(0)),
    out_dtypes: [max_outputs]mlx.mlx_dtype = @splat(.float32),
};

// ── Golden digests the decode self-checks compare against (exl3_ref, via the extractor) ──

pub const GoldenPlane = struct {
    in_dim: u32,
    out_dim: u32,
    /// trellis bits per weight (16 K int16 per tile)
    k: u32,
    seed: u64,
    code_sha256: [32]u8,
    w_hat_sha256: [32]u8,
    onehot_rows: u32,
    onehot_rows_sha256: [32]u8,
};

pub const Golden = struct {
    mul1_table_sha256: [32]u8,
    gate_up: GoldenPlane,
    down: GoldenPlane,
    gate_up_k2: GoldenPlane,
    down_k2: GoldenPlane,

    /// The plane of a projection (its input width) at K.
    pub fn plane(self: *const Golden, in_dim: u32, k: u32) *const GoldenPlane {
        const gu = in_dim == self.gate_up.in_dim;
        return if (k == 2) (if (gu) &self.gate_up_k2 else &self.down_k2) else (if (gu) &self.gate_up else &self.down);
    }
};

// ── The registry ──

pub const Registry = struct {
    arena: std.heap.ArenaAllocator,
    entries: [n_kernels]Entry,
    /// the lane's header texts (what the lane pins cover)
    headers: [n_headers][:0]const u8,
    /// what a kernel compiles with: the lane header, + the port addendum where the manifest lists one
    compiled: [n_headers][:0]const u8,
    golden: Golden,
    codebook: []const u8,
    multiplier: u64,
    ks: []const u32,
    /// the manifest's own sha256 (hex) and the superseded manifests it lists as predecessors:
    /// every kernel entry, text and header of a predecessor is byte-identical here (the exporter
    /// checks it), so a fixture dumped against one still describes those kernels exactly.
    manifest_hex: [64]u8,
    predecessors: []const [64]u8,

    /// Parses the manifest, checks it against `pin` (hex sha256) and every text against it,
    /// once. Production passes `embedded` and `manifest_sha256`.
    pub fn init(gpa: Allocator, texts: *const Texts, pin: []const u8, diag: ?*Diag) (Refusal || Allocator.Error)!Registry {
        const got = try kset.checkPin(texts.manifest, pin, "exl3 kernels", diag);
        var reg: Registry = .{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .entries = undefined,
            .headers = texts.headers,
            .compiled = texts.headers,
            .golden = undefined,
            .codebook = "",
            .multiplier = 0,
            .ks = &.{},
            .manifest_hex = got,
            .predecessors = &.{},
        };
        errdefer reg.arena.deinit();
        const a = reg.arena.allocator();
        const m = std.json.parseFromSliceLeaky(JManifest, a, texts.manifest, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ManifestSyntax, "exl3 kernels: manifest does not parse ({t})", .{e}),
        };
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, "exl3 kernels: format \"{s}\" is not {s}", .{ m.format, format });
        if (!std.mem.eql(u8, m.bank.codebook, bank_codebook) or m.bank.multiplier != bank_multiplier or !std.mem.eql(u32, m.bank.K, &bank_ks))
            return refuse(diag, error.BankNotImplemented, "exl3 kernels: the manifest's bank ({s}, K {any}) is not the mul1 K {any} bank these texts decode", .{ m.bank.codebook, m.bank.K, bank_ks });
        reg.codebook = m.bank.codebook;
        reg.multiplier = m.bank.multiplier;
        reg.ks = m.bank.K;
        try kset.adoptHeaders(Header, &texts.headers, m.headers, "exl3 kernels", diag);
        try reg.adoptAddenda(a, texts, m.port_addenda, diag);
        try reg.adoptKernels(a, texts, m.kernels, diag);
        reg.golden = try adoptGolden(m.golden, diag);
        reg.predecessors = try adoptPredecessors(a, m.predecessors, diag);
        return reg;
    }

    pub fn deinit(self: *Registry) void {
        self.arena.deinit();
        self.* = undefined;
    }

};

/// One launch's mlx config (output shapes and dtypes, grid, threadgroup, template arguments),
/// built once (`Bound.prepare`) and handed to every launch at that geometry (`applyPrepared`).
/// mlx copies what a launch needs when it is applied, so `deinit` is safe once the last
/// `applyPrepared` using it has returned (its outputs need not be evaluated yet).
pub const Prepared = kset.Prepared(Kernel);

/// The kernels built on one GPU stream (`Registry.bind`); freed after that stream has drained: `prepare` builds a
/// launch's mlx config once (a route's construction), `applyPrepared` launches it, `apply` builds one for one launch
/// (the per-call path of the prefill routes whose geometry varies per call), `observe` installs the profile hook's
/// launch probe.
pub const Bound = kset.BoundOf(@This());

pub const launchKey = kset.launchKey;

// ── Manifest adoption (once, at init) ──

const JDim = struct { m: u32, v: ?[]const u8 = null, max: ?u32 = null, div: u32 = 1, add: u32 = 0 };
const JDomain = struct {
    kind: []const u8,
    scale: ?f64 = null,
    lo: ?f64 = null,
    hi: ?f64 = null,
    of: ?[]const u8 = null,
    used: ?[]const u8 = null,
    ints: ?[]const i64 = null,
    floats: ?[]const f64 = null,
    tiles: ?u32 = null,
    bm: ?u32 = null,
};
const JArg = struct { name: []const u8, dtype: []const u8, shape: []const JDim, role: ?[]const u8 = null, domain: ?JDomain = null, row_axis: u8 = 0 };
const JTemplate = struct { name: []const u8, int: ?i64 = null, dtype: ?[]const u8 = null };
const JPlan = struct { site: []const u8, rows: u32, grid: [3]u32, threadgroup: [3]u32, template: []const JTemplate, output_shapes: []const []const u32 };
const JVarBound = struct { name: []const u8, lo: u64, hi: u64 };
const JVarValue = struct { name: []const u8, value: u64 };
const JSample = struct {
    site: ?[]const u8 = null,
    vars: []const JVarValue,
    grid: [3]u32,
    threadgroup: [3]u32,
    output_shapes: []const []const u32,
    output_dtypes: []const []const u8,
    template: []const JTemplate,
};
const JText = struct { id: ?[]const u8 = null, file: []const u8, sha256: []const u8, bytes: u64 };
const JPin = struct { symbol: []const u8, hash_of: []const u8, sha256: []const u8, join: ?[]const u8 = null };
const JSelfCheck = struct { checks: []const []const u8, rows_max: u32 = 0 };
const JSite = struct { name: []const u8, N: u32, K: u32, G: u32, XS: u32, XG: u32, YS: u32, YG: u32 };
const JVariant = struct { of: []const u8, mlx_name: []const u8, launch_only: bool = false };
const JKernel = struct {
    name: []const u8,
    variant: ?JVariant = null,
    family: []const u8,
    phase: []const u8,
    source: JText,
    header: ?JText = null,
    lane_pins: []const JPin,
    ensure_row_contiguous: bool,
    atomic_outputs: bool,
    math_mode: []const u8,
    inputs: []const JArg,
    outputs: []const JArg,
    template: []const JTemplate,
    vars: []const JVarBound,
    grid: ?[3]JDim = null,
    threadgroup: ?[3]u32 = null,
    threadgroup_rule: ?[3]JDim = null,
    plans: ?[]const JPlan = null,
    sites: []const JSite = &.{},
    launch_samples: []const JSample,
    self_check: JSelfCheck,
};
const JPlane = struct { projection: [2]u32, K: u32, codebook: []const u8, seed: u64, code_sha256: []const u8, w_hat_sha256: []const u8, onehot_rows: u32, onehot_rows_sha256: []const u8, states_covered_by_onehot_rows: u32 };
const JGolden = struct {
    mul1_table: struct { states: u32, sha256: []const u8 },
    planes: struct { gate_up: JPlane, down: JPlane, gate_up_k2: JPlane, down_k2: JPlane },
};
const JBank = struct { codebook: []const u8, multiplier: u64, K: []const u32 };
const JPredecessor = struct { manifest_sha256: []const u8, kernels: []const []const u8, headers: []const []const u8 };
const JManifest = struct { format: []const u8, bank: JBank, headers: []const JText, kernels: []const JKernel, golden: JGolden, predecessors: []const JPredecessor = &.{}, port_addenda: []const JText = &.{} };

/// The predecessors' sha256s, each checked: 64 hex, and every kernel / header it lists is one this
/// build embeds (a predecessor names a subset of this manifest).
fn adoptPredecessors(a: Allocator, js: []const JPredecessor, diag: ?*Diag) (Refusal || Allocator.Error)![]const [64]u8 {
    const out = try a.alloc([64]u8, js.len);
    for (js, out) |j, *o| {
        if (hexSha(j.manifest_sha256) == null) return refuse(diag, error.SchemaInvalid, "exl3 kernels: predecessor sha256 \"{s}\" is not 64 hex", .{j.manifest_sha256});
        for (j.kernels) |name| if (std.meta.stringToEnum(Kernel, name) == null) return refuse(diag, error.SchemaInvalid, "exl3 kernels: predecessor {s} lists {s}, not a kernel of this build", .{ j.manifest_sha256[0..8], name });
        for (j.headers) |name| if (std.meta.stringToEnum(Header, name) == null) return refuse(diag, error.SchemaInvalid, "exl3 kernels: predecessor {s} lists header {s}, not one of this build", .{ j.manifest_sha256[0..8], name });
        @memcpy(o, j.manifest_sha256[0..64]);
    }
    return out;
}

const hexSha = kset.hexSha;

fn adoptGolden(g: JGolden, diag: ?*Diag) Refusal!Golden {
    const bad = error.SchemaInvalid;
    return .{
        .mul1_table_sha256 = hexSha(g.mul1_table.sha256) orelse return refuse(diag, bad, "exl3 kernels: golden mul1 table sha256", .{}),
        .gate_up = try adoptPlane(g.planes.gate_up, 3, diag),
        .down = try adoptPlane(g.planes.down, 3, diag),
        .gate_up_k2 = try adoptPlane(g.planes.gate_up_k2, 2, diag),
        .down_k2 = try adoptPlane(g.planes.down_k2, 2, diag),
    };
}

fn adoptPlane(p: JPlane, k: u32, diag: ?*Diag) Refusal!GoldenPlane {
    if (p.K != k or std.mem.indexOfScalar(u32, &bank_ks, k) == null or !std.mem.eql(u8, p.codebook, "mul1") or p.states_covered_by_onehot_rows != 65536 or p.onehot_rows == 0 or p.onehot_rows > p.projection[0])
        return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden plane {d}x{d}", .{ p.projection[0], p.projection[1] });
    return .{
        .in_dim = p.projection[0],
        .out_dim = p.projection[1],
        .k = k,
        .seed = p.seed,
        .code_sha256 = hexSha(p.code_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden code sha256", .{}),
        .w_hat_sha256 = hexSha(p.w_hat_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden w_hat sha256", .{}),
        .onehot_rows = p.onehot_rows,
        .onehot_rows_sha256 = hexSha(p.onehot_rows_sha256) orelse return refuse(diag, error.SchemaInvalid, "exl3 kernels: golden rows sha256", .{}),
    };
}

// ── The EXL3 host decode the decode self-checks compare against ──

/// exllamav3 tensor_core_perm (quantize.py L22-44): the row-major tile slot of code position p.
pub const tile_perm: [256]u8 = blk: {
    var perm: [256]u8 = undefined;
    for (0..32) |t| {
        const r0 = (t % 4) * 2;
        const rows = [4]usize{ r0, r0 + 1, r0 + 8, r0 + 9 };
        for (0..8) |j| perm[t * 8 + j] = @intCast(rows[j % 4] * 16 + t / 4 + (if (j < 4) 0 else 8));
    }
    break :blk perm;
};

/// The trellis states of one tile's 256 code positions (exl3_dq.cuh L15-31): the 16 stream
/// bits ending at bit (p + 1) K, circularly, of the little-endian u32 view of the tile.
fn tileStates(words: []const i16, K: usize, out: *[256]u16) void {
    const nw = 8 * K;
    var u: [24]u64 = undefined;
    for (0..nw) |m| u[m] = @as(u64, @as(u16, @bitCast(words[2 * m]))) | (@as(u64, @as(u16, @bitCast(words[2 * m + 1]))) << 16);
    for (out, 0..) |*o, p| {
        const b1 = (p + 1) * K + 256 * K;
        const hi_word = (b1 - 16) / 32;
        const lo_word = (b1 - 1) / 32;
        const s0: u6 = @intCast((lo_word + 1) * 32 - b1);
        o.* = @truncate(((u[hi_word % nw] << 32) | u[lo_word % nw]) >> s0);
    }
}

/// exl3_ref.reconstruct: code int16 [nI, nJ, 16K] -> W_hat f16 bits [16 nI, 16 nJ];
/// `states` (optional) receives each weight's trellis state in the same place.
pub fn reconstruct(code: []const i16, n_i: usize, n_j: usize, K: usize, table: *const [65536]u16, w: []u16, states: ?[]u16) void {
    const tw = 16 * K;
    const cols = n_j * 16;
    var st: [256]u16 = undefined;
    for (0..n_i) |ti| {
        for (0..n_j) |tj| {
            tileStates(code[(ti * n_j + tj) * tw ..][0..tw], K, &st);
            for (st, 0..) |s, p| {
                const pos: usize = tile_perm[p];
                const at = (ti * 16 + pos / 16) * cols + tj * 16 + pos % 16;
                w[at] = table[s];
                if (states) |o| o[at] = s;
            }
        }
    }
}

pub fn splitmix64(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// A golden plane's seeded code: the little-endian int16 view of splitmix64(seed).
pub fn synthPlane(a: Allocator, g: *const GoldenPlane) ![]i16 {
    const n = (g.in_dim / 16) * (g.out_dim / 16) * 16 * g.k;
    const out = try a.alloc(i16, n);
    var s = g.seed;
    var i: usize = 0;
    while (i < n) : (i += 4) {
        const z = splitmix64(&s);
        inline for (0..4) |j| out[i + j] = @bitCast(@as(u16, @truncate(z >> (16 * j))));
    }
    return out;
}

/// W_hat of a golden plane (f16 bits, row-major [IN, OUT]) and each weight's state.
pub const HostPlane = struct {
    code: []i16,
    w: []u16,
    states: []u16,

    pub fn init(a: Allocator, g: *const GoldenPlane, table: *const [65536]u16) !HostPlane {
        const code = try synthPlane(a, g);
        errdefer a.free(code);
        const n = @as(usize, g.in_dim) * g.out_dim;
        const w = try a.alloc(u16, n);
        errdefer a.free(w);
        const states = try a.alloc(u16, n);
        reconstruct(code, g.in_dim / 16, g.out_dim / 16, g.k, table, w, states);
        return .{ .code = code, .w = w, .states = states };
    }

    pub fn deinit(self: *HostPlane, a: Allocator) void {
        a.free(self.code);
        a.free(self.w);
        a.free(self.states);
        self.* = undefined;
    }
};

// ── Tests (host only: nothing here creates an MLX array or kernel) ──

const testing = std.testing;

const shaHex = kset.shaHex;
