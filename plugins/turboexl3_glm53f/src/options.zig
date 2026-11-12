//! turboexl3_glm53f's switches as `Manifest.options` (mlx-serve #1131: `sdk.Plugin.options`, `abi.dylib.Option`): every
//! `TURBOEXL3_GLM53F_*` variable the plugin reads at load, its kind (`bool` "0" or "1", `number` a plain decimal,
//! `text` up to 1 KiB), the plugin's own value when none is set and one line of help, which the app shows on the
//! plugin's page. The value when none is set is the frozen exact configuration's (2026-11-12: the 64-entry control of
//! 2026-11-10 with DSA_HOLD and PF_TAIL_EVAL on, plugin ac87fe1 on the A/B host, the standard 16K coding prompt's
//! greedy text c3c346222307): every switch on, the text and number switches at their accepted values. `defaultOf`
//! answers it for an unset variable (`routed_t3.processEnv`, plugin.zig's `envSpan`, the only two readers), so a plain
//! install serves that configuration; "0" (or another value) in the environment or the app's options turns a switch off
//! or changes it. A switch whose mechanism is a seam of the kit's stream (src/streamed/glm5n/experts.zig `seams`)
//! defaults off on a kit whose stream lacks it, and is refused at load when set there. PF_INT8 (not exact),
//! LOGITS_DUMP (a check), PF_RUN_TRIM and DEMAND_HELP (both rejected in their A/Bs) are off unset (`off`). A switch
//! that needs another (`With X:` in its help) is refused at load when that one is set off and it is not. The device
//! tests' gates (`TURBOEXL3_GLM53F_*_GPU`) are the tests' own, not the plugin's. MLX-free.
//! plugin.zig declares them only on a kit whose `sdk.Plugin` has `options`; an older kit builds the plugin unchanged.
const std = @import("std");
const kit = @import("mlx_serve");
/// The bank's switches whose mechanism is a seam of the kit's stream, and whether this kit's stream has each
/// (src/streamed/glm5n/experts.zig `seams`): without its seam a switch defaults off and is refused when set.
const seams = @import("mlx_stream").glm5n_source.seams;

/// An option's kind, as the kit's `abi.dylib.OptionKind` names it.
pub const Kind = enum { bool, number, text };
/// One switch: its variable, kind, the plugin's own value when none is set and one line of help.
pub const Decl = struct { name: []const u8, kind: Kind, default: []const u8, help: []const u8 };

pub const all = [_]Decl{
    .{ .name = "TURBOEXL3_GLM53F_LAYER_BILL", .kind = .bool, .default = "1", .help = "Bill the slot rows at each routed layer's own row size (the pack's v2 manifest), not the widest" },
    .{ .name = "TURBOEXL3_GLM53F_DECODE_ROOM", .kind = .bool, .default = "1", .help = "Give the decode phase the rows the prompt's bill leaves beside the decode's own" },
    .{ .name = "TURBOEXL3_GLM53F_GROW_UNFILLED", .kind = .bool, .default = "1", .help = "Take the decode grow's new slot rows as MLX buffers without the zero fill" },
    .{ .name = "TURBOEXL3_GLM53F_PROMPT_KEEP", .kind = .bool, .default = if (seams.prompt_keep) "1" else "0", .help = "No prompt-phase admission evicts a resident expert (needs the kit stream's prompt_keep seam)" },
    .{ .name = "TURBOEXL3_GLM53F_SPEC_PICK1", .kind = .bool, .default = "1", .help = "The decode lookahead reads ahead one record a layer call, its most likely (off: 2); the read pool unchanged" },
    .{ .name = "TURBOEXL3_GLM53F_DEMAND_SPLIT", .kind = .bool, .default = if (seams.demand_split) "1" else "0", .help = "A decode call's one record read in a page-aligned piece per read pool worker but one (needs the kit stream's demand_split seam)" },
    .{ .name = "TURBOEXL3_GLM53F_SHARED_HOIST", .kind = .bool, .default = "1", .help = "A decode or verify call builds its layer's shared expert before the routed call" },
    .{ .name = "TURBOEXL3_GLM53F_READ_AHEAD3", .kind = .bool, .default = "1", .help = "A decode or verify window scores the first routed layer's router a layer early" },
    .{ .name = "TURBOEXL3_GLM53F_DEMAND_HELP", .kind = .number, .default = "0", .help = "0 off (rejected in its A/B); 2 decode's read pool puts its demand first and a waiting demand read takes its claimed record's untaken pieces (the kit stream's demandFirst(2))" },
    .{ .name = "TURBOEXL3_GLM53F_ADOPT_PIECES", .kind = .bool, .default = if (seams.adopt_pieces) "1" else "0", .help = "A demand read served from a lookahead record copies its bytes in pieces across the read pool's workers (needs the kit stream's adopt_pieces seam)" },
    .{ .name = "TURBOEXL3_GLM53F_BILL_FIT", .kind = .bool, .default = "1", .help = "Fix the slot rows by the GPU ceiling and the context limit, not the memory available at load" },
    .{ .name = "TURBOEXL3_GLM53F_FIT_INUSE", .kind = .bool, .default = "1", .help = "With BILL_FIT: allow the memory outside the server that is in use at load plus 1 GiB, not the fit's fixed 13.9 GiB" },
    .{ .name = "TURBOEXL3_GLM53F_SEED_DECODE", .kind = .bool, .default = "1", .help = "Seed the residents from the pack's decode-only usage record" },
    .{ .name = "TURBOEXL3_GLM53F_KEEP_WARM", .kind = .bool, .default = "1", .help = "Keep the GPU warm through the expert stream's waits (one small FMA chain on its own queue)" },
    .{ .name = "TURBOEXL3_GLM53F_PF_ROWS", .kind = .bool, .default = "1", .help = "A prompt call and a cold-row chunk run turboexl3's moeClampedRows" },
    .{ .name = "TURBOEXL3_GLM53F_DEC_BF16", .kind = .bool, .default = "1", .help = "A decode wave's banked down finish stores bf16" },
    .{ .name = "TURBOEXL3_GLM53F_PF_FEED", .kind = .bool, .default = "1", .help = "A prompt call returns with its last chunks in flight; one wait covers its group" },
    .{ .name = "TURBOEXL3_GLM53F_PF_ALIGN", .kind = .bool, .default = "1", .help = "A prompt call's chunks end where no expert's 128-row run is split" },
    .{ .name = "TURBOEXL3_GLM53F_BANKED", .kind = .text, .default = "g4m3.t4.t4", .help = "The banked decode's schedule: g4m3.t4.t4" },
    .{ .name = "TURBOEXL3_GLM53F_DEC", .kind = .text, .default = "g1", .help = "The dense decode GEMV: g1" },
    .{ .name = "TURBOEXL3_GLM53F_PF_OPTS", .kind = .text, .default = "bk128", .help = "The routed prefill's options: bk128" },
    .{ .name = "TURBOEXL3_GLM53F_PG2", .kind = .text, .default = "50115151111", .help = "The trunk's prefill tile codes on turboexl3's pg2, one a linear (11)" },
    .{ .name = "TURBOEXL3_GLM53F_PF_STAGE", .kind = .bool, .default = "1", .help = "A prompt call routes its layer's misses before its routing barrier" },
    .{ .name = "TURBOEXL3_GLM53F_PF_DRAIN_LAG", .kind = .bool, .default = "1", .help = "With PF_FEED: a prompt MoE group's drain waits for the next group's first chunk" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_PROLOGUE", .kind = .bool, .default = "1", .help = "A KDA layer's prompt prologue (conv, slices, norms) in one pass over qkv" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_NORMGATE", .kind = .bool, .default = "1", .help = "A KDA layer's prompt gated norm in one pass" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_PERCORE", .kind = .bool, .default = "1", .help = "Not exact: the KDA prompt recurrence and gate on the per-core route" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_CHUNK", .kind = .bool, .default = "1", .help = "Not exact: the KDA prompt recurrence and gate as the chunkwise delta rule" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_FOLD", .kind = .bool, .default = "1", .help = "With KDA_CHUNK: the KDA prologue folded into the chunkwise launch" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_PIPE", .kind = .bool, .default = "1", .help = "With KDA_CHUNK: a KDA layer's prompt forward in 4,096-row spans, each span's qkv linear on a side stream" },
    .{ .name = "TURBOEXL3_GLM53F_ASYNC_EVAL", .kind = .bool, .default = "1", .help = "A prompt forward's cadence eval handed to the GPU without the host's wait" },
    .{ .name = "TURBOEXL3_GLM53F_PF_REUSE", .kind = .bool, .default = "1", .help = "A prompt pass's buffers reused: a KDA run's held in MLX's cache, the routed prefill's at size classes" },
    .{ .name = "TURBOEXL3_GLM53F_PF_MOE_HOLD", .kind = .bool, .default = "1", .help = "With PF_REUSE: a DSA layer's routed prefill finds its first two chunks' arrays in MLX's cache, held through its attention" },
    .{ .name = "TURBOEXL3_GLM53F_PF_MOE_HOLD_TOP", .kind = .bool, .default = "1", .help = "With PF_MOE_HOLD: the hold's hand-back raises the cache limit from the cache's bytes when they are over it, so every held buffer is cached" },
    .{ .name = "TURBOEXL3_GLM53F_PF_INT8", .kind = .number, .default = "0", .help = "Not exact: 0 off; 1 the routed prefill's gate|up and down GEMMs on int8, a scale each activation 128-block; 2 one scale a row; 3 1 and the trunk's KDA qkv and o_proj at prefill rows on int8; 4 1 and every trunk linear at prefill rows on int8" },
    .{ .name = "TURBOEXL3_GLM53F_LOGITS_DUMP", .kind = .text, .default = "", .help = "A check, never served: each forward's rows' top-64 log-probabilities written to this absolute directory" },
    .{ .name = "TURBOEXL3_GLM53F_HC_FUSE", .kind = .bool, .default = "1", .help = "A prompt sublayer's hyper-connection expand and the next sublayer's collapse as one text" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_VERIFY", .kind = .bool, .default = "1", .help = "The KDA prologue, gate and gated-norm texts in a verify forward too" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_LAZY_SEQ", .kind = .bool, .default = "1", .help = "A KDA layer's verify writes only its final state; its per-row states run only when a partial accept's rollback reads them" },
    .{ .name = "TURBOEXL3_GLM53F_ROUTER", .kind = .bool, .default = "1", .help = "A 2- to 4-row decode call's router in two launches" },
    .{ .name = "TURBOEXL3_GLM53F_PREDICT", .kind = .bool, .default = "1", .help = "The decode lookahead's next-layer scores in two launches" },
    .{ .name = "TURBOEXL3_GLM53F_ROUTER_ONE", .kind = .bool, .default = "1", .help = "With ROUTER or PREDICT: each of their calls in one launch, its last threadgroup running the second phase" },
    .{ .name = "TURBOEXL3_GLM53F_MOE_COMBINE1", .kind = .bool, .default = "1", .help = "A decode call's routed join and combine in one launch reading the routed rows in place" },
    .{ .name = "TURBOEXL3_GLM53F_ROUTE_VIEW", .kind = .bool, .default = "1", .help = "A decode call's route ids as an int32 view of the router's uint32 ids instead of a cast" },
    .{ .name = "TURBOEXL3_GLM53F_SMALLOP_FUSE", .kind = .bool, .default = "1", .help = "The small ops around the decode and verify forwards' big launches merged" },
    .{ .name = "TURBOEXL3_GLM53F_HEAD_LAST_ROW", .kind = .bool, .default = "1", .help = "The MTP head's merged first draft step runs its routed experts for its last row alone" },
    .{ .name = "TURBOEXL3_GLM53F_TRUNK_F16_FREE", .kind = .bool, .default = "1", .help = "The trunk's plain f16 matrices free their f16 sources once their bf16 copies exist" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_VATTN", .kind = .bool, .default = "1", .help = "A DSA layer's verify-window sparse attention without its gather: both matmuls read the latent at the selected rows" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_ONEP", .kind = .bool, .default = "1", .help = "Not exact (P rounded once to fp16): a DSA layer's prompt sparse attention as one P piece" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_VSEL", .kind = .bool, .default = "1", .help = "A DSA layer's verify window selects in two launches: every row's indexer scores in one, their top-k in one" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_GLM", .kind = .bool, .default = "1", .help = "A DSA layer's prompt indexer as one GLM-5.3-Flash launch and its output in one copy (the same bits)" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_LFUSE", .kind = .bool, .default = "1", .help = "A DSA layer's verify-window sparse latent attention in two launches at GLM-5.3-Flash's dims: scores and mask in one, softmax and values in one" },
    // the verify's launch trims (src/kda_trim.zig, src/cached_scalars.zig, src/host_zero.zig)
    .{ .name = "TURBOEXL3_GLM53F_KDA_CONVIN", .kind = .bool, .default = "1", .help = "With KDA_VERIFY: a KDA verify's prologue text writes the capture's conv input itself, no concatenate" },
    .{ .name = "TURBOEXL3_GLM53F_KDA_EXPSIG", .kind = .bool, .default = "1", .help = "A KDA verify takes exp(A_log) made once a layer and beta's sigmoid inside its capture launch" },
    .{ .name = "TURBOEXL3_GLM53F_CACHED_SCALARS", .kind = .bool, .default = "1", .help = "The clamped SwiGLU's limit made once, not cast to the activation's dtype at every call" },
    .{ .name = "TURBOEXL3_GLM53F_HOST_ZERO", .kind = .bool, .default = "1", .help = "DEC-ONE's arrival counters made as zeros on the host, not by a launch" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_TAILS", .kind = .bool, .default = "1", .help = "A decode or verify forward reads its DSA layers' indexer tail rows first, so each cache append reuses its buffer" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_ROWS", .kind = .bool, .default = "1", .help = "A decode or verify window's DSA indexer rows built by the pooled-keys launch: the gate's cast and concatenate folded in" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_WFOLD", .kind = .bool, .default = "1", .help = "A decode or verify window's DSA selection reads the head weights as projected: cast and scales in the scores launch" },
    // the MTP head's router in ROUTER's plan (src/fused_router.zig)
    .{ .name = "TURBOEXL3_GLM53F_MTP_ROUTER", .kind = .bool, .default = "1", .help = "With ROUTER: the MTP head's router registered with its plan too (a 2- to 4-row draft step), its routes counted apart" },
    .{ .name = "TURBOEXL3_GLM53F_ROUTER_ONE1", .kind = .bool, .default = "1", .help = "With ROUTER: a registered layer's one-row router call in one launch (MLX's gemv text, then the stock router text)" },
    // the MTP draft's step 0 on SMALLOP's texts (src/draft_s0.zig)
    .{ .name = "TURBOEXL3_GLM53F_DRAFT_S0", .kind = .bool, .default = "1", .help = "With SMALLOP_FUSE: the MTP draft's step 0 (2 to 4 rows) front, post-norm and last residual on SMALLOP's texts" },
    // a verify linear's producer writing its prepared input, DEC-ONE's GEMV alone after it (src/dec_prep.zig)
    .{ .name = "TURBOEXL3_GLM53F_KDA_OPREP", .kind = .bool, .default = "1", .help = "A verify's KDA gated norm writes the out projection's prepared input: one launch fewer a KDA layer" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_OPREP", .kind = .bool, .default = "1", .help = "A verify's DSA out projection prepares its input from the per-head values in place: one launch fewer a DSA layer" },
    .{ .name = "TURBOEXL3_GLM53F_DSA_HOLD", .kind = .bool, .default = "1", .help = "Memory for compute: a prompt DSA layer's q_abs and o_lat held for its next DSA layer's allocations" },
    .{ .name = "TURBOEXL3_GLM53F_PF_RUN_TRIM", .kind = .bool, .default = "0", .help = "With PF_REUSE: a KDA run's cache trimmed during its last layer's routed call, not at the next DSA layer" },
    .{ .name = "TURBOEXL3_GLM53F_PF_TAIL_EVAL", .kind = .bool, .default = "1", .help = "A prompt layer's tail handed to the GPU at the layer's end between the cadence's evals (the same bits)" },
};

/// The plugin's own value of switch `name` when the environment sets none (`all`'s default), or null for a name it does
/// not declare (the device tests' gates).
pub fn defaultOf(name: []const u8) ?[]const u8 {
    for (all) |d| if (std.mem.eql(u8, d.name, name)) return d.default;
    return null;
}

/// `all` as options of the kit's type `O` (`sdk.Plugin.options`' element: name, kind, default, help).
pub fn forKit(comptime O: type) [all.len]O {
    var out: [all.len]O = undefined;
    for (all, &out) |d, *o| o.* = .{ .name = d.name, .kind = switch (d.kind) {
        .bool => .bool,
        .number => .number,
        .text => .text,
    }, .default = d.default, .help = d.help };
    return out;
}

test "turboexl3_glm53f stream options: every switch declared once with a line of help, each name the kit's option name rule and each default a value of its kind, where the kit has the rules" {
    const t = std.testing;
    for (all, 0..) |d, i| {
        try t.expect(std.mem.startsWith(u8, d.name, "TURBOEXL3_GLM53F_") and d.name.len <= 128 and d.help.len > 0);
        for (all[0..i]) |e| try t.expect(!std.mem.eql(u8, d.name, e.name));
    }
    // mlx-serve #1131's own checks (`abi.dylib.optionNameOk`, `valueOk`; a 26.11.2 kit has neither)
    if (comptime @hasDecl(kit.abi, "dylib") and @hasDecl(kit.abi.dylib, "optionNameOk") and @hasDecl(kit.abi.dylib, "valueOk")) {
        const dy = kit.abi.dylib;
        for (all) |d| {
            try t.expect(dy.optionNameOk("turboexl3_glm53f", d.name));
            const k: dy.OptionKind = switch (d.kind) {
                .bool => .bool,
                .number => .number,
                .text => .text,
            };
            try t.expect(dy.valueOk(k, d.default));
        }
    }
}

/// The switches whose mechanism is a seam of the kit's stream, by name, and whether this kit's stream has it.
fn seamOf(name: []const u8) ?bool {
    const S = struct { n: []const u8, on: bool };
    const list = [_]S{
        .{ .n = "TURBOEXL3_GLM53F_PROMPT_KEEP", .on = seams.prompt_keep },
        .{ .n = "TURBOEXL3_GLM53F_DEMAND_SPLIT", .on = seams.demand_split },
        .{ .n = "TURBOEXL3_GLM53F_ADOPT_PIECES", .on = seams.adopt_pieces },
    };
    for (list) |x| if (std.mem.eql(u8, x.n, name)) return x.on;
    return null;
}

/// The switches the served configuration keeps off ("0", or "" for a text switch): PF_INT8 (not exact; a server-start
/// option, 1 to 4), LOGITS_DUMP (a check, never served), and PF_RUN_TRIM and DEMAND_HELP (exact, but their A/Bs
/// rejected them; DEMAND_HELP on the kit stream's `demandFirst(2)` decoded 3.1% slower in g53g2h1).
const off = [_][]const u8{
    "TURBOEXL3_GLM53F_PF_INT8",
    "TURBOEXL3_GLM53F_LOGITS_DUMP",
    "TURBOEXL3_GLM53F_PF_RUN_TRIM",
    "TURBOEXL3_GLM53F_DEMAND_HELP",
};
fn isOff(name: []const u8) bool {
    for (off) |o| if (std.mem.eql(u8, o, name)) return true;
    return false;
}

test "turboexl3_glm53f stream options: every default is the frozen exact configuration's: each bool on, each number and text switch its accepted value, a switch whose seam the kit's stream lacks off, the served configuration's off ones off, and defaultOf answers it" {
    const t = std.testing;
    for (off) |o| try t.expect(defaultOf(o) != null);
    for (all) |d| {
        if (isOff(d.name)) {
            try t.expectEqualStrings(if (d.kind == .text) "" else "0", d.default);
            try t.expectEqualStrings(d.default, defaultOf(d.name).?);
            continue;
        }
        if (seamOf(d.name)) |on| if (!on) {
            try t.expectEqualStrings("0", d.default);
            try t.expectEqualStrings(d.default, defaultOf(d.name).?);
            continue;
        };
        switch (d.kind) {
            .bool => try t.expectEqualStrings("1", d.default),
            .number, .text => try t.expect(d.default.len > 0 and !std.mem.eql(u8, d.default, "0")),
        }
        try t.expectEqualStrings(d.default, defaultOf(d.name).?);
    }
    try t.expectEqual(@as(usize, 67), all.len);
    try t.expectEqualStrings("g4m3.t4.t4", defaultOf("TURBOEXL3_GLM53F_BANKED").?);
    try t.expectEqualStrings("g1", defaultOf("TURBOEXL3_GLM53F_DEC").?);
    try t.expectEqualStrings("bk128", defaultOf("TURBOEXL3_GLM53F_PF_OPTS").?);
    try t.expectEqualStrings("50115151111", defaultOf("TURBOEXL3_GLM53F_PG2").?);
    try t.expectEqualStrings("0", defaultOf("TURBOEXL3_GLM53F_DEMAND_HELP").?);
    try t.expectEqualStrings("0", defaultOf("TURBOEXL3_GLM53F_PF_INT8").?);
    try t.expectEqualStrings("", defaultOf("TURBOEXL3_GLM53F_LOGITS_DUMP").?);
    try t.expectEqualStrings("0", defaultOf("TURBOEXL3_GLM53F_PF_RUN_TRIM").?);
    try t.expectEqualStrings("1", defaultOf("TURBOEXL3_GLM53F_DSA_HOLD").?);
    try t.expectEqualStrings("1", defaultOf("TURBOEXL3_GLM53F_PF_TAIL_EVAL").?);
    try t.expect(defaultOf("TURBOEXL3_GLM53F_ADOPT_ZC") == null);
    try t.expect(defaultOf("TURBOEXL3_GLM53F_KDA_GPU") == null);
}
