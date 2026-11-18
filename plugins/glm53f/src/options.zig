//! glm53f's switches as `Manifest.options`: every `GLM53F_*` variable the plugin reads, its kind (`bool` "0" or "1"),
//! the plugin's own value when none is set and one line of help. Every switch is on by default: each one won its
//! standard A/B, so the default is the fastest accepted stack (the three rounding-class ones, `KDA_PROMPT_CHUNK`,
//! `KDA_PROMPT_PERCORE` and `DSA_PROMPT_ONEP`, included). A switch reads its variable once; a value is on unless it
//! is empty or starts with '0'. MLX-free.
const std = @import("std");
const kit = @import("mlx_serve");

pub const Switch = enum {
    kda_verify,
    kda_expsig,
    kda_lazy_seq,
    kda_prompt_prologue,
    kda_prompt_normgate,
    kda_prompt_percore,
    kda_prompt_chunk,
    dsa_prompt_indexer,
    dsa_prompt_onep,
    dsa_tails,
    head_last_row,
    prompt_async_eval,
    router_nax,
    router_one1,
    whole_prompt,
    pf_reuse,
};

pub const Decl = struct { sw: Switch, name: []const u8, default: []const u8, help: []const u8 };

pub const all = [_]Decl{
    .{ .sw = .kda_verify, .name = "GLM53F_KDA_VERIFY", .default = "1", .help = "A KDA layer's verify rows take the packed prework, the bounded gate in one launch and the norm-gate epilogue (the chain's bytes)" },
    .{ .sw = .kda_expsig, .name = "GLM53F_KDA_EXPSIG", .default = "1", .help = "With KDA_VERIFY: the one-launch gate makes exp(A_log) in its launch (the same bits)" },
    .{ .sw = .kda_lazy_seq, .name = "GLM53F_KDA_LAZY_SEQ", .default = "1", .help = "A KDA verify's per-row states come from a launch MLX runs only when a rollback reads them (the same bits)" },
    .{ .sw = .kda_prompt_prologue, .name = "GLM53F_KDA_PROMPT_PROLOGUE", .default = "1", .help = "A KDA layer's prompt rows take the conv, SiLU, q/k norms and scales in one launch (the chain's bytes)" },
    .{ .sw = .kda_prompt_normgate, .name = "GLM53F_KDA_PROMPT_NORMGATE", .default = "1", .help = "A KDA layer's prompt rows take the norm-gate epilogue in one launch (the chain's bytes)" },
    .{ .sw = .kda_prompt_percore, .name = "GLM53F_KDA_PROMPT_PERCORE", .default = "1", .help = "Not exact: a KDA layer's prompt recurrence on the per-core kernel at up to two threadgroups a core" },
    .{ .sw = .kda_prompt_chunk, .name = "GLM53F_KDA_PROMPT_CHUNK", .default = "1", .help = "Not exact: a KDA layer's prompt recurrence as the chunkwise delta rule (16-row chunks); a load it cannot serve is refused" },
    .{ .sw = .dsa_prompt_indexer, .name = "GLM53F_DSA_PROMPT_INDEXER", .default = "1", .help = "A DSA layer's prompt rows: the indexer's scores, top-k and expansion in one launch, the output in one copy (the same tokens and bytes)" },
    .{ .sw = .dsa_prompt_onep, .name = "GLM53F_DSA_PROMPT_ONEP", .default = "1", .help = "Not exact: a DSA prompt pass's sparse latent attention with P as one fp16 piece" },
    .{ .sw = .dsa_tails, .name = "GLM53F_DSA_TAILS", .default = "1", .help = "A decode, verify or draft forward reads its DSA layers' indexer tail rows first, in one launch (the same rows)" },
    .{ .sw = .head_last_row, .name = "GLM53F_HEAD_LAST_ROW", .default = "1", .help = "The MTP head's merged first draft step runs its routed experts for its last row alone (that row's bytes)" },
    .{ .sw = .prompt_async_eval, .name = "GLM53F_PROMPT_ASYNC_EVAL", .default = "1", .help = "A prompt forward's cadence evals go to the GPU without the host's wait (the same arrays and order)" },
    .{ .sw = .router_nax, .name = "GLM53F_ROUTER_NAX", .default = "1", .help = "The 4096 x 288 sigmoid router keeps its stored copy and routes 2 to 4 rows on MLX's split-K NAX chains (the stock route's bytes)" },
    .{ .sw = .router_one1, .name = "GLM53F_ROUTER_ONE1", .default = "1", .help = "With ROUTER_NAX: a one-row router call in one launch (the stock route's bytes)" },
    .{ .sw = .whole_prompt, .name = "GLM53F_WHOLE_PROMPT", .default = "1", .help = "A streaming pack takes the whole prompt in one forward and runs its last chunk and held-back tail as a pair, their routed experts in one bank call (each span's bytes)" },
    .{ .sw = .pf_reuse, .name = "GLM53F_PF_REUSE", .default = "1", .help = "A prompt forward's KDA runs keep their buffers in MLX's cache from a run's first layer to its last, the limit raised by the bill's layer envelope a row (the same kernels and bytes)" },
};

comptime {
    for (all, 0..) |d, i| if (@backingInt(d.sw) != i) @compileError("options: `all` out of `Switch` order at " ++ d.name);
    if (all.len != @typeInfo(Switch).@"enum".field_names.len) @compileError("options: a switch without its declaration");
}

/// A test's value for a switch, over the environment's (null: the environment's).
pub var forced: [all.len]?bool = @splat(null);
var cached: [all.len]?bool = @splat(null);

/// Whether a raw value turns a switch on: set, and not empty or "0...".
pub fn valueOn(v: []const u8) bool {
    return v.len > 0 and v[0] != '0';
}

/// Switch `sw` as set (read once), else its default.
pub fn on(sw: Switch) bool {
    const i = @backingInt(sw);
    if (forced[i]) |v| return v;
    if (cached[i]) |v| return v;
    const d = all[i];
    const raw: []const u8 = if (std.c.getenv(@ptrCast(d.name.ptr))) |p| std.mem.span(p) else d.default;
    cached[i] = valueOn(raw);
    return cached[i].?;
}

/// The default of switch `sw`, as a bool.
pub fn defaultOn(sw: Switch) bool {
    return valueOn(all[@backingInt(sw)].default);
}

/// `all` as the kit's options (`sdk.Plugin.options`).
pub fn forKit(comptime O: type) [all.len]O {
    var out: [all.len]O = undefined;
    for (all, &out) |d, *o| o.* = .{ .name = d.name, .kind = .bool, .default = d.default, .help = d.help };
    return out;
}

test "glm53f options: every switch is a GLM53F_ option of the kit's rules, on by default, with a line of help" {
    const t = std.testing;
    const dy = kit.abi.dylib;
    for (all, 0..) |d, i| {
        try t.expect(dy.optionNameOk("glm53f", d.name));
        try t.expect(dy.valueOk(.bool, d.default));
        try t.expect(d.help.len > 0);
        for (all[0..i]) |e| try t.expect(!std.mem.eql(u8, d.name, e.name));
    }
    // The fastest accepted stack is the default: every switch on.
    for (all) |d| try t.expect(defaultOn(d.sw));
}

test "glm53f options: a value is on unless empty or starting with 0; a forced value wins" {
    const t = std.testing;
    try t.expect(valueOn("1") and valueOn("yes") and !valueOn("0") and !valueOn("") and !valueOn("00"));
    forced[@backingInt(Switch.kda_prompt_chunk)] = true;
    defer forced[@backingInt(Switch.kda_prompt_chunk)] = null;
    try t.expect(on(.kda_prompt_chunk));
}
