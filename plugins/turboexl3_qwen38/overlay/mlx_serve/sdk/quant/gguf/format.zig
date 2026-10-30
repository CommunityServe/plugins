//! This plugin's own, in place of mlx-serve's `src/sdk/quant/gguf/format.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the built-in formats' list names GGUF's table, which serves nothing here (no GGUF weight
//! loads: sdk/quant/gguf/claim.zig). Its claim is the built-ins' own (sdk/quant/block.zig); every call refuses by name.

const abi = @import("../../../abi/quant.zig");

const Str = abi.Str;
const Verdict = abi.Verdict;
const no = Verdict.refuse(.unsupported, "turboexl3_qwen38: GGUF weights are not served");

fn claims(_: Str, _: Str) callconv(.c) abi.Claim {
    return .no;
}
fn validate(_: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    return no;
}
fn bill(_: Str, _: Str, stored: u64) callconv(.c) u64 {
    return stored;
}
fn open(_: *abi.LoadCtx) callconv(.c) Verdict {
    return no;
}
fn close(_: *abi.LoadCtx) callconv(.c) void {}
fn load(_: *const abi.LoadCtx, _: Str, _: u32, _: *abi.Weight) callconv(.c) Verdict {
    return no;
}
fn release(_: *abi.Weight) callconv(.c) void {}
fn matmul(_: *const abi.Weight, _: abi.Array, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return no;
}
fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) Verdict {
    return no;
}

pub const table: abi.QuantTable = .{
    .name = Str.of("gguf"),
    .claims = &claims,
    .validate = &validate,
    .bill = &bill,
    .open = &open,
    .close = &close,
    .load_tensor = &load,
    .load_bank = &load,
    .release = &release,
    .matmul = &matmul,
    .moe = &moe,
};
