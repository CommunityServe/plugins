//! This plugin's own, in place of mlx-serve's `src/sdk/quant/formats.zig` (a cut edge of scripts/sync-mlx-serve.sh): no
//! built-in format. Every quant block of the packs this plugin serves (the trunk's and the experts') is turboexl3's, the
//! one `extra` table the trunk passes, so no other format is built in.

const abi = @import("../../abi/quant.zig");
const common = @import("mlx_serve").abi.common;

/// `claimant` among the model's own tables `extra` alone: the one claiming highest, null when none claims it, or when
/// two tie at the top (the built-in's tie rule, with no built-in to take the block).
pub fn claimantAmong(extra: []const *const abi.QuantTable, block_json: []const u8, config_json: []const u8) error{Ambiguous}!?*const abi.QuantTable {
    if (extra.len == 0 or extra.len > max_extra) return null;
    var ranks: [max_extra]abi.Claim = undefined;
    for (extra, ranks[0..extra.len]) |t, *c| c.* = t.claims(.of(block_json), .of(config_json));
    const i = (common.pick(ranks[0..extra.len]) catch null) orelse return null;
    return extra[i];
}

/// The extra tables weighed at most.
const max_extra = 64;
