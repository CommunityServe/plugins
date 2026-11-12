//! turboexl3_ling: the MTP head alone on the latent cache (MTP_HEAD_LATENT, S35). The head's one MLA layer caches its
//! normed latent and roped shared key (1,152 bytes a token instead of 40,960) and attends in the latent at its
//! decode-width calls. Those calls run turboexl3's fused launches where turboexl3's plan takes them
//! (`ling.mlaLatentDecode`, src/ling_mla_fused.zig), else the composed latent chain (src/ling_mla_latent.zig `decode`).
//! Both read kv_b's stored rows, which the loader holds transposed (`kvbOrient`). A prompt call's history only appends
//! to the cache. The trunk's MLA layers keep their own (rope-once) cache. Under the exact verify the head only proposes
//! drafts, so the served words are the trunk's: the head's cache moves acceptance and speed, never the text. The copied
//! mla.zig's `latentAttn` marks MTP_HEAD_LATENT's run-path bit at the head's decode-width calls. MLX-free.
const std = @import("std");

/// The head's MLA weights (src/ling_mtp.zig `load`, by address: the copied trunk's `LingHead.mla`); null while the head
/// is not served.
pub var head: ?*const anyopaque = null;

/// Whether `fa` is the head's layer on the latent cache: the copied mla.zig's `mlaAttn` attends on the latent cache for
/// that layer alone.
pub fn isHead(fa: *const anyopaque) bool {
    return if (head) |h| h == fa else false;
}

/// How an MLA layer's plain kv_b array holds its stored rows `[rows, kv_lora]` (rows = H (nope + v)).
pub const KvbOrient = enum {
    /// The array is the rows.
    rows,
    /// The array is the loader's `[kv_lora, rows]` view: a dense projection held as the `[in, out]` view that a plain
    /// matmul reads (the copied trunk.zig `proj`, `transposed`).
    loader_view,
};

/// `shape`'s orientation of kv_b's stored rows; null for any other shape, and where rows = kv_lora would leave it open.
pub fn kvbOrient(shape: []const c_int, rows: c_int, kv_lora: c_int) ?KvbOrient {
    if (shape.len != 2 or rows == kv_lora) return null;
    if (shape[0] == kv_lora and shape[1] == rows) return .loader_view;
    if (shape[0] == rows and shape[1] == kv_lora) return .rows;
    return null;
}

const testing = std.testing;

test "the head's layer alone takes the latent cache; the trunk's layers keep theirs" {
    const saved = head;
    defer head = saved;
    var trunk_layer: u8 = 0;
    var head_layer: u8 = 0;
    head = null;
    try testing.expect(!isHead(&head_layer));
    try testing.expect(!isHead(&trunk_layer));
    head = &head_layer;
    try testing.expect(isHead(&head_layer));
    try testing.expect(!isHead(&trunk_layer));
}

test "kv_b's stored rows: the loader's [kv_lora, H (nope + v)] view is read transposed, the rows' own shape as it is, any other shape refused" {
    // Ling-3.1-flash: 64 heads x (128 + 128) = 16,384 rows of kv_lora 512. The loader holds [512, 16384]: the shape
    // turboexl3's plan refuses (kv_b [H (DN + DV), DL]) and the composed chain's reshape would scramble.
    try testing.expectEqual(@as(?KvbOrient, .loader_view), kvbOrient(&.{ 512, 16384 }, 16384, 512));
    try testing.expectEqual(@as(?KvbOrient, .rows), kvbOrient(&.{ 16384, 512 }, 16384, 512));
    try testing.expectEqual(@as(?KvbOrient, null), kvbOrient(&.{ 512, 8192 }, 16384, 512));
    try testing.expectEqual(@as(?KvbOrient, null), kvbOrient(&.{ 1, 512, 16384 }, 16384, 512));
    try testing.expectEqual(@as(?KvbOrient, null), kvbOrient(&.{ 512, 512 }, 512, 512));
}

test "the trunk's MLA never reaches the latent decode: only ling_mla_fused calls turboexl3's ling.mlaLatentDecode, no trunk MLA module imports it or names MLA_LATENT_NAX's field, and the load's options set that field once" {
    try testing.expect(std.mem.indexOf(u8, @embedFile("ling_mla_fused.zig"), "t3.ling.mlaLatentDecode(") != null);
    // the trunk's MLA modules: the rope-once cache, the exact verify's attention and the flash prompt attention
    const trunk = .{ @embedFile("ling_mla_rope1.zig"), @embedFile("ling_verify_mla.zig"), @embedFile("ling_flash.zig") };
    inline for (trunk) |src| {
        try testing.expect(std.mem.indexOf(u8, src, "mlaLatentDecode") == null);
        try testing.expect(std.mem.indexOf(u8, src, "ling_mla_fused") == null);
        try testing.expect(std.mem.indexOf(u8, src, "mla_latent_nax") == null);
    }
    // the head's weights import it for their load line; the copied mla.zig's latentAttn calls it for the head's layer
    // alone (`isHead`, above)
    try testing.expect(std.mem.indexOf(u8, @embedFile("ling_mtp.zig"), "@import(\"ling_mla_fused.zig\")") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, @embedFile("ling_math.zig"), "c.mla_latent_nax = "));
}
