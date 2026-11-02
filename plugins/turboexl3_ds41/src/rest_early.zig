//! `MLX_SERVE_DSV41_REST_EARLY` (read at load; unset, empty or `1`: on; `0`: off; anything else is refused by name): a
//! wide prompt layer's new streams (its routed call's tail kernel, the TAIL_FUSE combine and hc expand, with its inputs'
//! small ops) committed at the call's end, before the layer's teardown, not by the layer's last eval after it.
//!
//! Off, the copied model's `wideLayer` returns after its last group's holds, its group scope and its layer scope free
//! every intermediate the layer made (thousands of handles, the attention's GB-size transients among them: MLX's 2 GiB
//! cache takes few, the rest go back to Metal), and only then does `runLayers` evaluate the layer's held arrays, the new
//! streams first among them: the GPU idles from the call's last wave through the teardown, the eval's walk and the new
//! streams' allocation, about 20.7 ms a 16K-row encoder layer (pt4 w-pio-20261102-0613's pk arm: 5.8 ms from the call's
//! end to the eval's first commit, 3.5 ms between its first two small kernels, 11.5 ms for the new streams' fresh
//! allocation), then runs the tail kernel (4.8 ms).
//!
//! On, the last group commits the new streams (`mlx_async_eval`) right after its holds: the GPU runs the tail kernel while
//! the host tears the layer down, and the layer's last eval finds it under way. The same kernels over the same inputs in
//! the same stream, only committed earlier: the same words (a text gate; there is no second path to compare). Memory: the
//! holds have just freed the previous streams (in a wide layer their only references are the spans' slices and the
//! whole stream, which the holds replace, and the x1 region's release token, replaced just before; the DSpark target
//! means, the one lazy graph that could hold them, are set on decoder layers only), so the new streams take their place
//! in the live set: the peak is the old flow's just before its holds, which the prompt bill covers (two stream sets a row
//! in its `held` term beside the attention's layer-long heads).
//!
//! Measured (re1, w-pio-20261102-0740, on zzz9c against ctl31, G_KITSP1 with keyed sampling; a1 b1 b2 a2, the cold 16K
//! card then the new-text slice 3): cold 1451.8 / 1459.4 / 1446.4 / 1441.3 tok/s (+0.44%, pairs +0.52% / +0.35%), new
//! text 1543.8 / 1543.3 / 1584.0 / 1518.1 (+2.14%, pairs -0.03% / +4.34%); 147 prompt and 176 decode rows in every arm;
//! the guard's accounted peak 106.4 / 100.2 GiB in the b arms against 107.8 / 107.8 in the a arms; the texts the same
//! bytes in every arm.

const std = @import("std");

/// The switch (`configure`, at load).
pub var on: bool = true;

pub const Error = error{RestEarlySetting};

pub fn configure(value: ?[]const u8) Error!void {
    on = try parse(value);
}

pub fn parse(value: ?[]const u8) Error!bool {
    const v = value orelse return true;
    if (v.len == 0 or std.mem.eql(u8, v, "1")) return true;
    if (std.mem.eql(u8, v, "0")) return false;
    return error.RestEarlySetting;
}

test "rest_early setting: on unset, empty or 1; 0 off; anything else refused by name" {
    try std.testing.expect(try parse(null));
    try std.testing.expect(try parse(""));
    try std.testing.expect(try parse("1"));
    try std.testing.expect(!try parse("0"));
    for ([_][]const u8{ "on", "2", "00", "check", "true", " 1" }) |v| try std.testing.expectError(error.RestEarlySetting, parse(v));
}
