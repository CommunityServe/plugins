//! The streamer's own math as src/routed_t3.zig `Math` reads it (`Stock`): the decode lane's bounds and phase routes,
//! the lookahead's predictor (MiMo's, `mimo_math.Predictor`) and the construction's argument types. turboexl3 runs the
//! projections.

const sdk = @import("sdk");
const mimo_math = @import("../mimo/predictor.zig");
const xf = @import("exl3_forms.zig");

/// A prompt call's tokens at most (the module's call rows).
const max_call_tokens: u32 = 16_384;
/// A prompt call's routed rows at most: its tokens x top-8.
pub const max_call_rows: u32 = max_call_tokens * 8;

/// The order of a prompt call's rows by slot (the construction's route): the stable block sort (the control) or the
/// stable counting sort (`prompt_route_cut: on`); both give the same permutation.
pub const Pack = enum { sort, count };

pub const Math = struct {
    /// MiMo's decode-lane bounds and phase routes (mimo_math `MathOf`): the decode lane takes at most 4 rows, the wide
    /// lane from 5; the wide waves are event-gated; the predictor runs in decode only.
    pub const decode_forward_rows: u32 = mimo_math.Math.decode_forward_rows;
    pub const prefill_min_tokens: u32 = decode_forward_rows + 1;
    pub const wide_event_gates = true;
    pub const predictor_decode_only = true;
    pub const Profile = mimo_math.Profile;
    /// The construction's argument types: the decode forms, the rows' order, the refusal message.
    pub const Forms = xf.Forms;
    pub const PackOf = Pack;
    pub const Diag = sdk.Diag;

    pub fn predictorScores(g: anytype, x: @TypeOf(g.*).T, router: @TypeOf(g.*).T, correction: @TypeOf(g.*).T) !@TypeOf(g.*).T {
        return mimo_math.Math.predictorScores(g, x, router, correction);
    }

    predictor: mimo_math.Predictor = .{},
};
