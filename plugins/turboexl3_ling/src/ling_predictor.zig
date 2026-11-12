//! turboexl3_ling's read-ahead predictor scores (`Math.predictorScores`, experts.zig `scoresOf`): the next routed layer's
//! gate on this call's rows, in f32, over the streamer's graph ops `G`. Residency only: the scores choose which records
//! are read ahead, never a routed expert. A call marks PREDICTOR on the run-path line (src/ling_ran.zig).
const ran = @import("ling_ran.zig");

/// Ling's router key (`sigmoid_bias_grouped`'s selection score): sigmoid of the f32 gate logits plus the expert bias.
pub fn lingScores(comptime G: type, g: *G, xf: G.T, w: G.T, bias: G.T) !G.T {
    ran.mark(.PREDICTOR);
    const z = try g.astype(try g.matmul(xf, try g.transpose(w)), .float32);
    return g.add(try g.sigmoid(z), bias);
}
