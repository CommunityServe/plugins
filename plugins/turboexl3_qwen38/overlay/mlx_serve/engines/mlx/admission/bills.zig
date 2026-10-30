//! This plugin's own, in place of mlx-serve's `src/engines/mlx/admission/bills.zig` (a cut edge of
//! scripts/sync-mlx-serve.sh): the one term of the engine's admission arithmetic the copied bill reads.

const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const qsaHistoryShareEnabled = @import("../../../plugins/qwen4_exp/qsa/gather.zig").qsaHistoryShareEnabled;

/// The QSA history a request holds per token, once when the slots share it and twice when they do not, with the score
/// bank's per-token bytes.
pub fn statePerTokenBilled(config: *const ModelConfig) u64 {
    const one = config.qsaHistoryBytesPerToken();
    if (one == 0) return 0;
    return one * @as(u64, if (qsaHistoryShareEnabled()) 1 else 2) + config.qsaScoreBankBytesPerToken();
}
