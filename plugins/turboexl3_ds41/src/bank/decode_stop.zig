//! The decode phase's budget takes part of the stop under the ceiling back for its slot rows, MLX's cache capped at
//! decode instead (read by the bill and source).
//!
//! The bank's bill keeps `stream_host_bytes` unplanned beside the host's wired-limit margin: a 2.0e9 B stop under the
//! ceiling and a 1 GiB host side, in both phases. The stop is there for what MLX holds beyond the bill (its buffer
//! cache, a phase's transients). At decode it holds little: MLX's cache was 97 to 109 MiB and its peak 51 MiB over its
//! active bytes at 16K; at a 32K context the cache 0 and the peak again 51 MiB over active, the footprint 3.51 GiB under
//! the 102,400 MiB ceiling. So the decode budget takes `cut_bytes` of the stop back (three more decode slot rows at 16K
//! and at 32K on the M5 Max: 470 MiB a row), and MLX's cache is capped at `cache_bytes` from the phase change to decode
//! (after the grow), the cap in effect before restored at the change back to a prompt (before decode's rows are let go);
//! the prompt's budget, rows and cache are as before. The 2 GiB wired margin is untouched.
//!
//! The cap is 512 MiB, twice decode's own early use of 235-252 MiB: a prompt's last chunk lets the attention ring go
//! before the phase change (the model), so its synchronize and clearCache take the ring's buffers and decode's
//! cache holds decode's own. Exact: a route's bytes are the records' in any row; only which rows hold them moves.

const std = @import("std");
const mlx = @import("mlx_serve").mlx;

/// MLX's cache cap at decode.
pub const cache_bytes: usize = 512 << 20;
/// The stop's bytes the decode budget takes back (the bill's `decode_stop_cut`): 1.0e9 B of the stop, and the 512 MiB
/// a cap of 512 MiB, not 1 GiB, frees.
pub const cut_bytes: u64 = 1_000_000_000 + ((1 << 30) - cache_bytes);

/// The cap in effect before decode's (restored at the change back to a prompt); null outside decode.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    prompt_cap: ?usize = null,
};
pub const shared_state = @import("sdk").shared.Shared(State, "dsv41_decode_stop", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// The phase change to decode, after the grow: MLX's cache capped at `cache_bytes` (never raised: a tighter cap stays),
/// the cap before it kept for the change back.
pub fn enterDecode() void {
    if (ps().prompt_cap != null) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, cache_bytes);
    if (prev < cache_bytes) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
    }
    ps().prompt_cap = prev;
}

/// The phase change back to a prompt, before decode's rows are let go: the cap before decode's restored.
pub fn enterPrompt() void {
    const prev = ps().prompt_cap orelse return;
    var tmp: usize = 0;
    _ = mlx.mlx_set_cache_limit(&tmp, prev);
    ps().prompt_cap = null;
}
