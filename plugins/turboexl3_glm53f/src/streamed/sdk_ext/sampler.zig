//! mlx-stream's serial sampler: the plugin's token for every row it hands the host. Every arch binds one per module.
//! The host arms a clean request through `draft_lane.arm` before its prompt pass (`arm`). The prompt pass takes the
//! binding (`begin`): a request the host did not arm (logprobs, grammar, penalties, `--no-mtp`) is unbound and keeps
//! the host's sampler.
//!
//! For a bound sampled request every logits row the arch returns becomes the one-hot of the token `acceptance` draws
//! at that position (`chosen`): the prompt's last row (the first token) and each serial step's. That token is
//! `acceptance.sample`: the row's tempered, filtered distribution, drawn at `drawAt(seed, position, token_draw)`. A
//! lane's verify emits the same token at the same position (`sampledRows`, sample-then-verify), so a seeded request
//! yields the same ids serial or drafted, on every arch. The host's own sampler then returns the plugin's token
//! whatever its stream. A greedy request keeps its logits (the host's argmax is the target's).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const acc = @import("acceptance.zig");

const testing = std.testing;

/// Seeded rows: each a leader over a spread of logits.
fn fixture(seed: u64, comptime w: usize, comptime v: usize, lg: *[w * v]f32) void {
    var rng: std.Random.DefaultPrng = .init(seed);
    for (lg) |*x| x.* = rng.random().floatNorm(f32) * 1.2;
    for (0..w) |r| lg[r * v + rng.random().uintLessThan(u32, @as(u32, v))] += 2.0;
}
