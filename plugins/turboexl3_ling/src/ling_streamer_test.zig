//! turboexl3_ling's MLX-free tests of the copied streamer (`zig build test`, build.zig): its routed calls on the trace
//! backend (src/mlx_serve/sdk/weights/streamed/experts.zig: ROUTED_JOIN's parts and the D2 family's read-ahead, holds,
//! parks and claims), and RQ's decode rows grown and trimmed through its hook (src/ling_rq_stream_test.zig).
test {
    _ = @import("mlx_serve/sdk/weights/streamed/experts.zig");
    _ = @import("ling_rq_stream_test.zig");
}
