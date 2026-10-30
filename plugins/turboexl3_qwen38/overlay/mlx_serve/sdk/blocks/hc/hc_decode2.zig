//! copied_from: src/sdk/blocks/hc/hc_decode2.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Hyper-connection read at ONE decode row in two dispatches instead of three. The stream norm
//! folds into the down launch (the same idea as oMLX's two-launch decode, Apache-2.0): the down
//! projection's K slice for stream s is exactly stream s, so a threadgroup normalizes only its
//! own stream and multiplies the slice, with its weight loads issued before the norm finishes.
//! The slice partial sums meet in the up/mix launch, which sums them into the activation once per
//! threadgroup. Rounding sites are the three-kernel path's (xn, silu, up, gate, mix); only the
//! f32 accumulation order differs. On an EXL3 trunk pack (`readTrunk`), from two rows up a threadgroup applies its
//! weight loads to several rows (the multi-row launches), and a dense bf16 read at hc 4 takes decode3's pair; a row's
//! bytes are the one-row launches' in every case.




/// decode3's kernel shape (none changes a byte, they place work): down rows a simdgroup computes in turn, rows per down
/// threadgroup, simdgroups (columns) per up threadgroup, the up weights unpacked once for every row, the weights' loads
/// issued first.
pub const D3_DRP: c_int = 4;
pub const D3_RBD: c_int = 2;
pub const D3_SGU: c_int = 16;
pub const D3_WH: c_int = 1;
pub const D3_LF: c_int = 0;
/// Simdgroups of decode3's down threadgroup: 256 threads, the norm's layout.
pub const D3_SGD: c_int = 8;


