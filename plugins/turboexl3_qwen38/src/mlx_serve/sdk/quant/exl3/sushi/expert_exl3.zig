//! copied_from: src/sdk/quant/exl3/sushi/expert_exl3.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! The EXL3 tile format on the host: rates, codebooks, windows, the trellis unpack and each codebook's decode.
const std = @import("std");
const codebook_tables = @import("codebook_tables.zig");

pub const TILE: usize = 16;
pub const TILE_VALUES: usize = TILE * TILE;
pub const HAD_DIM: usize = 128;
pub const HAD_SCALE: f32 = 0.08838834764831845;
pub const K4: u32 = 4;
pub const K4_PACKED: usize = TILE_VALUES * K4 / 16;
pub const MCG_MULT: u32 = 0xCBAC1FED;
pub const MUL1_MULT: u32 = 0x83DCD12D;

/// A hash-mask codebook: one multiply hashes the codeword, `mask` keeps two tiny (often subnormal)
/// halves, and their half sum is rescaled by `scale`. The decoders and kernels take the pair as a
/// parameter, so another member of the family is a new pair, not a new code path.
pub const HashMask = struct { mult: u32, mask: u32, scale: u32 = hash_mask_scale };
pub const hash_mask_scale: u32 = 4096;

/// PAIR1's pair codec: weights 2j and 2j+1 both decode from the window of weight 2j+1.
/// `p = cw * mult`; weight 2j is the half sum of `p & mask0`, weight 2j+1 that of
/// `rotl(p, rot) & mask1`, each times `scale`. One row per rate, so a bank decodes with its own.
pub const Pair = struct { mult: u32, rot: u5, mask0: u32, mask1: u32, scale: u32 = 4096 };
/// The pack contract's PAIR1 rows, one per even rate n = 24, 26 ... 64; equal to sashimi's `PAIR1_CONSTANTS`.
const PAIR1_ROWS = [_]Pair{
    .{ .mult = 0x58F41BF3, .rot = 3, .mask0 = 0x8ED78C13, .mask1 = 0x8EEE8C03 }, // n24
    .{ .mult = 0x58F41BF3, .rot = 3, .mask0 = 0x8ED78C13, .mask1 = 0x8EEE8C03 }, // n26
    .{ .mult = 0x7AA2BE15, .rot = 3, .mask0 = 0x8ED78C13, .mask1 = 0x8ECE8C03 }, // n28
    .{ .mult = 0x7B229E11, .rot = 3, .mask0 = 0x8ED78C13, .mask1 = 0x8ECE8C03 }, // n30
    .{ .mult = 0x3B22BE15, .rot = 3, .mask0 = 0x8ED78C13, .mask1 = 0x8ECE8C03 }, // n32
    .{ .mult = 0x46C63E17, .rot = 19, .mask0 = 0x8F9A8E6E, .mask1 = 0x8E258FFE }, // n34
    .{ .mult = 0x46C63E17, .rot = 19, .mask0 = 0x8F9A8E6E, .mask1 = 0x8E258FFE }, // n36
    .{ .mult = 0x46C63E17, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n38
    .{ .mult = 0x46C63E17, .rot = 19, .mask0 = 0x8F9A8E6E, .mask1 = 0x8E258FFE }, // n40
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n42
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n44
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n46
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n48
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n50
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n52
    .{ .mult = 0xF75D60FF, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FFE }, // n54
    .{ .mult = 0x82439F67, .rot = 19, .mask0 = 0x8E6E8F9A, .mask1 = 0x8E258FAE }, // n56
    .{ .mult = 0xE2C7FC0D, .rot = 19, .mask0 = 0x8F408FB0, .mask1 = 0x8F918F89 }, // n58
    .{ .mult = 0xE2C7FC0D, .rot = 19, .mask0 = 0x8F408FA0, .mask1 = 0x8F118FC9 }, // n60
    .{ .mult = 0xE2C7FC0D, .rot = 19, .mask0 = 0x8F408FA0, .mask1 = 0x8F118FC9 }, // n62
    .{ .mult = 0xE2C7FC0D, .rot = 19, .mask0 = 0x8F408FA0, .mask1 = 0x8F118FC9 }, // n64
};

/// The pack contract's PAIR1 row for `rate`; null refuses an odd or out-of-range rate.
pub fn pair1Pair(rate: Rate) ?Pair {
    if (rate.n % 2 != 0 or rate.n < 24 or rate.n > 64) return null;
    return PAIR1_ROWS[(rate.n - 24) / 2];
}

pub const Codebook = enum(u8) {
    mul1 = 0,
    mcg = 1,
    spec1 = 2,
    pair1 = 3,
    std1 = 4,

    pub const count = 5;

    pub fn fromName(name: []const u8) ?Codebook {
        if (std.mem.eql(u8, name, "mcg")) return .mcg;
        if (std.mem.eql(u8, name, "mul1")) return .mul1;
        if (std.mem.eql(u8, name, "std1")) return .std1;
        if (std.mem.eql(u8, name, "spec1")) return .spec1;
        if (std.mem.eql(u8, name, "pair1")) return .pair1;
        return null;
    }

    /// Whether the codebook decodes through the (multiplier, mask) row of its table: STD1 and SPEC1. A bank decodes
    /// with `Decode.hashMask`.
    pub fn tabledHashMask(self: Codebook) bool {
        return self == .spec1 or self == .std1;
    }

    /// Whether a pack of this codebook names a codebook table (STQ's codebooks).
    pub fn hasTable(self: Codebook) bool {
        return self.tabledHashMask() or self == .pair1;
    }

    /// The codebooks whose bytes GLM's lane, one-row, group2 and prefill-grid kernels are proven on.
    pub fn servesGlmFastPaths(self: Codebook) bool {
        return switch (self) {
            .mcg, .spec1, .std1, .pair1 => true,
            .mul1 => false,
        };
    }
};

/// The codeword width a pack's Viterbi search hashed: a decoder takes the low
/// `w` bits of the 16-bit sliding window before it decodes. The bitstream and
/// the bits per weight are the same at every width — only the value each
/// window decodes to changes. A pack that names no width is w16, which masks
/// nothing. (Unrelated to the GEMM's row windows.)
pub const Window = enum(u8) {
    /// w16 is tag 0, so a zeroed config means "mask nothing" — the pack every
    /// converter wrote before the field existed. The narrowed widths follow in
    /// ascending order from tag 1, which is what `bits` and `fromBits` index.
    w16 = 0,
    w8,
    w9,
    w10,
    w11,
    w12,
    w13,
    w14,
    w15,

    pub const min_bits: u8 = 8;
    pub const count = 16 - min_bits + 1;

    pub fn bits(self: Window) u8 {
        if (self == .w16) return 16;
        return @intFromEnum(self) + min_bits - 1;
    }

    pub fn mask(self: Window) u16 {
        return @truncate((@as(u32, 1) << @intCast(self.bits())) - 1);
    }

    pub fn index(self: Window) usize {
        return @intFromEnum(self);
    }

    pub fn fromBits(w: i64) ?Window {
        if (w == 16) return .w16;
        if (w < min_bits or w > 15) return null;
        return @enumFromInt(w - min_bits + 1);
    }
};

/// How a pack's trellis decodes. The two travel together — a codeword read
/// under the wrong window is as wrong as one read under the wrong codebook.
pub const Decode = struct {
    codebook: Codebook,
    window: Window = .w16,
    /// The pack's codebook table: an index into `codebook_tables.tables`, one row per rate. STD1 and SPEC1 always have
    /// one, a decode without it has no pair; PAIR1's is informational, every other codebook has none.
    table: ?u8 = null,

    pub const mul1: Decode = .{ .codebook = .mul1 };
    pub const mcg: Decode = .{ .codebook = .mcg };

    /// A table-keyed codebook (STD1, SPEC1) has no pair without its table; a dispatch refuses it by name.
    pub fn requireTable(self: Decode) error{Exl3CodebookTableUnknown}!void {
        if (self.codebook.tabledHashMask() and self.table == null) return error.Exl3CodebookTableUnknown;
    }

};

/// A trellis rate K = n/16 bits per weight, `n` being the halfwords a packed
/// 256-weight tile carries. Weight t's codeword is the 16-bit window ending at
/// `bitEnd(t)`, so it takes `bitEnd(t) - bitEnd(t-1)` fresh bits; the pattern
/// follows from n and is never stored. An integer K is the uniform case.
pub const Rate = struct {
    n: u32,

    pub const min_n: u32 = 16;
    pub const max_n: u32 = 128;

    /// Whether a set of rates (bit n/2 per rate) holds this one; null is every rate.
    pub fn in(self: Rate, set: ?u128) bool {
        const rates = set orelse return true;
        return (rates >> @intCast(self.n / 2)) & 1 == 1;
    }

};

/// K1 to K8 in 1/8-bit steps. An odd n would leave a tile's bitstream short of
/// a whole uint32 word, which every reader indexes by.
pub fn kFromPackedDim(packed_hw: usize) ?Rate {
    if (packed_hw % 2 != 0) return null;
    if (packed_hw < Rate.min_n or packed_hw > Rate.max_n) return null;
    return .{ .n = @intCast(packed_hw) };
}

/// `@embedFile` is byte-aligned and the linker lands it wherever it likes, so a
/// fixture read as u16 through a `[]const u8` is a coin flip on the blob's
/// address. These copies carry the alignment every reader assumes.
pub const fixtures = struct {
    pub const k4 = aligned(@embedFile("fixtures/exl3_k4_linear.safetensors"));
    pub const k3 = aligned(@embedFile("fixtures/exl3_k3_linear.safetensors"));
    pub const k2 = aligned(@embedFile("fixtures/exl3_k2_linear.safetensors"));
    pub const k2p5_mcg = aligned(@embedFile("fixtures/exl3_k2p5_mcg_linear.safetensors"));
    pub const k3_mcg = aligned(@embedFile("fixtures/exl3_k3_mcg_linear.safetensors"));
    /// Searched AND decoded at window 12 by sashimi: the only fixture that
    /// certifies a narrowed window against the library rather than against our
    /// own masking of a w16 bitstream.
    pub const k2p5_mcg_w12 = aligned(@embedFile("fixtures/exl3_k2p5_mcg_w12_linear.safetensors"));
    /// Random packed PAIR1 tiles at every even rate K1.5 to K4 (`trellis_n<n>`) and sashimi's numpy decode of
    /// each, row-major (`decoded_n<n>`): tests/dump_pair1_fixtures.py.
    pub const pair1_tiles = aligned(@embedFile("fixtures/exl3_pair1_tiles.safetensors"));

    fn aligned(comptime raw: []const u8) *align(8) const [raw.len]u8 {
        const holder = struct {
            const value: [raw.len]u8 align(8) = raw[0..raw.len].*;
        };
        return &holder.value;
    }
};

const fixture_bytes = fixtures.k4;

const TensorView = struct {
    dtype: []const u8,
    shape: []const usize,
    bytes: []const u8,
};

const fixture_k3_bytes = fixtures.k3;
const fixture_k2_bytes = fixtures.k2;

const fixture_k2p5_mcg_bytes = fixtures.k2p5_mcg;
const fixture_k3_mcg_bytes = fixtures.k3_mcg;
