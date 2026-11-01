//! The expert usage profile's settings (mlx-serve's `expert.usage`: the resident seed and the record), MLX-free, read at
//! load before the stream opens; the copied EXL3 source (exl3_source.zig, patches/turboexl3-zz4-usage-seed.patch) reads
//! them through this root. The model settings' keys of the in-tree archs (`resident_seed`, `expert_usage_record`,
//! `expert_usage_label`) do not reach an outside plugin's load, so they are environment variables here:
//! - `MLX_SERVE_DSV41_RESIDENT_SEED`: `usage` (the prompt rows filled at the load with each layer's most used experts of
//!   `expert-usage-v1.json`; refused without the file) or `none`. Unset: by the pack, `usage` when it has the file (as
//!   glm5_next).
//! - `MLX_SERVE_DSV41_EXPERT_USAGE_DIR`: the directory the seed reads `expert-usage-v1.json` from (absolute). Unset: the
//!   pack's.
//! - `MLX_SERVE_DSV41_EXPERT_USAGE_RECORD`: an absolute file path: each request's routed experts per layer counted and
//!   written there at its end (`expert-usage-v1.json`'s form). Unset: no record.
//! - `MLX_SERVE_DSV41_EXPERT_USAGE_LABEL`: the recorded prompt set's name (the file's provenance); needs the record.

const std = @import("std");

pub const max_path = 1024;
pub const max_label = 256;

pub const Seed = enum { pack, usage, none };

pub const Setting = struct {
    seed: Seed = .pack,
    dir_buf: [max_path]u8 = undefined,
    dir_len: usize = 0,
    record_buf: [max_path]u8 = undefined,
    record_len: usize = 0,
    label_buf: [max_label]u8 = undefined,
    label_len: usize = 0,

    /// The seed's directory: the setting's, else `pack`.
    pub fn dirOr(s: *const Setting, pack: []const u8) []const u8 {
        return if (s.dir_len > 0) s.dir_buf[0..s.dir_len] else pack;
    }

    pub fn recordPath(s: *const Setting) ?[]const u8 {
        return if (s.record_len > 0) s.record_buf[0..s.record_len] else null;
    }

    pub fn label(s: *const Setting) []const u8 {
        return if (s.label_len > 0) s.label_buf[0..s.label_len] else "unlabeled";
    }
};

/// Set at load.
pub var setting: Setting = .{};

pub const Error = error{ ResidentSeedSetting, ExpertUsageDirSetting, ExpertUsageRecordSetting, ExpertUsageLabelSetting, ExpertUsageLabelWithoutRecord };

/// The four values (null: unset), or the first bad one's error (the load refuses it by name).
pub fn parse(seed: ?[]const u8, dir: ?[]const u8, record: ?[]const u8, label: ?[]const u8) Error!Setting {
    var s: Setting = .{};
    if (seed) |v| if (v.len > 0) {
        s.seed = if (std.mem.eql(u8, v, "usage")) .usage else if (std.mem.eql(u8, v, "none")) .none else return error.ResidentSeedSetting;
    };
    if (dir) |v| if (v.len > 0) {
        if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.ExpertUsageDirSetting;
        @memcpy(s.dir_buf[0..v.len], v);
        s.dir_len = v.len;
    };
    if (record) |v| if (v.len > 0) {
        if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.ExpertUsageRecordSetting;
        @memcpy(s.record_buf[0..v.len], v);
        s.record_len = v.len;
    };
    if (label) |v| if (v.len > 0) {
        if (v.len > max_label) return error.ExpertUsageLabelSetting;
        if (s.record_len == 0) return error.ExpertUsageLabelWithoutRecord;
        @memcpy(s.label_buf[0..v.len], v);
        s.label_len = v.len;
    };
    return s;
}

// ── tests ──

const testing = std.testing;

test "expert usage settings: unset is the pack's seed and no record" {
    const s = try parse(null, null, null, null);
    try testing.expectEqual(Seed.pack, s.seed);
    try testing.expectEqualStrings("/p", s.dirOr("/p"));
    try testing.expect(s.recordPath() == null);
    try testing.expectEqualStrings("unlabeled", s.label());
    const e = try parse("", "", "", "");
    try testing.expectEqual(Seed.pack, e.seed);
}

test "expert usage settings: the seed, its directory, the record and its label; bad values refused by name" {
    const s = try parse("usage", "/data/profile", "/tmp/rec/expert-usage-v1.json", "code-slices");
    try testing.expectEqual(Seed.usage, s.seed);
    try testing.expectEqualStrings("/data/profile", s.dirOr("/p"));
    try testing.expectEqualStrings("/tmp/rec/expert-usage-v1.json", s.recordPath().?);
    try testing.expectEqualStrings("code-slices", s.label());
    try testing.expectEqual(Seed.none, (try parse("none", null, null, null)).seed);
    try testing.expectError(error.ResidentSeedSetting, parse("on", null, null, null));
    try testing.expectError(error.ExpertUsageDirSetting, parse(null, "rel/dir", null, null));
    try testing.expectError(error.ExpertUsageRecordSetting, parse(null, null, "rec.json", null));
    try testing.expectError(error.ExpertUsageLabelWithoutRecord, parse(null, null, null, "x"));
    var long: [max_label + 1]u8 = @splat('a');
    try testing.expectError(error.ExpertUsageLabelSetting, parse(null, null, "/r.json", &long));
}
