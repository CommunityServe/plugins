//! The expert usage profile's settings (mlx-serve's `expert.usage`: the resident seed), MLX-free, read at load before
//! the stream opens; the EXL3 source (exl3_source.zig) reads them through this root. The plugin's options:
//! - `TURBOEXL3_DS41_RESIDENT_SEED`: `usage` (the prompt rows filled at the load with each layer's most used experts of
//!   `expert-usage-v1.json`; refused without the file) or `none`. Unset: by the pack, `usage` when it has the file (as
//!   glm5_next).
//! - `TURBOEXL3_DS41_EXPERT_USAGE_DIR`: the directory the seed reads `expert-usage-v1.json` from (absolute). Unset: the
//!   pack's.

const std = @import("std");

pub const max_path = 1024;

pub const Seed = enum { pack, usage, none };

pub const Setting = struct {
    seed: Seed = .pack,
    dir_buf: [max_path]u8 = undefined,
    dir_len: usize = 0,

    /// The seed's directory: the setting's, else `pack`.
    pub fn dirOr(s: *const Setting, pack: []const u8) []const u8 {
        return if (s.dir_len > 0) s.dir_buf[0..s.dir_len] else pack;
    }
};

/// Set at load.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    setting: Setting = .{},
};
pub const shared_state = @import("sdk").shared.Shared(State, "dsv41_expert_usage", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

pub const Error = error{ ResidentSeedSetting, ExpertUsageDirSetting };

/// The two values (null: unset), or the first bad one's error (the load refuses it by name).
pub fn parse(seed: ?[]const u8, dir: ?[]const u8) Error!Setting {
    var s: Setting = .{};
    if (seed) |v| if (v.len > 0) {
        s.seed = if (std.mem.eql(u8, v, "usage")) .usage else if (std.mem.eql(u8, v, "none")) .none else return error.ResidentSeedSetting;
    };
    if (dir) |v| if (v.len > 0) {
        if (v.len > max_path or !std.fs.path.isAbsolute(v)) return error.ExpertUsageDirSetting;
        @memcpy(s.dir_buf[0..v.len], v);
        s.dir_len = v.len;
    };
    return s;
}
