//! copied_from: src/engines/mlx/ssm/group.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! engines/mlx/ssm/group.
//! Owns SsmMergeKind, ssm_member_gen_src, nextSsmMemberGen, SsmGroupMember, SsmPatchFields, PersistentSsmGroup, and 42 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const ssmFreeQsaState = @import("../../../plugins/qwen4_exp/qsa/history.zig").ssmFreeQsaState;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const MAX_BATCH_ROWS = @import("../batched/layered.zig").MAX_BATCH_ROWS;
const ForwardCtx = @import("../forward_ctx.zig").ForwardCtx;
const KVCache = @import("../kv/cache.zig").KVCache;
const Transformer = @import("../transformer.zig").Transformer;
const captureSsmCheckpoint = @import("checkpoint.zig").captureSsmCheckpoint;
const restoreSsmCheckpoint = @import("checkpoint.zig").restoreSsmCheckpoint;
const SSMCacheEntry = @import("entry.zig").SSMCacheEntry;
const splitSpecCaptures = @import("entry.zig").splitSpecCaptures;
const ssmFreeSpecCapture = @import("entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("entry.zig").ssmRollbackFromCapture;
const ssmSnapshot = @import("entry.zig").ssmSnapshot;
const ssmSnapshotDeinit = @import("entry.zig").ssmSnapshotDeinit;

pub const SsmMergeKind = packed struct {
    linear: bool = false,
    ple: bool = false,
};

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const SsmGroupState = struct {
    ssm_member_gen_src: std.atomic.Value(u64) = .init(1),
};
pub const ssmGroup_shared = @import("../../../sdk/util/shared.zig").Shared(SsmGroupState, "ssm_group", .{});
inline fn ssmGroup() *SsmGroupState {
    return ssmGroup_shared.get();
}

pub fn nextSsmMemberGen() u64 {
    return ssmGroup().ssm_member_gen_src.fetchAdd(1, .monotonic);
}

const SsmGroupMember = struct {
    key: usize,
    gen: u64,
    entries: []SSMCacheEntry,
};

const SsmPatchFields = packed struct {
    conv: bool = false,
    ssm: bool = false,
    aux: bool = false,
};

/// The batched tick's one owner of the merged GDN/conv/PLE state across ticks.
/// A stable group reads the current generation and writes the next; slots hold
/// (group, row, member_gen), never a slice view. Rows leave through
/// `ssmExportRow` (materialized + evaluated) before any generation is freed, and
/// `entries[].member_gen` is the ONE guard: any write outside the group bumps it
/// and `matches()` then refuses the slot. Retirement is MLX refcount: every
/// consumer holds its own handle.
pub const PersistentSsmGroup = struct {
    allocator: std.mem.Allocator = undefined,
    n: usize = 0,
    keys: []usize = &.{},
    gens: []u64 = &.{},
    member_entries: [][]SSMCacheEntry = &.{},
    layers: []SSMCacheEntry = &.{},
    kinds: []SsmMergeKind = &.{},
    live: bool = false,

    pub fn matches(self: *const PersistentSsmGroup, ctxs: []const *ForwardCtx) bool {
        if (!self.live or self.n != ctxs.len) return false;
        for (ctxs, 0..) |c, i| {
            const entries = c.ssm_entries orelse return false;
            if (self.keys[i] != @intFromPtr(entries.ptr)) return false;
            if (entries.len == 0 or self.gens[i] != entries[0].member_gen) return false;
        }
        return true;
    }

    fn clear(self: *PersistentSsmGroup) void {
        if (!self.live) return;
        ssmUnbindAll(self);
        for (self.layers) |*m| ssmFreeMergedLayer(m);
        self.allocator.free(self.layers);
        self.allocator.free(self.keys);
        self.allocator.free(self.gens);
        self.allocator.free(self.member_entries);
        self.allocator.free(self.kinds);
        self.layers = &.{};
        self.keys = &.{};
        self.gens = &.{};
        self.member_entries = &.{};
        self.kinds = &.{};
        self.n = 0;
        self.live = false;
    }

    pub fn deinit(self: *PersistentSsmGroup) void {
        self.clear();
    }
};

pub const ssmConcatAxis0 = @import("../../../mlx/array.zig").concatAxis0;

fn ssmRowView(s: mlx.mlx_stream, arr: mlx.mlx_array, i: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    if (shape.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(i);
    stop[0] = @intCast(i + 1);
    for (1..shape.len) |d| stop[d] = shape[d];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
    return out;
}

pub fn ssmFreeMergedLayer(m: *SSMCacheEntry) void {
    if (m.conv_state.ctx != null) _ = mlx.mlx_array_free(m.conv_state);
    if (m.ssm_state.ctx != null) _ = mlx.mlx_array_free(m.ssm_state);
    if (m.aux_state.ctx != null) _ = mlx.mlx_array_free(m.aux_state);
    ssmFreeSpecCapture(m);
    m.conv_state = .{ .ctx = null };
    m.ssm_state = .{ .ctx = null };
    m.aux_state = .{ .ctx = null };
    m.initialized = false;
}

pub fn ssmMergeLayer(s: mlx.mlx_stream, ctxs: []const *ForwardCtx, layer: usize, kind: SsmMergeKind) !SSMCacheEntry {
    // The qwen4 PLE layer's dilated-conv window is merged like the SSM state.
    var out: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = kind.linear };
    errdefer ssmFreeMergedLayer(&out);
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    const n = ctxs.len;
    if (kind.linear) {
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].conv_state;
        out.conv_state = try ssmConcatAxis0(s, parts[0..n]);
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].ssm_state;
        out.ssm_state = try ssmConcatAxis0(s, parts[0..n]);
    }
    if (kind.ple) {
        for (ctxs, 0..) |c, i| parts[i] = c.ssm_entries.?[layer].aux_state;
        out.aux_state = try ssmConcatAxis0(s, parts[0..n]);
    }
    return out;
}

pub fn ssmSplitLayer(s: mlx.mlx_stream, m: *const SSMCacheEntry, ctxs: []const *ForwardCtx, layer: usize, kind: SsmMergeKind) !void {
    for (ctxs, 0..) |c, i| {
        var e = &c.ssm_entries.?[layer];
        if (kind.linear) {
            const conv_own = try ssmRowView(s, m.conv_state, i);
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            e.conv_state = conv_own;
            const ssm_own = try ssmRowView(s, m.ssm_state, i);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            e.ssm_state = ssm_own;
            e.initialized = true;
        }
        if (kind.ple) {
            const aux_own = try ssmRowView(s, m.aux_state, i);
            ssmFreeQsaState(e);
            e.aux_state = aux_own;
        }
        try splitSpecCaptures(s, m, e, i);
    }
}

fn ssmMergeAll(alloc: std.mem.Allocator, s: mlx.mlx_stream, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) ![]SSMCacheEntry {
    const merged = try alloc.alloc(SSMCacheEntry, kinds.len);
    var built: usize = 0;
    errdefer {
        for (merged[0..built]) |*m| ssmFreeMergedLayer(m);
        alloc.free(merged);
    }
    for (merged, 0..) |*m, l| {
        m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
        if (!kinds[l].linear and !kinds[l].ple) {
            built = l + 1;
            continue;
        }
        m.* = try ssmMergeLayer(s, ctxs, l, kinds[l]);
        built = l + 1;
    }
    return merged;
}

pub fn ssmSplitAll(s: mlx.mlx_stream, merged: []const SSMCacheEntry, ctxs: []const *ForwardCtx, kinds: []const SsmMergeKind) !void {
    for (merged, 0..) |*m, l| {
        if (!kinds[l].linear and !kinds[l].ple) continue;
        try ssmSplitLayer(s, m, ctxs, l, kinds[l]);
    }
}

pub fn ssmBindSlotRefs(group: *PersistentSsmGroup, ctxs: []const *ForwardCtx) void {
    for (ctxs, 0..) |c, row| {
        const entries = c.ssm_entries orelse continue;
        for (entries, 0..) |*e, l| {
            e.group = group;
            e.group_row = @intCast(row);
            e.group_layer = @intCast(l);
        }
    }
}

fn ssmUnbindEntries(group: *PersistentSsmGroup, entries: []SSMCacheEntry) void {
    for (entries) |*e| {
        if (e.group == group) e.group = null;
    }
}

fn ssmUnbindAll(group: *PersistentSsmGroup) void {
    if (!group.live) return;
    for (group.member_entries) |entries| ssmUnbindEntries(group, entries);
}

fn ssmGroupAdopt(
    group: *PersistentSsmGroup,
    alloc: std.mem.Allocator,
    ctxs: []const *ForwardCtx,
    kinds: []const SsmMergeKind,
    merged: []SSMCacheEntry,
) !void {
    group.clear();
    group.allocator = alloc;
    const n = ctxs.len;
    const keys = try alloc.alloc(usize, n);
    errdefer alloc.free(keys);
    const gens = try alloc.alloc(u64, n);
    errdefer alloc.free(gens);
    const member_entries = try alloc.alloc([]SSMCacheEntry, n);
    errdefer alloc.free(member_entries);
    const kinds_copy = try alloc.dupe(SsmMergeKind, kinds);
    errdefer alloc.free(kinds_copy);
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries.?;
        keys[i] = @intFromPtr(entries.ptr);
        gens[i] = if (entries.len > 0) entries[0].member_gen else c.ssm_member_gen;
        member_entries[i] = entries;
        c.ssm_member_gen = gens[i];
        for (entries) |*e| e.member_gen = gens[i];
    }
    group.keys = keys;
    group.gens = gens;
    group.member_entries = member_entries;
    group.kinds = kinds_copy;
    group.layers = merged;
    group.n = n;
    group.live = true;
    for (ctxs) |c| {
        if (c.ssm_entries) |entries| {
            for (entries) |*e| e.member_gen = c.ssm_member_gen;
        }
    }
    ssmBindSlotRefs(group, ctxs);
}

pub fn ssmBumpMemberGenEntries(entries: []SSMCacheEntry) void {
    const g = nextSsmMemberGen();
    for (entries) |*e| e.member_gen = g;
}

pub fn ssmEvalHandles(arrs: []const mlx.mlx_array) void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var count: usize = 0;
    for (arrs) |a| {
        if (a.ctx == null) continue;
        _ = mlx.mlx_vector_array_append_value(vec, a);
        count += 1;
    }
    if (count > 0) _ = mlx.mlx_eval(vec);
}

/// Hands a row back to its slot as an owned, evaluated array and bumps the
/// slot's `member_gen`, so a later failed repack can never re-adopt it.
fn ssmExportRow(s: mlx.mlx_stream, group: *PersistentSsmGroup, row: usize) !void {
    const entries = group.member_entries[row];
    if (entries.len > 0 and entries[0].group == null) return;
    var exported: [3 * 256]mlx.mlx_array = undefined;
    if (group.layers.len * 3 > exported.len) return error.SsmGroupLayerCount;
    var n_exp: usize = 0;
    for (group.layers, 0..) |*m, l| {
        var e = &entries[l];
        const kind = group.kinds[l];
        if (kind.linear and m.conv_state.ctx != null) {
            const view = try ssmRowView(s, m.conv_state, row);
            defer _ = mlx.mlx_array_free(view);
            const owned = try materializedOwnedCopy(s, view);
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            e.conv_state = owned;
            exported[n_exp] = owned;
            n_exp += 1;
            const view_s = try ssmRowView(s, m.ssm_state, row);
            defer _ = mlx.mlx_array_free(view_s);
            const owned_s = try materializedOwnedCopy(s, view_s);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            e.ssm_state = owned_s;
            e.initialized = true;
            exported[n_exp] = owned_s;
            n_exp += 1;
        }
        if (kind.ple and m.aux_state.ctx != null) {
            const view = try ssmRowView(s, m.aux_state, row);
            defer _ = mlx.mlx_array_free(view);
            const owned = try materializedOwnedCopy(s, view);
            if (e.aux_state.ctx != null) _ = mlx.mlx_array_free(e.aux_state);
            e.aux_state = owned;
            exported[n_exp] = owned;
            n_exp += 1;
        }
        e.group = null;
    }
    ssmBumpMemberGenEntries(entries);
    ssmEvalHandles(exported[0..n_exp]);
}

fn ssmFindOldRow(group: *const PersistentSsmGroup, key: usize, gen: u64) ?usize {
    for (group.keys, 0..) |k, i| {
        if (k == key and group.gens[i] == gen) return i;
    }
    return null;
}

fn ssmMembersFromCtxs(ctxs: []const *ForwardCtx, buf: []SsmGroupMember) ![]SsmGroupMember {
    if (ctxs.len > buf.len) return error.SsmGroupTooWide;
    for (ctxs, 0..) |c, i| {
        const entries = c.ssm_entries.?;
        const gen = if (entries.len > 0) entries[0].member_gen else c.ssm_member_gen;
        buf[i] = .{ .key = @intFromPtr(entries.ptr), .gen = gen, .entries = entries };
    }
    return buf[0..ctxs.len];
}

fn ssmSliceUpdateRow(s: mlx.mlx_stream, parent: *mlx.mlx_array, row: usize, update: mlx.mlx_array) !void {
    const sh = mlx.getShape(parent.*);
    if (sh.len > 8) return error.SsmRankTooHigh;
    var start: [8]c_int = @splat(0);
    var stop: [8]c_int = @splat(0);
    const strides: [8]c_int = @splat(1);
    start[0] = @intCast(row);
    stop[0] = @intCast(row + 1);
    for (1..sh.len) |d| stop[d] = sh[d];
    var updated = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(updated);
    try mlx.check(mlx.mlx_slice_update(&updated, parent.*, update, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    _ = mlx.mlx_array_free(parent.*);
    parent.* = updated;
}

/// Builds the next generation for a new membership: survivors keep their rows,
/// joiners are imported, leavers are exported. The old `kinds` is duplicated and
/// the replacement descriptor is allocated before anything old is freed; on a
/// failure every surviving row is exported so no stream reverts to a stale entry.
fn ssmRepackMembers(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    members: []const SsmGroupMember,
    kinds: []const SsmMergeKind,
) !void {
    var i: usize = 0;
    while (i < group.n) : (i += 1) {
        var found = false;
        for (members) |m| {
            if (ssmFindOldRow(group, m.key, m.gen) == i) {
                found = true;
                break;
            }
        }
        if (!found) try ssmExportRow(s, group, i);
    }
    const kinds_local = try alloc.dupe(SsmMergeKind, kinds);
    errdefer alloc.free(kinds_local);
    const new_layers = try alloc.alloc(SSMCacheEntry, kinds_local.len);
    var built: usize = 0;
    errdefer {
        for (new_layers[0..built]) |*m| ssmFreeMergedLayer(m);
        alloc.free(new_layers);
    }
    const n = members.len;
    for (new_layers, 0..) |*m, l| {
        m.* = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = kinds_local[l].linear };
        if (!kinds_local[l].linear and !kinds_local[l].ple) {
            built = l + 1;
            continue;
        }
        var parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        // conv + ssm + aux per row.
        var views: [3 * MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var n_views: usize = 0;
        defer {
            var v: usize = 0;
            while (v < n_views) : (v += 1) _ = mlx.mlx_array_free(views[v]);
        }
        if (kinds_local[l].linear) {
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].conv_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].conv_state;
                }
            }
            m.conv_state = try ssmConcatAxis0(s, parts[0..n]);
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].ssm_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].ssm_state;
                }
            }
            m.ssm_state = try ssmConcatAxis0(s, parts[0..n]);
        }
        if (kinds_local[l].ple) {
            for (members, 0..) |mem, j| {
                if (ssmFindOldRow(group, mem.key, mem.gen)) |old| {
                    const view = try ssmRowView(s, group.layers[l].aux_state, old);
                    views[n_views] = view;
                    n_views += 1;
                    parts[j] = view;
                } else {
                    parts[j] = mem.entries[l].aux_state;
                }
            }
            m.aux_state = try ssmConcatAxis0(s, parts[0..n]);
        }
        built = l + 1;
    }
    const new_keys = try alloc.alloc(usize, n);
    errdefer alloc.free(new_keys);
    const new_gens = try alloc.alloc(u64, n);
    errdefer alloc.free(new_gens);
    const new_entries = try alloc.alloc([]SSMCacheEntry, n);
    errdefer alloc.free(new_entries);
    for (members, 0..) |mem, j| {
        new_keys[j] = mem.key;
        new_gens[j] = mem.gen;
        new_entries[j] = mem.entries;
    }
    const old_layers = group.layers;
    const old_keys = group.keys;
    const old_gens = group.gens;
    const old_member_entries = group.member_entries;
    const old_kinds = group.kinds;
    ssmUnbindAll(group);
    group.layers = new_layers;
    group.keys = new_keys;
    group.gens = new_gens;
    group.member_entries = new_entries;
    group.kinds = kinds_local;
    group.n = n;
    group.live = true;
    for (old_layers) |*m| ssmFreeMergedLayer(m);
    alloc.free(old_layers);
    alloc.free(old_keys);
    alloc.free(old_gens);
    alloc.free(old_member_entries);
    alloc.free(old_kinds);
    for (members, 0..) |mem, row| {
        for (mem.entries, 0..) |*e, l| {
            e.group = group;
            e.group_row = @intCast(row);
            e.group_layer = @intCast(l);
            e.member_gen = mem.gen;
        }
    }
}

pub fn ssmGroupDropMember(group: *PersistentSsmGroup, s: mlx.mlx_stream, entries: []SSMCacheEntry) !void {
    if (!group.live) return;
    const key = @intFromPtr(entries.ptr);
    var row: ?usize = null;
    for (group.keys, 0..) |k, i| {
        if (k == key) {
            row = i;
            break;
        }
    }
    const r = row orelse return;
    ssmUnbindEntries(group, entries);
    if (group.n == 1) {
        group.clear();
        return;
    }
    var keep: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    var n_keep: usize = 0;
    for (0..group.n) |i| {
        if (i == r) continue;
        keep[n_keep] = .{ .key = group.keys[i], .gen = group.gens[i], .entries = group.member_entries[i] };
        n_keep += 1;
    }
    ssmRepackMembers(group.allocator, s, group, keep[0..n_keep], group.kinds) catch |err| {
        var export_err: ?anyerror = null;
        for (0..group.n) |i| {
            if (i == r) continue;
            ssmExportRow(s, group, i) catch |e| {
                export_err = e;
            };
        }
        ssmUnbindAll(group);
        group.clear();
        if (export_err) |e| return e;
        return err;
    };
}

pub fn ssmGroupReleaseCtx(group: *PersistentSsmGroup, s: mlx.mlx_stream, ctx: *ForwardCtx) !void {
    const entries = ctx.ssm_entries orelse return;
    if (!group.live) return;
    const key = @intFromPtr(entries.ptr);
    var row: ?usize = null;
    for (group.keys, 0..) |k, i| {
        if (k == key) {
            row = i;
            break;
        }
    }
    const r = row orelse return;
    if (group.n == 1) {
        try ssmExportRow(s, group, r);
        if (entries.len > 0) ctx.ssm_member_gen = entries[0].member_gen;
        group.clear();
        return;
    }
    var keep: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
    var n_keep: usize = 0;
    for (0..group.n) |i| {
        if (i == r) continue;
        keep[n_keep] = .{ .key = group.keys[i], .gen = group.gens[i], .entries = group.member_entries[i] };
        n_keep += 1;
    }
    try ssmRepackMembers(group.allocator, s, group, keep[0..n_keep], group.kinds);
    if (entries.len > 0) ctx.ssm_member_gen = entries[0].member_gen;
}

/// The tick's bind: a stable membership returns the group's layers with no concat,
/// no slice and no allocation; a changed membership repacks once.
pub fn ssmBindTickState(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    ctxs: []const *ForwardCtx,
    kinds: []const SsmMergeKind,
    persist: bool,
) !struct { merged: []SSMCacheEntry, owned: bool } {
    if (!persist) {
        return .{ .merged = try ssmMergeAll(alloc, s, ctxs, kinds), .owned = true };
    }
    if (group.live and group.matches(ctxs) and group.kinds.len == kinds.len) {
        return .{ .merged = group.layers, .owned = false };
    }
    if (group.live) {
        var buf: [MAX_BATCH_ROWS]SsmGroupMember = undefined;
        try ssmRepackMembers(alloc, s, group, try ssmMembersFromCtxs(ctxs, &buf), kinds);
        return .{ .merged = group.layers, .owned = false };
    }
    const merged = try ssmMergeAll(alloc, s, ctxs, kinds);
    errdefer {
        for (merged) |*m| ssmFreeMergedLayer(m);
        alloc.free(merged);
    }
    try ssmGroupAdopt(group, alloc, ctxs, kinds, merged);
    return .{ .merged = group.layers, .owned = false };
}

pub fn ssmPatchGroupRow(
    s: mlx.mlx_stream,
    group: *PersistentSsmGroup,
    row: usize,
    layer: usize,
    conv: mlx.mlx_array,
    ssm: mlx.mlx_array,
    aux: mlx.mlx_array,
    fields: SsmPatchFields,
) !void {
    if (!group.live or layer >= group.layers.len) return;
    const m = &group.layers[layer];
    if (fields.conv and conv.ctx != null and m.conv_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.conv_state, row, conv);
    }
    if (fields.ssm and ssm.ctx != null and m.ssm_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.ssm_state, row, ssm);
    }
    if (fields.aux and aux.ctx != null and m.aux_state.ctx != null) {
        try ssmSliceUpdateRow(s, &m.aux_state, row, aux);
    }
}

pub fn ssmMaterializeLive(
    s: mlx.mlx_stream,
    src: *const SSMCacheEntry,
    arr: mlx.mlx_array,
) !mlx.mlx_array {
    if (arr.ctx == null) return mlx.mlx_array_new();
    if (src.group) |g| {
        if (g.live) {
            const view = try ssmRowView(s, arr, src.group_row);
            defer _ = mlx.mlx_array_free(view);
            return materializedOwnedCopy(s, view);
        }
    }
    return materializedOwnedCopy(s, arr);
}

pub fn ssmLiveAux(e: *const SSMCacheEntry) mlx.mlx_array {
    if (e.group) |g| {
        if (g.live and e.group_layer < g.layers.len) {
            const a = g.layers[e.group_layer].aux_state;
            if (a.ctx != null) return a;
        }
    }
    return e.aux_state;
}

// ── tests ──

const SsmTickSlot = struct {
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,

    fn deinit(self: *SsmTickSlot, alloc: std.mem.Allocator) void {
        for (self.entries) |*e| {
            if (e.conv_state.ctx != null) _ = mlx.mlx_array_free(e.conv_state);
            if (e.ssm_state.ctx != null) _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
            ssmFreeSpecCapture(e);
        }
        alloc.free(self.entries);
    }
};
