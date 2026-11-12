//! The pack's non-expert tensors as this plugin reads them (`LoadCtx.weights`: the engine loads none, keep_key refuses
//! every key). The engine's loader narrows every 1-D f16 tensor to bf16, and an EXL3 projection's suh and svh are 1-D
//! f16 that turboexl3 reads as stored: so the plugin opens the shards `model.safetensors.index.json` names itself, as
//! stored, and hands the copied trunk's load a `Source` over them; the MTP head reads the NextN layer's tensors (past the
//! trunk) from the same `Source` (the copied trunk.zig `lingHead`). Lazy until the trunk's load evaluates what it keeps;
//! `deinit` after the load drops the rest.

const std = @import("std");
const kit = @import("mlx_serve");
const mlx = kit.mlx;
const quant_abi = kit.abi.quant;

pub const Weights = struct {
    a: std.mem.Allocator,
    map: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty,
    src: quant_abi.Source,

    /// The shards in `dir`'s index, every tensor but those whose key starts with `skip`: the served load passes "\x00",
    /// which no key starts with (the MTP head reads the NextN layer); the GPU tests pass the NextN layer's prefix.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, skip: []const u8, s: mlx.mlx_stream) !*Weights {
        const self = try a.create(Weights);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .src = .{ .ctx = self, .get = &get } };
        errdefer self.deinit();
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const aa = arena.allocator();
        const index_path = try std.fs.path.join(aa, &.{ dir, "model.safetensors.index.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, index_path, aa, .limited(256 << 20));
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, aa, text, .{});
        if (doc != .object) return error.LingIndexSyntax;
        const wm = doc.object.get("weight_map") orelse return error.LingIndexSyntax;
        if (wm != .object) return error.LingIndexSyntax;
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = wm.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.LingIndexSyntax;
            const f = e.value_ptr.string;
            if (std.mem.indexOfScalar(u8, f, '/') != null or std.mem.indexOf(u8, f, "..") != null) return error.LingIndexSyntax;
            if (std.mem.startsWith(u8, e.key_ptr.*, skip)) continue;
            try files.put(aa, f, {});
        }
        _ = s;
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        for (files.keys()) |f| {
            const path = try std.fs.path.joinZ(aa, &.{ dir, f });
            var arrays = mlx.mlx_map_string_to_array_new();
            defer _ = mlx.mlx_map_string_to_array_free(arrays);
            var meta = mlx.mlx_map_string_to_string_new();
            defer _ = mlx.mlx_map_string_to_string_free(meta);
            // MLX loads on the CPU stream (its Load has no GPU kernel); the arrays are unified memory either way.
            try mlx.check(mlx.mlx_load_safetensors(&arrays, &meta, path.ptr, cpu));
            const iter = mlx.mlx_map_string_to_array_iterator_new(arrays);
            defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
            while (true) {
                var key: ?[*:0]const u8 = null;
                var v = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(v);
                if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, iter) != 0) break;
                const k = std.mem.span(key orelse break);
                if (std.mem.startsWith(u8, k, skip)) continue;
                var h = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(h);
                try mlx.check(mlx.mlx_array_set(&h, v));
                const owned = try a.dupe(u8, k);
                errdefer a.free(owned);
                const gop = try self.map.getOrPut(a, owned);
                if (gop.found_existing) return error.LingIndexDuplicateTensor;
                gop.value_ptr.* = h;
            }
        }
        return self;
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.a.free(e.key_ptr.*);
        }
        self.map.deinit(self.a);
        self.a.destroy(self);
    }

    pub fn count(self: *const Weights) usize {
        return self.map.count();
    }

    fn get(ctx: ?*anyopaque, name: quant_abi.Str) callconv(.c) quant_abi.Array {
        const self: *Weights = @ptrCast(@alignCast(ctx.?));
        const v = self.map.get(name.slice()) orelse return .{};
        return .{ .ctx = v.ctx };
    }
};
