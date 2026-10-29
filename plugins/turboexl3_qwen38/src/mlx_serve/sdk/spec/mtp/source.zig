//! copied_from: src/sdk/spec/mtp/source.zig@1b3e624b6cf44c6092c51683bd353ad70bd9718d
//! Where a pack keeps its MTP head: a sidecar file beside the trunk, or `mtp.*` tensors in the checkpoint itself.
//! Header and index reads only; no weight loads.

const std = @import("std");
const testing = std.testing;

/// Sidecar file layouts we accept, in priority order. The native layout wins
/// so a repo shipping several keeps loading exactly what it loaded before.
/// Root-level names are what others publish (mutual compat: their
/// loader accepts our `mtp/weights.safetensors` too).
pub const sidecar_rel_paths = [_][]const u8{
    "mtp/weights.safetensors", // mlx-serve native (ddalcu repos, build_mtp_sidecar.py)
    "mtp.safetensors", // others
    "model-mtp.safetensors", // others
    "optiq/mtp.safetensors", // oMLX OptiQ (delta-encoded norms — folded at load)
    "mtp_head.safetensors", // Nemotron-H heads (sevren-ai packs, bare keys)
    "mtp-4bit.safetensors", // Nemotron-H heads (TensorFold's Vontra packs)
};

/// Where a model's MTP head lives.
pub const MtpSource = union(enum) {
    /// A separate sidecar file — rel path, one of `sidecar_rel_paths`.
    sidecar_file: []const u8,
    /// Inside the main checkpoint safetensors (Qwen HF releases and oMLX
    /// oQ4e-class conversions ship `[language_model.]mtp.*` in the trunk
    /// shards). Loading reads ONLY the shards the index names for mtp keys.
    in_checkpoint,
};

/// Marker projections that prove a LOADABLE head: `fc` (Qwen dense/MoE
/// layouts) or `eh_proj` (hy3). Discovery and the shard sweep both gate on
/// this same set, so a checkpoint with stray `mtp.*` auxiliaries but no
/// marker never claims a head it can't bind.
const mtp_marker_keys = [_][]const u8{
    "mtp.fc.weight",
    "language_model.mtp.fc.weight",
    "mtp.eh_proj.weight",
    "language_model.mtp.eh_proj.weight",
    "attn.mixer.q_proj.weight", // Nemotron-H head (bare keys, dense `eh_proj`)
    "layers.0.eh_proj.weight", // Nemotron-H head (TensorFold names)
};

/// Sanity bound for index.json / safetensors headers (the Jundot 27B index
/// is ~212 KB; headers of the largest checkpoints stay well under this).
pub const checkpoint_header_limit: usize = 64 * 1024 * 1024;

/// Qwen3.8-Flash-Next's head is the checkpoint's own layer, loaded by the
/// trunk (`loadQwen4Mtp`), never through `resolveMtpSource`.
const qwen4_mtp_marker = "\"language_model.mtp.fc_hidden.weight\"";
const qwen4_root_mtp_marker = "\"mtp.fc_hidden.weight\"";
