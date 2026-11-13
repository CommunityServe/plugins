//! A plugin's one declaration and its negotiation with the host (`negotiate`, which the conformance suite runs
//! against `sdk.host`); each kind's `of` checks the kind's interface.

const std = @import("std");

pub const Version = struct { major: u16, minor: u16 };

/// The `plugin` declaration of a plugin's root file. Comptime only: `provides` names the namespace that implements
/// each kind, and the registry builds the kinds' tables from them (`sdk.Arch.of` and its siblings).
pub const Plugin = struct {
    name: []const u8,
    /// The SDK the plugin was built against.
    api: Version,
    /// The MLX the plugin was tested on: `v` + its version (`v0.32.3`).
    mlx: []const u8,
    /// Registers nothing on graphs without the macOS-only sources (Linux, iOS).
    macos_only: bool = false,
    provides: Provides,
};

/// Any subset of the three kinds. A quant or an expert source is its arch's internal (bound at comptime), not a kind.
pub const Provides = struct {
    source: ?type = null,
    arch: ?type = null,
    engine: ?type = null,
};

/// What the host checks a plugin against: its SDK version and its MLX.
pub const Host = struct { api: Version, mlx: []const u8 };

pub const NegotiationError = error{
    /// A breaking SDK change between the plugin and the host. A newer minor on either side is compatible.
    ApiMajorMismatch,
    /// One MLX per process: bumping MLX is one change that bumps every plugin's pin.
    MlxPinMismatch,
};

const testing = std.testing;
