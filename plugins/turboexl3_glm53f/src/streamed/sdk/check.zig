//! The kinds' comptime interface checks: a missing or mistyped declaration is a compile error that names it
//! (docs/plugins.md, negotiation step 3).

/// `T.name` is a function taking exactly `params` and returning `Payload`, or an error union of it.
pub fn fnDecl(comptime where: []const u8, comptime T: type, comptime name: []const u8, comptime params: []const type, comptime Payload: type) void {
    if (!@hasDecl(T, name)) @compileError(where ++ ": no " ++ name);
    const info = switch (@typeInfo(@TypeOf(@field(T, name)))) {
        .@"fn" => |f| f,
        else => @compileError(where ++ "." ++ name ++ " is not a function"),
    };
    if (info.param_types.len != params.len) @compileError(where ++ "." ++ name ++ ": takes a different parameter count than the SDK's");
    for (info.param_types, params) |p, t| {
        if (p.? != t) @compileError(where ++ "." ++ name ++ ": parameter " ++ @typeName(p.?) ++ " where the SDK has " ++ @typeName(t));
    }
    const R = info.return_type.?;
    const P = switch (@typeInfo(R)) {
        .error_union => |eu| eu.payload,
        else => R,
    };
    if (P != Payload) @compileError(where ++ "." ++ name ++ ": returns " ++ @typeName(P) ++ " where the SDK has " ++ @typeName(Payload));
}
