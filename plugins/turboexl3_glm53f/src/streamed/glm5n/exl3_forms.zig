//! GLM-5.3-Flash's EXL3 decode forms by name (settings.zig `Exl3Forms`): the gate / up form and the down form the load
//! line reports. turboexl3 runs the decode (src/routed_t3.zig); the names select nothing else.

pub const GuForm = enum { seam, guone };
pub const DnForm = enum { seam, pair };
pub const Forms = struct {
    gu: GuForm = .seam,
    dn: DnForm = .seam,
};
