#pragma METAL fp contract(off)
// turboexl3_ds41's tail fusion (src/tail_fuse.zig): the joinless combine's source pick (header_joinless's q3jl_pick).
template <typename S>
inline const device S* tf_pick(int s, const device S* s0, const device S* s1, const device S* s2, const device S* s3, const device S* s4, const device S* s5, const device S* s6, const device S* s7, const device S* s8, const device S* s9, const device S* s10, const device S* s11, const device S* s12, const device S* s13, const device S* s14, const device S* s15, const device S* s16, const device S* s17, const device S* s18, const device S* s19, const device S* s20, const device S* s21, const device S* s22, const device S* s23) {
    switch (s) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        case 3: return s3;
        case 4: return s4;
        case 5: return s5;
        case 6: return s6;
        case 7: return s7;
        case 8: return s8;
        case 9: return s9;
        case 10: return s10;
        case 11: return s11;
        case 12: return s12;
        case 13: return s13;
        case 14: return s14;
        case 15: return s15;
        case 16: return s16;
        case 17: return s17;
        case 18: return s18;
        case 19: return s19;
        case 20: return s20;
        case 21: return s21;
        case 22: return s22;
        default: return s23;
    }
}
