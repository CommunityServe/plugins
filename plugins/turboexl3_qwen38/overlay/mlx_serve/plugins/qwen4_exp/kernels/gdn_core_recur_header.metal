// SPDX-License-Identifier: Apache-2.0
// The GDN core recurrence's stage element types: gcore_sel<F, A, B>::type is A when F, else B.
template <bool F, typename A, typename B> struct gcore_sel { typedef B type; };
template <typename A, typename B> struct gcore_sel<true, A, B> { typedef A type; };
