#ifndef WCN_MATH_SOA_KERNELS_H
#define WCN_MATH_SOA_KERNELS_H

/*
 * Plane-wise kernels for the SoA (xN) operators.
 *
 * Each SoA type keeps one float plane per component, so an element-wise operator on,
 * say, Vec4xN is four independent plane operations of identical shape. Writing the
 * SIMD loop out per operator per backend would be ~200 operators x 5 backends of
 * near-identical code; here the loops exist once behind a backend-neutral vector type
 * and every operator composes them.
 *
 * Lane handling
 * -------------
 * A vector carries its own lane count, and every helper takes or inherits it, because
 * RVV's vl is dynamic: a confusing trap is that RVV vector ops use whatever vl the
 * *most recent* vsetvl set, so a helper that quietly called vsetvl could change the
 * length seen by a later operation on unrelated data. Passing vl explicitly (and
 * storing it in the vector) removes that whole class of bug. On the fixed-width ISAs
 * vl is a constant 4 that the compiler folds away.
 *
 * wcn_soa_vlanes() returns 0 when fewer than a full vector remains on the fixed-width
 * ISAs, which ends the vector loop and hands the remainder to the scalar tail; that is
 * what keeps the 16-byte loads from reading past the end of the plane. On RVV it
 * returns the native vl and the scalar tail simply never runs.
 */

/*
 * Deliberately does NOT include wcn_math_internal.h. The only thing this layer needed from it
 * was wcn_fma_mul_add_ps, and pulling in that header entangled the kernels with the project's
 * scalar SIMD branches -- whose RVV/LSX code does not compile, which made it impossible to
 * syntax-check these kernels on those targets. Declaring the dependency explicitly keeps the
 * two separable. See SOA_OPS.md.
 */
#include <stddef.h>
#include <string.h>   // memcpy, used by wcn_soa_copy

#include "WCN/WCN_PLATFORM_MACROS.h"
#include "WCN/WCN_MATH_TYPES.h"
#include "WCN/WCN_MATH_DST.h"

/* Intrinsics for the backend this build targets. */
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
#  include <emmintrin.h>
#  include <immintrin.h>
#  include <smmintrin.h>
#endif
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
#  include <arm_neon.h>
#endif
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
#  include <riscv_vector.h>
#endif
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
#  include <wasm_simd128.h>
#endif
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
#  include <lsxintrin.h>
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
/* Named distinctly from wcn_math_internal.h's helper so a translation unit that includes both
 * headers cannot end up with two definitions. */
#if defined(WCN_HAS_FMA)
static inline __m128 wcn_soa_fma_ps(const __m128 a, const __m128 b, const __m128 c) {
    return _mm_fmadd_ps(a, b, c);
}
#else
static inline __m128 wcn_soa_fma_ps(const __m128 a, const __m128 b, const __m128 c) {
    return _mm_add_ps(_mm_mul_ps(a, b), c);
}
#endif
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64

typedef struct { __m128 v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return (rem >= 4u) ? 4u : 0u; }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.v = _mm_loadu_ps(p); r.vl = vl; return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) {
    wcn_soa_vec r; r.v = _mm_set1_ps(x); r.vl = vl; return r;
}
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { (void)x.vl; _mm_storeu_ps(p, x.v); }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_add_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_sub_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_mul_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_div_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_min_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { a.v = _mm_max_ps(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) {
    /* sign-bit flip, not 0-x: IEEE gives (+0)-(+0) = +0, while the scalar negate in
     * this library (and NEON/WASM/RVV) flip the sign bit, so XOR matches the
     * reference exactly including signed zeros. */
    a.v = _mm_xor_ps(a.v, _mm_set1_ps(-0.0f)); return a;
}
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = _mm_sqrt_ps(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) {
    a.v = wcn_soa_fma_ps(b.v, c.v, a.v); return a;
}
#ifdef __SSE4_1__
#define WCN_SOA_HAS_NATIVE_ROUND 1
static inline wcn_soa_vec wcn_soa_vfloor(wcn_soa_vec a) { a.v = _mm_floor_ps(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vceil(wcn_soa_vec a) { a.v = _mm_ceil_ps(a.v); return a; }
#else
/* No native vector floor/ceil below SSE4.1; the scalar implementations fall back to a
 * scalar loop here too, and an integer-conversion emulation would be wrong for
 * |x| >= 2^31. */
#define WCN_SOA_HAS_NATIVE_ROUND 0
#endif

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    const __m128 m = _mm_cmpneq_ps(a.v, b.v);
    fv.v = _mm_or_ps(_mm_and_ps(m, tv.v), _mm_andnot_ps(m, fv.v));
    return fv;
}

static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) {
    /* clear the sign bit; no SSE2 absolute-value instruction exists */
    a.v = _mm_and_ps(a.v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
    return a;
}
static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                              wcn_soa_vec tv, wcn_soa_vec fv) {
    /* and/andnot rather than blendv: blendv is SSE4.1-only and this file must still
     * build on the SSE2 baseline. */
    const __m128 m = _mm_cmpgt_ps(a.v, b.v);
    fv.v = _mm_or_ps(_mm_and_ps(m, tv.v), _mm_andnot_ps(m, fv.v));
    return fv;
}


#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64

typedef struct { float32x4_t v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return (rem >= 4u) ? 4u : 0u; }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.v = vld1q_f32(p); r.vl = vl; return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) {
    wcn_soa_vec r; r.v = vdupq_n_f32(x); r.vl = vl; return r;
}
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { (void)x.vl; vst1q_f32(p, x.v); }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v = vaddq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v = vsubq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v = vmulq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v = vdivq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { a.v = vminq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { a.v = vmaxq_f32(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) { a.v = vnegq_f32(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = vsqrtq_f32(a.v); return a; }
/* NEON's vfmaq_f32(acc, x, y) is acc + x*y: accumulator first, unlike x86. */
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) {
    a.v = vfmaq_f32(a.v, b.v, c.v); return a;
}
#define WCN_SOA_HAS_NATIVE_ROUND 1
static inline wcn_soa_vec wcn_soa_vfloor(wcn_soa_vec a) { a.v = vrndmq_f32(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vceil(wcn_soa_vec a) { a.v = vrndpq_f32(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = vbslq_f32(vmvnq_u32(vceqq_f32(a.v, b.v)), tv.v, fv.v);
    return fv;
}

static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) { a.v = vabsq_f32(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = vbslq_f32(vcgtq_f32(a.v, b.v), tv.v, fv.v);
    return fv;
}


#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR

/*
 * RISC-V vector types are SIZELESS: the V C Intrinsic API forbids them as struct members,
 * array elements, or operands of sizeof -- and this abstraction stores one in a struct, so the
 * plain `vfloat32m1_t` cannot work here (measured: "field has sizeless type").
 *
 * `riscv_rvv_vector_bits` pins the type to a fixed bit width, making it sized and therefore
 * legal as a member. clang only accepts the attribute when the compilation also passes
 * `-mrvv-vector-bits=<bits>`; see the riscv64 target in .local/tools/crosscheck.js.
 *
 * 128 bits is 4 floats, which is the lane count wcn_soa_vlanes() yields for the default rv64gcv
 * Zvl, so the fixed type and the runtime vl agree.
 */
typedef vfloat32m1_t wcn_soa_vf32 __attribute__((riscv_rvv_vector_bits(128)));
typedef struct { wcn_soa_vf32 v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return __riscv_vsetvl_e32m1(rem); }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.vl = vl; r.v = __riscv_vle32_v_f32m1(p, vl); return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) {
    wcn_soa_vec r; r.vl = vl; r.v = __riscv_vfmv_v_f_f32m1(x, vl); return r;
}
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { __riscv_vse32_v_f32m1(p, x.v, x.vl); }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfadd_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfsub_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfmul_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfdiv_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfmin_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { a.v = __riscv_vfmax_vv_f32m1(a.v, b.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) { a.v = __riscv_vfneg_v_f32m1(a.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = __riscv_vfsqrt_v_f32m1(a.v, a.vl); return a; }
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) {
    a.v = __riscv_vfmadd_vv_f32m1(b.v, c.v, a.v, a.vl); return a;
}
/*
 * No vector round path on RVV. The base vector extension has no float->float rounding
 * instruction: vfcvt converts between floats and integers, and the integer route is wrong for
 * |x| >= 2^31 -- the same reason the x86 branch declines to emulate it below SSE4.1. So the
 * round family falls back to a scalar loop here.
 */
#define WCN_SOA_HAS_NATIVE_ROUND 0

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    const vbool32_t m = __riscv_vmfne_vv_f32m1_b32(a.v, b.v, a.vl);
    fv.v = __riscv_vmerge_vvm_f32m1(fv.v, tv.v, m, fv.vl);
    return fv;
}

static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) {
    a.v = __riscv_vfabs_v_f32m1(a.v, a.vl); return a;
}

static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    const vbool32_t m = __riscv_vmfgt_vv_f32m1_b32(a.v, b.v, a.vl);
    fv.v = __riscv_vmerge_vvm_f32m1(fv.v, tv.v, m, fv.vl);
    return fv;
}

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD

typedef struct { v128_t v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return (rem >= 4u) ? 4u : 0u; }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.v = wasm_v128_load(p); r.vl = vl; return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) {
    wcn_soa_vec r; r.v = wasm_f32x4_splat(x); r.vl = vl; return r;
}
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { (void)x.vl; wasm_v128_store(p, x.v); }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_add(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_sub(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_mul(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_div(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_pmin(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { a.v = wasm_f32x4_pmax(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) { a.v = wasm_f32x4_neg(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = wasm_f32x4_sqrt(a.v); return a; }
/* WASM SIMD has no fused multiply-add. */
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) {
    a.v = wasm_f32x4_add(a.v, wasm_f32x4_mul(b.v, c.v)); return a;
}
#define WCN_SOA_HAS_NATIVE_ROUND 1
static inline wcn_soa_vec wcn_soa_vfloor(wcn_soa_vec a) { a.v = wasm_f32x4_floor(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vceil(wcn_soa_vec a) { a.v = wasm_f32x4_ceil(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = wasm_v128_bitselect(tv.v, fv.v, wasm_f32x4_ne(a.v, b.v));
    return fv;
}

static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) { a.v = wasm_f32x4_abs(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = wasm_v128_bitselect(tv.v, fv.v, wasm_f32x4_gt(a.v, b.v));
    return fv;
}


#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX

typedef struct { __m128 v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return (rem >= 4u) ? 4u : 0u; }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.v = (__m128)__lsx_vld(p, 0); r.vl = vl; return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) {
    wcn_soa_vec r; /* LSX has no float-register splat; load-and-replicate one word instead.
     * __lsx_vreplfr2vr_s does not exist and silently became an implicit int. */
    r.v = (__m128)__lsx_vldrepl_w(&x, 0); r.vl = vl; return r;
}
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { (void)x.vl; __lsx_vst((__m128i)x.v, p, 0); }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfadd_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfsub_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfmul_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfdiv_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfmin_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { a.v = __lsx_vfmax_s(a.v, b.v); return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) {
    /* sign-bit flip for the same reason as the x86 branch */
    const float negzero = -0.0f;
    a.v = (__m128)__lsx_vxor_v((__m128i)a.v, (__m128i)__lsx_vldrepl_w(&negzero, 0));
    return a;
}
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = __lsx_vfsqrt_s(a.v); return a; }
/* LSX has no fused multiply-add. */
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) {
    a.v = __lsx_vfadd_s(a.v, __lsx_vfmul_s(b.v, c.v)); return a;
}
#define WCN_SOA_HAS_NATIVE_ROUND 1
static inline wcn_soa_vec wcn_soa_vfloor(wcn_soa_vec a) { a.v = __lsx_vfrintrm_s(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vceil(wcn_soa_vec a) { a.v = __lsx_vfrintrp_s(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = (__m128)__lsx_vbitsel_v((__m128i)fv.v, (__m128i)tv.v,
                                   (__m128i)__lsx_vfcmp_cune_s(a.v, b.v));
    return fv;
}

static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) {
    /* AND with a 0x7FFFFFFF pattern rather than an abs instruction, which not every LSX
     * header version exposes */
    const int absmask[4] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
    a.v = (__m128)__lsx_vand_v((__m128i)a.v, __lsx_vld(absmask, 0));
    return a;
}

static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = (__m128)__lsx_vbitsel_v((__m128i)fv.v, (__m128i)tv.v,
                                   (__m128i)__lsx_vfcmp_clt_s(b.v, a.v));
    return fv;
}


#else /* scalar-only build: one lane, so the kernels still work unchanged */

typedef struct { float v; size_t vl; } wcn_soa_vec;

static inline size_t wcn_soa_vlanes(size_t rem) { return rem ? 1u : 0u; }
static inline wcn_soa_vec wcn_soa_vload(const float* p, size_t vl) {
    wcn_soa_vec r; r.v = *p; r.vl = vl; return r;
}
static inline wcn_soa_vec wcn_soa_vsplat(float x, size_t vl) { wcn_soa_vec r; r.v = x; r.vl = vl; return r; }
static inline void wcn_soa_vstore(float* p, wcn_soa_vec x) { (void)x.vl; *p = x.v; }
static inline wcn_soa_vec wcn_soa_vadd(wcn_soa_vec a, wcn_soa_vec b) { a.v += b.v; return a; }
static inline wcn_soa_vec wcn_soa_vsub(wcn_soa_vec a, wcn_soa_vec b) { a.v -= b.v; return a; }
static inline wcn_soa_vec wcn_soa_vmul(wcn_soa_vec a, wcn_soa_vec b) { a.v *= b.v; return a; }
static inline wcn_soa_vec wcn_soa_vdiv(wcn_soa_vec a, wcn_soa_vec b) { a.v /= b.v; return a; }
static inline wcn_soa_vec wcn_soa_vmin(wcn_soa_vec a, wcn_soa_vec b) { if (b.v < a.v) a.v = b.v; return a; }
static inline wcn_soa_vec wcn_soa_vmax(wcn_soa_vec a, wcn_soa_vec b) { if (b.v > a.v) a.v = b.v; return a; }
static inline wcn_soa_vec wcn_soa_vneg(wcn_soa_vec a) { a.v = -a.v; return a; }
static inline wcn_soa_vec wcn_soa_vsqrt(wcn_soa_vec a) { a.v = sqrtf(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vfma(wcn_soa_vec a, wcn_soa_vec b, wcn_soa_vec c) { a.v += b.v * c.v; return a; }
#define WCN_SOA_HAS_NATIVE_ROUND 1
static inline wcn_soa_vec wcn_soa_vfloor(wcn_soa_vec a) { a.v = floorf(a.v); return a; }
static inline wcn_soa_vec wcn_soa_vceil(wcn_soa_vec a) { a.v = ceilf(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_ne(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = (a.v != b.v) ? tv.v : fv.v;
    return fv;
}
static inline wcn_soa_vec wcn_soa_vabs(wcn_soa_vec a) { a.v = fabsf(a.v); return a; }

static inline wcn_soa_vec wcn_soa_vsel_gt(wcn_soa_vec a, wcn_soa_vec b,
                                          wcn_soa_vec tv, wcn_soa_vec fv) {
    fv.v = (a.v > b.v) ? tv.v : fv.v;
    return fv;
}

#endif

/* ------------------------------------------------------------------------- */
/* Plane loop skeleton                                                        */
/*                                                                            */
/* Every kernel is: advance while a full vector fits (RVV: while any remains),   */
/* then a scalar tail. The tail is empty on RVV because wcn_soa_vlanes() never    */
/* returns 0 while elements remain.                                              */
/* ------------------------------------------------------------------------- */

#define WCN_SOA_LOOP_BEGIN(I, N)                                                       \
    do {                                                                               \
        size_t I = 0;                                                                  \
        while (I < (N)) {                                                              \
            const size_t wcn_vl_ = wcn_soa_vlanes((N) - I);                            \
            if (wcn_vl_ == 0) break;                                                   \
            {                                                                          \
                (void)wcn_vl_;

#define WCN_SOA_LOOP_END(I, N, SCALAR_BODY);                                            \
            }                                                                          \
            I += wcn_vl_;                                                              \
        }                                                                              \
        for (; I < (N); I++) {                                                         \
            SCALAR_BODY;                                                               \
        }                                                                              \
    } while (0)

/* --- plane-wise binary ---------------------------------------------------- */

#define WCN_DEF_SOA_BINOP(NAME, VOP, SOPEXPR)                                              \
static inline void NAME(float* dst, const float* a, const float* b, size_t n) {        \
    WCN_SOA_LOOP_BEGIN(i, n)                                                           \
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);                                \
        wcn_soa_vec vb = wcn_soa_vload(b + i, wcn_vl_);                                \
        wcn_soa_vstore(dst + i, VOP(va, vb));                                          \
    WCN_SOA_LOOP_END(i, n, dst[i] = SOPEXPR);        \
}

WCN_DEF_SOA_BINOP(wcn_soa_add, wcn_soa_vadd, (a[i] + b[i]))
WCN_DEF_SOA_BINOP(wcn_soa_sub, wcn_soa_vsub, (a[i] - b[i]))
WCN_DEF_SOA_BINOP(wcn_soa_mul, wcn_soa_vmul, (a[i] * b[i]))
WCN_DEF_SOA_BINOP(wcn_soa_div, wcn_soa_vdiv, (a[i] / b[i]))
WCN_DEF_SOA_BINOP(wcn_soa_min, wcn_soa_vmin, (a[i] < b[i] ? a[i] : b[i]))
WCN_DEF_SOA_BINOP(wcn_soa_max, wcn_soa_vmax, (a[i] > b[i] ? a[i] : b[i]))

#undef WCN_DEF_SOA_BINOP

/* --- plane-wise scalar rhs ------------------------------------------------ */

#define WCN_DEF_SOA_SCALAR(NAME, VOP, SOPEXPR)                                             \
static inline void NAME(float* dst, const float* a, float s, size_t n) {               \
    WCN_SOA_LOOP_BEGIN(i, n)                                                           \
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);                                \
        wcn_soa_vec vs = wcn_soa_vsplat(s, wcn_vl_);                                   \
        wcn_soa_vstore(dst + i, VOP(va, vs));                                          \
    WCN_SOA_LOOP_END(i, n, dst[i] = SOPEXPR);        \
}

WCN_DEF_SOA_SCALAR(wcn_soa_mul_s, wcn_soa_vmul, (a[i] * s))
WCN_DEF_SOA_SCALAR(wcn_soa_div_s, wcn_soa_vdiv, (a[i] / s))
WCN_DEF_SOA_SCALAR(wcn_soa_min_s, wcn_soa_vmin, (a[i] < s ? a[i] : s))
WCN_DEF_SOA_SCALAR(wcn_soa_max_s, wcn_soa_vmax, (a[i] > s ? a[i] : s))

#undef WCN_DEF_SOA_SCALAR

/* --- plane-wise unary ----------------------------------------------------- */

#define WCN_DEF_SOA_UNARY(NAME, VOP, SOPEXPR)                                              \
static inline void NAME(float* dst, const float* a, size_t n) {                        \
    WCN_SOA_LOOP_BEGIN(i, n)                                                           \
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);                                \
        wcn_soa_vstore(dst + i, VOP(va));                                              \
    WCN_SOA_LOOP_END(i, n, dst[i] = SOPEXPR);        \
}

WCN_DEF_SOA_UNARY(wcn_soa_neg, wcn_soa_vneg, (-a[i]))
WCN_DEF_SOA_UNARY(wcn_soa_sqrt, wcn_soa_vsqrt, sqrtf(a[i]))

#undef WCN_DEF_SOA_UNARY

/* --- copy / fill (these ignore vl and use plain memory ops) --------------- */

static inline void wcn_soa_copy(float* dst, const float* a, size_t n) {
    if (dst != a) memcpy(dst, a, n * sizeof(float));
}
static inline void wcn_soa_fill(float* dst, float v, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = v;
}

/* --- dst = a + b*s -------------------------------------------------------- */

static inline void wcn_soa_add_scaled(float* dst, const float* a, const float* b, float s,
                                      size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vec vb = wcn_soa_vload(b + i, wcn_vl_);
        wcn_soa_vec vs = wcn_soa_vsplat(s, wcn_vl_);
        /* one rounding instead of two when the target has FMA */
        wcn_soa_vstore(dst + i, wcn_soa_vfma(va, vb, vs));
    WCN_SOA_LOOP_END(i, n, dst[i] = a[i] + b[i] * s);
}

/* --- dst = a + (b-a)*t --------------------------------------------------- */

static inline void wcn_soa_lerp(float* dst, const float* a, const float* b, float t,
                                size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vec vb = wcn_soa_vload(b + i, wcn_vl_);
        wcn_soa_vec vt = wcn_soa_vsplat(t, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vfma(va, wcn_soa_vsub(vb, va), vt));
    WCN_SOA_LOOP_END(i, n, dst[i] = a[i] + (b[i] - a[i]) * t);
}


/* --- plane-wise lerp with a per-element weight ---------------------------- */
/*
 * dst = a + (b - a) * t, with t a plane rather than a broadcast scalar. Used by the
 * xN LERP_V operators, where each component can carry a different interpolation weight.
 * The scalar-weight form above covers WMATH_LERP.
 */
static inline void wcn_soa_lerp_plane(float* dst, const float* a, const float* b,
                                      const float* tp, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vec vb = wcn_soa_vload(b + i, wcn_vl_);
        wcn_soa_vec vt = wcn_soa_vload(tp + i, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vfma(va, wcn_soa_vsub(vb, va), vt));
    WCN_SOA_LOOP_END(i, n, dst[i] = a[i] + (b[i] - a[i]) * tp[i]);
}

/* --- dst = clamp(a, lo, hi) ---------------------------------------------- */

static inline void wcn_soa_clamp(float* dst, const float* a, float lo, float hi, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vec vlo = wcn_soa_vsplat(lo, wcn_vl_);
        wcn_soa_vec vhi = wcn_soa_vsplat(hi, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vmin(wcn_soa_vmax(va, vlo), vhi));
    WCN_SOA_LOOP_END(i, n, dst[i] = (a[i] < lo) ? lo : ((a[i] > hi) ? hi : a[i]));
}

/* --- dst = 1/a ------------------------------------------------------------ */

static inline void wcn_soa_inv(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vec one = wcn_soa_vsplat(1.0f, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vdiv(one, va));
    WCN_SOA_LOOP_END(i, n, dst[i] = 1.0f / a[i]);
}

/* --- dst = sqrt(a*a) ------------------------------------------------------ */

static inline void wcn_soa_len_sq(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vmul(va, va));
    WCN_SOA_LOOP_END(i, n, dst[i] = a[i] * a[i]);
}

/* --- fused per-element reductions over 2..4 planes ------------------------ */

/*
 * dot2..dot4 compute dst[k] = sum_p a_p[k] * b_p[k] for one element at a time, so the
 * result plane holds a per-element scalar rather than a vector. This is a single pass
 * over the planes with one accumulator, which is why it needs its own kernel instead
 * of being composed from wcn_soa_mul + wcn_soa_add (that would need a temporary plane
 * and three passes).
 */
static inline void wcn_soa_dot2(float* dst, const float* ax, const float* ay,
                                const float* bx, const float* by, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec a0 = wcn_soa_vload(ax + i, wcn_vl_);
        wcn_soa_vec a1 = wcn_soa_vload(ay + i, wcn_vl_);
        wcn_soa_vec b0 = wcn_soa_vload(bx + i, wcn_vl_);
        wcn_soa_vec b1 = wcn_soa_vload(by + i, wcn_vl_);
        wcn_soa_vec acc = wcn_soa_vmul(a0, b0);
        acc = wcn_soa_vfma(acc, a1, b1);
        wcn_soa_vstore(dst + i, acc);
    WCN_SOA_LOOP_END(i, n, dst[i] = ax[i] * bx[i] + ay[i] * by[i]);
}

static inline void wcn_soa_dot3(float* dst, const float* ax, const float* ay,
                                const float* az, const float* bx, const float* by,
                                const float* bz, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec a0 = wcn_soa_vload(ax + i, wcn_vl_);
        wcn_soa_vec a1 = wcn_soa_vload(ay + i, wcn_vl_);
        wcn_soa_vec a2 = wcn_soa_vload(az + i, wcn_vl_);
        wcn_soa_vec b0 = wcn_soa_vload(bx + i, wcn_vl_);
        wcn_soa_vec b1 = wcn_soa_vload(by + i, wcn_vl_);
        wcn_soa_vec b2 = wcn_soa_vload(bz + i, wcn_vl_);
        wcn_soa_vec acc = wcn_soa_vmul(a0, b0);
        acc = wcn_soa_vfma(acc, a1, b1);
        acc = wcn_soa_vfma(acc, a2, b2);
        wcn_soa_vstore(dst + i, acc);
    WCN_SOA_LOOP_END(i, n, dst[i] = ax[i] * bx[i] + ay[i] * by[i] + az[i] * bz[i]);
}

static inline void wcn_soa_dot4(float* dst, const float* ax, const float* ay,
                                const float* az, const float* aw, const float* bx,
                                const float* by, const float* bz, const float* bw,
                                size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec a0 = wcn_soa_vload(ax + i, wcn_vl_);
        wcn_soa_vec a1 = wcn_soa_vload(ay + i, wcn_vl_);
        wcn_soa_vec a2 = wcn_soa_vload(az + i, wcn_vl_);
        wcn_soa_vec a3 = wcn_soa_vload(aw + i, wcn_vl_);
        wcn_soa_vec b0 = wcn_soa_vload(bx + i, wcn_vl_);
        wcn_soa_vec b1 = wcn_soa_vload(by + i, wcn_vl_);
        wcn_soa_vec b2 = wcn_soa_vload(bz + i, wcn_vl_);
        wcn_soa_vec b3 = wcn_soa_vload(bw + i, wcn_vl_);
        wcn_soa_vec acc = wcn_soa_vmul(a0, b0);
        acc = wcn_soa_vfma(acc, a1, b1);
        acc = wcn_soa_vfma(acc, a2, b2);
        acc = wcn_soa_vfma(acc, a3, b3);
        wcn_soa_vstore(dst + i, acc);
    WCN_SOA_LOOP_END(i, n,
                     dst[i] = ax[i] * bx[i] + ay[i] * by[i] + az[i] * bz[i] + aw[i] * bw[i]);
}

/* --- length squared / length / distance over planes ----------------------- */

static inline void wcn_soa_lensq2(float* dst, const float* ax, const float* ay, size_t n) {
    wcn_soa_dot2(dst, ax, ay, ax, ay, n);
}
static inline void wcn_soa_lensq3(float* dst, const float* ax, const float* ay,
                                 const float* az, size_t n) {
    wcn_soa_dot3(dst, ax, ay, az, ax, ay, az, n);
}
static inline void wcn_soa_lensq4(float* dst, const float* ax, const float* ay,
                                 const float* az, const float* aw, size_t n) {
    wcn_soa_dot4(dst, ax, ay, az, aw, ax, ay, az, aw, n);
}

static inline void wcn_soa_len2(float* dst, const float* ax, const float* ay, size_t n) {
    wcn_soa_lensq2(dst, ax, ay, n);
    wcn_soa_sqrt(dst, dst, n);
}
static inline void wcn_soa_len3(float* dst, const float* ax, const float* ay,
                                const float* az, size_t n) {
    wcn_soa_lensq3(dst, ax, ay, az, n);
    wcn_soa_sqrt(dst, dst, n);
}
static inline void wcn_soa_len4(float* dst, const float* ax, const float* ay,
                                const float* az, const float* aw, size_t n) {
    wcn_soa_lensq4(dst, ax, ay, az, aw, n);
    wcn_soa_sqrt(dst, dst, n);
}

static inline void wcn_soa_distsq2(float* dst, const float* ax, const float* ay,
                                   const float* bx, const float* by, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec d0 = wcn_soa_vsub(wcn_soa_vload(ax + i, wcn_vl_),
                                      wcn_soa_vload(bx + i, wcn_vl_));
        wcn_soa_vec d1 = wcn_soa_vsub(wcn_soa_vload(ay + i, wcn_vl_),
                                      wcn_soa_vload(by + i, wcn_vl_));
        wcn_soa_vstore(dst + i, wcn_soa_vfma(wcn_soa_vmul(d0, d0), d1, d1));
    WCN_SOA_LOOP_END(i, n,
                     dst[i] = (ax[i] - bx[i]) * (ax[i] - bx[i]) +
                              (ay[i] - by[i]) * (ay[i] - by[i]));
}

static inline void wcn_soa_distsq3(float* dst, const float* ax, const float* ay,
                                   const float* az, const float* bx, const float* by,
                                   const float* bz, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec d0 = wcn_soa_vsub(wcn_soa_vload(ax + i, wcn_vl_),
                                      wcn_soa_vload(bx + i, wcn_vl_));
        wcn_soa_vec d1 = wcn_soa_vsub(wcn_soa_vload(ay + i, wcn_vl_),
                                      wcn_soa_vload(by + i, wcn_vl_));
        wcn_soa_vec d2 = wcn_soa_vsub(wcn_soa_vload(az + i, wcn_vl_),
                                      wcn_soa_vload(bz + i, wcn_vl_));
        wcn_soa_vec acc = wcn_soa_vmul(d0, d0);
        acc = wcn_soa_vfma(acc, d1, d1);
        acc = wcn_soa_vfma(acc, d2, d2);
        wcn_soa_vstore(dst + i, acc);
    WCN_SOA_LOOP_END(i, n,
                     dst[i] = (ax[i] - bx[i]) * (ax[i] - bx[i]) +
                              (ay[i] - by[i]) * (ay[i] - by[i]) +
                              (az[i] - bz[i]) * (az[i] - bz[i]));
}

static inline void wcn_soa_distsq4(float* dst, const float* ax, const float* ay,
                                   const float* az, const float* aw, const float* bx,
                                   const float* by, const float* bz, const float* bw,
                                   size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec d0 = wcn_soa_vsub(wcn_soa_vload(ax + i, wcn_vl_),
                                      wcn_soa_vload(bx + i, wcn_vl_));
        wcn_soa_vec d1 = wcn_soa_vsub(wcn_soa_vload(ay + i, wcn_vl_),
                                      wcn_soa_vload(by + i, wcn_vl_));
        wcn_soa_vec d2 = wcn_soa_vsub(wcn_soa_vload(az + i, wcn_vl_),
                                      wcn_soa_vload(bz + i, wcn_vl_));
        wcn_soa_vec d3 = wcn_soa_vsub(wcn_soa_vload(aw + i, wcn_vl_),
                                      wcn_soa_vload(bw + i, wcn_vl_));
        wcn_soa_vec acc = wcn_soa_vmul(d0, d0);
        acc = wcn_soa_vfma(acc, d1, d1);
        acc = wcn_soa_vfma(acc, d2, d2);
        acc = wcn_soa_vfma(acc, d3, d3);
        wcn_soa_vstore(dst + i, acc);
    WCN_SOA_LOOP_END(i, n,
                     dst[i] = (ax[i] - bx[i]) * (ax[i] - bx[i]) +
                              (ay[i] - by[i]) * (ay[i] - by[i]) +
                              (az[i] - bz[i]) * (az[i] - bz[i]) +
                              (aw[i] - bw[i]) * (aw[i] - bw[i]));
}

/* --- fused normalize, 2 and 3 planes -------------------------------------- */
/*
 * dst = a / |a| when |a|^2 > eps^2, else dst = 0.
 *
 * That threshold-and-zero rule is what the scalar WMATH_NORMALIZE(Vec3) does, so it is
 * reproduced exactly; an earlier draft used "leave the vector unchanged when the length
 * is 0", which would have made the SIMD body and the scalar tail disagree on
 * zero-length input.
 *
 * It cannot be composed from the plane kernels: the reciprocal length must be held
 * while all planes are scaled, which would need a temporary plane as long as the input.
 * The comparison is a select rather than a branch, so the vector body stays
 * branch-free.
 */

static inline void wcn_soa_normalize2(float* dx, float* dy, const float* sx,
                                      const float* sy, size_t n) {
    const float eps = wcn_math_get_epsilon();
    const float eps2 = eps * eps;
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        wcn_soa_vec len2 = wcn_soa_vfma(wcn_soa_vmul(x, x), y, y);
        wcn_soa_vec inv = wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), wcn_soa_vsqrt(len2));
        wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        wcn_soa_vec thresh = wcn_soa_vsplat(eps2, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(x, inv), zero));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(y, inv), zero));
    WCN_SOA_LOOP_END(i, n, {
        const float l2 = sx[i] * sx[i] + sy[i] * sy[i];
        if (l2 > eps2) {
            const float il = 1.0f / sqrtf(l2);
            dx[i] = sx[i] * il;
            dy[i] = sy[i] * il;
        } else {
            dx[i] = 0.0f;
            dy[i] = 0.0f;
        }
    });
}

static inline void wcn_soa_normalize3(float* dx, float* dy, float* dz, const float* sx,
                                      const float* sy, const float* sz, size_t n) {
    const float eps = wcn_math_get_epsilon();
    const float eps2 = eps * eps;
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        wcn_soa_vec len2 = wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z);
        wcn_soa_vec inv = wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), wcn_soa_vsqrt(len2));
        wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        wcn_soa_vec thresh = wcn_soa_vsplat(eps2, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(x, inv), zero));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(y, inv), zero));
        wcn_soa_vstore(dz + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(z, inv), zero));
    WCN_SOA_LOOP_END(i, n, {
        const float l2 = sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i];
        if (l2 > eps2) {
            const float il = 1.0f / sqrtf(l2);
            dx[i] = sx[i] * il;
            dy[i] = sy[i] * il;
            dz[i] = sz[i] * il;
        } else {
            dx[i] = 0.0f;
            dy[i] = 0.0f;
            dz[i] = 0.0f;
        }
    });
}


static inline void wcn_soa_normalize4(float* dx, float* dy, float* dz, float* dw,
                                      const float* sx, const float* sy, const float* sz,
                                      const float* sw, size_t n) {
    const float eps = wcn_math_get_epsilon();
    const float eps2 = eps * eps;
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        wcn_soa_vec w = wcn_soa_vload(sw + i, wcn_vl_);
        wcn_soa_vec len2 = wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z), w, w);
        wcn_soa_vec inv = wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), wcn_soa_vsqrt(len2));
        wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        wcn_soa_vec thresh = wcn_soa_vsplat(eps2, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(x, inv), zero));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(y, inv), zero));
        wcn_soa_vstore(dz + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(z, inv), zero));
        wcn_soa_vstore(dw + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(w, inv), zero));
    WCN_SOA_LOOP_END(i, n, {
        const float l2 = sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i] + sw[i] * sw[i];
        if (l2 > eps2) {
            const float il = 1.0f / sqrtf(l2);
            dx[i] = sx[i] * il;
            dy[i] = sy[i] * il;
            dz[i] = sz[i] * il;
            dw[i] = sw[i] * il;
        } else {
            dx[i] = 0.0f; dy[i] = 0.0f; dz[i] = 0.0f; dw[i] = 0.0f;
        }
    });
}

/* dst = v scaled down to max_length when |v| > max_length, else v unchanged */
static inline void wcn_soa_truncate4(float* dx, float* dy, float* dz, float* dw,
                                     const float* sx, const float* sy, const float* sz,
                                     const float* sw, float max_length, size_t n) {
    const float max2 = max_length * max_length;
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        wcn_soa_vec w = wcn_soa_vload(sw + i, wcn_vl_);
        wcn_soa_vec len2 = wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z), w, w);
        /* scale = max_length / sqrt(len_sq); only used where len_sq > max^2 */
        wcn_soa_vec scale = wcn_soa_vmul(
            wcn_soa_vsplat(max_length, wcn_vl_),
            wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), wcn_soa_vsqrt(len2)));
        wcn_soa_vec thresh = wcn_soa_vsplat(max2, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(x, scale), x));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(y, scale), y));
        wcn_soa_vstore(dz + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(z, scale), z));
        wcn_soa_vstore(dw + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(w, scale), w));
    WCN_SOA_LOOP_END(i, n, {
        const float l2 = sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i] + sw[i] * sw[i];
        if (l2 > max2 && l2 > 0.0f) {
            const float s = max_length / sqrtf(l2);
            dx[i] = sx[i] * s;
            dy[i] = sy[i] * s;
            dz[i] = sz[i] * s;
            dw[i] = sw[i] * s;
        } else {
            dx[i] = sx[i];
            dy[i] = sy[i];
            dz[i] = sz[i];
            dw[i] = sw[i];
        }
    });
}

/* --- fused cross product, 3 output planes -------------------------------- */
/*
 * dst = a x b, one fused pass: the components share the loaded operands, whereas as
 * independent plane-wise steps every plane would be reloaded three times.
 */
static inline void wcn_soa_cross3(float* dx, float* dy, float* dz, const float* ax,
                                  const float* ay, const float* az, const float* bx,
                                  const float* by, const float* bz, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vec x0 = wcn_soa_vload(ax + i, wcn_vl_);
        wcn_soa_vec y0 = wcn_soa_vload(ay + i, wcn_vl_);
        wcn_soa_vec z0 = wcn_soa_vload(az + i, wcn_vl_);
        wcn_soa_vec x1 = wcn_soa_vload(bx + i, wcn_vl_);
        wcn_soa_vec y1 = wcn_soa_vload(by + i, wcn_vl_);
        wcn_soa_vec z1 = wcn_soa_vload(bz + i, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsub(wcn_soa_vmul(y0, z1), wcn_soa_vmul(z0, y1)));
        wcn_soa_vstore(dy + i, wcn_soa_vsub(wcn_soa_vmul(z0, x1), wcn_soa_vmul(x0, z1)));
        wcn_soa_vstore(dz + i, wcn_soa_vsub(wcn_soa_vmul(x0, y1), wcn_soa_vmul(y0, x1)));
    WCN_SOA_LOOP_END(i, n, {
        dx[i] = ay[i] * bz[i] - az[i] * by[i];
        dy[i] = az[i] * bx[i] - ax[i] * bz[i];
        dz[i] = ax[i] * by[i] - ay[i] * bx[i];
    });
}

/* --- quaternion kernels -------------------------------------------------- */
/*
 * Quaternion layout matches Vec4xN: planes x, y, z, w with w the scalar part.
 *
 * MULTIPLY, one fused pass over four output planes:
 *     w = aw*bw - (ax*bx + ay*by + az*bz)
 *     x = aw*bx + ax*bw + (ay*bz - az*by)
 *     y = aw*by + ay*bw + (az*bx - ax*bz)
 *     z = aw*bz + az*bw + (ax*by - ay*bx)
 * The parenthesised terms are the components of a.xyz x b.xyz, which is why every
 * output needs all eight inputs and cannot be reached plane by plane.
 */
static inline void wcn_soa_quat_mul4(float* dx, float* dy, float* dz, float* dw,
                                     const float* ax, const float* ay, const float* az,
                                     const float* aw, const float* bx, const float* by,
                                     const float* bz, const float* bw, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec a0 = wcn_soa_vload(ax + i, wcn_vl_);
        const wcn_soa_vec a1 = wcn_soa_vload(ay + i, wcn_vl_);
        const wcn_soa_vec a2 = wcn_soa_vload(az + i, wcn_vl_);
        const wcn_soa_vec a3 = wcn_soa_vload(aw + i, wcn_vl_);
        const wcn_soa_vec b0 = wcn_soa_vload(bx + i, wcn_vl_);
        const wcn_soa_vec b1 = wcn_soa_vload(by + i, wcn_vl_);
        const wcn_soa_vec b2 = wcn_soa_vload(bz + i, wcn_vl_);
        const wcn_soa_vec b3 = wcn_soa_vload(bw + i, wcn_vl_);

        const wcn_soa_vec dotxyz =
            wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a0, b0), a1, b1), a2, b2);
        wcn_soa_vstore(dw + i, wcn_soa_vsub(wcn_soa_vmul(a3, b3), dotxyz));

        /* a.xyz x b.xyz components */
        const wcn_soa_vec cx = wcn_soa_vsub(wcn_soa_vmul(a1, b2), wcn_soa_vmul(a2, b1));
        const wcn_soa_vec cy = wcn_soa_vsub(wcn_soa_vmul(a2, b0), wcn_soa_vmul(a0, b2));
        const wcn_soa_vec cz = wcn_soa_vsub(wcn_soa_vmul(a0, b1), wcn_soa_vmul(a1, b0));

        /* aw*b + bw*a + cross, each as acc + m*n through vfma */
        wcn_soa_vstore(dx + i,
                       wcn_soa_vadd(wcn_soa_vfma(wcn_soa_vmul(a0, b3), a3, b0), cx));
        wcn_soa_vstore(dy + i,
                       wcn_soa_vadd(wcn_soa_vfma(wcn_soa_vmul(a1, b3), a3, b1), cy));
        wcn_soa_vstore(dz + i,
                       wcn_soa_vadd(wcn_soa_vfma(wcn_soa_vmul(a2, b3), a3, b2), cz));
    WCN_SOA_LOOP_END(i, n, {
        const float a0 = ax[i];
        const float a1 = ay[i];
        const float a2 = az[i];
        const float a3 = aw[i];
        const float b0 = bx[i];
        const float b1 = by[i];
        const float b2 = bz[i];
        const float b3 = bw[i];
        dw[i] = a3 * b3 - (a0 * b0 + a1 * b1 + a2 * b2);
        dx[i] = a3 * b0 + a0 * b3 + (a1 * b2 - a2 * b1);
        dy[i] = a3 * b1 + a1 * b3 + (a2 * b0 - a0 * b2);
        dz[i] = a3 * b2 + a2 * b3 + (a0 * b1 - a1 * b0);
    });
}

/*
 * Normalize, falling back to the quaternion identity (0,0,0,1) below the threshold --
 * NOT zeros, which is what the Vec4 normalize does. A zero quaternion is not a valid
 * basis for later multiplications, so the scalar implementation substitutes the
 * identity and this must match it exactly.
 */
static inline void wcn_soa_quat_normalize4(float* dx, float* dy, float* dz, float* dw,
                                           const float* sx, const float* sy,
                                           const float* sz, const float* sw, size_t n) {
    const float eps = wcn_math_get_epsilon();
    const float eps2 = eps * eps;
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        const wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        const wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        const wcn_soa_vec w = wcn_soa_vload(sw + i, wcn_vl_);
        const wcn_soa_vec len2 =
            wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z), w, w);
        const wcn_soa_vec inv =
            wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), wcn_soa_vsqrt(len2));
        const wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        const wcn_soa_vec one = wcn_soa_vsplat(1.0f, wcn_vl_);
        const wcn_soa_vec thresh = wcn_soa_vsplat(eps2, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(x, inv), zero));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(y, inv), zero));
        wcn_soa_vstore(dz + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(z, inv), zero));
        wcn_soa_vstore(dw + i, wcn_soa_vsel_gt(len2, thresh, wcn_soa_vmul(w, inv), one));
    WCN_SOA_LOOP_END(i, n, {
        const float l2 = sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i] + sw[i] * sw[i];
        if (l2 > eps2) {
            const float il = 1.0f / sqrtf(l2);
            dx[i] = sx[i] * il;
            dy[i] = sy[i] * il;
            dz[i] = sz[i] * il;
            dw[i] = sw[i] * il;
        } else {
            dx[i] = 0.0f;
            dy[i] = 0.0f;
            dz[i] = 0.0f;
            dw[i] = 1.0f;
        }
    });
}

/*
 * Inverse: conjugate(q) / dot(q,q), with a zero norm mapped to 0 rather than dividing.
 * dot is a sum of squares so it is never negative, which is what lets the guard be the
 * same "greater than zero" select the vector paths already have.
 */
static inline void wcn_soa_quat_inverse4(float* dx, float* dy, float* dz, float* dw,
                                         const float* sx, const float* sy, const float* sz,
                                         const float* sw, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        const wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        const wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        const wcn_soa_vec w = wcn_soa_vload(sw + i, wcn_vl_);
        const wcn_soa_vec dot =
            wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z), w, w);
        const wcn_soa_vec one = wcn_soa_vsplat(1.0f, wcn_vl_);
        const wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        const wcn_soa_vec inv = wcn_soa_vsel_gt(dot, zero, wcn_soa_vdiv(one, dot), zero);
        wcn_soa_vstore(dx + i, wcn_soa_vmul(wcn_soa_vneg(x), inv));
        wcn_soa_vstore(dy + i, wcn_soa_vmul(wcn_soa_vneg(y), inv));
        wcn_soa_vstore(dz + i, wcn_soa_vmul(wcn_soa_vneg(z), inv));
        wcn_soa_vstore(dw + i, wcn_soa_vmul(w, inv));
    WCN_SOA_LOOP_END(i, n, {
        const float d = sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i] + sw[i] * sw[i];
        const float k = (d != 0.0f) ? 1.0f / d : 0.0f;
        dx[i] = -sx[i] * k;
        dy[i] = -sy[i] * k;
        dz[i] = -sz[i] * k;
        dw[i] = sw[i] * k;
    });
}

/* --- Mat3xN kernels ------------------------------------------------------ */
/*
 * Packed plane layout: p = row*3 + col, i.e.
 *     p0 p1 p2      m00 m01 m02
 *     p3 p4 p5  ==  m10 m11 m12
 *     p6 p7 p8      m20 m21 m22
 * (Mat3xN carries nine planes; Mat3's in-memory form pads each row to four floats, so
 * these indices are not Mat3's m[] indices.)
 *
 * Fused rather than composed from the plane kernels because each output reads several
 * planes at once: MULTIPLY's nine outputs are each a sum of three cross-plane products,
 * DETERMINANT collapses nine inputs to one, and INVERSE's cofactors each read four.
 */
/*
 * Read-only view of a plane array, for passing a container's `float* m[N]` into a kernel that
 * declares `const float* const*`.
 *
 * C17 6.5.16.1 forbids that conversion implicitly: const must be added at every pointer level
 * down to the one being qualified, and `float** -> const float* const*` adds it to the pointee
 * two levels down. So the call site has to spell it out. This cast only ever ADDS const, so it
 * cannot hide a write -- which is why the kernels keep the stronger signature rather than the
 * cast being avoided by weakening it.
 */
#define WCN_SOA_RO(planes) ((const float* const*)(planes))

static inline void wcn_soa_m3_mul9(float* const* d, const float* const* A,
                                   const float* const* B, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec a0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec a1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec a2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec a3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec a4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec a5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec a6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec a7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec a8 = wcn_soa_vload(A[8] + i, wcn_vl_);
        const wcn_soa_vec b0 = wcn_soa_vload(B[0] + i, wcn_vl_);
        const wcn_soa_vec b1 = wcn_soa_vload(B[1] + i, wcn_vl_);
        const wcn_soa_vec b2 = wcn_soa_vload(B[2] + i, wcn_vl_);
        const wcn_soa_vec b3 = wcn_soa_vload(B[3] + i, wcn_vl_);
        const wcn_soa_vec b4 = wcn_soa_vload(B[4] + i, wcn_vl_);
        const wcn_soa_vec b5 = wcn_soa_vload(B[5] + i, wcn_vl_);
        const wcn_soa_vec b6 = wcn_soa_vload(B[6] + i, wcn_vl_);
        const wcn_soa_vec b7 = wcn_soa_vload(B[7] + i, wcn_vl_);
        const wcn_soa_vec b8 = wcn_soa_vload(B[8] + i, wcn_vl_);
        /* d[row*3+col] = sum_k a[row*3+k] * b[k*3+col] */
        wcn_soa_vstore(d[0] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a0, b0), a1, b3), a2, b6));
        wcn_soa_vstore(d[1] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a0, b1), a1, b4), a2, b7));
        wcn_soa_vstore(d[2] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a0, b2), a1, b5), a2, b8));
        wcn_soa_vstore(d[3] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a3, b0), a4, b3), a5, b6));
        wcn_soa_vstore(d[4] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a3, b1), a4, b4), a5, b7));
        wcn_soa_vstore(d[5] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a3, b2), a4, b5), a5, b8));
        wcn_soa_vstore(d[6] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a6, b0), a7, b3), a8, b6));
        wcn_soa_vstore(d[7] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a6, b1), a7, b4), a8, b7));
        wcn_soa_vstore(d[8] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(a6, b2), a7, b5), a8, b8));
    WCN_SOA_LOOP_END(i, n, {
        const float a0 = A[0][i];
        const float a1 = A[1][i];
        const float a2 = A[2][i];
        const float a3 = A[3][i];
        const float a4 = A[4][i];
        const float a5 = A[5][i];
        const float a6 = A[6][i];
        const float a7 = A[7][i];
        const float a8 = A[8][i];
        const float b0 = B[0][i];
        const float b1 = B[1][i];
        const float b2 = B[2][i];
        const float b3 = B[3][i];
        const float b4 = B[4][i];
        const float b5 = B[5][i];
        const float b6 = B[6][i];
        const float b7 = B[7][i];
        const float b8 = B[8][i];
        d[0][i] = a0 * b0 + a1 * b3 + a2 * b6;
        d[1][i] = a0 * b1 + a1 * b4 + a2 * b7;
        d[2][i] = a0 * b2 + a1 * b5 + a2 * b8;
        d[3][i] = a3 * b0 + a4 * b3 + a5 * b6;
        d[4][i] = a3 * b1 + a4 * b4 + a5 * b7;
        d[5][i] = a3 * b2 + a4 * b5 + a5 * b8;
        d[6][i] = a6 * b0 + a7 * b3 + a8 * b6;
        d[7][i] = a6 * b1 + a7 * b4 + a8 * b7;
        d[8][i] = a6 * b2 + a7 * b5 + a8 * b8;
    });
}

/* det = p0*(p4*p8 - p7*p5) - p3*(p1*p8 - p7*p2) + p6*(p1*p5 - p4*p2) */
static inline void wcn_soa_m3_det(float* out, const float* const* A, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec p0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec p1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec p2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec p3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec p4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec p5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec p6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec p7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec p8 = wcn_soa_vload(A[8] + i, wcn_vl_);
        const wcn_soa_vec t0 = wcn_soa_vsub(wcn_soa_vmul(p4, p8), wcn_soa_vmul(p7, p5));
        const wcn_soa_vec t1 = wcn_soa_vsub(wcn_soa_vmul(p1, p8), wcn_soa_vmul(p7, p2));
        const wcn_soa_vec t2 = wcn_soa_vsub(wcn_soa_vmul(p1, p5), wcn_soa_vmul(p4, p2));
        const wcn_soa_vec acc = wcn_soa_vsub(wcn_soa_vmul(p0, t0), wcn_soa_vmul(p3, t1));
        wcn_soa_vstore(out + i, wcn_soa_vfma(acc, p6, t2));
    WCN_SOA_LOOP_END(i, n, out[i] = A[0][i] * (A[4][i] * A[8][i] - A[7][i] * A[5][i]) -
                                    A[3][i] * (A[1][i] * A[8][i] - A[7][i] * A[2][i]) +
                                    A[6][i] * (A[1][i] * A[5][i] - A[4][i] * A[2][i]));
}

/*
 * Inverse: the nine cofactors over the determinant, with the scalar singular guard.
 * Below |det| < 1e-12 the result is the IDENTITY matrix, not inf/nan -- reproduced with
 * an abs and a masked select per component, so the vector body stays branch-free.
 */
static inline void wcn_soa_m3_inv9(float* const* d, const float* const* A, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec p0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec p1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec p2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec p3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec p4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec p5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec p6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec p7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec p8 = wcn_soa_vload(A[8] + i, wcn_vl_);

        const wcn_soa_vec c00 = wcn_soa_vsub(wcn_soa_vmul(p4, p8), wcn_soa_vmul(p5, p7));
        const wcn_soa_vec c01 = wcn_soa_vsub(wcn_soa_vmul(p2, p7), wcn_soa_vmul(p1, p8));
        const wcn_soa_vec c02 = wcn_soa_vsub(wcn_soa_vmul(p1, p5), wcn_soa_vmul(p2, p4));
        const wcn_soa_vec c10 = wcn_soa_vsub(wcn_soa_vmul(p5, p6), wcn_soa_vmul(p3, p8));
        const wcn_soa_vec c11 = wcn_soa_vsub(wcn_soa_vmul(p0, p8), wcn_soa_vmul(p2, p6));
        const wcn_soa_vec c12 = wcn_soa_vsub(wcn_soa_vmul(p2, p3), wcn_soa_vmul(p0, p5));
        const wcn_soa_vec c20 = wcn_soa_vsub(wcn_soa_vmul(p3, p7), wcn_soa_vmul(p4, p6));
        const wcn_soa_vec c21 = wcn_soa_vsub(wcn_soa_vmul(p1, p6), wcn_soa_vmul(p0, p7));
        const wcn_soa_vec c22 = wcn_soa_vsub(wcn_soa_vmul(p0, p4), wcn_soa_vmul(p1, p3));

        /* det = p0*c00 + p1*c10 + p2*c20 */
        const wcn_soa_vec det = wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(p0, c00), p1, c10), p2, c20);
        const wcn_soa_vec kinv =
            wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), det);
        const wcn_soa_vec thresh = wcn_soa_vsplat(1e-12f, wcn_vl_);
        const wcn_soa_vec absdet = wcn_soa_vabs(det);
        const wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        const wcn_soa_vec one = wcn_soa_vsplat(1.0f, wcn_vl_);
        /* (thresh > absdet) ? identity_component : cofactor * kinv */
        wcn_soa_vstore(d[0] + i, wcn_soa_vsel_gt(thresh, absdet, one, wcn_soa_vmul(c00, kinv)));
        wcn_soa_vstore(d[1] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c01, kinv)));
        wcn_soa_vstore(d[2] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c02, kinv)));
        wcn_soa_vstore(d[3] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c10, kinv)));
        wcn_soa_vstore(d[4] + i, wcn_soa_vsel_gt(thresh, absdet, one, wcn_soa_vmul(c11, kinv)));
        wcn_soa_vstore(d[5] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c12, kinv)));
        wcn_soa_vstore(d[6] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c20, kinv)));
        wcn_soa_vstore(d[7] + i, wcn_soa_vsel_gt(thresh, absdet, zero, wcn_soa_vmul(c21, kinv)));
        wcn_soa_vstore(d[8] + i, wcn_soa_vsel_gt(thresh, absdet, one, wcn_soa_vmul(c22, kinv)));
    WCN_SOA_LOOP_END(i, n, {
        const float m00 = A[0][i];
        const float m01 = A[1][i];
        const float m02 = A[2][i];
        const float m10 = A[3][i];
        const float m11 = A[4][i];
        const float m12 = A[5][i];
        const float m20 = A[6][i];
        const float m21 = A[7][i];
        const float m22 = A[8][i];
        const float c00 = m11 * m22 - m12 * m21;
        const float c01 = m02 * m21 - m01 * m22;
        const float c02 = m01 * m12 - m02 * m11;
        const float c10 = m12 * m20 - m10 * m22;
        const float c11 = m00 * m22 - m02 * m20;
        const float c12 = m02 * m10 - m00 * m12;
        const float c20 = m10 * m21 - m11 * m20;
        const float c21 = m01 * m20 - m00 * m21;
        const float c22 = m00 * m11 - m01 * m10;
        const float det = m00 * c00 + m01 * c10 + m02 * c20;
        if (fabsf(det) < 1e-12f) {
            d[0][i] = 1.0f; d[1][i] = 0.0f; d[2][i] = 0.0f;
            d[3][i] = 0.0f; d[4][i] = 1.0f; d[5][i] = 0.0f;
            d[6][i] = 0.0f; d[7][i] = 0.0f; d[8][i] = 1.0f;
        } else {
            const float k = 1.0f / det;
            d[0][i] = c00 * k; d[1][i] = c01 * k; d[2][i] = c02 * k;
            d[3][i] = c10 * k; d[4][i] = c11 * k; d[5][i] = c12 * k;
            d[6][i] = c20 * k; d[7][i] = c21 * k; d[8][i] = c22 * k;
        }
    });
}

/* --- Mat4xN kernels ----------------------------------------------------- */
/*
 * Plane layout: sixteen planes, plane p is Mat4's m[p] directly (row-major, no
 * padding), so no index mapping is required here -- unlike Mat3xN, whose packed
 * nine-plane layout had to be mapped into Mat3's padded m[].
 *
 * Vector bodies are generated from the definition; scalar tails call the scalar
 * library functions, which keeps them from drifting from the reference and costs
 * nothing (they run only for the final three elements).
 */

static inline void wcn_soa_m4_mul16(float* const* d, const float* const* A,
                                    const float* const* B, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec A0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec A1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec A2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec A3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec A4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec A5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec A6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec A7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec A8 = wcn_soa_vload(A[8] + i, wcn_vl_);
        const wcn_soa_vec A9 = wcn_soa_vload(A[9] + i, wcn_vl_);
        const wcn_soa_vec A10 = wcn_soa_vload(A[10] + i, wcn_vl_);
        const wcn_soa_vec A11 = wcn_soa_vload(A[11] + i, wcn_vl_);
        const wcn_soa_vec A12 = wcn_soa_vload(A[12] + i, wcn_vl_);
        const wcn_soa_vec A13 = wcn_soa_vload(A[13] + i, wcn_vl_);
        const wcn_soa_vec A14 = wcn_soa_vload(A[14] + i, wcn_vl_);
        const wcn_soa_vec A15 = wcn_soa_vload(A[15] + i, wcn_vl_);
        const wcn_soa_vec B0 = wcn_soa_vload(B[0] + i, wcn_vl_);
        const wcn_soa_vec B1 = wcn_soa_vload(B[1] + i, wcn_vl_);
        const wcn_soa_vec B2 = wcn_soa_vload(B[2] + i, wcn_vl_);
        const wcn_soa_vec B3 = wcn_soa_vload(B[3] + i, wcn_vl_);
        const wcn_soa_vec B4 = wcn_soa_vload(B[4] + i, wcn_vl_);
        const wcn_soa_vec B5 = wcn_soa_vload(B[5] + i, wcn_vl_);
        const wcn_soa_vec B6 = wcn_soa_vload(B[6] + i, wcn_vl_);
        const wcn_soa_vec B7 = wcn_soa_vload(B[7] + i, wcn_vl_);
        const wcn_soa_vec B8 = wcn_soa_vload(B[8] + i, wcn_vl_);
        const wcn_soa_vec B9 = wcn_soa_vload(B[9] + i, wcn_vl_);
        const wcn_soa_vec B10 = wcn_soa_vload(B[10] + i, wcn_vl_);
        const wcn_soa_vec B11 = wcn_soa_vload(B[11] + i, wcn_vl_);
        const wcn_soa_vec B12 = wcn_soa_vload(B[12] + i, wcn_vl_);
        const wcn_soa_vec B13 = wcn_soa_vload(B[13] + i, wcn_vl_);
        const wcn_soa_vec B14 = wcn_soa_vload(B[14] + i, wcn_vl_);
        const wcn_soa_vec B15 = wcn_soa_vload(B[15] + i, wcn_vl_);
        wcn_soa_vstore(d[0] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A0, B0), A1, B4), A2, B8), A3, B12));
        wcn_soa_vstore(d[1] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A0, B1), A1, B5), A2, B9), A3, B13));
        wcn_soa_vstore(d[2] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A0, B2), A1, B6), A2, B10), A3, B14));
        wcn_soa_vstore(d[3] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A0, B3), A1, B7), A2, B11), A3, B15));
        wcn_soa_vstore(d[4] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A4, B0), A5, B4), A6, B8), A7, B12));
        wcn_soa_vstore(d[5] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A4, B1), A5, B5), A6, B9), A7, B13));
        wcn_soa_vstore(d[6] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A4, B2), A5, B6), A6, B10), A7, B14));
        wcn_soa_vstore(d[7] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A4, B3), A5, B7), A6, B11), A7, B15));
        wcn_soa_vstore(d[8] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A8, B0), A9, B4), A10, B8), A11, B12));
        wcn_soa_vstore(d[9] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A8, B1), A9, B5), A10, B9), A11, B13));
        wcn_soa_vstore(d[10] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A8, B2), A9, B6), A10, B10), A11, B14));
        wcn_soa_vstore(d[11] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A8, B3), A9, B7), A10, B11), A11, B15));
        wcn_soa_vstore(d[12] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A12, B0), A13, B4), A14, B8), A15, B12));
        wcn_soa_vstore(d[13] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A12, B1), A13, B5), A14, B9), A15, B13));
        wcn_soa_vstore(d[14] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A12, B2), A13, B6), A14, B10), A15, B14));
        wcn_soa_vstore(d[15] + i, wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(A12, B3), A13, B7), A14, B11), A15, B15));
    WCN_SOA_LOOP_END(i, n, {
        /* The remainder reuses the scalar library function rather than restating the index
         * arithmetic: at most three elements, so its cost is irrelevant, and a reused
         * implementation cannot drift from the reference the test compares against.
         * The call is wrapped in parentheses because an unparenthesised comma would split
         * the macro argument. */
        WMATH_TYPE(Mat4) mA;
        WMATH_TYPE(Mat4) mB;
        WMATH_TYPE(Mat4) r;
        for (int k = 0; k < 16; k++) mA.m[k] = A[k][i];
        for (int k = 0; k < 16; k++) mB.m[k] = B[k][i];
        (WMATH_MULTIPLY(Mat4)(&r, mA, mB));
        for (int k = 0; k < 16; k++) d[k][i] = r.m[k];
    });
}

static inline void wcn_soa_m4_det(float* out, const float* const* A, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec A0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec A1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec A2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec A3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec A4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec A5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec A6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec A7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec A8 = wcn_soa_vload(A[8] + i, wcn_vl_);
        const wcn_soa_vec A9 = wcn_soa_vload(A[9] + i, wcn_vl_);
        const wcn_soa_vec A10 = wcn_soa_vload(A[10] + i, wcn_vl_);
        const wcn_soa_vec A11 = wcn_soa_vload(A[11] + i, wcn_vl_);
        const wcn_soa_vec A12 = wcn_soa_vload(A[12] + i, wcn_vl_);
        const wcn_soa_vec A13 = wcn_soa_vload(A[13] + i, wcn_vl_);
        const wcn_soa_vec A14 = wcn_soa_vload(A[14] + i, wcn_vl_);
        const wcn_soa_vec A15 = wcn_soa_vload(A[15] + i, wcn_vl_);
        const wcn_soa_vec tmp0 = wcn_soa_vmul(A10, A15);
        const wcn_soa_vec tmp1 = wcn_soa_vmul(A14, A11);
        const wcn_soa_vec tmp2 = wcn_soa_vmul(A6, A15);
        const wcn_soa_vec tmp3 = wcn_soa_vmul(A14, A7);
        const wcn_soa_vec tmp4 = wcn_soa_vmul(A6, A11);
        const wcn_soa_vec tmp5 = wcn_soa_vmul(A10, A7);
        const wcn_soa_vec tmp6 = wcn_soa_vmul(A2, A15);
        const wcn_soa_vec tmp7 = wcn_soa_vmul(A14, A3);
        const wcn_soa_vec tmp8 = wcn_soa_vmul(A2, A11);
        const wcn_soa_vec tmp9 = wcn_soa_vmul(A10, A3);
        const wcn_soa_vec tmp10 = wcn_soa_vmul(A2, A7);
        const wcn_soa_vec tmp11 = wcn_soa_vmul(A6, A3);
        const wcn_soa_vec t0 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp0, A5), wcn_soa_vmul(tmp3, A9)), wcn_soa_vmul(tmp4, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp1, A5), wcn_soa_vmul(tmp2, A9)), wcn_soa_vmul(tmp5, A13)));
        const wcn_soa_vec t1 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp1, A1), wcn_soa_vmul(tmp6, A9)), wcn_soa_vmul(tmp9, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp0, A1), wcn_soa_vmul(tmp7, A9)), wcn_soa_vmul(tmp8, A13)));
        const wcn_soa_vec t2 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp2, A1), wcn_soa_vmul(tmp7, A5)), wcn_soa_vmul(tmp10, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp3, A1), wcn_soa_vmul(tmp6, A5)), wcn_soa_vmul(tmp11, A13)));
        const wcn_soa_vec t3 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp5, A1), wcn_soa_vmul(tmp8, A5)), wcn_soa_vmul(tmp11, A9)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp4, A1), wcn_soa_vmul(tmp9, A5)), wcn_soa_vmul(tmp10, A9)));
        const wcn_soa_vec acc = wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(A0, t0), wcn_soa_vmul(A4, t1)),
                                       wcn_soa_vadd(wcn_soa_vmul(A8, t2), wcn_soa_vmul(A12, t3)));
        wcn_soa_vstore(out + i, acc);
    WCN_SOA_LOOP_END(i, n, {
        WMATH_TYPE(Mat4) m;
        for (int k = 0; k < 16; k++) m.m[k] = A[k][i];
        out[i] = WMATH_DETERMINANT(Mat4)(m);
    });
}

static inline void wcn_soa_m4_inv16(float* const* d, const float* const* A, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec A0 = wcn_soa_vload(A[0] + i, wcn_vl_);
        const wcn_soa_vec A1 = wcn_soa_vload(A[1] + i, wcn_vl_);
        const wcn_soa_vec A2 = wcn_soa_vload(A[2] + i, wcn_vl_);
        const wcn_soa_vec A3 = wcn_soa_vload(A[3] + i, wcn_vl_);
        const wcn_soa_vec A4 = wcn_soa_vload(A[4] + i, wcn_vl_);
        const wcn_soa_vec A5 = wcn_soa_vload(A[5] + i, wcn_vl_);
        const wcn_soa_vec A6 = wcn_soa_vload(A[6] + i, wcn_vl_);
        const wcn_soa_vec A7 = wcn_soa_vload(A[7] + i, wcn_vl_);
        const wcn_soa_vec A8 = wcn_soa_vload(A[8] + i, wcn_vl_);
        const wcn_soa_vec A9 = wcn_soa_vload(A[9] + i, wcn_vl_);
        const wcn_soa_vec A10 = wcn_soa_vload(A[10] + i, wcn_vl_);
        const wcn_soa_vec A11 = wcn_soa_vload(A[11] + i, wcn_vl_);
        const wcn_soa_vec A12 = wcn_soa_vload(A[12] + i, wcn_vl_);
        const wcn_soa_vec A13 = wcn_soa_vload(A[13] + i, wcn_vl_);
        const wcn_soa_vec A14 = wcn_soa_vload(A[14] + i, wcn_vl_);
        const wcn_soa_vec A15 = wcn_soa_vload(A[15] + i, wcn_vl_);
        const wcn_soa_vec cf00 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A5, wcn_soa_vsub(wcn_soa_vmul(A10, A15), wcn_soa_vmul(A11, A14))), wcn_soa_vmul(A7, wcn_soa_vsub(wcn_soa_vmul(A9, A14), wcn_soa_vmul(A10, A13)))), wcn_soa_vmul(A6, wcn_soa_vsub(wcn_soa_vmul(A9, A15), wcn_soa_vmul(A11, A13))));
        const wcn_soa_vec cf01 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A4, wcn_soa_vsub(wcn_soa_vmul(A10, A15), wcn_soa_vmul(A11, A14))), wcn_soa_vmul(A7, wcn_soa_vsub(wcn_soa_vmul(A8, A14), wcn_soa_vmul(A10, A12)))), wcn_soa_vmul(A6, wcn_soa_vsub(wcn_soa_vmul(A8, A15), wcn_soa_vmul(A11, A12)))));
        const wcn_soa_vec cf02 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A4, wcn_soa_vsub(wcn_soa_vmul(A9, A15), wcn_soa_vmul(A11, A13))), wcn_soa_vmul(A7, wcn_soa_vsub(wcn_soa_vmul(A8, A13), wcn_soa_vmul(A9, A12)))), wcn_soa_vmul(A5, wcn_soa_vsub(wcn_soa_vmul(A8, A15), wcn_soa_vmul(A11, A12))));
        const wcn_soa_vec cf03 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A4, wcn_soa_vsub(wcn_soa_vmul(A9, A14), wcn_soa_vmul(A10, A13))), wcn_soa_vmul(A6, wcn_soa_vsub(wcn_soa_vmul(A8, A13), wcn_soa_vmul(A9, A12)))), wcn_soa_vmul(A5, wcn_soa_vsub(wcn_soa_vmul(A8, A14), wcn_soa_vmul(A10, A12)))));
        const wcn_soa_vec cf10 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A10, A15), wcn_soa_vmul(A11, A14))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A9, A14), wcn_soa_vmul(A10, A13)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A9, A15), wcn_soa_vmul(A11, A13)))));
        const wcn_soa_vec cf11 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A10, A15), wcn_soa_vmul(A11, A14))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A8, A14), wcn_soa_vmul(A10, A12)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A8, A15), wcn_soa_vmul(A11, A12))));
        const wcn_soa_vec cf12 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A9, A15), wcn_soa_vmul(A11, A13))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A8, A13), wcn_soa_vmul(A9, A12)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A8, A15), wcn_soa_vmul(A11, A12)))));
        const wcn_soa_vec cf13 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A9, A14), wcn_soa_vmul(A10, A13))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A8, A13), wcn_soa_vmul(A9, A12)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A8, A14), wcn_soa_vmul(A10, A12))));
        const wcn_soa_vec cf20 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A6, A15), wcn_soa_vmul(A7, A14))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A5, A14), wcn_soa_vmul(A6, A13)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A5, A15), wcn_soa_vmul(A7, A13))));
        const wcn_soa_vec cf21 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A6, A15), wcn_soa_vmul(A7, A14))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A4, A14), wcn_soa_vmul(A6, A12)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A4, A15), wcn_soa_vmul(A7, A12)))));
        const wcn_soa_vec cf22 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A5, A15), wcn_soa_vmul(A7, A13))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A4, A13), wcn_soa_vmul(A5, A12)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A4, A15), wcn_soa_vmul(A7, A12))));
        const wcn_soa_vec cf23 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A5, A14), wcn_soa_vmul(A6, A13))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A4, A13), wcn_soa_vmul(A5, A12)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A4, A14), wcn_soa_vmul(A6, A12)))));
        const wcn_soa_vec cf30 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A6, A11), wcn_soa_vmul(A7, A10))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A5, A10), wcn_soa_vmul(A6, A9)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A5, A11), wcn_soa_vmul(A7, A9)))));
        const wcn_soa_vec cf31 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A6, A11), wcn_soa_vmul(A7, A10))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A4, A10), wcn_soa_vmul(A6, A8)))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A4, A11), wcn_soa_vmul(A7, A8))));
        const wcn_soa_vec cf32 = wcn_soa_vneg(wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A5, A11), wcn_soa_vmul(A7, A9))), wcn_soa_vmul(A3, wcn_soa_vsub(wcn_soa_vmul(A4, A9), wcn_soa_vmul(A5, A8)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A4, A11), wcn_soa_vmul(A7, A8)))));
        const wcn_soa_vec cf33 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vmul(A0, wcn_soa_vsub(wcn_soa_vmul(A5, A10), wcn_soa_vmul(A6, A9))), wcn_soa_vmul(A2, wcn_soa_vsub(wcn_soa_vmul(A4, A9), wcn_soa_vmul(A5, A8)))), wcn_soa_vmul(A1, wcn_soa_vsub(wcn_soa_vmul(A4, A10), wcn_soa_vmul(A6, A8))));
        const wcn_soa_vec tmp0 = wcn_soa_vmul(A10, A15);
        const wcn_soa_vec tmp1 = wcn_soa_vmul(A14, A11);
        const wcn_soa_vec tmp2 = wcn_soa_vmul(A6, A15);
        const wcn_soa_vec tmp3 = wcn_soa_vmul(A14, A7);
        const wcn_soa_vec tmp4 = wcn_soa_vmul(A6, A11);
        const wcn_soa_vec tmp5 = wcn_soa_vmul(A10, A7);
        const wcn_soa_vec tmp6 = wcn_soa_vmul(A2, A15);
        const wcn_soa_vec tmp7 = wcn_soa_vmul(A14, A3);
        const wcn_soa_vec tmp8 = wcn_soa_vmul(A2, A11);
        const wcn_soa_vec tmp9 = wcn_soa_vmul(A10, A3);
        const wcn_soa_vec tmp10 = wcn_soa_vmul(A2, A7);
        const wcn_soa_vec tmp11 = wcn_soa_vmul(A6, A3);
        const wcn_soa_vec t0 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp0, A5), wcn_soa_vmul(tmp3, A9)), wcn_soa_vmul(tmp4, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp1, A5), wcn_soa_vmul(tmp2, A9)), wcn_soa_vmul(tmp5, A13)));
        const wcn_soa_vec t1 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp1, A1), wcn_soa_vmul(tmp6, A9)), wcn_soa_vmul(tmp9, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp0, A1), wcn_soa_vmul(tmp7, A9)), wcn_soa_vmul(tmp8, A13)));
        const wcn_soa_vec t2 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp2, A1), wcn_soa_vmul(tmp7, A5)), wcn_soa_vmul(tmp10, A13)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp3, A1), wcn_soa_vmul(tmp6, A5)), wcn_soa_vmul(tmp11, A13)));
        const wcn_soa_vec t3 = wcn_soa_vsub(wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp5, A1), wcn_soa_vmul(tmp8, A5)), wcn_soa_vmul(tmp11, A9)), wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(tmp4, A1), wcn_soa_vmul(tmp9, A5)), wcn_soa_vmul(tmp10, A9)));
        const wcn_soa_vec acc = wcn_soa_vadd(wcn_soa_vadd(wcn_soa_vmul(A0, t0), wcn_soa_vmul(A4, t1)),
                                       wcn_soa_vadd(wcn_soa_vmul(A8, t2), wcn_soa_vmul(A12, t3)));
        const wcn_soa_vec kinv = wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), acc);
        wcn_soa_vstore(d[0] + i, wcn_soa_vmul(cf00, kinv));
        wcn_soa_vstore(d[1] + i, wcn_soa_vmul(cf10, kinv));
        wcn_soa_vstore(d[2] + i, wcn_soa_vmul(cf20, kinv));
        wcn_soa_vstore(d[3] + i, wcn_soa_vmul(cf30, kinv));
        wcn_soa_vstore(d[4] + i, wcn_soa_vmul(cf01, kinv));
        wcn_soa_vstore(d[5] + i, wcn_soa_vmul(cf11, kinv));
        wcn_soa_vstore(d[6] + i, wcn_soa_vmul(cf21, kinv));
        wcn_soa_vstore(d[7] + i, wcn_soa_vmul(cf31, kinv));
        wcn_soa_vstore(d[8] + i, wcn_soa_vmul(cf02, kinv));
        wcn_soa_vstore(d[9] + i, wcn_soa_vmul(cf12, kinv));
        wcn_soa_vstore(d[10] + i, wcn_soa_vmul(cf22, kinv));
        wcn_soa_vstore(d[11] + i, wcn_soa_vmul(cf32, kinv));
        wcn_soa_vstore(d[12] + i, wcn_soa_vmul(cf03, kinv));
        wcn_soa_vstore(d[13] + i, wcn_soa_vmul(cf13, kinv));
        wcn_soa_vstore(d[14] + i, wcn_soa_vmul(cf23, kinv));
        wcn_soa_vstore(d[15] + i, wcn_soa_vmul(cf33, kinv));
    WCN_SOA_LOOP_END(i, n, {
        /* No singular guard, matching wcn_math_Mat4_inverse, which deliberately does not
         * special-case singular input (the Mat3 inverse does). */
        WMATH_TYPE(Mat4) m;
        WMATH_TYPE(Mat4) r;
        for (int k = 0; k < 16; k++) m.m[k] = A[k][i];
        (WMATH_INVERSE(Mat4)(&r, m));
        for (int k = 0; k < 16; k++) d[k][i] = r.m[k];
    });
}


/* --- 1/a with a zero guard, matching WMATH_INVERSE(Vec2) ------------------- */
/*
 * The scalar vector types disagree here: Vec2's inverse maps zero to zero, while Vec3's and
 * Vec4's divide unconditionally and yield inf. Only the Vec2xN operator uses this kernel.
 */
static inline void wcn_soa_inv_guarded(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        const wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        const wcn_soa_vec inv = wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), va);
        wcn_soa_vstore(dst + i, wcn_soa_vsel_ne(va, zero, inv, zero));
    WCN_SOA_LOOP_END(i, n, dst[i] = (a[i] != 0.0f) ? (1.0f / a[i]) : 0.0f);
}


/* --- floor / ceil / round ------------------------------------------------- */
/*
 * The round rule is WMATH_ROUND's, not C's: dst = floor(x + 0.5f), i.e. half toward
 * +infinity. C's roundf goes half away from zero and the default FP rounding is
 * half-to-even; on -2.5 this yields -2 where roundf gives -3. The rule was read off
 * wcn_math_vec2/3/4.c, not assumed.
 */
#if WCN_SOA_HAS_NATIVE_ROUND

static inline void wcn_soa_floor(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vstore(dst + i, wcn_soa_vfloor(wcn_soa_vload(a + i, wcn_vl_)));
    WCN_SOA_LOOP_END(i, n, dst[i] = floorf(a[i]));
}

static inline void wcn_soa_ceil(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        wcn_soa_vstore(dst + i, wcn_soa_vceil(wcn_soa_vload(a + i, wcn_vl_)));
    WCN_SOA_LOOP_END(i, n, dst[i] = ceilf(a[i]));
}

static inline void wcn_soa_round(float* dst, const float* a, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec va = wcn_soa_vload(a + i, wcn_vl_);
        const wcn_soa_vec half = wcn_soa_vsplat(0.5f, wcn_vl_);
        wcn_soa_vstore(dst + i, wcn_soa_vfloor(wcn_soa_vadd(va, half)));
    WCN_SOA_LOOP_END(i, n, dst[i] = floorf(a[i] + 0.5f));
}

#else /* no native vector floor, e.g. x86 below SSE4.1: scalar, as the library already does */

static inline void wcn_soa_floor(float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = floorf(a[i]);
}
static inline void wcn_soa_ceil(float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = ceilf(a[i]);
}
static inline void wcn_soa_round(float* dst, const float* a, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = floorf(a[i] + 0.5f);
}

#endif


/* --- dst = v scaled down to max_length when |v| > max_length, else v ------- */
/*
 * Matches WMATH_TRUNCATE: the comparison is against the LENGTH (a square root), not the
 * squared length -- which is what the Vec2/Vec3 scalars do, while Vec4's compares squares.
 * Each variant reproduces its own scalar type.
 */
static inline void wcn_soa_truncate2(float* dx, float* dy, const float* sx,
                                     const float* sy, float max_length, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        const wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        const wcn_soa_vec len = wcn_soa_vsqrt(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y));
        const wcn_soa_vec scale = wcn_soa_vmul(
            wcn_soa_vsplat(max_length, wcn_vl_),
            wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), len));
        const wcn_soa_vec thresh = wcn_soa_vsplat(max_length, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len, thresh, wcn_soa_vmul(x, scale), x));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len, thresh, wcn_soa_vmul(y, scale), y));
    WCN_SOA_LOOP_END(i, n, {
        const float l = sqrtf(sx[i] * sx[i] + sy[i] * sy[i]);
        if (l > max_length) {
            const float s = max_length / l;
            dx[i] = sx[i] * s;
            dy[i] = sy[i] * s;
        } else {
            dx[i] = sx[i];
            dy[i] = sy[i];
        }
    });
}

static inline void wcn_soa_truncate3(float* dx, float* dy, float* dz, const float* sx,
                                     const float* sy, const float* sz, float max_length,
                                     size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec x = wcn_soa_vload(sx + i, wcn_vl_);
        const wcn_soa_vec y = wcn_soa_vload(sy + i, wcn_vl_);
        const wcn_soa_vec z = wcn_soa_vload(sz + i, wcn_vl_);
        const wcn_soa_vec len =
            wcn_soa_vsqrt(wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vmul(x, x), y, y), z, z));
        const wcn_soa_vec scale = wcn_soa_vmul(
            wcn_soa_vsplat(max_length, wcn_vl_),
            wcn_soa_vdiv(wcn_soa_vsplat(1.0f, wcn_vl_), len));
        const wcn_soa_vec thresh = wcn_soa_vsplat(max_length, wcn_vl_);
        wcn_soa_vstore(dx + i, wcn_soa_vsel_gt(len, thresh, wcn_soa_vmul(x, scale), x));
        wcn_soa_vstore(dy + i, wcn_soa_vsel_gt(len, thresh, wcn_soa_vmul(y, scale), y));
        wcn_soa_vstore(dz + i, wcn_soa_vsel_gt(len, thresh, wcn_soa_vmul(z, scale), z));
    WCN_SOA_LOOP_END(i, n, {
        const float l = sqrtf(sx[i] * sx[i] + sy[i] * sy[i] + sz[i] * sz[i]);
        if (l > max_length) {
            const float s = max_length / l;
            dx[i] = sx[i] * s;
            dy[i] = sy[i] * s;
            dz[i] = sz[i] * s;
        } else {
            dx[i] = sx[i];
            dy[i] = sy[i];
            dz[i] = sz[i];
        }
    });
}


/* --- quaternion to rotation matrix, fused --------------------------------- */
/*
 * WMATH_CALL(Mat4, from_quat) written out plane by plane. Each of the nine outputs draws on
 * the same xx/yy/zz/xy/xz/yz/wx/wy/wz products, so they are computed once per vector and the
 * nine stores follow. The subtraction order matches the scalar source exactly
 * (`1.0f - yy - zz` is (1 - yy) - zz, not 1 - (yy + zz)).
 */
static inline void wcn_soa_quat_to_m4(float* const* d, const float* qx, const float* qy,
                                      const float* qz, const float* qw, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec x = wcn_soa_vload(qx + i, wcn_vl_);
        const wcn_soa_vec y = wcn_soa_vload(qy + i, wcn_vl_);
        const wcn_soa_vec z = wcn_soa_vload(qz + i, wcn_vl_);
        const wcn_soa_vec w = wcn_soa_vload(qw + i, wcn_vl_);
        /* The scalar doubles each component first (x2 = x + x) and builds the products from
         * it, so every product below carries a factor of two. Using plain x*x here yields
         * exactly half the correct off-diagonal entries -- which is what the test caught. */
        const wcn_soa_vec two = wcn_soa_vsplat(2.0f, wcn_vl_);
        const wcn_soa_vec x2 = wcn_soa_vmul(x, two);
        const wcn_soa_vec y2 = wcn_soa_vmul(y, two);
        const wcn_soa_vec z2 = wcn_soa_vmul(z, two);
        const wcn_soa_vec xx = wcn_soa_vmul(x, x2);
        const wcn_soa_vec xy = wcn_soa_vmul(x, y2);
        const wcn_soa_vec xz = wcn_soa_vmul(x, z2);
        const wcn_soa_vec yy = wcn_soa_vmul(y, y2);
        const wcn_soa_vec yz = wcn_soa_vmul(y, z2);
        const wcn_soa_vec zz = wcn_soa_vmul(z, z2);
        const wcn_soa_vec wx = wcn_soa_vmul(w, x2);
        const wcn_soa_vec wy = wcn_soa_vmul(w, y2);
        const wcn_soa_vec wz = wcn_soa_vmul(w, z2);
        const wcn_soa_vec one = wcn_soa_vsplat(1.0f, wcn_vl_);
        const wcn_soa_vec zero = wcn_soa_vsplat(0.0f, wcn_vl_);
        wcn_soa_vstore(d[0] + i, wcn_soa_vsub(wcn_soa_vsub(one, yy), zz));
        wcn_soa_vstore(d[1] + i, wcn_soa_vadd(xy, wz));
        wcn_soa_vstore(d[2] + i, wcn_soa_vsub(xz, wy));
        wcn_soa_vstore(d[3] + i, zero);
        wcn_soa_vstore(d[4] + i, wcn_soa_vsub(xy, wz));
        wcn_soa_vstore(d[5] + i, wcn_soa_vsub(wcn_soa_vsub(one, xx), zz));
        wcn_soa_vstore(d[6] + i, wcn_soa_vadd(yz, wx));
        wcn_soa_vstore(d[7] + i, zero);
        wcn_soa_vstore(d[8] + i, wcn_soa_vadd(xz, wy));
        wcn_soa_vstore(d[9] + i, wcn_soa_vsub(yz, wx));
        wcn_soa_vstore(d[10] + i, wcn_soa_vsub(wcn_soa_vsub(one, xx), yy));
        wcn_soa_vstore(d[11] + i, zero);
        wcn_soa_vstore(d[12] + i, zero);
        wcn_soa_vstore(d[13] + i, zero);
        wcn_soa_vstore(d[14] + i, zero);
        wcn_soa_vstore(d[15] + i, one);
    WCN_SOA_LOOP_END(i, n, {
        WMATH_TYPE(Quat) q;
        WMATH_TYPE(Mat4) r;
        q.v[0] = qx[i];
        q.v[1] = qy[i];
        q.v[2] = qz[i];
        q.v[3] = qw[i];
        WMATH_CALL(Mat4, from_quat)(&r, q);
        for (int k = 0; k < 16; k++) d[k][i] = r.m[k];
    });
}


/* --- dst = m * T(v): the fourth row of a translation multiply ------------------ */
/*
 * WMATH_CALL(Mat4, translate) is a matrix multiply, not an in-place offset. Since T(v) only
 * carries the translation in its last row, planes 0..11 are copied and planes 12..15 become
 * m[0]*vx + m[4]*vy + m[8]*vz + m[12] and its analogues. With v per element those are plane
 * sums, so the row is computed with fma rather than looping over elements.
 */
static inline void wcn_soa_m4_translate(float* const* d, const float* const* m,
                                        const float* vx, const float* vy, const float* vz,
                                        size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec X = wcn_soa_vload(vx + i, wcn_vl_);
        const wcn_soa_vec Y = wcn_soa_vload(vy + i, wcn_vl_);
        const wcn_soa_vec Z = wcn_soa_vload(vz + i, wcn_vl_);
        for (int c = 0; c < 4; c++) {
            const wcn_soa_vec a = wcn_soa_vload(m[c] + i, wcn_vl_);
            const wcn_soa_vec b = wcn_soa_vload(m[4 + c] + i, wcn_vl_);
            const wcn_soa_vec e = wcn_soa_vload(m[8 + c] + i, wcn_vl_);
            const wcn_soa_vec base = wcn_soa_vload(m[12 + c] + i, wcn_vl_);
            const wcn_soa_vec r =
                wcn_soa_vfma(wcn_soa_vfma(wcn_soa_vfma(base, a, X), b, Y), e, Z);
            wcn_soa_vstore(d[12 + c] + i, r);
        }
    WCN_SOA_LOOP_END(i, n, {
        const float vx0 = vx[i];
        const float vy0 = vy[i];
        const float vz0 = vz[i];
        for (int c = 0; c < 4; c++) {
            d[12 + c][i] = m[c][i] * vx0 + m[4 + c][i] * vy0 + m[8 + c][i] * vz0 + m[12 + c][i];
        }
    });
}


/* --- dst = m * T2(v): the third row of a 2D translation multiply --------------- */
/*
 * WMATH_CALL(Mat3, translate) is a matrix multiply like the Mat4 form, and only row 2
 * changes: planes 6,7,8 become m[0]*vx + m[3]*vy + m[6] and its row-mates. With v per
 * element those are plane sums, so this is vectorised.
 */
static inline void wcn_soa_m3_translate(float* const* d, const float* const* m,
                                        const float* vx, const float* vy, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec X = wcn_soa_vload(vx + i, wcn_vl_);
        const wcn_soa_vec Y = wcn_soa_vload(vy + i, wcn_vl_);
        for (int c = 0; c < 3; c++) {
            const wcn_soa_vec a = wcn_soa_vload(m[c] + i, wcn_vl_);
            const wcn_soa_vec b = wcn_soa_vload(m[3 + c] + i, wcn_vl_);
            const wcn_soa_vec base = wcn_soa_vload(m[6 + c] + i, wcn_vl_);
            const wcn_soa_vec r = wcn_soa_vfma(wcn_soa_vfma(base, a, X), b, Y);
            wcn_soa_vstore(d[6 + c] + i, r);
        }
    WCN_SOA_LOOP_END(i, n, {
        const float vx0 = vx[i];
        const float vy0 = vy[i];
        for (int c = 0; c < 3; c++) {
            d[6 + c][i] = m[c][i] * vx0 + m[3 + c][i] * vy0 + m[6 + c][i];
        }
    });
}


/* --- z = ax*by - ay*bx, the 2D cross product's only non-zero component -------- */
/*
 * WMATH_CROSS(Vec2) produces a Vec3 whose x and y are exactly zero; only z is computed. Kept
 * as a kernel rather than two composed multiplies so the subtraction order is visible in one
 * place -- the sign of a cross product is the whole point of it.
 */
static inline void wcn_soa_cross2_z(float* dz, const float* ax, const float* ay,
                                    const float* bx, const float* by, size_t n) {
    WCN_SOA_LOOP_BEGIN(i, n)
        const wcn_soa_vec a0 = wcn_soa_vload(ax + i, wcn_vl_);
        const wcn_soa_vec a1 = wcn_soa_vload(ay + i, wcn_vl_);
        const wcn_soa_vec b0 = wcn_soa_vload(bx + i, wcn_vl_);
        const wcn_soa_vec b1 = wcn_soa_vload(by + i, wcn_vl_);
        wcn_soa_vstore(dz + i, wcn_soa_vsub(wcn_soa_vmul(a0, b1), wcn_soa_vmul(a1, b0)));
    WCN_SOA_LOOP_END(i, n, dz[i] = ax[i] * by[i] - ay[i] * bx[i]);
}

#endif /* WCN_MATH_SOA_KERNELS_H */