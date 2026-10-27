#include "WCN/WCN_MATH_SOA.h"
#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include "../common/wcn_math_soa_kernels.h"
#include <assert.h>
static inline void __prev_clangd_check() {
wcn_math_get_epsilon();
}
/* Mat4xN inverse.
 *
 * The algorithm is the pair-sharing adjugate from wcn_math_mat4.c, restated so it
 * works on four MOVING rows rather than four named ones: each lane holds one row
 * of one matrix, and the three shifted copies come from the shuffle. Writing it
 * this way means the same source serves both the AVX2 two-at-a-time path and the
 * portable one-at-a-time path.
 *
 * The key structural fact that keeps this cheap: only three row pairs are ever
 * needed.
 *     cofactor rows 0 and 1 -> (row2, row3)
 *     cofactor row 2        -> (row1, row3)
 *     cofactor row 3        -> (row1, row2)
 * so there are nine minors, not thirty-six, and each cofactor row is one
 * three-term combination with the remaining row.
 *
 * Two index facts that are easy to get wrong (both cost time earlier):
 *   - the vector built for cofactor row c is ROW c of the cofactor matrix, hence
 *     COLUMN c of the inverse; a transpose is required on the way out.
 *   - det is hsum(row0 * signed cofactor row 0). Contracting row c against
 *     cofactor row c is wrong.
 */

#if !defined(WMATH_DISABLE_SIMD) && (defined(WCN_HAS_AVX2) || WCN_HAS_AVX512)

#include <immintrin.h>

/* Shift lanes right by d within each 128-bit half (d = 1,2,3).
 * _mm256_shuffle_ps operates independently on the two halves, which is exactly
 * the semantics needed: half 0 is matrix A, half 1 is matrix B. */
#define WCN_SH8(v, d)                                                                 \
    _mm256_shuffle_ps((v), (v),                                                       \
                      ((d) == 1) ? _MM_SHUFFLE(0, 3, 2, 1)                            \
                                 : (((d) == 2) ? _MM_SHUFFLE(1, 0, 3, 2) : _MM_SHUFFLE(2, 1, 0, 3)))

#define WCN_COMB8(s1, s2, s3, m0, m1, m2)                                             \
    _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps((s1), (m0)), _mm256_mul_ps((s2), (m1))), \
                  _mm256_mul_ps((s3), (m2)))


/* Four matrices per zmm. Blocks 0..3 of each register are matrices at+0..at+3, kept
 * independent because every shuffle and arithmetic op here is per-128-bit-block. */
static void wcn_m4xn_store_quad(__m512 i0, __m512 i1, __m512 i2, __m512 i3, float* dst,
                                size_t at) {
    /* i_c is column c of each result; transpose within each 128-bit block so the four
     * columns become four rows, then each block is one matrix's row set. */
    const __m512 t0 = _mm512_unpacklo_ps(i0, i1);
    const __m512 t1 = _mm512_unpackhi_ps(i0, i1);
    const __m512 t2 = _mm512_unpacklo_ps(i2, i3);
    const __m512 t3 = _mm512_unpackhi_ps(i2, i3);

    const __m512 p0 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 p1 = _mm512_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m512 p2 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m512 p3 = _mm512_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

    /* Unrolled: the extractf32x4 lane index is encoded in the instruction, so it must
     * be a compile-time constant rather than a loop variable. */
    float* d0 = dst + 16 * at;
    float* d1 = d0 + 16;
    float* d2 = d1 + 16;
    float* d3 = d2 + 16;

    _mm_storeu_ps(d0 + 0, _mm512_extractf32x4_ps(p0, 0));
    _mm_storeu_ps(d0 + 4, _mm512_extractf32x4_ps(p1, 0));
    _mm_storeu_ps(d0 + 8, _mm512_extractf32x4_ps(p2, 0));
    _mm_storeu_ps(d0 + 12, _mm512_extractf32x4_ps(p3, 0));

    _mm_storeu_ps(d1 + 0, _mm512_extractf32x4_ps(p0, 1));
    _mm_storeu_ps(d1 + 4, _mm512_extractf32x4_ps(p1, 1));
    _mm_storeu_ps(d1 + 8, _mm512_extractf32x4_ps(p2, 1));
    _mm_storeu_ps(d1 + 12, _mm512_extractf32x4_ps(p3, 1));

    _mm_storeu_ps(d2 + 0, _mm512_extractf32x4_ps(p0, 2));
    _mm_storeu_ps(d2 + 4, _mm512_extractf32x4_ps(p1, 2));
    _mm_storeu_ps(d2 + 8, _mm512_extractf32x4_ps(p2, 2));
    _mm_storeu_ps(d2 + 12, _mm512_extractf32x4_ps(p3, 2));

    _mm_storeu_ps(d3 + 0, _mm512_extractf32x4_ps(p0, 3));
    _mm_storeu_ps(d3 + 4, _mm512_extractf32x4_ps(p1, 3));
    _mm_storeu_ps(d3 + 8, _mm512_extractf32x4_ps(p2, 3));
    _mm_storeu_ps(d3 + 12, _mm512_extractf32x4_ps(p3, 3));
}

#define WCN_SH16(v, d)                                                                 \
    _mm512_shuffle_ps((v), (v),                                                        \
                      ((d) == 1) ? _MM_SHUFFLE(0, 3, 2, 1)                              \
                                 : (((d) == 2) ? _MM_SHUFFLE(1, 0, 3, 2) : _MM_SHUFFLE(2, 1, 0, 3)))

#define WCN_COMB16(s1, s2, s3, m0, m1, m2)                                             \
    _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps((s1), (m0)), _mm512_mul_ps((s2), (m1))), \
                  _mm512_mul_ps((s3), (m2)))

static void wcn_m4xn_quad(__m512 r0, __m512 r1, __m512 r2, __m512 r3, float* dst,
                          size_t at) {
    const __m512 sgE = _mm512_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                                      1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    const __m512 sgO = _mm512_sub_ps(_mm512_setzero_ps(), sgE);

    const __m512 R0a = WCN_SH16(r0, 1), R0b = WCN_SH16(r0, 2), R0c = WCN_SH16(r0, 3);
    const __m512 R1a = WCN_SH16(r1, 1), R1b = WCN_SH16(r1, 2), R1c = WCN_SH16(r1, 3);
    const __m512 R2a = WCN_SH16(r2, 1), R2b = WCN_SH16(r2, 2), R2c = WCN_SH16(r2, 3);
    const __m512 R3a = WCN_SH16(r3, 1), R3b = WCN_SH16(r3, 2), R3c = WCN_SH16(r3, 3);

    const __m512 p0 = _mm512_sub_ps(_mm512_mul_ps(R2b, R3c), _mm512_mul_ps(R2c, R3b));
    const __m512 p1 = _mm512_sub_ps(_mm512_mul_ps(R2a, R3c), _mm512_mul_ps(R2c, R3a));
    const __m512 p2 = _mm512_sub_ps(_mm512_mul_ps(R2a, R3b), _mm512_mul_ps(R2b, R3a));

    const __m512 u0 = WCN_COMB16(R1a, R1b, R1c, p0, p1, p2);
    const __m512 u1 = WCN_COMB16(R0a, R0b, R0c, p0, p1, p2);

    const __m512 C0 = _mm512_mul_ps(u0, sgE);
    const __m512 C1 = _mm512_mul_ps(u1, sgO);

    /* det = hsum within each 128-bit block of (row0 * signed cofactor row 0) */
    const __m512 d0 = _mm512_mul_ps(r0, C0);
    const __m512 h1 = _mm512_add_ps(d0, _mm512_shuffle_ps(d0, d0, _MM_SHUFFLE(2, 3, 0, 1)));
    const __m512 h2 = _mm512_add_ps(h1, _mm512_shuffle_ps(h1, h1, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m512 dinv = _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_shuffle_ps(h2, h2, 0));

    const __m512 i0 = _mm512_mul_ps(C0, dinv);
    const __m512 i1 = _mm512_mul_ps(C1, dinv);

    const __m512 q0 = _mm512_sub_ps(_mm512_mul_ps(R1b, R3c), _mm512_mul_ps(R1c, R3b));
    const __m512 q1 = _mm512_sub_ps(_mm512_mul_ps(R1a, R3c), _mm512_mul_ps(R1c, R3a));
    const __m512 q2 = _mm512_sub_ps(_mm512_mul_ps(R1a, R3b), _mm512_mul_ps(R1b, R3a));
    const __m512 i2 =
        _mm512_mul_ps(_mm512_mul_ps(WCN_COMB16(R0a, R0b, R0c, q0, q1, q2), sgE), dinv);

    const __m512 s0 = _mm512_sub_ps(_mm512_mul_ps(R1b, R2c), _mm512_mul_ps(R1c, R2b));
    const __m512 s1 = _mm512_sub_ps(_mm512_mul_ps(R1a, R2c), _mm512_mul_ps(R1c, R2a));
    const __m512 s2 = _mm512_sub_ps(_mm512_mul_ps(R1a, R2b), _mm512_mul_ps(R1b, R2a));
    const __m512 i3 =
        _mm512_mul_ps(_mm512_mul_ps(WCN_COMB16(R0a, R0b, R0c, s0, s1, s2), sgO), dinv);

    wcn_m4xn_store_quad(i0, i1, i2, i3, dst, at);
}

#undef WCN_COMB16
#undef WCN_SH16

/* Transpose the 4x4 formed by four 256-bit values whose halves are two separate
 * matrices, then store both into SoA planes. */
static void wcn_m4xn_store(__m256 i0, __m256 i1, __m256 i2, __m256 i3, float* dst,
                           size_t at) {
    /* i_c is column c of each result. unpack/movelh act within each 128-bit half,
     * so the transpose turns the four columns into four rows and the halves stay
     * separate: low half is matrix `at`, high half is `at + 1`. */
    const __m256 t0 = _mm256_unpacklo_ps(i0, i1);
    const __m256 t1 = _mm256_unpackhi_ps(i0, i1);
    const __m256 t2 = _mm256_unpacklo_ps(i2, i3);
    const __m256 t3 = _mm256_unpackhi_ps(i2, i3);

    float* pa = dst + 16 * at;
    float* pb = dst + 16 * (at + 1);

    /* 256-bit movelh/movehl do not exist; shuffle_ps with a 128-bit pattern does
     * the same thing per half, which is what we want because the halves are
     * different matrices:
     *   movelh(a,b) == shuffle_ps(a, b, (1,0,1,0))
     *   movehl(a,b) == shuffle_ps(a, b, (3,2,3,2)) */
    const __m256 p0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 p1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 p2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 p3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));

    _mm_storeu_ps(pa + 0, _mm256_castps256_ps128(p0));
    _mm_storeu_ps(pa + 4, _mm256_castps256_ps128(p1));
    _mm_storeu_ps(pa + 8, _mm256_castps256_ps128(p2));
    _mm_storeu_ps(pa + 12, _mm256_castps256_ps128(p3));

    _mm_storeu_ps(pb + 0, _mm256_extractf128_ps(p0, 1));
    _mm_storeu_ps(pb + 4, _mm256_extractf128_ps(p1, 1));
    _mm_storeu_ps(pb + 8, _mm256_extractf128_ps(p2, 1));
    _mm_storeu_ps(pb + 12, _mm256_extractf128_ps(p3, 1));
}

/* One pair of matrices. Rows are gathered straight from the SoA planes; the two
 * matrices' rows are combined into the two halves of each ymm. */
static void wcn_m4xn_pair(__m256 r0, __m256 r1, __m256 r2, __m256 r3, float* dst, size_t at) {
    const __m256 sgE = _mm256_setr_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    const __m256 sgO = _mm256_sub_ps(_mm256_setzero_ps(), sgE);

    const __m256 R0a = WCN_SH8(r0, 1), R0b = WCN_SH8(r0, 2), R0c = WCN_SH8(r0, 3);
    const __m256 R1a = WCN_SH8(r1, 1), R1b = WCN_SH8(r1, 2), R1c = WCN_SH8(r1, 3);
    const __m256 R2a = WCN_SH8(r2, 1), R2b = WCN_SH8(r2, 2), R2c = WCN_SH8(r2, 3);
    const __m256 R3a = WCN_SH8(r3, 1), R3b = WCN_SH8(r3, 2), R3c = WCN_SH8(r3, 3);

    const __m256 p0 = _mm256_sub_ps(_mm256_mul_ps(R2b, R3c), _mm256_mul_ps(R2c, R3b));
    const __m256 p1 = _mm256_sub_ps(_mm256_mul_ps(R2a, R3c), _mm256_mul_ps(R2c, R3a));
    const __m256 p2 = _mm256_sub_ps(_mm256_mul_ps(R2a, R3b), _mm256_mul_ps(R2b, R3a));

    const __m256 u0 = WCN_COMB8(R1a, R1b, R1c, p0, p1, p2);
    const __m256 u1 = WCN_COMB8(R0a, R0b, R0c, p0, p1, p2);

    const __m256 C0 = _mm256_mul_ps(u0, sgE);
    const __m256 C1 = _mm256_mul_ps(u1, sgO);

    /* det = hsum within each 128-bit half of (row0 * signed cofactor row 0). */
    const __m256 d0 = _mm256_mul_ps(r0, C0);
    const __m256 h1 = _mm256_add_ps(d0, _mm256_shuffle_ps(d0, d0, _MM_SHUFFLE(2, 3, 0, 1)));
    const __m256 h2 = _mm256_add_ps(h1, _mm256_shuffle_ps(h1, h1, _MM_SHUFFLE(1, 0, 3, 2)));
    /* h2 lane 0 of each 128-bit half already holds that half's complete sum.
     * shuffle_ps applies the lane pattern per half, so this broadcasts lane 0
     * within each matrix independently -- no cross-half mixing. */
    const __m256 dinv = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_shuffle_ps(h2, h2, 0));

    const __m256 i0 = _mm256_mul_ps(C0, dinv);
    const __m256 i1 = _mm256_mul_ps(C1, dinv);

    const __m256 q0 = _mm256_sub_ps(_mm256_mul_ps(R1b, R3c), _mm256_mul_ps(R1c, R3b));
    const __m256 q1 = _mm256_sub_ps(_mm256_mul_ps(R1a, R3c), _mm256_mul_ps(R1c, R3a));
    const __m256 q2 = _mm256_sub_ps(_mm256_mul_ps(R1a, R3b), _mm256_mul_ps(R1b, R3a));
    const __m256 i2 = _mm256_mul_ps(_mm256_mul_ps(WCN_COMB8(R0a, R0b, R0c, q0, q1, q2), sgE),
                                    dinv);

    const __m256 s0 = _mm256_sub_ps(_mm256_mul_ps(R1b, R2c), _mm256_mul_ps(R1c, R2b));
    const __m256 s1 = _mm256_sub_ps(_mm256_mul_ps(R1a, R2c), _mm256_mul_ps(R1c, R2a));
    const __m256 s2 = _mm256_sub_ps(_mm256_mul_ps(R1a, R2b), _mm256_mul_ps(R1b, R2a));
    const __m256 i3 = _mm256_mul_ps(_mm256_mul_ps(WCN_COMB8(R0a, R0b, R0c, s0, s1, s2), sgO),
                                    dinv);

    wcn_m4xn_store(i0, i1, i2, i3, dst, at);
}

void wcn_math_Mat4xN_inverse(float* dst, const float* src, size_t count) {
    if (!dst || !src || count == 0) return;

    size_t k = 0;

#if WCN_HAS_AVX512
    /* Four matrices per zmm. This is where the width actually is: a 128-bit register
     * holds exactly one 4x4 row, so SSE/NEON/WASM/RVV/LSX cannot share a register
     * between two matrices, whereas a zmm holds four of them in four independent
     * 128-bit blocks. The gap the loop has to cover is the gather: four loads plus
     * three insertf32x4 per row, against AVX2's two loads plus one insertf128. */
    for (; k + 4 <= count; k += 4) {
        const float* p0 = src + 16 * k;
        const float* p1 = p0 + 16;
        const float* p2 = p1 + 16;
        const float* p3 = p2 + 16;

#define WCN_GATHER4(off)                                                               \
    _mm512_insertf32x4(                                                                \
        _mm512_insertf32x4(                                                            \
            _mm512_insertf32x4(_mm512_castps128_ps512(_mm_loadu_ps(p0 + (off))),        \
                               _mm_loadu_ps(p1 + (off)), 1),                            \
            _mm_loadu_ps(p2 + (off)), 2),                                              \
        _mm_loadu_ps(p3 + (off)), 3)

        const __m512 r0 = WCN_GATHER4(0);
        const __m512 r1 = WCN_GATHER4(4);
        const __m512 r2 = WCN_GATHER4(8);
        const __m512 r3 = WCN_GATHER4(12);

#undef WCN_GATHER4

        wcn_m4xn_quad(r0, r1, r2, r3, dst, k);
    }
#endif

#if defined(WCN_HAS_AVX2)
    /* Two matrices per ymm: matrix k in the low 128 bits, k+1 in the high.
     * loadu + insertf128 is two loads and no shuffling, because a row of a
     * row-major matrix is already four consecutive floats. Measured 6.05-6.64
     * ns/matrix against 12.50 for the per-matrix loop. */
    for (; k + 2 <= count; k += 2) {
        const float* a = src + 16 * k;
        const float* b = src + 16 * (k + 1);

        const __m256 r0 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(a + 0)),
                                               _mm_loadu_ps(b + 0), 1);
        const __m256 r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(a + 4)),
                                               _mm_loadu_ps(b + 4), 1);
        const __m256 r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(a + 8)),
                                               _mm_loadu_ps(b + 8), 1);
        const __m256 r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(a + 12)),
                                               _mm_loadu_ps(b + 12), 1);

        wcn_m4xn_pair(r0, r1, r2, r3, dst, k);
    }
#endif

    /* Remainder, and the whole job on backends without a 256-bit register.
     *
     * WMATH_INVERSE(Mat4) is itself a per-platform SIMD branch, so this loop gives
     * SSE / NEON / WASM SIMD / RVV / LSX their vectorised inverse. What it must NOT
     * do is round-trip through a temporary: the previous form copied all sixteen
     * floats out of `o` into dst with a scalar loop, which measured 23.05 ns/matrix
     * against 12.50 ns/matrix for a plain per-matrix loop -- i.e. the helper was
     * 1.8x slower than not using it.
     *
     * Mat4 is a union whose layout is exactly float m[16] with no alignment
     * attribute, and dst + 16*k is sixteen contiguous floats, so the destination can
     * be handed over directly and the sixteen stores disappear. The input copy stays:
     * the parameter is by value, so a copy is required by the ABI regardless and the
     * compiler emits it as one struct load.
     *
     * There is no wider form to reach for on these backends: a 128-bit register holds
     * exactly one 4x4 row, so two matrices cannot share a register the way AVX2's two
     * ymm halves let them. */
    for (; k < count; k++) {
        /* Struct-assign the input rather than looping over its sixteen floats: the
         * measured difference is 16.60 -> 13.28 ns/matrix, and it matches the plain
         * per-matrix loop the benchmark uses as its reference (13.09). */
        const WMATH_TYPE(Mat4) a = *(const WMATH_TYPE(Mat4)*)(const void*)(src + 16 * k);
        WMATH_INVERSE(Mat4)((WMATH_TYPE(Mat4)*)(void*)(dst + 16 * k), a);
    }
}

#undef WCN_COMB8
#undef WCN_SH8

#else

void wcn_math_Mat4xN_inverse(float* dst, const float* src, size_t count) {
    if (!dst || !src || count == 0) return;
    /* Same direct-pointer form as the remainder loop above; see the note there for
     * why the destination does not need a temporary. */
    for (size_t k = 0; k < count; k++) {
        const WMATH_TYPE(Mat4) a = *(const WMATH_TYPE(Mat4)*)(const void*)(src + 16 * k);
        WMATH_INVERSE(Mat4)((WMATH_TYPE(Mat4)*)(void*)(dst + 16 * k), a);
    }
}

#endif


/* ==========================================================================
 * SoA operators for Mat4xN (sixteen planes)
 *
 * Plane layout: Mat4xN has sixteen planes and plane p is Mat4's m[p] directly --
 * row-major, no padding. This is the one xN type where the plane index and the scalar m[]
 * index coincide, so unlike Mat3xN there is no mapping to get wrong; the header comment on
 * the type says the same ("m[0..15] 映射 4x4 矩阵的 16 个分量").
 * ========================================================================== */

static inline int m4xn_ok2(const WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a,
                           const WMATH_TYPE(Mat4xN)* b) {
    if (!dst || !a || !b) return 0;
    if (dst->count != a->count || dst->count != b->count) return 0;
    for (int p = 0; p < 16; p++)
        if (!dst->m[p] || !a->m[p] || !b->m[p]) return 0;
    return 1;
}
static inline int m4xn_ok1(const WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    if (!dst || !a || dst->count != a->count) return 0;
    for (int p = 0; p < 16; p++)
        if (!dst->m[p] || !a->m[p]) return 0;
    return 1;
}

/* ---------------------------------------------------------------- memory */

WMATH_TYPE(Mat4xN)* wcn_math_Mat4xN_malloc(const size_t count) {
    WMATH_TYPE(Mat4xN)* v = (WMATH_TYPE(Mat4xN)*)malloc(sizeof(WMATH_TYPE(Mat4xN)));
    if (!v) return nullptr;
    v->count = count;
    for (int p = 0; p < 16; p++) {
        /* wcn_aligned_alloc takes (bytes, alignment); reversed it allocates 16 bytes and
         * corrupts on the first store (LOGIC_BUGS.md). */
        v->m[p] = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    }
    return v;
}

void wcn_math_Mat4xN_free(WMATH_TYPE(Mat4xN)* ptr) {
    if (ptr == nullptr) return;
    /* Planes only. Mat4xN instances are filled in by callers -- the Flat conversions do
     * exactly that -- so freeing the shell here would corrupt the heap on a
     * stack-allocated one. Shells from _malloc go through _destroy. */
    for (int p = 0; p < 16; p++) {
        if (ptr->m[p]) { wcn_aligned_free(ptr->m[p]); ptr->m[p] = nullptr; }
    }
    ptr->count = 0;
}

void wcn_math_Mat4xN_destroy(WMATH_TYPE(Mat4xN)* ptr) {
    if (ptr == nullptr) return;
    wcn_math_Mat4xN_free(ptr);
    free(ptr);
}

float* wcn_math_Mat4xN_get_element_array(WMATH_TYPE(Mat4xN)* ptr, size_t plane) {
    if (!ptr || plane >= 16) return nullptr;
    return ptr->m[plane];
}

size_t wcn_math_Mat4xN_get_count(const WMATH_TYPE(Mat4xN)* ptr) {
    return ptr ? ptr->count : 0;
}

bool wcn_math_Mat4xN_get(WMATH_TYPE(Mat4)* out, const WMATH_TYPE(Mat4xN)* src, size_t index) {
    if (!out || !src || index >= src->count) return false;
    for (int k = 0; k < 16; k++) out->m[k] = src->m[k][index];
    return true;
}

bool wcn_math_Mat4xN_set(WMATH_TYPE(Mat4xN)* dst, const size_t index,
                         const WMATH_TYPE(Mat4)* input) {
    if (!dst || !input || index >= dst->count) return false;
    for (int k = 0; k < 16; k++) dst->m[k][index] = input->m[k];
    return true;
}

/* ------------------------------------------------------------- element-wise */

void WMATH_ADD(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a,
                       const WMATH_TYPE(Mat4xN)* b) {
    assert(m4xn_ok2(dst, a, b));
    if (!m4xn_ok2(dst, a, b)) return;
    for (int p = 0; p < 16; p++) wcn_soa_add(dst->m[p], a->m[p], b->m[p], a->count);
}

void WMATH_SUB(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a,
                       const WMATH_TYPE(Mat4xN)* b) {
    assert(m4xn_ok2(dst, a, b));
    if (!m4xn_ok2(dst, a, b)) return;
    for (int p = 0; p < 16; p++) wcn_soa_sub(dst->m[p], a->m[p], b->m[p], a->count);
}

void WMATH_MULTIPLY_SCALAR(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a,
                                   const float s) {
    assert(m4xn_ok1(dst, a));
    if (!m4xn_ok1(dst, a)) return;
    for (int p = 0; p < 16; p++) wcn_soa_mul_s(dst->m[p], a->m[p], s, a->count);
}

void WMATH_NEGATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    assert(m4xn_ok1(dst, a));
    if (!m4xn_ok1(dst, a)) return;
    for (int p = 0; p < 16; p++) wcn_soa_neg(dst->m[p], a->m[p], a->count);
}

void WMATH_MULTIPLY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a,
                            const WMATH_TYPE(Mat4xN)* b) {
    assert(m4xn_ok2(dst, a, b));
    if (!m4xn_ok2(dst, a, b)) return;
    /* Fused: each of the sixteen outputs is a sum of four cross-plane products. */
    wcn_soa_m4_mul16(dst->m, WCN_SOA_RO(a->m), WCN_SOA_RO(b->m), a->count);
}

void WMATH_TRANSPOSE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    assert(m4xn_ok1(dst, a));
    /* dst must not alias a: the six off-diagonal plane pairs are exchanged, so copying one
     * before reading its partner would lose a value. The scalar WMATH_TRANSPOSE(Mat4) takes
     * its source by value and is safe in place; this pointer form is not. The Mat3xN
     * transpose carries the same contract for the same reason. */
    assert(dst != a);
    if (!m4xn_ok1(dst, a) || dst == a) return;
    for (int p = 0; p < 16; p++) {
        const int row = p / 4;
        const int col = p % 4;
        const int src = col * 4 + row; /* the transposed partner */
        wcn_soa_copy(dst->m[p], a->m[src], a->count);
    }
}

void WMATH_DETERMINANT(Mat4xN)(float* out, const WMATH_TYPE(Mat4xN)* a) {
    if (!out || !a || !a->m[0]) return;
    wcn_soa_m4_det(out, WCN_SOA_RO(a->m), a->count);
}

void WMATH_CALL(Mat4xN, inverse_soa)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    assert(m4xn_ok1(dst, a));
    if (!m4xn_ok1(dst, a)) return;
    /* No singular guard, matching wcn_math_Mat4_inverse, which deliberately does not
     * special-case singular input. (Mat3 and Mat3xN do.) */
    wcn_soa_m4_inv16(dst->m, WCN_SOA_RO(a->m), a->count);
}

/* ------------------------------------------------------------- whole-container */

void WMATH_COPY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    assert(m4xn_ok1(dst, a));
    if (!m4xn_ok1(dst, a)) return;
    for (int p = 0; p < 16; p++) wcn_soa_copy(dst->m[p], a->m[p], a->count);
}

void WMATH_ZERO(Mat4xN)(WMATH_TYPE(Mat4xN)* dst) {
    if (!dst || !dst->m[0]) return;
    for (int p = 0; p < 16; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
}

void WMATH_IDENTITY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst) {
    if (!dst || !dst->m[0]) return;
    /* WMATH_IDENTITY(Mat4) sets m[0], m[5], m[10], m[15] to 1 and the rest to 0. Because
     * the plane index equals the m[] index for this type, those are planes 0, 5, 10, 15. */
    for (int p = 0; p < 16; p++) {
        wcn_soa_fill(dst->m[p], (p == 0 || p == 5 || p == 10 || p == 15) ? 1.0f : 0.0f,
                     dst->count);
    }
}

bool WMATH_EQUALS(Mat4xN)(const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    for (size_t i = 0; i < a->count; i++) {
        for (int p = 0; p < 16; p++) {
            if (a->m[p][i] != b->m[p][i]) return false;
        }
    }
    return true;
}

bool WMATH_EQUALS_APPROXIMATELY(Mat4xN)(const WMATH_TYPE(Mat4xN)* a,
                                        const WMATH_TYPE(Mat4xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    const float ep = wcn_math_get_epsilon();
    for (size_t i = 0; i < a->count; i++) {
        for (int p = 0; p < 16; p++) {
            if (!(fabsf(a->m[p][i] - b->m[p][i]) < ep)) return false;
        }
    }
    return true;
}

void WMATH_CREATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_CREATE_TYPE(Mat4) c) {
    if (!dst || !dst->m[0]) return;
    /* fills every element from the named CREATE_TYPE fields; the plane/field mapping is
     * plane p = row*4 + col, i.e. m_<row><col>. */
    wcn_soa_fill(dst->m[0], c.m_00, dst->count);
    wcn_soa_fill(dst->m[1], c.m_01, dst->count);
    wcn_soa_fill(dst->m[2], c.m_02, dst->count);
    wcn_soa_fill(dst->m[3], c.m_03, dst->count);
    wcn_soa_fill(dst->m[4], c.m_10, dst->count);
    wcn_soa_fill(dst->m[5], c.m_11, dst->count);
    wcn_soa_fill(dst->m[6], c.m_12, dst->count);
    wcn_soa_fill(dst->m[7], c.m_13, dst->count);
    wcn_soa_fill(dst->m[8], c.m_20, dst->count);
    wcn_soa_fill(dst->m[9], c.m_21, dst->count);
    wcn_soa_fill(dst->m[10], c.m_22, dst->count);
    wcn_soa_fill(dst->m[11], c.m_23, dst->count);
    wcn_soa_fill(dst->m[12], c.m_30, dst->count);
    wcn_soa_fill(dst->m[13], c.m_31, dst->count);
    wcn_soa_fill(dst->m[14], c.m_32, dst->count);
    wcn_soa_fill(dst->m[15], c.m_33, dst->count);
}

/* ------------------------------------------------------- translation family */

void WMATH_GET_TRANSLATION(Mat4xN)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m) {
    if (!out || !m || out->count != m->count) return;
    /* WMATH_GET_TRANSLATION(Mat4) reads m[12], m[13], m[14] -- the fourth row, since this
     * library's Mat4 is row-major with the translation in row 3. The plane index equals the
     * m[] index for Mat4xN, so those are planes 12, 13 and 14. */
    wcn_soa_copy(out->x, m->m[12], m->count);
    wcn_soa_copy(out->y, m->m[13], m->count);
    wcn_soa_copy(out->z, m->m[14], m->count);
}

void WMATH_SET_TRANSLATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                    const WMATH_TYPE(Vec3) v) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m)) return;
    /* The scalar form copies the whole matrix and then overwrites the fourth row with
     * (v.x, v.y, v.z, 1). One Vec3 applies to every element, which is the container lift
     * and matches how WMATH_CREATE broadcasts its components. */
    for (int p = 0; p < 16; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    wcn_soa_fill(dst->m[12], v.v[0], m->count);
    wcn_soa_fill(dst->m[13], v.v[1], m->count);
    wcn_soa_fill(dst->m[14], v.v[2], m->count);
    wcn_soa_fill(dst->m[15], 1.0f, m->count);
}

/* --------------------------------------------------------- scaling family */

void WMATH_SCALE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                          const WMATH_TYPE(Vec3) v) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m)) return;
    /* WMATH_SCALE(Mat4) scales rows 0, 1, 2 by v.x, v.y, v.z and copies row 3 unchanged --
     * the translation survives a scale, which is the point of the operation. Planes 0..3 are
     * row 0, 4..7 row 1, 8..11 row 2, 12..15 row 3. */
    const float s[3] = {v.v[0], v.v[1], v.v[2]};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 4; c++) {
            wcn_soa_mul_s(dst->m[r * 4 + c], m->m[r * 4 + c], s[r], m->count);
        }
    }
    for (int c = 0; c < 4; c++) wcn_soa_copy(dst->m[12 + c], m->m[12 + c], m->count);
}

void WMATH_CALL(Mat4xN, uniform_scale)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                        float s) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m)) return;
    /* Planes 0..11 are rows 0..2; row 3 -- the translation -- is copied unchanged. My first
     * version scaled all sixteen, which the test caught: WMATH_CALL(Mat4, uniform_scale) also
     * preserves the translation, exactly as WMATH_SCALE does. The two differ only in whether
     * the factor is one scalar or per-row, not in what they leave alone. */
    for (int p = 0; p < 12; p++) wcn_soa_mul_s(dst->m[p], m->m[p], s, m->count);
    for (int p = 12; p < 16; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
}

void WMATH_CALL(Mat4xN, scaling)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3) v) {
    if (!dst || !dst->m[0]) return;
    /* Builds a diagonal scaling matrix: m00 = v.x, m11 = v.y, m22 = v.z, m33 = 1, rest zero.
     * Unlike WMATH_SCALE this takes no input matrix -- it constructs one. */
    for (int p = 0; p < 16; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_fill(dst->m[0], v.v[0], dst->count);
    wcn_soa_fill(dst->m[5], v.v[1], dst->count);
    wcn_soa_fill(dst->m[10], v.v[2], dst->count);
    wcn_soa_fill(dst->m[15], 1.0f, dst->count);
}

/* --------------------------------------------------------- rotation family */
/*
 * These are per-element scalar loops, not SIMD kernels, and that is forced: no ISA in this
 * library has a vector sin/cos (SSE/AVX outside SVML, NEON, WASM SIMD, RVV base, LSX all
 * lack them), so the trig cannot be widened. Each element builds/rotates a plain Mat4 with
 * the scalar routine and the sixteen planes are gathered from it. Same shape as ANGLE.
 *
 * The angle is one value per element. A uniform rotation is a caller filling the array.
 */
void WMATH_ROTATION_X(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_ROTATION_X(Mat4)(&r, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_ROTATE_X(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                const float* radians) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat4) mi;
        WMATH_TYPE(Mat4) r;
        for (int k = 0; k < 16; k++) mi.m[k] = m->m[k][i];
        WMATH_ROTATE_X(Mat4)(&r, mi, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}
void WMATH_ROTATION_Y(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_ROTATION_Y(Mat4)(&r, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_ROTATE_Y(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                const float* radians) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat4) mi;
        WMATH_TYPE(Mat4) r;
        for (int k = 0; k < 16; k++) mi.m[k] = m->m[k][i];
        WMATH_ROTATE_Y(Mat4)(&r, mi, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}
void WMATH_ROTATION_Z(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_ROTATION_Z(Mat4)(&r, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_ROTATE_Z(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                const float* radians) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat4) mi;
        WMATH_TYPE(Mat4) r;
        for (int k = 0; k < 16; k++) mi.m[k] = m->m[k][i];
        WMATH_ROTATE_Z(Mat4)(&r, mi, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

/* ------------------------------------------------- type conversions */

void WMATH_CALL(Mat4xN, from_mat3)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat3xN)* a) {
    if (!dst || !a || dst->count != a->count) return;
    /* WMATH_CALL(Mat4, from_mat3) embeds the upper-left 3x3, zeroes the fourth column (planes
     * 3, 7, 11) and the fourth row (planes 12..14), and sets m[15] = 1. Mat3xN is packed
     * (plane p = row*3 + col) while Mat4xN's plane p is m[p] directly, so the mapping is:
     * 1->4, 2->8 (column 0); 3->5, 7->9 (column 1); 6->10 (column 2). */
    static const int SRC[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    for (int p = 0; p < 9; p++) wcn_soa_copy(dst->m[SRC[p]], a->m[p], a->count);
    wcn_soa_fill(dst->m[3], 0.0f, a->count);
    wcn_soa_fill(dst->m[7], 0.0f, a->count);
    wcn_soa_fill(dst->m[11], 0.0f, a->count);
    wcn_soa_fill(dst->m[12], 0.0f, a->count);
    wcn_soa_fill(dst->m[13], 0.0f, a->count);
    wcn_soa_fill(dst->m[14], 0.0f, a->count);
    wcn_soa_fill(dst->m[15], 1.0f, a->count);
}

void WMATH_CALL(Mat4xN, from_quat)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(QuatxN)* q) {
    if (!dst || !q || dst->count != q->count) return;
    /* Vectorised: the whole conversion is multiply/add, with no trig anywhere, so unlike the
     * rotation family it has a real SIMD body. */
    wcn_soa_quat_to_m4(dst->m, q->x, q->y, q->z, q->w, q->count);
}

void WMATH_CALL(Mat4xN, get_scaling)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m) {
    if (!out || !m || out->count != m->count) return;
    /* WMATH_CALL(Mat4, get_scaling) takes the length of planes (0,1,2), then (4,5,6), then
     * (8,9,10) -- the three groups the scalar source calls columns. */
    wcn_soa_len3(out->x, m->m[0], m->m[1], m->m[2], m->count);
    wcn_soa_len3(out->y, m->m[4], m->m[5], m->m[6], m->count);
    wcn_soa_len3(out->z, m->m[8], m->m[9], m->m[10], m->count);
}

/* --------------------------------------------------- axis rotation family */
/*
 * Per-element scalar: the axis-angle routine calls cosf/sinf, and no backend here has a
 * vector sin/cos (the scalar source says the same of itself). Both the axis and the angle
 * carry one value per element -- matching the convention chosen for the rotation family, so
 * a single call never mixes a shared parameter with a per-element one.
 */

void WMATH_CALL(Mat4xN, axis_rotation)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* axis,
                                        const float* radians) {
    if (!dst || !dst->m[0] || !axis || !radians) return;
    if (axis->count != dst->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec3) ax;
        WMATH_TYPE(Mat4) r;
        ax.v[0] = axis->x[i];
        ax.v[1] = axis->y[i];
        ax.v[2] = axis->z[i];
        WMATH_CALL(Mat4, axis_rotation)(&r, ax, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_CALL(Mat4xN, axis_rotate)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                      const WMATH_TYPE(Vec3xN)* axis, const float* radians) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !axis || !radians) return;
    if (axis->count != m->count) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Vec3) ax;
        WMATH_TYPE(Mat4) mi;
        WMATH_TYPE(Mat4) r;
        ax.v[0] = axis->x[i];
        ax.v[1] = axis->y[i];
        ax.v[2] = axis->z[i];
        for (int k = 0; k < 16; k++) mi.m[k] = m->m[k][i];
        WMATH_CALL(Mat4, axis_rotate)(&r, mi, ax, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

/* ------------------------------------------------------- projections */

void WMATH_CALL(Mat4xN, perspective)(WMATH_TYPE(Mat4xN)* dst, const float* fov_y,
                                      const float* aspect, const float* z_near,
                                      const float* z_far) {
    if (!dst || !dst->m[0] || !fov_y || !aspect || !z_near || !z_far) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_CALL(Mat4, perspective)(&r, fov_y[i], aspect[i], z_near[i], z_far[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_CALL(Mat4xN, ortho)(WMATH_TYPE(Mat4xN)* dst, const float* left, const float* right,
                                const float* bottom, const float* top, const float* znear,
                                const float* zfar) {
    if (!dst || !dst->m[0] || !left || !right || !bottom || !top || !znear || !zfar) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_CALL(Mat4, ortho)(&r, left[i], right[i], bottom[i], top[i], znear[i], zfar[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

/* ------------------------------------------------------- camera family */
/*
 * Three Vec3xN parameters, one value per element each. Per-element scalar loops: these
 * routines are not free of scalar-only work, and camera_aim in particular is a long routine
 * with its own branches -- re-deriving it in vector form would be a second implementation to
 * keep in step, for something no inner loop calls.
 */

#define WCN_M4XN_V3(OUT, SRC, IDX)                                                     \
    do {                                                                               \
        (OUT).v[0] = (SRC)->x[(IDX)];                                                  \
        (OUT).v[1] = (SRC)->y[(IDX)];                                                  \
        (OUT).v[2] = (SRC)->z[(IDX)];                                                  \
    } while (0)

#define WCN_M4XN_CAMERA(NAME)                                                          \
void WMATH_CALL(Mat4xN, NAME)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* a,    \
                               const WMATH_TYPE(Vec3xN)* b,                            \
                               const WMATH_TYPE(Vec3xN)* c) {                          \
    if (!dst || !dst->m[0] || !a || !b || !c) return;                                  \
    if (a->count != dst->count || b->count != dst->count || c->count != dst->count)     \
        return;                                                                        \
    for (size_t i = 0; i < dst->count; i++) {                                          \
        WMATH_TYPE(Vec3) va;                                                           \
        WMATH_TYPE(Vec3) vb;                                                           \
        WMATH_TYPE(Vec3) vc;                                                           \
        WMATH_TYPE(Mat4) r;                                                            \
        WCN_M4XN_V3(va, a, i);                                                         \
        WCN_M4XN_V3(vb, b, i);                                                         \
        WCN_M4XN_V3(vc, c, i);                                                         \
        WMATH_CALL(Mat4, NAME)(&r, va, vb, vc);                                        \
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];                            \
    }                                                                                  \
}

WCN_M4XN_CAMERA(aim)
WCN_M4XN_CAMERA(look_at)
WCN_M4XN_CAMERA(camera_aim)

#undef WCN_M4XN_CAMERA
#undef WCN_M4XN_V3

/* -------------------------------------------- remaining projections */

void WMATH_CALL(Mat4xN, frustum)(WMATH_TYPE(Mat4xN)* dst, const float* left, const float* right,
                                  const float* bottom, const float* top, const float* znear,
                                  const float* zfar) {
    if (!dst || !dst->m[0] || !left || !right || !bottom || !top || !znear || !zfar) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_CALL(Mat4, frustum)(&r, left[i], right[i], bottom[i], top[i], znear[i], zfar[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_CALL(Mat4xN, frustum_reverse_z)(WMATH_TYPE(Mat4xN)* dst, const float* left,
                                            const float* right, const float* bottom,
                                            const float* top, const float* znear,
                                            const float* zfar) {
    if (!dst || !dst->m[0] || !left || !right || !bottom || !top || !znear || !zfar) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_CALL(Mat4, frustum_reverse_z)(&r, left[i], right[i], bottom[i], top[i],
                                            znear[i], zfar[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_CALL(Mat4xN, perspective_reverse_z)(WMATH_TYPE(Mat4xN)* dst, const float* fov_y,
                                                const float* aspect, const float* z_near,
                                                const float* z_far) {
    if (!dst || !dst->m[0] || !fov_y || !aspect || !z_near || !z_far) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat4) r;
        WMATH_CALL(Mat4, perspective_reverse_z)(&r, fov_y[i], aspect[i], z_near[i], z_far[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

/* --------------------------------------------------- translation family */

void WMATH_CALL(Mat4xN, translate)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                    const WMATH_TYPE(Vec3xN)* v) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !v || v->count != m->count) return;
    /* Rows 0..2 pass through; row 3 is the translation multiply. */
    for (int p = 0; p < 12; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    wcn_soa_m4_translate(dst->m, WCN_SOA_RO(m->m), v->x, v->y, v->z, m->count);
}

void WMATH_TRANSLATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* v) {
    if (!dst || !dst->m[0] || !v || v->count != dst->count) return;
    /* Builds an identity whose fourth row is (vx, vy, vz, 1) -- WMATH_TRANSLATION(Mat4). */
    for (int p = 0; p < 16; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_fill(dst->m[0], 1.0f, dst->count);
    wcn_soa_fill(dst->m[5], 1.0f, dst->count);
    wcn_soa_fill(dst->m[10], 1.0f, dst->count);
    wcn_soa_copy(dst->m[12], v->x, dst->count);
    wcn_soa_copy(dst->m[13], v->y, dst->count);
    wcn_soa_copy(dst->m[14], v->z, dst->count);
    wcn_soa_fill(dst->m[15], 1.0f, dst->count);
}

void WMATH_CALL(Mat4xN, uniform_scaling)(WMATH_TYPE(Mat4xN)* dst, const float* s) {
    if (!dst || !dst->m[0] || !s) return;
    /* Diagonal of s with m[15] = 1; one scale per element. */
    for (int p = 0; p < 16; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_copy(dst->m[0], s, dst->count);
    wcn_soa_copy(dst->m[5], s, dst->count);
    wcn_soa_copy(dst->m[10], s, dst->count);
    wcn_soa_fill(dst->m[15], 1.0f, dst->count);
}

/* --------------------------------------------------------- axis accessors */

void WMATH_CALL(Mat4xN, get_axis)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m,
                                   int axis) {
    if (!out || !m || out->count != m->count) return;
    if (axis < 0 || axis > 3) return;
    /* WMATH_CALL(Mat4, get_axis) indexes off = axis*4 and reads off+0..2 -- ROWS, not
     * columns. The `int axis` is applied to the whole container on purpose: which row to read
     * is a structural choice, not data that varies between the N items. */
    const int off = axis * 4;
    wcn_soa_copy(out->x, m->m[off + 0], m->count);
    wcn_soa_copy(out->y, m->m[off + 1], m->count);
    wcn_soa_copy(out->z, m->m[off + 2], m->count);
}

void WMATH_CALL(Mat4xN, set_axis)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                                   const WMATH_TYPE(Vec3xN)* v, int axis) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !v || v->count != m->count) return;
    if (axis < 0 || axis > 3) return;
    for (int p = 0; p < 16; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    const int off = axis * 4;
    wcn_soa_copy(dst->m[off + 0], v->x, m->count);
    wcn_soa_copy(dst->m[off + 1], v->y, m->count);
    wcn_soa_copy(dst->m[off + 2], v->z, m->count);
}

/* --------------------------------------------------------- axis rotation */

void WMATH_ROTATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* axis,
                             const float* radians) {
    if (!dst || !dst->m[0] || !axis || !radians || axis->count != dst->count) return;
    /* Per-element scalar: the axis-angle routine calls cosf/sinf and no backend has vector
     * trig. Both the axis and the angle carry one value per element. */
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec3) ax;
        WMATH_TYPE(Mat4) r;
        ax.v[0] = axis->x[i];
        ax.v[1] = axis->y[i];
        ax.v[2] = axis->z[i];
        WMATH_ROTATION(Mat4)(&r, ax, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}

void WMATH_ROTATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m,
                           const WMATH_TYPE(Vec3xN)* axis, const float* radians) {
    assert(m4xn_ok1(dst, m));
    if (!m4xn_ok1(dst, m) || !axis || !radians) return;
    if (axis->count != m->count) return;
    /* WMATH_ROTATE(Mat4) has the same shape as axis_rotate and calls trig, so per element,
     * with both the axis and the angle carrying one value per element. */
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Vec3) ax;
        WMATH_TYPE(Mat4) mi;
        WMATH_TYPE(Mat4) r;
        ax.v[0] = axis->x[i];
        ax.v[1] = axis->y[i];
        ax.v[2] = axis->z[i];
        wcn_math_Mat4xN_get(&mi, m, i);
        WMATH_ROTATE(Mat4)(&r, mi, ax, radians[i]);
        for (int k = 0; k < 16; k++) dst->m[k][i] = r.m[k];
    }
}
