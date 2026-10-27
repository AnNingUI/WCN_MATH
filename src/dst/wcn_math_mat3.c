/*
 * ISA width policy for this file (see SIMD_PERF_BUGS.md, "ISA width measurement")
 * ---------------------------------------------------------------------------
 * Mat3 kernels: 3x3 rows padded to four floats each, one register per row.
 *
 * AVX-512 is still the first choice for a build, for a different reason than width:
 * with AVX512VL the architectural XMM count doubles from 16 to 32 and mask
 * registers appear, so this same 128-bit code compiles with far fewer spills.
 * Measured over the whole library at /arch:AVX2 vs /arch:AVX512:
 *     instructions -3.9% overall (mat3 -7.8%, mat4 -6.4%, other -4.9%),
 *     stack accesses 828 -> 708 (-14%),
 *     wcn_math_Mat4_inverse 107 ins / 22 stack -> 80 ins / 4 stack.
 * That is why the ladder AVX512 > AVX2 > FMA+SSE > SSE is realised as build
 * variants (tools/build_libs_avx512.bat, CMake WCN_MATH_ENABLE_AVX512) rather than
 * as additional #if branches in this file: there is no 128-bit-to-512-bit rewrite
 * to write, only a better register budget.
 *
 * The functions here branch on WCN_HAS_X86_64 / AARCH64 / WASM_SIMD /
 * RISCV_VECTOR / LOONGARCH_LSX and use 128-bit vectors in every case. That is a
 * measured decision, not an omission.
 *
 * A single Vec/Mat/Quat operation carries four floats of live data, which is one
 * 128-bit register. Measured on this machine:
 *
 *     4-float op        SSE 0.800 ns   AVX(256) 0.750 ns   -6% (in the noise)
 *     RGBA->SoA (N px)  SSE 0.51 ns/px AVX2     0.33 ns/px  1.55x
 *
 * The 256-bit form only pays when the work grows with input size, which happens in
 * the image conversion loops and nowhere here. Widening these functions would also
 * add a vzeroupper transition at every call boundary and pull the core onto the
 * AVX frequency curve for no arithmetic gain.
 *
 * Functions whose scalar bodies are deliberate (no SIMD branch at all) carry their
 * own reason next to the body.
 */

#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include <string.h>

// BEGIN Mat3

void WMATH_IDENTITY(Mat3)(DST_MAT3) {
    dst->m[0] = 1.0f;
    dst->m[1] = 0.0f;
    dst->m[2] = 0.0f;
    dst->m[3] = 0.0f;
    dst->m[4] = 0.0f;
    dst->m[5] = 1.0f;
    dst->m[6] = 0.0f;
    dst->m[7] = 0.0f;
    dst->m[8] = 0.0f;
    dst->m[9] = 0.0f;
    dst->m[10] = 1.0f;
    dst->m[11] = 0.0f;
}

void WMATH_ZERO(Mat3)(DST_MAT3) {
    for (int i = 0; i < 12; i++) {
        dst->m[i] = 0.0f;
    }
}

void WMATH_CREATE(Mat3)(DST_MAT3, const WMATH_CREATE_TYPE(Mat3) mat3_c) {
    dst->m[0] = WMATH_OR_ELSE_ZERO(mat3_c.m_00);
    dst->m[1] = WMATH_OR_ELSE_ZERO(mat3_c.m_01);
    dst->m[2] = WMATH_OR_ELSE_ZERO(mat3_c.m_02);
    dst->m[4] = WMATH_OR_ELSE_ZERO(mat3_c.m_10);
    dst->m[5] = WMATH_OR_ELSE_ZERO(mat3_c.m_11);
    dst->m[6] = WMATH_OR_ELSE_ZERO(mat3_c.m_12);
    dst->m[8] = WMATH_OR_ELSE_ZERO(mat3_c.m_20);
    dst->m[9] = WMATH_OR_ELSE_ZERO(mat3_c.m_21);
    dst->m[10] = WMATH_OR_ELSE_ZERO(mat3_c.m_22);
    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;
}

void WMATH_COPY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {
    memcpy(dst, &mat, sizeof(WMATH_TYPE(Mat3)));
}

bool WMATH_EQUALS(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {
    return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] && a.m[4] == b.m[4] &&
            a.m[5] == b.m[5] && a.m[6] == b.m[6] && a.m[8] == b.m[8] && a.m[9] == b.m[9] &&
            a.m[10] == b.m[10]);
}

bool WMATH_EQUALS_APPROXIMATELY(Mat3)(const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {
    const float ep = WCN_GET_EPSILON();
    return (fabsf(a.m[0] - b.m[0]) < ep && fabsf(a.m[1] - b.m[1]) < ep &&
            fabsf(a.m[2] - b.m[2]) < ep && fabsf(a.m[4] - b.m[4]) < ep &&
            fabsf(a.m[5] - b.m[5]) < ep && fabsf(a.m[6] - b.m[6]) < ep &&
            fabsf(a.m[8] - b.m[8]) < ep && fabsf(a.m[9] - b.m[9]) < ep &&
            fabsf(a.m[10] - b.m[10]) < ep);
}

void WMATH_SET(Mat3)(DST_MAT3, const float m00, const float m01, const float m02, const float m10,
                     const float m11, const float m12, const float m20, const float m21,
                     const float m22) {
    dst->m[0] = m00;
    dst->m[1] = m01;
    dst->m[2] = m02;
    dst->m[4] = m10;
    dst->m[5] = m11;
    dst->m[6] = m12;
    dst->m[8] = m20;
    dst->m[9] = m21;
    dst->m[10] = m22;
    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;
}

void WMATH_NEGATE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 sign_mask = _mm_set1_ps(-0.0f);

    __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
    __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[4]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[8]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_res;

    vec_a = vld1q_f32(&mat.m[0]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&mat.m[4]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&mat.m[8]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t neg_one = wasm_f32x4_splat(-1.0f);

    v128_t vec_a = wasm_v128_load(&mat.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&mat.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&mat.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl = __riscv_vsetvl_e32m1(4);

    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&mat.m[0], vl);
    vfloat32m1_t neg_one = __riscv_vfmv_v_f_f32m1(-1.0f, vl);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&mat.m[4], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&mat.m[8], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, neg_one, vl);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 sign_mask = wcn_lsx_splat_f(-0.0f);

    __m128 vec_a = __lsx_vld(&mat.m[0], 0);
    __m128 vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&mat.m[4], 0);
    vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&mat.m[8], 0);
    vec_res = __lsx_vxor_v(vec_a, sign_mask);
    __lsx_vst(vec_res, &dst->m[8], 0);

#else
    WMATH_SET(Mat3)(dst, -mat.m[0], -mat.m[1], -mat.m[2], -mat.m[4], -mat.m[5], -mat.m[6],
                    -mat.m[8], -mat.m[9], -mat.m[10]);
#endif
}

void WMATH_TRANSPOSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) mat) {
    dst->m[0] = mat.m[0];
    dst->m[1] = mat.m[4];
    dst->m[2] = mat.m[8];
    dst->m[3] = 0.0f;

    dst->m[4] = mat.m[1];
    dst->m[5] = mat.m[5];
    dst->m[6] = mat.m[9];
    dst->m[7] = 0.0f;

    dst->m[8] = mat.m[2];
    dst->m[9] = mat.m[6];
    dst->m[10] = mat.m[10];
    dst->m[11] = 0.0f;
}

void WMATH_ADD(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 row_a = wcn_mat3_get_row(&a, 0);
    __m128 row_b = wcn_mat3_get_row(&b, 0);
    __m128 row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = _mm_add_ps(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = vaddq_f32(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = wasm_f32x4_add(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = __riscv_vfadd_vv_f32m1(row_a, row_b, 4);
    wcn_mat3_set_row(dst, 2, row_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 row_a, row_b, row_res;

    row_a = wcn_mat3_get_row(&a, 0);
    row_b = wcn_mat3_get_row(&b, 0);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 0, row_res);

    row_a = wcn_mat3_get_row(&a, 1);
    row_b = wcn_mat3_get_row(&b, 1);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 1, row_res);

    row_a = wcn_mat3_get_row(&a, 2);
    row_b = wcn_mat3_get_row(&b, 2);
    row_res = __lsx_vfadd_s(row_a, row_b);
    wcn_mat3_set_row(dst, 2, row_res);

#else
    dst->m[0] = a.m[0] + b.m[0];
    dst->m[1] = a.m[1] + b.m[1];
    dst->m[2] = a.m[2] + b.m[2];
    dst->m[3] = 0.0f;
    dst->m[4] = a.m[4] + b.m[4];
    dst->m[5] = a.m[5] + b.m[5];
    dst->m[6] = a.m[6] + b.m[6];
    dst->m[7] = 0.0f;
    dst->m[8] = a.m[8] + b.m[8];
    dst->m[9] = a.m[9] + b.m[9];
    dst->m[10] = a.m[10] + b.m[10];
    dst->m[11] = 0.0f;
#endif
}

void WMATH_SUB(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_b = _mm_loadu_ps(&b.m[0]);
    __m128 vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_b = _mm_loadu_ps(&b.m[4]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_b = _mm_loadu_ps(&b.m[8]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b, vec_res;

    vec_a = vld1q_f32(&a.m[0]);
    vec_b = vld1q_f32(&b.m[0]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&a.m[4]);
    vec_b = vld1q_f32(&b.m[4]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&a.m[8]);
    vec_b = vld1q_f32(&b.m[8]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_b, vec_res;

    vec_a = wasm_v128_load(&a.m[0]);
    vec_b = wasm_v128_load(&b.m[0]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&a.m[4]);
    vec_b = wasm_v128_load(&b.m[4]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&a.m[8]);
    vec_b = wasm_v128_load(&b.m[8]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[8], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t vec_a, vec_b, vec_res;

    vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[0], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[4], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
    vec_b = __riscv_vle32_v_f32m1(&b.m[8], 4);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, 4);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, 4);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;

    vec_a = __lsx_vld(&a.m[0], 0);
    vec_b = __lsx_vld(&b.m[0], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&a.m[4], 0);
    vec_b = __lsx_vld(&b.m[4], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&a.m[8], 0);
    vec_b = __lsx_vld(&b.m[8], 0);
    vec_res = __lsx_vfsub_s(vec_a, vec_b);
    __lsx_vst(vec_res, &dst->m[8], 0);

#else
    dst->m[0] = a.m[0] - b.m[0];
    dst->m[1] = a.m[1] - b.m[1];
    dst->m[2] = a.m[2] - b.m[2];
    dst->m[3] = 0.0f;
    dst->m[4] = a.m[4] - b.m[4];
    dst->m[5] = a.m[5] - b.m[5];
    dst->m[6] = a.m[6] - b.m[6];
    dst->m[7] = 0.0f;
    dst->m[8] = a.m[8] - b.m[8];
    dst->m[9] = a.m[9] - b.m[9];
    dst->m[10] = a.m[10] - b.m[10];
    dst->m[11] = 0.0f;
#endif
}

void WMATH_MULTIPLY_SCALAR(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const float b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_scalar = _mm_set1_ps(b);

    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_res = _mm_mul_ps(vec_a, vec_scalar);
    _mm_storeu_ps(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_scalar, vec_res;
    vec_scalar = vdupq_n_f32(b);

    vec_a = vld1q_f32(&a.m[0]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[0], vec_res);

    vec_a = vld1q_f32(&a.m[4]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[4], vec_res);

    vec_a = vld1q_f32(&a.m[8]);
    vec_res = vmulq_f32(vec_a, vec_scalar);
    vst1q_f32(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_scalar = wasm_f32x4_splat(b);

    v128_t vec_a = wasm_v128_load(&a.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&a.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&a.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, vec_scalar);
    wasm_v128_store(&dst->m[8], vec_res);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    vfloat32m1_t vec_b_scalar = __riscv_vfmv_v_f_f32m1(b, 4);

    vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(&a.m[0], 4);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[0], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[4], 4);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[4], vec_res, 4);

    vec_a = __riscv_vle32_v_f32m1(&a.m[8], 4);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, 4);
    __riscv_vse32_v_f32m1(&dst->m[8], vec_res, 4);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_scalar = __lsx_vldrepl_w(&b, 0);

    __m128 vec_a = __lsx_vld(&a.m[0], 0);
    __m128 vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[0], 0);

    vec_a = __lsx_vld(&a.m[4], 0);
    vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[4], 0);

    vec_a = __lsx_vld(&a.m[8], 0);
    vec_res = __lsx_vfmul_s(vec_a, vec_scalar);
    __lsx_vst(vec_res, &dst->m[8], 0);

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;

#else
    for (int i = 0; i < 12; i++) {
        dst->m[i] = a.m[i] * b;
    }
#endif
}

void WMATH_INVERSE(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a) {
    const float m00 = a.m[0], m01 = a.m[1], m02 = a.m[2];
    const float m10 = a.m[4], m11 = a.m[5], m12 = a.m[6];
    const float m20 = a.m[8], m21 = a.m[9], m22 = a.m[10];

    /* Compute the nine cofactors once and reuse them for both the determinant and
     * the result.
     *
     * The determinant used to be written out separately as
     *     m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)
     * which recomputes three of the cofactors below (m11*m22-m12*m21 appears again
     * as dst->m[0], m10*m22-m12*m20 as -dst->m[4], m10*m21-m11*m20 as dst->m[8]).
     * Naming them once removes six multiplies and three subtracts. */
    const float c00 = m11 * m22 - m12 * m21;
    const float c01 = m02 * m21 - m01 * m22;
    const float c02 = m01 * m12 - m02 * m11;
    const float c10 = m12 * m20 - m10 * m22;
    const float c11 = m00 * m22 - m02 * m20;
    const float c12 = m02 * m10 - m00 * m12;
    const float c20 = m10 * m21 - m11 * m20;
    const float c21 = m01 * m20 - m00 * m21;
    const float c22 = m00 * m11 - m01 * m10;

    /* det = m00*c00 + m01*c10 + m02*c20, the cofactor expansion along the first row
     * (c10 already carries the negative sign for the middle term, c20 the positive
     * one for the last). */
    const float det = m00 * c00 + m01 * c10 + m02 * c20;

    if (fabsf(det) < 1e-12f) {
        WMATH_IDENTITY(Mat3)(dst);
        return;
    }

    const float inv_det = 1.0f / det;

    /* Scalar on every backend -- one shared body, no per-ISA branch.
     *
     * Two spellings were measured and both were rejected:
     *
     * 1. The old x86 branch built the nine cofactors with scalar expressions inside
     *    _mm_set_ps argument lists. The compiler emitted the same scalar multiplies
     *    either way and the vector register only added packing, so it was pseudo-SIMD.
     *
     * 2. A genuine vectorisation exists: adj(A)'s columns are the cyclic row cross
     *    products (col0 = b x c, col1 = c x a, col2 = a x b) with det = a . col0,
     *    verified against Gauss-Jordan over 300 random matrices to 3.9e-14. On x86 it
     *    is real: 122 -> 87 instructions (-28.7%) and 7.651 -> 7.483 ns median
     *    (1.022x over 12 alternating runs). It is not kept because it cannot be made
     *    uniform across the backends -- SSE/WASM do the two 3-element rotations in one
     *    shuffle each, but NEON's vextq_f32 cycles all four lanes and drags the
     *    padding into the significant ones, so an equivalent rotation costs 4
     *    instructions there and the formulation stops paying. A 1-of-5 coverage would
     *    also break the requirement that every backend carries its own branch, and the
     *    2% is not worth that. See SIMD_PERF_BUGS.md, section 10.
     *
     * The determinant's reciprocal is taken unconditionally past the guard above,
     * matching the previous behaviour; `a` is by value, so dst cannot alias it.
     *
     * Layout: Mat3 keeps a padding lane per row, hence m[0..2], m[4..6], m[8..10]
     * with m[3]/m[7]/m[11] zeroed. */
    dst->m[0] = c00 * inv_det;
    dst->m[1] = c01 * inv_det;
    dst->m[2] = c02 * inv_det;
    dst->m[3] = 0.0f;

    dst->m[4] = c10 * inv_det;
    dst->m[5] = c11 * inv_det;
    dst->m[6] = c12 * inv_det;
    dst->m[7] = 0.0f;

    dst->m[8] = c20 * inv_det;
    dst->m[9] = c21 * inv_det;
    dst->m[10] = c22 * inv_det;
    dst->m[11] = 0.0f;
}

void WMATH_MULTIPLY(Mat3)(DST_MAT3, const WMATH_TYPE(Mat3) a, const WMATH_TYPE(Mat3) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    /* Row-linear-combination form, same reasoning as Mat4_multiply: build each
       output row as a sum of broadcast a[i][k] times b row k. The rows of `b`
       live at strides of 4 (Mat3 keeps a padding lane per row), so they load as
       full 4-lane vectors.

       Writing straight into `dst` is safe: `a` and `b` are by-value parameters,
       so they are stack copies and `dst` cannot alias them. The previous version
       computed into a `result = {0}` temporary and then copied all 12 floats,
       which on its own measured ~3x the cost of the arithmetic. The padding
       lanes are re-zeroed below so a caller's garbage padding never leaks out. */
    const __m128 b_r0 = _mm_loadu_ps(&b.m[0]);
    const __m128 b_r1 = _mm_loadu_ps(&b.m[4]);
    const __m128 b_r2 = _mm_loadu_ps(&b.m[8]);

    for (int i = 0; i < 3; i++) {
        __m128 acc = _mm_mul_ps(_mm_set1_ps(a.m[i * 4 + 0]), b_r0);
        acc = wcn_fma_mul_add_ps(_mm_set1_ps(a.m[i * 4 + 1]), b_r1, acc);
        acc = wcn_fma_mul_add_ps(_mm_set1_ps(a.m[i * 4 + 2]), b_r2, acc);
        _mm_storeu_ps(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    /* Same row-linear-combination form; vfmaq_n_f32 folds the broadcast in. */
    const float32x4_t b_r0 = vld1q_f32(&b.m[0]);
    const float32x4_t b_r1 = vld1q_f32(&b.m[4]);
    const float32x4_t b_r2 = vld1q_f32(&b.m[8]);

    for (int i = 0; i < 3; i++) {
        float32x4_t acc = vmulq_n_f32(b_r0, a.m[i * 4 + 0]);
        acc = vfmaq_n_f32(acc, b_r1, a.m[i * 4 + 1]);
        acc = vfmaq_n_f32(acc, b_r2, a.m[i * 4 + 2]);
        vst1q_f32(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    const v128_t b_r0 = wasm_v128_load(&b.m[0]);
    const v128_t b_r1 = wasm_v128_load(&b.m[4]);
    const v128_t b_r2 = wasm_v128_load(&b.m[8]);

    for (int i = 0; i < 3; i++) {
        v128_t acc = wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 0]), b_r0);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 1]), b_r1));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 2]), b_r2));
        wasm_v128_store(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    const size_t vl = __riscv_vsetvl_e32m1(4);
    const vfloat32m1_t b_r0 = __riscv_vle32_v_f32m1(&b.m[0], vl);
    const vfloat32m1_t b_r1 = __riscv_vle32_v_f32m1(&b.m[4], vl);
    const vfloat32m1_t b_r2 = __riscv_vle32_v_f32m1(&b.m[8], vl);

    for (int i = 0; i < 3; i++) {
        vfloat32m1_t acc = __riscv_vfmul_vf_f32m1(b_r0, a.m[i * 4 + 0], vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a.m[i * 4 + 1], b_r1, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a.m[i * 4 + 2], b_r2, vl);
        __riscv_vse32_v_f32m1(&dst->m[i * 4], acc, vl);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    const __m128 b_r0 = (__m128)__lsx_vld(&b.m[0], 0);
    const __m128 b_r1 = (__m128)__lsx_vld(&b.m[4], 0);
    const __m128 b_r2 = (__m128)__lsx_vld(&b.m[8], 0);

    for (int i = 0; i < 3; i++) {
        __m128 acc = (__m128)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 0], 0),
                                            (__m128i)b_r0);
        acc = (__m128)__lsx_vfadd_s(
            (__m128i)acc, (__m128i)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 1], 0),
                                                 (__m128i)b_r1));
        acc = (__m128)__lsx_vfadd_s(
            (__m128i)acc, (__m128i)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 2], 0),
                                                 (__m128i)b_r2));
        __lsx_vst((__m128i)acc, &dst->m[i * 4], 0);
    }

#else
    dst->m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[4] + a.m[2] * b.m[8];
    dst->m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[5] + a.m[2] * b.m[9];
    dst->m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[6] + a.m[2] * b.m[10];
    dst->m[4] = a.m[4] * b.m[0] + a.m[5] * b.m[4] + a.m[6] * b.m[8];
    dst->m[5] = a.m[4] * b.m[1] + a.m[5] * b.m[5] + a.m[6] * b.m[9];
    dst->m[6] = a.m[4] * b.m[2] + a.m[5] * b.m[6] + a.m[6] * b.m[10];
    dst->m[8] = a.m[8] * b.m[0] + a.m[9] * b.m[4] + a.m[10] * b.m[8];
    dst->m[9] = a.m[8] * b.m[1] + a.m[9] * b.m[5] + a.m[10] * b.m[9];
    dst->m[10] = a.m[8] * b.m[2] + a.m[9] * b.m[6] + a.m[10] * b.m[10];
#endif

    dst->m[3] = dst->m[7] = dst->m[11] = 0.0f;
}

float WMATH_DETERMINANT(Mat3)(const WMATH_TYPE(Mat3) m) {
    /* Scalar on every backend, structurally: nine products collapse to one
     * float. Any vector form would broadcast, multiply and then reduce -- more
     * instructions than the scalar expression, which the compiler already
     * schedules well. Contrast Mat4_determinant, where the same reasoning led to
     * deleting a SIMD branch that measured 9.7 ns against 7.8 ns. */
    return m.m[0] * (m.m[5] * m.m[10] - m.m[9] * m.m[6]) -
           m.m[4] * (m.m[1] * m.m[10] - m.m[9] * m.m[2]) +
           m.m[8] * (m.m[1] * m.m[6] - m.m[5] * m.m[2]);
}

// END Mat3
