/*
 * ISA width policy for this file (see SIMD_PERF_BUGS.md, "ISA width measurement")
 * ---------------------------------------------------------------------------
 * Mat4 kernels: 4x4 rows of four floats, one register per row.
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
static inline void __prev_clangd_check() {
wcn_math_get_epsilon();
}
// BEGIN Mat4

void WMATH_IDENTITY(Mat4)(DST_MAT4) {
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
    dst->m[12] = 0.0f;
    dst->m[13] = 0.0f;
    dst->m[14] = 0.0f;
    dst->m[15] = 1.0f;
}

void WMATH_ZERO(Mat4)(DST_MAT4) {
    for (int i = 0; i < 16; i++) {
        dst->m[i] = 0.0f;
    }
}

void WMATH_CREATE(Mat4)(DST_MAT4, const WMATH_CREATE_TYPE(Mat4) mat4_c) {
    dst->m[0] = WMATH_OR_ELSE_ZERO(mat4_c.m_00);
    dst->m[1] = WMATH_OR_ELSE_ZERO(mat4_c.m_01);
    dst->m[2] = WMATH_OR_ELSE_ZERO(mat4_c.m_02);
    dst->m[3] = WMATH_OR_ELSE_ZERO(mat4_c.m_03);
    dst->m[4] = WMATH_OR_ELSE_ZERO(mat4_c.m_10);
    dst->m[5] = WMATH_OR_ELSE_ZERO(mat4_c.m_11);
    dst->m[6] = WMATH_OR_ELSE_ZERO(mat4_c.m_12);
    dst->m[7] = WMATH_OR_ELSE_ZERO(mat4_c.m_13);
    dst->m[8] = WMATH_OR_ELSE_ZERO(mat4_c.m_20);
    dst->m[9] = WMATH_OR_ELSE_ZERO(mat4_c.m_21);
    dst->m[10] = WMATH_OR_ELSE_ZERO(mat4_c.m_22);
    dst->m[11] = WMATH_OR_ELSE_ZERO(mat4_c.m_23);
    dst->m[12] = WMATH_OR_ELSE_ZERO(mat4_c.m_30);
    dst->m[13] = WMATH_OR_ELSE_ZERO(mat4_c.m_31);
    dst->m[14] = WMATH_OR_ELSE_ZERO(mat4_c.m_32);
    dst->m[15] = WMATH_OR_ELSE_ZERO(mat4_c.m_33);
}

void WMATH_COPY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat) {
    memcpy(dst, &mat, sizeof(WMATH_TYPE(Mat4)));
}

void WMATH_SET(Mat4)(DST_MAT4, const float m00, const float m01, const float m02, const float m03,
                     const float m10, const float m11, const float m12, const float m13,
                     const float m20, const float m21, const float m22, const float m23,
                     const float m30, const float m31, const float m32, const float m33) {
    dst->m[0] = m00;
    dst->m[1] = m01;
    dst->m[2] = m02;
    dst->m[3] = m03;
    dst->m[4] = m10;
    dst->m[5] = m11;
    dst->m[6] = m12;
    dst->m[7] = m13;
    dst->m[8] = m20;
    dst->m[9] = m21;
    dst->m[10] = m22;
    dst->m[11] = m23;
    dst->m[12] = m30;
    dst->m[13] = m31;
    dst->m[14] = m32;
    dst->m[15] = m33;
}

void WMATH_NEGATE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) mat) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    const __m128 sign_mask = _mm_set1_ps(-0.0f);

    __m128 vec_a = _mm_loadu_ps(&mat.m[0]);
    __m128 vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[0], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[4]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[4], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[8]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[8], vec_res);

    vec_a = _mm_loadu_ps(&mat.m[12]);
    vec_res = _mm_xor_ps(vec_a, sign_mask);
    _mm_storeu_ps(&dst->m[12], vec_res);

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

    vec_a = vld1q_f32(&mat.m[12]);
    vec_res = vnegq_f32(vec_a);
    vst1q_f32(&dst->m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_res;
    const v128_t neg_one = wasm_f32x4_splat(-1.0f);

    vec_a = wasm_v128_load(&mat.m[0]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[0], vec_res);

    vec_a = wasm_v128_load(&mat.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[4], vec_res);

    vec_a = wasm_v128_load(&mat.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[8], vec_res);

    vec_a = wasm_v128_load(&mat.m[12]);
    vec_res = wasm_f32x4_mul(vec_a, neg_one);
    wasm_v128_store(&dst->m[12], vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_res;
    float* a_ptr = (float*)&mat;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvl_e32m1(4);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_res = __riscv_vfneg_v_f32m1(vec_a, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_res;
    const __m128 sign_mask = wcn_lsx_splat_f(-0.0f);

    vec_a = (__m128)__lsx_vld(&mat.m[0], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[4], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[8], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);

    vec_a = (__m128)__lsx_vld(&mat.m[12], 0);
    vec_res = (__m128)__lsx_vxor_v((__m128i)vec_a, (__m128i)sign_mask);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);

#else
    for (int i = 0; i < 16; i++) {
        dst->m[i] = -mat.m[i];
    }
#endif
}

bool WMATH_EQUALS(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
    return (a.m[0] == b.m[0] && a.m[1] == b.m[1] && a.m[2] == b.m[2] && a.m[3] == b.m[3] &&
            a.m[4] == b.m[4] && a.m[5] == b.m[5] && a.m[6] == b.m[6] && a.m[7] == b.m[7] &&
            a.m[8] == b.m[8] && a.m[9] == b.m[9] && a.m[10] == b.m[10] && a.m[11] == b.m[11] &&
            a.m[12] == b.m[12] && a.m[13] == b.m[13] && a.m[14] == b.m[14] && a.m[15] == b.m[15]);
}

bool WMATH_EQUALS_APPROXIMATELY(Mat4)(const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
    return (
        fabsf(a.m[0] - b.m[0]) < WCN_GET_EPSILON() && fabsf(a.m[1] - b.m[1]) < WCN_GET_EPSILON() &&
        fabsf(a.m[2] - b.m[2]) < WCN_GET_EPSILON() && fabsf(a.m[3] - b.m[3]) < WCN_GET_EPSILON() &&
        fabsf(a.m[4] - b.m[4]) < WCN_GET_EPSILON() && fabsf(a.m[5] - b.m[5]) < WCN_GET_EPSILON() &&
        fabsf(a.m[6] - b.m[6]) < WCN_GET_EPSILON() && fabsf(a.m[7] - b.m[7]) < WCN_GET_EPSILON() &&
        fabsf(a.m[8] - b.m[8]) < WCN_GET_EPSILON() && fabsf(a.m[9] - b.m[9]) < WCN_GET_EPSILON() &&
        fabsf(a.m[10] - b.m[10]) < WCN_GET_EPSILON() &&
        fabsf(a.m[11] - b.m[11]) < WCN_GET_EPSILON() &&
        fabsf(a.m[12] - b.m[12]) < WCN_GET_EPSILON() &&
        fabsf(a.m[13] - b.m[13]) < WCN_GET_EPSILON() &&
        fabsf(a.m[14] - b.m[14]) < WCN_GET_EPSILON() &&
        fabsf(a.m[15] - b.m[15]) < WCN_GET_EPSILON());
}

void WMATH_ADD(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_b = _mm_loadu_ps(&b.m[0]);
    __m128 vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[0], vec_res);
    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_b = _mm_loadu_ps(&b.m[4]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[4], vec_res);
    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_b = _mm_loadu_ps(&b.m[8]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[8], vec_res);
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_b = _mm_loadu_ps(&b.m[12]);
    vec_res = _mm_add_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b, vec_res;
    vec_a = vld1q_f32(&a.m[0]);
    vec_b = vld1q_f32(&b.m[0]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[0], vec_res);
    vec_a = vld1q_f32(&a.m[4]);
    vec_b = vld1q_f32(&b.m[4]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[4], vec_res);
    vec_a = vld1q_f32(&a.m[8]);
    vec_b = vld1q_f32(&b.m[8]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[8], vec_res);
    vec_a = vld1q_f32(&a.m[12]);
    vec_b = vld1q_f32(&b.m[12]);
    vec_res = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_a, vec_b, vec_res;
    vec_a = wasm_v128_load(&a.m[0]);
    vec_b = wasm_v128_load(&b.m[0]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[0], vec_res);
    vec_a = wasm_v128_load(&a.m[4]);
    vec_b = wasm_v128_load(&b.m[4]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[4], vec_res);
    vec_a = wasm_v128_load(&a.m[8]);
    vec_b = wasm_v128_load(&b.m[8]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[8], vec_res);
    vec_a = wasm_v128_load(&a.m[12]);
    vec_b = wasm_v128_load(&b.m[12]);
    vec_res = wasm_f32x4_add(vec_a, vec_b);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b, vec_res;
    float* a_ptr = (float*)&a;
    float* b_ptr = (float*)&b;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvl_e32m1(4);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
    vec_res = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;
    vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    vec_b = (__m128)__lsx_vld(&b.m[0], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_b = (__m128)__lsx_vld(&b.m[4], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_b = (__m128)__lsx_vld(&b.m[8], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_b = (__m128)__lsx_vld(&b.m[12], 0);
    vec_res = (__m128)__lsx_vfadd_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] + b.m[i];
    }
#endif
}

void WMATH_SUB(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {
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
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_b = _mm_loadu_ps(&b.m[12]);
    vec_res = _mm_sub_ps(vec_a, vec_b);
    _mm_storeu_ps(&dst->m[12], vec_res);
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
    vec_a = vld1q_f32(&a.m[12]);
    vec_b = vld1q_f32(&b.m[12]);
    vec_res = vsubq_f32(vec_a, vec_b);
    vst1q_f32(&dst->m[12], vec_res);
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
    vec_a = wasm_v128_load(&a.m[12]);
    vec_b = wasm_v128_load(&b.m[12]);
    vec_res = wasm_f32x4_sub(vec_a, vec_b);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b, vec_res;
    float* a_ptr = (float*)&a;
    float* b_ptr = (float*)&b;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvl_e32m1(4);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[0], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[4], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[8], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_b = __riscv_vle32_v_f32m1(&b_ptr[12], vl);
    vec_res = __riscv_vfsub_vv_f32m1(vec_a, vec_b, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_a, vec_b, vec_res;
    vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    vec_b = (__m128)__lsx_vld(&b.m[0], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_b = (__m128)__lsx_vld(&b.m[4], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_b = (__m128)__lsx_vld(&b.m[8], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_b = (__m128)__lsx_vld(&b.m[12], 0);
    vec_res = (__m128)__lsx_vfsub_s((__m128i)vec_a, (__m128i)vec_b);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] - b.m[i];
    }
#endif
}

void WMATH_MULTIPLY_SCALAR(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const float b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 vec_b_scalar = _mm_set1_ps(b);
    __m128 vec_a = _mm_loadu_ps(&a.m[0]);
    __m128 vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[0], vec_res);
    vec_a = _mm_loadu_ps(&a.m[4]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[4], vec_res);
    vec_a = _mm_loadu_ps(&a.m[8]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[8], vec_res);
    vec_a = _mm_loadu_ps(&a.m[12]);
    vec_res = _mm_mul_ps(vec_a, vec_b_scalar);
    _mm_storeu_ps(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t vec_a, vec_b_scalar, vec_res;
    vec_b_scalar = vdupq_n_f32(b);
    vec_a = vld1q_f32(&a.m[0]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[0], vec_res);
    vec_a = vld1q_f32(&a.m[4]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[4], vec_res);
    vec_a = vld1q_f32(&a.m[8]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[8], vec_res);
    vec_a = vld1q_f32(&a.m[12]);
    vec_res = vmulq_f32(vec_a, vec_b_scalar);
    vst1q_f32(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t vec_b_scalar = wasm_f32x4_splat(b);
    v128_t vec_a = wasm_v128_load(&a.m[0]);
    v128_t vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[0], vec_res);
    vec_a = wasm_v128_load(&a.m[4]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[4], vec_res);
    vec_a = wasm_v128_load(&a.m[8]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[8], vec_res);
    vec_a = wasm_v128_load(&a.m[12]);
    vec_res = wasm_f32x4_mul(vec_a, vec_b_scalar);
    wasm_v128_store(&dst->m[12], vec_res);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    size_t vl;
    vfloat32m1_t vec_a, vec_b_scalar, vec_res;
    float* a_ptr = (float*)&a;
    float* dst_ptr = (float*)dst;

    vl = __riscv_vsetvl_e32m1(4);
    vec_b_scalar = __riscv_vfmv_v_f_f32m1(b, vl);
    vec_a = __riscv_vle32_v_f32m1(&a_ptr[0], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[0], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[4], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[4], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[8], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[8], vec_res, vl);

    vec_a = __riscv_vle32_v_f32m1(&a_ptr[12], vl);
    vec_res = __riscv_vfmul_vv_f32m1(vec_a, vec_b_scalar, vl);
    __riscv_vse32_v_f32m1(&dst_ptr[12], vec_res, vl);
#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    __m128 vec_b_scalar = __lsx_vldrepl_w(&b, 0);
    __m128 vec_a = (__m128)__lsx_vld(&a.m[0], 0);
    __m128 vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[0], 0);
    vec_a = (__m128)__lsx_vld(&a.m[4], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[4], 0);
    vec_a = (__m128)__lsx_vld(&a.m[8], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[8], 0);
    vec_a = (__m128)__lsx_vld(&a.m[12], 0);
    vec_res = (__m128)__lsx_vfmul_s((__m128i)vec_a, (__m128i)vec_b_scalar);
    __lsx_vst((__m128i)vec_res, &dst->m[12], 0);
#else
    for (int i = 0; i < 16; ++i) {
        dst->m[i] = a.m[i] * b;
    }
#endif
}

void WMATH_MULTIPLY(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a, const WMATH_TYPE(Mat4) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    /* Row-linear-combination form:
     *   dst_row_i = a[i][0]*b_row0 + a[i][1]*b_row1 + a[i][2]*b_row2 + a[i][3]*b_row3
     * Each b row is loaded once and reused by all four output rows, and each
     * output row costs four multiply-adds plus one store -- no horizontal
     * reduction. The previous form built transposed columns with _mm_set_ps and
     * reduced every element with two _mm_hadd_ps: 16 mul + 32 hadd + 16 vmovss,
     * which measured slower than the plain scalar loop (22.2 ns vs 15.2 ns);
     * this form is 6.8 ns. */
    const __m128 b_r0 = _mm_loadu_ps(&b.m[0]);
    const __m128 b_r1 = _mm_loadu_ps(&b.m[4]);
    const __m128 b_r2 = _mm_loadu_ps(&b.m[8]);
    const __m128 b_r3 = _mm_loadu_ps(&b.m[12]);

    for (int i = 0; i < 4; i++) {
        __m128 acc = _mm_mul_ps(_mm_set1_ps(a.m[i * 4 + 0]), b_r0);
        acc = wcn_fma_mul_add_ps(_mm_set1_ps(a.m[i * 4 + 1]), b_r1, acc);
        acc = wcn_fma_mul_add_ps(_mm_set1_ps(a.m[i * 4 + 2]), b_r2, acc);
        acc = wcn_fma_mul_add_ps(_mm_set1_ps(a.m[i * 4 + 3]), b_r3, acc);
        _mm_storeu_ps(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    /* Same row-linear-combination form as the x86 branch. vfmaq_n_f32 folds the
     * broadcast into the multiply-add, so each output row is one vmulq_n_f32 plus
     * three vfmaq_n_f32 and one store. */
    const float32x4_t b_r0 = vld1q_f32(&b.m[0]);
    const float32x4_t b_r1 = vld1q_f32(&b.m[4]);
    const float32x4_t b_r2 = vld1q_f32(&b.m[8]);
    const float32x4_t b_r3 = vld1q_f32(&b.m[12]);

    for (int i = 0; i < 4; i++) {
        float32x4_t acc = vmulq_n_f32(b_r0, a.m[i * 4 + 0]);
        acc = vfmaq_n_f32(acc, b_r1, a.m[i * 4 + 1]);
        acc = vfmaq_n_f32(acc, b_r2, a.m[i * 4 + 2]);
        acc = vfmaq_n_f32(acc, b_r3, a.m[i * 4 + 3]);
        vst1q_f32(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    /* Same form; WASM SIMD has no fused multiply-add, so each term is an explicit
     * acc + a*b. */
    const v128_t b_r0 = wasm_v128_load(&b.m[0]);
    const v128_t b_r1 = wasm_v128_load(&b.m[4]);
    const v128_t b_r2 = wasm_v128_load(&b.m[8]);
    const v128_t b_r3 = wasm_v128_load(&b.m[12]);

    for (int i = 0; i < 4; i++) {
        v128_t acc = wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 0]), b_r0);
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 1]), b_r1));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 2]), b_r2));
        acc = wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(a.m[i * 4 + 3]), b_r3));
        wasm_v128_store(&dst->m[i * 4], acc);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    /* Same form; vfmacc_vf_f32m1 multiplies by a scalar and accumulates. */
    const size_t vl = __riscv_vsetvl_e32m1(4);
    const vfloat32m1_t b_r0 = __riscv_vle32_v_f32m1(&b.m[0], vl);
    const vfloat32m1_t b_r1 = __riscv_vle32_v_f32m1(&b.m[4], vl);
    const vfloat32m1_t b_r2 = __riscv_vle32_v_f32m1(&b.m[8], vl);
    const vfloat32m1_t b_r3 = __riscv_vle32_v_f32m1(&b.m[12], vl);

    for (int i = 0; i < 4; i++) {
        vfloat32m1_t acc = __riscv_vfmul_vf_f32m1(b_r0, a.m[i * 4 + 0], vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a.m[i * 4 + 1], b_r1, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a.m[i * 4 + 2], b_r2, vl);
        acc = __riscv_vfmacc_vf_f32m1(acc, a.m[i * 4 + 3], b_r3, vl);
        __riscv_vse32_v_f32m1(&dst->m[i * 4], acc, vl);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    /* Same form; LSX has no fused multiply-add, so each term is mul then add. */
    const __m128 b_r0 = (__m128)__lsx_vld(&b.m[0], 0);
    const __m128 b_r1 = (__m128)__lsx_vld(&b.m[4], 0);
    const __m128 b_r2 = (__m128)__lsx_vld(&b.m[8], 0);
    const __m128 b_r3 = (__m128)__lsx_vld(&b.m[12], 0);

    for (int i = 0; i < 4; i++) {
        __m128 acc = (__m128)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 0], 0),
                                            (__m128i)b_r0);
        acc = (__m128)__lsx_vfadd_s(
            (__m128i)acc, (__m128i)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 1], 0),
                                                 (__m128i)b_r1));
        acc = (__m128)__lsx_vfadd_s(
            (__m128i)acc, (__m128i)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 2], 0),
                                                 (__m128i)b_r2));
        acc = (__m128)__lsx_vfadd_s(
            (__m128i)acc, (__m128i)__lsx_vfmul_s((__m128i)__lsx_vldrepl_w(&a.m[i * 4 + 3], 0),
                                                 (__m128i)b_r3));
        __lsx_vst((__m128i)acc, &dst->m[i * 4], 0);
    }

#else
    dst->m[0] = a.m[0] * b.m[0] + a.m[1] * b.m[4] + a.m[2] * b.m[8] + a.m[3] * b.m[12];
    dst->m[1] = a.m[0] * b.m[1] + a.m[1] * b.m[5] + a.m[2] * b.m[9] + a.m[3] * b.m[13];
    dst->m[2] = a.m[0] * b.m[2] + a.m[1] * b.m[6] + a.m[2] * b.m[10] + a.m[3] * b.m[14];
    dst->m[3] = a.m[0] * b.m[3] + a.m[1] * b.m[7] + a.m[2] * b.m[11] + a.m[3] * b.m[15];

    dst->m[4] = a.m[4] * b.m[0] + a.m[5] * b.m[4] + a.m[6] * b.m[8] + a.m[7] * b.m[12];
    dst->m[5] = a.m[4] * b.m[1] + a.m[5] * b.m[5] + a.m[6] * b.m[9] + a.m[7] * b.m[13];
    dst->m[6] = a.m[4] * b.m[2] + a.m[5] * b.m[6] + a.m[6] * b.m[10] + a.m[7] * b.m[14];
    dst->m[7] = a.m[4] * b.m[3] + a.m[5] * b.m[7] + a.m[6] * b.m[11] + a.m[7] * b.m[15];

    dst->m[8] = a.m[8] * b.m[0] + a.m[9] * b.m[4] + a.m[10] * b.m[8] + a.m[11] * b.m[12];
    dst->m[9] = a.m[8] * b.m[1] + a.m[9] * b.m[5] + a.m[10] * b.m[9] + a.m[11] * b.m[13];
    dst->m[10] = a.m[8] * b.m[2] + a.m[9] * b.m[6] + a.m[10] * b.m[10] + a.m[11] * b.m[14];
    dst->m[11] = a.m[8] * b.m[3] + a.m[9] * b.m[7] + a.m[10] * b.m[11] + a.m[11] * b.m[15];

    dst->m[12] = a.m[12] * b.m[0] + a.m[13] * b.m[4] + a.m[14] * b.m[8] + a.m[15] * b.m[12];
    dst->m[13] = a.m[12] * b.m[1] + a.m[13] * b.m[5] + a.m[14] * b.m[9] + a.m[15] * b.m[13];
    dst->m[14] = a.m[12] * b.m[2] + a.m[13] * b.m[6] + a.m[14] * b.m[10] + a.m[15] * b.m[14];
    dst->m[15] = a.m[12] * b.m[3] + a.m[13] * b.m[7] + a.m[14] * b.m[11] + a.m[15] * b.m[15];
#endif
}

void WMATH_INVERSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    /* Four-lane adjugate. Lane r of every vector below is the r-th entry of a
     * cofactor row, so the whole adjugate is built with 97 vector operations
     * (measured) against 324 scalar instructions -- and, unlike the earlier
     * attempts, with no local arrays, because MSVC spills __m128 arrays to the
     * stack (41 movaps + 28 stack accesses, which cost more than the ALU work
     * saved).
     *
     * The structural saving that makes this fit in registers: only THREE row
     * pairs are ever needed.
     *     cofactor row 0 and 1 -> pair (R2,R3)
     *     cofactor row 2       -> pair (R1,R3)
     *     cofactor row 3       -> pair (R1,R2)
     * Each pair gives three minor vectors; each cofactor row is then a single
     * three-term combination with the remaining row, so the minors cost 9
     * vector ops per pair instead of 9 per cofactor.
     *
     * The two index facts that broke every previous attempt:
     *   - Cc (the vector for cofactor row c) is ROW c of the cofactor matrix,
     *     so it is COLUMN c of the inverse; a transpose is needed on the way
     *     out because dst is row-major.
     *   - det is therefore hsum(R0 * C0): the contraction runs along row 0 of
     *     both A and C. Summing Rc * Cc over c is wrong.
     *
     * Shift helper: WCN_SH(v,d) puts column (r+d)&3 of v in lane r. */
#define WCN_SH(v, d)                                                                  \
    _mm_shuffle_ps((v), (v),                                                          \
                   ((d) == 1) ? _MM_SHUFFLE(0, 3, 2, 1)                               \
                              : (((d) == 2) ? _MM_SHUFFLE(1, 0, 3, 2) : _MM_SHUFFLE(2, 1, 0, 3)))

    const __m128 R0 = _mm_loadu_ps(&a.m[0]);
    const __m128 R1 = _mm_loadu_ps(&a.m[4]);
    const __m128 R2 = _mm_loadu_ps(&a.m[8]);
    const __m128 R3 = _mm_loadu_ps(&a.m[12]);

    /* Three minor vectors per row pair. */
    /* The three shifted copies of each row are computed ONCE and shared. R1
     * takes part in two of the three row pairs, so a per-pair expansion shuffles
     * it twice; the disassembly showed 14 shufps where 12 suffice, plus the
     * extra register pressure that comes with the duplicates. */
    /* Shifted rows, built on demand rather than all up front: every named
     * vector here stays live until its last use, and MSVC spills to xmm6..xmm15
     * once more than ~12 are live (that shows up as 20 save/restore instructions
     * in the prologue). Ordering the work by minor group keeps at most three
     * minors live at a time.
     *
     * The sign convention forces one ordering constraint: det uses the SIGNED
     * cofactor row 0, so that row must be built before the divisor and the
     * divisor before the scaling. */

    /* ---- (R2,R3) group: serves cofactor rows 0 and 1 ---- */
    const __m128 R2a = WCN_SH(R2, 1), R2b = WCN_SH(R2, 2), R2c = WCN_SH(R2, 3);
    const __m128 R3a = WCN_SH(R3, 1), R3b = WCN_SH(R3, 2), R3c = WCN_SH(R3, 3);
    const __m128 R0a = WCN_SH(R0, 1), R0b = WCN_SH(R0, 2), R0c = WCN_SH(R0, 3);

    const __m128 p0 = wcn_fma_mul_sub_ps(R2b, R3c, _mm_mul_ps(R2c, R3b));
    const __m128 p1 = wcn_fma_mul_sub_ps(R2a, R3c, _mm_mul_ps(R2c, R3a));
    const __m128 p2 = wcn_fma_mul_sub_ps(R2a, R3b, _mm_mul_ps(R2b, R3a));

    /* Cofactor row c = sh(P,1)*m0 - sh(P,2)*m1 + sh(P,3)*m2, P the remaining
     * row. */
/* FMA form: one multiply and two fused multiply-adds instead of three multiplies
 * plus two separate adds. On this microarch every mulps competes for the same
 * issue port, so the multiply count is what sets the floor. */
#define WCN_COMB(s1, s2, s3, m0, m1, m2)                                              \
    wcn_fma_mul_add_ps((s3), (m2),                                                     \
                       wcn_fma_neg_mul_add_ps((s2), (m1), _mm_mul_ps((s1), (m0))))

    __m128 u0 = WCN_COMB(WCN_SH(R1, 1), WCN_SH(R1, 2), WCN_SH(R1, 3), p0, p1, p2);
    __m128 u1 = WCN_COMB(R0a, R0b, R0c, p0, p1, p2);

    /* Signed row 0, then det = hsum(R0 * signed row 0). */
    const __m128 sgE = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);
    const __m128 sgO = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
    u0 = _mm_mul_ps(u0, sgE);
    u1 = _mm_mul_ps(u1, sgO);

    const __m128 s0 = _mm_mul_ps(R0, u0);
    const __m128 s1 = _mm_add_ps(s0, _mm_shuffle_ps(s0, s0, _MM_SHUFFLE(2, 3, 0, 1)));
    const __m128 s2 = _mm_add_ps(s1, _mm_shuffle_ps(s1, s1, _MM_SHUFFLE(1, 0, 3, 2)));
    const __m128 dinv = _mm_div_ps(_mm_set1_ps(1.0f), s2);

    /* u0/u1 already carry the cofactor sign, so the divisor here is plain.
     * Column 2 and 3 still need the sign applied when they are formed. */
    const __m128 dE = dinv;
    const __m128 dO = _mm_mul_ps(dinv, sgE);
    const __m128 i0 = _mm_mul_ps(u0, dinv);
    const __m128 i1 = _mm_mul_ps(u1, dinv);

    /* ---- (R1,R3) group: cofactor row 2 ---- */
    const __m128 R1a = WCN_SH(R1, 1), R1b = WCN_SH(R1, 2), R1c = WCN_SH(R1, 3);
    const __m128 q0 = wcn_fma_mul_sub_ps(R1b, R3c, _mm_mul_ps(R1c, R3b));
    const __m128 q1 = wcn_fma_mul_sub_ps(R1a, R3c, _mm_mul_ps(R1c, R3a));
    const __m128 q2 = wcn_fma_mul_sub_ps(R1a, R3b, _mm_mul_ps(R1b, R3a));
    const __m128 i2 = _mm_mul_ps(
        _mm_mul_ps(WCN_COMB(R0a, R0b, R0c, q0, q1, q2), sgE), dinv);

    /* ---- (R1,R2) group: cofactor row 3 ---- */
    const __m128 R2a2 = WCN_SH(R2, 1), R2b2 = WCN_SH(R2, 2), R2c2 = WCN_SH(R2, 3);
    const __m128 r0 = wcn_fma_mul_sub_ps(R1b, R2c2, _mm_mul_ps(R1c, R2b2));
    const __m128 r1 = wcn_fma_mul_sub_ps(R1a, R2c2, _mm_mul_ps(R1c, R2a2));
    const __m128 r2 = wcn_fma_mul_sub_ps(R1a, R2b2, _mm_mul_ps(R1b, R2a2));
    const __m128 i3 = _mm_mul_ps(
        _mm_mul_ps(WCN_COMB(R0a, R0b, R0c, r0, r1, r2), sgO), dinv);

#undef WCN_COMB

    /* i_c is column c of the result; dst is row-major, so transpose out. */
    const __m128 t0 = _mm_unpacklo_ps(i0, i1);
    const __m128 t1 = _mm_unpackhi_ps(i0, i1);
    const __m128 t2 = _mm_unpacklo_ps(i2, i3);
    const __m128 t3 = _mm_unpackhi_ps(i2, i3);
    _mm_storeu_ps(&dst->m[0],  _mm_movelh_ps(t0, t2));
    _mm_storeu_ps(&dst->m[4],  _mm_movehl_ps(t2, t0));
    _mm_storeu_ps(&dst->m[8],  _mm_movelh_ps(t1, t3));
    _mm_storeu_ps(&dst->m[12], _mm_movehl_ps(t3, t1));

#undef WCN_SH

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    /* Same pair-sharing formulation as the SSE branch above; only the shift and
     * sign-mask spellings differ. Lane layout and the det contraction are
     * identical, so the correctness argument carries over unchanged. */
    const float32x4_t R0 = vld1q_f32(&a.m[0]);
    const float32x4_t R1 = vld1q_f32(&a.m[4]);
    const float32x4_t R2 = vld1q_f32(&a.m[8]);
    const float32x4_t R3 = vld1q_f32(&a.m[12]);

    /* Two +/-1 masks: lanes 0..3 = (+1,-1,+1,-1) and its negation. */
    const float32x4_t sgE = (float32x4_t){1.0f, -1.0f, 1.0f, -1.0f};
    const float32x4_t sgO = vnegq_f32(sgE);

    /* minors of (R2,R3) */
    const float32x4_t R2a = vextq_f32(R2, R2, 1), R2b = vextq_f32(R2, R2, 2),
                      R2c = vextq_f32(R2, R2, 3);
    const float32x4_t R3a = vextq_f32(R3, R3, 1), R3b = vextq_f32(R3, R3, 2),
                      R3c = vextq_f32(R3, R3, 3);
    const float32x4_t R0a = vextq_f32(R0, R0, 1), R0b = vextq_f32(R0, R0, 2),
                      R0c = vextq_f32(R0, R0, 3);

    const float32x4_t p0 = vfmsq_f32(R2b, R3c, vmulq_f32(R2c, R3b));
    const float32x4_t p1 = vfmsq_f32(R2a, R3c, vmulq_f32(R2c, R3a));
    const float32x4_t p2 = vfmsq_f32(R2a, R3b, vmulq_f32(R2b, R3a));

#define WCN_COMB4(s1, s2, s3, m0, m1, m2)                                             \
    vaddq_f32(vfmsq_f32((s1), (m0), vmulq_f32((s2), (m1))),                 \
              vmulq_f32((s3), (m2)))

    const float32x4_t R1a0 = vextq_f32(R1, R1, 1);
    const float32x4_t R1b0 = vextq_f32(R1, R1, 2);
    const float32x4_t R1c0 = vextq_f32(R1, R1, 3);
    float32x4_t u0 = WCN_COMB4(R1a0, R1b0, R1c0, p0, p1, p2);
    float32x4_t u1 = WCN_COMB4(R0a, R0b, R0c, p0, p1, p2);

    u0 = vmulq_f32(u0, sgE);
    u1 = vmulq_f32(u1, sgO);

    /* det = hsum(R0 * signed row 0). */
    float32x4_t s0 = vmulq_f32(R0, u0);
    s0 = vaddq_f32(s0, vextq_f32(s0, s0, 2));
    s0 = vaddq_f32(s0, vextq_f32(s0, s0, 1));
    const float32x4_t dinv = vdupq_n_f32(1.0f / vgetq_lane_f32(s0, 0));

    const float32x4_t i0 = vmulq_f32(u0, dinv);
    const float32x4_t i1 = vmulq_f32(u1, dinv);

    /* minors of (R1,R3) -> cofactor row 2 */
    const float32x4_t R1a = vextq_f32(R1, R1, 1), R1b = vextq_f32(R1, R1, 2),
                      R1c = vextq_f32(R1, R1, 3);
    const float32x4_t q0 = vfmsq_f32(R1b, R3c, vmulq_f32(R1c, R3b));
    const float32x4_t q1 = vfmsq_f32(R1a, R3c, vmulq_f32(R1c, R3a));
    const float32x4_t q2 = vfmsq_f32(R1a, R3b, vmulq_f32(R1b, R3a));
    const float32x4_t i2 = vmulq_f32(vmulq_f32(WCN_COMB4(R0a, R0b, R0c, q0, q1, q2), sgE),
                                     dinv);

    /* minors of (R1,R2) -> cofactor row 3 */
    const float32x4_t r0 = vfmsq_f32(R1b, R2c, vmulq_f32(R1c, R2b));
    const float32x4_t r1 = vfmsq_f32(R1a, R2c, vmulq_f32(R1c, R2a));
    const float32x4_t r2 = vfmsq_f32(R1a, R2b, vmulq_f32(R1b, R2a));
    const float32x4_t i3 = vmulq_f32(vmulq_f32(WCN_COMB4(R0a, R0b, R0c, r0, r1, r2), sgO),
                                     dinv);

#undef WCN_COMB4

    /* i_c is column c; transpose into row-major dst. */
    const float32x4x2_t t01 = vtrnq_f32(i0, i1);
    const float32x4x2_t t23 = vtrnq_f32(i2, i3);
    vst1q_f32(&dst->m[0],  vcombine_f32(vget_low_f32(t01.val[0]),  vget_low_f32(t23.val[0])));
    vst1q_f32(&dst->m[4],  vcombine_f32(vget_low_f32(t01.val[1]),  vget_low_f32(t23.val[1])));
    vst1q_f32(&dst->m[8],  vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0])));
    vst1q_f32(&dst->m[12], vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1])));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    /* Same pair-sharing adjugate as the SSE/NEON branches. WASM SIMD has no fused
     * multiply-add, so WCN_COMB8 spells out multiply then add. Sign masks build
     * from a splat and a select-free constant literal. */
    const v128_t R0 = wasm_v128_load(&a.m[0]);
    const v128_t R1 = wasm_v128_load(&a.m[4]);
    const v128_t R2 = wasm_v128_load(&a.m[8]);
    const v128_t R3 = wasm_v128_load(&a.m[12]);

    /* lane r -> column (r+d) & 3 */
#define WCN_WSH(v, i0, i1, i2, i3) wasm_i32x4_shuffle((v), (v), i0, i1, i2, i3)

    const v128_t sgE = wasm_f32x4_make(1.0f, -1.0f, 1.0f, -1.0f);
    const v128_t sgO = wasm_f32x4_make(-1.0f, 1.0f, -1.0f, 1.0f);

    const v128_t R0a = WCN_WSH(R0, 1, 2, 3, 0), R0b = WCN_WSH(R0, 2, 3, 0, 1),
                 R0c = WCN_WSH(R0, 3, 0, 1, 2);
    const v128_t R1a = WCN_WSH(R1, 1, 2, 3, 0), R1b = WCN_WSH(R1, 2, 3, 0, 1),
                 R1c = WCN_WSH(R1, 3, 0, 1, 2);
    const v128_t R2a = WCN_WSH(R2, 1, 2, 3, 0), R2b = WCN_WSH(R2, 2, 3, 0, 1),
                 R2c = WCN_WSH(R2, 3, 0, 1, 2);
    const v128_t R3a = WCN_WSH(R3, 1, 2, 3, 0), R3b = WCN_WSH(R3, 2, 3, 0, 1),
                 R3c = WCN_WSH(R3, 3, 0, 1, 2);

#undef WCN_WSH

#define WCN_WCOMB(s1, s2, s3, m0, m1, m2)                                            \
    wasm_f32x4_add(wasm_f32x4_sub(wasm_f32x4_mul((s1), (m0)), wasm_f32x4_mul((s2), (m1))), \
                   wasm_f32x4_mul((s3), (m2)))

    const v128_t p0 = wasm_f32x4_sub(wasm_f32x4_mul(R2b, R3c), wasm_f32x4_mul(R2c, R3b));
    const v128_t p1 = wasm_f32x4_sub(wasm_f32x4_mul(R2a, R3c), wasm_f32x4_mul(R2c, R3a));
    const v128_t p2 = wasm_f32x4_sub(wasm_f32x4_mul(R2a, R3b), wasm_f32x4_mul(R2b, R3a));

    const v128_t u0 = WCN_WCOMB(R1a, R1b, R1c, p0, p1, p2);
    const v128_t u1 = WCN_WCOMB(R0a, R0b, R0c, p0, p1, p2);

    const v128_t C0 = wasm_f32x4_mul(u0, sgE);
    const v128_t C1 = wasm_f32x4_mul(u1, sgO);

    /* det = hsum(R0 * signed cofactor row 0) */
    v128_t s0 = wasm_f32x4_mul(R0, C0);
    s0 = wasm_f32x4_add(s0, wasm_i32x4_shuffle(s0, s0, 2, 3, 0, 1));
    s0 = wasm_f32x4_add(s0, wasm_i32x4_shuffle(s0, s0, 1, 0, 3, 2));
    const v128_t dinv = wasm_f32x4_div(wasm_f32x4_splat(1.0f),
                                       wasm_i32x4_shuffle(s0, s0, 0, 0, 0, 0));

    const v128_t i0 = wasm_f32x4_mul(C0, dinv);
    const v128_t i1 = wasm_f32x4_mul(C1, dinv);

    const v128_t q0 = wasm_f32x4_sub(wasm_f32x4_mul(R1b, R3c), wasm_f32x4_mul(R1c, R3b));
    const v128_t q1 = wasm_f32x4_sub(wasm_f32x4_mul(R1a, R3c), wasm_f32x4_mul(R1c, R3a));
    const v128_t q2 = wasm_f32x4_sub(wasm_f32x4_mul(R1a, R3b), wasm_f32x4_mul(R1b, R3a));
    const v128_t i2 =
        wasm_f32x4_mul(wasm_f32x4_mul(WCN_WCOMB(R0a, R0b, R0c, q0, q1, q2), sgE), dinv);

    const v128_t r0 = wasm_f32x4_sub(wasm_f32x4_mul(R1b, R2c), wasm_f32x4_mul(R1c, R2b));
    const v128_t r1 = wasm_f32x4_sub(wasm_f32x4_mul(R1a, R2c), wasm_f32x4_mul(R1c, R2a));
    const v128_t r2 = wasm_f32x4_sub(wasm_f32x4_mul(R1a, R2b), wasm_f32x4_mul(R1b, R2a));
    const v128_t i3 =
        wasm_f32x4_mul(wasm_f32x4_mul(WCN_WCOMB(R0a, R0b, R0c, r0, r1, r2), sgO), dinv);

#undef WCN_WCOMB

    /* i_c is column c; transpose into row-major dst. */
    const v128_t t0 = wasm_i32x4_shuffle(i0, i1, 0, 4, 1, 5);
    const v128_t t1 = wasm_i32x4_shuffle(i0, i1, 2, 6, 3, 7);
    const v128_t t2 = wasm_i32x4_shuffle(i2, i3, 0, 4, 1, 5);
    const v128_t t3 = wasm_i32x4_shuffle(i2, i3, 2, 6, 3, 7);
    wasm_v128_store(&dst->m[0], wasm_i32x4_shuffle(t0, t2, 0, 1, 4, 5));
    wasm_v128_store(&dst->m[4], wasm_i32x4_shuffle(t0, t2, 2, 3, 6, 7));
    wasm_v128_store(&dst->m[8], wasm_i32x4_shuffle(t1, t3, 0, 1, 4, 5));
    wasm_v128_store(&dst->m[12], wasm_i32x4_shuffle(t1, t3, 2, 3, 6, 7));

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    /* Same pair-sharing adjugate. The lane shift is a vrgather with an index
     * vector, because RVV has no single-instruction 4-lane rotate. */
    const size_t vl = __riscv_vsetvl_e32m1(4);
    const vfloat32m1_t R0 = __riscv_vle32_v_f32m1(&a.m[0], vl);
    const vfloat32m1_t R1 = __riscv_vle32_v_f32m1(&a.m[4], vl);
    const vfloat32m1_t R2 = __riscv_vle32_v_f32m1(&a.m[8], vl);
    const vfloat32m1_t R3 = __riscv_vle32_v_f32m1(&a.m[12], vl);

    /*
     * vcreate is not a public intrinsic; materialise the rotation indices from memory instead.
     * This is the supported construction route and produces the identical index vectors.
     */
    static const uint32_t ix1_v[4] = {1, 2, 3, 0};
    static const uint32_t ix2_v[4] = {2, 3, 0, 1};
    static const uint32_t ix3_v[4] = {3, 0, 1, 2};
    const vuint32m1_t ix1 = __riscv_vle32_v_u32m1(ix1_v, vl);
    const vuint32m1_t ix2 = __riscv_vle32_v_u32m1(ix2_v, vl);
    const vuint32m1_t ix3 = __riscv_vle32_v_u32m1(ix3_v, vl);

#define WCN_RSH1(v) __riscv_vrgather_vv_f32m1((v), ix1, vl)
#define WCN_RSH2(v) __riscv_vrgather_vv_f32m1((v), ix2, vl)
#define WCN_RSH3(v) __riscv_vrgather_vv_f32m1((v), ix3, vl)

    static const float sgE_v[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    static const float sgO_v[4] = {-1.0f, 1.0f, -1.0f, 1.0f};
    const vfloat32m1_t sgE = __riscv_vle32_v_f32m1(sgE_v, vl);
    const vfloat32m1_t sgO = __riscv_vle32_v_f32m1(sgO_v, vl);

    const vfloat32m1_t R0a = WCN_RSH1(R0), R0b = WCN_RSH2(R0), R0c = WCN_RSH3(R0);
    const vfloat32m1_t R1a = WCN_RSH1(R1), R1b = WCN_RSH2(R1), R1c = WCN_RSH3(R1);
    const vfloat32m1_t R2a = WCN_RSH1(R2), R2b = WCN_RSH2(R2), R2c = WCN_RSH3(R2);
    const vfloat32m1_t R3a = WCN_RSH1(R3), R3b = WCN_RSH2(R3), R3c = WCN_RSH3(R3);

#undef WCN_RSH1
#undef WCN_RSH2
#undef WCN_RSH3

#define WCN_RCOMB(s1, s2, s3, m0, m1, m2)                                            \
    __riscv_vfadd_vv_f32m1(                                                          \
        __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1((s1), (m0), vl),                \
                               __riscv_vfmul_vv_f32m1((s2), (m1), vl), vl),           \
        __riscv_vfmul_vv_f32m1((s3), (m2), vl), vl)

    const vfloat32m1_t p0 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R2b, R3c, vl),
                                                   __riscv_vfmul_vv_f32m1(R2c, R3b, vl), vl);
    const vfloat32m1_t p1 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R2a, R3c, vl),
                                                   __riscv_vfmul_vv_f32m1(R2c, R3a, vl), vl);
    const vfloat32m1_t p2 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R2a, R3b, vl),
                                                   __riscv_vfmul_vv_f32m1(R2b, R3a, vl), vl);

    const vfloat32m1_t u0 = WCN_RCOMB(R1a, R1b, R1c, p0, p1, p2);
    const vfloat32m1_t u1 = WCN_RCOMB(R0a, R0b, R0c, p0, p1, p2);

    const vfloat32m1_t C0 = __riscv_vfmul_vv_f32m1(u0, sgE, vl);
    const vfloat32m1_t C1 = __riscv_vfmul_vv_f32m1(u1, sgO, vl);

    vfloat32m1_t s0 = __riscv_vfmul_vv_f32m1(R0, C0, vl);
    s0 = __riscv_vfadd_vv_f32m1(s0, __riscv_vrgather_vv_f32m1(s0, ix2, vl), vl);
    s0 = __riscv_vfadd_vv_f32m1(s0, __riscv_vrgather_vv_f32m1(s0, ix1, vl), vl);
    const vfloat32m1_t dinv =
        __riscv_vfdiv_vf_f32m1(__riscv_vfmv_v_f_f32m1(1.0f, vl),
                               __riscv_vfmv_f_s_f32m1_f32(s0), vl);

    const vfloat32m1_t i0 = __riscv_vfmul_vv_f32m1(C0, dinv, vl);
    const vfloat32m1_t i1 = __riscv_vfmul_vv_f32m1(C1, dinv, vl);

    const vfloat32m1_t q0 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1b, R3c, vl),
                                                   __riscv_vfmul_vv_f32m1(R1c, R3b, vl), vl);
    const vfloat32m1_t q1 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1a, R3c, vl),
                                                   __riscv_vfmul_vv_f32m1(R1c, R3a, vl), vl);
    const vfloat32m1_t q2 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1a, R3b, vl),
                                                   __riscv_vfmul_vv_f32m1(R1b, R3a, vl), vl);
    const vfloat32m1_t i2 =
        __riscv_vfmul_vv_f32m1(__riscv_vfmul_vv_f32m1(WCN_RCOMB(R0a, R0b, R0c, q0, q1, q2),
                                                      sgE, vl), dinv, vl);

    const vfloat32m1_t r0 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1b, R2c, vl),
                                                   __riscv_vfmul_vv_f32m1(R1c, R2b, vl), vl);
    const vfloat32m1_t r1 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1a, R2c, vl),
                                                   __riscv_vfmul_vv_f32m1(R1c, R2a, vl), vl);
    const vfloat32m1_t r2 = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(R1a, R2b, vl),
                                                   __riscv_vfmul_vv_f32m1(R1b, R2a, vl), vl);
    const vfloat32m1_t i3 =
        __riscv_vfmul_vv_f32m1(__riscv_vfmul_vv_f32m1(WCN_RCOMB(R0a, R0b, R0c, r0, r1, r2),
                                                      sgO, vl), dinv, vl);

#undef WCN_RCOMB

    /* i_c is column c; scatter to row-major dst element by element. RVV has no
     * cheap 4x4 float transpose, and this runs once per call. */
    float c0[4], c1[4], c2[4], c3[4];
    __riscv_vse32_v_f32m1(c0, i0, vl);
    __riscv_vse32_v_f32m1(c1, i1, vl);
    __riscv_vse32_v_f32m1(c2, i2, vl);
    __riscv_vse32_v_f32m1(c3, i3, vl);
    for (int rr = 0; rr < 4; rr++) {
        dst->m[rr * 4 + 0] = c0[rr];
        dst->m[rr * 4 + 1] = c1[rr];
        dst->m[rr * 4 + 2] = c2[rr];
        dst->m[rr * 4 + 3] = c3[rr];
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    /* Same pair-sharing adjugate. LSX spells the lane rotate as vshuf4i_w with a
     * 1,2,3,0 / 2,3,0,1 / 3,0,1,2 element pattern. */
    const __m128 R0 = (__m128)__lsx_vld(&a.m[0], 0);
    const __m128 R1 = (__m128)__lsx_vld(&a.m[4], 0);
    const __m128 R2 = (__m128)__lsx_vld(&a.m[8], 0);
    const __m128 R3 = (__m128)__lsx_vld(&a.m[12], 0);

    /* vshuf4i_w immediate layout: two bits per lane, lane i takes src[imm>>(2i)]. */
#define WCN_LSH1(v) (__m128)__lsx_vshuf4i_w((__m128i)(v), 0x39) /* 1,2,3,0 */
#define WCN_LSH2(v) (__m128)__lsx_vshuf4i_w((__m128i)(v), 0x4E) /* 2,3,0,1 */
#define WCN_LSH3(v) (__m128)__lsx_vshuf4i_w((__m128i)(v), 0x93) /* 3,0,1,2 */

    const __m128 R0a = WCN_LSH1(R0), R0b = WCN_LSH2(R0), R0c = WCN_LSH3(R0);
    const __m128 R1a = WCN_LSH1(R1), R1b = WCN_LSH2(R1), R1c = WCN_LSH3(R1);
    const __m128 R2a = WCN_LSH1(R2), R2b = WCN_LSH2(R2), R2c = WCN_LSH3(R2);
    const __m128 R3a = WCN_LSH1(R3), R3b = WCN_LSH2(R3), R3c = WCN_LSH3(R3);

#undef WCN_LSH1
#undef WCN_LSH2
#undef WCN_LSH3

    /* +/-1 masks as float bit patterns. */
    const float kSgE[4] = {1.0f, -1.0f, 1.0f, -1.0f};
    const float kSgO[4] = {-1.0f, 1.0f, -1.0f, 1.0f};
    const __m128 vSgE = (__m128)__lsx_vld(kSgE, 0);
    const __m128 vSgO = (__m128)__lsx_vld(kSgO, 0);

#define WCN_LCOMB(s1, s2, s3, m0, m1, m2)                                            \
    (__m128)__lsx_vfadd_s(                                                           \
        (__m128i)(__m128)__lsx_vfsub_s(                                              \
            (__m128i)(__m128)__lsx_vfmul_s((__m128i)(s1), (__m128i)(m0)),            \
            (__m128i)(__m128)__lsx_vfmul_s((__m128i)(s2), (__m128i)(m1))),           \
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)(s3), (__m128i)(m2)))

    const __m128 p0 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2b, (__m128i)R3c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2c, (__m128i)R3b));
    const __m128 p1 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2a, (__m128i)R3c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2c, (__m128i)R3a));
    const __m128 p2 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2a, (__m128i)R3b),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R2b, (__m128i)R3a));

    const __m128 u0 = WCN_LCOMB(R1a, R1b, R1c, p0, p1, p2);
    const __m128 u1 = WCN_LCOMB(R0a, R0b, R0c, p0, p1, p2);

    const __m128 C0 = (__m128)__lsx_vfmul_s((__m128i)u0, (__m128i)vSgE);
    const __m128 C1 = (__m128)__lsx_vfmul_s((__m128i)u1, (__m128i)vSgO);

    __m128 s0 = (__m128)__lsx_vfmul_s((__m128i)R0, (__m128i)C0);
    s0 = (__m128)__lsx_vfadd_s((__m128i)s0, (__m128i)__lsx_vshuf4i_w((__m128i)s0, 0x4E));
    s0 = (__m128)__lsx_vfadd_s((__m128i)s0, (__m128i)__lsx_vshuf4i_w((__m128i)s0, 0x39));
    const float det_s = wcn_lsx_extract_f(s0, 0);
    const float kOne = 1.0f;
    const __m128 dinv = (__m128)__lsx_vfdiv_s((__m128i)__lsx_vldrepl_w((void *)&kOne, 0),
                                              (__m128i)__lsx_vldrepl_w((void *)&det_s, 0));

    const __m128 i0 = (__m128)__lsx_vfmul_s((__m128i)C0, (__m128i)dinv);
    const __m128 i1 = (__m128)__lsx_vfmul_s((__m128i)C1, (__m128i)dinv);

    const __m128 q0 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1b, (__m128i)R3c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1c, (__m128i)R3b));
    const __m128 q1 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1a, (__m128i)R3c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1c, (__m128i)R3a));
    const __m128 q2 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1a, (__m128i)R3b),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1b, (__m128i)R3a));
    const __m128 i2 = (__m128)__lsx_vfmul_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)WCN_LCOMB(R0a, R0b, R0c, q0, q1, q2),
                                       (__m128i)vSgE),
        (__m128i)dinv);

    const __m128 r0 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1b, (__m128i)R2c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1c, (__m128i)R2b));
    const __m128 r1 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1a, (__m128i)R2c),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1c, (__m128i)R2a));
    const __m128 r2 = (__m128)__lsx_vfsub_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1a, (__m128i)R2b),
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)R1b, (__m128i)R2a));
    const __m128 i3 = (__m128)__lsx_vfmul_s(
        (__m128i)(__m128)__lsx_vfmul_s((__m128i)WCN_LCOMB(R0a, R0b, R0c, r0, r1, r2),
                                       (__m128i)vSgO),
        (__m128i)dinv);

#undef WCN_LCOMB

    float c0[4], c1[4], c2[4], c3[4];
    __lsx_vst((__m128i)i0, c0, 0);
    __lsx_vst((__m128i)i1, c1, 0);
    __lsx_vst((__m128i)i2, c2, 0);
    __lsx_vst((__m128i)i3, c3, 0);
    for (int rr = 0; rr < 4; rr++) {
        dst->m[rr * 4 + 0] = c0[rr];
        dst->m[rr * 4 + 1] = c1[rr];
        dst->m[rr * 4 + 2] = c2[rr];
        dst->m[rr * 4 + 3] = c3[rr];
    }

#else
    /* Scalar fallback: shared pool of eighteen 2x2 minors.
     *
     * The sixteen adjugate entries are three-term combinations over the pool, so
     * nothing is computed twice. This replaced a form with twenty-four tmp_*
     * products and fully expanded cofactors (~120 multiplies, 33.7 ns); the pool
     * form is 17.8 ns.
     *
     * Note the ordering: det needs COLUMN 0 of the adjugate (A0, A4, A8, A12)
     * because adj[r][c] = C(c,r), i.e. the cofactors are transposed. */
    const float* m = a.m;

    const float M0  = m[10] * m[15] - m[11] * m[14];
    const float M1  = m[9]  * m[15] - m[11] * m[13];
    const float M2  = m[9]  * m[14] - m[10] * m[13];
    const float M3  = m[6]  * m[15] - m[7]  * m[14];
    const float M4  = m[5]  * m[15] - m[7]  * m[13];
    const float M5  = m[5]  * m[14] - m[6]  * m[13];
    const float M6  = m[6]  * m[11] - m[7]  * m[10];
    const float M7  = m[5]  * m[11] - m[7]  * m[9];
    const float M8  = m[5]  * m[10] - m[6]  * m[9];
    const float M9  = m[8]  * m[15] - m[11] * m[12];
    const float M10 = m[8]  * m[14] - m[10] * m[12];
    const float M11 = m[4]  * m[15] - m[7]  * m[12];
    const float M12 = m[4]  * m[14] - m[6]  * m[12];
    const float M13 = m[4]  * m[11] - m[7]  * m[8];
    const float M14 = m[4]  * m[10] - m[6]  * m[8];
    const float M15 = m[8]  * m[13] - m[9]  * m[12];
    const float M16 = m[4]  * m[13] - m[5]  * m[12];
    const float M17 = m[4]  * m[9]  - m[5]  * m[8];

    const float A0  =  (m[5] * M0 - m[6] * M1 + m[7] * M2);
    const float A1  = -(m[1] * M0 - m[2] * M1 + m[3] * M2);
    const float A2  =  (m[1] * M3 - m[2] * M4 + m[3] * M5);
    const float A3  = -(m[1] * M6 - m[2] * M7 + m[3] * M8);
    const float A4  = -(m[4] * M0 - m[6] * M9 + m[7] * M10);
    const float A5  =  (m[0] * M0 - m[2] * M9 + m[3] * M10);
    const float A6  = -(m[0] * M3 - m[2] * M11 + m[3] * M12);
    const float A7  =  (m[0] * M6 - m[2] * M13 + m[3] * M14);
    const float A8  =  (m[4] * M1 - m[5] * M9 + m[7] * M15);
    const float A9  = -(m[0] * M1 - m[1] * M9 + m[3] * M15);
    const float A10 =  (m[0] * M4 - m[1] * M11 + m[3] * M16);
    const float A11 = -(m[0] * M7 - m[1] * M13 + m[3] * M17);
    const float A12 = -(m[4] * M2 - m[5] * M10 + m[6] * M15);
    const float A13 =  (m[0] * M2 - m[1] * M10 + m[2] * M15);
    const float A14 = -(m[0] * M5 - m[1] * M12 + m[2] * M16);
    const float A15 =  (m[0] * M8 - m[1] * M14 + m[2] * M17);

    const float det = m[0] * A0 + m[1] * A4 + m[2] * A8 + m[3] * A12;
    const float d = 1.0f / det;

    dst->m[0]  = A0  * d;
    dst->m[1]  = A1  * d;
    dst->m[2]  = A2  * d;
    dst->m[3]  = A3  * d;
    dst->m[4]  = A4  * d;
    dst->m[5]  = A5  * d;
    dst->m[6]  = A6  * d;
    dst->m[7]  = A7  * d;
    dst->m[8]  = A8  * d;
    dst->m[9]  = A9  * d;
    dst->m[10] = A10 * d;
    dst->m[11] = A11 * d;
    dst->m[12] = A12 * d;
    dst->m[13] = A13 * d;
    dst->m[14] = A14 * d;
    dst->m[15] = A15 * d;
#endif
}

void WMATH_TRANSPOSE(Mat4)(DST_MAT4, const WMATH_TYPE(Mat4) a) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    __m128 row0 = _mm_loadu_ps(&a.m[0]);
    __m128 row1 = _mm_loadu_ps(&a.m[4]);
    __m128 row2 = _mm_loadu_ps(&a.m[8]);
    __m128 row3 = _mm_loadu_ps(&a.m[12]);

    __m128 tmp0 = _mm_unpacklo_ps(row0, row1);
    __m128 tmp1 = _mm_unpacklo_ps(row2, row3);
    __m128 tmp2 = _mm_unpackhi_ps(row0, row1);
    __m128 tmp3 = _mm_unpackhi_ps(row2, row3);

    __m128 col0 = _mm_movelh_ps(tmp0, tmp1);
    __m128 col1 = _mm_movehl_ps(tmp1, tmp0);
    __m128 col2 = _mm_movelh_ps(tmp2, tmp3);
    __m128 col3 = _mm_movehl_ps(tmp3, tmp2);

    _mm_storeu_ps(&dst->m[0], col0);
    _mm_storeu_ps(&dst->m[4], col1);
    _mm_storeu_ps(&dst->m[8], col2);
    _mm_storeu_ps(&dst->m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    float32x4_t row0 = vld1q_f32(&a.m[0]);
    float32x4_t row1 = vld1q_f32(&a.m[4]);
    float32x4_t row2 = vld1q_f32(&a.m[8]);
    float32x4_t row3 = vld1q_f32(&a.m[12]);

    float32x4x2_t t01 = vtrnq_f32(row0, row1);
    float32x4x2_t t23 = vtrnq_f32(row2, row3);

    float32x4_t col0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    float32x4_t col1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    float32x4_t col2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    float32x4_t col3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

    vst1q_f32(&dst->m[0], col0);
    vst1q_f32(&dst->m[4], col1);
    vst1q_f32(&dst->m[8], col2);
    vst1q_f32(&dst->m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    v128_t row0 = wasm_v128_load(&a.m[0]);
    v128_t row1 = wasm_v128_load(&a.m[4]);
    v128_t row2 = wasm_v128_load(&a.m[8]);
    v128_t row3 = wasm_v128_load(&a.m[12]);

    v128_t tmp0 = wasm_i32x4_shuffle(row0, row1, 0, 1, 0, 1);
    v128_t tmp1 = wasm_i32x4_shuffle(row2, row3, 0, 1, 0, 1);
    v128_t tmp2 = wasm_i32x4_shuffle(row0, row1, 2, 3, 2, 3);
    v128_t tmp3 = wasm_i32x4_shuffle(row2, row3, 2, 3, 2, 3);

    v128_t col0 = wasm_i32x4_shuffle(tmp0, tmp1, 0, 2, 0, 2);
    v128_t col1 = wasm_i32x4_shuffle(tmp0, tmp1, 1, 3, 1, 3);
    v128_t col2 = wasm_i32x4_shuffle(tmp2, tmp3, 0, 2, 0, 2);
    v128_t col3 = wasm_i32x4_shuffle(tmp2, tmp3, 1, 3, 1, 3);

    wasm_v128_store(&dst->m[0], col0);
    wasm_v128_store(&dst->m[4], col1);
    wasm_v128_store(&dst->m[8], col2);
    wasm_v128_store(&dst->m[12], col3);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    /* Pure lane permutation, so no arithmetic and therefore bit-identical to the
     * scalar path. Two gather stages: pair rows, then exchange quarters -- the
     * same structure as the SSE unpack/movelh sequence. */
    /* Transposing four 4-element rows with vrgather needs the source in a single
     * vector, which RVV expresses as a 4*vl m4 register group. Load the matrix as
     * one group and gather with per-lane indices -- that is the idiomatic form. */
    /* Each output row is a strided read of the source: elements k, k+4, k+8, k+12. */
    const size_t vl1 = __riscv_vsetvl_e32m1(4);
    const ptrdiff_t stride = 4 * (ptrdiff_t)sizeof(float);
    __riscv_vse32_v_f32m1(&dst->m[0],  __riscv_vlse32_v_f32m1(&a.m[0], stride, vl1), vl1);
    __riscv_vse32_v_f32m1(&dst->m[4],  __riscv_vlse32_v_f32m1(&a.m[1], stride, vl1), vl1);
    __riscv_vse32_v_f32m1(&dst->m[8],  __riscv_vlse32_v_f32m1(&a.m[2], stride, vl1), vl1);
    __riscv_vse32_v_f32m1(&dst->m[12], __riscv_vlse32_v_f32m1(&a.m[3], stride, vl1), vl1);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    /* Pure lane permutation. vshuf4i_w reorders within a register; vshuf_b merges
     * two registers by byte selector, which covers the cross-row combine that the
     * SSE path does with movelh/movehl. */
    const __m128 row0 = (__m128)__lsx_vld(&a.m[0], 0);
    const __m128 row1 = (__m128)__lsx_vld(&a.m[4], 0);
    const __m128 row2 = (__m128)__lsx_vld(&a.m[8], 0);
    const __m128 row3 = (__m128)__lsx_vld(&a.m[12], 0);

    /* vshuf_b is not exposed by every LSX header version; the equivalent byte
     * blend is vbitsel with a constant mask, or two vshuf4i_w plus vbitsel.
     * Use the portable spellings above; these two instructions suffice for a
     * 4x4: ilvlo/ilvhi (vpackev/vpackod style) then the quarter exchange. */
    const __m128 t0 = (__m128)__lsx_vilvl_w((__m128i)row1, (__m128i)row0);
    const __m128 t1 = (__m128)__lsx_vilvh_w((__m128i)row1, (__m128i)row0);
    const __m128 t2 = (__m128)__lsx_vilvl_w((__m128i)row3, (__m128i)row2);
    const __m128 t3 = (__m128)__lsx_vilvh_w((__m128i)row3, (__m128i)row2);

    __lsx_vst((__m128i)__lsx_vshuf4i_d((__m128i)t0, (__m128i)t2, 0x88), &dst->m[0], 0);
    __lsx_vst((__m128i)__lsx_vshuf4i_d((__m128i)t0, (__m128i)t2, 0xDD), &dst->m[4], 0);
    __lsx_vst((__m128i)__lsx_vshuf4i_d((__m128i)t1, (__m128i)t3, 0x88), &dst->m[8], 0);
    __lsx_vst((__m128i)__lsx_vshuf4i_d((__m128i)t1, (__m128i)t3, 0xDD), &dst->m[12], 0);

#else
    dst->m[0] = a.m[0];
    dst->m[1] = a.m[4];
    dst->m[2] = a.m[8];
    dst->m[3] = a.m[12];
    dst->m[4] = a.m[1];
    dst->m[5] = a.m[5];
    dst->m[6] = a.m[9];
    dst->m[7] = a.m[13];
    dst->m[8] = a.m[2];
    dst->m[9] = a.m[6];
    dst->m[10] = a.m[10];
    dst->m[11] = a.m[14];
    dst->m[12] = a.m[3];
    dst->m[13] = a.m[7];
    dst->m[14] = a.m[11];
    dst->m[15] = a.m[15];
#endif
}

float WMATH_DETERMINANT(Mat4)(const WMATH_TYPE(Mat4) m) {
    /* Scalar on every backend, deliberately.
     *
     * Everything here ends in one float, so the SIMD branch could only broadcast
     * scalars into all four lanes, multiply, and pull lane 0 back out: 12
     * _mm_set1_ps + 12 _mm_mul_ps + 12 _mm_cvtss_f32 for the minors, then 8 more
     * broadcasts, 4 multiplies and 3 adds for the cofactor sum. Strictly more
     * uops than the scalar products it computes; measured 9.7 ns vs 7.8 ns. It
     * also opened with four dead _mm_loadu_ps whose results were never used. */
    float m00 = m.m[0], m01 = m.m[1], m02 = m.m[2], m03 = m.m[3];
    float m10 = m.m[4], m11 = m.m[5], m12 = m.m[6], m13 = m.m[7];
    float m20 = m.m[8], m21 = m.m[9], m22 = m.m[10], m23 = m.m[11];
    float m30 = m.m[12], m31 = m.m[13], m32 = m.m[14], m33 = m.m[15];

    float tmp0 = m22 * m33;
    float tmp1 = m32 * m23;
    float tmp2 = m12 * m33;
    float tmp3 = m32 * m13;
    float tmp4 = m12 * m23;
    float tmp5 = m22 * m13;
    float tmp6 = m02 * m33;
    float tmp7 = m32 * m03;
    float tmp8 = m02 * m23;
    float tmp9 = m22 * m03;
    float tmp10 = m02 * m13;
    float tmp11 = m12 * m03;

    float t0 = (tmp0 * m11 + tmp3 * m21 + tmp4 * m31) - (tmp1 * m11 + tmp2 * m21 + tmp5 * m31);
    float t1 = (tmp1 * m01 + tmp6 * m21 + tmp9 * m31) - (tmp0 * m01 + tmp7 * m21 + tmp8 * m31);
    float t2 = (tmp2 * m01 + tmp7 * m11 + tmp10 * m31) - (tmp3 * m01 + tmp6 * m11 + tmp11 * m31);
    float t3 = (tmp5 * m01 + tmp8 * m11 + tmp11 * m21) - (tmp4 * m01 + tmp9 * m11 + tmp10 * m21);

    return m00 * t0 + m10 * t1 + m20 * t2 + m30 * t3;
}

void WMATH_CALL(Mat4, aim)(DST_MAT4, const WMATH_TYPE(Vec3) position, const WMATH_TYPE(Vec3) target,
                           const WMATH_TYPE(Vec3) up) {
    /* Basis built from scalars in this frame.
     *
     * The SUB/NORMALIZE/cross/NORMALIZE/cross chain passed a 12-byte Vec3 by
     * value at every step -- six struct round trips -- which made this the
     * slowest kernel in the suite. Holding the axis components in scalars
     * removes the copies, folds the two normalisations into an explicit
     * 1/sqrt, and turns the dot products into plain sums. The basis maths is
     * unchanged. */

    float zx = target.v[0] - position.v[0], zy = target.v[1] - position.v[1], zz = target.v[2] - position.v[2];
    const float z_len_sq = zx * zx + zy * zy + zz * zz;
    const float eps_sq = wcn_math_get_epsilon() * wcn_math_get_epsilon();
    if (z_len_sq > eps_sq) {
        const float inv = 1.0f / sqrtf(z_len_sq);
        zx *= inv;
        zy *= inv;
        zz *= inv;
    } else {
        zx = zy = zz = 0.0f;
    }

    /* x = up x z */
    float xx = up.v[1] * zz - up.v[2] * zy;
    float xy = up.v[2] * zx - up.v[0] * zz;
    float xz = up.v[0] * zy - up.v[1] * zx;
    const float x_len_sq = xx * xx + xy * xy + xz * xz;
    if (x_len_sq > eps_sq) {
        const float inv = 1.0f / sqrtf(x_len_sq);
        xx *= inv;
        xy *= inv;
        xz *= inv;
    } else {
        xx = xy = xz = 0.0f;
    }

    /* y = z x x -- perpendicular unit vectors, so already unit length. */
    const float yx = zy * xz - zz * xy;
    const float yy = zz * xx - zx * xz;
    const float yz = zx * xy - zy * xx;

    /* aim() is a model matrix: axes ROW-wise, translation in m[12..14]. */
    dst->m[0] = xx;  dst->m[1] = xy;  dst->m[2] = xz;   dst->m[3] = 0.0f;
    dst->m[4] = yx;  dst->m[5] = yy;  dst->m[6] = yz;   dst->m[7] = 0.0f;
    dst->m[8] = zx;  dst->m[9] = zy;  dst->m[10] = zz;  dst->m[11] = 0.0f;
    dst->m[12] = position.v[0];
    dst->m[13] = position.v[1];
    dst->m[14] = position.v[2];
    dst->m[15] = 1.0f;
}


void WMATH_CALL(Mat4, look_at)(DST_MAT4, const WMATH_TYPE(Vec3) eye, const WMATH_TYPE(Vec3) target,
                               const WMATH_TYPE(Vec3) up) {
    /* Basis built from scalars in this frame.
     *
     * The SUB/NORMALIZE/cross/NORMALIZE/cross chain passed a 12-byte Vec3 by
     * value at every step -- six struct round trips -- which made this the
     * slowest kernel in the suite. Holding the axis components in scalars
     * removes the copies, folds the two normalisations into an explicit
     * 1/sqrt, and turns the dot products into plain sums. The basis maths is
     * unchanged. */

    float zx = eye.v[0] - target.v[0], zy = eye.v[1] - target.v[1], zz = eye.v[2] - target.v[2];
    const float z_len_sq = zx * zx + zy * zy + zz * zz;
    const float eps_sq = wcn_math_get_epsilon() * wcn_math_get_epsilon();
    if (z_len_sq > eps_sq) {
        const float inv = 1.0f / sqrtf(z_len_sq);
        zx *= inv;
        zy *= inv;
        zz *= inv;
    } else {
        zx = zy = zz = 0.0f;
    }

    /* x = up x z */
    float xx = up.v[1] * zz - up.v[2] * zy;
    float xy = up.v[2] * zx - up.v[0] * zz;
    float xz = up.v[0] * zy - up.v[1] * zx;
    const float x_len_sq = xx * xx + xy * xy + xz * xz;
    if (x_len_sq > eps_sq) {
        const float inv = 1.0f / sqrtf(x_len_sq);
        xx *= inv;
        xy *= inv;
        xz *= inv;
    } else {
        xx = xy = xz = 0.0f;
    }

    /* y = z x x -- perpendicular unit vectors, so already unit length. */
    const float yx = zy * xz - zz * xy;
    const float yy = zz * xx - zx * xz;
    const float yz = zx * xy - zy * xx;

    // The original (mat4-impl.ts:1405) stores the axes COLUMN-wise and puts
    // -dot(axis,eye) at m[12..14]. That differs from aim(), whose axes are
    // row-wise -- the asymmetry is in the original itself (aim builds a model
    // matrix, look_at a view matrix). The port had the transpose of look_at
    // (axes contiguous, translation at m[3]/m[7]/m[11]), so transforming a
    // point through it gave wrong results (examples/R.c: target mapped to
    // z=0 instead of z=-5).
    const float ex = eye.v[0], ey = eye.v[1], ez = eye.v[2];
    dst->m[0] = xx;   dst->m[1] = yx;   dst->m[2] = zx;   dst->m[3] = 0.0f;
    dst->m[4] = xy;   dst->m[5] = yy;   dst->m[6] = zy;   dst->m[7] = 0.0f;
    dst->m[8] = xz;   dst->m[9] = yz;   dst->m[10] = zz;  dst->m[11] = 0.0f;
    dst->m[12] = -(xx * ex + xy * ey + xz * ez);
    dst->m[13] = -(yx * ex + yy * ey + yz * ez);
    dst->m[14] = -(zx * ex + zy * ey + zz * ez);
    dst->m[15] = 1.0f;
}


// END Mat4
