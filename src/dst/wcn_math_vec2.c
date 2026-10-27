/*
 * ISA width policy for this file (see SIMD_PERF_BUGS.md, "ISA width measurement")
 * ---------------------------------------------------------------------------
 * Vec2 kernels: two floats of live data, below the SSE lane count.
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

#include "../common/wcn_math_internal.h"
#include "WCN/WCN_MATH_DST.h"
// BEGIN Vec2

// create
void
WMATH_CREATE(Vec2)(DST_VEC2, const WMATH_CREATE_TYPE(Vec2) vec2_c) {
  dst->v[0] = WMATH_OR_ELSE_ZERO(vec2_c.v_x);
  dst->v[1] = WMATH_OR_ELSE_ZERO(vec2_c.v_y);
}

// set
void
WMATH_SET(Vec2)(DST_VEC2, const float x, const float y) {
  dst->v[0] = x;
  dst->v[1] = y;
}

// copy
void
WMATH_COPY(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) vec2) {
  dst->v[0] = vec2.v[0];
  dst->v[1] = vec2.v[1];
}

// 0
// Field writes, not `*dst = (T){...}`: the compound literal makes MSVC build
// the value in a stack temporary and copy it (PS-01 in SIMD_PERF_BUGS.md).
void
WMATH_ZERO(Vec2)(DST_VEC2) {
  dst->v[0] = 0.0f;
  dst->v[1] = 0.0f;
}

// 1
void
WMATH_IDENTITY(Vec2)(DST_VEC2) {
  dst->v[0] = 1.0f;
  dst->v[1] = 1.0f;
}

// ceil
void
WMATH_CEIL(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a) {
  // Scalar on every backend - see the note on Vec2 length: per-component rounding
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = ceilf(a.v[0]);
    dst->v[1] = ceilf(a.v[1]);
}

// floor
void
WMATH_FLOOR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a) {
  // Scalar on every backend - see the note on Vec2 length: per-component rounding
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
   dst->v[0] = floorf(a.v[0]);
   dst->v[1] = floorf(a.v[1]);
}

// round
void
WMATH_ROUND(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a) {
  // Scalar on every backend - see the note on Vec2 length: per-component rounding
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = floorf(a.v[0] + 0.5f);
    dst->v[1] = floorf(a.v[1] + 0.5f);
}

// clamp
void
WMATH_CLAMP(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float min_val, const float max_val) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation with branchless operations
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_min = _mm_set1_ps(min_val);
  __m128 vec_max = _mm_set1_ps(max_val);

  // Branchless clamp using min/max
  __m128 vec_clamped = _mm_min_ps(_mm_max_ps(vec_a, vec_min), vec_max);
  wcn_store_vec2_partial(dst->v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // ARM NEON implementation
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_min = vdupq_n_f32(min_val);
  float32x4_t vec_max = vdupq_n_f32(max_val);

  // Branchless clamp: max(min(a, max_val), min_val)
  float32x4_t vec_clamped = vmaxq_f32(vminq_f32(vec_a, vec_max), vec_min);
  wcn_store_vec2_partial(dst->v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  size_t vl = __riscv_vsetvl_e32m1(2);
  vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a.v, vl);
  vfloat32m1_t vec_min = __riscv_vfmv_v_f_f32m1(min_val, vl);
  vfloat32m1_t vec_max = __riscv_vfmv_v_f_f32m1(max_val, vl);

  // Clamp: min(max(a, min_val), max_val)
  vfloat32m1_t vec_temp = __riscv_vfmax_vv_f32m1(vec_a, vec_min, vl);
  vfloat32m1_t vec_clamped = __riscv_vfmin_vv_f32m1(vec_temp, vec_max, vl);
  __riscv_vse32_v_f32m1(dst->v, vec_clamped, vl);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WASM SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_min = wasm_f32x4_splat(min_val);
  v128_t vec_max = wasm_f32x4_splat(max_val);

  // Branchless clamp
  v128_t vec_temp = wasm_f32x4_max(vec_a, vec_min);
  v128_t vec_clamped = wasm_f32x4_min(vec_temp, vec_max);
  wcn_store_vec2_partial(dst->v, vec_clamped);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_min = wcn_lsx_splat_f(min_val);
  __m128 vec_max = wcn_lsx_splat_f(max_val);

  // Clamp using min/max operations
  __m128 vec_temp = __lsx_vfmax_s(vec_a, vec_min);
  __m128 vec_clamped = __lsx_vfmin_s(vec_temp, vec_max);
  wcn_store_vec2_partial(dst->v, vec_clamped);
#else
  // Scalar fallback
  dst->v[0] = WMATH_CLAMP(float)(a.v[0], min_val, max_val);
  dst->v[1] = WMATH_CLAMP(float)(a.v[1], min_val, max_val);
#endif

}

// dot
float WMATH_DOT(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length. Two multiplies and
  // one add beat the build-a-4-lane-vector-then-reduce sequence, which measured
  // 2.0x the cost here (SIMD_PERF_BUGS.md PS-07).
  return a.v[0] * b.v[0] + a.v[1] * b.v[1];
}

// add
void
WMATH_ADD(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length: per-component add
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = a.v[0] + b.v[0];
    dst->v[1] = a.v[1] + b.v[1];
}

// addScaled
void
WMATH_ADD_SCALED(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b, const float scale) {
  // Scalar on every backend - see the note on Vec2 length: per-component multiply-add
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = a.v[0] + b.v[0] * scale;
    dst->v[1] = a.v[1] + b.v[1] * scale;
}

// sub
void
WMATH_SUB(Vec2)(DST_VEC2, WMATH_TYPE(Vec2) a, WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length: per-component subtract
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = a.v[0] - b.v[0];
    dst->v[1] = a.v[1] - b.v[1];
}

// angle
float WMATH_ANGLE(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE optimized implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  __m128 va_sq = _mm_mul_ps(va, va);
  __m128 vb_sq = _mm_mul_ps(vb, vb);
  float mag_1_sq = _mm_cvtss_f32(wcn_hadd_ps(va_sq));
  float mag_2_sq = _mm_cvtss_f32(wcn_hadd_ps(vb_sq));

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON optimized implementation
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);

  float32x4_t va_sq = vmulq_f32(va, va);
  float32x4_t vb_sq = vmulq_f32(vb, vb);
  float mag_1_sq = wcn_hadd_f32(va_sq);
  float mag_2_sq = wcn_hadd_f32(vb_sq);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  vfloat32m1_t va_sq = __riscv_vfmul_vv_f32m1(va, va, 4);
  vfloat32m1_t vb_sq = __riscv_vfmul_vv_f32m1(vb, vb, 4);
  float mag_1_sq = wcn_hadd_f32(va_sq);
  float mag_2_sq = wcn_hadd_f32(vb_sq);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  v128_t va_sq = wasm_f32x4_mul(va, va);
  v128_t vb_sq = wasm_f32x4_mul(vb, vb);
  // Extract and add each component to get sum of squares
  v128_t v = wasm_i32x4_shuffle(va_sq, va_sq, 0, 1, 0, 1);
  v128_t v2 = wasm_i32x4_shuffle(va_sq, va_sq, 2, 3, 2, 3);
  v128_t sum = wasm_f32x4_add(v, v2);
  float mag_1_sq = wasm_f32x4_extract_lane(sum, 0);

  v = wasm_i32x4_shuffle(vb_sq, vb_sq, 0, 1, 0, 1);
  v2 = wasm_i32x4_shuffle(vb_sq, vb_sq, 2, 3, 2, 3);
  sum = wasm_f32x4_add(v, v2);
  float mag_2_sq = wasm_f32x4_extract_lane(sum, 0);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);

  // Calculate magnitudes using SIMD
  __m128 va_sq = __lsx_vfmul_s(va, va);
  __m128 vb_sq = __lsx_vfmul_s(vb, vb);
  // Horizontally add to get sum of squares
  __m128 high = __lsx_vshuf4i_w(va_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(va_sq, high);
  float mag_1_sq = wcn_lsx_extract_f(sum, 0);

  high = __lsx_vshuf4i_w(vb_sq, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  sum = __lsx_vfadd_s(vb_sq, high);
  float mag_2_sq = wcn_lsx_extract_f(sum, 0);

  float mag_1 = sqrtf(mag_1_sq);
  float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  if (mag < wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);

#else
  // Scalar fallback with safety checks
  const float mag_1_sq = a.v[0] * a.v[0] + a.v[1] * a.v[1];
  const float mag_2_sq = b.v[0] * b.v[0] + b.v[1] * b.v[1];

  if (mag_1_sq < wcn_math_get_epsilon() * wcn_math_get_epsilon() ||
      mag_2_sq < wcn_math_get_epsilon() * wcn_math_get_epsilon()) {
    return 0.0f; // Prevent division by zero
  }

  const float mag_1 = sqrtf(mag_1_sq);
  const float mag_2 = sqrtf(mag_2_sq);
  const float mag = mag_1 * mag_2;

  float dot_product = WMATH_DOT(Vec2)(a, b);
  float cosine = dot_product / mag;

  // Clamp cosine to [-1, 1] to prevent domain errors in acosf
  cosine = fmaxf(-1.0f, fminf(1.0f, cosine));
  return acosf(cosine);
#endif
}

// equalsApproximately
bool WMATH_EQUALS_APPROXIMATELY(Vec2)(const WMATH_TYPE(Vec2) a,
                                      const WMATH_TYPE(Vec2) b) {
  float ep = wcn_math_get_epsilon();
  return fabsf(a.v[0] - b.v[0]) < ep && fabsf(a.v[1] - b.v[1]) < ep;
}

// equals
bool WMATH_EQUALS(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  return (a.v[0] == b.v[0] && a.v[1] == b.v[1]);
}

// Linear Interpolation
void
WMATH_LERP(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                 const float t) {
  // Scalar on every backend - see the note on Vec2 length: per-component lerp
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    dst->v[0] = a.v[0] + (b.v[0] - a.v[0]) * t;
    dst->v[1] = a.v[1] + (b.v[1] - a.v[1]) * t;
}

// Linear Interpolation V
void
WMATH_LERP_V(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b,
                   const WMATH_TYPE(Vec2) t) {
  // Scalar on every backend - see the note on Vec2 length: per-component lerp
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    dst->v[0] = a.v[0] + (b.v[0] - a.v[0]) * t.v[0];
    dst->v[1] = a.v[1] + (b.v[1] - a.v[1]) * t.v[1];
}

// f max
void
WMATH_FMAX(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_max_ps(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vmaxq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmax_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_max(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfmax_s(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#else
  // Scalar fallback
  dst->v[0] = fmaxf(a.v[0], b.v[0]);
  dst->v[1] = fmaxf(a.v[1], b.v[1]);
#endif

}

// f min
void
WMATH_FMIN(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation - using helper functions
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = _mm_min_ps(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation - using helper functions
  float32x4_t vec_a = wcn_load_vec2_partial(a.v);
  float32x4_t vec_b = wcn_load_vec2_partial(b.v);
  float32x4_t vec_res = vminq_f32(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation - using helper functions
  vfloat32m1_t vec_a = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vec_b = wcn_load_vec2_partial(b.v);
  vfloat32m1_t vec_res = __riscv_vfmin_vv_f32m1(vec_a, vec_b, 4); // Use full vector length for helper
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_a = wcn_load_vec2_partial(a.v);
  v128_t vec_b = wcn_load_vec2_partial(b.v);
  v128_t vec_res = wasm_f32x4_min(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_a = wcn_load_vec2_partial(a.v);
  __m128 vec_b = wcn_load_vec2_partial(b.v);
  __m128 vec_res = __lsx_vfmin_s(vec_a, vec_b);
  wcn_store_vec2_partial(dst->v, vec_res);

#else
  // Scalar fallback
  dst->v[0] = fminf(a.v[0], b.v[0]);
  dst->v[1] = fminf(a.v[1], b.v[1]);
#endif

}

// multiplyScalar
void
WMATH_MULTIPLY_SCALAR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float scalar) {
  // Scalar on every backend - see the note on Vec2 length: per-component scalar multiply
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = a.v[0] * scalar;
    dst->v[1] = a.v[1] * scalar;
}

// multiply
void
WMATH_MULTIPLY(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length: per-component multiply
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = a.v[0] * b.v[0];
    dst->v[1] = a.v[1] * b.v[1];
}

// divScalar
/**
 * (divScalar) if scalar is 0, returns a zero vector
 */
void
WMATH_DIV_SCALAR(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float scalar) {
  // Scalar on every backend - see the note on Vec2 length: per-component scalar divide
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar implementation with branchless operations
    dst->v[0] = wcn_safe_div_float(a.v[0], scalar);
    dst->v[1] = wcn_safe_div_float(a.v[1], scalar);
}

// div
void
WMATH_DIV(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length: per-component divide
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    dst->v[0] = a.v[0] / b.v[0];
    dst->v[1] = a.v[1] / b.v[1];
}

// inverse
void
WMATH_INVERSE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a) {
  // Scalar on every backend - see the note on Vec2 length: per-component multiply by -1
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback with zero division check
    dst->v[0] = (a.v[0] != 0.0f) ? 1.0f / a.v[0] : 0.0f;
    dst->v[1] = (a.v[1] != 0.0f) ? 1.0f / a.v[1] : 0.0f;
}

// cross
void
WMATH_CROSS(Vec2)(DST_VEC3, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // WMATH_TYPE(Vec3) vec3;
  dst->v[0] = 0;
  dst->v[1] = 0;
  // z
  dst->v[2] = a.v[0] * b.v[1] - a.v[1] * b.v[0];
  // return vec3;
}

// length
float WMATH_LENGTH(Vec2)(const WMATH_TYPE(Vec2) v) {
  // Scalar on every backend, deliberately.
  //
  // This is a 2-element horizontal reduction: the whole computation is two
  // multiplies and one add, which the compiler already contracts to a pair of
  // FMA-able instructions. The SIMD versions had to build a 4-lane vector from
  // two floats and then reduce it across lanes (movlps + mul + hadd + hadd +
  // vmovss on SSE, vpaddq twice on NEON), which measured 2.2x SLOWER than this
  // line (SIMD_PERF_BUGS.md PS-07). Same reasoning applies to length_squared,
  // dot and distance below.
  return sqrtf(v.v[0] * v.v[0] + v.v[1] * v.v[1]);
}

// lengthSquared
float WMATH_LENGTH_SQ(Vec2)(const WMATH_TYPE(Vec2) v) {
  /* Scalar on every backend; same reasoning as Vec2_length above. Two
   multiplies and one add, measured 1.6x faster than the SSE form that had to
   build a 4-lane vector and reduce across lanes (SIMD_PERF_BUGS.md PS-07). */
  return v.v[0] * v.v[0] + v.v[1] * v.v[1];
}

// distance
float WMATH_DISTANCE(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  // Scalar on every backend - see the note on Vec2 length (SIMD_PERF_BUGS.md PS-07).
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return sqrtf(dx * dx + dy * dy);
}

// distance_squared
float WMATH_DISTANCE_SQ(Vec2)(const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  float dx = a.v[0] - b.v[0];
  float dy = a.v[1] - b.v[1];
  return dx * dx + dy * dy;
}

// negate
void
WMATH_NEGATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a) {
  // Scalar on every backend - see the note on Vec2 length: per-component sign flip
  // on two lanes costs more in intrinsics than it saves (SIMD_PERF_BUGS.md PS-07).
    // Scalar fallback
    dst->v[0] = -a.v[0];
    dst->v[1] = -a.v[1];
}

// random
void
WMATH_RANDOM(Vec2)(DST_VEC2, float scale) {
  float angle = WMATH_RANDOM(float)() * WMATH_2PI;
  dst->v[0] = cosf(angle) * scale;
  dst->v[1] = sinf(angle) * scale;
}

// normalize
void
WMATH_NORMALIZE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) v) {
  // WMATH_TYPE(Vec2) vec2;
  const float epsilon = wcn_math_get_epsilon();

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // Plain scalar on x86 deliberately (SIMD_PERF_BUGS.md PS-07): for a 2-wide
  // vector the packed path (movsd load, mulps, hadd, broadcast, movlps store)
  // measured ~1.7x slower than the six SSE2 scalar ops MSVC emits for this
  // form anyway. Exact sqrt, matching the original 1/Math.sqrt.
  float len_sq = v.v[0] * v.v[0] + v.v[1] * v.v[1];
  if (len_sq > epsilon * epsilon) {
    const float inv_len = 1.0f / sqrtf(len_sq);
    dst->v[0] = v.v[0] * inv_len;
    dst->v[1] = v.v[1] * inv_len;
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // Optimized NEON implementation
  float32x4_t vec_v = wcn_load_vec2_partial(v.v);
  float32x4_t vec_squared = vmulq_f32(vec_v, vec_v);
  float len_sq = wcn_hadd_f32(vec_squared);

  if (len_sq > epsilon * epsilon) {
    // Exact reciprocal square root (matches the original's 1/Math.sqrt);
    // the previous vrsqrte estimate + 1 refinement kept ~6e-7 relative error.
    float32x2_t len_sq_vec = vdup_n_f32(len_sq);
    float32x2_t inv_len = vdiv_f32(vdup_n_f32(1.0f), vsqrt_f32(len_sq_vec));
    float32x4_t inv_len_broadcast = vcombine_f32(inv_len, inv_len);
    float32x4_t vec_res = vmulq_f32(vec_v, inv_len_broadcast);
    wcn_store_vec2_partial(dst->v, vec_res);
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  vfloat32m1_t vec_v = __riscv_vle32_v_f32m1(v.v, 2);
  vfloat32m1_t vec_squared = __riscv_vfmul_vv_f32m1(vec_v, vec_v, 2);
  float len_sq = wcn_hadd_f32(vec_squared);

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    vfloat32m1_t inv_len_vec = __riscv_vfmv_v_f_f32m1(inv_len, 2);
    vfloat32m1_t vec_res = __riscv_vfmul_vv_f32m1(vec_v, inv_len_vec, 2);
    __riscv_vse32_v_f32m1(dst->v, vec_res, 2);
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  v128_t vec_v = wcn_load_vec2_partial(v.v);
  v128_t vec_squared = wasm_f32x4_mul(vec_v, vec_v);
  float len_sq = wasm_f32x4_extract_lane(vec_squared, 0) + wasm_f32x4_extract_lane(vec_squared, 1);

  if (len_sq > epsilon * epsilon) {
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    v128_t inv_len_vec = wasm_f32x4_splat(inv_len);
    v128_t vec_res = wasm_f32x4_mul(vec_v, inv_len_vec);
    wcn_store_vec2_partial(dst->v, vec_res);
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  __m128 vec_v = wcn_load_vec2_partial(v.v);
  __m128 vec_squared = __lsx_vfmul_s(vec_v, vec_v);

  // Horizontally add to get length squared
  __m128 high = __lsx_vshuf4i_w(vec_squared, 0x1B);  // 0x1B = 0b00011011 = (2, 3, 0, 1)
  __m128 sum = __lsx_vfadd_s(vec_squared, high);
  float len_sq = wcn_lsx_extract_f(sum, 0);

  if (len_sq > epsilon * epsilon) {
    // Calculate inverse square root and multiply
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    __m128 inv_len_vec = __lsx_vldrepl_w(&inv_len, 0);
    __m128 vec_res = __lsx_vfmul_s(vec_v, inv_len_vec);
    wcn_store_vec2_partial(dst->v, vec_res);
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }

#else
  // Optimized scalar fallback using fast inverse square root
  float len_sq = v.v[0] * v.v[0] + v.v[1] * v.v[1];
  if (len_sq > epsilon * epsilon) {
    float inv_len = wcn_fast_inv_sqrt(len_sq);
    dst->v[0] = v.v[0] * inv_len;
    dst->v[1] = v.v[1] * inv_len;
  } else {
    dst->v[0] = 0.0f;
    dst->v[1] = 0.0f;
  }
#endif
}

// rotate
void
WMATH_ROTATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b, const float rad) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
  // SSE implementation
  float s = sinf(rad);
  float c = cosf(rad);
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 v = _mm_sub_ps(va, vb); // [p0, p1, 0, 0]

  __m128 px = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 py = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 sc = _mm_set1_ps(s);
  __m128 cc = _mm_set1_ps(c);

  __m128 rx = wcn_fma_mul_sub_ps(px, cc, _mm_mul_ps(py, sc));
  __m128 ry = wcn_fma_mul_add_ps(px, sc, _mm_mul_ps(py, cc));

  float rx_s = _mm_cvtss_f32(rx);
  float ry_s = _mm_cvtss_f32(ry);
  __m128 res = _mm_set_ps(0.0f, 0.0f, ry_s, rx_s);
  // add back center
  res = _mm_add_ps(res, vb);
  wcn_store_vec2_partial(dst->v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
  // NEON implementation
  float s = sinf(rad);
  float c = cosf(rad);
  float32x4_t va = wcn_load_vec2_partial(a.v);
  float32x4_t vb = wcn_load_vec2_partial(b.v);
  float32x4_t v = vsubq_f32(va, vb);
  float32x4_t px = vdupq_n_f32(vgetq_lane_f32(v, 0));
  float32x4_t py = vdupq_n_f32(vgetq_lane_f32(v, 1));
  float32x4_t sc = vdupq_n_f32(s);
  float32x4_t cc = vdupq_n_f32(c);
  float32x4_t rx = vfmsq_f32(px, cc, vmulq_f32(py, sc));
  float32x4_t ry = vfmaq_f32(vmulq_f32(py, cc), px, sc);
  float rx_s = vgetq_lane_f32(rx, 0);
  float ry_s = vgetq_lane_f32(ry, 0);
  float32x4_t res = {rx_s + vgetq_lane_f32(vb, 0), ry_s + vgetq_lane_f32(vb, 1),
                     0.0f, 0.0f};
  wcn_store_vec2_partial(dst->v, res);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
  // RISC-V Vector Extension implementation
  float s = sinf(rad);
  float c = cosf(rad);
  vfloat32m1_t va = wcn_load_vec2_partial(a.v);
  vfloat32m1_t vb = wcn_load_vec2_partial(b.v);
  vfloat32m1_t v = __riscv_vfsub_vv_f32m1(va, vb, 4); // [p0, p1, 0, 0] (using helper function)

  // Extract the x and y components - use first element and second element
  float p0 = __riscv_vfmv_f_s_f32m1_f32(v);  // Extract x component (first lane)
  float temp[4];
  __riscv_vse32_v_f32m1(temp, v, 4);  // Store vector to temp array
  float p1 = temp[1];  // Extract y component (second lane)

  // Create vectors with the components broadcasted
  vfloat32m1_t px = __riscv_vfmv_v_f_f32m1(p0, 4);
  vfloat32m1_t py = __riscv_vfmv_v_f_f32m1(p1, 4);
  vfloat32m1_t sc = __riscv_vfmv_v_f_f32m1(s, 4);
  vfloat32m1_t cc = __riscv_vfmv_v_f_f32m1(c, 4);
  vfloat32m1_t rx = __riscv_vfsub_vv_f32m1(__riscv_vfmul_vv_f32m1(px, cc, 4), __riscv_vfmul_vv_f32m1(py, sc, 4), 4);
  vfloat32m1_t ry = __riscv_vfadd_vv_f32m1(__riscv_vfmul_vv_f32m1(px, sc, 4), __riscv_vfmul_vv_f32m1(py, cc, 4), 4);

  // Extract the results
  float rx_s = __riscv_vfmv_f_s_f32m1_f32(rx);
  float temp2[4];
  __riscv_vse32_v_f32m1(temp2, ry, 4);  // Store ry vector to temp array
  float ry_s = temp2[1];  // Extract result from first lane

  // Add back center point
  dst->v[0] = rx_s + b.v[0];
  dst->v[1] = ry_s + b.v[1];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
  // WebAssembly SIMD implementation
  float s = sinf(rad);
  float c = cosf(rad);
  v128_t va = wcn_load_vec2_partial(a.v);
  v128_t vb = wcn_load_vec2_partial(b.v);
  v128_t v = wasm_f32x4_sub(va, vb);

  // Extract the x and y components
  float p0 = wasm_f32x4_extract_lane(v, 0);
  float p1 = wasm_f32x4_extract_lane(v, 1);

  // Perform rotation: rx = p0*c - p1*s, ry = p0*s + p1*c
  float rx_s = p0 * c - p1 * s;
  float ry_s = p0 * s + p1 * c;

  // Add back center point
  dst->v[0] = rx_s + b.v[0];
  dst->v[1] = ry_s + b.v[1];

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
  // LoongArch LSX implementation
  float s = sinf(rad);
  float c = cosf(rad);
  __m128 va = wcn_load_vec2_partial(a.v);
  __m128 vb = wcn_load_vec2_partial(b.v);
  __m128 v = __lsx_vfsub_s(va, vb); // [p0, p1, 0, 0]

  // Extract the x and y components using shuffle
  __m128 px = __lsx_vshuf4i_w(v, 0x00); // duplicate x component (lane 0)
  __m128 py = __lsx_vshuf4i_w(v, 0x55); // duplicate y component (lane 1 with pattern 0x55=0b01010101)

  __m128 sc = __lsx_vldrepl_w(&s, 0);
  __m128 cc = __lsx_vldrepl_w(&c, 0);
  __m128 rx = __lsx_vfsub_s(__lsx_vfmul_s(px, cc), __lsx_vfmul_s(py, sc));
  __m128 ry = __lsx_vfadd_s(__lsx_vfmul_s(px, sc), __lsx_vfmul_s(py, cc));

  float rx_s = wcn_lsx_extract_f(rx, 0);
  float ry_s = wcn_lsx_extract_f(ry, 0);  // Get from first element as both lanes are the same

  // Add back center
  dst->v[0] = rx_s + b.v[0];
  dst->v[1] = ry_s + b.v[1];

#else
  // Scalar fallback
  float p0 = a.v[0] - b.v[0];
  float p1 = a.v[1] - b.v[1];
  float s = sinf(rad);
  float c = cosf(rad);
  dst->v[0] = p0 * c - p1 * s + b.v[0];
  dst->v[1] = p0 * s + p1 * c + b.v[1];
#endif
}

// set length
void
WMATH_SET_LENGTH(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float length) {
  WMATH_TYPE(Vec2) temp;
  WMATH_NORMALIZE(Vec2)(&temp, a);
  WMATH_MULTIPLY_SCALAR(Vec2)(dst, temp, length);
}

// truncate
void
WMATH_TRUNCATE(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const float length) {
  // The original returns from the truncating branch:
  //   if (length(a) > maxLen) return setLength(a, maxLen, newDst);
  //   return copy(a, newDst);
  // Without the else the truncated result was immediately overwritten with
  // the untruncated input, so truncate() was a no-op copy for every input.
  if (WMATH_LENGTH(Vec2)(a) > length) {
    WMATH_SET_LENGTH(Vec2)(dst, a, length);
  } else {
    WMATH_COPY(Vec2)(dst, a);
  }
}

// midpoint
void
WMATH_MIDPOINT(Vec2)(DST_VEC2, const WMATH_TYPE(Vec2) a, const WMATH_TYPE(Vec2) b) {
  WMATH_LERP(Vec2)(dst, a, b, 0.5f);
}

// END Vec2
