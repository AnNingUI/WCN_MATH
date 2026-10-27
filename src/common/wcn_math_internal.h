#ifndef WCN_MATH_INTERNAL_H
#define WCN_MATH_INTERNAL_H

#ifdef __cplusplus
extern "C" {
#endif
#include "WCN/WCN_PLATFORM_MACROS.h"
#include <math.h>   // sqrtf/acosf/sinf/cosf/fabsf/floorf used in this header
#include <stdlib.h>
#include <string.h> // memcpy/memset used across all modules
// ========================================================================
#include "WCN/WCN_MATH_TYPES.h"
#include <stdbool.h>

static inline void* wcn_aligned_alloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#if defined(_WIN32)
    // Windows (MSVC)
    ptr = _aligned_malloc(size, alignment);
#else
    // Linux / macOS / POSIX
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

static inline void wcn_aligned_free(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
#include <emmintrin.h>
#include <immintrin.h> // For additional SSE/AVX intrinsics
#include <smmintrin.h> // For SSE 4.1, which has better float operations

// Check for AVX/AVX2 support at compile time
#if defined(__AVX2__)
#define WCN_HAS_AVX2 1
#elif defined(__AVX__)
#define WCN_HAS_AVX 1
#endif

// SIMD helper functions for x86
static inline __m128 wcn_load_vec2_partial(const float *v) {
  // One 8-byte packed load into lane 0-1 (movlps), no 16-byte over-read and
  // no per-element insert shuffles. The old `_mm_set_ps(0,0,v[1],v[0])` made
  // MSVC emit two scalar loads plus inserts; measured cost made the SSE
  // Vec2 kernels slower than the plain scalar ones (SIMD_PERF_BUGS.md PS-07).
  return _mm_castpd_ps(_mm_load_sd((const double*)v));
}

// Cross product helper function for x86 SSE
static inline __m128 wcn_cross_ps(__m128 a, __m128 b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y -
  // a.y*b.x)
  const __m128 a_yzx =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 2, 1)); // a.y, a.z, a.x, a.y
  const __m128 b_zxy =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 0, 2)); // b.z, b.x, b.y, b.z
  const __m128 a_zxy =
      _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 0, 2)); // a.z, a.x, a.y, a.z
  const __m128 b_yzx =
      _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 0, 2, 1)); // b.y, b.z, b.x, b.y

  const __m128 mul1 = _mm_mul_ps(a_yzx, b_zxy);
  const __m128 mul2 = _mm_mul_ps(a_zxy, b_yzx);
  return _mm_sub_ps(mul1, mul2);
}

static inline void wcn_store_vec2_partial(float *v, const __m128 vec) {
  // One packed 8-byte store of lanes 0-1 straight from the register. The old
  // pattern (16-byte store into a stack scratch, then scalar reloads) pays a
  // store-forwarding stall for every 4-byte reload (SIMD_PERF_BUGS.md PS-03),
  // and a 16-byte store would write past float v[2].
  _mm_storel_pi((__m64*)v, vec);
}

static inline __m128 wcn_load_vec3_partial(const float *v) {
  // 3 instructions, no over-read: movsd loads lanes 0-1 packed, movss loads
  // lane 2, one shuffle merges -> [v0, v1, v2, 0]. (GCC spells the merge
  // _mm_movelhps; shuffle_ps(1,0,1,0) is the same single instruction and is
  // portable to MSVC.)
  // The old _mm_setr_ps form made MSVC emit per-lane inserts (5-6 moves),
  // which is what made SSE Vec3 kernels cost more than the scalar path.
  const __m128 lo = _mm_castpd_ps(_mm_load_sd((const double*)v));
  const __m128 z = _mm_load_ss(v + 2);
  return _mm_shuffle_ps(lo, z, _MM_SHUFFLE(1, 0, 1, 0));
}

static inline void wcn_store_vec3_partial(float *v, const __m128 vec) {
  // Packed 8-byte store (lanes 0-1) + one 4-byte store of lane 2: no scratch
  // array, and never writes float v[3].
  _mm_storel_pi((__m64*)v, vec);
  _mm_store_ss(v + 2, _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(2, 2, 2, 2)));
}

static inline __m128 wcn_hadd_ps(const __m128 vec) {
  const __m128 temp = _mm_hadd_ps(vec, vec);
  return _mm_hadd_ps(temp, temp);
}

// Matrix helper functions
static inline __m128 wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, const int row) {
  return _mm_loadu_ps(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, const int row,
                                    const __m128 vec) {
  _mm_storeu_ps(&mat->m[row * 4], vec);
}

static inline __m128 wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, const int row) {
  return _mm_loadu_ps(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, const int row,
                                    const __m128 vec) {
  _mm_storeu_ps(&mat->m[row * 4], vec);
}

static inline __m128 wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, const int col) {
  return _mm_set_ps(mat->m[col + 12], mat->m[col + 8], mat->m[col + 4],
                    mat->m[col]);
}

// AVX/AVX2 helper functions
#if defined(WCN_HAS_AVX) || defined(WCN_HAS_AVX2)

#endif

// FMA helper functions
#if defined(WCN_HAS_FMA)

// FMA-optimized vector multiply-add: a * b + c
static inline __m128 wcn_fma_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fmadd_ps(a, b, c);
}

// FMA-optimized vector multiply-sub: a * b - c
static inline __m128 wcn_fma_mul_sub_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fmsub_ps(a, b, c);
}

// FMA-optimized vector negate-multiply-add: -(a * b) + c
static inline __m128 wcn_fma_neg_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_fnmadd_ps(a, b, c);
}

// AVX FMA versions
#if defined(WCN_HAS_AVX) || defined(WCN_HAS_AVX2)

// FMA-optimized AVX vector multiply-add: a * b + c
static inline __m256 wcn_avx_fma_mul_add_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}

// FMA-optimized AVX vector multiply-sub: a * b - c
static inline __m256 wcn_avx_fma_mul_sub_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fmsub_ps(a, b, c);
}

// FMA-optimized AVX vector negate-multiply-add: -(a * b) + c
static inline __m256 wcn_avx_fma_neg_mul_add_ps(const __m256 a, const __m256 b, const __m256 c) {
  return _mm256_fnmadd_ps(a, b, c);
}

#endif

#endif

// AVX2 specific helper functions
#if defined(WCN_HAS_AVX2)

#endif

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
#include <arm_neon.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// Helper: store a float32x4_t into a 4-element array and copy first 3 elements
static inline void wcn_neon_store_vec3_to_array(float32x4_t v, float out[3]) {
  float tmp[4];
  vst1q_f32(tmp, v);
  out[0] = tmp[0];
  out[1] = tmp[1];
  out[2] = tmp[2];
}
#endif

// SIMD helper functions for ARM NEON
static inline float32x4_t wcn_load_vec2_partial(const float *v) {
  return (float32x4_t){v[0], v[1], 0.0f, 0.0f};
}

// NEON cross product helper for 3D vectors
static inline float32x4_t wcn_cross_neon(float32x4_t a, float32x4_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y -
  // a.y*b.x)
  float32x4_t a_yzx = (float32x4_t){vgetq_lane_f32(a, 1), vgetq_lane_f32(a, 2),
                                    vgetq_lane_f32(a, 0), 0.0f};
  float32x4_t b_zxy = (float32x4_t){vgetq_lane_f32(b, 2), vgetq_lane_f32(b, 0),
                                    vgetq_lane_f32(b, 1), 0.0f};
  float32x4_t a_zxy = (float32x4_t){vgetq_lane_f32(a, 2), vgetq_lane_f32(a, 0),
                                    vgetq_lane_f32(a, 1), 0.0f};
  float32x4_t b_yzx = (float32x4_t){vgetq_lane_f32(b, 1), vgetq_lane_f32(b, 2),
                                    vgetq_lane_f32(b, 0), 0.0f};

  float32x4_t mul1 = vmulq_f32(a_yzx, b_zxy);
  float32x4_t mul2 = vmulq_f32(a_zxy, b_yzx);
  return vsubq_f32(mul1, mul2);
}

static inline void wcn_store_vec2_partial(float *v, float32x4_t vec) {
  v[0] = vgetq_lane_f32(vec, 0);
  v[1] = vgetq_lane_f32(vec, 1);
}

static inline float32x4_t wcn_load_vec3_partial(const float *v) {
  return (float32x4_t){v[0], v[1], v[2], 0.0f};
}

static inline void wcn_store_vec3_partial(float *v, float32x4_t vec) {
  v[0] = vgetq_lane_f32(vec, 0);
  v[1] = vgetq_lane_f32(vec, 1);
  v[2] = vgetq_lane_f32(vec, 2);
}

static inline float wcn_hadd_f32(float32x4_t vec) {
  float32x2_t low = vget_low_f32(vec);
  float32x2_t high = vget_high_f32(vec);
  float32x2_t sum = vadd_f32(low, high);
  return vget_lane_f32(vpadd_f32(sum, sum), 0);
}

// Matrix helper functions
static inline float32x4_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat,
                                           int row) {
  return vld1q_f32(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row,
                                    float32x4_t vec) {
  vst1q_f32(&mat->m[row * 4], vec);
}

static inline float32x4_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat,
                                           int row) {
  return vld1q_f32(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row,
                                    float32x4_t vec) {
  vst1q_f32(&mat->m[row * 4], vec);
}

static inline float32x4_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat,
                                           int col) {
  return (float32x4_t){mat->m[col], mat->m[col + 4], mat->m[col + 8],
                       mat->m[col + 12]};
}

#endif

// Platform-specific SIMD implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
// RISC-V Vector Extension
#include <riscv_vector.h>


/*
 * Materialise a 4-lane vector from scalar arguments.
 *
 * `__riscv_vcreate_v_*` is an internal compiler builtin and is not part of the public RVV
 * intrinsic API, so these calls did not resolve. The supported route is to store and load.
 * vl is 4 because this build pins the vector type to 128 bits (riscv_rvv_vector_bits), i.e.
 * exactly four 32-bit lanes.
 */
static inline vfloat32m1_t wcn_rvv_make4_f(float a, float b, float c, float d) {
  float tmp[4] = {a, b, c, d};
  return __riscv_vle32_v_f32m1(tmp, 4);
}

static inline vuint32m1_t wcn_rvv_make4_u(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  uint32_t tmp[4] = {a, b, c, d};
  return __riscv_vle32_v_u32m1(tmp, 4);
}

// Helper functions for RISC-V
static inline vfloat32m1_t wcn_load_vec2_partial(const float *v) {
  float temp[4] = {v[0], v[1], 0.0f, 0.0f};
  return __riscv_vle32_v_f32m1(temp, 4);
}

static inline void wcn_store_vec2_partial(float *v, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(v, vec, 2); // Only store first 2 elements
}

static inline vfloat32m1_t wcn_load_vec3_partial(const float *v) {
  float temp[4] = {v[0], v[1], v[2], 0.0f};
  return __riscv_vle32_v_f32m1(temp, 4);
}

static inline void wcn_store_vec3_partial(float *v, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(v, vec, 3); // Only store first 3 elements
}

/*
 * Lane extraction on RVV. `vfmv.f.s` only reads lane 0; to read lane N the vector is slid down
 * by N first. The previous code passed the index as a second argument to vfmv.f.s, which does
 * not take one.
 */
/*
 * Lane extraction when there is no vl in scope. This build pins the vector type to 128 bits
 * (riscv_rvv_vector_bits in the kernel header), so there are always exactly four 32-bit lanes and
 * the length is a constant. wcn_rvv_lane takes vl instead, for callers that have one.
 */
static inline float wcn_rvv_lane4(vfloat32m1_t v, const size_t i) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(v, i, 4));
}

/*
 * Insert one scalar into lane i, leaving the other lanes alone.
 *
 * RVV has no single-instruction "set lane i" for a runtime i: vfmv.v.f splats and
 * vfslide1up/down only shift by one position. Storing, poking and reloading is exact for any i
 * and needs no immediate. These call sites previously invoked vfmv.v.f with an index argument,
 * which it does not take.
 */
static inline vfloat32m1_t wcn_rvv_set_lane(vfloat32m1_t v, const float x, const size_t i,
                                            const size_t vl) {
  float tmp[4];
  __riscv_vse32_v_f32m1(tmp, v, vl);
  tmp[i] = x;
  return __riscv_vle32_v_f32m1(tmp, vl);
}

static inline float wcn_rvv_lane(vfloat32m1_t v, size_t i, size_t vl) {
  return __riscv_vfmv_f_s_f32m1_f32(__riscv_vslidedown_vx_f32m1(v, i, vl));
}

static inline float wcn_hadd_f32(vfloat32m1_t vec) {
  // Perform horizontal sum of vector elements
  vfloat32m1_t initial = __riscv_vfmv_v_f_f32m1(0.0f, 4); // broadcast 0.0f to all elements
  vfloat32m1_t sum = __riscv_vfredusum_vs_f32m1_f32m1(vec, initial, 4);

  return wcn_rvv_lane(sum, 0, 4);
}

// RISC-V cross product helper for 3D vectors

static inline vfloat32m1_t wcn_cross_rvv(vfloat32m1_t a, vfloat32m1_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  float a_vals[4] = {wcn_rvv_lane(a, 1, 4), wcn_rvv_lane(a, 2, 4),
                     wcn_rvv_lane(a, 0, 4), 0.0f};
  vfloat32m1_t a_yzx = __riscv_vle32_v_f32m1(a_vals, 4);

  float b_vals[4] = {wcn_rvv_lane(b, 2, 4), wcn_rvv_lane(b, 0, 4),
                     wcn_rvv_lane(b, 1, 4), 0.0f};
  vfloat32m1_t b_zxy = __riscv_vle32_v_f32m1(b_vals, 4);

  float a_vals2[4] = {wcn_rvv_lane(a, 2, 4), wcn_rvv_lane(a, 0, 4),
                      wcn_rvv_lane(a, 1, 4), 0.0f};
  vfloat32m1_t a_zxy = __riscv_vle32_v_f32m1(a_vals2, 4);

  float b_vals2[4] = {wcn_rvv_lane(b, 1, 4), wcn_rvv_lane(b, 2, 4),
                      wcn_rvv_lane(b, 0, 4), 0.0f};
  vfloat32m1_t b_yzx = __riscv_vle32_v_f32m1(b_vals2, 4);

  vfloat32m1_t mul1 = __riscv_vfmul_vv_f32m1(a_yzx, b_zxy, 4);
  vfloat32m1_t mul2 = __riscv_vfmul_vv_f32m1(a_zxy, b_yzx, 4);
  return __riscv_vfsub_vv_f32m1(mul1, mul2, 4);
}

// Matrix helper functions for RISC-V
static inline vfloat32m1_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return __riscv_vle32_v_f32m1(&mat->m[row * 4], 4);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(&mat->m[row * 4], vec, 4);
}

static inline vfloat32m1_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return __riscv_vle32_v_f32m1(&mat->m[row * 4], 4);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, vfloat32m1_t vec) {
  __riscv_vse32_v_f32m1(&mat->m[row * 4], vec, 4);
}

static inline vfloat32m1_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  float temp[4] = {mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]};
  return __riscv_vle32_v_f32m1(temp, 4);
}

#endif

// WebAssembly SIMD
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
#include <wasm_simd128.h>

// Helper functions for WebAssembly SIMD
static inline v128_t wcn_load_vec2_partial(const float *v) {
  // Vec2 只有 8 字节 (2 floats)，直接读取避免越界
  return wasm_f32x4_make(v[0], v[1], 0.0f, 0.0f);
}

static inline void wcn_store_vec2_partial(float *v, v128_t vec) {
  v[0] = wasm_f32x4_extract_lane(vec, 0);
  v[1] = wasm_f32x4_extract_lane(vec, 1);
}

static inline v128_t wcn_load_vec3_partial(const float *v) {
  return wasm_f32x4_make(v[0], v[1], v[2], 0.0f);
}

static inline void wcn_store_vec3_partial(float *v, v128_t vec) {
  v[0] = wasm_f32x4_extract_lane(vec, 0);
  v[1] = wasm_f32x4_extract_lane(vec, 1);
  v[2] = wasm_f32x4_extract_lane(vec, 2);
}

static inline float wcn_hadd_f32(v128_t vec) {
  // Perform horizontal sum using WASM intrinsics
  v128_t shuf = wasm_i32x4_shuffle(vec, vec, 2, 3, 0, 1);
  v128_t sums = wasm_f32x4_add(vec, shuf);
  shuf = wasm_i32x4_shuffle(sums, sums, 1, 0, 0, 0);
  return wasm_f32x4_extract_lane(wasm_f32x4_add(sums, shuf), 0);
}

// WASM cross product helper for 3D vectors
static inline v128_t wcn_cross_wasm(v128_t a, v128_t b) {
  // Cross product: a x b = (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
  v128_t a_yzx = wasm_i32x4_shuffle(a, a, 1, 2, 0, 3);
  v128_t b_zxy = wasm_i32x4_shuffle(b, b, 2, 0, 1, 3);
  v128_t a_zxy = wasm_i32x4_shuffle(a, a, 2, 0, 1, 3);
  v128_t b_yzx = wasm_i32x4_shuffle(b, b, 1, 2, 0, 3);

  v128_t mul1 = wasm_f32x4_mul(a_yzx, b_zxy);
  v128_t mul2 = wasm_f32x4_mul(a_zxy, b_yzx);
  return wasm_f32x4_sub(mul1, mul2);
}

// Matrix helper functions for WASM
static inline v128_t wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return wasm_v128_load(&mat->m[row * 4]);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, v128_t vec) {
  wasm_v128_store(&mat->m[row * 4], vec);
}

static inline v128_t wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return wasm_v128_load(&mat->m[row * 4]);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, v128_t vec) {
  wasm_v128_store(&mat->m[row * 4], vec);
}

static inline v128_t wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  return wasm_f32x4_make(mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]);
}

#endif

// LoongArch SIMD
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
#include <lsxintrin.h>

/*
 * LSX lane extraction and splat.
 *
 * clang's lsxintrin.h provides `__lsx_vpickve2gr_w(__m128, int)`, which yields the raw 32-bit
 * pattern of one lane, and `__lsx_vldrepl_w(const void*, int)` for load-and-replicate. The
 * names this file used before (vextractw, vfrep2vr_w, vfmov_s_f, vpickve2gr_w_s, vreplfr2vr_s,
 * vfrstps_s) appear nowhere in that header.
 *
 * The reinterpretation is the whole point: assigning vpickve2gr_w's `int` directly to a `float`
 * converts numerically (3 -> 3.0f) instead of recovering the float whose bits are 3. That would
 * compile cleanly and be wrong on every LSX build.
 */
static inline float wcn_lsx_extract_f(__m128 v, const int idx) {
  /*
   * NOT __lsx_vpickve2gr_w(v, idx): its lane operand must be a compile-time constant (clang:
   * "argument to '__builtin_lsx_vpickve2gr_w' must be a constant integer"), and callers pass a
   * runtime index. Storing the vector and indexing the array sidesteps the immediate entirely.
   * The bit reinterpretation is unnecessary here -- reading the float back from memory IS the
   * reinterpretation, and it cannot be confused with a numeric int->float conversion.
   */
  float lanes[4];
  __lsx_vst(v, lanes, 0);
  return lanes[idx];
}

static inline __m128 wcn_lsx_splat_f(const float x) {
  /* takes the address of its parameter, so a non-lvalue argument is fine */
  return (__m128)__lsx_vldrepl_w(&x, 0);
}

// Helper functions for LoongArch LSX
static inline __m128 wcn_load_vec2_partial(const float *v) {
  __m128 temp = __lsx_vldrepl_w(v, 0);
  temp = __lsx_vinsgr2vr_w(temp, v[1], 1);
  return temp;
}

static inline void wcn_store_vec2_partial(float *v, __m128 vec) {
  v[0] = wcn_lsx_extract_f(vec, 0);
  v[1] = wcn_lsx_extract_f(vec, 1);
}

static inline __m128 wcn_load_vec3_partial(const float *v) {
  __m128 temp = __lsx_vldrepl_w(v, 0);
  temp = __lsx_vinsgr2vr_w(temp, v[1], 1);
  temp = __lsx_vinsgr2vr_w(temp, v[2], 2);
  return temp;
}

static inline void wcn_store_vec3_partial(float *v, __m128 vec) {
  v[0] = wcn_lsx_extract_f(vec, 0);
  v[1] = wcn_lsx_extract_f(vec, 1);
  v[2] = wcn_lsx_extract_f(vec, 2);
}

static inline float wcn_hadd_f32(__m128 vec) {
  __m128 shuf = __lsx_vshuf4i_w(vec, 0x1B);  // Shuffle: [2, 3, 0, 1]
  __m128 sums = __lsx_vfadd_s(vec, shuf);
  shuf = __lsx_vshuf4i_w(sums, 0x01);  // Shuffle: [1, 0, x, x]
  sums = __lsx_vfadd_s(sums, shuf);
  return wcn_lsx_extract_f(sums, 0);
}

// LoongArch cross product helper for 3D vectors
static inline __m128 wcn_cross_lsx(__m128 a, __m128 b) {
  __m128 a_yzx = __lsx_vshuf4i_w(a, 0x69);  // Shuffle: [1, 2, 0, 3] -> 01101001 binary = 0x69
  __m128 b_zxy = __lsx_vshuf4i_w(b, 0x96);  // Shuffle: [2, 0, 1, 3] -> 10010110 binary = 0x96
  __m128 a_zxy = __lsx_vshuf4i_w(a, 0x96);  // Shuffle: [2, 0, 1, 3] -> 10010110 binary = 0x96
  __m128 b_yzx = __lsx_vshuf4i_w(b, 0x69);  // Shuffle: [1, 2, 0, 3] -> 01101001 binary = 0x69

  __m128 mul1 = __lsx_vfmul_s(a_yzx, b_zxy);
  __m128 mul2 = __lsx_vfmul_s(a_zxy, b_yzx);
  return __lsx_vfsub_s(mul1, mul2);
}

// Matrix helper functions for LoongArch
static inline __m128 wcn_mat3_get_row(const WMATH_TYPE(Mat3) * mat, int row) {
  return __lsx_vld(&mat->m[row * 4], 0);
}

static inline void wcn_mat3_set_row(WMATH_TYPE(Mat3) * mat, int row, __m128 vec) {
  __lsx_vst(vec, &mat->m[row * 4], 0);
}

static inline __m128 wcn_mat4_get_row(const WMATH_TYPE(Mat4) * mat, int row) {
  return __lsx_vld(&mat->m[row * 4], 0);
}

static inline void wcn_mat4_set_row(WMATH_TYPE(Mat4) * mat, int row, __m128 vec) {
  __lsx_vst(vec, &mat->m[row * 4], 0);
}

static inline __m128 wcn_mat4_get_col(const WMATH_TYPE(Mat4) * mat, int col) {
  return (const __m128){mat->m[col], mat->m[col + 4], mat->m[col + 8], mat->m[col + 12]};
}

#endif

// Missing FMA functions for x86_64 SSE (when FMA not available)
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64 && !defined(WCN_HAS_FMA)
static inline __m128 wcn_fma_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_add_ps(_mm_mul_ps(a, b), c);
}

static inline __m128 wcn_fma_mul_sub_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_sub_ps(_mm_mul_ps(a, b), c);
}

static inline __m128 wcn_fma_neg_mul_add_ps(const __m128 a, const __m128 b, const __m128 c) {
  return _mm_sub_ps(c, _mm_mul_ps(a, b));
}
#endif

// SIMD branchless selection for SSE
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_select_ps(const __m128 condition_mask, const __m128 a, const __m128 b) {
  // Use bitwise operations to select between a and b without branching
  return _mm_or_ps(_mm_and_ps(condition_mask, a),
                   _mm_andnot_ps(condition_mask, b));
}

#endif

// Inverse square root for SSE: exact sqrt + div.
// The previous _mm_rsqrt_ps + one Newton-Raphson step left ~6e-7 relative error
// (enough to fail a 1e-5 comparison on normalize) and did not match the
// original wgpu-matrix, which computes 1/Math.sqrt(lenSq).
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_fast_inv_sqrt_ps(const __m128 x) {
  const __m128 one = _mm_set1_ps(1.0f);
  return _mm_div_ps(one, _mm_sqrt_ps(x));
}

#endif

// Platform-specific inverse square root implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_fast_inv_sqrt_wasm(const v128_t x) {
  // WASM does not have a direct reciprocal square root approximation
  // Use standard division and square root for precision
  v128_t one = wasm_f32x4_splat(1.0f);
  return wasm_f32x4_div(one, wasm_f32x4_sqrt(x));
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_fast_inv_sqrt_lsx(const __m128 x) {
  // Use standard division and sqrt for precision with LoongArch
  __m128 one = __lsx_vldrepl_w((float[]){1.0f}, 0);
  __m128 sqrt_x = __lsx_vfsqrt_s(x);
  return __lsx_vfdiv_s(one, sqrt_x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_fast_inv_sqrt_rvv(const vfloat32m1_t x) {
  // Use standard division and sqrt for precision with RISC-V vector extension
  vfloat32m1_t one = __riscv_vfmv_v_f_f32m1(1.0f, 4);
  vfloat32m1_t sqrt_x = __riscv_vfsqrt_v_f32m1(x, 4); // assuming SEW=32, LMUL=1
  return __riscv_vfdiv_vv_f32m1(one, sqrt_x, 4);
}
#endif

// Unified cross product function that uses platform-specific implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_cross_platform(__m128 a, __m128 b) {
  return wcn_cross_ps(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
static inline float32x4_t wcn_cross_platform(float32x4_t a, float32x4_t b) {
  return wcn_cross_neon(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_cross_platform(vfloat32m1_t a, vfloat32m1_t b) {
  return wcn_cross_rvv(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_cross_platform(v128_t a, v128_t b) {
  return wcn_cross_wasm(a, b);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_cross_platform(__m128 a, __m128 b) {
  return wcn_cross_lsx(a, b);
}
#endif

// Unified fast inverse square root function that uses platform-specific implementations
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
static inline __m128 wcn_fast_inv_sqrt_platform(__m128 x) {
  return wcn_fast_inv_sqrt_ps(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
static inline float32x4_t wcn_fast_inv_sqrt_platform(float32x4_t x) {
  // Exact sqrt + divide, matching the original's 1/Math.sqrt. The previous
  // vrsqrteq + one refinement step kept ~6e-7 relative error.
  const float32x4_t one = vdupq_n_f32(1.0f);
  return vdivq_f32(one, vsqrtq_f32(x));
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
static inline vfloat32m1_t wcn_fast_inv_sqrt_platform(vfloat32m1_t x) {
  return wcn_fast_inv_sqrt_rvv(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
static inline v128_t wcn_fast_inv_sqrt_platform(v128_t x) {
  return wcn_fast_inv_sqrt_wasm(x);
}
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
static inline __m128 wcn_fast_inv_sqrt_platform(__m128 x) {
  return wcn_fast_inv_sqrt_lsx(x);
}
#endif

// Scalar inverse square root.
// This used to be the Quake `0x5f3759df` bit-hack with one Newton-Raphson step:
// only ~1.5e-3 relative error, so every normalize()/set_length()/truncate()/
// rotation_to() result was corrupted beyond any reasonable tolerance, and the
// int/float type punning is strict-aliasing UB in the first place.
// The original wgpu-matrix computes 1/Math.sqrt; sqrt+div is a hardware
// instruction here, so exactness costs almost nothing.
static inline float wcn_fast_inv_sqrt(const float x) {
  return 1.0f / sqrtf(x);
}

static inline float wcn_safe_div_float(const float a, const float b) {
  return b == 0.0f ? 0.0f : a / b;
}

extern const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4];

extern float EPSILON;

extern bool EPSILON_IS_SET;

float wcn_math_set_epsilon(const float epsilon);

float wcn_math_get_epsilon();

// ========================================================================
#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_INTERNAL_H
