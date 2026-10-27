/*
- [ ] Vec2xN
  - [x] WMATH_ADD, WMATH_ADD_SCALED, WMATH_SUB
  - [x] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_DIV, WMATH_DIV_SCALAR
  - [x] WMATH_CLAMP, WMATH_NEGATE, WMATH_INVERSE, WMATH_NORMALIZE
  - [x] WMATH_LERP, WMATH_LERP_V, WMATH_MIDPOINT, WMATH_SET_LENGTH
  - [x] WMATH_DOT, WMATH_LENGTH, WMATH_LENGTH_SQ
  - [x] WMATH_DISTANCE, WMATH_DISTANCE_SQ
  - [x] WMATH_FMIN, WMATH_FMAX
  - [x] WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY, WMATH_CREATE, WMATH_SET
  - [x] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
- [ ] Vec2xN remaining: WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND (need new plane kernels),
      WMATH_ANGLE, WMATH_RANDOM, WMATH_ROTATE, WMATH_TRUNCATE, WMATH_CROSS (2D has no
      cross product scalar-side)
*/
#include <assert.h> // 用于断言
#include <math.h>
#include <string.h>
#include "WCN/WCN_MATH_SOA.h"
#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include "../common/wcn_math_soa_kernels.h"

///| 堆
WMATH_TYPE(Vec2xN) * wcn_math_Vec2xN_malloc(const size_t count) 
{
    WMATH_TYPE(Vec2xN)* v = malloc(sizeof(WMATH_TYPE(Vec2xN)));
    v->count = count;
    v->x = (float*)wcn_aligned_alloc((count) * sizeof(float), 16);
    v->y = (float*)wcn_aligned_alloc((count) * sizeof(float), 16);
    return v;
}

///| 栈写法
///| ```c
///| WCN_ALIGN16 float x_array[100];
///| WCN_ALIGN16 float y_array[100];
///| WMATH_TYPE(Vec2xN) v = {
///|    .count = 100,
///|    .x = x_array,
///|    .y = y_array
///| };
///| ```
///| 

void wcn_math_Vec2xN_free(WMATH_TYPE(Vec2xN) * ptr) 
{
    if (ptr == nullptr) return;

    // 1. 先释放内部的数据（必须用对应的 aligned_free）
    if (ptr->x) { wcn_aligned_free(ptr->x); ptr->x = nullptr; }
    if (ptr->y) { wcn_aligned_free(ptr->y); ptr->y = nullptr; }
    ptr->count = 0;
    // 2. 最后释放结构体壳子（必须用普通 free）
    free(ptr);
}

float * wcn_math_Vec2xN_get_xarray(WMATH_TYPE(Vec2xN) * ptr) 
{
    if (ptr == nullptr) return nullptr;
    return ptr->x;
}

float * wcn_math_Vec2xN_get_yarray(WMATH_TYPE(Vec2xN) * ptr) 
{
    if (ptr == nullptr) return nullptr;
    return ptr->y;
}

size_t wcn_math_Vec2xN_get_count(const WMATH_TYPE(Vec2xN) * ptr) 
{
    if (ptr == nullptr) return 0;
    return ptr->count;
}

// ==========================================
// 获取 (Read): 从 Vec2xN 中读取第 index 个向量到 out
// ==========================================
bool wcn_math_Vec2xN_get(
    WMATH_TYPE(Vec2)* out,            
    const WMATH_TYPE(Vec2xN)* src,    
    size_t index
) 
{
    // 安全检查
    if (!out || !src || index >= src->count) {
        return false;
    }

    // SoA 布局读取：分别读取 x 和 y 数组
    out->v[0] = src->x[index];
    out->v[1] = src->y[index];

    return true;
}

// ==========================================
// 设置 (Write): 将 input 写入到 Vec2xN 的第 index 位置
// ==========================================
bool wcn_math_Vec2xN_set(
    WMATH_TYPE(Vec2xN)* dst,          
    const size_t index,
    const WMATH_TYPE(Vec2)* input     
) 
{
    // 安全检查
    if (!dst || !input || index >= dst->count) {
        return false;
    }

    // SoA 布局写入
    dst->x[index] = input->v[0];
    dst->y[index] = input->v[1];

    return true;
}

bool wcn_math_Vec2xN_set_by_xy
(
    const WMATH_TYPE(Vec2xN)* v,
    const size_t index, const float x, const float y
)
{
    if (!v) return false;
    v->x[index] = x;
    v->y[index] = y;
    return true;
}

void WMATH_ADD(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b
) 
{
    // 1. 安全检查：长度必须一致
    // 使用 assert 在开发阶段直接拦截错误，release 版保留 if 防止越界
    assert(dst->count == a->count && dst->count == b->count);
    if (dst->count != a->count || dst->count != b->count) {
        return; // 或者返回错误码
    }

    const size_t count = a->count;
    size_t i = 0; // 2. 修正：必须用 size_t，否则大数组会溢出

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE
    for (; i + 4 <= count; i += 4) {
        // 3. 修正：使用 loadu/storeu 处理非对齐内存，避免崩溃
        __m128 ax = _mm_loadu_ps(&a->x[i]);
        __m128 ay = _mm_loadu_ps(&a->y[i]);
        __m128 bx = _mm_loadu_ps(&b->x[i]);
        __m128 by = _mm_loadu_ps(&b->y[i]);

        _mm_storeu_ps(&dst->x[i], _mm_add_ps(ax, bx));
        _mm_storeu_ps(&dst->y[i], _mm_add_ps(ay, by));
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // NEON
    for (; i + 4 <= count; i += 4) {
        float32x4_t ax = vld1q_f32(&a->x[i]);
        float32x4_t ay = vld1q_f32(&a->y[i]);
        float32x4_t bx = vld1q_f32(&b->x[i]);
        float32x4_t by = vld1q_f32(&b->y[i]);

        vst1q_f32(&dst->x[i], vaddq_f32(ax, bx));
        vst1q_f32(&dst->y[i], vaddq_f32(ay, by));
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector
    size_t vl;
    for (; i < count; i += vl) {
        vl = __riscv_vsetvl_e32m1(count - i);

        vfloat32m1_t ax = __riscv_vle32_v_f32m1(&a->x[i], vl);
        vfloat32m1_t ay = __riscv_vle32_v_f32m1(&a->y[i], vl);
        vfloat32m1_t bx = __riscv_vle32_v_f32m1(&b->x[i], vl);
        vfloat32m1_t by = __riscv_vle32_v_f32m1(&b->y[i], vl);

        __riscv_vse32_v_f32m1(&dst->x[i], __riscv_vfadd_vv_f32m1(ax, bx, vl), vl);
        __riscv_vse32_v_f32m1(&dst->y[i], __riscv_vfadd_vv_f32m1(ay, by, vl), vl);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WASM
    for (; i + 4 <= count; i += 4) {
        v128_t ax = wasm_v128_load(&a->x[i]);
        v128_t ay = wasm_v128_load(&a->y[i]);
        v128_t bx = wasm_v128_load(&b->x[i]);
        v128_t by = wasm_v128_load(&b->y[i]);

        wasm_v128_store(&dst->x[i], wasm_f32x4_add(ax, bx));
        wasm_v128_store(&dst->y[i], wasm_f32x4_add(ay, by));
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX
    for (; i + 4 <= count; i += 4) {
        __m128 ax = __lsx_vld(&a->x[i], 0);
        __m128 ay = __lsx_vld(&a->y[i], 0);
        __m128 bx = __lsx_vld(&b->x[i], 0);
        __m128 by = __lsx_vld(&b->y[i], 0);

        __lsx_vst(__lsx_vfadd_s(ax, bx), &dst->x[i], 0);
        __lsx_vst(__lsx_vfadd_s(ay, by), &dst->y[i], 0);
    }

#endif

    // Scalar tail
    for (; i < count; i++) {
        dst->x[i] = a->x[i] + b->x[i];
        dst->y[i] = a->y[i] + b->y[i];
    }
}

void WMATH_ADD_SCALED(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b,
    const float s
) 
{
    // ===============================
    // Safety check
    // ===============================
    assert(dst && a && b);
    assert(dst->count == a->count && dst->count == b->count);
    if (!dst || !a || !b ||
        dst->count != a->count ||
        dst->count != b->count) {
        return;
    }

    const size_t count = a->count;
    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // ===============================
    // x86 SSE / FMA
    // ===============================
    __m128 ss = _mm_set1_ps(s);

    for (; i + 4 <= count; i += 4) {
        __m128 ax = _mm_loadu_ps(&a->x[i]);
        __m128 ay = _mm_loadu_ps(&a->y[i]);
        __m128 bx = _mm_loadu_ps(&b->x[i]);
        __m128 by = _mm_loadu_ps(&b->y[i]);

    #if defined(__FMA__)
        // dst = a + b * s
        __m128 rx = _mm_fmadd_ps(bx, ss, ax);
        __m128 ry = _mm_fmadd_ps(by, ss, ay);
    #else
        __m128 rx = _mm_add_ps(ax, _mm_mul_ps(bx, ss));
        __m128 ry = _mm_add_ps(ay, _mm_mul_ps(by, ss));
    #endif

        _mm_storeu_ps(&dst->x[i], rx);
        _mm_storeu_ps(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ===============================
    // ARM NEON
    // ===============================
    float32x4_t ss = vdupq_n_f32(s);

    for (; i + 4 <= count; i += 4) {
        float32x4_t ax = vld1q_f32(&a->x[i]);
        float32x4_t ay = vld1q_f32(&a->y[i]);
        float32x4_t bx = vld1q_f32(&b->x[i]);
        float32x4_t by = vld1q_f32(&b->y[i]);

        float32x4_t rx = vmlaq_f32(ax, bx, ss); // a + b*s
        float32x4_t ry = vmlaq_f32(ay, by, ss);

        vst1q_f32(&dst->x[i], rx);
        vst1q_f32(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // ===============================
    // RISC-V Vector
    // ===============================
    size_t vl;
    for (; i < count; i += vl) {
        vl = __riscv_vsetvl_e32m1(count - i);

        vfloat32m1_t ax = __riscv_vle32_v_f32m1(&a->x[i], vl);
        vfloat32m1_t ay = __riscv_vle32_v_f32m1(&a->y[i], vl);
        vfloat32m1_t bx = __riscv_vle32_v_f32m1(&b->x[i], vl);
        vfloat32m1_t by = __riscv_vle32_v_f32m1(&b->y[i], vl);

        vfloat32m1_t ss = __riscv_vfmv_v_f_f32m1(s, vl);

        vfloat32m1_t rx = __riscv_vfmadd_vv_f32m1(bx, ss, ax, vl);
        vfloat32m1_t ry = __riscv_vfmadd_vv_f32m1(by, ss, ay, vl);

        __riscv_vse32_v_f32m1(&dst->x[i], rx, vl);
        __riscv_vse32_v_f32m1(&dst->y[i], ry, vl);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // ===============================
    // WASM SIMD
    // ===============================
    v128_t ss = wasm_f32x4_splat(s);

    for (; i + 4 <= count; i += 4) {
        v128_t ax = wasm_v128_load(&a->x[i]);
        v128_t ay = wasm_v128_load(&a->y[i]);
        v128_t bx = wasm_v128_load(&b->x[i]);
        v128_t by = wasm_v128_load(&b->y[i]);

        v128_t rx = wasm_f32x4_add(ax, wasm_f32x4_mul(bx, ss));
        v128_t ry = wasm_f32x4_add(ay, wasm_f32x4_mul(by, ss));

        wasm_v128_store(&dst->x[i], rx);
        wasm_v128_store(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // ===============================
    // LoongArch LSX
    // ===============================
    __m128 ss = wcn_lsx_splat_f(s);

    for (; i + 4 <= count; i += 4) {
        __m128 ax = __lsx_vld(&a->x[i], 0);
        __m128 ay = __lsx_vld(&a->y[i], 0);
        __m128 bx = __lsx_vld(&b->x[i], 0);
        __m128 by = __lsx_vld(&b->y[i], 0);

        __m128 rx = __lsx_vfadd_s(ax, __lsx_vfmul_s(bx, ss));
        __m128 ry = __lsx_vfadd_s(ay, __lsx_vfmul_s(by, ss));

        __lsx_vst(rx, &dst->x[i], 0);
        __lsx_vst(ry, &dst->y[i], 0);
    }

#endif

    // ===============================
    // Scalar tail
    // ===============================
    for (; i < count; i++) {
        dst->x[i] = a->x[i] + b->x[i] * s;
        dst->y[i] = a->y[i] + b->y[i] * s;
    }
}

void WMATH_SUB(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b
) 
{
    // ===============================
    // Safety check
    // ===============================
    assert(dst && a && b);
    assert(dst->count == a->count && dst->count == b->count);
    if (!dst || !a || !b ||
        dst->count != a->count ||
        dst->count != b->count) {
        return;
    }

    const size_t count = a->count;
    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // ===============================
    // x86 SSE
    // ===============================
    for (; i + 4 <= count; i += 4) {
        __m128 ax = _mm_loadu_ps(&a->x[i]);
        __m128 ay = _mm_loadu_ps(&a->y[i]);
        __m128 bx = _mm_loadu_ps(&b->x[i]);
        __m128 by = _mm_loadu_ps(&b->y[i]);

        __m128 rx = _mm_sub_ps(ax, bx);
        __m128 ry = _mm_sub_ps(ay, by);

        _mm_storeu_ps(&dst->x[i], rx);
        _mm_storeu_ps(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ===============================
    // ARM NEON
    // ===============================
    for (; i + 4 <= count; i += 4) {
        float32x4_t ax = vld1q_f32(&a->x[i]);
        float32x4_t ay = vld1q_f32(&a->y[i]);
        float32x4_t bx = vld1q_f32(&b->x[i]);
        float32x4_t by = vld1q_f32(&b->y[i]);

        float32x4_t rx = vsubq_f32(ax, bx);
        float32x4_t ry = vsubq_f32(ay, by);

        vst1q_f32(&dst->x[i], rx);
        vst1q_f32(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // ===============================
    // RISC-V Vector
    // ===============================
    size_t vl;
    for (; i < count; i += vl) {
        vl = __riscv_vsetvl_e32m1(count - i);

        vfloat32m1_t ax = __riscv_vle32_v_f32m1(&a->x[i], vl);
        vfloat32m1_t ay = __riscv_vle32_v_f32m1(&a->y[i], vl);
        vfloat32m1_t bx = __riscv_vle32_v_f32m1(&b->x[i], vl);
        vfloat32m1_t by = __riscv_vle32_v_f32m1(&b->y[i], vl);

        vfloat32m1_t rx = __riscv_vfsub_vv_f32m1(ax, bx, vl);
        vfloat32m1_t ry = __riscv_vfsub_vv_f32m1(ay, by, vl);

        __riscv_vse32_v_f32m1(&dst->x[i], rx, vl);
        __riscv_vse32_v_f32m1(&dst->y[i], ry, vl);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // ===============================
    // WASM SIMD
    // ===============================
    for (; i + 4 <= count; i += 4) {
        v128_t ax = wasm_v128_load(&a->x[i]);
        v128_t ay = wasm_v128_load(&a->y[i]);
        v128_t bx = wasm_v128_load(&b->x[i]);
        v128_t by = wasm_v128_load(&b->y[i]);

        v128_t rx = wasm_f32x4_sub(ax, bx);
        v128_t ry = wasm_f32x4_sub(ay, by);

        wasm_v128_store(&dst->x[i], rx);
        wasm_v128_store(&dst->y[i], ry);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // ===============================
    // LoongArch LSX
    // ===============================
    for (; i + 4 <= count; i += 4) {
        __m128 ax = __lsx_vld(&a->x[i], 0);
        __m128 ay = __lsx_vld(&a->y[i], 0);
        __m128 bx = __lsx_vld(&b->x[i], 0);
        __m128 by = __lsx_vld(&b->y[i], 0);

        __m128 rx = __lsx_vfsub_s(ax, bx);
        __m128 ry = __lsx_vfsub_s(ay, by);

        __lsx_vst(rx, &dst->x[i], 0);
        __lsx_vst(ry, &dst->y[i], 0);
    }

#endif

    // ===============================
    // Scalar tail
    // ===============================
    for (; i < count; i++) {
        dst->x[i] = a->x[i] - b->x[i];
        dst->y[i] = a->y[i] - b->y[i];
    }
}


/* ==========================================================================
 * Vec2xN: the remainder of the operator surface
 *
 * Wiring onto the two-plane kernels in wcn_math_soa_kernels.h; the only new logic is the
 * operand checks. Every reduction here produces one scalar per element, hence a bare
 * float* out rather than a Vec2xN.
 * ========================================================================== */

static inline int v2xn_ok2(const WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                           const WMATH_TYPE(Vec2xN)* b) {
    return dst && a && b && dst->count == a->count && dst->count == b->count && dst->x &&
           dst->y && a->x && a->y && b->x && b->y;
}
static inline int v2xn_ok1(const WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    return dst && a && dst->count == a->count && dst->x && dst->y && a->x && a->y;
}

#define V2XN_BINOP(NAME, KERN)                                                         \
void NAME(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,                        \
          const WMATH_TYPE(Vec2xN)* b) {                                               \
    assert(v2xn_ok2(dst, a, b));                                                       \
    if (!v2xn_ok2(dst, a, b)) return;                                                  \
    KERN(dst->x, a->x, b->x, a->count);                                                \
    KERN(dst->y, a->y, b->y, a->count);                                                \
}

V2XN_BINOP(WMATH_MULTIPLY(Vec2xN), wcn_soa_mul)
V2XN_BINOP(WMATH_DIV(Vec2xN), wcn_soa_div)
V2XN_BINOP(WMATH_FMIN(Vec2xN), wcn_soa_min)
V2XN_BINOP(WMATH_FMAX(Vec2xN), wcn_soa_max)

#undef V2XN_BINOP

#define V2XN_SCALAR(NAME, KERN)                                                        \
void NAME(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float s) {       \
    assert(v2xn_ok1(dst, a));                                                          \
    if (!v2xn_ok1(dst, a)) return;                                                     \
    KERN(dst->x, a->x, s, a->count);                                                   \
    KERN(dst->y, a->y, s, a->count);                                                   \
}

V2XN_SCALAR(WMATH_MULTIPLY_SCALAR(Vec2xN), wcn_soa_mul_s)
V2XN_SCALAR(WMATH_DIV_SCALAR(Vec2xN), wcn_soa_div_s)

#undef V2XN_SCALAR

void WMATH_CLAMP(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                         const float lo, const float hi) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    wcn_soa_clamp(dst->x, a->x, lo, hi, a->count);
    wcn_soa_clamp(dst->y, a->y, lo, hi, a->count);
}

void WMATH_NEGATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    wcn_soa_neg(dst->x, a->x, a->count);
    wcn_soa_neg(dst->y, a->y, a->count);
}

void WMATH_INVERSE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* Zero maps to zero, matching WMATH_INVERSE(Vec2). The Vec3/Vec4 inverses divide
     * unconditionally instead -- the scalar types disagree, so this is not a mistake in
     * either place. */
    wcn_soa_inv_guarded(dst->x, a->x, a->count);
    wcn_soa_inv_guarded(dst->y, a->y, a->count);
}

void WMATH_NORMALIZE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* Fused: the reciprocal length has to be held while both planes are scaled. Below eps
     * the result is ZERO, matching WMATH_NORMALIZE(Vec2). */
    wcn_soa_normalize2(dst->x, dst->y, a->x, a->y, a->count);
}

void WMATH_LERP(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                        const WMATH_TYPE(Vec2xN)* b, const float t) {
    assert(v2xn_ok2(dst, a, b));
    if (!v2xn_ok2(dst, a, b)) return;
    wcn_soa_lerp(dst->x, a->x, b->x, t, a->count);
    wcn_soa_lerp(dst->y, a->y, b->y, t, a->count);
}

void WMATH_LERP_V(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                          const WMATH_TYPE(Vec2xN)* b, const WMATH_TYPE(Vec2xN)* t) {
    assert(v2xn_ok2(dst, a, b));
    assert(t && t->count == a->count);
    if (!v2xn_ok2(dst, a, b) || !t || t->count != a->count) return;
    wcn_soa_lerp_plane(dst->x, a->x, b->x, t->x, a->count);
    wcn_soa_lerp_plane(dst->y, a->y, b->y, t->y, a->count);
}

void WMATH_MIDPOINT(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                            const WMATH_TYPE(Vec2xN)* b) {
    assert(v2xn_ok2(dst, a, b));
    if (!v2xn_ok2(dst, a, b)) return;
    /* WMATH_MIDPOINT(Vec2) is literally WMATH_LERP(Vec2)(dst, a, b, 0.5f). */
    wcn_soa_lerp(dst->x, a->x, b->x, 0.5f, a->count);
    wcn_soa_lerp(dst->y, a->y, b->y, 0.5f, a->count);
}

void WMATH_SET_LENGTH(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                              const float len) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* WMATH_SET_LENGTH(Vec2) is NORMALIZE then MULTIPLY_SCALAR, so a sub-epsilon input
     * stays zero rather than becoming NaN. */
    wcn_soa_normalize2(dst->x, dst->y, a->x, a->y, a->count);
    wcn_soa_mul_s(dst->x, dst->x, len, a->count);
    wcn_soa_mul_s(dst->y, dst->y, len, a->count);
}

void WMATH_DOT(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a,
                       const WMATH_TYPE(Vec2xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_dot2(out, a->x, a->y, b->x, b->y, a->count);
}

void WMATH_LENGTH_SQ(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a) {
    if (!out || !a) return;
    wcn_soa_lensq2(out, a->x, a->y, a->count);
}

void WMATH_LENGTH(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a) {
    if (!out || !a) return;
    wcn_soa_len2(out, a->x, a->y, a->count);
}

void WMATH_DISTANCE_SQ(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a,
                               const WMATH_TYPE(Vec2xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_distsq2(out, a->x, a->y, b->x, b->y, a->count);
}

void WMATH_DISTANCE(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a,
                            const WMATH_TYPE(Vec2xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_distsq2(out, a->x, a->y, b->x, b->y, a->count);
    wcn_soa_sqrt(out, out, a->count);
}

void WMATH_COPY(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    wcn_soa_copy(dst->x, a->x, a->count);
    wcn_soa_copy(dst->y, a->y, a->count);
}

void WMATH_ZERO(Vec2xN)(WMATH_TYPE(Vec2xN)* dst) {
    if (!dst || !dst->x) return;
    wcn_soa_fill(dst->x, 0.0f, dst->count);
    wcn_soa_fill(dst->y, 0.0f, dst->count);
}

void WMATH_IDENTITY(Vec2xN)(WMATH_TYPE(Vec2xN)* dst) {
    if (!dst || !dst->x) return;
    /* WMATH_IDENTITY(Vec2) is (1,1). */
    wcn_soa_fill(dst->x, 1.0f, dst->count);
    wcn_soa_fill(dst->y, 1.0f, dst->count);
}

void WMATH_CREATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_CREATE_TYPE(Vec2) c) {
    if (!dst || !dst->x) return;
    wcn_soa_fill(dst->x, c.x, dst->count);
    wcn_soa_fill(dst->y, c.y, dst->count);
}


bool WMATH_EQUALS(Vec2xN)(const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    for (size_t i = 0; i < a->count; i++) {
        if (a->x[i] != b->x[i] || a->y[i] != b->y[i]) return false;
    }
    return true;
}

bool WMATH_EQUALS_APPROXIMATELY(Vec2xN)(const WMATH_TYPE(Vec2xN)* a,
                                        const WMATH_TYPE(Vec2xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    const float ep = wcn_math_get_epsilon();
    for (size_t i = 0; i < a->count; i++) {
        if (!(fabsf(a->x[i] - b->x[i]) < ep)) return false;
        if (!(fabsf(a->y[i] - b->y[i]) < ep)) return false;
    }
    return true;
}

void WMATH_FLOOR(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_FLOOR's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_floor(dst->x, a->x, a->count);
    wcn_soa_floor(dst->y, a->y, a->count);
}

void WMATH_CEIL(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_CEIL's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_ceil(dst->x, a->x, a->count);
    wcn_soa_ceil(dst->y, a->y, a->count);
}

void WMATH_ROUND(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_ROUND's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_round(dst->x, a->x, a->count);
    wcn_soa_round(dst->y, a->y, a->count);
}

void WMATH_TRUNCATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                            const float max_length) {
    assert(v2xn_ok1(dst, a));
    if (!v2xn_ok1(dst, a)) return;
    wcn_soa_truncate2(dst->x, dst->y, a->x, a->y, max_length, a->count);
}

/* ------------------------------------------------------- last Vec2xN operators */

void WMATH_CROSS(Vec2xN)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Vec2xN)* a,
                          const WMATH_TYPE(Vec2xN)* b) {
    if (!out || !a || !b || out->count != a->count || out->count != b->count) return;
    /* WMATH_CROSS(Vec2) writes a Vec3: (0, 0, ax*by - ay*bx). This is the only operator whose
     * output container differs from its inputs; the first two components are zero by
     * definition, so they are filled rather than computed. */
    wcn_soa_fill(out->x, 0.0f, out->count);
    wcn_soa_fill(out->y, 0.0f, out->count);
    wcn_soa_cross2_z(out->z, a->x, a->y, b->x, b->y, out->count);
}

void WMATH_ROTATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a,
                           const WMATH_TYPE(Vec2xN)* b, const float* rad) {
    assert(v2xn_ok2(dst, a, b));
    if (!v2xn_ok2(dst, a, b) || !rad) return;
    /* trig, so per element -- the same wall every rotation in this library hits. */
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec2) va;
        WMATH_TYPE(Vec2) vb;
        WMATH_TYPE(Vec2) r;
        va.v[0] = a->x[i]; va.v[1] = a->y[i];
        vb.v[0] = b->x[i]; vb.v[1] = b->y[i];
        WMATH_ROTATE(Vec2)(&r, va, vb, rad[i]);
        dst->x[i] = r.v[0];
        dst->y[i] = r.v[1];
    }
}

void WMATH_SCALE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* v,
                          const float* scale) {
    assert(v2xn_ok1(dst, v));
    if (!v2xn_ok1(dst, v) || !scale) return;
    /* WMATH_SCALE(Vec2) is MULTIPLY_SCALAR; the factor is a plane, so element-wise. */
    wcn_soa_mul(dst->x, v->x, scale, v->count);
    wcn_soa_mul(dst->y, v->y, scale, v->count);
}

void WMATH_CALL(Vec2xN, transform_mat3)(WMATH_TYPE(Vec2xN)* dst,
                                         const WMATH_TYPE(Vec2xN)* v,
                                         const WMATH_TYPE(Mat3xN)* m) {
    if (!dst || !v || !m || dst->count != v->count || dst->count != m->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec2) vv;
        WMATH_TYPE(Mat3) mm;
        WMATH_TYPE(Vec2) r;
        vv.v[0] = v->x[i]; vv.v[1] = v->y[i];
        wcn_math_Mat3xN_get(&mm, m, i);
        WMATH_CALL(Vec2, transform_mat3)(&r, vv, mm);
        dst->x[i] = r.v[0];
        dst->y[i] = r.v[1];
    }
}

void WMATH_CALL(Vec2xN, transform_mat4)(WMATH_TYPE(Vec2xN)* dst,
                                         const WMATH_TYPE(Vec2xN)* v,
                                         const WMATH_TYPE(Mat4xN)* m) {
    if (!dst || !v || !m || dst->count != v->count || dst->count != m->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec2) vv;
        WMATH_TYPE(Mat4) mm;
        WMATH_TYPE(Vec2) r;
        vv.v[0] = v->x[i]; vv.v[1] = v->y[i];
        wcn_math_Mat4xN_get(&mm, m, i);
        WMATH_CALL(Vec2, transform_mat4)(&r, vv, mm);
        dst->x[i] = r.v[0];
        dst->y[i] = r.v[1];
    }
}

void WMATH_ANGLE(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a,
                          const WMATH_TYPE(Vec2xN)* b) {
    /* Takes float* out and returns void, matching DOT/LENGTH on the xN side: the scalar
     * returns one float, the container produces N of them, so the result is a plane. */
    if (!out || !a || !b || a->count != b->count) return;
    for (size_t i = 0; i < a->count; i++) {
        WMATH_TYPE(Vec2) va;
        WMATH_TYPE(Vec2) vb;
        va.v[0] = a->x[i]; va.v[1] = a->y[i];
        vb.v[0] = b->x[i]; vb.v[1] = b->y[i];
        out[i] = WMATH_ANGLE(Vec2)(va, vb);
    }
}

void WMATH_RANDOM(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const float* scale) {
    if (!dst || !dst->x || !scale) return;
    /* WMATH_RANDOM(Vec2) draws one angle from the global rand() and returns
     * (cos*scale, sin*scale). With `scale` a plane this is per element, so it is a scalar
     * loop -- cosf/sinf again, and the draw order must stay element 0..N-1 for the sequence
     * to match the scalar reference. */
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec2) r;
        WMATH_RANDOM(Vec2)(&r, scale[i]);
        dst->x[i] = r.v[0];
        dst->y[i] = r.v[1];
    }
}
