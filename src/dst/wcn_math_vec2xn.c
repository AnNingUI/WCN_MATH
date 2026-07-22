/*
- [ ] Vec2xN
  - [x] WMATH_ADD
  - [ ] WMATH_CREATE, WMATH_SET, WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY
  - [ ] WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND, WMATH_CLAMP
  - [ ] WMATH_SUB, WMATH_ADD_SCALED
  - [ ] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_DIV, WMATH_DIV_SCALAR
  - [ ] WMATH_DOT, WMATH_CROSS, WMATH_LENGTH, WMATH_LENGTH_SQ
  - [ ] WMATH_DISTANCE, WMATH_DISTANCE_SQ, WMATH_NORMALIZE
  - [ ] WMATH_NEGATE, WMATH_INVERSE, WMATH_LERP, WMATH_LERP_V
  - [ ] WMATH_FMAX, WMATH_FMIN, WMATH_ANGLE
  - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
  - [ ] WMATH_RANDOM, WMATH_ROTATE, WMATH_SET_LENGTH, WMATH_TRUNCATE, WMATH_MIDPOINT
*/
#include <assert.h> // 用于断言
#include <math.h>
#include "WCN/WCN_MATH_SOA.h"
#include "common/wcn_math_internal.h"

///| 堆
WMATH_TYPE(Vec2xN) * wcn_math_Vec2xN_malloc(const size_t count) 
{
    WMATH_TYPE(Vec2xN)* v = malloc(sizeof(WMATH_TYPE(Vec2xN)));
    v->count = count;
    v->x = (float*)wcn_aligned_alloc(16, count * sizeof(float));
    v->y = (float*)wcn_aligned_alloc(16, count * sizeof(float));
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
    __m128 ss = __lsx_vreplfr2vr_s(s);

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
