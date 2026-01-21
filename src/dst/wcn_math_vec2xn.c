#include <assert.h> // 用于断言
#include <math.h>
#include <stdbool.h>
#include "WCN/WCN_MATH_MACROS.h"
#include "common/wcn_math_internal.h"
WMATH_TYPE(Vec2xN) * wcn_math_Vec2xN_malloc(const size_t count) {
    WMATH_TYPE(Vec2xN)* v = malloc(sizeof(WMATH_TYPE(Vec2xN)));
    v->count = count;
    v->x = (float*)wcn_aligned_alloc(16, count * sizeof(float));
    v->y = (float*)wcn_aligned_alloc(16, count * sizeof(float));
    return v;
}


void wcn_math_Vec2xN_free(WMATH_TYPE(Vec2xN) * ptr) {
    if (ptr == NULL) return;

    // 1. 先释放内部的数据（必须用对应的 aligned_free）
    if (ptr->x) wcn_aligned_free(ptr->x);
    if (ptr->y) wcn_aligned_free(ptr->y);

    // 2. 最后释放结构体壳子（必须用普通 free）
    free(ptr);
}

// ==========================================
// 获取 (Read): 从 Vec2xN 中读取第 index 个向量到 out
// ==========================================
bool wcn_math_Vec2xN_get(
    WMATH_TYPE(Vec2)* out,            // 输出目标 (相当于你原来的 dst)
    const WMATH_TYPE(Vec2xN)* src,    // 输入源 (只读，加 const)
    size_t index
) {
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
    WMATH_TYPE(Vec2xN)* dst,          // 目标容器 (会被修改)
    const size_t index,
    const WMATH_TYPE(Vec2)* input     // 输入值 (只读)
) {
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
) {
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