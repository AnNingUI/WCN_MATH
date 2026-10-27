/*
- [x] Vec4xN
  - [x] malloc, free, get_*array, get_count, get, set, set_by_xyzw
  - [x] WMATH_ADD, WMATH_SUB, WMATH_MULTIPLY, WMATH_DIV, WMATH_FMIN, WMATH_FMAX
  - [x] WMATH_MULTIPLY_SCALAR, WMATH_DIV_SCALAR, WMATH_ADD_SCALED, WMATH_CLAMP
  - [x] WMATH_NEGATE, WMATH_NORMALIZE, WMATH_INVERSE
  - [x] WMATH_LERP, WMATH_LERP_V, WMATH_MIDPOINT
  - [x] WMATH_SET_LENGTH, WMATH_TRUNCATE
  - [x] WMATH_DOT, WMATH_LENGTH, WMATH_LENGTH_SQ
  - [x] WMATH_DISTANCE, WMATH_DISTANCE_SQ
  - [x] WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY, WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
- [ ] Vec4xN remaining: WMATH_CREATE, WMATH_SET, WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND,
      WMATH_ANGLE, WMATH_RANDOM
*/
#include <assert.h>
#include <math.h>

#include "WCN/WCN_MATH_SOA.h"
#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include "../common/wcn_math_soa_kernels.h"

/* Operand checks.
 *
 * The count check is not redundant with the assert: asserts vanish in a release build,
 * and a pair of mismatched containers would otherwise walk off the shorter plane. */
static inline int wcn_v4xn_ok2(const WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                               const WMATH_TYPE(Vec4xN)* b) {
    return dst && a && b && dst->count == a->count && dst->count == b->count && dst->x &&
           dst->y && dst->z && dst->w && a->x && a->y && a->z && a->w && b->x && b->y &&
           b->z && b->w;
}
static inline int wcn_v4xn_ok1(const WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    return dst && a && dst->count == a->count && dst->x && dst->y && dst->z && dst->w &&
           a->x && a->y && a->z && a->w;
}

/* ---------------------------------------------------------------- memory */

WMATH_TYPE(Vec4xN)* wcn_math_Vec4xN_malloc(const size_t count) {
    WMATH_TYPE(Vec4xN)* v = (WMATH_TYPE(Vec4xN)*)malloc(sizeof(WMATH_TYPE(Vec4xN)));
    if (!v) return nullptr;
    v->count = count;
    /* wcn_aligned_alloc takes (bytes, alignment); the reversed order allocates 16 bytes
     * and corrupts on the first store, which is what happened at eight other sites in
     * this library (LOGIC_BUGS.md). */
    v->x = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    v->y = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    v->z = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    v->w = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    return v;
}

float* wcn_math_Vec4xN_get_xarray(WMATH_TYPE(Vec4xN)* ptr) { return ptr ? ptr->x : nullptr; }
float* wcn_math_Vec4xN_get_yarray(WMATH_TYPE(Vec4xN)* ptr) { return ptr ? ptr->y : nullptr; }
float* wcn_math_Vec4xN_get_zarray(WMATH_TYPE(Vec4xN)* ptr) { return ptr ? ptr->z : nullptr; }
float* wcn_math_Vec4xN_get_warray(WMATH_TYPE(Vec4xN)* ptr) { return ptr ? ptr->w : nullptr; }
size_t wcn_math_Vec4xN_get_count(const WMATH_TYPE(Vec4xN)* ptr) { return ptr ? ptr->count : 0; }

bool wcn_math_Vec4xN_get(WMATH_TYPE(Vec4)* out, const WMATH_TYPE(Vec4xN)* src, size_t index) {
    if (!out || !src || index >= src->count) return false;
    out->v[0] = src->x[index];
    out->v[1] = src->y[index];
    out->v[2] = src->z[index];
    out->v[3] = src->w[index];
    return true;
}

bool wcn_math_Vec4xN_set(WMATH_TYPE(Vec4xN)* dst, const size_t index,
                         const WMATH_TYPE(Vec4)* input) {
    if (!dst || !input || index >= dst->count) return false;
    dst->x[index] = input->v[0];
    dst->y[index] = input->v[1];
    dst->z[index] = input->v[2];
    dst->w[index] = input->v[3];
    return true;
}


void wcn_math_Vec4xN_free(WMATH_TYPE(Vec4xN)* vec4xN) {
    if (vec4xN == nullptr) return;

    /* Releases the four planes only; the struct itself belongs to the caller and stays
     * theirs. This must not free the shell: every producer (ImageBitmap_to_Vec4xN and
     * friends) fills in a caller-owned struct, and the documented usage in
     * WCN_MATH_IMAGE.h is that the caller frees such a one with this function. A bare
     * free(vec4xN) here aborted with 0xC0000374 as soon as a stack-allocated Vec4xN was
     * freed. Shells from wcn_math_Vec4xN_malloc are released with _destroy instead.
     *
     * Contrast wcn_math_Vec2xN_free, which does free the shell -- correct there because
     * Vec2xN has no other producer and always comes from its own _malloc. */
    if (vec4xN->x) { wcn_aligned_free(vec4xN->x); vec4xN->x = nullptr; }
    if (vec4xN->y) { wcn_aligned_free(vec4xN->y); vec4xN->y = nullptr; }
    if (vec4xN->z) { wcn_aligned_free(vec4xN->z); vec4xN->z = nullptr; }
    if (vec4xN->w) { wcn_aligned_free(vec4xN->w); vec4xN->w = nullptr; }
    vec4xN->count = 0;
}

void wcn_math_Vec4xN_destroy(WMATH_TYPE(Vec4xN)* vec4xN) {
    if (vec4xN == nullptr) return;
    /* For structs from wcn_math_Vec4xN_malloc: the caller owns the shell there, and it
     * was allocated with malloc, so it is released the same way the plane pointers are
     * cleared first to keep a double call from touching freed memory. */
    wcn_math_Vec4xN_free(vec4xN);
    free(vec4xN);
}

bool wcn_math_Vec4xN_set_by_xyzw(const WMATH_TYPE(Vec4xN)* v, const size_t index,
                                 const float x, const float y, const float z, const float w) {
    if (!v || index >= v->count) return false;
    v->x[index] = x;
    v->y[index] = y;
    v->z[index] = z;
    v->w[index] = w;
    return true;
}

/* ------------------------------------------------------- element-wise binary */

#define WCN_V4X_BINOP(NAME, KERN)                                                      \
void NAME(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,                        \
          const WMATH_TYPE(Vec4xN)* b) {                                               \
    assert(wcn_v4xn_ok2(dst, a, b));                                                   \
    if (!wcn_v4xn_ok2(dst, a, b)) return;                                              \
    KERN(dst->x, a->x, b->x, a->count);                                                \
    KERN(dst->y, a->y, b->y, a->count);                                                \
    KERN(dst->z, a->z, b->z, a->count);                                                \
    KERN(dst->w, a->w, b->w, a->count);                                                \
}

WCN_V4X_BINOP(WMATH_ADD(Vec4xN), wcn_soa_add)
WCN_V4X_BINOP(WMATH_SUB(Vec4xN), wcn_soa_sub)
WCN_V4X_BINOP(WMATH_MULTIPLY(Vec4xN), wcn_soa_mul)
WCN_V4X_BINOP(WMATH_DIV(Vec4xN), wcn_soa_div)
WCN_V4X_BINOP(WMATH_FMIN(Vec4xN), wcn_soa_min)
WCN_V4X_BINOP(WMATH_FMAX(Vec4xN), wcn_soa_max)

#undef WCN_V4X_BINOP

/* ---------------------------------------------------- element-wise, scalar */

#define WCN_V4X_SCALAR(NAME, KERN)                                                     \
void NAME(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float s) {       \
    assert(wcn_v4xn_ok1(dst, a));                                                      \
    if (!wcn_v4xn_ok1(dst, a)) return;                                                 \
    KERN(dst->x, a->x, s, a->count);                                                   \
    KERN(dst->y, a->y, s, a->count);                                                   \
    KERN(dst->z, a->z, s, a->count);                                                   \
    KERN(dst->w, a->w, s, a->count);                                                   \
}

WCN_V4X_SCALAR(WMATH_MULTIPLY_SCALAR(Vec4xN), wcn_soa_mul_s)
WCN_V4X_SCALAR(WMATH_DIV_SCALAR(Vec4xN), wcn_soa_div_s)

#undef WCN_V4X_SCALAR

void WMATH_ADD_SCALED(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                              const WMATH_TYPE(Vec4xN)* b, const float s) {
    assert(wcn_v4xn_ok2(dst, a, b));
    if (!wcn_v4xn_ok2(dst, a, b)) return;
    wcn_soa_add_scaled(dst->x, a->x, b->x, s, a->count);
    wcn_soa_add_scaled(dst->y, a->y, b->y, s, a->count);
    wcn_soa_add_scaled(dst->z, a->z, b->z, s, a->count);
    wcn_soa_add_scaled(dst->w, a->w, b->w, s, a->count);
}

void WMATH_CLAMP(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                         const float lo, const float hi) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    wcn_soa_clamp(dst->x, a->x, lo, hi, a->count);
    wcn_soa_clamp(dst->y, a->y, lo, hi, a->count);
    wcn_soa_clamp(dst->z, a->z, lo, hi, a->count);
    wcn_soa_clamp(dst->w, a->w, lo, hi, a->count);
}

/* ------------------------------------------------------------- unary / fused */

#define WCN_V4X_UNARY(NAME, KERN)                                                      \
void NAME(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {                      \
    assert(wcn_v4xn_ok1(dst, a));                                                      \
    if (!wcn_v4xn_ok1(dst, a)) return;                                                 \
    KERN(dst->x, a->x, a->count);                                                      \
    KERN(dst->y, a->y, a->count);                                                      \
    KERN(dst->z, a->z, a->count);                                                      \
    KERN(dst->w, a->w, a->count);                                                      \
}

WCN_V4X_UNARY(WMATH_NEGATE(Vec4xN), wcn_soa_neg)

#undef WCN_V4X_UNARY

void WMATH_INVERSE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    wcn_soa_inv(dst->x, a->x, a->count);
    wcn_soa_inv(dst->y, a->y, a->count);
    wcn_soa_inv(dst->z, a->z, a->count);
    wcn_soa_inv(dst->w, a->w, a->count);
}

void WMATH_NORMALIZE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    /* Fused: the reciprocal length must be held while all four planes are written.
     * In-place safe (dst == a) because each lane reads its four inputs before any store.
     * The threshold rule matches WMATH_NORMALIZE(Vec4): below eps the result is zero. */
    wcn_soa_normalize4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w, a->count);
}

void WMATH_LERP(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                        const WMATH_TYPE(Vec4xN)* b, const float t) {
    assert(wcn_v4xn_ok2(dst, a, b));
    if (!wcn_v4xn_ok2(dst, a, b)) return;
    wcn_soa_lerp(dst->x, a->x, b->x, t, a->count);
    wcn_soa_lerp(dst->y, a->y, b->y, t, a->count);
    wcn_soa_lerp(dst->z, a->z, b->z, t, a->count);
    wcn_soa_lerp(dst->w, a->w, b->w, t, a->count);
}

void WMATH_LERP_V(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                          const WMATH_TYPE(Vec4xN)* b, const WMATH_TYPE(Vec4xN)* t) {
    assert(wcn_v4xn_ok2(dst, a, b));
    assert(t && t->count == a->count);
    if (!wcn_v4xn_ok2(dst, a, b) || !t || t->count != a->count) return;
    wcn_soa_lerp_plane(dst->x, a->x, b->x, t->x, a->count);
    wcn_soa_lerp_plane(dst->y, a->y, b->y, t->y, a->count);
    wcn_soa_lerp_plane(dst->z, a->z, b->z, t->z, a->count);
    wcn_soa_lerp_plane(dst->w, a->w, b->w, t->w, a->count);
}

void WMATH_MIDPOINT(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                            const WMATH_TYPE(Vec4xN)* b) {
    assert(wcn_v4xn_ok2(dst, a, b));
    if (!wcn_v4xn_ok2(dst, a, b)) return;
    /* WMATH_MIDPOINT(Vec4) is LERP(a, b, 0.5f), i.e. a + (b-a)*0.5 -- not (a+b)*0.5,
     * which differs for any non-zero a. */
    wcn_soa_lerp(dst->x, a->x, b->x, 0.5f, a->count);
    wcn_soa_lerp(dst->y, a->y, b->y, 0.5f, a->count);
    wcn_soa_lerp(dst->z, a->z, b->z, 0.5f, a->count);
    wcn_soa_lerp(dst->w, a->w, b->w, 0.5f, a->count);
}

void WMATH_SET_LENGTH(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                              const float len) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    /* WMATH_SET_LENGTH(Vec4) is NORMALIZE then MULTIPLY_SCALAR(len), so it inherits the
     * normalize rule: a sub-epsilon input yields zero, which the scale then keeps at
     * zero rather than producing NaN. */
    wcn_soa_normalize4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w, a->count);
    wcn_soa_mul_s(dst->x, dst->x, len, a->count);
    wcn_soa_mul_s(dst->y, dst->y, len, a->count);
    wcn_soa_mul_s(dst->z, dst->z, len, a->count);
    wcn_soa_mul_s(dst->w, dst->w, len, a->count);
}

void WMATH_TRUNCATE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a,
                            const float max_length) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    wcn_soa_truncate4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w, max_length,
                      a->count);
}

/* ------------------------------------------------------- per-element scalars */

void WMATH_DOT(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a,
                       const WMATH_TYPE(Vec4xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_dot4(out, a->x, a->y, a->z, a->w, b->x, b->y, b->z, b->w, a->count);
}

void WMATH_LENGTH_SQ(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a) {
    if (!out || !a) return;
    wcn_soa_lensq4(out, a->x, a->y, a->z, a->w, a->count);
}

void WMATH_LENGTH(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a) {
    if (!out || !a) return;
    wcn_soa_len4(out, a->x, a->y, a->z, a->w, a->count);
}

void WMATH_DISTANCE_SQ(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a,
                               const WMATH_TYPE(Vec4xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_distsq4(out, a->x, a->y, a->z, a->w, b->x, b->y, b->z, b->w, a->count);
}

void WMATH_DISTANCE(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a,
                            const WMATH_TYPE(Vec4xN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_distsq4(out, a->x, a->y, a->z, a->w, b->x, b->y, b->z, b->w, a->count);
    wcn_soa_sqrt(out, out, a->count);
}

/* ------------------------------------------------------------ whole-container */

void WMATH_COPY(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    wcn_soa_copy(dst->x, a->x, a->count);
    wcn_soa_copy(dst->y, a->y, a->count);
    wcn_soa_copy(dst->z, a->z, a->count);
    wcn_soa_copy(dst->w, a->w, a->count);
}

void WMATH_ZERO(Vec4xN)(WMATH_TYPE(Vec4xN)* dst) {
    if (!dst || !dst->x) return;
    wcn_soa_fill(dst->x, 0.0f, dst->count);
    wcn_soa_fill(dst->y, 0.0f, dst->count);
    wcn_soa_fill(dst->z, 0.0f, dst->count);
    wcn_soa_fill(dst->w, 0.0f, dst->count);
}

void WMATH_IDENTITY(Vec4xN)(WMATH_TYPE(Vec4xN)* dst) {
    if (!dst || !dst->x) return;
    /* (1,1,1,1), the multiplicative unit. This is the value WMATH_IDENTITY(Vec4) uses
     * after LOGIC_BUGS L-04; (0,0,0,1) would make identity*v == (0,0,0,v.w) and is the
     * quaternion identity instead. */
    wcn_soa_fill(dst->x, 1.0f, dst->count);
    wcn_soa_fill(dst->y, 1.0f, dst->count);
    wcn_soa_fill(dst->z, 1.0f, dst->count);
    wcn_soa_fill(dst->w, 1.0f, dst->count);
}

bool WMATH_EQUALS(Vec4xN)(const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    for (size_t i = 0; i < a->count; i++) {
        if (a->x[i] != b->x[i] || a->y[i] != b->y[i] || a->z[i] != b->z[i] ||
            a->w[i] != b->w[i])
            return false;
    }
    return true;
}

bool WMATH_EQUALS_APPROXIMATELY(Vec4xN)(const WMATH_TYPE(Vec4xN)* a,
                                        const WMATH_TYPE(Vec4xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    const float ep = wcn_math_get_epsilon();
    for (size_t i = 0; i < a->count; i++) {
        if (!(fabsf(a->x[i] - b->x[i]) < ep)) return false;
        if (!(fabsf(a->y[i] - b->y[i]) < ep)) return false;
        if (!(fabsf(a->z[i] - b->z[i]) < ep)) return false;
        if (!(fabsf(a->w[i] - b->w[i]) < ep)) return false;
    }
    return true;
}
void WMATH_CREATE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_CREATE_TYPE(Vec4) c) {
    if (!dst || !dst->x) return;
    /* applies one set of component values to EVERY element -- the container lift of the
     * scalar operation, with the same shape as WMATH_ZERO / WMATH_IDENTITY. Note this is
     * not the per-element accessor, wcn_math_Vec4xN_set(dst, index, input). */
    wcn_soa_fill(dst->x, c.x, dst->count);
    wcn_soa_fill(dst->y, c.y, dst->count);
    wcn_soa_fill(dst->z, c.z, dst->count);
    wcn_soa_fill(dst->w, c.w, dst->count);
}

void WMATH_FLOOR(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_FLOOR's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_floor(dst->x, a->x, a->count);
    wcn_soa_floor(dst->y, a->y, a->count);
    wcn_soa_floor(dst->z, a->z, a->count);
    wcn_soa_floor(dst->w, a->w, a->count);
}

void WMATH_CEIL(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_CEIL's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_ceil(dst->x, a->x, a->count);
    wcn_soa_ceil(dst->y, a->y, a->count);
    wcn_soa_ceil(dst->z, a->z, a->count);
    wcn_soa_ceil(dst->w, a->w, a->count);
}

void WMATH_ROUND(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a) {
    assert(wcn_v4xn_ok1(dst, a));
    if (!wcn_v4xn_ok1(dst, a)) return;
    /* The rounding rule is WMATH_ROUND's own -- see the kernels in wcn_math_soa_kernels.h;
     * ROUND in particular is floor(x + 0.5), not C's roundf. */
    wcn_soa_round(dst->x, a->x, a->count);
    wcn_soa_round(dst->y, a->y, a->count);
    wcn_soa_round(dst->z, a->z, a->count);
    wcn_soa_round(dst->w, a->w, a->count);
}

void WMATH_CALL(Vec4xN, transform_mat4)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* v,
                                         const WMATH_TYPE(Mat4xN)* m) {
    if (!dst || !v || !m || dst->count != v->count || dst->count != m->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec4) vv;
        WMATH_TYPE(Mat4) mm;
        WMATH_TYPE(Vec4) r;
        vv.v[0] = v->x[i]; vv.v[1] = v->y[i]; vv.v[2] = v->z[i]; vv.v[3] = v->w[i];
        wcn_math_Mat4xN_get(&mm, m, i);
        WMATH_CALL(Vec4, transform_mat4)(&r, vv, mm);
        dst->x[i] = r.v[0]; dst->y[i] = r.v[1];
        dst->z[i] = r.v[2]; dst->w[i] = r.v[3];
    }
}
