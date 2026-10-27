/*
- [x] QuatxN
  - [x] malloc, destroy, free, get_*array, get_count, get, set, set_by_xyzw
  - [x] WMATH_ADD, WMATH_SUB
  - [x] WMATH_MULTIPLY (quaternion product), WMATH_MULTIPLY_SCALAR, WMATH_DIV_SCALAR
  - [x] WMATH_SCALE (alias of WMATH_MULTIPLY_SCALAR, as in the scalar implementation)
  - [x] WMATH_CONJUGATE, WMATH_NORMALIZE, WMATH_INVERSE
  - [x] WMATH_LERP
  - [x] WMATH_DOT, WMATH_LENGTH, WMATH_LENGTH_SQ
  - [x] WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY, WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
- [ ] QuatxN remaining: WMATH_ANGLE, WMATH_SLERP, WMATH_SQLERP, WMATH_ROTATE_X/Y/Z,
      WMATH_CREATE, WMATH_SET, and the axis-angle / euler / matrix conversions.
      These need transcendentals (acos, sqrt of a cosine) or produce a different type
      per element (a Vec3), so they are left rather than guessed at.
*/
#include <assert.h>
#include <math.h>

#include "WCN/WCN_MATH_SOA.h"
#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include "../common/wcn_math_soa_kernels.h"

static inline int wcn_qxn_ok2(const WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                              const WMATH_TYPE(QuatxN)* b) {
    return dst && a && b && dst->count == a->count && dst->count == b->count && dst->x &&
           dst->y && dst->z && dst->w && a->x && a->y && a->z && a->w && b->x && b->y &&
           b->z && b->w;
}
static inline int wcn_qxn_ok1(const WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    return dst && a && dst->count == a->count && dst->x && dst->y && dst->z && dst->w &&
           a->x && a->y && a->z && a->w;
}

/* ---------------------------------------------------------------- memory */

WMATH_TYPE(QuatxN)* wcn_math_QuatxN_malloc(const size_t count) {
    WMATH_TYPE(QuatxN)* q = (WMATH_TYPE(QuatxN)*)malloc(sizeof(WMATH_TYPE(QuatxN)));
    if (!q) return nullptr;
    q->count = count;
    /* wcn_aligned_alloc takes (bytes, alignment) -- reversed it allocates 16 bytes and
     * corrupts on the first store (LOGIC_BUGS.md). */
    q->x = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    q->y = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    q->z = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    q->w = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    return q;
}

void wcn_math_QuatxN_free(WMATH_TYPE(QuatxN)* ptr) {
    if (ptr == nullptr) return;
    /* Planes only, matching wcn_math_Vec4xN_free: a QuatxN may be caller-owned (stack
     * WCN_ALIGN16 arrays are the documented usage), so freeing the shell here would
     * corrupt the heap on that path. malloc'd shells go through _destroy. */
    if (ptr->x) { wcn_aligned_free(ptr->x); ptr->x = nullptr; }
    if (ptr->y) { wcn_aligned_free(ptr->y); ptr->y = nullptr; }
    if (ptr->z) { wcn_aligned_free(ptr->z); ptr->z = nullptr; }
    if (ptr->w) { wcn_aligned_free(ptr->w); ptr->w = nullptr; }
    ptr->count = 0;
}

void wcn_math_QuatxN_destroy(WMATH_TYPE(QuatxN)* ptr) {
    if (ptr == nullptr) return;
    wcn_math_QuatxN_free(ptr);
    free(ptr);
}

float* wcn_math_QuatxN_get_xarray(WMATH_TYPE(QuatxN)* ptr) { return ptr ? ptr->x : nullptr; }
float* wcn_math_QuatxN_get_yarray(WMATH_TYPE(QuatxN)* ptr) { return ptr ? ptr->y : nullptr; }
float* wcn_math_QuatxN_get_zarray(WMATH_TYPE(QuatxN)* ptr) { return ptr ? ptr->z : nullptr; }
float* wcn_math_QuatxN_get_warray(WMATH_TYPE(QuatxN)* ptr) { return ptr ? ptr->w : nullptr; }
size_t wcn_math_QuatxN_get_count(const WMATH_TYPE(QuatxN)* ptr) { return ptr ? ptr->count : 0; }

bool wcn_math_QuatxN_get(WMATH_TYPE(Quat)* out, const WMATH_TYPE(QuatxN)* src, size_t index) {
    if (!out || !src || index >= src->count) return false;
    out->v[0] = src->x[index];
    out->v[1] = src->y[index];
    out->v[2] = src->z[index];
    out->v[3] = src->w[index];
    return true;
}

bool wcn_math_QuatxN_set(WMATH_TYPE(QuatxN)* dst, const size_t index,
                         const WMATH_TYPE(Quat)* input) {
    if (!dst || !input || index >= dst->count) return false;
    dst->x[index] = input->v[0];
    dst->y[index] = input->v[1];
    dst->z[index] = input->v[2];
    dst->w[index] = input->v[3];
    return true;
}

bool wcn_math_QuatxN_set_by_xyzw(const WMATH_TYPE(QuatxN)* q, const size_t index,
                                 const float x, const float y, const float z, const float w) {
    if (!q || index >= q->count) return false;
    q->x[index] = x;
    q->y[index] = y;
    q->z[index] = z;
    q->w[index] = w;
    return true;
}

/* ------------------------------------------------------------- arithmetic */

void WMATH_ADD(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                       const WMATH_TYPE(QuatxN)* b) {
    assert(wcn_qxn_ok2(dst, a, b));
    if (!wcn_qxn_ok2(dst, a, b)) return;
    wcn_soa_add(dst->x, a->x, b->x, a->count);
    wcn_soa_add(dst->y, a->y, b->y, a->count);
    wcn_soa_add(dst->z, a->z, b->z, a->count);
    wcn_soa_add(dst->w, a->w, b->w, a->count);
}

void WMATH_SUB(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                       const WMATH_TYPE(QuatxN)* b) {
    assert(wcn_qxn_ok2(dst, a, b));
    if (!wcn_qxn_ok2(dst, a, b)) return;
    wcn_soa_sub(dst->x, a->x, b->x, a->count);
    wcn_soa_sub(dst->y, a->y, b->y, a->count);
    wcn_soa_sub(dst->z, a->z, b->z, a->count);
    wcn_soa_sub(dst->w, a->w, b->w, a->count);
}

void WMATH_MULTIPLY(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                            const WMATH_TYPE(QuatxN)* b) {
    assert(wcn_qxn_ok2(dst, a, b));
    if (!wcn_qxn_ok2(dst, a, b)) return;
    /* Fused: every output component depends on all eight inputs, so this cannot be
     * reached plane by plane.
     * NOTE: not safe when dst aliases a or b. The kernel reads all eight planes of the
     * current element before writing, so dst == a is in fact fine, but dst == b is not
     * worth relying on -- callers pass separate containers, as the scalar API does. */
    wcn_soa_quat_mul4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w, b->x, b->y,
                      b->z, b->w, a->count);
}

void WMATH_MULTIPLY_SCALAR(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                                   const float s) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    wcn_soa_mul_s(dst->x, a->x, s, a->count);
    wcn_soa_mul_s(dst->y, a->y, s, a->count);
    wcn_soa_mul_s(dst->z, a->z, s, a->count);
    wcn_soa_mul_s(dst->w, a->w, s, a->count);
}

void WMATH_DIV_SCALAR(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                              const float s) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    wcn_soa_div_s(dst->x, a->x, s, a->count);
    wcn_soa_div_s(dst->y, a->y, s, a->count);
    wcn_soa_div_s(dst->z, a->z, s, a->count);
    wcn_soa_div_s(dst->w, a->w, s, a->count);
}

void WMATH_SCALE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                         const float s) {
    /* WMATH_SCALE(Quat) is literally WMATH_CALL(Quat, multiply_scalar), so this forwards
     * rather than duplicating the four plane calls. */
    WMATH_MULTIPLY_SCALAR(QuatxN)(dst, a, s);
}

void WMATH_NORMALIZE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    /* Below eps the result is the quaternion IDENTITY (0,0,0,1) -- not zeros. See the
     * kernel's note: a zero quaternion cannot be used as a basis for later products, and
     * WMATH_NORMALIZE(Quat) substitutes the identity. */
    wcn_soa_quat_normalize4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w,
                            a->count);
}

void WMATH_CALL(QuatxN, conjugate)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    /* x, y, z negated, w untouched -- matching WMATH_CALL(Quat, conjugate), which XORs a
     * sign mask of (0, -0, -0, -0). With the exact sign-flip negate this agrees bit for
     * bit, signed zeros included. */
    wcn_soa_neg(dst->x, a->x, a->count);
    wcn_soa_neg(dst->y, a->y, a->count);
    wcn_soa_neg(dst->z, a->z, a->count);
    wcn_soa_copy(dst->w, a->w, a->count);
}

void WMATH_INVERSE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    /* conjugate / dot(q,q), with a zero norm yielding zeros rather than dividing. */
    wcn_soa_quat_inverse4(dst->x, dst->y, dst->z, dst->w, a->x, a->y, a->z, a->w, a->count);
}

void WMATH_LERP(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                        const WMATH_TYPE(QuatxN)* b, const float t) {
    assert(wcn_qxn_ok2(dst, a, b));
    if (!wcn_qxn_ok2(dst, a, b)) return;
    /* a + (b-a)*t, the same shape as WMATH_LERP(Quat). */
    wcn_soa_lerp(dst->x, a->x, b->x, t, a->count);
    wcn_soa_lerp(dst->y, a->y, b->y, t, a->count);
    wcn_soa_lerp(dst->z, a->z, b->z, t, a->count);
    wcn_soa_lerp(dst->w, a->w, b->w, t, a->count);
}

/* ------------------------------------------------------- per-element scalars */

void WMATH_DOT(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b) {
    if (!out || !a || !b || a->count != b->count) return;
    wcn_soa_dot4(out, a->x, a->y, a->z, a->w, b->x, b->y, b->z, b->w, a->count);
}

void WMATH_LENGTH_SQ(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a) {
    if (!out || !a) return;
    wcn_soa_lensq4(out, a->x, a->y, a->z, a->w, a->count);
}

void WMATH_LENGTH(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a) {
    if (!out || !a) return;
    wcn_soa_len4(out, a->x, a->y, a->z, a->w, a->count);
}

/* ------------------------------------------------------------ whole-container */

void WMATH_COPY(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    assert(wcn_qxn_ok1(dst, a));
    if (!wcn_qxn_ok1(dst, a)) return;
    wcn_soa_copy(dst->x, a->x, a->count);
    wcn_soa_copy(dst->y, a->y, a->count);
    wcn_soa_copy(dst->z, a->z, a->count);
    wcn_soa_copy(dst->w, a->w, a->count);
}

void WMATH_ZERO(QuatxN)(WMATH_TYPE(QuatxN)* dst) {
    if (!dst || !dst->x) return;
    wcn_soa_fill(dst->x, 0.0f, dst->count);
    wcn_soa_fill(dst->y, 0.0f, dst->count);
    wcn_soa_fill(dst->z, 0.0f, dst->count);
    wcn_soa_fill(dst->w, 0.0f, dst->count);
}

void WMATH_IDENTITY(QuatxN)(WMATH_TYPE(QuatxN)* dst) {
    if (!dst || !dst->x) return;
    /* (0,0,0,1): the quaternion identity, which is NOT the Vec4 identity (1,1,1,1).
     * The two differ deliberately -- see LOGIC_BUGS.md L-04. */
    wcn_soa_fill(dst->x, 0.0f, dst->count);
    wcn_soa_fill(dst->y, 0.0f, dst->count);
    wcn_soa_fill(dst->z, 0.0f, dst->count);
    wcn_soa_fill(dst->w, 1.0f, dst->count);
}

bool WMATH_EQUALS(QuatxN)(const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b) {
    if (!a || !b || a->count != b->count) return false;
    for (size_t i = 0; i < a->count; i++) {
        if (a->x[i] != b->x[i] || a->y[i] != b->y[i] || a->z[i] != b->z[i] ||
            a->w[i] != b->w[i])
            return false;
    }
    return true;
}

bool WMATH_EQUALS_APPROXIMATELY(QuatxN)(const WMATH_TYPE(QuatxN)* a,
                                        const WMATH_TYPE(QuatxN)* b) {
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
void WMATH_CREATE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_CREATE_TYPE(Quat) c) {
    if (!dst || !dst->x) return;
    /* applies one set of component values to EVERY element -- the container lift of the
     * scalar operation, with the same shape as WMATH_ZERO / WMATH_IDENTITY. Note this is
     * not the per-element accessor, wcn_math_QuatxN_set(dst, index, input). */
    wcn_soa_fill(dst->x, c.x, dst->count);
    wcn_soa_fill(dst->y, c.y, dst->count);
    wcn_soa_fill(dst->z, c.z, dst->count);
    wcn_soa_fill(dst->w, c.w, dst->count);
}

/* ---------------------------------------------------- remaining operators */
/*
 * All per-element scalar: every one of these either calls trig or delegates to a routine
 * that does, and no backend here has a vector sin/cos (see the rotation-family note). Each
 * value parameter is one plane, i.e. one value per element.
 */

static inline int quatxn_ok2(const WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                             const WMATH_TYPE(QuatxN)* b) {
    return dst && a && b && dst->count == a->count && dst->count == b->count && dst->x &&
           dst->y && dst->z && dst->w && a->x && b->x;
}
static inline int quatxn_ok1(const WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a) {
    return dst && a && dst->count == a->count && dst->x && dst->y && dst->z && dst->w &&
           a->x && a->y && a->z && a->w;
}

/* one Get/Set round trip per element, shared by the element-wise entry points below */
static inline void qxn_get(WMATH_TYPE(Quat)* q, const WMATH_TYPE(QuatxN)* s, size_t i) {
    q->v[0] = s->x[i];
    q->v[1] = s->y[i];
    q->v[2] = s->z[i];
    q->v[3] = s->w[i];
}
static inline void qxn_put(WMATH_TYPE(QuatxN)* d, size_t i, const WMATH_TYPE(Quat)* q) {
    d->x[i] = q->v[0];
    d->y[i] = q->v[1];
    d->z[i] = q->v[2];
    d->w[i] = q->v[3];
}

void WMATH_ANGLE(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a,
                           const WMATH_TYPE(QuatxN)* b) {
    /* Takes float* out and returns void, matching the convention already used by
     * DOT/LENGTH/LENGTH_SQ/DISTANCE on the xN side: the scalar returns one float, the
     * container produces N of them, so the result is a plane. Returning some aggregate here
     * would be a number with no meaning. */
    if (!out || !a || !b || a->count != b->count) return;
    for (size_t i = 0; i < a->count; i++) {
        WMATH_TYPE(Quat) qa;
        WMATH_TYPE(Quat) qb;
        qxn_get(&qa, a, i);
        qxn_get(&qb, b, i);
        out[i] = WMATH_ANGLE(Quat)(qa, qb);
    }
}

void WMATH_CALL(QuatxN, from_mat3)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Mat3xN)* m) {
    if (!dst || !m || dst->count != m->count) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat3) mm;
        WMATH_TYPE(Quat) r;
        wcn_math_Mat3xN_get(&mm, m, i);
        WMATH_CALL(Quat, from_mat3)(&r, mm);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, from_mat4)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Mat4xN)* m) {
    if (!dst || !m || dst->count != m->count) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat4) mm;
        WMATH_TYPE(Quat) r;
        wcn_math_Mat4xN_get(&mm, m, i);
        WMATH_CALL(Quat, from_mat4)(&r, mm);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, rotation_to)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Vec3xN)* a,
                                      const WMATH_TYPE(Vec3xN)* b) {
    if (!dst || !a || !b || dst->count != a->count || dst->count != b->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec3) va;
        WMATH_TYPE(Vec3) vb;
        WMATH_TYPE(Quat) r;
        va.v[0] = a->x[i]; va.v[1] = a->y[i]; va.v[2] = a->z[i];
        vb.v[0] = b->x[i]; vb.v[1] = b->y[i]; vb.v[2] = b->z[i];
        WMATH_CALL(Quat, rotation_to)(&r, va, vb);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, slerp)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                                const WMATH_TYPE(QuatxN)* b, const float* t) {
    assert(quatxn_ok2(dst, a, b));
    if (!quatxn_ok2(dst, a, b) || !t) return;
    for (size_t i = 0; i < a->count; i++) {
        WMATH_TYPE(Quat) qa;
        WMATH_TYPE(Quat) qb;
        WMATH_TYPE(Quat) r;
        qxn_get(&qa, a, i);
        qxn_get(&qb, b, i);
        WMATH_CALL(Quat, slerp)(&r, qa, qb, t[i]);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, sqlerp)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a,
                                 const WMATH_TYPE(QuatxN)* b, const WMATH_TYPE(QuatxN)* c,
                                 const WMATH_TYPE(QuatxN)* d, const float* t) {
    if (!dst || !a || !b || !c || !d || !t) return;
    if (a->count != dst->count || b->count != dst->count || c->count != dst->count ||
        d->count != dst->count) {
        return;
    }
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Quat) qa, qb, qc, qd, r;
        qxn_get(&qa, a, i);
        qxn_get(&qb, b, i);
        qxn_get(&qc, c, i);
        qxn_get(&qd, d, i);
        WMATH_CALL(Quat, sqlerp)(&r, qa, qb, qc, qd, t[i]);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, from_axis_angle)(WMATH_TYPE(QuatxN)* dst,
                                          const WMATH_TYPE(Vec3xN)* axis,
                                          const float* radians) {
    if (!dst || !axis || !radians || axis->count != dst->count) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Vec3) ax;
        WMATH_TYPE(Quat) r;
        ax.v[0] = axis->x[i];
        ax.v[1] = axis->y[i];
        ax.v[2] = axis->z[i];
        WMATH_CALL(Quat, from_axis_angle)(&r, ax, radians[i]);
        qxn_put(dst, i, &r);
    }
}

void WMATH_CALL(QuatxN, from_euler)(WMATH_TYPE(QuatxN)* dst, const float* x, const float* y,
                                     const float* z, enum WCN_Math_RotationOrder order) {
    if (!dst || !x || !y || !z) return;
    /* `order` is a plain enum for the whole container -- the same reasoning as `int axis`:
     * which order to compose in is a structural choice about the call, not per-element data. */
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Quat) r;
        WMATH_CALL(Quat, from_euler)(&r, x[i], y[i], z[i], order);
        qxn_put(dst, i, &r);
    }
}

/* Written out rather than generated by a SUFFIX macro: the audit tool reads definitions
 * textually, and a name assembled by token pasting is invisible to it -- which would
 * make these three look unimplemented in the very report used to judge completeness.
 */

void WMATH_ROTATE_X(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q,
                                const float* radians) {
    assert(quatxn_ok1(dst, q));
    if (!quatxn_ok1(dst, q) || !radians) return;
    for (size_t i = 0; i < q->count; i++) {
        WMATH_TYPE(Quat) qq;
        WMATH_TYPE(Quat) r;
        qxn_get(&qq, q, i);
        WMATH_ROTATE_X(Quat)(&r, qq, radians[i]);
        qxn_put(dst, i, &r);
    }
}

void WMATH_ROTATE_Y(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q,
                                const float* radians) {
    assert(quatxn_ok1(dst, q));
    if (!quatxn_ok1(dst, q) || !radians) return;
    for (size_t i = 0; i < q->count; i++) {
        WMATH_TYPE(Quat) qq;
        WMATH_TYPE(Quat) r;
        qxn_get(&qq, q, i);
        WMATH_ROTATE_Y(Quat)(&r, qq, radians[i]);
        qxn_put(dst, i, &r);
    }
}

void WMATH_ROTATE_Z(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q,
                                const float* radians) {
    assert(quatxn_ok1(dst, q));
    if (!quatxn_ok1(dst, q) || !radians) return;
    for (size_t i = 0; i < q->count; i++) {
        WMATH_TYPE(Quat) qq;
        WMATH_TYPE(Quat) r;
        qxn_get(&qq, q, i);
        WMATH_ROTATE_Z(Quat)(&r, qq, radians[i]);
        qxn_put(dst, i, &r);
    }
}
