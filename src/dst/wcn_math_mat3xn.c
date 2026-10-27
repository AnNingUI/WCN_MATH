/*
- [x] Mat3xN
  - [x] malloc, destroy, free, get_element_array, get_count, get, set
  - [x] WMATH_ADD, WMATH_SUB, WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR
  - [x] WMATH_NEGATE, WMATH_TRANSPOSE, WMATH_INVERSE, WMATH_DETERMINANT
  - [x] WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY
  - [x] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
- [ ] Mat3xN remaining: WMATH_SET, WMATH_CREATE, WMATH_SCALE, WMATH_ROTATE*,
      WMATH_ROTATION*, WMATH_TRANSLATION, WMATH_GET_TRANSLATION, WMATH_SET_TRANSLATION.
      Those are parameterised constructors (an angle, an axis, a vector) rather than
      per-element operators on an existing container, so each needs its own treatment.
*/
#include <assert.h>
#include <math.h>

#include "WCN/WCN_MATH_SOA.h"
#include "WCN/WCN_MATH_DST.h"
#include "../common/wcn_math_internal.h"
#include "../common/wcn_math_soa_kernels.h"

/*
 * Plane layout (documented here because it is the one thing most likely to be got wrong):
 * Mat3xN has NINE planes in a packed layout, plane p == Mat3 element (row*3 + col).
 *
 *     plane:  0  1  2  3  4  5  6  7  8
 *     Mat3:  m00 m01 m02 m10 m11 m12 m20 m21 m22     (via Mat3's m[row*4+col])
 *
 * Mat3's in-memory form pads each row to four floats, so Mat3xN's plane indices are NOT
 * Mat3's m[] indices. m3_p2m() below is the mapping, used by get/set and mirrored by the
 * kernels in wcn_math_soa_kernels.h.
 */
static inline size_t m3_p2m(size_t plane) { return (plane / 3) * 4 + (plane % 3); }

static inline int m3xn_ok2(const WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a,
                           const WMATH_TYPE(Mat3xN)* b) {
    if (!dst || !a || !b) return 0;
    if (dst->count != a->count || dst->count != b->count) return 0;
    for (int p = 0; p < 9; p++)
        if (!dst->m[p] || !a->m[p] || !b->m[p]) return 0;
    return 1;
}
static inline int m3xn_ok1(const WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a) {
    if (!dst || !a || dst->count != a->count) return 0;
    for (int p = 0; p < 9; p++)
        if (!dst->m[p] || !a->m[p]) return 0;
    return 1;
}

/* ---------------------------------------------------------------- memory */

WMATH_TYPE(Mat3xN)* wcn_math_Mat3xN_malloc(const size_t count) {
    WMATH_TYPE(Mat3xN)* v = (WMATH_TYPE(Mat3xN)*)malloc(sizeof(WMATH_TYPE(Mat3xN)));
    if (!v) return nullptr;
    v->count = count;
    for (int p = 0; p < 9; p++) {
        /* wcn_aligned_alloc takes (bytes, alignment); reversed it allocates 16 bytes and
         * corrupts on the first store (LOGIC_BUGS.md). */
        v->m[p] = (float*)wcn_aligned_alloc(count * sizeof(float), 16);
    }
    return v;
}

void wcn_math_Mat3xN_free(WMATH_TYPE(Mat3xN)* ptr) {
    if (ptr == nullptr) return;
    /* Planes only: a Mat3xN may be caller-owned (the Flat conversions and the documented
     * usage both hand in caller-filled structs), so freeing the shell here would corrupt
     * the heap on that path. Shells from _malloc go through _destroy. */
    for (int p = 0; p < 9; p++) {
        if (ptr->m[p]) { wcn_aligned_free(ptr->m[p]); ptr->m[p] = nullptr; }
    }
    ptr->count = 0;
}

void wcn_math_Mat3xN_destroy(WMATH_TYPE(Mat3xN)* ptr) {
    if (ptr == nullptr) return;
    wcn_math_Mat3xN_free(ptr);
    free(ptr);
}

float* wcn_math_Mat3xN_get_element_array(WMATH_TYPE(Mat3xN)* ptr, size_t plane) {
    if (!ptr || plane >= 9) return nullptr;
    return ptr->m[plane];
}

size_t wcn_math_Mat3xN_get_count(const WMATH_TYPE(Mat3xN)* ptr) {
    return ptr ? ptr->count : 0;
}

bool wcn_math_Mat3xN_get(WMATH_TYPE(Mat3)* out, const WMATH_TYPE(Mat3xN)* src, size_t index) {
    if (!out || !src || index >= src->count) return false;
    /* Mat3 has twelve slots with a padding lane per row; Mat3xN has only the nine real
     * ones, so the padding is filled with zero here exactly as WMATH_IDENTITY and the
     * scalar setters leave it. */
    for (int i = 0; i < 12; i++) out->m[i] = 0.0f;
    for (size_t p = 0; p < 9; p++) out->m[m3_p2m(p)] = src->m[p][index];
    return true;
}

bool wcn_math_Mat3xN_set(WMATH_TYPE(Mat3xN)* dst, const size_t index,
                         const WMATH_TYPE(Mat3)* input) {
    if (!dst || !input || index >= dst->count) return false;
    for (size_t p = 0; p < 9; p++) dst->m[p][index] = input->m[m3_p2m(p)];
    return true;
}

/* ------------------------------------------------------------- element-wise */

#define M3XN_BINOP(NAME, KERN)                                                         \
void NAME(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a,                        \
          const WMATH_TYPE(Mat3xN)* b) {                                               \
    assert(m3xn_ok2(dst, a, b));                                                       \
    if (!m3xn_ok2(dst, a, b)) return;                                                  \
    for (int p = 0; p < 9; p++) KERN(dst->m[p], a->m[p], b->m[p], a->count);           \
}

M3XN_BINOP(WMATH_ADD(Mat3xN), wcn_soa_add)
M3XN_BINOP(WMATH_SUB(Mat3xN), wcn_soa_sub)

#undef M3XN_BINOP

#define M3XN_UNARY(NAME, KERN)                                                         \
void NAME(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a) {                      \
    assert(m3xn_ok1(dst, a));                                                          \
    if (!m3xn_ok1(dst, a)) return;                                                     \
    for (int p = 0; p < 9; p++) KERN(dst->m[p], a->m[p], a->count);                    \
}

M3XN_UNARY(WMATH_NEGATE(Mat3xN), wcn_soa_neg)

#undef M3XN_UNARY

void WMATH_MULTIPLY_SCALAR(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a,
                                   const float s) {
    assert(m3xn_ok1(dst, a));
    if (!m3xn_ok1(dst, a)) return;
    for (int p = 0; p < 9; p++) wcn_soa_mul_s(dst->m[p], a->m[p], s, a->count);
}

void WMATH_MULTIPLY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a,
                            const WMATH_TYPE(Mat3xN)* b) {
    assert(m3xn_ok2(dst, a, b));
    if (!m3xn_ok2(dst, a, b)) return;
    /* Fused: each of the nine outputs is a sum of three cross-plane products, so a
     * plane-by-plane composition would need a temporary plane per term. */
    wcn_soa_m3_mul9(dst->m, WCN_SOA_RO(a->m), WCN_SOA_RO(b->m), a->count);
}

void WMATH_TRANSPOSE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a) {
    assert(m3xn_ok1(dst, a));
    /* dst must not alias a: the three plane swaps below would read values they have
     * already overwritten. The scalar WMATH_TRANSPOSE(Mat3) takes its source by value and
     * is therefore safe in place; this pointer-based form is not, so the restriction is
     * checked rather than papered over with a full scratch plane. */
    assert(dst != a);
    if (!m3xn_ok1(dst, a) || dst == a) return;
    /* Transposing swaps the off-diagonal planes and leaves the diagonal ones alone:
     *   plane 1 (m01) <-> plane 3 (m10)
     *   plane 2 (m02) <-> plane 6 (m20)
     *   plane 5 (m12) <-> plane 7 (m21)
     * The nine plane pointers stay where they are; only the data moves. */
    wcn_soa_copy(dst->m[0], a->m[0], a->count);
    wcn_soa_copy(dst->m[1], a->m[3], a->count);
    wcn_soa_copy(dst->m[2], a->m[6], a->count);
    wcn_soa_copy(dst->m[3], a->m[1], a->count);
    wcn_soa_copy(dst->m[4], a->m[4], a->count);
    wcn_soa_copy(dst->m[5], a->m[7], a->count);
    wcn_soa_copy(dst->m[6], a->m[2], a->count);
    wcn_soa_copy(dst->m[7], a->m[5], a->count);
    wcn_soa_copy(dst->m[8], a->m[8], a->count);
}

void WMATH_DETERMINANT(Mat3xN)(float* out, const WMATH_TYPE(Mat3xN)* a) {
    if (!out || !a || !a->m[0]) return;
    wcn_soa_m3_det(out, WCN_SOA_RO(a->m), a->count);
}

void WMATH_INVERSE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a) {
    assert(m3xn_ok1(dst, a));
    if (!m3xn_ok1(dst, a)) return;
    /* Keeps the scalar guard: |det| < 1e-12 gives the IDENTITY matrix, not inf/nan. */
    wcn_soa_m3_inv9(dst->m, WCN_SOA_RO(a->m), a->count);
}

/* ------------------------------------------------------------- whole-container */

void WMATH_COPY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a) {
    assert(m3xn_ok1(dst, a));
    if (!m3xn_ok1(dst, a)) return;
    for (int p = 0; p < 9; p++) wcn_soa_copy(dst->m[p], a->m[p], a->count);
}

void WMATH_ZERO(Mat3xN)(WMATH_TYPE(Mat3xN)* dst) {
    if (!dst || !dst->m[0]) return;
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
}

void WMATH_IDENTITY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst) {
    if (!dst || !dst->m[0]) return;
    /* WMATH_IDENTITY(Mat3) sets m[0], m[5], m[10] to 1 and the rest to 0, which in the
     * packed plane layout is planes 0, 4 and 8. */
    for (int p = 0; p < 9; p++) {
        wcn_soa_fill(dst->m[p], (p == 0 || p == 4 || p == 8) ? 1.0f : 0.0f, dst->count);
    }
}

bool WMATH_EQUALS(Mat3xN)(const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    for (size_t i = 0; i < a->count; i++) {
        for (int p = 0; p < 9; p++) {
            if (a->m[p][i] != b->m[p][i]) return false;
        }
    }
    return true;
}

bool WMATH_EQUALS_APPROXIMATELY(Mat3xN)(const WMATH_TYPE(Mat3xN)* a,
                                        const WMATH_TYPE(Mat3xN)* b) {
    if (!a || !b || a->count != b->count) return false;
    const float ep = wcn_math_get_epsilon();
    for (size_t i = 0; i < a->count; i++) {
        for (int p = 0; p < 9; p++) {
            if (!(fabsf(a->m[p][i] - b->m[p][i]) < ep)) return false;
        }
    }
    return true;
}
void WMATH_CREATE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_CREATE_TYPE(Mat3) c) {
    if (!dst || !dst->m[0]) return;
    /* fills every element from the named CREATE_TYPE fields; the plane/field mapping is
     * plane p = row*3 + col, i.e. m_<row><col>. */
    wcn_soa_fill(dst->m[0], c.m_00, dst->count);
    wcn_soa_fill(dst->m[1], c.m_01, dst->count);
    wcn_soa_fill(dst->m[2], c.m_02, dst->count);
    wcn_soa_fill(dst->m[3], c.m_10, dst->count);
    wcn_soa_fill(dst->m[4], c.m_11, dst->count);
    wcn_soa_fill(dst->m[5], c.m_12, dst->count);
    wcn_soa_fill(dst->m[6], c.m_20, dst->count);
    wcn_soa_fill(dst->m[7], c.m_21, dst->count);
    wcn_soa_fill(dst->m[8], c.m_22, dst->count);
}

/* ------------------------------------------------------- translation family */

void WMATH_GET_TRANSLATION(Mat3xN)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m) {
    if (!out || !m || out->count != m->count) return;
    /* WMATH_GET_TRANSLATION(Mat3) reads m[8] and m[9], i.e. m20 and m21 -- the first two
     * slots of the third row. Mat3xN is packed (plane p = row*3 + col), so those are planes
     * 6 and 7. */
    wcn_soa_copy(out->x, m->m[6], m->count);
    wcn_soa_copy(out->y, m->m[7], m->count);
}

void WMATH_SET_TRANSLATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                    const WMATH_TYPE(Vec2) v) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m)) return;
    /* The scalar form copies the matrix, writes v into m[8], m[9] and FORCES m[10] to 1 --
     * the affine convention. It never touches the padding lanes m[3]/m[7]/m[11], which have
     * no plane here. One Vec2 applies to every element, as with WMATH_CREATE. */
    for (int p = 0; p < 9; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    wcn_soa_fill(dst->m[6], v.v[0], m->count);
    wcn_soa_fill(dst->m[7], v.v[1], m->count);
    wcn_soa_fill(dst->m[8], 1.0f, m->count);
}

/* ------------------------------------------------- type conversion */

void WMATH_CALL(Mat3xN, from_mat4)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat4xN)* a) {
    if (!dst || !a || dst->count != a->count) return;
    /* Upper-left 3x3 of the Mat4: planes 0,1,2 / 4,5,6 / 8,9,10. The translation row (12..15)
     * and column (3,7,11) are excluded, matching WMATH_CALL(Mat3, from_mat4). */
    static const int SRC[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    for (int p = 0; p < 9; p++) wcn_soa_copy(dst->m[p], a->m[SRC[p]], a->count);
}

/* --------------------------------------------------------- rotation family */
/*
 * Per-element scalar, for the same reason as Mat4xN's: no ISA in this library has a vector
 * sin/cos (SSE/AVX outside SVML, NEON, WASM SIMD, RVV base, LSX all lack them), so the trig
 * cannot be widened. The angle is one value per element.
 *
 * This family was reverted in round 13 because it produced identity matrices. The cause was
 * not here: this file did not include WCN_MATH_DST.h, so `WMATH_ROTATION(Mat3)` below was an
 * implicit declaration, the float argument was promoted to double, and the callee read a
 * garbage angle. See the note at the top of the file.
 */
void WMATH_ROTATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat3) r;
        WMATH_ROTATION(Mat3)(&r, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATION_X(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat3) r;
        WMATH_ROTATION_X(Mat3)(&r, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATION_Y(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat3) r;
        WMATH_ROTATION_Y(Mat3)(&r, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATION_Z(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians) {
    if (!dst || !dst->m[0] || !radians) return;
    for (size_t i = 0; i < dst->count; i++) {
        WMATH_TYPE(Mat3) r;
        WMATH_ROTATION_Z(Mat3)(&r, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                         const float* radians) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat3) mi;
        WMATH_TYPE(Mat3) r;
        wcn_math_Mat3xN_get(&mi, m, i);
        WMATH_ROTATE(Mat3)(&r, mi, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATE_X(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                         const float* radians) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat3) mi;
        WMATH_TYPE(Mat3) r;
        wcn_math_Mat3xN_get(&mi, m, i);
        WMATH_ROTATE_X(Mat3)(&r, mi, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATE_Y(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                         const float* radians) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat3) mi;
        WMATH_TYPE(Mat3) r;
        wcn_math_Mat3xN_get(&mi, m, i);
        WMATH_ROTATE_Y(Mat3)(&r, mi, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
void WMATH_ROTATE_Z(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                         const float* radians) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !radians) return;
    for (size_t i = 0; i < m->count; i++) {
        WMATH_TYPE(Mat3) mi;
        WMATH_TYPE(Mat3) r;
        wcn_math_Mat3xN_get(&mi, m, i);
        WMATH_ROTATE_Z(Mat3)(&r, mi, radians[i]);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
/* --------------------------------------------------------- scaling family */

void WMATH_SCALE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                          const WMATH_TYPE(Vec2) v) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m)) return;
    /* WMATH_SCALE(Mat3) scales rows 0 and 1 by v.x and v.y and copies row 2 unchanged -- the
     * translation survives, exactly as in the Mat4 form. Packed planes: 0-2 row 0,
     * 3-5 row 1, 6-8 row 2. */
    const float s[2] = {v.v[0], v.v[1]};
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            wcn_soa_mul_s(dst->m[r * 3 + c], m->m[r * 3 + c], s[r], m->count);
        }
    }
    for (int c = 0; c < 3; c++) wcn_soa_copy(dst->m[6 + c], m->m[6 + c], m->count);
}

void WMATH_CALL(Mat3xN, scale3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                  const WMATH_TYPE(Vec3) v) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m)) return;
    /* Three-dimensional scaling: all three rows scaled by v.x, v.y, v.z. Unlike WMATH_SCALE
     * there is no untouched row. */
    const float s[3] = {v.v[0], v.v[1], v.v[2]};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            wcn_soa_mul_s(dst->m[r * 3 + c], m->m[r * 3 + c], s[r], m->count);
        }
    }
}

void WMATH_CALL(Mat3xN, scaling)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec2) v) {
    if (!dst || !dst->m[0]) return;
    /* Builds a scaling matrix from a Vec2; the bottom-right convention is verified against
     * WMATH_CALL(Mat3, scaling) by the test rather than assumed. */
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_fill(dst->m[0], v.v[0], dst->count);
    wcn_soa_fill(dst->m[4], v.v[1], dst->count);
    wcn_soa_fill(dst->m[8], 1.0f, dst->count);
}

/* -------------------------------------------------- scaling / translation */

void WMATH_CALL(Mat3xN, uniform_scaling)(WMATH_TYPE(Mat3xN)* dst, const float* s) {
    if (!dst || !dst->m[0] || !s) return;
    /* diagonal s, but m22 is forced to 1 -- unlike uniform_scaling_3D below. */
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_copy(dst->m[0], s, dst->count);
    wcn_soa_copy(dst->m[4], s, dst->count);
    wcn_soa_fill(dst->m[8], 1.0f, dst->count);
}

void WMATH_CALL(Mat3xN, uniform_scaling_3D)(WMATH_TYPE(Mat3xN)* dst, const float* s) {
    if (!dst || !dst->m[0] || !s) return;
    /* the same diagonal, but m22 takes s as well -- this one element is the entire
     * difference from uniform_scaling. */
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_copy(dst->m[0], s, dst->count);
    wcn_soa_copy(dst->m[4], s, dst->count);
    wcn_soa_copy(dst->m[8], s, dst->count);
}

void WMATH_CALL(Mat3xN, uniform_scale)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                        const float* s) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !s) return;
    /* rows 0 and 1 scaled; row 2 (the translation) copied. The scale is a PLANE here, so
     * this is the element-wise multiply -- my first version passed s[0], which broadcast one
     * element's scale across the whole container and the test caught it. */
    for (int p = 0; p < 6; p++) wcn_soa_mul(dst->m[p], m->m[p], s, m->count);
    for (int p = 6; p < 9; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
}

void WMATH_CALL(Mat3xN, uniform_scale_3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                           const float* s) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !s) return;
    /* all three rows scaled -- the difference from uniform_scale. Per-element scale, as
     * above. */
    for (int p = 0; p < 9; p++) wcn_soa_mul(dst->m[p], m->m[p], s, m->count);
}

void WMATH_CALL(Mat3xN, scaling3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec3xN)* v) {
    if (!dst || !dst->m[0] || !v || v->count != dst->count) return;
    /* diagonal (vx, vy, vz); nothing forced to 1 -- contrast WMATH_CALL(Mat3xN, scaling),
     * whose bottom-right element is 1. */
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_copy(dst->m[0], v->x, dst->count);
    wcn_soa_copy(dst->m[4], v->y, dst->count);
    wcn_soa_copy(dst->m[8], v->z, dst->count);
}

void WMATH_TRANSLATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec2xN)* v) {
    if (!dst || !dst->m[0] || !v || v->count != dst->count) return;
    /* identity with the third row's first two slots set and m22 = 1. */
    for (int p = 0; p < 9; p++) wcn_soa_fill(dst->m[p], 0.0f, dst->count);
    wcn_soa_fill(dst->m[0], 1.0f, dst->count);
    wcn_soa_fill(dst->m[4], 1.0f, dst->count);
    wcn_soa_copy(dst->m[6], v->x, dst->count);
    wcn_soa_copy(dst->m[7], v->y, dst->count);
    wcn_soa_fill(dst->m[8], 1.0f, dst->count);
}

void WMATH_CALL(Mat3xN, translate)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                    const WMATH_TYPE(Vec2xN)* v) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !v || v->count != m->count) return;
    for (int p = 0; p < 6; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    wcn_soa_m3_translate(dst->m, WCN_SOA_RO(m->m), v->x, v->y, m->count);
}

/* ------------------------------------------------------------ accessors */

void WMATH_CALL(Mat3xN, get_axis)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m,
                                   int axis) {
    if (!out || !m || out->count != m->count) return;
    if (axis < 0 || axis > 2) return;
    /* row `axis`, first two columns. The int selector applies to the whole container: which
     * row to read is a structural choice, not data varying between the N items. */
    const int base = axis * 3;
    wcn_soa_copy(out->x, m->m[base + 0], m->count);
    wcn_soa_copy(out->y, m->m[base + 1], m->count);
}

void WMATH_CALL(Mat3xN, set_axis)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m,
                                   const WMATH_TYPE(Vec2xN)* v, int axis) {
    assert(m3xn_ok1(dst, m));
    if (!m3xn_ok1(dst, m) || !v || v->count != m->count) return;
    if (axis < 0 || axis > 2) return;
    for (int p = 0; p < 9; p++) wcn_soa_copy(dst->m[p], m->m[p], m->count);
    const int base = axis * 3;
    wcn_soa_copy(dst->m[base + 0], v->x, m->count);
    wcn_soa_copy(dst->m[base + 1], v->y, m->count);
}

void WMATH_CALL(Mat3xN, get_scaling)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m) {
    if (!out || !m || out->count != m->count) return;
    /* two components of each of rows 0 and 1 -- the 2D form. */
    wcn_soa_len2(out->x, m->m[0], m->m[1], m->count);
    wcn_soa_len2(out->y, m->m[3], m->m[4], m->count);
}

void WMATH_CALL(Mat3xN, get_3D_scaling)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat3xN)* m) {
    if (!out || !m || out->count != m->count) return;
    /* all three components of each row -- the 3D form. */
    wcn_soa_len3(out->x, m->m[0], m->m[1], m->m[2], m->count);
    wcn_soa_len3(out->y, m->m[3], m->m[4], m->m[5], m->count);
    wcn_soa_len3(out->z, m->m[6], m->m[7], m->m[8], m->count);
}

/* ----------------------------------------------------------------- conversion */

void WMATH_CALL(Mat3xN, from_quat)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(QuatxN)* q) {
    if (!dst || !q || dst->count != q->count) return;
    /* Per-element scalar: the algebra could share the Mat4xN kernel's products, but this is
     * not on an inner loop, so a second implementation would be upkeep without gain. */
    for (size_t i = 0; i < q->count; i++) {
        WMATH_TYPE(Quat) qq;
        WMATH_TYPE(Mat3) r;
        qq.v[0] = q->x[i];
        qq.v[1] = q->y[i];
        qq.v[2] = q->z[i];
        qq.v[3] = q->w[i];
        WMATH_CALL(Mat3, from_quat)(&r, qq);
        wcn_math_Mat3xN_set(dst, i, &r);
    }
}
