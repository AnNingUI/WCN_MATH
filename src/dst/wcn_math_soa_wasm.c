/*
 * SoA (xN) WASM export layer.
 *
 * Plane arrays plus a count cross the boundary; the SoA structs do not. Each wrapper builds the
 * structs on the stack from the caller's pointers, calls the operator, and returns -- JS passes
 * pointers straight into wasm memory, needs no knowledge of the layout, and the caller's planes
 * are written in place, so there is no ownership question.
 *
 * GENERATED -- see SOA_OPS.md. Four of the six SoA types hold one pointer per component while
 * Mat3xN/Mat4xN hold an array (float* m[9] / m[16]), so their initialiser carries an inner brace
 * pair. The Mat4xN wrappers take sixteen plane pointers each; a hand-written copy would be a
 * transposition waiting to happen, whereas a wrong operator name simply fails to compile.
 *
 * Mat4xN_inverse_wasm calls WMATH_CALL(Mat4xN, inverse_soa): WMATH_INVERSE(Mat4xN) would expand to
 * wcn_math_Mat4xN_inverse, which is the AoS batch entry point taking (float*, const float*, size_t).
 *
 * Compiled only for the WASM build, matching the guard in wcn_math_wasm.c.
 */
#ifdef __EMSCRIPTEN__

#include <WCN/WCN_MATH_DST.h>
#include <WCN/WCN_MATH_SOA.h>

#include <stddef.h>

/* ============================ Vec2xN ============================ */
WCN_WASM_EXPORT void wcn_math_Vec2xN_add_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_ADD(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_sub_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_SUB(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_multiply_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_MULTIPLY(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_div_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_DIV(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_fmin_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_FMIN(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_fmax_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_FMAX(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_midpoint_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_MIDPOINT(Vec2xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_negate_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_NEGATE(Vec2xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_inverse_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_INVERSE(Vec2xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_normalize_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_NORMALIZE(Vec2xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_copy_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_COPY(Vec2xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_multiply_scalar_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float s, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_MULTIPLY_SCALAR(Vec2xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_div_scalar_wasm(float* d_x, float* d_y, const float* a_x, const float* a_y, const float s, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_DIV_SCALAR(Vec2xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_dot_wasm(float* out, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_DOT(Vec2xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_distance_wasm(float* out, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_DISTANCE(Vec2xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_distance_sq_wasm(float* out, const float* a_x, const float* a_y, const float* b_x, const float* b_y, size_t count) {
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_TYPE(Vec2xN) b = { b_x, b_y, count };
    WMATH_DISTANCE_SQ(Vec2xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_length_wasm(float* out, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_LENGTH(Vec2xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_length_sq_wasm(float* out, const float* a_x, const float* a_y, size_t count) {
    WMATH_TYPE(Vec2xN) a = { a_x, a_y, count };
    WMATH_LENGTH_SQ(Vec2xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_zero_wasm(float* d_x, float* d_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_ZERO(Vec2xN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec2xN_identity_wasm(float* d_x, float* d_y, size_t count) {
    WMATH_TYPE(Vec2xN) d = { d_x, d_y, count };
    WMATH_IDENTITY(Vec2xN)(&d);
    (void)d;
}
/* ============================ Vec3xN ============================ */
WCN_WASM_EXPORT void wcn_math_Vec3xN_add_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_ADD(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_sub_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_SUB(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_multiply_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_MULTIPLY(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_div_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_DIV(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_fmin_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_FMIN(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_fmax_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_FMAX(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_midpoint_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_MIDPOINT(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_cross_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_CROSS(Vec3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_negate_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_NEGATE(Vec3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_inverse_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_INVERSE(Vec3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_normalize_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_NORMALIZE(Vec3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_copy_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_COPY(Vec3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_multiply_scalar_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float s, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_MULTIPLY_SCALAR(Vec3xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_div_scalar_wasm(float* d_x, float* d_y, float* d_z, const float* a_x, const float* a_y, const float* a_z, const float s, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_DIV_SCALAR(Vec3xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_dot_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_DOT(Vec3xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_distance_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_DISTANCE(Vec3xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_distance_sq_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* b_x, const float* b_y, const float* b_z, size_t count) {
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_TYPE(Vec3xN) b = { b_x, b_y, b_z, count };
    WMATH_DISTANCE_SQ(Vec3xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_length_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_LENGTH(Vec3xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_length_sq_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, size_t count) {
    WMATH_TYPE(Vec3xN) a = { a_x, a_y, a_z, count };
    WMATH_LENGTH_SQ(Vec3xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_zero_wasm(float* d_x, float* d_y, float* d_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_ZERO(Vec3xN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec3xN_identity_wasm(float* d_x, float* d_y, float* d_z, size_t count) {
    WMATH_TYPE(Vec3xN) d = { d_x, d_y, d_z, count };
    WMATH_IDENTITY(Vec3xN)(&d);
    (void)d;
}
/* ============================ Vec4xN ============================ */
WCN_WASM_EXPORT void wcn_math_Vec4xN_add_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_ADD(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_sub_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_SUB(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_multiply_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_MULTIPLY(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_div_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_DIV(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_fmin_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_FMIN(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_fmax_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_FMAX(Vec4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_negate_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_NEGATE(Vec4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_inverse_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_INVERSE(Vec4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_normalize_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_NORMALIZE(Vec4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_copy_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_COPY(Vec4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_multiply_scalar_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float s, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_MULTIPLY_SCALAR(Vec4xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_div_scalar_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float s, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_DIV_SCALAR(Vec4xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_dot_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_DOT(Vec4xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_distance_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_DISTANCE(Vec4xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_distance_sq_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(Vec4xN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_DISTANCE_SQ(Vec4xN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_length_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_LENGTH(Vec4xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_length_sq_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(Vec4xN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_LENGTH_SQ(Vec4xN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_zero_wasm(float* d_x, float* d_y, float* d_z, float* d_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_ZERO(Vec4xN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Vec4xN_identity_wasm(float* d_x, float* d_y, float* d_z, float* d_w, size_t count) {
    WMATH_TYPE(Vec4xN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_IDENTITY(Vec4xN)(&d);
    (void)d;
}
/* ============================ QuatxN ============================ */
WCN_WASM_EXPORT void wcn_math_QuatxN_add_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(QuatxN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_ADD(QuatxN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_sub_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(QuatxN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_SUB(QuatxN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_multiply_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(QuatxN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_MULTIPLY(QuatxN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_inverse_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_INVERSE(QuatxN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_normalize_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_NORMALIZE(QuatxN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_copy_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_COPY(QuatxN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_multiply_scalar_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float s, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_MULTIPLY_SCALAR(QuatxN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_div_scalar_wasm(float* d_x, float* d_y, float* d_z, float* d_w, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float s, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_DIV_SCALAR(QuatxN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_dot_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, const float* b_x, const float* b_y, const float* b_z, const float* b_w, size_t count) {
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_TYPE(QuatxN) b = { b_x, b_y, b_z, b_w, count };
    WMATH_DOT(QuatxN)(out, &a, &b);
}
WCN_WASM_EXPORT void wcn_math_QuatxN_length_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_LENGTH(QuatxN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_QuatxN_length_sq_wasm(float* out, const float* a_x, const float* a_y, const float* a_z, const float* a_w, size_t count) {
    WMATH_TYPE(QuatxN) a = { a_x, a_y, a_z, a_w, count };
    WMATH_LENGTH_SQ(QuatxN)(out, &a);
}
WCN_WASM_EXPORT void wcn_math_QuatxN_zero_wasm(float* d_x, float* d_y, float* d_z, float* d_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_ZERO(QuatxN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_QuatxN_identity_wasm(float* d_x, float* d_y, float* d_z, float* d_w, size_t count) {
    WMATH_TYPE(QuatxN) d = { d_x, d_y, d_z, d_w, count };
    WMATH_IDENTITY(QuatxN)(&d);
    (void)d;
}
/* ============================ Mat3xN ============================ */
WCN_WASM_EXPORT void wcn_math_Mat3xN_add_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_TYPE(Mat3xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8 }, count };
    WMATH_ADD(Mat3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_sub_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_TYPE(Mat3xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8 }, count };
    WMATH_SUB(Mat3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_multiply_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_TYPE(Mat3xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8 }, count };
    WMATH_MULTIPLY(Mat3xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_negate_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_NEGATE(Mat3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_inverse_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_INVERSE(Mat3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_transpose_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_TRANSPOSE(Mat3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_copy_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_COPY(Mat3xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_multiply_scalar_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float s, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_TYPE(Mat3xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8 }, count };
    WMATH_MULTIPLY_SCALAR(Mat3xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_zero_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_ZERO(Mat3xN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat3xN_identity_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, size_t count) {
    WMATH_TYPE(Mat3xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8 }, count };
    WMATH_IDENTITY(Mat3xN)(&d);
    (void)d;
}
/* ============================ Mat4xN ============================ */
WCN_WASM_EXPORT void wcn_math_Mat4xN_add_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, const float* b_m9, const float* b_m10, const float* b_m11, const float* b_m12, const float* b_m13, const float* b_m14, const float* b_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_TYPE(Mat4xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9, b_m10, b_m11, b_m12, b_m13, b_m14, b_m15 }, count };
    WMATH_ADD(Mat4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_sub_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, const float* b_m9, const float* b_m10, const float* b_m11, const float* b_m12, const float* b_m13, const float* b_m14, const float* b_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_TYPE(Mat4xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9, b_m10, b_m11, b_m12, b_m13, b_m14, b_m15 }, count };
    WMATH_SUB(Mat4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_multiply_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, const float* b_m0, const float* b_m1, const float* b_m2, const float* b_m3, const float* b_m4, const float* b_m5, const float* b_m6, const float* b_m7, const float* b_m8, const float* b_m9, const float* b_m10, const float* b_m11, const float* b_m12, const float* b_m13, const float* b_m14, const float* b_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_TYPE(Mat4xN) b = { { b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9, b_m10, b_m11, b_m12, b_m13, b_m14, b_m15 }, count };
    WMATH_MULTIPLY(Mat4xN)(&d, &a, &b);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_negate_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_NEGATE(Mat4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_inverse_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_CALL(Mat4xN, inverse_soa)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_transpose_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_TRANSPOSE(Mat4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_copy_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_COPY(Mat4xN)(&d, &a);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_multiply_scalar_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, const float* a_m0, const float* a_m1, const float* a_m2, const float* a_m3, const float* a_m4, const float* a_m5, const float* a_m6, const float* a_m7, const float* a_m8, const float* a_m9, const float* a_m10, const float* a_m11, const float* a_m12, const float* a_m13, const float* a_m14, const float* a_m15, const float s, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_TYPE(Mat4xN) a = { { a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10, a_m11, a_m12, a_m13, a_m14, a_m15 }, count };
    WMATH_MULTIPLY_SCALAR(Mat4xN)(&d, &a, s);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_zero_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_ZERO(Mat4xN)(&d);
    (void)d;
}
WCN_WASM_EXPORT void wcn_math_Mat4xN_identity_wasm(float* d_m0, float* d_m1, float* d_m2, float* d_m3, float* d_m4, float* d_m5, float* d_m6, float* d_m7, float* d_m8, float* d_m9, float* d_m10, float* d_m11, float* d_m12, float* d_m13, float* d_m14, float* d_m15, size_t count) {
    WMATH_TYPE(Mat4xN) d = { { d_m0, d_m1, d_m2, d_m3, d_m4, d_m5, d_m6, d_m7, d_m8, d_m9, d_m10, d_m11, d_m12, d_m13, d_m14, d_m15 }, count };
    WMATH_IDENTITY(Mat4xN)(&d);
    (void)d;
}
#endif /* __EMSCRIPTEN__ */
