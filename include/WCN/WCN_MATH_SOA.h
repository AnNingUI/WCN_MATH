#ifndef WCN_MATH_SOA_H
#define WCN_MATH_SOA_H
#ifdef __cplusplus
extern "C" {
#endif
#include <WCN/WCN_MATH_TYPES.h>
#include <WCN/WCN_MATH_MACROS.h>
#include <stdbool.h>
#include <stdlib.h>

#if defined(_MSC_VER)
    #define WCN_ALIGN16 __declspec(align(16))
#else
    #define WCN_ALIGN16 alignas(16)
#endif

// BIGIN Vec2xN
WMATH_TYPE(Vec2xN) * wcn_math_Vec2xN_malloc(const size_t count);
void wcn_math_Vec2xN_free(WMATH_TYPE(Vec2xN) * ptr);
float * wcn_math_Vec2xN_get_xarray(WMATH_TYPE(Vec2xN) * ptr);
float * wcn_math_Vec2xN_get_yarray(WMATH_TYPE(Vec2xN) * ptr);
size_t wcn_math_Vec2xN_get_count(const WMATH_TYPE(Vec2xN) * ptr);
bool wcn_math_Vec2xN_get(
    WMATH_TYPE(Vec2)* out,            
    const WMATH_TYPE(Vec2xN)* src,    
    size_t index
);
bool wcn_math_Vec2xN_set(
    WMATH_TYPE(Vec2xN)* dst,          
    const size_t index,
    const WMATH_TYPE(Vec2)* input     
);
bool wcn_math_Vec2xN_set_by_xy
(
    const WMATH_TYPE(Vec2xN)* v,
    const size_t index, const float x, const float y
);
void WMATH_ADD(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b
);
void WMATH_ADD_SCALED(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b,
    const float s
);
void WMATH_SUB(Vec2xN)(
    WMATH_TYPE(Vec2xN)* dst,
    const WMATH_TYPE(Vec2xN)* a,
    const WMATH_TYPE(Vec2xN)* b
);
void WMATH_MULTIPLY(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_DIV(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_FMIN(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_FMAX(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_MULTIPLY_SCALAR(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float s);
void WMATH_DIV_SCALAR(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float s);
void WMATH_CLAMP(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float lo, const float hi);
void WMATH_NEGATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_INVERSE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_NORMALIZE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_LERP(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b, const float t);
void WMATH_LERP_V(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b, const WMATH_TYPE(Vec2xN)* t);
void WMATH_MIDPOINT(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_SET_LENGTH(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float len);
void WMATH_DOT(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_LENGTH(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a);
void WMATH_LENGTH_SQ(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a);
void WMATH_DISTANCE(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_DISTANCE_SQ(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_COPY(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_ZERO(Vec2xN)(WMATH_TYPE(Vec2xN)* dst);
void WMATH_IDENTITY(Vec2xN)(WMATH_TYPE(Vec2xN)* dst);
/* CREATE/SET apply one set of component values to EVERY element (a fill), which is the
 * container-level lift of the scalar operation and is distinct from set_by_xy below,
 * which writes a single element. */
/* No WMATH_SET(Vec2xN): that name expands to wcn_math_Vec2xN_set, which is already the
 * per-element accessor (dst, index, input) declared above. See wcn_math_vec2xn.c for why
 * the operator is not provided under a different spelling either. */
void WMATH_CREATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_CREATE_TYPE(Vec2) c);
bool WMATH_EQUALS(Vec2xN)(const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
bool WMATH_EQUALS_APPROXIMATELY(Vec2xN)(const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_FLOOR(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_CEIL(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_ROUND(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a);
void WMATH_TRUNCATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const float max_length);
void WMATH_CROSS(Vec2xN)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_ROTATE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b, const float* rad);
void WMATH_SCALE(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* v, const float* scale);
void WMATH_CALL(Vec2xN, transform_mat3)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* v, const WMATH_TYPE(Mat3xN)* m);
void WMATH_CALL(Vec2xN, transform_mat4)(WMATH_TYPE(Vec2xN)* dst, const WMATH_TYPE(Vec2xN)* v, const WMATH_TYPE(Mat4xN)* m);
void WMATH_ANGLE(Vec2xN)(float* out, const WMATH_TYPE(Vec2xN)* a, const WMATH_TYPE(Vec2xN)* b);
void WMATH_RANDOM(Vec2xN)(WMATH_TYPE(Vec2xN)* dst, const float* scale);
// END Vec2xN

// BIGIN Vec3xN
WMATH_TYPE(Vec3xN) * wcn_math_Vec3xN_malloc(const size_t count);
void wcn_math_Vec3xN_free(WMATH_TYPE(Vec3xN) * ptr);
float * wcn_math_Vec3xN_get_xarray(WMATH_TYPE(Vec3xN) * ptr);
float * wcn_math_Vec3xN_get_yarray(WMATH_TYPE(Vec3xN) * ptr);
float * wcn_math_Vec3xN_get_zarray(WMATH_TYPE(Vec3xN) * ptr);
size_t wcn_math_Vec3xN_get_count(const WMATH_TYPE(Vec3xN) * ptr);

bool wcn_math_Vec3xN_get(
    WMATH_TYPE(Vec3)* out,
    const WMATH_TYPE(Vec3xN)* src,
    size_t index
);
bool wcn_math_Vec3xN_set(
    WMATH_TYPE(Vec3xN)* dst,
    const size_t index,
    const WMATH_TYPE(Vec3)* input
);
bool wcn_math_Vec3xN_set_by_xyz(
    const WMATH_TYPE(Vec3xN)* v,
    const size_t index, const float x, const float y, const float z
);

/* element-wise binary */
void WMATH_ADD(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_SUB(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_MULTIPLY(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_DIV(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_FMIN(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_FMAX(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);

/* element-wise with a scalar */
void WMATH_MULTIPLY_SCALAR(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const float s);
void WMATH_DIV_SCALAR(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const float s);
void WMATH_ADD_SCALED(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const float s);
void WMATH_CLAMP(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const float lo, const float hi);

/* element-wise unary / in-place-shaped */
void WMATH_NEGATE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_NORMALIZE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_INVERSE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_LERP(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const float t);
void WMATH_LERP_V(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const WMATH_TYPE(Vec3xN)* t);
void WMATH_MIDPOINT(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_CROSS(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);

/* per-element reductions: one scalar per element, hence a bare float array */
void WMATH_DOT(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_LENGTH(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a);
void WMATH_LENGTH_SQ(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a);
void WMATH_DISTANCE(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_DISTANCE_SQ(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);

/* whole-container */
void WMATH_COPY(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_ZERO(Vec3xN)(WMATH_TYPE(Vec3xN)* dst);
bool WMATH_EQUALS(Vec3xN)(const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
bool WMATH_EQUALS_APPROXIMATELY(Vec3xN)(const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_CREATE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_CREATE_TYPE(Vec3) c);
void WMATH_FLOOR(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_CEIL(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_ROUND(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a);
void WMATH_IDENTITY(Vec3xN)(WMATH_TYPE(Vec3xN)* dst);
void WMATH_SET_LENGTH(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const float len);
void WMATH_TRUNCATE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const float max_length);
void WMATH_ANGLE(Vec3xN)(float* out, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b);
void WMATH_CALL(Vec3xN, get_axis)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m, int axis);
void WMATH_CALL(Vec3xN, get_scale)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m);
void WMATH_GET_TRANSLATION(Vec3xN)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m);
void WMATH_ROTATE_X(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const float* rad);
void WMATH_ROTATE_Y(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const float* rad);
void WMATH_ROTATE_Z(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* a, const WMATH_TYPE(Vec3xN)* b, const float* rad);
void WMATH_SCALE(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* v, const float* scale);
void WMATH_CALL(Vec3xN, transform_mat3)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* v, const WMATH_TYPE(Mat3xN)* m);
void WMATH_CALL(Vec3xN, transform_mat4)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* v, const WMATH_TYPE(Mat4xN)* m);
void WMATH_CALL(Vec3xN, transform_mat4_upper3x3)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* v, const WMATH_TYPE(Mat4xN)* m);
void WMATH_CALL(Vec3xN, transform_quat)(WMATH_TYPE(Vec3xN)* dst, const WMATH_TYPE(Vec3xN)* v, const WMATH_TYPE(QuatxN)* q);
void WMATH_RANDOM(Vec3xN)(WMATH_TYPE(Vec3xN)* dst, const float* scale);
// END Vec3xN

// BIGIN Vec4xN
WMATH_TYPE(Vec4xN) * wcn_math_Vec4xN_malloc(const size_t count);
void wcn_math_Vec4xN_free(WMATH_TYPE(Vec4xN)* vec4xN);
void wcn_math_Vec4xN_destroy(WMATH_TYPE(Vec4xN)* vec4xN);
float * wcn_math_Vec4xN_get_xarray(WMATH_TYPE(Vec4xN) * ptr);
float * wcn_math_Vec4xN_get_yarray(WMATH_TYPE(Vec4xN) * ptr);
float * wcn_math_Vec4xN_get_zarray(WMATH_TYPE(Vec4xN) * ptr);
float * wcn_math_Vec4xN_get_warray(WMATH_TYPE(Vec4xN) * ptr);
size_t wcn_math_Vec4xN_get_count(const WMATH_TYPE(Vec4xN) * ptr);

bool wcn_math_Vec4xN_get(
    WMATH_TYPE(Vec4)* out,
    const WMATH_TYPE(Vec4xN)* src,
    size_t index
);
bool wcn_math_Vec4xN_set(
    WMATH_TYPE(Vec4xN)* dst,
    const size_t index,
    const WMATH_TYPE(Vec4)* input
);
bool wcn_math_Vec4xN_set_by_xyzw(
    const WMATH_TYPE(Vec4xN)* v,
    const size_t index, const float x, const float y, const float z, const float w
);

/* element-wise binary */
void WMATH_ADD(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_SUB(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_MULTIPLY(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_DIV(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_FMIN(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_FMAX(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);

/* element-wise with a scalar */
void WMATH_MULTIPLY_SCALAR(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float s);
void WMATH_DIV_SCALAR(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float s);
void WMATH_ADD_SCALED(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b, const float s);
void WMATH_CLAMP(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float lo, const float hi);

/* unary and fused */
void WMATH_NEGATE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_NORMALIZE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_INVERSE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_LERP(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b, const float t);
void WMATH_LERP_V(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b, const WMATH_TYPE(Vec4xN)* t);
void WMATH_MIDPOINT(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_SET_LENGTH(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float len);
void WMATH_TRUNCATE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a, const float max_length);

/* per-element reductions: one scalar per element, hence a bare float array */
void WMATH_DOT(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_LENGTH(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a);
void WMATH_LENGTH_SQ(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a);
void WMATH_DISTANCE(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_DISTANCE_SQ(Vec4xN)(float* out, const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);

/* whole-container */
void WMATH_COPY(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_ZERO(Vec4xN)(WMATH_TYPE(Vec4xN)* dst);
void WMATH_IDENTITY(Vec4xN)(WMATH_TYPE(Vec4xN)* dst);
bool WMATH_EQUALS(Vec4xN)(const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
bool WMATH_EQUALS_APPROXIMATELY(Vec4xN)(const WMATH_TYPE(Vec4xN)* a, const WMATH_TYPE(Vec4xN)* b);
void WMATH_CREATE(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_CREATE_TYPE(Vec4) c);
void WMATH_FLOOR(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_CEIL(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_ROUND(Vec4xN)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* a);
void WMATH_CALL(Vec4xN, transform_mat4)(WMATH_TYPE(Vec4xN)* dst, const WMATH_TYPE(Vec4xN)* v, const WMATH_TYPE(Mat4xN)* m);
// END Vec4xN


// BIGIN QuatxN
WMATH_TYPE(QuatxN) * wcn_math_QuatxN_malloc(const size_t count);
void wcn_math_QuatxN_destroy(WMATH_TYPE(QuatxN) * ptr);
void wcn_math_QuatxN_free(WMATH_TYPE(QuatxN) * ptr);
float * wcn_math_QuatxN_get_xarray(WMATH_TYPE(QuatxN) * ptr);
float * wcn_math_QuatxN_get_yarray(WMATH_TYPE(QuatxN) * ptr);
float * wcn_math_QuatxN_get_zarray(WMATH_TYPE(QuatxN) * ptr);
float * wcn_math_QuatxN_get_warray(WMATH_TYPE(QuatxN) * ptr);
size_t wcn_math_QuatxN_get_count(const WMATH_TYPE(QuatxN) * ptr);

bool wcn_math_QuatxN_get(
    WMATH_TYPE(Quat)* out,
    const WMATH_TYPE(QuatxN)* src,
    size_t index
);
bool wcn_math_QuatxN_set(
    WMATH_TYPE(QuatxN)* dst,
    const size_t index,
    const WMATH_TYPE(Quat)* input
);
bool wcn_math_QuatxN_set_by_xyzw(
    const WMATH_TYPE(QuatxN)* q,
    const size_t index, const float x, const float y, const float z, const float w
);

void WMATH_ADD(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_SUB(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_MULTIPLY(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_MULTIPLY_SCALAR(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const float s);
void WMATH_DIV_SCALAR(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const float s);
void WMATH_SCALE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const float s);

void WMATH_NORMALIZE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a);
void WMATH_CALL(QuatxN, conjugate)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a);
void WMATH_INVERSE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a);
void WMATH_LERP(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b, const float t);

void WMATH_DOT(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_LENGTH(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a);
void WMATH_LENGTH_SQ(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a);

void WMATH_COPY(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a);
void WMATH_ZERO(QuatxN)(WMATH_TYPE(QuatxN)* dst);
void WMATH_IDENTITY(QuatxN)(WMATH_TYPE(QuatxN)* dst);
bool WMATH_EQUALS(QuatxN)(const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
bool WMATH_EQUALS_APPROXIMATELY(QuatxN)(const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_CREATE(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_CREATE_TYPE(Quat) c);
void WMATH_ANGLE(QuatxN)(float* out, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b);
void WMATH_CALL(QuatxN, from_mat3)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Mat3xN)* m);
void WMATH_CALL(QuatxN, from_mat4)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Mat4xN)* m);
void WMATH_CALL(QuatxN, rotation_to)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Vec3xN)* a_unit, const WMATH_TYPE(Vec3xN)* b_unit);
void WMATH_CALL(QuatxN, slerp)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b, const float* t);
void WMATH_CALL(QuatxN, sqlerp)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* a, const WMATH_TYPE(QuatxN)* b, const WMATH_TYPE(QuatxN)* c, const WMATH_TYPE(QuatxN)* d, const float* t);
void WMATH_CALL(QuatxN, from_axis_angle)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(Vec3xN)* axis, const float* radians);
void WMATH_CALL(QuatxN, from_euler)(WMATH_TYPE(QuatxN)* dst, const float* x, const float* y, const float* z, enum WCN_Math_RotationOrder order);
void WMATH_ROTATE_X(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q, const float* radians);
void WMATH_ROTATE_Y(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q, const float* radians);
void WMATH_ROTATE_Z(QuatxN)(WMATH_TYPE(QuatxN)* dst, const WMATH_TYPE(QuatxN)* q, const float* radians);
// END QuatxN

// BEGIN Mat3xN
/**
 * Mat3xN -> Mat3xN_Flat
 */ 
inline void wcn_math_Mat3xN_to_Mat3xN_Flat(WMATH_TYPE(Mat3xN_Flat)* dst, WMATH_TYPE(Mat3xN)* mat3xN) 
{
    dst->m00 = mat3xN->m[0];
    dst->m01 = mat3xN->m[1];
    dst->m02 = mat3xN->m[2];
    dst->m10 = mat3xN->m[3];
    dst->m11 = mat3xN->m[4];
    dst->m12 = mat3xN->m[5];
    dst->m20 = mat3xN->m[6];
    dst->m21 = mat3xN->m[7];
    dst->m22 = mat3xN->m[8];
    dst->count = mat3xN->count;
};

inline void wcn_math_Mat3xN_Flat_to_Mat3xN(WMATH_TYPE(Mat3xN)* dst, WMATH_TYPE(Mat3xN_Flat)* mat3xN_flat) 
{
    dst->m[0] = mat3xN_flat->m00;
    dst->m[1] = mat3xN_flat->m01;
    dst->m[2] = mat3xN_flat->m02;
    dst->m[3] = mat3xN_flat->m10;
    dst->m[4] = mat3xN_flat->m11;
    dst->m[5] = mat3xN_flat->m12;
    dst->m[6] = mat3xN_flat->m20;
    dst->m[7] = mat3xN_flat->m21;
    dst->m[8] = mat3xN_flat->m22;
    dst->count = mat3xN_flat->count;
}
WMATH_TYPE(Mat3xN) * wcn_math_Mat3xN_malloc(const size_t count);
void wcn_math_Mat3xN_destroy(WMATH_TYPE(Mat3xN) * ptr);
void wcn_math_Mat3xN_free(WMATH_TYPE(Mat3xN) * ptr);
float * wcn_math_Mat3xN_get_element_array(WMATH_TYPE(Mat3xN) * ptr, size_t plane);
size_t wcn_math_Mat3xN_get_count(const WMATH_TYPE(Mat3xN) * ptr);

bool wcn_math_Mat3xN_get(WMATH_TYPE(Mat3)* out, const WMATH_TYPE(Mat3xN)* src, size_t index);
bool wcn_math_Mat3xN_set(WMATH_TYPE(Mat3xN)* dst, const size_t index, const WMATH_TYPE(Mat3)* input);

void WMATH_ADD(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b);
void WMATH_SUB(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b);
void WMATH_MULTIPLY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b);
void WMATH_MULTIPLY_SCALAR(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a, const float s);
void WMATH_NEGATE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a);
void WMATH_TRANSPOSE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a);
void WMATH_INVERSE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a);
void WMATH_DETERMINANT(Mat3xN)(float* out, const WMATH_TYPE(Mat3xN)* a);

void WMATH_COPY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* a);
void WMATH_ZERO(Mat3xN)(WMATH_TYPE(Mat3xN)* dst);
void WMATH_IDENTITY(Mat3xN)(WMATH_TYPE(Mat3xN)* dst);
bool WMATH_EQUALS(Mat3xN)(const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b);
bool WMATH_EQUALS_APPROXIMATELY(Mat3xN)(const WMATH_TYPE(Mat3xN)* a, const WMATH_TYPE(Mat3xN)* b);
void WMATH_CREATE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_CREATE_TYPE(Mat3) c);
void WMATH_GET_TRANSLATION(Mat3xN)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m);
void WMATH_SET_TRANSLATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const WMATH_TYPE(Vec2) v);
void WMATH_CALL(Mat3xN, from_mat4)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat4xN)* a);
void WMATH_ROTATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians);
void WMATH_ROTATION_X(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians);
void WMATH_ROTATION_Y(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians);
void WMATH_ROTATION_Z(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const float* radians);
void WMATH_ROTATE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* radians);
void WMATH_ROTATE_X(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* radians);
void WMATH_ROTATE_Y(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* radians);
void WMATH_ROTATE_Z(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* radians);
void WMATH_SCALE(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const WMATH_TYPE(Vec2) v);
void WMATH_CALL(Mat3xN, scale3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const WMATH_TYPE(Vec3) v);
void WMATH_CALL(Mat3xN, scaling)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec2) v);
void WMATH_CALL(Mat3xN, get_axis)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m, int axis);
void WMATH_CALL(Mat3xN, set_axis)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const WMATH_TYPE(Vec2xN)* v, int axis);
void WMATH_CALL(Mat3xN, get_scaling)(WMATH_TYPE(Vec2xN)* out, const WMATH_TYPE(Mat3xN)* m);
void WMATH_CALL(Mat3xN, get_3D_scaling)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat3xN)* m);
void WMATH_TRANSLATION(Mat3xN)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec2xN)* v);
void WMATH_CALL(Mat3xN, translate)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const WMATH_TYPE(Vec2xN)* v);
void WMATH_CALL(Mat3xN, scaling3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Vec3xN)* v);
void WMATH_CALL(Mat3xN, uniform_scaling)(WMATH_TYPE(Mat3xN)* dst, const float* s);
void WMATH_CALL(Mat3xN, uniform_scaling_3D)(WMATH_TYPE(Mat3xN)* dst, const float* s);
void WMATH_CALL(Mat3xN, uniform_scale)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* s);
void WMATH_CALL(Mat3xN, uniform_scale_3D)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(Mat3xN)* m, const float* s);
void WMATH_CALL(Mat3xN, from_quat)(WMATH_TYPE(Mat3xN)* dst, const WMATH_TYPE(QuatxN)* q);
// END Mat3xN

// BEGIN Mat4xN
/**
 * Mat3xN -> Mat3xN_Flat
 */ 
inline void wcn_math_Mat4xN_to_Mat4xN_Flat(WMATH_TYPE(Mat4xN_Flat)* dst, WMATH_TYPE(Mat4xN)* mat3xN) 
{
    dst->m00 = mat3xN->m[0];
    dst->m01 = mat3xN->m[1];
    dst->m02 = mat3xN->m[2];
    dst->m03 = mat3xN->m[3];

    dst->m10 = mat3xN->m[4];
    dst->m11 = mat3xN->m[5];
    dst->m12 = mat3xN->m[6];
    dst->m13 = mat3xN->m[7];

    dst->m20 = mat3xN->m[8];
    dst->m21 = mat3xN->m[9];
    dst->m22 = mat3xN->m[10];
    dst->m23 = mat3xN->m[11];

    dst->m30 = mat3xN->m[12];
    dst->m31 = mat3xN->m[13];
    dst->m32 = mat3xN->m[14];
    dst->m33 = mat3xN->m[15];

    dst->count = mat3xN->count;
};

inline void wcn_math_Mat4xN_Flat_to_Mat4xN(WMATH_TYPE(Mat4xN)* dst, WMATH_TYPE(Mat4xN_Flat)* mat3xN_flat) 
{
    dst->m[0] = mat3xN_flat->m00;
    dst->m[1] = mat3xN_flat->m01;
    dst->m[2] = mat3xN_flat->m02;
    dst->m[3] = mat3xN_flat->m03;

    dst->m[4] = mat3xN_flat->m10;
    dst->m[5] = mat3xN_flat->m11;
    dst->m[6] = mat3xN_flat->m12;
    dst->m[7] = mat3xN_flat->m13;

    dst->m[8] = mat3xN_flat->m20;
    dst->m[9] = mat3xN_flat->m21;
    dst->m[10] = mat3xN_flat->m22;
    dst->m[11] = mat3xN_flat->m23;

    dst->m[12] = mat3xN_flat->m30;
    dst->m[13] = mat3xN_flat->m31;
    dst->m[14] = mat3xN_flat->m32;
    dst->m[15] = mat3xN_flat->m33;

    dst->count = mat3xN_flat->count;
}

/**
 * @brief Batched inverse of `count` 4x4 matrices stored AoS, row-major.
 * @param dst Output, 16*count floats. Must not alias src.
 * @param src Input, 16*count floats. Matrix i occupies src[16*i .. 16*i+15],
 *            row-major (m[row*4 + col]).
 * @param count Number of matrices.
 * @note AoS is deliberate. The adjugate needs each matrix's rows as contiguous
 *       vectors; a SoA (plane-per-component) batch was written first and measured
 *       19.5 ns/matrix against 9.8 ns/matrix for calling wcn_math_Mat4_inverse in
 *       a loop, because gathering a row out of eight planes cost more than the
 *       arithmetic it fed.
 * @note On x86-64 with AVX2 the batch runs two matrices per ymm register;
 *       elsewhere it falls back to one matrix at a time.
 * @note Singular matrices are not special-cased, matching wcn_math_Mat4_inverse.
 */
void wcn_math_Mat4xN_inverse(float* dst, const float* src, size_t count);
WMATH_TYPE(Mat4xN) * wcn_math_Mat4xN_malloc(const size_t count);
void wcn_math_Mat4xN_destroy(WMATH_TYPE(Mat4xN) * ptr);
float * wcn_math_Mat4xN_get_element_array(WMATH_TYPE(Mat4xN) * ptr, size_t plane);
size_t wcn_math_Mat4xN_get_count(const WMATH_TYPE(Mat4xN) * ptr);

bool wcn_math_Mat4xN_get(WMATH_TYPE(Mat4)* out, const WMATH_TYPE(Mat4xN)* src, size_t index);
bool wcn_math_Mat4xN_set(WMATH_TYPE(Mat4xN)* dst, const size_t index, const WMATH_TYPE(Mat4)* input);

void WMATH_ADD(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b);
void WMATH_SUB(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b);
void WMATH_MULTIPLY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b);
void WMATH_MULTIPLY_SCALAR(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a, const float s);
void WMATH_NEGATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a);
void WMATH_TRANSPOSE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a);
/* NOTE: not WMATH_INVERSE(Mat4xN). That expands to wcn_math_Mat4xN_inverse, which
 * is already the AoS batched inverse declared above (16*count contiguous floats).
 * The plane-based inverse therefore takes a distinct name rather than colliding
 * with, or forcing a rename of, that tested entry point. */
void WMATH_CALL(Mat4xN, inverse_soa)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a);
void WMATH_DETERMINANT(Mat4xN)(float* out, const WMATH_TYPE(Mat4xN)* a);

void WMATH_COPY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* a);
void WMATH_ZERO(Mat4xN)(WMATH_TYPE(Mat4xN)* dst);
void WMATH_IDENTITY(Mat4xN)(WMATH_TYPE(Mat4xN)* dst);
bool WMATH_EQUALS(Mat4xN)(const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b);
bool WMATH_EQUALS_APPROXIMATELY(Mat4xN)(const WMATH_TYPE(Mat4xN)* a, const WMATH_TYPE(Mat4xN)* b);
void WMATH_CREATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_CREATE_TYPE(Mat4) c);
void WMATH_GET_TRANSLATION(Mat4xN)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m);
void WMATH_SET_TRANSLATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3) v);
void WMATH_SCALE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3) v);
void WMATH_CALL(Mat4xN, uniform_scale)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, float s);
void WMATH_CALL(Mat4xN, scaling)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3) v);
void WMATH_ROTATION_X(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians);
void WMATH_ROTATION_Y(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians);
void WMATH_ROTATION_Z(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const float* radians);
void WMATH_ROTATE_X(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const float* radians);
void WMATH_ROTATE_Y(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const float* radians);
void WMATH_ROTATE_Z(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const float* radians);
void WMATH_CALL(Mat4xN, from_mat3)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat3xN)* a);
void WMATH_CALL(Mat4xN, from_quat)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(QuatxN)* q);
void WMATH_CALL(Mat4xN, get_scaling)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m);
void WMATH_CALL(Mat4xN, axis_rotation)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* axis, const float* radians);
void WMATH_CALL(Mat4xN, axis_rotate)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3xN)* axis, const float* radians);
void WMATH_CALL(Mat4xN, perspective)(WMATH_TYPE(Mat4xN)* dst, const float* fov_y, const float* aspect, const float* z_near, const float* z_far);
void WMATH_CALL(Mat4xN, ortho)(WMATH_TYPE(Mat4xN)* dst, const float* left, const float* right, const float* bottom, const float* top, const float* znear, const float* zfar);
void WMATH_CALL(Mat4xN, aim)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* position, const WMATH_TYPE(Vec3xN)* target, const WMATH_TYPE(Vec3xN)* up);
void WMATH_CALL(Mat4xN, look_at)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* eye, const WMATH_TYPE(Vec3xN)* target, const WMATH_TYPE(Vec3xN)* up);
void WMATH_CALL(Mat4xN, camera_aim)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* eye, const WMATH_TYPE(Vec3xN)* target, const WMATH_TYPE(Vec3xN)* up);
void WMATH_CALL(Mat4xN, frustum)(WMATH_TYPE(Mat4xN)* dst, const float* left, const float* right, const float* bottom, const float* top, const float* znear, const float* zfar);
void WMATH_CALL(Mat4xN, frustum_reverse_z)(WMATH_TYPE(Mat4xN)* dst, const float* left, const float* right, const float* bottom, const float* top, const float* znear, const float* zfar);
void WMATH_CALL(Mat4xN, perspective_reverse_z)(WMATH_TYPE(Mat4xN)* dst, const float* fov_y, const float* aspect, const float* z_near, const float* z_far);
void WMATH_CALL(Mat4xN, translate)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3xN)* v);
void WMATH_TRANSLATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* v);
void WMATH_CALL(Mat4xN, uniform_scaling)(WMATH_TYPE(Mat4xN)* dst, const float* s);
void WMATH_CALL(Mat4xN, get_axis)(WMATH_TYPE(Vec3xN)* out, const WMATH_TYPE(Mat4xN)* m, int axis);
void WMATH_CALL(Mat4xN, set_axis)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3xN)* v, int axis);
void WMATH_ROTATION(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Vec3xN)* axis, const float* radians);
void WMATH_ROTATE(Mat4xN)(WMATH_TYPE(Mat4xN)* dst, const WMATH_TYPE(Mat4xN)* m, const WMATH_TYPE(Vec3xN)* axis, const float* radians);
// END Mat4xN

#endif // WCN_MATH_SOA_H