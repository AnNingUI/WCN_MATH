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
// END Vec2xN

// BIGIN Vec3xN
// END Vec3xN

// BIGIN Vec4xN

/**
 * @brief Free Vec4xN arrays (caller must call this to release memory)
 * @param vec4xN Pointer to Vec4xN structure
 * @note After calling this function, all array pointers will be set to NULL
 */
void wcn_math_Vec4xN_free(WMATH_TYPE(Vec4xN)* vec4xN);

// END Vec4xN

// BIGIN QuatxN
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
// END Mat4xN

#ifdef __cplusplus
}
#endif
#endif // WCN_MATH_SOA_H