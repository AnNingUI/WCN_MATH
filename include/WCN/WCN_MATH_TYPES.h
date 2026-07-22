#ifndef WCN_MATH_TYPES_H
#define WCN_MATH_TYPES_H

#include "WCN/Rtype.h"
#include "WCN/WCN_MATH_MACROS.h"
#include <stddef.h>
#include <stdint.h>

// TYPE

// Mat3 Type

typedef struct {
  float m[12]; // Using 12 elements for better SIMD alignment
} WMATH_TYPE(Mat3);

typedef struct {
  float m_00;
  float m_01;
  float m_02;
  // next row
  float m_10;
  float m_11;
  float m_12;
  // next row
  float m_20;
  float m_21;
  float m_22;
} WMATH_CREATE_TYPE(Mat3);

// Mat4 Type

typedef struct {
  float m[16];
} WMATH_TYPE(Mat4);

typedef struct {
  float m_00;
  float m_01;
  float m_02;
  float m_03;
  // next row
  float m_10;
  float m_11;
  float m_12;
  float m_13;
  // next row
  float m_20;
  float m_21;
  float m_22;
  float m_23;
  // next row
  float m_30;
  float m_31;
  float m_32;
  float m_33;
} WMATH_CREATE_TYPE(Mat4);

// Quat Type
#ifdef _MSC_VER
    #define WCN_ALIGNED_STRUCT(x) __declspec(align(x)) struct
#else
    // GCC / Clang 支持将 __attribute__ 放在 struct 关键字之后
    #define WCN_ALIGNED_STRUCT(x) struct __attribute__((aligned(x)))
#endif
typedef WCN_ALIGNED_STRUCT(16) {
  float v[4];
} WMATH_TYPE(Quat);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
  float v_w;
} WMATH_CREATE_TYPE(Quat);

enum WCN_Math_RotationOrder {
  WCN_Math_RotationOrder_XYZ = 0,
  WCN_Math_RotationOrder_XZY = 1,
  WCN_Math_RotationOrder_YXZ = 2,
  WCN_Math_RotationOrder_YZX = 3,
  WCN_Math_RotationOrder_ZXY = 4,
  WCN_Math_RotationOrder_ZYX = 5,
};
#define WCN_MATH_IS_VALID_ROTATION_ORDER(order)                                \
  ((order) >= WCN_Math_RotationOrder_XYZ &&                                    \
   (order) <= WCN_Math_RotationOrder_ZYX)

// 或者使用数组大小来确保一致性
#define WCN_MATH_ROTATION_ORDER_COUNT 6
extern const int WCN_MATH_ROTATION_SIGN_TABLE[WCN_MATH_ROTATION_ORDER_COUNT][4];

// Vec2 Type

typedef struct {
  float v[2];
} WMATH_TYPE(Vec2);

typedef struct {
  float v_x;
  float v_y;
} WMATH_CREATE_TYPE(Vec2);

// Vec3 Type

typedef struct {
  float v[3];
} WMATH_TYPE(Vec3);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
} WMATH_CREATE_TYPE(Vec3);

typedef struct {
  float angle;
  WMATH_TYPE(Vec3) axis;
} WCN_Math_Vec3_WithAngleAxis;

// Vec4 Type

typedef WCN_ALIGNED_STRUCT(16) {
  float v[4];
} WMATH_TYPE(Vec4);

typedef struct {
  float v_x;
  float v_y;
  float v_z;
  float v_w;
} WMATH_CREATE_TYPE(Vec4);

// #ifdef __WMATH_SOA__
// Vec2xN: 适用于路径点、UV坐标流
typedef struct {
    float* x; // 必须 16 字节对齐
    float* y;
    size_t count;
} WMATH_TYPE(Vec2xN);

// Vec3xN: 适用于基础粒子位置、速度、颜色(RGB)
typedef struct {
    float* x;
    float* y;
    float* z;
    size_t count;
} WMATH_TYPE(Vec3xN);

// Vec4xN: 适用于带 Alpha 的颜色、切线流
typedef struct {
    float* x;
    float* y;
    float* z;
    float* w;
    size_t count;
} WMATH_TYPE(Vec4xN);

// QuatxN: 适用于大量旋转数据的更新
typedef struct {
    float* x;
    float* y;
    float* z;
    float* w;
    size_t count;
} WMATH_TYPE(QuatxN);

// Mat3xN: 适用于 3D 旋转/缩放矩阵序列
typedef struct {
    float* m[9]; // m[0] 指向所有矩阵的 m00，m[1] 指向 m01...
    size_t count;
} WMATH_TYPE(Mat3xN);
typedef NO_OP_IMPL struct {
    float * m00;
    float * m01;
    float * m02;
    float * m10;
    float * m11;
    float * m12;
    float * m20;
    float * m21;
    float * m22;
    size_t count;
} WMATH_TYPE(Mat3xN_Flat); // 本结构体不会有算子实现
// Mat4xN: 适用于骨骼变换矩阵、实例渲染世界矩阵
typedef struct {
    float* m[16]; // m[0..15] 映射 4x4 矩阵的 16 个分量
    size_t count;
} WMATH_TYPE(Mat4xN);
typedef NO_OP_IMPL struct {
    float * m00;
    float * m01;
    float * m02;
    float * m03;

    float * m10;
    float * m11;
    float * m12;
    float * m13;

    float * m20;
    float * m21;
    float * m22;
    float * m23;

    float * m30;
    float * m31;
    float * m32;
    float * m33;
    size_t count;
} WMATH_TYPE(Mat4xN_Flat); // 本结构体不会有算子实现
// ImageRGBA
typedef struct {
    f32* r;
    f32* g;
    f32* b;
    f32* a;
    size_t width;
    size_t height;
} WMATH_TYPE(ImageRGBA);

typedef struct {
    u8* data;
    size_t width;
    size_t height;
} WMATH_TYPE(ImageBitmap);

typedef WMATH_TYPE(ImageBitmap) WMATH_TYPE(ImageBitmapU8);

typedef struct {
    u16* data;
    size_t width;
    size_t height;
} WMATH_TYPE(ImageBitmapU16);

typedef struct {
    f32* data;
    size_t width;
    size_t height;
} WMATH_TYPE(ImageBitmapF32);

// #endif // __WMATH_SOA__
#endif // WCN_MATH_TYPES_H




