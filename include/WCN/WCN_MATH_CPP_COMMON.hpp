#pragma once 
#include <cstdint>
#include "WCN/WCN_MATH_TYPES.h"
namespace WCN {
    inline namespace Math {
        using Vec2          = WMATH_TYPE(Vec2);
        using Vec3          = WMATH_TYPE(Vec3);
        using Vec4          = WMATH_TYPE(Vec4);
        using Quat          = WMATH_TYPE(Quat);
        using Mat3          = WMATH_TYPE(Mat3);
        using Mat4          = WMATH_TYPE(Mat4);
        enum class RotationOrder: std::uint8_t {
            XYZ = WCN_Math_RotationOrder_XYZ,
            XZY = WCN_Math_RotationOrder_XZY,
            YXZ = WCN_Math_RotationOrder_YXZ,
            YZX = WCN_Math_RotationOrder_YZX,
            ZYX = WCN_Math_RotationOrder_ZYX,
            ZXY = WCN_Math_RotationOrder_ZXY,
        };
    }
}