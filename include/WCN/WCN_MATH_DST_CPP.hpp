#pragma once

#include "WCN/WCN_MATH_DST.h"
#include "WCN_MATH_CPP_COMMON.hpp"
#include <utility> // for std::forward

namespace WCN
{
    inline namespace Math {
    // =========================================================================
    // 1. 管道基础设施 (Pipe Infrastructure)
    // =========================================================================

    namespace Internal {
        template <typename T>
        struct Pipe { T* dst; };

        template <typename Func>
        struct Operation { Func func; };

        template <typename Func>
        inline Operation<Func> make_op(Func&& f) {
            return { std::forward<Func>(f) };
        }

        // --- Dispatchers: 将 C++ 类型映射到 C 宏 ---
        
        #define DEF_DISPATCH_VOID(NAME, TYPE, MACRO) \
            inline void dispatch_##NAME(TYPE* d) { MACRO(TYPE)(d); }
        
        #define DEF_DISPATCH_UNARY(NAME, TYPE, MACRO) \
            inline void dispatch_##NAME(TYPE* d) { MACRO(TYPE)(d, *d); }
        
        #define DEF_DISPATCH_SCALAR(NAME, TYPE, MACRO) \
            inline void dispatch_##NAME(TYPE* d, float s) { MACRO(TYPE)(d, *d, s); }
        
        #define DEF_DISPATCH_BINARY(NAME, TYPE, MACRO) \
            inline void dispatch_##NAME(TYPE* d, const TYPE& b) { MACRO(TYPE)(d, *d, b); }

        #define DEF_DISPATCH_CLAMP(NAME, TYPE, MACRO) \
            inline void dispatch_##NAME(TYPE* d, float min, float max) { MACRO(TYPE)(d, *d, min, max); }

        // --- Vec2 Dispatch ---
        DEF_DISPATCH_VOID(identity,     Vec2, WMATH_IDENTITY)
        DEF_DISPATCH_VOID(zero,         Vec2, WMATH_ZERO)
        DEF_DISPATCH_UNARY(negate,      Vec2, WMATH_NEGATE)
        DEF_DISPATCH_UNARY(normalize,   Vec2, WMATH_NORMALIZE)
        DEF_DISPATCH_UNARY(inverse,     Vec2, WMATH_INVERSE)
        DEF_DISPATCH_UNARY(ceil,        Vec2, WMATH_CEIL)
        DEF_DISPATCH_UNARY(floor,       Vec2, WMATH_FLOOR)
        DEF_DISPATCH_UNARY(round,       Vec2, WMATH_ROUND)
        DEF_DISPATCH_SCALAR(mul_s,      Vec2, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_SCALAR(div_s,      Vec2, WMATH_DIV_SCALAR)
        DEF_DISPATCH_SCALAR(set_len,    Vec2, WMATH_SET_LENGTH)
        DEF_DISPATCH_SCALAR(truncate,   Vec2, WMATH_TRUNCATE)
        DEF_DISPATCH_CLAMP(clamp,       Vec2, WMATH_CLAMP)
        DEF_DISPATCH_BINARY(add,        Vec2, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Vec2, WMATH_SUB)
        DEF_DISPATCH_BINARY(mul,        Vec2, WMATH_MULTIPLY)
        DEF_DISPATCH_BINARY(div,        Vec2, WMATH_DIV)
        DEF_DISPATCH_BINARY(midpoint,   Vec2, WMATH_MIDPOINT)
        DEF_DISPATCH_BINARY(fmax,       Vec2, WMATH_FMAX)
        DEF_DISPATCH_BINARY(fmin,       Vec2, WMATH_FMIN)

        // --- Vec3 Dispatch ---
        DEF_DISPATCH_VOID(identity,     Vec3, WMATH_IDENTITY) 
        DEF_DISPATCH_VOID(zero,         Vec3, WMATH_ZERO)
        DEF_DISPATCH_UNARY(negate,      Vec3, WMATH_NEGATE)
        DEF_DISPATCH_UNARY(normalize,   Vec3, WMATH_NORMALIZE)
        DEF_DISPATCH_UNARY(inverse,     Vec3, WMATH_INVERSE)
        DEF_DISPATCH_UNARY(ceil,        Vec3, WMATH_CEIL)
        DEF_DISPATCH_UNARY(floor,       Vec3, WMATH_FLOOR)
        DEF_DISPATCH_UNARY(round,       Vec3, WMATH_ROUND)
        DEF_DISPATCH_SCALAR(mul_s,      Vec3, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_SCALAR(div_s,      Vec3, WMATH_DIV_SCALAR)
        DEF_DISPATCH_SCALAR(set_len,    Vec3, WMATH_SET_LENGTH)
        DEF_DISPATCH_SCALAR(truncate,   Vec3, WMATH_TRUNCATE)
        DEF_DISPATCH_CLAMP(clamp,       Vec3, WMATH_CLAMP)
        DEF_DISPATCH_BINARY(add,        Vec3, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Vec3, WMATH_SUB)
        DEF_DISPATCH_BINARY(mul,        Vec3, WMATH_MULTIPLY)
        DEF_DISPATCH_BINARY(div,        Vec3, WMATH_DIV)
        DEF_DISPATCH_BINARY(midpoint,   Vec3, WMATH_MIDPOINT)
        DEF_DISPATCH_BINARY(fmax,       Vec3, WMATH_FMAX)
        DEF_DISPATCH_BINARY(fmin,       Vec3, WMATH_FMIN)
        DEF_DISPATCH_BINARY(cross,      Vec3, WMATH_CROSS)

        // --- Vec4 Dispatch ---
        DEF_DISPATCH_VOID(identity,     Vec4, WMATH_IDENTITY)
        DEF_DISPATCH_VOID(zero,         Vec4, WMATH_ZERO)
        DEF_DISPATCH_UNARY(negate,      Vec4, WMATH_NEGATE)
        DEF_DISPATCH_UNARY(normalize,   Vec4, WMATH_NORMALIZE)
        DEF_DISPATCH_UNARY(inverse,     Vec4, WMATH_INVERSE)
        DEF_DISPATCH_UNARY(ceil,        Vec4, WMATH_CEIL)
        DEF_DISPATCH_UNARY(floor,       Vec4, WMATH_FLOOR)
        DEF_DISPATCH_UNARY(round,       Vec4, WMATH_ROUND)
        DEF_DISPATCH_SCALAR(mul_s,      Vec4, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_SCALAR(div_s,      Vec4, WMATH_DIV_SCALAR)
        DEF_DISPATCH_SCALAR(set_len,    Vec4, WMATH_SET_LENGTH)
        DEF_DISPATCH_SCALAR(truncate,   Vec4, WMATH_TRUNCATE)
        DEF_DISPATCH_CLAMP(clamp,       Vec4, WMATH_CLAMP)
        DEF_DISPATCH_BINARY(add,        Vec4, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Vec4, WMATH_SUB)
        DEF_DISPATCH_BINARY(mul,        Vec4, WMATH_MULTIPLY)
        DEF_DISPATCH_BINARY(div,        Vec4, WMATH_DIV)
        DEF_DISPATCH_BINARY(midpoint,   Vec4, WMATH_MIDPOINT)
        DEF_DISPATCH_BINARY(fmax,       Vec4, WMATH_FMAX)
        DEF_DISPATCH_BINARY(fmin,       Vec4, WMATH_FMIN)

        // --- Quat Dispatch ---
        DEF_DISPATCH_VOID(identity,     Quat, WMATH_IDENTITY)
        DEF_DISPATCH_VOID(zero,         Quat, WMATH_ZERO)
        DEF_DISPATCH_UNARY(normalize,   Quat, WMATH_NORMALIZE)
        DEF_DISPATCH_UNARY(invert,      Quat, WMATH_INVERSE)
        DEF_DISPATCH_SCALAR(mul_s,      Quat, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_SCALAR(div_s,      Quat, WMATH_DIV_SCALAR)
        DEF_DISPATCH_BINARY(mul,        Quat, WMATH_MULTIPLY)
        DEF_DISPATCH_BINARY(add,        Quat, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Quat, WMATH_SUB)

        // --- Mat3 Dispatch ---
        DEF_DISPATCH_VOID(identity,     Mat3, WMATH_IDENTITY)
        DEF_DISPATCH_VOID(zero,         Mat3, WMATH_ZERO)
        DEF_DISPATCH_UNARY(transpose,   Mat3, WMATH_TRANSPOSE)
        DEF_DISPATCH_UNARY(invert,      Mat3, WMATH_INVERSE)
        DEF_DISPATCH_UNARY(negate,      Mat3, WMATH_NEGATE)
        DEF_DISPATCH_SCALAR(mul_s,      Mat3, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_BINARY(add,        Mat3, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Mat3, WMATH_SUB)
        DEF_DISPATCH_BINARY(mul,        Mat3, WMATH_MULTIPLY)

        // --- Mat4 Dispatch ---
        DEF_DISPATCH_VOID(identity,     Mat4, WMATH_IDENTITY)
        DEF_DISPATCH_VOID(zero,         Mat4, WMATH_ZERO)
        DEF_DISPATCH_UNARY(transpose,   Mat4, WMATH_TRANSPOSE)
        DEF_DISPATCH_UNARY(invert,      Mat4, WMATH_INVERSE)
        DEF_DISPATCH_UNARY(negate,      Mat4, WMATH_NEGATE)
        DEF_DISPATCH_SCALAR(mul_s,      Mat4, WMATH_MULTIPLY_SCALAR)
        DEF_DISPATCH_BINARY(add,        Mat4, WMATH_ADD)
        DEF_DISPATCH_BINARY(sub,        Mat4, WMATH_SUB)
        DEF_DISPATCH_BINARY(mul,        Mat4, WMATH_MULTIPLY)

        // --- Special Overloaded Dispatches (Fixes Ambiguity) ---
        
        // Transform (Vec <- Mat)
        inline void dispatch_transform(Vec2* v, const Mat3& m) { WMATH_CALL(Vec2, transform_mat3)(v, *v, m); }
        inline void dispatch_transform(Vec3* v, const Mat3& m) { WMATH_CALL(Vec3, transform_mat3)(v, *v, m); }
        
        inline void dispatch_transform(Vec2* v, const Mat4& m) { WMATH_CALL(Vec2, transform_mat4)(v, *v, m); }
        inline void dispatch_transform(Vec3* v, const Mat4& m) { WMATH_CALL(Vec3, transform_mat4)(v, *v, m); }
        inline void dispatch_transform(Vec4* v, const Mat4& m) { WMATH_CALL(Vec4, transform_mat4)(v, *v, m); }

        // Rotate X/Y/Z (Quat/Mat3/Mat4 <- float)
        inline void dispatch_rotate_x(Quat* q, float rad) { WMATH_ROTATE_X(Quat)(q, *q, rad); }
        inline void dispatch_rotate_x(Mat3* m, float rad) { WMATH_ROTATE_X(Mat3)(m, *m, rad); }
        inline void dispatch_rotate_x(Mat4* m, float rad) { WMATH_ROTATE_X(Mat4)(m, *m, rad); }
        
        inline void dispatch_rotate_y(Quat* q, float rad) { WMATH_ROTATE_Y(Quat)(q, *q, rad); }
        inline void dispatch_rotate_y(Mat3* m, float rad) { WMATH_ROTATE_Y(Mat3)(m, *m, rad); }
        inline void dispatch_rotate_y(Mat4* m, float rad) { WMATH_ROTATE_Y(Mat4)(m, *m, rad); }
        
        inline void dispatch_rotate_z(Quat* q, float rad) { WMATH_ROTATE_Z(Quat)(q, *q, rad); }
        inline void dispatch_rotate_z(Mat3* m, float rad) { WMATH_ROTATE_Z(Mat3)(m, *m, rad); }
        inline void dispatch_rotate_z(Mat4* m, float rad) { WMATH_ROTATE_Z(Mat4)(m, *m, rad); }

        // Scale (Mat3/Mat4 <- Vec)
        // Note: Generic `dispatch_mul_s` handles scalar scaling for vectors/matrices.
        // These handle non-uniform scaling via vectors.
        inline void dispatch_scale(Mat3* m, const Vec2& v) { WMATH_SCALE(Mat3)(m, *m, v); }
        inline void dispatch_scale(Mat3* m, const Vec3& v) { WMATH_CALL(Mat3, scale3D)(m, *m, v); }
        inline void dispatch_scale(Mat4* m, const Vec3& v) { WMATH_SCALE(Mat4)(m, *m, v); }

        // Uniform Scale (Mat3/Mat4 <- float)
        inline void dispatch_uniform_scale(Mat3* m, float s) { WMATH_CALL(Mat3, uniform_scale)(m, *m, s); }
        inline void dispatch_uniform_scale(Mat4* m, float s) { WMATH_CALL(Mat4, uniform_scale)(m, *m, s); }

        // Get Translation
        inline void dispatch_get_translation(Vec2* v, const Mat3& m) { WMATH_GET_TRANSLATION(Mat3)(v, m); }
        inline void dispatch_get_translation(Vec3* v, const Mat4& m) { WMATH_GET_TRANSLATION(Mat4)(v, m); }

        // Get Axis 
        inline void dispatch_get_axis(Vec2* v, const Mat3& m, int axis) { WMATH_CALL(Mat3, get_axis)(v, m, axis); }
        inline void dispatch_get_axis(Vec3* v, const Mat4& m, int axis) { WMATH_CALL(Mat4, get_axis)(v, m, axis); }

        // Get Scaling
        inline void dispatch_get_scaling(Vec2* v, const Mat3& m) { WMATH_CALL(Mat3, get_scaling)(v, m); }
        inline void dispatch_get_scaling(Vec3* v, const Mat4& m) { WMATH_CALL(Mat4, get_scaling)(v, m); }

        // To Mat3 
        inline void dispatch_to_mat3(Mat3* m, const Quat& q) { WMATH_CALL(Mat3, from_quat)(m, q); }
        inline void dispatch_to_mat3(Mat3* m, const Mat4& m4) { WMATH_CALL(Mat3, from_mat4)(m, m4); }

        // To Mat4
        inline void dispatch_to_mat4(Mat4* m, const Quat& q) { WMATH_CALL(Mat4, from_quat)(m, q); }
        inline void dispatch_to_mat4(Mat4* m, const Mat3& m3) { WMATH_CALL(Mat4, from_mat3)(m, m3); }

        // To Quat
        inline void dispatch_to_quat(Quat* q, const Mat4& m) { WMATH_CALL(Quat, from_mat4)(q, m); }
        inline void dispatch_to_quat(Quat* q, const Mat3& m) { WMATH_CALL(Quat, from_mat3)(q, m); }

        #undef DEF_DISPATCH_VOID
        #undef DEF_DISPATCH_UNARY
        #undef DEF_DISPATCH_SCALAR
        #undef DEF_DISPATCH_BINARY
        #undef DEF_DISPATCH_CLAMP
    }

    // --- Pipe Creation ---
    template <typename T>
    inline Internal::Pipe<T> pipe(T& target) { return { &target }; }

    template <typename T>
    inline Internal::Pipe<T> pipe(T* target) { return { target }; }

    // --- Pipe Operators ---
    
    // Pipe | Operation (Execute)
    template <typename T, typename Func>
    inline Internal::Pipe<T> operator|(Internal::Pipe<T> p, const Internal::Operation<Func>& op) {
        op.func(p.dst); 
        return p;
    }

    // Pipe |= Operation (Execute, nicer syntax for init)
    template <typename T, typename Func>
    inline Internal::Pipe<T> operator|=(Internal::Pipe<T> p, const Internal::Operation<Func>& op) {
        op.func(p.dst);
        return p;
    }

    // Operation | Operation (Compose)
    template <typename FuncA, typename FuncB>
    inline auto operator|(const Internal::Operation<FuncA>& opA, const Internal::Operation<FuncB>& opB) {
        return Internal::make_op([=](auto* dst) {
            opA.func(dst);
            opB.func(dst);
        });
    }

    // =========================================================================
    // 2. Operations (Ops Namespace)
    // =========================================================================
    namespace Index {
        static constexpr struct X {} x{};
        static constexpr struct Y {} y{};
        static constexpr struct Z {} z{};
        static constexpr struct W {} w{};
        template<int R, int C>
        struct M3x3 {
            static_assert(R >= 0 && R <= 2, "R must be in [0, 2]");
            static_assert(C >= 0 && C <= 2, "C must be in [0, 2]");
        };
        template<int R, int C>
        struct M4x4 {
            static_assert(R >= 0 && R <= 3, "R must be in [0, 3]");
            static_assert(C >= 0 && C <= 3, "C must be in [0, 3]");
        };

        // vec2
        inline float operator|(const Vec2& v, const Index::X&) { return v.v[0]; }
        inline float operator|(const Vec2& v, const Index::Y&) { return v.v[1]; }

        // vec3
        inline float operator|(const Vec3& v, const Index::X&) { return v.v[0]; }
        inline float operator|(const Vec3& v, const Index::Y&) { return v.v[1]; }
        inline float operator|(const Vec3& v, const Index::Z&) { return v.v[2]; }

        // vec4
        inline float operator|(const Vec4& v, const Index::X&) { return v.v[0]; }
        inline float operator|(const Vec4& v, const Index::Y&) { return v.v[1]; }
        inline float operator|(const Vec4& v, const Index::Z&) { return v.v[2]; }
        inline float operator|(const Vec4& v, const Index::W&) { return v.v[3]; }

        // Quat
        inline float operator|(const Quat& q, const Index::X&) { return q.v[0]; }
        inline float operator|(const Quat& q, const Index::Y&) { return q.v[1]; }
        inline float operator|(const Quat& q, const Index::Z&) { return q.v[2]; }
        inline float operator|(const Quat& q, const Index::W&) { return q.v[3]; }

        // Mat3
        template <int Row, int Col>
        
        inline float operator|(const Mat3& k, const Index::M3x3<Row, Col>&) {
            return k.m[Row * 4 + Col];
        }

        // Mat4
        template <int Row, int Col>
        inline float operator|(const Mat4& k, const Index::M4x4<Row, Col>&) {
            return k.m[Row * 4 + Col];
        }
    }
    namespace Ops {
        
        // ---------------------------------------------------------------------
        // Generic / Common Ops (Work on multiple types)
        // ---------------------------------------------------------------------
        inline auto identity()  { return Internal::make_op([](auto* d){ Internal::dispatch_identity(d); }); }
        inline auto zero()      { return Internal::make_op([](auto* d){ Internal::dispatch_zero(d); }); }
        inline auto normalize() { return Internal::make_op([](auto* d){ Internal::dispatch_normalize(d); }); }
        inline auto negate()    { return Internal::make_op([](auto* d){ Internal::dispatch_negate(d); }); }
        inline auto invert()    { return Internal::make_op([](auto* d){ Internal::dispatch_invert(d); }); } // Vec(Inverse component), Mat/Quat(Inverse)
        inline auto transpose() { return Internal::make_op([](auto* d){ Internal::dispatch_transpose(d); }); }
        
        inline auto ceil()      { return Internal::make_op([](auto* d){ Internal::dispatch_ceil(d); }); }
        inline auto floor()     { return Internal::make_op([](auto* d){ Internal::dispatch_floor(d); }); }
        inline auto round()     { return Internal::make_op([](auto* d){ Internal::dispatch_round(d); }); }
        inline auto clamp(float min, float max) { 
            return Internal::make_op([=](auto* d){ Internal::dispatch_clamp(d, min, max); }); 
        }

        // Scalar arithmetic
        inline auto mul(float s)   { return Internal::make_op([=](auto* d){ Internal::dispatch_mul_s(d, s); }); }
        inline auto div(float s)   { return Internal::make_op([=](auto* d){ Internal::dispatch_div_s(d, s); }); }
        // Alias for mul(scalar)
        inline auto scale(float s) { return Internal::make_op([=](auto* d){ Internal::dispatch_mul_s(d, s); }); } 

        // ---------------------------------------------------------------------
        // Vector Specific Ops (Vec2, Vec3, Vec4)
        // ---------------------------------------------------------------------
        
        // Creation
        inline auto make_vec2(float x, float y) { return Internal::make_op([=](Vec2* d){ WMATH_CREATE(Vec2)(d, {x,y}); }); }
        inline auto make_vec3(float x, float y, float z) { return Internal::make_op([=](Vec3* d){ WMATH_CREATE(Vec3)(d, {x,y,z}); }); }
        inline auto make_vec4(float x, float y, float z, float w) { return Internal::make_op([=](Vec4* d){ WMATH_CREATE(Vec4)(d, {x,y,z,w}); }); }
        
        // Binary Ops (add, sub, mul(comp), div(comp))
        // 使用模板 lambda 自动匹配 dispatch_add 等
        #define W_OP_GENERIC_BINARY(NAME, DISPATCH_NAME) \
            inline auto NAME(const Vec2& b) { return Internal::make_op([=](Vec2* d){ Internal::DISPATCH_NAME(d, b); }); } \
            inline auto NAME(const Vec3& b) { return Internal::make_op([=](Vec3* d){ Internal::DISPATCH_NAME(d, b); }); } \
            inline auto NAME(const Vec4& b) { return Internal::make_op([=](Vec4* d){ Internal::DISPATCH_NAME(d, b); }); }
        
        W_OP_GENERIC_BINARY(add, dispatch_add)
        W_OP_GENERIC_BINARY(sub, dispatch_sub)
        W_OP_GENERIC_BINARY(mul, dispatch_mul) // Component-wise multiply for Vectors, Matrix multiply for Matrices
        W_OP_GENERIC_BINARY(div, dispatch_div) // Component-wise divide
        W_OP_GENERIC_BINARY(max, dispatch_fmax)
        W_OP_GENERIC_BINARY(min, dispatch_fmin)
        W_OP_GENERIC_BINARY(midpoint, dispatch_midpoint)
        #undef W_OP_GENERIC_BINARY

        // Additional Vector Ops
        inline auto set_length(float len) { return Internal::make_op([=](auto* d){ Internal::dispatch_set_len(d, len); }); }
        inline auto truncate(float max_len) { return Internal::make_op([=](auto* d){ Internal::dispatch_truncate(d, max_len); }); }

        // Lerp
        inline auto lerp(Vec2 b, float t) { return Internal::make_op([=](Vec2* d){ WMATH_LERP(Vec2)(d, *d, b, t); }); }
        inline auto lerp(Vec3 b, float t) { return Internal::make_op([=](Vec3* d){ WMATH_LERP(Vec3)(d, *d, b, t); }); }
        inline auto lerp(Vec4 b, float t) { return Internal::make_op([=](Vec4* d){ WMATH_LERP(Vec4)(d, *d, b, t); }); }
        // Lerp Vector T
        inline auto lerp(Vec2 b, Vec2 t) { return Internal::make_op([=](Vec2* d){ WMATH_LERP_V(Vec2)(d, *d, b, t); }); }
        inline auto lerp(Vec3 b, Vec3 t) { return Internal::make_op([=](Vec3* d){ WMATH_LERP_V(Vec3)(d, *d, b, t); }); }
        inline auto lerp(Vec4 b, Vec4 t) { return Internal::make_op([=](Vec4* d){ WMATH_LERP_V(Vec4)(d, *d, b, t); }); }

        // Vec3 Special
        inline auto cross(Vec3 b) { return Internal::make_op([=](Vec3* d){ Internal::dispatch_cross(d, b); }); }
        
        // Transform Application (Applies Matrix m to Vector d)
        // Solved Ambiguity: Uses generic lambda to dispatch based on type of 'd'
        inline auto transform(const Mat3& m) { return Internal::make_op([=](auto* d){ Internal::dispatch_transform(d, m); }); }
        inline auto transform(const Mat4& m) { return Internal::make_op([=](auto* d){ Internal::dispatch_transform(d, m); }); }
        
        inline auto transform(const Quat& q) { return Internal::make_op([=](Vec3* d){ WMATH_CALL(Vec3, transform_quat)(d, *d, q); }); }
        inline auto transform_normal(const Mat4& m) { return Internal::make_op([=](Vec3* d){ WMATH_CALL(Vec3, transform_mat4_upper3x3)(d, *d, m); }); }

        // ---------------------------------------------------------------------
        // Quaternion Ops
        // ---------------------------------------------------------------------
        inline auto make_quat(float x, float y, float z, float w) { return Internal::make_op([=](Quat* d){ WMATH_CREATE(Quat)(d, {x,y,z,w}); }); }
        
        inline auto mul(const Quat& b) { return Internal::make_op([=](Quat* d){ Internal::dispatch_mul(d, b); }); }
        
        // Ambiguity Fix: rotate_x/y/z for Quat/Mat3/Mat4 merged into generic ops
        inline auto rotate_x(float rad) { return Internal::make_op([=](auto* d){ Internal::dispatch_rotate_x(d, rad); }); }
        inline auto rotate_y(float rad) { return Internal::make_op([=](auto* d){ Internal::dispatch_rotate_y(d, rad); }); }
        inline auto rotate_z(float rad) { return Internal::make_op([=](auto* d){ Internal::dispatch_rotate_z(d, rad); }); }
        
        inline auto conjugate() { return Internal::make_op([](Quat* d){ WMATH_CALL(Quat, conjugate)(d, *d); }); }
        
        inline auto slerp(Quat b, float t) { return Internal::make_op([=](Quat* d){ WMATH_CALL(Quat, slerp)(d, *d, b, t); }); }
        inline auto sqlerp(Quat b, Quat c, Quat d_val, float t) { 
            return Internal::make_op([=](Quat* d){ WMATH_CALL(Quat, sqlerp)(d, *d, b, c, d_val, t); }); 
        }

        inline auto from_euler(float x, float y, float z, RotationOrder order = RotationOrder::XYZ) {
            return Internal::make_op([=](Quat* d){ WMATH_CALL(Quat, from_euler)(d, x, y, z, static_cast<enum WCN_Math_RotationOrder>(order)); });
        }
        inline auto from_axis_angle(Vec3 axis, float rad) { return Internal::make_op([=](Quat* d){ WMATH_CALL(Quat, from_axis_angle)(d, axis, rad); }); }
        inline auto rotation_to(Vec3 from, Vec3 to) { return Internal::make_op([=](Quat* d){ WMATH_CALL(Quat, rotation_to)(d, from, to); }); }
        
        // ---------------------------------------------------------------------
        // Matrix Ops (Mat3 / Mat4)
        // ---------------------------------------------------------------------
        inline auto mul(const Mat3& b) { return Internal::make_op([=](Mat3* d){ Internal::dispatch_mul(d, b); }); }
        inline auto mul(const Mat4& b) { return Internal::make_op([=](Mat4* d){ Internal::dispatch_mul(d, b); }); }
        
        inline auto add(const Mat3& b) { return Internal::make_op([=](Mat3* d){ Internal::dispatch_add(d, b); }); }
        inline auto add(const Mat4& b) { return Internal::make_op([=](Mat4* d){ Internal::dispatch_add(d, b); }); }
        
        inline auto sub(const Mat3& b) { return Internal::make_op([=](Mat3* d){ Internal::dispatch_sub(d, b); }); }
        inline auto sub(const Mat4& b) { return Internal::make_op([=](Mat4* d){ Internal::dispatch_sub(d, b); }); }

        // Modifiers
        inline auto translate(Vec2 v)       { return Internal::make_op([=](Mat3* d){ WMATH_CALL(Mat3, translate)(d, *d, v); }); }
        inline auto translate(Vec3 v)       { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, translate)(d, *d, v); }); }
        
        inline auto rotate(float rad)       { return Internal::make_op([=](Mat3* d){ WMATH_ROTATE(Mat3)(d, *d, rad); }); }
        inline auto rotate(Vec3 axis, float r) { return Internal::make_op([=](Mat4* d){ WMATH_ROTATE(Mat4)(d, *d, axis, r); }); }
        inline auto axis_rotate(Vec3 axis, float r) { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, axis_rotate)(d, *d, axis, r); }); }
        
        // Scaling (Ambiguity Fix)
        inline auto scale(Vec2 v)           { return Internal::make_op([=](auto* d){ Internal::dispatch_scale(d, v); }); }
        inline auto scale(Vec3 v)           { return Internal::make_op([=](auto* d){ Internal::dispatch_scale(d, v); }); }
        inline auto uniform_scale(float s)  { return Internal::make_op([=](auto* d){ Internal::dispatch_uniform_scale(d, s); }); }
        
        // Setters
        inline auto set_translation(Vec2 v) { return Internal::make_op([=](Mat3* d){ WMATH_SET_TRANSLATION(Mat3)(d, *d, v); }); }
        inline auto set_translation(Vec3 v) { return Internal::make_op([=](Mat4* d){ WMATH_SET_TRANSLATION(Mat4)(d, *d, v); }); }
        
        inline auto set_axis(int axis, Vec2 v) { return Internal::make_op([=](Mat3* d){ WMATH_CALL(Mat3, set_axis)(d, *d, v, axis); }); }
        inline auto set_axis(int axis, Vec3 v) { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, set_axis)(d, *d, v, axis); }); }

        // Factories (Overwrite current)
        inline auto make_translation(Vec2 v) { return Internal::make_op([=](Mat3* d){ WMATH_TRANSLATION(Mat3)(d, v); }); }
        inline auto make_rotation(float r)   { return Internal::make_op([=](Mat3* d){ WMATH_ROTATION(Mat3)(d, r); }); }
        inline auto make_scaling(Vec2 v)     { return Internal::make_op([=](Mat3* d){ WMATH_CALL(Mat3, scaling)(d, v); }); }

        // Camera / View Factories (Overwrite current)
        inline auto look_at(Vec3 eye, Vec3 target, Vec3 up) { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, look_at)(d, eye, target, up); }); }
        inline auto aim(Vec3 pos, Vec3 target, Vec3 up)     { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, aim)(d, pos, target, up); }); }
        inline auto camera_aim(Vec3 eye, Vec3 target, Vec3 up) { return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, camera_aim)(d, eye, target, up); }); }
        
        // Projection Factories (Overwrite current)
        inline auto ortho(float l, float r, float b, float t, float n, float f) {
            return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, ortho)(d, l, r, b, t, n, f); });
        }
        inline auto frustum(float l, float r, float b, float t, float n, float f) {
            return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, frustum)(d, l, r, b, t, n, f); });
        }
        inline auto frustum_reverse_z(float l, float r, float b, float t, float n, float f) {
            return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, frustum_reverse_z)(d, l, r, b, t, n, f); });
        }
        inline auto perspective(float fov, float aspect, float n, float f) {
            return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, perspective)(d, fov, aspect, n, f); });
        }
        inline auto perspective_reverse_z(float fov, float aspect, float n, float f) {
            return Internal::make_op([=](Mat4* d){ WMATH_CALL(Mat4, perspective_reverse_z)(d, fov, aspect, n, f); });
        }

        // Get Translation
        inline auto get_translation(const Mat3& m) {
            return Internal::make_op([=](Vec2* d){ Internal::dispatch_get_translation(d, m); });
        }
        inline auto get_translation(const Mat4& m) {
            return Internal::make_op([=](Vec3* d){ Internal::dispatch_get_translation(d, m); });
        }

        // Get Axis
        inline auto get_axis(const Mat3& m, int axis) {
            return Internal::make_op([=](Vec2* d){ Internal::dispatch_get_axis(d, m, axis); });
        }
        inline auto get_axis(const Mat4& m, int axis) {
            return Internal::make_op([=](Vec3* d){ Internal::dispatch_get_axis(d, m, axis); });
        }

        // Get Scaling
        inline auto get_scaling(const Mat3& m) {
            return Internal::make_op([=](Vec2* d){ Internal::dispatch_get_scaling(d, m); });
        }
        inline auto get_scaling(const Mat4& m) {
            return Internal::make_op([=](Vec3* d){ Internal::dispatch_get_scaling(d, m); });
        }

        // To Mat3 
        inline auto to_mat3(const Mat4& m) {
            return Internal::make_op([=](Mat3* d){ Internal::dispatch_to_mat3(d, m); });
        }
        inline auto to_mat3(const Quat& q) {
            return Internal::make_op([=](Mat3* d){ Internal::dispatch_to_mat3(d, q); });
        }

        // To Mat4
        inline auto to_mat4(const Mat3& m) {
            return Internal::make_op([=](Mat4* d){ Internal::dispatch_to_mat4(d, m); });    
        }
        inline auto to_mat4(const Quat& q) {
            return Internal::make_op([=](Mat4* d){ Internal::dispatch_to_mat4(d, q); });
        }
        
        // To Quat 
        inline auto to_quat(const Mat3& m) {
            return Internal::make_op([=](Quat* d){ Internal::dispatch_to_quat(d, m); });
        }
        inline auto to_quat(const Mat4& m) {
            return Internal::make_op([=](Quat* d){ Internal::dispatch_to_quat(d, m); });
        }

        // set index
        
        template<typename T,
         typename = typename std::enable_if<
             std::is_same<T, Vec2>::value ||
             std::is_same<T, Vec3>::value ||
             std::is_same<T, Vec4>::value ||
             std::is_same<T, Quat>::value
         >::type>
        inline auto set_x(const float x) noexcept {
            return Internal::make_op([=](T* d){ d->v[0] = x; });
        }

        template<typename T,
         typename = typename std::enable_if<
             std::is_same<T, Vec2>::value ||
             std::is_same<T, Vec3>::value ||
             std::is_same<T, Vec4>::value ||
             std::is_same<T, Quat>::value
         >::type>
        inline auto set_y(const float y) noexcept {
            return Internal::make_op([=](T* d){ d->v[1] = y; });
        }

        template<typename T,
         typename = typename std::enable_if<
             std::is_same<T, Vec3>::value ||
             std::is_same<T, Vec4>::value ||
             std::is_same<T, Quat>::value
         >::type>
        inline auto set_z(const float z) noexcept {
            return Internal::make_op([=](T* d){ d->v[2] = z; });
        }

        template<typename T,
         typename = typename std::enable_if<
             std::is_same<T, Vec4>::value ||
             std::is_same<T, Quat>::value
         >::type>
        inline auto set_w(const float w) noexcept {
            return Internal::make_op([=](T* d){ d->v[2] = w; });
        }

        template<int R, int C>
        inline auto set_index(Index::M3x3<R, C> idx, float value) noexcept {
            return Internal::make_op([=](Mat3* d){ d->m[R * 4 + C] = value; });
        }
        template<int R, int C>
        inline auto set_index(Index::M4x4<R, C> idx, float value) noexcept {
            return Internal::make_op([=](Mat4* d){ d->m[R * 4 + C] = value; });
        }
    } // namespace Ops

    // =========================================================================
    // 3. Standalone Utils (No Pipe)
    // =========================================================================
    
    // Config
    inline float get_epsilon() { return wcn_math_get_epsilon(); }
    inline float set_epsilon(float e) { return wcn_math_set_epsilon(e); }
    static constexpr float PI = WMATH_PI;

    // Dot Product
    inline float dot(Vec2 a, Vec2 b) { return WMATH_DOT(Vec2)(a, b); }
    inline float dot(Vec3 a, Vec3 b) { return WMATH_DOT(Vec3)(a, b); }
    inline float dot(Vec4 a, Vec4 b) { return WMATH_DOT(Vec4)(a, b); }
    inline float dot(Quat a, Quat b) { return WMATH_DOT(Quat)(a, b); }

    // Length / Distance
    inline float length(Vec2 a) { return WMATH_LENGTH(Vec2)(a); }
    inline float length(Vec3 a) { return WMATH_LENGTH(Vec3)(a); }
    inline float length(Vec4 a) { return WMATH_LENGTH(Vec4)(a); }
    inline float length(Quat a) { return WMATH_LENGTH(Quat)(a); }
    
    inline float length_sq(Vec2 a) { return WMATH_LENGTH_SQ(Vec2)(a); }
    inline float length_sq(Vec3 a) { return WMATH_LENGTH_SQ(Vec3)(a); }
    inline float length_sq(Vec4 a) { return WMATH_LENGTH_SQ(Vec4)(a); }
    inline float length_sq(Quat a) { return WMATH_LENGTH_SQ(Quat)(a); }

    inline float distance(Vec2 a, Vec2 b) { return WMATH_DISTANCE(Vec2)(a, b); }
    inline float distance(Vec3 a, Vec3 b) { return WMATH_DISTANCE(Vec3)(a, b); }
    inline float distance(Vec4 a, Vec4 b) { return WMATH_DISTANCE(Vec4)(a, b); }
    
    inline float distance_sq(Vec2 a, Vec2 b) { return WMATH_DISTANCE_SQ(Vec2)(a, b); }
    inline float distance_sq(Vec3 a, Vec3 b) { return WMATH_DISTANCE_SQ(Vec3)(a, b); }
    inline float distance_sq(Vec4 a, Vec4 b) { return WMATH_DISTANCE_SQ(Vec4)(a, b); }

    inline float angle(Vec2 a, Vec2 b) { return WMATH_ANGLE(Vec2)(a, b); }
    inline float angle(Vec3 a, Vec3 b) { return WMATH_ANGLE(Vec3)(a, b); }
    inline float angle(Quat a, Quat b) { return WMATH_ANGLE(Quat)(a, b); }

    // Comparisons
    inline bool operator==(Vec2 a, Vec2 b) { return WMATH_EQUALS(Vec2)(a, b); }
    inline bool operator==(Vec3 a, Vec3 b) { return WMATH_EQUALS(Vec3)(a, b); }
    inline bool operator==(Vec4 a, Vec4 b) { return WMATH_EQUALS(Vec4)(a, b); }
    inline bool operator==(Mat3 a, Mat3 b) { return WMATH_EQUALS(Mat3)(a, b); }
    inline bool operator==(Mat4 a, Mat4 b) { return WMATH_EQUALS(Mat4)(a, b); }
    inline bool operator==(Quat a, Quat b) { return WMATH_EQUALS(Quat)(a, b); }

    inline bool equals_approx(Vec2 a, Vec2 b) { return WMATH_EQUALS_APPROXIMATELY(Vec2)(a, b); }
    inline bool equals_approx(Vec3 a, Vec3 b) { return WMATH_EQUALS_APPROXIMATELY(Vec3)(a, b); }
    inline bool equals_approx(Vec4 a, Vec4 b) { return WMATH_EQUALS_APPROXIMATELY(Vec4)(a, b); }
    inline bool equals_approx(Mat3 a, Mat3 b) { return WMATH_EQUALS_APPROXIMATELY(Mat3)(a, b); }
    inline bool equals_approx(Mat4 a, Mat4 b) { return WMATH_EQUALS_APPROXIMATELY(Mat4)(a, b); }
    inline bool equals_approx(Quat a, Quat b) { return WMATH_EQUALS_APPROXIMATELY(Quat)(a, b); }

    // Matrix Utils
    inline float determinant(const Mat3& m) { return WMATH_DETERMINANT(Mat3)(m); }
    inline float determinant(const Mat4& m) { return WMATH_DETERMINANT(Mat4)(m); }
    
    } // namespace Math
} // namespace WCN