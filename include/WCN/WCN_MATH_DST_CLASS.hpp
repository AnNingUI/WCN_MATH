#pragma once
#include "WCN/WCN_MATH_CPP_COMMON.hpp"
#include "WCN/WCN_MATH_DST.h"
// #include <iostream>

namespace WCN {
    inline namespace Math {
        class z_vec2 {
            private:
                T$(Vec2) value;
            public:
                z_vec2() {
                    wcn_math_Vec2_zero(&value);
                };
                z_vec2(float x, float y) {
                    value = T$(Vec2){
                        .v = {x, y}
                    };
                }
                z_vec2(const z_vec2& other) = default;
                // copy 
                z_vec2(z_vec2&& other) {
                    wcn_math_Vec2_copy(&value, other.value);
                };
                z_vec2& operator=(const z_vec2& other) = default;
                
                ~z_vec2() = default;
            
                inline T$(Vec2) get() const {
                    return value;
                }
                inline T$(Vec2)* get_mut_ref() {
                    return &value;
                }
                inline float x() const {
                    return value.v[0];
                }
                inline float y() const {
                    return value.v[1];
                }
                inline auto operator+(const z_vec2& other) const noexcept {
                    z_vec2 result = *this;
                    wcn_math_Vec2_add(&result.value, result.value, other.get());
                    return result;
                }
                inline auto operator+(const float other) const noexcept {
                    z_vec2 result = *this;
                    result.value.v[0] += other;
                    result.value.v[1] += other;
                    return result;
                }
                inline auto operator+=(const z_vec2& other) noexcept {
                    wcn_math_Vec2_add(&value, value, other.get());
                    return this;
                }
                inline auto operator+=(const float other) noexcept {
                    value.v[0] += other;
                    value.v[1] += other;
                    return this;
                }
                inline auto add_scaled(const z_vec2& other, float scale) const noexcept {
                    z_vec2 result = *this;
                    wcn_math_Vec2_add_scaled(&result.value, result.value, other.get(), scale);
                    return result;
                }
                inline auto add_scaled_self(const z_vec2& other, float scale) noexcept {
                    wcn_math_Vec2_add_scaled(&value, value, other.get(), scale);
                    return this;
                }
        };
    }
}

