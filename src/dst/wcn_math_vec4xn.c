#include "WCN/WCN_MATH_SOA.h"
#include "common/wcn_math_internal.h"

void wcn_math_Vec4xN_free(WMATH_TYPE(Vec4xN)* vec4xN) {
    if (vec4xN == nullptr) return;
    if (vec4xN->x) { wcn_aligned_free(vec4xN->x); vec4xN->x = nullptr; }
    if (vec4xN->y) { wcn_aligned_free(vec4xN->y); vec4xN->y = nullptr; }
    if (vec4xN->z) { wcn_aligned_free(vec4xN->z); vec4xN->z = nullptr; }
    if (vec4xN->w) { wcn_aligned_free(vec4xN->w); vec4xN->w = nullptr; }
    vec4xN->count = 0;
    free(vec4xN);
}