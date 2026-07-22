#ifndef WCN_MATH_IMAGE_H
#define WCN_MATH_IMAGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <WCN/WCN_MATH_TYPES.h>
#include <WCN/WCN_MATH_SOA.h>
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// Memory Management Functions
// ============================================================================

/**
 * @brief Free ImageBitmap data (caller must call this to release memory)
 * @param image Pointer to ImageBitmap structure
 * @note After calling this function, image->data will be set to NULL
 */
void wcn_math_ImageBitmap_free(WMATH_TYPE(ImageBitmap)* image);

/**
 * @brief Free ImageRGBA arrays (caller must call this to release memory)
 * @param image Pointer to ImageRGBA structure
 * @note After calling this function, all array pointers will be set to NULL
 */
void wcn_math_ImageRGBA_free(WMATH_TYPE(ImageRGBA)* image);

// ============================================================================
// Conversion Functions: Vec4xN (SoA) <-> ImageBitmap (packed RGBA)
// ============================================================================

/**
 * @brief Convert Vec4xN (SoA) to ImageBitmap (packed RGBA)
 * @param image Output ImageBitmap (caller must free with wcn_math_ImageBitmap_free)
 * @param vec4xN Input Vec4xN (x=R, y=G, z=B, w=A)
 * @param width Image width
 * @param height Image height
 * @note SIMD-accelerated with scalar fallback for remaining pixels
 * @note Rounding: round-half-up (consistent across all platforms)
 */
void wcn_math_Vec4xN_to_ImageBitmap(
    WMATH_TYPE(ImageBitmap)* image,
    const WMATH_TYPE(Vec4xN)* vec4xN,
    size_t width,
    size_t height
);

/**
 * @brief Convert ImageBitmap (packed RGBA) to Vec4xN (SoA)
 * @param vec4xN Output Vec4xN (caller must free with wcn_math_Vec4xN_free)
 * @param image Input ImageBitmap
 * @param width Image width
 * @param height Image height
 * @note SIMD-accelerated with scalar fallback for remaining pixels
 */
void wcn_math_ImageBitmap_to_Vec4xN(
    WMATH_TYPE(Vec4xN)* vec4xN,
    const WMATH_TYPE(ImageBitmap)* image,
    size_t width,
    size_t height
);

// ============================================================================
// Conversion Functions: ImageRGBA (SoA) <-> ImageBitmap (packed RGBA)
// ============================================================================

/**
 * @brief Convert ImageRGBA (SoA) to ImageBitmap (packed RGBA)
 * @param image Output ImageBitmap (caller must free with wcn_math_ImageBitmap_free)
 * @param image_rgba Input ImageRGBA
 * @param width Image width
 * @param height Image height
 * @note SIMD-accelerated with scalar fallback for remaining pixels
 * @note Rounding: round-half-up (consistent across all platforms)
 */
void wcn_math_ImageRGBA_to_ImageBitmap(
    WMATH_TYPE(ImageBitmap)* image,
    const WMATH_TYPE(ImageRGBA)* image_rgba,
    size_t width,
    size_t height
);

/**
 * @brief Convert ImageBitmap (packed RGBA) to ImageRGBA (SoA)
 * @param image_rgba Output ImageRGBA (caller must free with wcn_math_ImageRGBA_free)
 * @param image_bitmap Input ImageBitmap
 * @param width Image width
 * @param height Image height
 * @note SIMD-accelerated with scalar fallback for remaining pixels
 */
void wcn_math_ImageBitmap_to_ImageRGBA(
    WMATH_TYPE(ImageRGBA)* image_rgba,
    const WMATH_TYPE(ImageBitmap)* image_bitmap,
    size_t width,
    size_t height
);

// ============================================================================
// SIMD Kernels (for unit testing and verification)
// ============================================================================

/**
 * @brief SIMD kernel: convert 4-channel SoA floats to packed RGBA uint8
 * @param dst_rgba Output RGBA data (16 bytes for 4 pixels)
 * @param src_r Input R channel (4 floats)
 * @param src_g Input G channel (4 floats)
 * @param src_b Input B channel (4 floats)
 * @param src_a Input A channel (4 floats)
 * @note Rounding: round-half-up (add 0.5 before truncation)
 * @note Clamping: values are clamped to [0, 1] before conversion
 */
void wcn_math_simd_soa_to_rgba_u8(
    uint8_t dst_rgba[16],
    const float src_r[4],
    const float src_g[4],
    const float src_b[4],
    const float src_a[4]
);

/**
 * @brief SIMD kernel: convert packed RGBA uint8 to 4-channel SoA floats
 * @param dst_r Output R channel (4 floats)
 * @param dst_g Output G channel (4 floats)
 * @param dst_b Output B channel (4 floats)
 * @param dst_a Output A channel (4 floats)
 * @param src_rgba Input RGBA data (16 bytes for 4 pixels)
 */
void wcn_math_simd_rgba_to_soa_u8(
    float dst_r[4],
    float dst_g[4],
    float dst_b[4],
    float dst_a[4],
    const uint8_t src_rgba[16]
);

// ============================================================================
// Zero-copy Conversion Functions: Vec4xN <-> ImageRGBA (compatible layouts)
// ============================================================================

inline void wcn_math_Vec4xN_ref_to_ImageRGBA(
    WMATH_TYPE(ImageRGBA)* image,
    const WMATH_TYPE(Vec4xN)* vec4xN,
    const size_t width,
    const size_t height
) {
  image->r = vec4xN->x;
  image->g = vec4xN->y;
  image->b = vec4xN->z;
  image->a = vec4xN->w;
  image->width  = width;
  image->height = height;
}

inline void wcn_math_ImageRGBA_ref_to_Vec4xN(
    WMATH_TYPE(Vec4xN)* vec4xN,
    const WMATH_TYPE(ImageRGBA)* image,
    const size_t width,
    const size_t height
) {
  vec4xN->x = image->r;
  vec4xN->y = image->g;
  vec4xN->z = image->b;
  vec4xN->w = image->a;
  vec4xN->count = width * height;
}

#ifdef __cplusplus
}
#endif

#endif // WCN_MATH_IMAGE_H