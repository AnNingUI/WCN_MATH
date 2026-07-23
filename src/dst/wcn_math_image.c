#include "WCN/WCN_MATH_TYPES.h"
#include "../common/wcn_math_internal.h"
#include <math.h>
#include <string.h>

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
#include <emmintrin.h>
#include <immintrin.h>
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
#include <arm_neon.h>
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
#include <wasm_simd128.h>
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
#include <lsxintrin.h>
#endif

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
#include <riscv_vector.h>
#endif

// Helper: clamp float to [0, 1] and convert to uint8_t [0, 255]
static inline uint8_t float_to_u8_clamped(float f) {
    if (f <= 0.0f) return 0;
    if (f >= 1.0f) return 255;
    return (uint8_t)(f * 255.0f + 0.5f); // round to nearest
}

// Helper: convert uint8_t [0, 255] to float [0, 1]
static inline float u8_to_float(uint8_t u) {
    return (float)u / 255.0f;
}

void wcn_math_Vec4xN_to_ImageBitmap(
    WMATH_TYPE(ImageBitmap)* image,
    const WMATH_TYPE(Vec4xN)* vec4xN,
    const size_t width,
    const size_t height
) {
    if (!image || !vec4xN || width == 0 || height == 0) {
        return;
    }

    const size_t total_pixels = width * height;
    if (vec4xN->count < total_pixels) {
        return;
    }

    image->width = width;
    image->height = height;

    // Allocate bitmap data (4 bytes per pixel: RGBA)
    image->data = (uint8_t*)malloc(total_pixels * 4);
    if (!image->data) {
        return;
    }

    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE - process 4 pixels at a time
    const __m128 scale = _mm_set1_ps(255.0f);
    const __m128 half = _mm_set1_ps(0.5f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load 4 color values for each channel
        __m128 x = _mm_loadu_ps(&vec4xN->x[i]); // R
        __m128 y = _mm_loadu_ps(&vec4xN->y[i]); // G
        __m128 z = _mm_loadu_ps(&vec4xN->z[i]); // B
        __m128 w = _mm_loadu_ps(&vec4xN->w[i]); // A

        // Clamp to [0, 1]
        x = _mm_max_ps(x, _mm_setzero_ps());
        y = _mm_max_ps(y, _mm_setzero_ps());
        z = _mm_max_ps(z, _mm_setzero_ps());
        w = _mm_max_ps(w, _mm_setzero_ps());

        x = _mm_min_ps(x, _mm_set1_ps(1.0f));
        y = _mm_min_ps(y, _mm_set1_ps(1.0f));
        z = _mm_min_ps(z, _mm_set1_ps(1.0f));
        w = _mm_min_ps(w, _mm_set1_ps(1.0f));

        // Convert to uint8_t: float * 255 + 0.5, then pack
        const __m128i r_i32 = _mm_cvtps_epi32(_mm_mul_ps(x, scale));
        const __m128i g_i32 = _mm_cvtps_epi32(_mm_mul_ps(y, scale));
        const __m128i b_i32 = _mm_cvtps_epi32(_mm_mul_ps(z, scale));
        const __m128i a_i32 = _mm_cvtps_epi32(_mm_mul_ps(w, scale));

        // Pack to 16-bit and then to 8-bit
        const __m128i rg_16 = _mm_packs_epi32(r_i32, g_i32);
        const __m128i ba_16 = _mm_packs_epi32(b_i32, a_i32);
        const __m128i rgba_8 = _mm_packus_epi16(rg_16, ba_16);

        // Store 16 bytes (4 pixels RGBA)
        _mm_storeu_si128((__m128i*)&image->data[i * 4], rgba_8);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ARM NEON
    const float32x4_t scale = vdupq_n_f32(255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load 4 color values
        float32x4_t x = vld1q_f32(&vec4xN->x[i]);
        float32x4_t y = vld1q_f32(&vec4xN->y[i]);
        float32x4_t z = vld1q_f32(&vec4xN->z[i]);
        float32x4_t w = vld1q_f32(&vec4xN->w[i]);

        // Clamp
        x = vmaxq_f32(x, vdupq_n_f32(0.0f));
        y = vmaxq_f32(y, vdupq_n_f32(0.0f));
        z = vmaxq_f32(z, vdupq_n_f32(0.0f));
        w = vmaxq_f32(w, vdupq_n_f32(0.0f));

        x = vminq_f32(x, vdupq_n_f32(1.0f));
        y = vminq_f32(y, vdupq_n_f32(1.0f));
        z = vminq_f32(z, vdupq_n_f32(1.0f));
        w = vminq_f32(w, vdupq_n_f32(1.0f));

        // Convert to int16, then to uint8 and interleave
        int16x8_t r = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(x, scale)));
        int16x8_t g = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(y, scale)));
        int16x8_t b = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(z, scale)));
        int16x8_t a = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(w, scale)));

        // Narrow to 8-bit and pack
        uint8x8_t r_u8 = vqmovun_s16(vget_low_s16(r));
        uint8x8_t g_u8 = vqmovun_s16(vget_low_s16(g));
        uint8x8_t b_u8 = vqmovun_s16(vget_low_s16(b));
        uint8x8_t a_u8 = vqmovun_s16(vget_low_s16(a));

        // Interleave RGBA channels
        uint8x8x4_t rgba = {{r_u8, g_u8, b_u8, a_u8}};
        vst4_u8(&image->data[i * 4], rgba);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD
    const v128_t scale = wasm_f32x4_splat(255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        v128_t x = wasm_v128_load(&vec4xN->x[i]);
        v128_t y = wasm_v128_load(&vec4xN->y[i]);
        v128_t z = wasm_v128_load(&vec4xN->z[i]);
        v128_t w = wasm_v128_load(&vec4xN->w[i]);

        // Clamp
        x = wasm_f32x4_max(x, wasm_f32x4_splat(0.0f));
        y = wasm_f32x4_max(y, wasm_f32x4_splat(0.0f));
        z = wasm_f32x4_max(z, wasm_f32x4_splat(0.0f));
        w = wasm_f32x4_max(w, wasm_f32x4_splat(0.0f));

        x = wasm_f32x4_min(x, wasm_f32x4_splat(1.0f));
        y = wasm_f32x4_min(y, wasm_f32x4_splat(1.0f));
        z = wasm_f32x4_min(z, wasm_f32x4_splat(1.0f));
        w = wasm_f32x4_min(w, wasm_f32x4_splat(1.0f));

        // Convert and pack manually for WASM
        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        wasm_v128_store(temp_r, x);
        wasm_v128_store(temp_g, y);
        wasm_v128_store(temp_b, z);
        wasm_v128_store(temp_a, w);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] * 255.0f + 0.5f);
            image->data[offset + 1] = (uint8_t)(temp_g[j] * 255.0f + 0.5f);
            image->data[offset + 2] = (uint8_t)(temp_b[j] * 255.0f + 0.5f);
            image->data[offset + 3] = (uint8_t)(temp_a[j] * 255.0f + 0.5f);
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX
    const __m128 scale = __lsx_vfrstps_s(255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        __m128 x = __lsx_vld(&vec4xN->x[i], 0);
        __m128 y = __lsx_vld(&vec4xN->y[i], 0);
        __m128 z = __lsx_vld(&vec4xN->z[i], 0);
        __m128 w = __lsx_vld(&vec4xN->w[i], 0);

        // Clamp
        x = __lsx_vfmax_s(x, __lsx_vfrstps_s(0.0f));
        y = __lsx_vfmax_s(y, __lsx_vfrstps_s(0.0f));
        z = __lsx_vfmax_s(z, __lsx_vfrstps_s(0.0f));
        w = __lsx_vfmax_s(w, __lsx_vfrstps_s(0.0f));

        x = __lsx_vfmin_s(x, __lsx_vfrstps_s(1.0f));
        y = __lsx_vfmin_s(y, __lsx_vfrstps_s(1.0f));
        z = __lsx_vfmin_s(z, __lsx_vfrstps_s(1.0f));
        w = __lsx_vfmin_s(w, __lsx_vfrstps_s(1.0f));

        // Convert and pack
        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        __lsx_vst(temp_r, x, 0);
        __lsx_vst(temp_g, y, 0);
        __lsx_vst(temp_b, z, 0);
        __lsx_vst(temp_a, w, 0);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] * 255.0f + 0.5f);
            image->data[offset + 1] = (uint8_t)(temp_g[j] * 255.0f + 0.5f);
            image->data[offset + 2] = (uint8_t)(temp_b[j] * 255.0f + 0.5f);
            image->data[offset + 3] = (uint8_t)(temp_a[j] * 255.0f + 0.5f);
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector
    for (; i + 4 <= total_pixels; i += 4) {
        size_t vl = __riscv_vsetvl_e32m1(4);

        vfloat32m1_t vx = __riscv_vle32_v_f32m1(&vec4xN->x[i], vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(&vec4xN->y[i], vl);
        vfloat32m1_t vz = __riscv_vle32_v_f32m1(&vec4xN->z[i], vl);
        vfloat32m1_t vw = __riscv_vle32_v_f32m1(&vec4xN->w[i], vl);

        // Clamp to [0, 1]
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vx = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vx, vone, vl), vzero, vl);
        vy = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vy, vone, vl), vzero, vl);
        vz = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vz, vone, vl), vzero, vl);
        vw = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vw, vone, vl), vzero, vl);

        // Scale to [0, 255]
        vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(255.0f, vl);
        vx = __riscv_vfmul_vv_f32m1(vx, vscale, vl);
        vy = __riscv_vfmul_vv_f32m1(vy, vscale, vl);
        vz = __riscv_vfmul_vv_f32m1(vz, vscale, vl);
        vw = __riscv_vfmul_vv_f32m1(vw, vscale, vl);

        // Convert to int32
        vint32m1_t vx_i32 = __riscv_vfcvt_x_f_v_i32m1(vx, vl);
        vint32m1_t vy_i32 = __riscv_vfcvt_x_f_v_i32m1(vy, vl);
        vint32m1_t vz_i32 = __riscv_vfcvt_x_f_v_i32m1(vz, vl);
        vint32m1_t vw_i32 = __riscv_vfcvt_x_f_v_i32m1(vw, vl);

        // Store interleaved RGBA
        int32_t temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        __riscv_vse32_v_i32m1(temp_r, vx_i32, vl);
        __riscv_vse32_v_i32m1(temp_g, vy_i32, vl);
        __riscv_vse32_v_i32m1(temp_b, vz_i32, vl);
        __riscv_vse32_v_i32m1(temp_a, vw_i32, vl);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] < 0 ? 0 : (temp_r[j] > 255 ? 255 : temp_r[j]));
            image->data[offset + 1] = (uint8_t)(temp_g[j] < 0 ? 0 : (temp_g[j] > 255 ? 255 : temp_g[j]));
            image->data[offset + 2] = (uint8_t)(temp_b[j] < 0 ? 0 : (temp_b[j] > 255 ? 255 : temp_b[j]));
            image->data[offset + 3] = (uint8_t)(temp_a[j] < 0 ? 0 : (temp_a[j] > 255 ? 255 : temp_a[j]));
        }
    }
#endif

    // Scalar tail
    for (; i < total_pixels; i++) {
        const size_t pixel_offset = i * 4;
        image->data[pixel_offset + 0] = float_to_u8_clamped(vec4xN->x[i]); // R
        image->data[pixel_offset + 1] = float_to_u8_clamped(vec4xN->y[i]); // G
        image->data[pixel_offset + 2] = float_to_u8_clamped(vec4xN->z[i]); // B
        image->data[pixel_offset + 3] = float_to_u8_clamped(vec4xN->w[i]); // A
    }
}

void wcn_math_ImageBitmap_to_Vec4xN(
    WMATH_TYPE(Vec4xN)* vec4xN,
    const WMATH_TYPE(ImageBitmap)* image,
    const size_t width,
    const size_t height
) {
    if (!vec4xN || !image || width == 0 || height == 0) {
        return;
    }

    const size_t total_pixels = width * height;
    if (image->width != width || image->height != height) {
        return;
    }

    // Allocate SoA arrays (16-byte aligned)
    vec4xN->count = total_pixels;
    vec4xN->x = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    vec4xN->y = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    vec4xN->z = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    vec4xN->w = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));

    if (!vec4xN->x || !vec4xN->y || !vec4xN->z || !vec4xN->w) {
        // Cleanup on allocation failure
        if (vec4xN->x) wcn_aligned_free(vec4xN->x);
        if (vec4xN->y) wcn_aligned_free(vec4xN->y);
        if (vec4xN->z) wcn_aligned_free(vec4xN->z);
        if (vec4xN->w) wcn_aligned_free(vec4xN->w);
        vec4xN->x = vec4xN->y = vec4xN->z = vec4xN->w = nullptr;
        vec4xN->count = 0;
        return;
    }

    // Convert packed RGBA to SoA (x=R, y=G, z=B, w=A)
    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE - process 4 pixels at a time
    const __m128 inv_scale = _mm_set1_ps(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load 16 bytes (4 pixels RGBA)
        __m128i rgba = _mm_loadu_si128((__m128i*)&image->data[i * 4]);

        // Unpack to 16-bit
        __m128i rgba_16 = _mm_unpacklo_epi8(rgba, _mm_setzero_si128());
        const __m128i r_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
        const __m128i g_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

        rgba_16 = _mm_unpackhi_epi8(rgba, _mm_setzero_si128());
        const __m128i b_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
        const __m128i a_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

        // Convert to float
        __m128 r = _mm_cvtepi32_ps(r_16);
        __m128 g = _mm_cvtepi32_ps(g_16);
        __m128 b = _mm_cvtepi32_ps(b_16);
        __m128 a = _mm_cvtepi32_ps(a_16);

        // Scale to [0, 1]
        r = _mm_mul_ps(r, inv_scale);
        g = _mm_mul_ps(g, inv_scale);
        b = _mm_mul_ps(b, inv_scale);
        a = _mm_mul_ps(a, inv_scale);

        _mm_storeu_ps(&vec4xN->x[i], r);
        _mm_storeu_ps(&vec4xN->y[i], g);
        _mm_storeu_ps(&vec4xN->z[i], b);
        _mm_storeu_ps(&vec4xN->w[i], a);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ARM NEON
    const float32x4_t inv_scale = vdupq_n_f32(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load 16 bytes
        uint8x16_t rgba = vld1q_u8(&image->data[i * 4]);

        // Deinterleave RGBA
        uint8x8x4_t rgba_8 = vuzp4_u8(vget_low_u8(rgba), vget_high_u8(rgba));

        // Convert to float
        float32x4_t r = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[0])), inv_scale);
        float32x4_t g = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[1])), inv_scale);
        float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[2])), inv_scale);
        float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[3])), inv_scale);

        vst1q_f32(&vec4xN->x[i], r);
        vst1q_f32(&vec4xN->y[i], g);
        vst1q_f32(&vec4xN->z[i], b);
        vst1q_f32(&vec4xN->w[i], a);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD
    const v128_t inv_scale = wasm_f32x4_splat(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load and manually deinterleave for WASM
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            vec4xN->x[i + j] = (float)image->data[offset + 0] / 255.0f;
            vec4xN->y[i + j] = (float)image->data[offset + 1] / 255.0f;
            vec4xN->z[i + j] = (float)image->data[offset + 2] / 255.0f;
            vec4xN->w[i + j] = (float)image->data[offset + 3] / 255.0f;
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX
    const __m128 inv_scale = __lsx_vfrstps_s(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load and manually convert for LoongArch
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            vec4xN->x[i + j] = (float)image->data[offset + 0] / 255.0f;
            vec4xN->y[i + j] = (float)image->data[offset + 1] / 255.0f;
            vec4xN->z[i + j] = (float)image->data[offset + 2] / 255.0f;
            vec4xN->w[i + j] = (float)image->data[offset + 3] / 255.0f;
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector
    const float inv_scale_val = 1.0f / 255.0f;

    for (; i + 4 <= total_pixels; i += 4) {
        size_t vl = __riscv_vsetvl_e32m1(4);

        // Deinterleave RGBA to separate channels
        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            temp_r[j] = (float)image->data[offset + 0];
            temp_g[j] = (float)image->data[offset + 1];
            temp_b[j] = (float)image->data[offset + 2];
            temp_a[j] = (float)image->data[offset + 3];
        }

        // Load and scale with RISC-V Vector
        vfloat32m1_t vr = __riscv_vle32_v_f32m1(temp_r, vl);
        vfloat32m1_t vg = __riscv_vle32_v_f32m1(temp_g, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(temp_b, vl);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(temp_a, vl);

        vfloat32m1_t vinv_scale = __riscv_vfmv_v_f_f32m1(inv_scale_val, vl);
        vr = __riscv_vfmul_vv_f32m1(vr, vinv_scale, vl);
        vg = __riscv_vfmul_vv_f32m1(vg, vinv_scale, vl);
        vb = __riscv_vfmul_vv_f32m1(vb, vinv_scale, vl);
        va = __riscv_vfmul_vv_f32m1(va, vinv_scale, vl);

        __riscv_vse32_v_f32m1(&vec4xN->x[i], vr, vl);
        __riscv_vse32_v_f32m1(&vec4xN->y[i], vg, vl);
        __riscv_vse32_v_f32m1(&vec4xN->z[i], vb, vl);
        __riscv_vse32_v_f32m1(&vec4xN->w[i], va, vl);
    }
#endif

    // Scalar tail
    for (; i < total_pixels; i++) {
        const size_t pixel_offset = i * 4;
        vec4xN->x[i] = u8_to_float(image->data[pixel_offset + 0]); // R
        vec4xN->y[i] = u8_to_float(image->data[pixel_offset + 1]); // G
        vec4xN->z[i] = u8_to_float(image->data[pixel_offset + 2]); // B
        vec4xN->w[i] = u8_to_float(image->data[pixel_offset + 3]); // A
    }
}

void wcn_math_ImageRGBA_to_ImageBitmap(
    WMATH_TYPE(ImageBitmap)* image,
    const WMATH_TYPE(ImageRGBA)* image_rgba,
    const size_t width,
    const size_t height
) {
    "SIMD OPEN";
    if (!image || !image_rgba || width == 0 || height == 0) {
        return;
    }

    if (image_rgba->width != width || image_rgba->height != height) {
        return;
    }

    const size_t total_pixels = width * height;

    image->width = width;
    image->height = height;

    // Allocate bitmap data (4 bytes per pixel: RGBA)
    image->data = (uint8_t*)malloc(total_pixels * 4);
    if (!image->data) {
        return;
    }

    // Convert SoA (r, g, b, a) to packed RGBA
    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE - process 4 pixels at a time
    const __m128 scale = _mm_set1_ps(255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        // Load 4 color values
        __m128 r = _mm_loadu_ps(&image_rgba->r[i]);
        __m128 g = _mm_loadu_ps(&image_rgba->g[i]);
        __m128 b = _mm_loadu_ps(&image_rgba->b[i]);
        __m128 a = _mm_loadu_ps(&image_rgba->a[i]);

        // Clamp to [0, 1]
        r = _mm_max_ps(r, _mm_setzero_ps());
        g = _mm_max_ps(g, _mm_setzero_ps());
        b = _mm_max_ps(b, _mm_setzero_ps());
        a = _mm_max_ps(a, _mm_setzero_ps());

        r = _mm_min_ps(r, _mm_set1_ps(1.0f));
        g = _mm_min_ps(g, _mm_set1_ps(1.0f));
        b = _mm_min_ps(b, _mm_set1_ps(1.0f));
        a = _mm_min_ps(a, _mm_set1_ps(1.0f));

        // Convert to uint8_t and pack
        const __m128i r_i32 = _mm_cvtps_epi32(_mm_mul_ps(r, scale));
        const __m128i g_i32 = _mm_cvtps_epi32(_mm_mul_ps(g, scale));
        const __m128i b_i32 = _mm_cvtps_epi32(_mm_mul_ps(b, scale));
        const __m128i a_i32 = _mm_cvtps_epi32(_mm_mul_ps(a, scale));

        const __m128i rg_16 = _mm_packs_epi32(r_i32, g_i32);
        const __m128i ba_16 = _mm_packs_epi32(b_i32, a_i32);
        const __m128i rgba_8 = _mm_packus_epi16(rg_16, ba_16);

        _mm_storeu_si128((__m128i*)&image->data[i * 4], rgba_8);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ARM NEON
    const float32x4_t scale = vdupq_n_f32(255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        float32x4_t r = vld1q_f32(&image_rgba->r[i]);
        float32x4_t g = vld1q_f32(&image_rgba->g[i]);
        float32x4_t b = vld1q_f32(&image_rgba->b[i]);
        float32x4_t a = vld1q_f32(&image_rgba->a[i]);

        // Clamp
        r = vmaxq_f32(r, vdupq_n_f32(0.0f));
        g = vmaxq_f32(g, vdupq_n_f32(0.0f));
        b = vmaxq_f32(b, vdupq_n_f32(0.0f));
        a = vmaxq_f32(a, vdupq_n_f32(0.0f));

        r = vminq_f32(r, vdupq_n_f32(1.0f));
        g = vminq_f32(g, vdupq_n_f32(1.0f));
        b = vminq_f32(b, vdupq_n_f32(1.0f));
        a = vminq_f32(a, vdupq_n_f32(1.0f));

        // Convert and interleave
        int16x8_t r_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(r, scale)));
        int16x8_t g_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(g, scale)));
        int16x8_t b_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(b, scale)));
        int16x8_t a_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(a, scale)));

        uint8x8_t r_u8 = vqmovun_s16(vget_low_s16(r_s16));
        uint8x8_t g_u8 = vqmovun_s16(vget_low_s16(g_s16));
        uint8x8_t b_u8 = vqmovun_s16(vget_low_s16(b_s16));
        uint8x8_t a_u8 = vqmovun_s16(vget_low_s16(a_s16));

        uint8x8x4_t rgba = {{r_u8, g_u8, b_u8, a_u8}};
        vst4_u8(&image->data[i * 4], rgba);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD
    for (; i + 4 <= total_pixels; i += 4) {
        v128_t r = wasm_v128_load(&image_rgba->r[i]);
        v128_t g = wasm_v128_load(&image_rgba->g[i]);
        v128_t b = wasm_v128_load(&image_rgba->b[i]);
        v128_t a = wasm_v128_load(&image_rgba->a[i]);

        // Clamp
        r = wasm_f32x4_max(r, wasm_f32x4_splat(0.0f));
        g = wasm_f32x4_max(g, wasm_f32x4_splat(0.0f));
        b = wasm_f32x4_max(b, wasm_f32x4_splat(0.0f));
        a = wasm_f32x4_max(a, wasm_f32x4_splat(0.0f));

        r = wasm_f32x4_min(r, wasm_f32x4_splat(1.0f));
        g = wasm_f32x4_min(g, wasm_f32x4_splat(1.0f));
        b = wasm_f32x4_min(b, wasm_f32x4_splat(1.0f));
        a = wasm_f32x4_min(a, wasm_f32x4_splat(1.0f));

        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        wasm_v128_store(temp_r, r);
        wasm_v128_store(temp_g, g);
        wasm_v128_store(temp_b, b);
        wasm_v128_store(temp_a, a);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] * 255.0f + 0.5f);
            image->data[offset + 1] = (uint8_t)(temp_g[j] * 255.0f + 0.5f);
            image->data[offset + 2] = (uint8_t)(temp_b[j] * 255.0f + 0.5f);
            image->data[offset + 3] = (uint8_t)(temp_a[j] * 255.0f + 0.5f);
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX
    for (; i + 4 <= total_pixels; i += 4) {
        __m128 r = __lsx_vld(&image_rgba->r[i], 0);
        __m128 g = __lsx_vld(&image_rgba->g[i], 0);
        __m128 b = __lsx_vld(&image_rgba->b[i], 0);
        __m128 a = __lsx_vld(&image_rgba->a[i], 0);

        // Clamp
        r = __lsx_vfmax_s(r, __lsx_vfrstps_s(0.0f));
        g = __lsx_vfmax_s(g, __lsx_vfrstps_s(0.0f));
        b = __lsx_vfmax_s(b, __lsx_vfrstps_s(0.0f));
        a = __lsx_vfmax_s(a, __lsx_vfrstps_s(0.0f));

        r = __lsx_vfmin_s(r, __lsx_vfrstps_s(1.0f));
        g = __lsx_vfmin_s(g, __lsx_vfrstps_s(1.0f));
        b = __lsx_vfmin_s(b, __lsx_vfrstps_s(1.0f));
        a = __lsx_vfmin_s(a, __lsx_vfrstps_s(1.0f));

        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        __lsx_vst(temp_r, r, 0);
        __lsx_vst(temp_g, g, 0);
        __lsx_vst(temp_b, b, 0);
        __lsx_vst(temp_a, a, 0);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] * 255.0f + 0.5f);
            image->data[offset + 1] = (uint8_t)(temp_g[j] * 255.0f + 0.5f);
            image->data[offset + 2] = (uint8_t)(temp_b[j] * 255.0f + 0.5f);
            image->data[offset + 3] = (uint8_t)(temp_a[j] * 255.0f + 0.5f);
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector
    for (; i + 4 <= total_pixels; i += 4) {
        size_t vl = __riscv_vsetvl_e32m1(4);

        vfloat32m1_t vr = __riscv_vle32_v_f32m1(&image_rgba->r[i], vl);
        vfloat32m1_t vg = __riscv_vle32_v_f32m1(&image_rgba->g[i], vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(&image_rgba->b[i], vl);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(&image_rgba->a[i], vl);

        // Clamp to [0, 1]
        vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.0f, vl);
        vr = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vr, vone, vl), vzero, vl);
        vg = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vg, vone, vl), vzero, vl);
        vb = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vb, vone, vl), vzero, vl);
        va = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(va, vone, vl), vzero, vl);

        // Scale to [0, 255]
        vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(255.0f, vl);
        vr = __riscv_vfmul_vv_f32m1(vr, vscale, vl);
        vg = __riscv_vfmul_vv_f32m1(vg, vscale, vl);
        vb = __riscv_vfmul_vv_f32m1(vb, vscale, vl);
        va = __riscv_vfmul_vv_f32m1(va, vscale, vl);

        // Convert to int32
        vint32m1_t vr_i32 = __riscv_vfcvt_x_f_v_i32m1(vr, vl);
        vint32m1_t vg_i32 = __riscv_vfcvt_x_f_v_i32m1(vg, vl);
        vint32m1_t vb_i32 = __riscv_vfcvt_x_f_v_i32m1(vb, vl);
        vint32m1_t va_i32 = __riscv_vfcvt_x_f_v_i32m1(va, vl);

        // Store interleaved RGBA
        int32_t temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        __riscv_vse32_v_i32m1(temp_r, vr_i32, vl);
        __riscv_vse32_v_i32m1(temp_g, vg_i32, vl);
        __riscv_vse32_v_i32m1(temp_b, vb_i32, vl);
        __riscv_vse32_v_i32m1(temp_a, va_i32, vl);

        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->data[offset + 0] = (uint8_t)(temp_r[j] < 0 ? 0 : (temp_r[j] > 255 ? 255 : temp_r[j]));
            image->data[offset + 1] = (uint8_t)(temp_g[j] < 0 ? 0 : (temp_g[j] > 255 ? 255 : temp_g[j]));
            image->data[offset + 2] = (uint8_t)(temp_b[j] < 0 ? 0 : (temp_b[j] > 255 ? 255 : temp_b[j]));
            image->data[offset + 3] = (uint8_t)(temp_a[j] < 0 ? 0 : (temp_a[j] > 255 ? 255 : temp_a[j]));
        }
    }
#endif

    // Scalar tail
    for (; i < total_pixels; i++) {
        const size_t pixel_offset = i * 4;
        image->data[pixel_offset + 0] = float_to_u8_clamped(image_rgba->r[i]); // R
        image->data[pixel_offset + 1] = float_to_u8_clamped(image_rgba->g[i]); // G
        image->data[pixel_offset + 2] = float_to_u8_clamped(image_rgba->b[i]); // B
        image->data[pixel_offset + 3] = float_to_u8_clamped(image_rgba->a[i]); // A
    }
}

void wcn_math_ImageBitmap_to_ImageRGBA(
    WMATH_TYPE(ImageRGBA)* image,
    const WMATH_TYPE(ImageBitmap)* image_bitmap,
    const size_t width,
    const size_t height
) {
    if (!image || !image_bitmap || width == 0 || height == 0) {
        return;
    }

    if (image_bitmap->width != width || image_bitmap->height != height) {
        return;
    }

    const size_t total_pixels = width * height;

    image->width = width;
    image->height = height;

    // Allocate SoA arrays (16-byte aligned)
    image->r = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    image->g = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    image->b = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));
    image->a = (float*)wcn_aligned_alloc(16, total_pixels * sizeof(float));

    if (!image->r || !image->g || !image->b || !image->a) {
        // Cleanup on allocation failure
        if (image->r) wcn_aligned_free(image->r);
        if (image->g) wcn_aligned_free(image->g);
        if (image->b) wcn_aligned_free(image->b);
        if (image->a) wcn_aligned_free(image->a);
        image->r = image->g = image->b = image->a = nullptr;
        image->width = image->height = 0;
        return;
    }

    // Convert packed RGBA to SoA (r, g, b, a)
    size_t i = 0;

#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE - process 4 pixels at a time
    const __m128 inv_scale = _mm_set1_ps(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        __m128i rgba = _mm_loadu_si128((__m128i*)&image_bitmap->data[i * 4]);

        // Unpack to 16-bit
        __m128i rgba_16 = _mm_unpacklo_epi8(rgba, _mm_setzero_si128());
        const __m128i r_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
        const __m128i g_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

        rgba_16 = _mm_unpackhi_epi8(rgba, _mm_setzero_si128());
        const __m128i b_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
        const __m128i a_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

        // Convert to float
        __m128 r = _mm_cvtepi32_ps(r_16);
        __m128 g = _mm_cvtepi32_ps(g_16);
        __m128 b = _mm_cvtepi32_ps(b_16);
        __m128 a = _mm_cvtepi32_ps(a_16);

        // Scale to [0, 1]
        r = _mm_mul_ps(r, inv_scale);
        g = _mm_mul_ps(g, inv_scale);
        b = _mm_mul_ps(b, inv_scale);
        a = _mm_mul_ps(a, inv_scale);

        _mm_storeu_ps(&image->r[i], r);
        _mm_storeu_ps(&image->g[i], g);
        _mm_storeu_ps(&image->b[i], b);
        _mm_storeu_ps(&image->a[i], a);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // ARM NEON
    const float32x4_t inv_scale = vdupq_n_f32(1.0f / 255.0f);

    for (; i + 4 <= total_pixels; i += 4) {
        uint8x16_t rgba = vld1q_u8(&image_bitmap->data[i * 4]);

        // Deinterleave RGBA
        uint8x8x4_t rgba_8 = vuzp4_u8(vget_low_u8(rgba), vget_high_u8(rgba));

        // Convert to float
        float32x4_t r = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[0])), inv_scale);
        float32x4_t g = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[1])), inv_scale);
        float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[2])), inv_scale);
        float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u8(rgba_8.val[3])), inv_scale);

        vst1q_f32(&image->r[i], r);
        vst1q_f32(&image->g[i], g);
        vst1q_f32(&image->b[i], b);
        vst1q_f32(&image->a[i], a);
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD
    for (; i + 4 <= total_pixels; i += 4) {
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->r[i + j] = (float)image_bitmap->data[offset + 0] / 255.0f;
            image->g[i + j] = (float)image_bitmap->data[offset + 1] / 255.0f;
            image->b[i + j] = (float)image_bitmap->data[offset + 2] / 255.0f;
            image->a[i + j] = (float)image_bitmap->data[offset + 3] / 255.0f;
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX
    for (; i + 4 <= total_pixels; i += 4) {
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            image->r[i + j] = (float)image_bitmap->data[offset + 0] / 255.0f;
            image->g[i + j] = (float)image_bitmap->data[offset + 1] / 255.0f;
            image->b[i + j] = (float)image_bitmap->data[offset + 2] / 255.0f;
            image->a[i + j] = (float)image_bitmap->data[offset + 3] / 255.0f;
        }
    }

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector
    const float inv_scale_val = 1.0f / 255.0f;

    for (; i + 4 <= total_pixels; i += 4) {
        size_t vl = __riscv_vsetvl_e32m1(4);

        // Deinterleave RGBA to separate channels
        float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
        for (size_t j = 0; j < 4; j++) {
            size_t offset = (i + j) * 4;
            temp_r[j] = (float)image_bitmap->data[offset + 0];
            temp_g[j] = (float)image_bitmap->data[offset + 1];
            temp_b[j] = (float)image_bitmap->data[offset + 2];
            temp_a[j] = (float)image_bitmap->data[offset + 3];
        }

        // Load and scale with RISC-V Vector
        vfloat32m1_t vr = __riscv_vle32_v_f32m1(temp_r, vl);
        vfloat32m1_t vg = __riscv_vle32_v_f32m1(temp_g, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(temp_b, vl);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(temp_a, vl);

        vfloat32m1_t vinv_scale = __riscv_vfmv_v_f_f32m1(inv_scale_val, vl);
        vr = __riscv_vfmul_vv_f32m1(vr, vinv_scale, vl);
        vg = __riscv_vfmul_vv_f32m1(vg, vinv_scale, vl);
        vb = __riscv_vfmul_vv_f32m1(vb, vinv_scale, vl);
        va = __riscv_vfmul_vv_f32m1(va, vinv_scale, vl);

        __riscv_vse32_v_f32m1(&image->r[i], vr, vl);
        __riscv_vse32_v_f32m1(&image->g[i], vg, vl);
        __riscv_vse32_v_f32m1(&image->b[i], vb, vl);
        __riscv_vse32_v_f32m1(&image->a[i], va, vl);
    }
#endif

    // Scalar tail
    for (; i < total_pixels; i++) {
        const size_t pixel_offset = i * 4;
        image->r[i] = u8_to_float(image_bitmap->data[pixel_offset + 0]); // R
        image->g[i] = u8_to_float(image_bitmap->data[pixel_offset + 1]); // G
        image->b[i] = u8_to_float(image_bitmap->data[pixel_offset + 2]); // B
        image->a[i] = u8_to_float(image_bitmap->data[pixel_offset + 3]); // A
    }
}

// ============================================================================
// Memory Management Functions
// ============================================================================

void wcn_math_ImageBitmap_free(WMATH_TYPE(ImageBitmap)* image) {
    if (!image) return;
    if (image->data) {
        free(image->data);
        image->data = nullptr;
    }
    image->width = 0;
    image->height = 0;
}

void wcn_math_ImageRGBA_free(WMATH_TYPE(ImageRGBA)* image) {
    if (!image) return;
    if (image->r) { wcn_aligned_free(image->r); image->r = NULL; }
    if (image->g) { wcn_aligned_free(image->g); image->g = NULL; }
    if (image->b) { wcn_aligned_free(image->b); image->b = NULL; }
    if (image->a) { wcn_aligned_free(image->a); image->a = NULL; }
    image->width = 0;
    image->height = 0;
}

// ============================================================================
// SIMD Kernels (for unit testing and verification)
// ============================================================================

void wcn_math_simd_soa_to_rgba_u8(
    uint8_t dst_rgba[16],
    const float src_r[4],
    const float src_g[4],
    const float src_b[4],
    const float src_a[4]
) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE implementation
    const __m128 scale = _mm_set1_ps(255.0f);

    __m128 r = _mm_loadu_ps(src_r);
    __m128 g = _mm_loadu_ps(src_g);
    __m128 b = _mm_loadu_ps(src_b);
    __m128 a = _mm_loadu_ps(src_a);

    // Clamp to [0, 1]
    r = _mm_max_ps(_mm_min_ps(r, _mm_set1_ps(1.0f)), _mm_setzero_ps());
    g = _mm_max_ps(_mm_min_ps(g, _mm_set1_ps(1.0f)), _mm_setzero_ps());
    b = _mm_max_ps(_mm_min_ps(b, _mm_set1_ps(1.0f)), _mm_setzero_ps());
    a = _mm_max_ps(_mm_min_ps(a, _mm_set1_ps(1.0f)), _mm_setzero_ps());

    // Convert to int32
    __m128i r_i32 = _mm_cvtps_epi32(_mm_mul_ps(r, scale));
    __m128i g_i32 = _mm_cvtps_epi32(_mm_mul_ps(g, scale));
    __m128i b_i32 = _mm_cvtps_epi32(_mm_mul_ps(b, scale));
    __m128i a_i32 = _mm_cvtps_epi32(_mm_mul_ps(a, scale));

    // Pack to 16-bit then 8-bit
    __m128i rg_16 = _mm_packs_epi32(r_i32, g_i32);
    __m128i ba_16 = _mm_packs_epi32(b_i32, a_i32);
    __m128i rgba_8 = _mm_packus_epi16(rg_16, ba_16);

    _mm_storeu_si128((__m128i*)dst_rgba, rgba_8);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // NEON implementation
    const float32x4_t scale = vdupq_n_f32(255.0f);

    float32x4_t r = vld1q_f32(src_r);
    float32x4_t g = vld1q_f32(src_g);
    float32x4_t b = vld1q_f32(src_b);
    float32x4_t a = vld1q_f32(src_a);

    // Clamp
    r = vmaxq_f32(vminq_f32(r, vdupq_n_f32(1.0f)), vdupq_n_f32(0.0f));
    g = vmaxq_f32(vminq_f32(g, vdupq_n_f32(1.0f)), vdupq_n_f32(0.0f));
    b = vmaxq_f32(vminq_f32(b, vdupq_n_f32(1.0f)), vdupq_n_f32(0.0f));
    a = vmaxq_f32(vminq_f32(a, vdupq_n_f32(1.0f)), vdupq_n_f32(0.0f));

    // Convert and narrow
    int16x8_t r_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(r, scale)));
    int16x8_t g_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(g, scale)));
    int16x8_t b_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(b, scale)));
    int16x8_t a_s16 = vreinterpretq_s16_s32(vcvtaq_s32_f32(vmulq_f32(a, scale)));

    uint8x8_t r_u8 = vqmovun_s16(vget_low_s16(r_s16));
    uint8x8_t g_u8 = vqmovun_s16(vget_low_s16(g_s16));
    uint8x8_t b_u8 = vqmovun_s16(vget_low_s16(b_s16));
    uint8x8_t a_u8 = vqmovun_s16(vget_low_s16(a_s16));

    uint8x8x4_t rgba = {{r_u8, g_u8, b_u8, a_u8}};
    vst4_u8(dst_rgba, rgba);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD implementation
    v128_t r = wasm_v128_load(src_r);
    v128_t g = wasm_v128_load(src_g);
    v128_t b = wasm_v128_load(src_b);
    v128_t a = wasm_v128_load(src_a);

    // Clamp to [0, 1]
    r = wasm_f32x4_max(wasm_f32x4_min(r, wasm_f32x4_splat(1.0f)), wasm_f32x4_splat(0.0f));
    g = wasm_f32x4_max(wasm_f32x4_min(g, wasm_f32x4_splat(1.0f)), wasm_f32x4_splat(0.0f));
    b = wasm_f32x4_max(wasm_f32x4_min(b, wasm_f32x4_splat(1.0f)), wasm_f32x4_splat(0.0f));
    a = wasm_f32x4_max(wasm_f32x4_min(a, wasm_f32x4_splat(1.0f)), wasm_f32x4_splat(0.0f));

    // Scale to [0, 255]
    const v128_t scale = wasm_f32x4_splat(255.0f);
    r = wasm_f32x4_mul(r, scale);
    g = wasm_f32x4_mul(g, scale);
    b = wasm_f32x4_mul(b, scale);
    a = wasm_f32x4_mul(a, scale);

    // Convert to i32 and saturate to u8
    v128_t r_i32 = wasm_i32x4_trunc_sat_f32x4(r);
    v128_t g_i32 = wasm_i32x4_trunc_sat_f32x4(g);
    v128_t b_i32 = wasm_i32x4_trunc_sat_f32x4(b);
    v128_t a_i32 = wasm_i32x4_trunc_sat_f32x4(a);

    // Pack to 16-bit then 8-bit
    v128_t rg_16 = wasm_i16x8_narrow_i32x4(r_i32, g_i32);
    v128_t ba_16 = wasm_i16x8_narrow_i32x4(b_i32, a_i32);
    v128_t rgba_8 = wasm_u8x16_narrow_i16x8(rg_16, ba_16);

    wasm_v128_store(dst_rgba, rgba_8);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX implementation
    __m128 r = __lsx_vld(src_r, 0);
    __m128 g = __lsx_vld(src_g, 0);
    __m128 b = __lsx_vld(src_b, 0);
    __m128 a = __lsx_vld(src_a, 0);

    // Clamp to [0, 1]
    __m128 zero = __lsx_vfrstps_s(0.0f);
    __m128 one = __lsx_vfrstps_s(1.0f);
    r = __lsx_vfmax_s(__lsx_vfmin_s(r, one), zero);
    g = __lsx_vfmax_s(__lsx_vfmin_s(g, one), zero);
    b = __lsx_vfmax_s(__lsx_vfmin_s(b, one), zero);
    a = __lsx_vfmax_s(__lsx_vfmin_s(a, one), zero);

    // Scale and convert
    __m128 scale = __lsx_vfrstps_s(255.0f);
    r = __lsx_vfmul_s(r, scale);
    g = __lsx_vfmul_s(g, scale);
    b = __lsx_vfmul_s(b, scale);
    a = __lsx_vfmul_s(a, scale);

    // Convert to int and pack
    __m128i r_i32 = __lsx_vftint_w_s(r);
    __m128i g_i32 = __lsx_vftint_w_s(g);
    __m128i b_i32 = __lsx_vftint_w_s(b);
    __m128i a_i32 = __lsx_vftint_w_s(a);

    // Pack to 16-bit then 8-bit with saturation
    __m128i rg_16 = __lsx_vssrani_h_w(g_i32, r_i32, 0);
    __m128i ba_16 = __lsx_vssrani_h_w(a_i32, b_i32, 0);
    __m128i rgba_8 = __lsx_vssrani_bu_h(ba_16, rg_16, 0);

    __lsx_vst(rgba_8, dst_rgba, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector implementation
    size_t vl = __riscv_vsetvl_e32m1(4);

    vfloat32m1_t vr = __riscv_vle32_v_f32m1(src_r, vl);
    vfloat32m1_t vg = __riscv_vle32_v_f32m1(src_g, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(src_b, vl);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(src_a, vl);

    // Clamp to [0, 1]
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t vone = __riscv_vfmv_v_f_f32m1(1.0f, vl);
    vr = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vr, vone, vl), vzero, vl);
    vg = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vg, vone, vl), vzero, vl);
    vb = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(vb, vone, vl), vzero, vl);
    va = __riscv_vfmax_vv_f32m1(__riscv_vfmin_vv_f32m1(va, vone, vl), vzero, vl);

    // Scale to [0, 255]
    vfloat32m1_t vscale = __riscv_vfmv_v_f_f32m1(255.0f, vl);
    vr = __riscv_vfmul_vv_f32m1(vr, vscale, vl);
    vg = __riscv_vfmul_vv_f32m1(vg, vscale, vl);
    vb = __riscv_vfmul_vv_f32m1(vb, vscale, vl);
    va = __riscv_vfmul_vv_f32m1(va, vscale, vl);

    // Convert to int32 (round to nearest)
    vint32m1_t vr_i32 = __riscv_vfcvt_x_f_v_i32m1(vr, vl);
    vint32m1_t vg_i32 = __riscv_vfcvt_x_f_v_i32m1(vg, vl);
    vint32m1_t vb_i32 = __riscv_vfcvt_x_f_v_i32m1(vb, vl);
    vint32m1_t va_i32 = __riscv_vfcvt_x_f_v_i32m1(va, vl);

    // Narrow to 16-bit then 8-bit with saturation
    vint16m1_t vr_i16 = __riscv_vnclip_wx_i16m1(__riscv_vwcvt_x_x_v_i64m2(vr_i32, vl), 0, __RISCV_VXRM_RNU, vl);
    vint16m1_t vg_i16 = __riscv_vnclip_wx_i16m1(__riscv_vwcvt_x_x_v_i64m2(vg_i32, vl), 0, __RISCV_VXRM_RNU, vl);
    vint16m1_t vb_i16 = __riscv_vnclip_wx_i16m1(__riscv_vwcvt_x_x_v_i64m2(vb_i32, vl), 0, __RISCV_VXRM_RNU, vl);
    vint16m1_t va_i16 = __riscv_vnclip_wx_i16m1(__riscv_vwcvt_x_x_v_i64m2(va_i32, vl), 0, __RISCV_VXRM_RNU, vl);

    // Store interleaved RGBA (use scalar for simplicity with 4 pixels)
    int32_t temp_r[4], temp_g[4], temp_b[4], temp_a[4];
    __riscv_vse32_v_i32m1(temp_r, vr_i32, vl);
    __riscv_vse32_v_i32m1(temp_g, vg_i32, vl);
    __riscv_vse32_v_i32m1(temp_b, vb_i32, vl);
    __riscv_vse32_v_i32m1(temp_a, va_i32, vl);

    for (size_t i = 0; i < 4; i++) {
        dst_rgba[i * 4 + 0] = (uint8_t)(temp_r[i] < 0 ? 0 : (temp_r[i] > 255 ? 255 : temp_r[i]));
        dst_rgba[i * 4 + 1] = (uint8_t)(temp_g[i] < 0 ? 0 : (temp_g[i] > 255 ? 255 : temp_g[i]));
        dst_rgba[i * 4 + 2] = (uint8_t)(temp_b[i] < 0 ? 0 : (temp_b[i] > 255 ? 255 : temp_b[i]));
        dst_rgba[i * 4 + 3] = (uint8_t)(temp_a[i] < 0 ? 0 : (temp_a[i] > 255 ? 255 : temp_a[i]));
    }

#else
    // Scalar fallback
    for (size_t i = 0; i < 4; i++) {
        dst_rgba[i * 4 + 0] = float_to_u8_clamped(src_r[i]);
        dst_rgba[i * 4 + 1] = float_to_u8_clamped(src_g[i]);
        dst_rgba[i * 4 + 2] = float_to_u8_clamped(src_b[i]);
        dst_rgba[i * 4 + 3] = float_to_u8_clamped(src_a[i]);
    }
#endif
}

void wcn_math_simd_rgba_to_soa_u8(
    float dst_r[4],
    float dst_g[4],
    float dst_b[4],
    float dst_a[4],
    const uint8_t src_rgba[16]
) {
#if !defined(WMATH_DISABLE_SIMD) && WCN_HAS_X86_64
    // SSE implementation
    const __m128 inv_scale = _mm_set1_ps(1.0f / 255.0f);

    __m128i rgba = _mm_loadu_si128((const __m128i*)src_rgba);

    // Unpack bytes to 16-bit
    __m128i rgba_16 = _mm_unpacklo_epi8(rgba, _mm_setzero_si128());
    __m128i r_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
    const __m128i g_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

    rgba_16 = _mm_unpackhi_epi8(rgba, _mm_setzero_si128());
    __m128i b_16 = _mm_unpacklo_epi8(rgba_16, _mm_setzero_si128());
    const __m128i a_16 = _mm_unpackhi_epi8(rgba_16, _mm_setzero_si128());

    // Convert to float and scale
    __m128 r = _mm_mul_ps(_mm_cvtepi32_ps(r_16), inv_scale);
    __m128 g = _mm_mul_ps(_mm_cvtepi32_ps(g_16), inv_scale);
    __m128 b = _mm_mul_ps(_mm_cvtepi32_ps(b_16), inv_scale);
    __m128 a = _mm_mul_ps(_mm_cvtepi32_ps(a_16), inv_scale);

    _mm_storeu_ps(dst_r, r);
    _mm_storeu_ps(dst_g, g);
    _mm_storeu_ps(dst_b, b);
    _mm_storeu_ps(dst_a, a);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_AARCH64
    // NEON implementation
    const float32x4_t inv_scale = vdupq_n_f32(1.0f / 255.0f);

    // Load and deinterleave
    uint8x16_t rgba = vld1q_u8(src_rgba);
    uint8x8x4_t rgba_8 = vuzp4_u8(vget_low_u8(rgba), vget_high_u8(rgba));

    // Widen to 16-bit then 32-bit, convert to float
    float32x4_t r = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgba_8.val[0])))), inv_scale);
    float32x4_t g = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgba_8.val[1])))), inv_scale);
    float32x4_t b = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgba_8.val[2])))), inv_scale);
    float32x4_t a = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(rgba_8.val[3])))), inv_scale);

    vst1q_f32(dst_r, r);
    vst1q_f32(dst_g, g);
    vst1q_f32(dst_b, b);
    vst1q_f32(dst_a, a);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_WASM_SIMD
    // WebAssembly SIMD implementation
    const v128_t inv_scale = wasm_f32x4_splat(1.0f / 255.0f);

    v128_t rgba = wasm_v128_load(src_rgba);

    // Extract each byte to separate lanes and widen
    // WASM doesn't have efficient deinterleave, use scalar approach for correctness
    float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
    for (size_t i = 0; i < 4; i++) {
        temp_r[i] = (float)src_rgba[i * 4 + 0];
        temp_g[i] = (float)src_rgba[i * 4 + 1];
        temp_b[i] = (float)src_rgba[i * 4 + 2];
        temp_a[i] = (float)src_rgba[i * 4 + 3];
    }

    v128_t r = wasm_f32x4_mul(wasm_v128_load(temp_r), inv_scale);
    v128_t g = wasm_f32x4_mul(wasm_v128_load(temp_g), inv_scale);
    v128_t b = wasm_f32x4_mul(wasm_v128_load(temp_b), inv_scale);
    v128_t a = wasm_f32x4_mul(wasm_v128_load(temp_a), inv_scale);

    wasm_v128_store(dst_r, r);
    wasm_v128_store(dst_g, g);
    wasm_v128_store(dst_b, b);
    wasm_v128_store(dst_a, a);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_LOONGARCH_LSX
    // LoongArch LSX implementation
    const __m128 inv_scale = __lsx_vfrstps_s(1.0f / 255.0f);

    __m128i rgba = __lsx_vld(src_rgba, 0);

    // Unpack bytes to 16-bit then 32-bit
    __m128i zero = __lsx_vldi(0);
    __m128i rgba_lo = __lsx_vilvl_b(zero, rgba);  // lower 8 bytes -> 16-bit
    __m128i rgba_hi = __lsx_vilvh_b(zero, rgba);  // upper 8 bytes -> 16-bit

    // Extract R, G, B, A (interleaved as RGBARGBARGBARGBA)
    __m128i r_16 = __lsx_vilvl_h(zero, rgba_lo);  // R0, G0, R1, G1 -> 32-bit
    __m128i b_16 = __lsx_vilvh_h(zero, rgba_lo);  // B0, A0, B1, A1 -> 32-bit
    __m128i r2_16 = __lsx_vilvl_h(zero, rgba_hi); // R2, G2, R3, G3 -> 32-bit
    __m128i b2_16 = __lsx_vilvh_h(zero, rgba_hi); // B2, A2, B3, A3 -> 32-bit

    // Convert to float and scale (use scalar for correct channel separation)
    float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
    for (size_t i = 0; i < 4; i++) {
        temp_r[i] = (float)src_rgba[i * 4 + 0];
        temp_g[i] = (float)src_rgba[i * 4 + 1];
        temp_b[i] = (float)src_rgba[i * 4 + 2];
        temp_a[i] = (float)src_rgba[i * 4 + 3];
    }

    __m128 r = __lsx_vfmul_s(__lsx_vld(temp_r, 0), inv_scale);
    __m128 g = __lsx_vfmul_s(__lsx_vld(temp_g, 0), inv_scale);
    __m128 b = __lsx_vfmul_s(__lsx_vld(temp_b, 0), inv_scale);
    __m128 a = __lsx_vfmul_s(__lsx_vld(temp_a, 0), inv_scale);

    __lsx_vst(r, dst_r, 0);
    __lsx_vst(g, dst_g, 0);
    __lsx_vst(b, dst_b, 0);
    __lsx_vst(a, dst_a, 0);

#elif !defined(WMATH_DISABLE_SIMD) && WCN_HAS_RISCV_VECTOR
    // RISC-V Vector implementation
    size_t vl = __riscv_vsetvl_e32m1(4);
    const float inv_scale_val = 1.0f / 255.0f;

    // Extract bytes to floats (deinterleave RGBA)
    float temp_r[4], temp_g[4], temp_b[4], temp_a[4];
    for (size_t i = 0; i < 4; i++) {
        temp_r[i] = (float)src_rgba[i * 4 + 0];
        temp_g[i] = (float)src_rgba[i * 4 + 1];
        temp_b[i] = (float)src_rgba[i * 4 + 2];
        temp_a[i] = (float)src_rgba[i * 4 + 3];
    }

    // Load and scale with RISC-V Vector
    vfloat32m1_t vr = __riscv_vle32_v_f32m1(temp_r, vl);
    vfloat32m1_t vg = __riscv_vle32_v_f32m1(temp_g, vl);
    vfloat32m1_t vb = __riscv_vle32_v_f32m1(temp_b, vl);
    vfloat32m1_t va = __riscv_vle32_v_f32m1(temp_a, vl);

    vfloat32m1_t vinv_scale = __riscv_vfmv_v_f_f32m1(inv_scale_val, vl);
    vr = __riscv_vfmul_vv_f32m1(vr, vinv_scale, vl);
    vg = __riscv_vfmul_vv_f32m1(vg, vinv_scale, vl);
    vb = __riscv_vfmul_vv_f32m1(vb, vinv_scale, vl);
    va = __riscv_vfmul_vv_f32m1(va, vinv_scale, vl);

    __riscv_vse32_v_f32m1(dst_r, vr, vl);
    __riscv_vse32_v_f32m1(dst_g, vg, vl);
    __riscv_vse32_v_f32m1(dst_b, vb, vl);
    __riscv_vse32_v_f32m1(dst_a, va, vl);

#else
    // Scalar fallback
    for (size_t i = 0; i < 4; i++) {
        dst_r[i] = u8_to_float(src_rgba[i * 4 + 0]);
        dst_g[i] = u8_to_float(src_rgba[i * 4 + 1]);
        dst_b[i] = u8_to_float(src_rgba[i * 4 + 2]);
        dst_a[i] = u8_to_float(src_rgba[i * 4 + 3]);
    }
#endif
}