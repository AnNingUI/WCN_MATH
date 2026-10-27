#ifndef WCN_PLATFORM_MACROS_H
#define WCN_PLATFORM_MACROS_H

#if !defined(__cplusplus) && \
    (!defined(__STDC_VERSION__) || __STDC_VERSION__ < 202311L)
    #define nullptr NULL
#endif

// ============================================================================
// Platform and Architecture Detection
// ============================================================================

// x86/x86_64 detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64__)
#define WCN_HAS_X86_64 1
#define WCN_HAS_X86 1
#elif defined(__i386__) || defined(_M_IX86) || defined(__i686__)
#define WCN_HAS_X86 1
#define WCN_HAS_X86_64 0
#else
#define WCN_HAS_X86 0
#define WCN_HAS_X86_64 0
#endif

// ARM/AArch64 detection
#if defined(__aarch64__) || defined(_M_ARM64)
#define WCN_HAS_AARCH64 1
#define WCN_HAS_ARM 1
#elif defined(__arm__) || defined(_M_ARM)
#define WCN_HAS_ARM 1
#define WCN_HAS_AARCH64 0
#else
#define WCN_HAS_ARM 0
#define WCN_HAS_AARCH64 0
#endif

// WebAssembly detection
#if defined(__EMSCRIPTEN__) || defined(__wasm__)
#define WCN_HAS_WASM 1
#if defined(__wasm_simd128__)
#define WCN_HAS_WASM_SIMD 1
#else
#define WCN_HAS_WASM_SIMD 0
#endif
#else
#define WCN_HAS_WASM 0
#define WCN_HAS_WASM_SIMD 0
#endif

// RISC-V Vector Extension detection
#if defined(__riscv) && defined(__riscv_vector)
#define WCN_HAS_RISCV_VECTOR 1
#else
#define WCN_HAS_RISCV_VECTOR 0
#endif

// LoongArch LSX detection
#if defined(__loongarch__) && defined(__loongarch_sx)
#define WCN_HAS_LOONGARCH_LSX 1
#else
#define WCN_HAS_LOONGARCH_LSX 0
#endif

// ============================================================================
// SIMD Feature Detection
// ============================================================================

// FMA detection
#if defined(__FMA__) || defined(WCN_ENABLE_FMA)
#define WCN_HAS_FMA 1
#else
#define WCN_HAS_FMA 0
#endif

// AVX detection.
//
// Two sources: the compiler's own feature macro, and the build system's
// WCN_ENABLE_AVX/WCN_ENABLE_AVX2 switches. The latter matter because the library
// is normally built without /arch:AVX*: the sources are meant to opt in per file
// via these defines, and a build that passes /arch:AVX2 does not always set
// __AVX2__ on every compiler (MSVC's /arch:AVX2 does, but the CMake option path
// uses the defines). Accepting both keeps one spelling honest.
#if defined(__AVX__) || defined(WCN_ENABLE_AVX) || defined(WCN_ENABLE_AVX2)
#ifndef WCN_HAS_AVX
#define WCN_HAS_AVX 1
#endif
#endif

#if defined(__AVX2__) || defined(WCN_ENABLE_AVX2)
#ifndef WCN_HAS_AVX2
#define WCN_HAS_AVX2 1
#endif
#endif

// AVX-512 detection.
//
// WCN_HAS_AVX512 gates the widest branch of the x86 dispatch ladder and is set
// only when the whole set of sub-features the 512-bit code paths actually use is
// present, not merely __AVX512F__:
//   F   - the 512-bit float arithmetic itself
//   DQ  - 64-bit lane logic, used by the mask/pack sequences in image.c
//   BW  - byte/word lane shifts, used by the RGBA packing kernels
//   VL  - 128/256-bit forms of the same instructions, needed for the narrow tail
// Passing -mavx512f alone without BW/DQ would leave those helpers unavailable, so
// requiring all four keeps a single macro honest.
#if (defined(__AVX512F__) && defined(__AVX512DQ__) && defined(__AVX512BW__) && \
     defined(__AVX512VL__)) ||                                                      \
    defined(WCN_ENABLE_AVX512)
#ifndef WCN_HAS_AVX512
#define WCN_HAS_AVX512 1
#endif
#endif

#ifndef WCN_HAS_AVX512
#define WCN_HAS_AVX512 0
#endif

// Precedence helpers: exactly one of these is 1 for a given build, and they exist
// so a function can express "use the widest available x86 path" as a chain of
// #elif without repeating the feature tests.
#if WCN_HAS_AVX512
#define WCN_X86_TIER 4
#elif defined(WCN_HAS_AVX2)
#define WCN_X86_TIER 3
#elif defined(WCN_HAS_AVX)
#define WCN_X86_TIER 2
#elif WCN_HAS_FMA
#define WCN_X86_TIER 1
#else
#define WCN_X86_TIER 0
#endif

// NEON detection (ARM)
#if WCN_HAS_AARCH64 || (WCN_HAS_ARM && (defined(__ARM_NEON) || defined(__ARM_NEON__)))
#define WCN_HAS_NEON 1
#else
#define WCN_HAS_NEON 0
#endif

// ============================================================================
// Export Macros
// ============================================================================

// WebAssembly export macro
#if WCN_HAS_WASM
#define WCN_WASM_EXPORT __attribute__((used)) __attribute__((visibility("default")))
#else
#define WCN_WASM_EXPORT
#endif

// General export macro for shared libraries
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef WCN_MATH_EXPORT
#define WCN_API __declspec(dllexport)
#else
#define WCN_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#define WCN_API __attribute__((visibility("default")))
#else
#define WCN_API
#endif

// ============================================================================
// Compiler-specific attributes
// ============================================================================

// Force inline
#if defined(_MSC_VER)
#define WCN_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define WCN_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define WCN_FORCE_INLINE inline
#endif

// Alignment
#if defined(_MSC_VER)
#define WCN_ALIGN(n) __declspec(align(n))
#elif defined(__GNUC__) || defined(__clang__)
#define WCN_ALIGN(n) __attribute__((aligned(n)))
#else
#define WCN_ALIGN(n)
#endif

#endif // WCN_PLATFORM_MACROS_H
