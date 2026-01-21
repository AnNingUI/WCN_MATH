#ifndef WCN_PLATFORM_MACROS_H
#define WCN_PLATFORM_MACROS_H

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

// AVX detection
#if defined(__AVX__)
#ifndef WCN_HAS_AVX
#define WCN_HAS_AVX 1
#endif
#endif

// AVX2 detection
#if defined(__AVX2__)
#ifndef WCN_HAS_AVX2
#define WCN_HAS_AVX2 1
#endif
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
