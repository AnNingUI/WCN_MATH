#ifndef WCN_RTYPE_H
#define WCN_RTYPE_H
#ifdef __cplusplus
#include <cstdint>
    using i8  = int8_t;
    using u8  = uint8_t;
    using i16 = int16_t;
    using u16 = uint16_t;
    using i32 = int32_t;
    using u32 = uint32_t;
    using i64 = int64_t;
    using u63 = uint64_t;
    using f32 = float;
    using f64 = double;
#else 
    typedef signed char i8;
    typedef unsigned char u8;
    typedef signed short i16;
    typedef unsigned short u16;
    typedef signed int i32;
    typedef unsigned int u32;
    typedef signed long long i64;
    typedef unsigned long long u64;
    typedef float f32;
    typedef double f64;
#endif
#endif // WCN_RTYPE_H
