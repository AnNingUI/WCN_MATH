cmake_minimum_required(VERSION 3.11)

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(WCN_MATH C CXX)

# 包含必要的 CMake 模块
include(CheckCCompilerFlag)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置包含目录
include_directories(include)
include_directories(src)

# 收集源文件
file(GLOB MATH_COMMON_SOURCES "src/common/*.c")
file(GLOB MATH_SOURCES "src/*.c")
file(GLOB MATH_DST_SOURCES "src/dst/*.c")

# 合并所有源文件
set(ALL_SOURCES ${MATH_COMMON_SOURCES} ${MATH_SOURCES} ${MATH_DST_SOURCES})

# 创建数学库
add_library(wcn_math ${ALL_SOURCES})

# 链接数学库
target_link_libraries(wcn_math m)

# SIMD 检测和配置选项
option(WCN_MATH_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(WCN_MATH_ENABLE_SSE "Enable SSE instructions" ON)
option(WCN_MATH_ENABLE_SSE2 "Enable SSE2 instructions" ON)
option(WCN_MATH_ENABLE_SSE3 "Enable SSE3 instructions" ON)
option(WCN_MATH_ENABLE_SSSE3 "Enable SSSE3 instructions" ON)
option(WCN_MATH_ENABLE_SSE41 "Enable SSE4.1 instructions" ON)
option(WCN_MATH_ENABLE_SSE42 "Enable SSE4.2 instructions" ON)
option(WCN_MATH_ENABLE_AVX "Enable AVX instructions" ON)
option(WCN_MATH_ENABLE_AVX2 "Enable AVX2 instructions" ON)
option(WCN_MATH_ENABLE_FMA "Enable FMA instructions" ON)
option(WCN_MATH_ENABLE_NEON "Enable NEON instructions" ON)

# SIMD 检测宏定义
if(WCN_MATH_ENABLE_SIMD)
    if(WCN_MATH_ENABLE_SSE)
        check_c_compiler_flag(-msse SUPPORTS_SSE)
        if(SUPPORTS_SSE)
            target_compile_options(wcn_math PRIVATE -msse)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE)
        endif()
    endif()

    if(WCN_MATH_ENABLE_SSE2)
        check_c_compiler_flag(-msse2 SUPPORTS_SSE2)
        if(SUPPORTS_SSE2)
            target_compile_options(wcn_math PRIVATE -msse2)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE2)
        endif()
    endif()

    if(WCN_MATH_ENABLE_SSE3)
        check_c_compiler_flag(-msse3 SUPPORTS_SSE3)
        if(SUPPORTS_SSE3)
            target_compile_options(wcn_math PRIVATE -msse3)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE3)
        endif()
    endif()

    if(WCN_MATH_ENABLE_SSSE3)
        check_c_compiler_flag(-mssse3 SUPPORTS_SSSE3)
        if(SUPPORTS_SSSE3)
            target_compile_options(wcn_math PRIVATE -mssse3)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSSE3)
        endif()
    endif()

    if(WCN_MATH_ENABLE_SSE41)
        check_c_compiler_flag(-msse4.1 SUPPORTS_SSE41)
        if(SUPPORTS_SSE41)
            target_compile_options(wcn_math PRIVATE -msse4.1)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE41)
        endif()
    endif()

    if(WCN_MATH_ENABLE_SSE42)
        check_c_compiler_flag(-msse4.2 SUPPORTS_SSE42)
        if(SUPPORTS_SSE42)
            target_compile_options(wcn_math PRIVATE -msse4.2)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE42)
        endif()
    endif()

    if(WCN_MATH_ENABLE_AVX)
        check_c_compiler_flag(-mavx SUPPORTS_AVX)
        if(SUPPORTS_AVX)
            target_compile_options(wcn_math PRIVATE -mavx)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_AVX)
        endif()
    endif()

    if(WCN_MATH_ENABLE_AVX2)
        check_c_compiler_flag(-mavx2 SUPPORTS_AVX2)
        if(SUPPORTS_AVX2)
            target_compile_options(wcn_math PRIVATE -mavx2)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_AVX2)
        endif()
    endif()

    if(WCN_MATH_ENABLE_FMA)
        check_c_compiler_flag(-mfma SUPPORTS_FMA)
        if(SUPPORTS_FMA)
            target_compile_options(wcn_math PRIVATE -mfma)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_FMA)
        endif()
    endif()

    if(WCN_MATH_ENABLE_NEON)
        check_c_compiler_flag(-mfpu=neon SUPPORTS_NEON)
        if(SUPPORTS_NEON)
            target_compile_options(wcn_math PRIVATE -mfpu=neon)
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_NEON)
        endif()
    endif()

    target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SIMD)
else()
    target_compile_definitions(wcn_math PRIVATE WMATH_DISABLE_SIMD)
endif()

# 平台特定的 SIMD 检测
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86")
    target_compile_definitions(wcn_math PRIVATE WCN_ARCH_X86)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_definitions(wcn_math PRIVATE WCN_ARCH_X64)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM")
    target_compile_definitions(wcn_math PRIVATE WCN_ARCH_ARM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
        target_compile_definitions(wcn_math PRIVATE WCN_ARCH_ARM64)
    endif()
endif()

# WASM 构建配置
option(WCN_MATH_BUILD_WASM "Build WebAssembly version" OFF)

if(EMSCRIPTEN OR WCN_MATH_BUILD_WASM)
    message(STATUS "Building WCN_MATH for WebAssembly")

    # 为 WASM 构建收集源文件
    file(GLOB WASM_SOURCES
        "src/common/wcn_math_base.c"
        "src/common/wcn_math_globals.c"
        "src/dst/wcn_math_mat3.c"
        "src/dst/wcn_math_mat4.c"
        "src/dst/wcn_math_other.c"
        "src/dst/wcn_math_quat.c"
        "src/dst/wcn_math_vec2.c"
        "src/dst/wcn_math_vec3.c"
        "src/dst/wcn_math_vec4.c"
        "src/dst/wcn_math_wasm.c"
    )

    add_executable(wcn_math_wasm ${WASM_SOURCES})

    target_include_directories(wcn_math_wasm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    set(WASM_ONLY_COMPILE_FLAGS -O3)
    set(WASM_ONLY_LINK_FLAGS
        -O3
        -sWASM=1
        -sALLOW_MEMORY_GROWTH=1
        -sMODULARIZE=1
        -sEXPORT_NAME=createWCNMathModule
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,getValue,setValue
    )

    if(WCN_MATH_ENABLE_SIMD)
        list(APPEND WASM_ONLY_COMPILE_FLAGS -msimd128)
        list(APPEND WASM_ONLY_LINK_FLAGS -msimd128)
    endif()

    target_compile_options(wcn_math_wasm PRIVATE ${WASM_ONLY_COMPILE_FLAGS})
    target_link_options(wcn_math_wasm PRIVATE ${WASM_ONLY_LINK_FLAGS})

    set_target_properties(wcn_math_wasm PROPERTIES
        OUTPUT_NAME "wcn_math"
        SUFFIX ".js"
    )
endif()

# 安装规则
install(TARGETS wcn_math
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/WCN
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
