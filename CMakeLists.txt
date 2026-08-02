cmake_minimum_required(VERSION 3.11)

# 生成 compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(WCN_MATH C CXX)

# 包含必要的 CMake 模块
include(CheckCCompilerFlag)

# 设置 C 标准
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 收集源文件
file(GLOB MATH_COMMON_SOURCES "src/common/*.c")
file(GLOB MATH_NO_DST_SOURCES "src/no_dst/*.c")
file(GLOB MATH_DST_SOURCES "src/dst/*.c")

# 合并所有源文件
option(WCN_MATH_IS_DST "Use Dst mode" ON)
if(WCN_MATH_IS_DST)
    message("    USE DST MODE")
    set(ALL_SOURCES ${MATH_COMMON_SOURCES} ${MATH_DST_SOURCES})
else()
    message("    USE NO DST MODE")
    set(ALL_SOURCES ${MATH_COMMON_SOURCES} ${MATH_NO_DST_SOURCES})
endif()

# ===============================
# 创建静态库
# ===============================
add_library(
    wcn_math
    STATIC
    ${ALL_SOURCES}
)
# ===============================
# include
# ===============================
target_include_directories(
    wcn_math
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接数学库
if(NOT MSVC)
    target_link_libraries(wcn_math m)
endif()

if(MSVC)
    target_compile_options(wcn_math PRIVATE /utf-8)
endif()

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

if(WCN_MATH_ENABLE_SIMD)
    target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SIMD)

    if(MSVC)
        # ==========================================
        # MSVC (Visual Studio) 编译器处理逻辑
        # ==========================================
        # MSVC 的 /arch: 标志是互斥的，只能指定一个最高基准架构
        set(MSVC_ARCH_FLAG "")

        # 从高到低判断用户开启的最高指令集
        if(WCN_MATH_ENABLE_AVX2)
            set(MSVC_ARCH_FLAG "/arch:AVX2")
        elseif(WCN_MATH_ENABLE_AVX)
            set(MSVC_ARCH_FLAG "/arch:AVX")
        elseif(WCN_MATH_ENABLE_SSE2)
            # 注意: /arch:SSE2 仅在 32 位(x86)下有意义，64 位(x64)默认强制开启 SSE2
            if(CMAKE_SIZEOF_VOID_P EQUAL 4) # 检查是否为 32 位
                set(MSVC_ARCH_FLAG "/arch:SSE2")
            endif()
        elseif(WCN_MATH_ENABLE_SSE)
            if(CMAKE_SIZEOF_VOID_P EQUAL 4)
                set(MSVC_ARCH_FLAG "/arch:SSE")
            endif()
        endif()

        if(MSVC_ARCH_FLAG)
            target_compile_options(wcn_math PRIVATE ${MSVC_ARCH_FLAG})
        endif()

        # MSVC 下：只要用户开启了选项，就定义对应的宏，方便 C 代码中使用 #ifdef
        # (因为 MSVC 只要支持该架构，包含对应的 intrinsics 头文件即可使用)
        if(WCN_MATH_ENABLE_SSE)   
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE) 
        endif()
        if(WCN_MATH_ENABLE_SSE2)  
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE2) 
        endif()
        if(WCN_MATH_ENABLE_SSE3)  
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE3) 
        endif()
        if(WCN_MATH_ENABLE_SSSE3) 
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSSE3)
        endif()
        if(WCN_MATH_ENABLE_SSE41) 
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE41)
        endif()
        if(WCN_MATH_ENABLE_SSE42) 
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_SSE42)
        endif()
        if(WCN_MATH_ENABLE_AVX)   
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_AVX) 
        endif()
        if(WCN_MATH_ENABLE_AVX2)  
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_AVX2) 
        endif()
        if(WCN_MATH_ENABLE_FMA)   
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_FMA) 
        endif()

        # MSVC ARM64 默认支持 NEON
        if(WCN_MATH_ENABLE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64")
            target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_NEON)
        endif()

    else()
        # ==========================================
        # GCC / Clang 编译器处理逻辑
        # ==========================================
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
            # 区分 32位 ARM 和 64位 ARM
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
                # AArch64 默认支持 NEON，直接定义宏即可，不需要额外编译标志
                target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_NEON)
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
                # 32位 ARM 需要检测并添加 -mfpu=neon
                check_c_compiler_flag("-mfpu=neon" SUPPORTS_NEON)
                if(SUPPORTS_NEON)
                    target_compile_options(wcn_math PRIVATE -mfpu=neon)
                    target_compile_definitions(wcn_math PRIVATE WCN_ENABLE_NEON)
                endif()
            endif()
        endif()
    endif()

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
        "src/dst/wcn_math_vec2xn.c"
        "src/dst/wcn_math_vec3.c"
        "src/dst/wcn_math_vec4.c"
        "src/dst/wcn_math_vec4xn.c"
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
        -sEXPORTED_FUNCTIONS=_malloc,_free,_wcn_math_Mat3_add_wasm,_wcn_math_Mat3_copy_wasm,_wcn_math_Mat3_create_wasm,_wcn_math_Mat3_determinant_wasm,_wcn_math_Mat3_equals_approximately_wasm,_wcn_math_Mat3_equals_wasm,_wcn_math_Mat3_from_mat4_wasm,_wcn_math_Mat3_from_quat_wasm,_wcn_math_Mat3_get_3D_scaling_wasm,_wcn_math_Mat3_get_axis_wasm,_wcn_math_Mat3_get_scaling_wasm,_wcn_math_Mat3_get_translation_wasm,_wcn_math_Mat3_identity_wasm,_wcn_math_Mat3_inverse_wasm,_wcn_math_Mat3_multiply_scalar_wasm,_wcn_math_Mat3_multiply_wasm,_wcn_math_Mat3_negate_wasm,_wcn_math_Mat3_rotate_wasm,_wcn_math_Mat3_rotate_x_wasm,_wcn_math_Mat3_rotate_y_wasm,_wcn_math_Mat3_rotate_z_wasm,_wcn_math_Mat3_rotation_wasm,_wcn_math_Mat3_rotation_x_wasm,_wcn_math_Mat3_rotation_y_wasm,_wcn_math_Mat3_rotation_z_wasm,_wcn_math_Mat3_scale3D_wasm,_wcn_math_Mat3_scale_wasm,_wcn_math_Mat3_scaling3D_wasm,_wcn_math_Mat3_scaling_wasm,_wcn_math_Mat3_set_axis_wasm,_wcn_math_Mat3_set_translation_wasm,_wcn_math_Mat3_set_wasm,_wcn_math_Mat3_set_with_index_wasm,_wcn_math_Mat3_sub_wasm,_wcn_math_Mat3_translate_wasm,_wcn_math_Mat3_translation_wasm,_wcn_math_Mat3_transpose_wasm,_wcn_math_Mat3_uniform_scale_3D_wasm,_wcn_math_Mat3_uniform_scale_wasm,_wcn_math_Mat3_uniform_scaling_3D_wasm,_wcn_math_Mat3_uniform_scaling_wasm,_wcn_math_Mat3_zero_wasm,_wcn_math_Mat4_add_wasm,_wcn_math_Mat4_aim_wasm,_wcn_math_Mat4_axis_rotate_wasm,_wcn_math_Mat4_axis_rotation_wasm,_wcn_math_Mat4_camera_aim_wasm,_wcn_math_Mat4_copy_wasm,_wcn_math_Mat4_determinant_wasm,_wcn_math_Mat4_equals_approximately_wasm,_wcn_math_Mat4_equals_wasm,_wcn_math_Mat4_from_mat3_wasm,_wcn_math_Mat4_from_quat_wasm,_wcn_math_Mat4_frustum_reverse_z_wasm,_wcn_math_Mat4_frustum_wasm,_wcn_math_Mat4_get_axis_wasm,_wcn_math_Mat4_get_scaling_wasm,_wcn_math_Mat4_get_translation_wasm,_wcn_math_Mat4_identity_wasm,_wcn_math_Mat4_inverse_wasm,_wcn_math_Mat4_look_at_wasm,_wcn_math_Mat4_multiply_scalar_wasm,_wcn_math_Mat4_multiply_wasm,_wcn_math_Mat4_negate_wasm,_wcn_math_Mat4_ortho_wasm,_wcn_math_Mat4_perspective_reverse_z_wasm,_wcn_math_Mat4_perspective_wasm,_wcn_math_Mat4_rotate_wasm,_wcn_math_Mat4_rotate_x_wasm,_wcn_math_Mat4_rotate_y_wasm,_wcn_math_Mat4_rotate_z_wasm,_wcn_math_Mat4_rotation_wasm,_wcn_math_Mat4_rotation_x_wasm,_wcn_math_Mat4_rotation_y_wasm,_wcn_math_Mat4_rotation_z_wasm,_wcn_math_Mat4_scale_wasm,_wcn_math_Mat4_scaling_wasm,_wcn_math_Mat4_set_axis_wasm,_wcn_math_Mat4_set_translation_wasm,_wcn_math_Mat4_set_wasm,_wcn_math_Mat4_set_with_index_wasm,_wcn_math_Mat4_sub_wasm,_wcn_math_Mat4_translate_wasm,_wcn_math_Mat4_translation_wasm,_wcn_math_Mat4_transpose_wasm,_wcn_math_Mat4_uniform_scale_wasm,_wcn_math_Mat4_uniform_scaling_wasm,_wcn_math_Mat4_zero_wasm,_wcn_math_Quat_add_wasm,_wcn_math_Quat_angle_wasm,_wcn_math_Quat_conjugate_wasm,_wcn_math_Quat_copy_wasm,_wcn_math_Quat_create_wasm,_wcn_math_Quat_div_scalar_wasm,_wcn_math_Quat_dot_wasm,_wcn_math_Quat_equals_approximately_wasm,_wcn_math_Quat_equals_wasm,_wcn_math_Quat_from_axis_angle_wasm,_wcn_math_Quat_from_euler_wasm,_wcn_math_Quat_from_mat3_wasm,_wcn_math_Quat_from_mat4_wasm,_wcn_math_Quat_identity_wasm,_wcn_math_Quat_inverse_wasm,_wcn_math_Quat_length_squared_wasm,_wcn_math_Quat_length_wasm,_wcn_math_Quat_lerp_wasm,_wcn_math_Quat_multiply_scalar_wasm,_wcn_math_Quat_multiply_wasm,_wcn_math_Quat_normalize_wasm,_wcn_math_Quat_rotate_x_wasm,_wcn_math_Quat_rotate_y_wasm,_wcn_math_Quat_rotate_z_wasm,_wcn_math_Quat_rotation_to_wasm,_wcn_math_Quat_scale_wasm,_wcn_math_Quat_set_w_wasm,_wcn_math_Quat_set_wasm,_wcn_math_Quat_set_x_wasm,_wcn_math_Quat_set_y_wasm,_wcn_math_Quat_set_z_wasm,_wcn_math_Quat_slerp_wasm,_wcn_math_Quat_sqlerp_wasm,_wcn_math_Quat_sub_wasm,_wcn_math_Quat_to_axis_angle_wasm,_wcn_math_Quat_zero_wasm,_wcn_math_Vec2_add_scaled_wasm,_wcn_math_Vec2_add_wasm,_wcn_math_Vec2_angle_wasm,_wcn_math_Vec2_ceil_wasm,_wcn_math_Vec2_clamp_wasm,_wcn_math_Vec2_copy_wasm,_wcn_math_Vec2_create_wasm,_wcn_math_Vec2_cross_wasm,_wcn_math_Vec2_distance_squared_wasm,_wcn_math_Vec2_distance_wasm,_wcn_math_Vec2_div_scalar_wasm,_wcn_math_Vec2_div_wasm,_wcn_math_Vec2_dot_wasm,_wcn_math_Vec2_equals_approximately_wasm,_wcn_math_Vec2_equals_wasm,_wcn_math_Vec2_floor_wasm,_wcn_math_Vec2_fmax_wasm,_wcn_math_Vec2_fmin_wasm,_wcn_math_Vec2_identity_wasm,_wcn_math_Vec2_inverse_wasm,_wcn_math_Vec2_length_squared_wasm,_wcn_math_Vec2_length_wasm,_wcn_math_Vec2_lerp_v_wasm,_wcn_math_Vec2_lerp_wasm,_wcn_math_Vec2_midpoint_wasm,_wcn_math_Vec2_multiply_scalar_wasm,_wcn_math_Vec2_multiply_wasm,_wcn_math_Vec2_negate_wasm,_wcn_math_Vec2_normalize_wasm,_wcn_math_Vec2_random_wasm,_wcn_math_Vec2_rotate_wasm,_wcn_math_Vec2_round_wasm,_wcn_math_Vec2_scale_wasm,_wcn_math_Vec2_set_length_wasm,_wcn_math_Vec2_set_wasm,_wcn_math_Vec2_set_x_wasm,_wcn_math_Vec2_set_y_wasm,_wcn_math_Vec2_sub_wasm,_wcn_math_Vec2_transform_mat3_wasm,_wcn_math_Vec2_transform_mat4_wasm,_wcn_math_Vec2_truncate_wasm,_wcn_math_Vec2_zero_wasm,_wcn_math_Vec2xN_free,_wcn_math_Vec2xN_get,_wcn_math_Vec2xN_get_count,_wcn_math_Vec2xN_get_xarray,_wcn_math_Vec2xN_get_yarray,_wcn_math_Vec2xN_malloc,_wcn_math_Vec2xN_set,_wcn_math_Vec2xN_set_by_xy,_wcn_math_Vec3_WithAngleAxis_get_angle,_wcn_math_Vec3_add_scaled_wasm,_wcn_math_Vec3_add_wasm,_wcn_math_Vec3_angle_wasm,_wcn_math_Vec3_ceil_wasm,_wcn_math_Vec3_clamp_wasm,_wcn_math_Vec3_copy_wasm,_wcn_math_Vec3_create_wasm,_wcn_math_Vec3_cross_wasm,_wcn_math_Vec3_distance_squared_wasm,_wcn_math_Vec3_distance_wasm,_wcn_math_Vec3_div_scalar_wasm,_wcn_math_Vec3_div_wasm,_wcn_math_Vec3_dot_wasm,_wcn_math_Vec3_equals_approximately_wasm,_wcn_math_Vec3_equals_wasm,_wcn_math_Vec3_floor_wasm,_wcn_math_Vec3_fmax_wasm,_wcn_math_Vec3_fmin_wasm,_wcn_math_Vec3_get_axis_wasm,_wcn_math_Vec3_get_scale_wasm,_wcn_math_Vec3_get_translation_wasm,_wcn_math_Vec3_identity_wasm,_wcn_math_Vec3_inverse_wasm,_wcn_math_Vec3_length_squared_wasm,_wcn_math_Vec3_length_wasm,_wcn_math_Vec3_lerp_v_wasm,_wcn_math_Vec3_lerp_wasm,_wcn_math_Vec3_midpoint_wasm,_wcn_math_Vec3_multiply_scalar_wasm,_wcn_math_Vec3_multiply_wasm,_wcn_math_Vec3_negate_wasm,_wcn_math_Vec3_normalize_wasm,_wcn_math_Vec3_random_wasm,_wcn_math_Vec3_rotate_x_wasm,_wcn_math_Vec3_rotate_y_wasm,_wcn_math_Vec3_rotate_z_wasm,_wcn_math_Vec3_round_wasm,_wcn_math_Vec3_scale_wasm,_wcn_math_Vec3_set_length_wasm,_wcn_math_Vec3_set_wasm,_wcn_math_Vec3_set_x_wasm,_wcn_math_Vec3_set_y_wasm,_wcn_math_Vec3_set_z_wasm,_wcn_math_Vec3_sub_wasm,_wcn_math_Vec3_transform_mat3_wasm,_wcn_math_Vec3_transform_mat4_upper3x3_wasm,_wcn_math_Vec3_transform_mat4_wasm,_wcn_math_Vec3_transform_quat_wasm,_wcn_math_Vec3_truncate_wasm,_wcn_math_Vec3_zero_wasm,_wcn_math_Vec4_add_scaled_wasm,_wcn_math_Vec4_add_wasm,_wcn_math_Vec4_ceil_wasm,_wcn_math_Vec4_clamp_wasm,_wcn_math_Vec4_copy_wasm,_wcn_math_Vec4_create_wasm,_wcn_math_Vec4_distance_squared_wasm,_wcn_math_Vec4_distance_wasm,_wcn_math_Vec4_div_scalar_wasm,_wcn_math_Vec4_div_wasm,_wcn_math_Vec4_dot_wasm,_wcn_math_Vec4_equals_approximately_wasm,_wcn_math_Vec4_equals_wasm,_wcn_math_Vec4_floor_wasm,_wcn_math_Vec4_fmax_wasm,_wcn_math_Vec4_fmin_wasm,_wcn_math_Vec4_identity_wasm,_wcn_math_Vec4_inverse_wasm,_wcn_math_Vec4_length_squared_wasm,_wcn_math_Vec4_length_wasm,_wcn_math_Vec4_lerp_v_wasm,_wcn_math_Vec4_lerp_wasm,_wcn_math_Vec4_midpoint_wasm,_wcn_math_Vec4_multiply_scalar_wasm,_wcn_math_Vec4_multiply_wasm,_wcn_math_Vec4_negate_wasm,_wcn_math_Vec4_normalize_wasm,_wcn_math_Vec4_round_wasm,_wcn_math_Vec4_set_length_wasm,_wcn_math_Vec4_set_w_wasm,_wcn_math_Vec4_set_wasm,_wcn_math_Vec4_set_x_wasm,_wcn_math_Vec4_set_y_wasm,_wcn_math_Vec4_set_z_wasm,_wcn_math_Vec4_sub_wasm,_wcn_math_Vec4_transform_mat4_wasm,_wcn_math_Vec4_truncate_wasm,_wcn_math_Vec4_zero_wasm,_wcn_math_Vec4xN_free
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
