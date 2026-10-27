# TODO — 219 个函数的确定性测试

> 每行一个 Task，三个后端各跑一遍。期望值由 `.local/tools/oracle_*.js`
> （纯 JS，binary32 归一）独立算出，再由 `.local/tools/gen_branch.js` 生成 `branch.c` 的断言。
>
> * `C-`   = C 版，no-SIMD（`/DWMATH_DISABLE_SIMD`）
> * `C+`   = C 版，SIMD（SSE4.2 基线 + `WCN_ENABLE_*`；AVX2 默认已关闭）
> * `Rust` = Rust 移植，经其 C ABI 链接（LLVM 自动向量化，无手写 SIMD）
>
> 缺陷详情：[LOGIC_BUGS.md](LOGIC_BUGS.md)（逻辑）、[SIMD_PERF_BUGS.md](SIMD_PERF_BUGS.md)（性能）
>
> 复现：先 `cmd /c "tools\msvc_env.bat cmd /c tools\build_libs_msvc.bat"`，再
> `cmd /c "tools\msvc_env.bat cmd /c tools\run3.bat nosimd"`（simd / rust 同理）


## Vec2

- [x] `001` `wcn_math_Vec2_create`  C-PASS C+PASS Rust-PASS
- [x] `002` `wcn_math_Vec2_set`  C-PASS C+PASS Rust-PASS
- [x] `003` `wcn_math_Vec2_copy`  C-PASS C+PASS Rust-PASS
- [x] `004` `wcn_math_Vec2_zero`  C-PASS C+PASS Rust-PASS
- [x] `005` `wcn_math_Vec2_identity`  C-PASS C+PASS Rust-PASS
- [x] `006` `wcn_math_Vec2_ceil`  C-PASS C+PASS Rust-PASS
- [x] `007` `wcn_math_Vec2_floor`  C-PASS C+PASS Rust-PASS
- [x] `008` `wcn_math_Vec2_round`  C-PASS C+PASS Rust-PASS
- [x] `009` `wcn_math_Vec2_clamp`  C-PASS C+PASS Rust-PASS
- [x] `010` `wcn_math_Vec2_dot`  C-PASS C+PASS Rust-PASS
- [x] `011` `wcn_math_Vec2_add`  C-PASS C+PASS Rust-PASS
- [x] `012` `wcn_math_Vec2_add_scaled`  C-PASS C+PASS Rust-PASS
- [x] `013` `wcn_math_Vec2_sub`  C-PASS C+PASS Rust-PASS
- [x] `014` `wcn_math_Vec2_angle`  C-PASS C+PASS Rust-PASS
- [x] `015` `wcn_math_Vec2_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `016` `wcn_math_Vec2_equals`  C-PASS C+PASS Rust-PASS
- [x] `017` `wcn_math_Vec2_lerp`  C-PASS C+PASS Rust-PASS
- [x] `018` `wcn_math_Vec2_lerp_v`  C-PASS C+PASS Rust-PASS
- [x] `019` `wcn_math_Vec2_fmax`  C-PASS C+PASS Rust-PASS
- [x] `020` `wcn_math_Vec2_fmin`  C-PASS C+PASS Rust-PASS
- [x] `021` `wcn_math_Vec2_multiply`  C-PASS C+PASS Rust-PASS
- [x] `022` `wcn_math_Vec2_div_scalar`  C-PASS C+PASS Rust-PASS
- [x] `023` `wcn_math_Vec2_div`  C-PASS C+PASS Rust-PASS
- [x] `024` `wcn_math_Vec2_inverse`  C-PASS C+PASS Rust-PASS
- [x] `025` `wcn_math_Vec2_cross`  C-PASS C+PASS Rust-PASS
- [x] `026` `wcn_math_Vec2_length`  C-PASS C+PASS Rust-PASS
- [x] `027` `wcn_math_Vec2_length_squared`  C-PASS C+PASS Rust-PASS
- [x] `028` `wcn_math_Vec2_distance`  C-PASS C+PASS Rust-PASS
- [x] `029` `wcn_math_Vec2_distance_squared`  C-PASS C+PASS Rust-PASS
- [x] `030` `wcn_math_Vec2_negate`  C-PASS C+PASS Rust-PASS
- [x] `031` `wcn_math_Vec2_random`  C-PASS C+PASS Rust-PASS
- [x] `032` `wcn_math_Vec2_normalize`  C-PASS C+PASS Rust-PASS
- [x] `033` `wcn_math_Vec2_rotate`  C-PASS C+PASS Rust-PASS
- [x] `034` `wcn_math_Vec2_set_length`  C-PASS C+PASS Rust-PASS
- [x] `035` `wcn_math_Vec2_truncate`  C-PASS C+PASS Rust-PASS
- [x] `036` `wcn_math_Vec2_transform_mat4`  C-PASS C+PASS Rust-PASS
- [x] `037` `wcn_math_Vec2_transform_mat3`  C-PASS C+PASS Rust-PASS
- [x] `038` `wcn_math_Vec2_scale`  C-PASS C+PASS Rust-PASS

## Vec3

- [x] `039` `wcn_math_Vec3_create`  C-PASS C+PASS Rust-PASS
- [x] `040` `wcn_math_Vec3_copy`  C-PASS C+PASS Rust-PASS
- [x] `041` `wcn_math_Vec3_set`  C-PASS C+PASS Rust-PASS
- [x] `042` `wcn_math_Vec3_zero`  C-PASS C+PASS Rust-PASS
- [x] `043` `wcn_math_Vec3_identity`  C-PASS C+PASS Rust-PASS
- [x] `044` `wcn_math_Vec3_ceil`  C-PASS C+PASS Rust-PASS
- [x] `045` `wcn_math_Vec3_floor`  C-PASS C+PASS Rust-PASS
- [x] `046` `wcn_math_Vec3_round`  C-PASS C+PASS Rust-PASS
- [x] `047` `wcn_math_Vec3_dot`  C-PASS C+PASS Rust-PASS
- [x] `048` `wcn_math_Vec3_cross`  C-PASS C+PASS Rust-PASS
- [x] `049` `wcn_math_Vec3_normalize`  C-PASS C+PASS Rust-PASS
- [x] `050` `wcn_math_Vec3_clamp`  C-PASS C+PASS Rust-PASS
- [x] `051` `wcn_math_Vec3_add`  C-PASS C+PASS Rust-PASS
- [x] `052` `wcn_math_Vec3_add_scaled`  C-PASS C+PASS Rust-PASS
- [x] `053` `wcn_math_Vec3_sub`  C-PASS C+PASS Rust-PASS
- [x] `054` `wcn_math_Vec3_angle`  C-PASS C+PASS Rust-PASS
- [x] `055` `wcn_math_Vec3_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `056` `wcn_math_Vec3_equals`  C-PASS C+PASS Rust-PASS
- [x] `057` `wcn_math_Vec3_lerp`  C-PASS C+PASS Rust-PASS
- [x] `058` `wcn_math_Vec3_lerp_v`  C-PASS C+PASS Rust-PASS
- [x] `059` `wcn_math_Vec3_fmax`  C-PASS C+PASS Rust-PASS
- [x] `060` `wcn_math_Vec3_fmin`  C-PASS C+PASS Rust-PASS
- [x] `061` `wcn_math_Vec3_multiply`  C-PASS C+PASS Rust-PASS
- [x] `062` `wcn_math_Vec3_multiply_scalar`  C-PASS C+PASS Rust-PASS
- [x] `063` `wcn_math_Vec3_div`  C-PASS C+PASS Rust-PASS
- [x] `064` `wcn_math_Vec3_div_scalar`  C-PASS C+PASS Rust-PASS
- [x] `065` `wcn_math_Vec3_inverse`  C-PASS C+PASS Rust-PASS
- [x] `066` `wcn_math_Vec3_distance`  C-PASS C+PASS Rust-PASS
- [x] `067` `wcn_math_Vec3_distance_squared`  C-PASS C+PASS Rust-PASS
- [x] `068` `wcn_math_Vec3_negate`  C-PASS C+PASS Rust-PASS
- [x] `069` `wcn_math_Vec3_random`  C-PASS C+PASS Rust-PASS
- [x] `070` `wcn_math_Vec3_set_length`  C-PASS C+PASS Rust-PASS
- [x] `071` `wcn_math_Vec3_truncate`  C-PASS C+PASS Rust-PASS
- [x] `072` `wcn_math_Vec3_transform_mat4`  C-PASS C+PASS Rust-PASS
- [x] `073` `wcn_math_Vec3_transform_mat4_upper3x3`  C-PASS C+PASS Rust-PASS
- [x] `074` `wcn_math_Vec3_transform_mat3`  C-PASS C+PASS Rust-PASS
- [x] `075` `wcn_math_Vec3_transform_quat`  C-PASS C+PASS Rust-PASS
- [x] `076` `wcn_math_Vec3_get_translation`  C-PASS C+PASS Rust-PASS
- [x] `077` `wcn_math_Vec3_get_axis`  C-PASS C+PASS Rust-PASS
- [x] `078` `wcn_math_Vec3_get_scale`  C-PASS C+PASS Rust-PASS
- [x] `079` `wcn_math_Vec3_rotate_x`  C-PASS C+PASS Rust-PASS
- [x] `080` `wcn_math_Vec3_rotate_y`  C-PASS C+PASS Rust-PASS
- [x] `081` `wcn_math_Vec3_rotate_z`  C-PASS C+PASS Rust-PASS

## Vec4

- [x] `082` `wcn_math_Vec4_create`  C-PASS C+PASS Rust-PASS
- [x] `083` `wcn_math_Vec4_set`  C-PASS C+PASS Rust-PASS
- [x] `084` `wcn_math_Vec4_copy`  C-PASS C+PASS Rust-PASS
- [x] `085` `wcn_math_Vec4_zero`  C-PASS C+PASS Rust-PASS
- [x] `086` `wcn_math_Vec4_identity`  C-PASS C+PASS Rust-PASS
- [x] `087` `wcn_math_Vec4_ceil`  C-PASS C+PASS Rust-PASS
- [x] `088` `wcn_math_Vec4_floor`  C-PASS C+PASS Rust-PASS
- [x] `089` `wcn_math_Vec4_round`  C-PASS C+PASS Rust-PASS
- [x] `090` `wcn_math_Vec4_clamp`  C-PASS C+PASS Rust-PASS
- [x] `091` `wcn_math_Vec4_add`  C-PASS C+PASS Rust-PASS
- [x] `092` `wcn_math_Vec4_add_scaled`  C-PASS C+PASS Rust-PASS
- [x] `093` `wcn_math_Vec4_sub`  C-PASS C+PASS Rust-PASS
- [x] `094` `wcn_math_Vec4_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `095` `wcn_math_Vec4_equals`  C-PASS C+PASS Rust-PASS
- [x] `096` `wcn_math_Vec4_lerp`  C-PASS C+PASS Rust-PASS
- [x] `097` `wcn_math_Vec4_lerp_v`  C-PASS C+PASS Rust-PASS
- [x] `098` `wcn_math_Vec4_fmax`  C-PASS C+PASS Rust-PASS
- [x] `099` `wcn_math_Vec4_fmin`  C-PASS C+PASS Rust-PASS
- [x] `100` `wcn_math_Vec4_multiply`  C-PASS C+PASS Rust-PASS
- [x] `101` `wcn_math_Vec4_multiply_scalar`  C-PASS C+PASS Rust-PASS
- [x] `102` `wcn_math_Vec4_div`  C-PASS C+PASS Rust-PASS
- [x] `103` `wcn_math_Vec4_div_scalar`  C-PASS C+PASS Rust-PASS
- [x] `104` `wcn_math_Vec4_inverse`  C-PASS C+PASS Rust-PASS
- [x] `105` `wcn_math_Vec4_dot`  C-PASS C+PASS Rust-PASS
- [x] `106` `wcn_math_Vec4_length_squared`  C-PASS C+PASS Rust-PASS
- [x] `107` `wcn_math_Vec4_distance_squared`  C-PASS C+PASS Rust-PASS
- [x] `108` `wcn_math_Vec4_normalize`  C-PASS C+PASS Rust-PASS
- [x] `109` `wcn_math_Vec4_negate`  C-PASS C+PASS Rust-PASS
- [x] `110` `wcn_math_Vec4_set_length`  C-PASS C+PASS Rust-PASS
- [x] `111` `wcn_math_Vec4_truncate`  C-PASS C+PASS Rust-PASS
- [x] `112` `wcn_math_Vec4_transform_mat4`  C-PASS C+PASS Rust-PASS

## Mat3

- [x] `113` `wcn_math_Mat3_identity`  C-PASS C+PASS Rust-PASS
- [x] `114` `wcn_math_Mat3_zero`  C-PASS C+PASS Rust-PASS
- [x] `115` `wcn_math_Mat3_create`  C-PASS C+PASS Rust-PASS
- [x] `116` `wcn_math_Mat3_copy`  C-PASS C+PASS Rust-PASS
- [x] `117` `wcn_math_Mat3_equals`  C-PASS C+PASS Rust-PASS
- [x] `118` `wcn_math_Mat3_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `119` `wcn_math_Mat3_set`  C-PASS C+PASS Rust-PASS
- [x] `120` `wcn_math_Mat3_transpose`  C-PASS C+PASS Rust-PASS
- [x] `121` `wcn_math_Mat3_add`  C-PASS C+PASS Rust-PASS
- [x] `122` `wcn_math_Mat3_sub`  C-PASS C+PASS Rust-PASS
- [x] `123` `wcn_math_Mat3_multiply_scalar`  C-PASS C+PASS Rust-PASS
- [x] `124` `wcn_math_Mat3_inverse`  C-PASS C+PASS Rust-PASS
- [x] `125` `wcn_math_Mat3_multiply`  C-PASS C+PASS Rust-PASS
- [x] `126` `wcn_math_Mat3_determinant`  C-PASS C+PASS Rust-PASS
- [x] `127` `wcn_math_Mat3_from_mat4`  C-PASS C+PASS Rust-PASS
- [x] `128` `wcn_math_Mat3_from_quat`  C-PASS C+PASS Rust-PASS
- [x] `129` `wcn_math_Mat3_rotate`  C-PASS C+PASS Rust-PASS
- [x] `130` `wcn_math_Mat3_rotate_x`  C-PASS C+PASS Rust-PASS
- [x] `131` `wcn_math_Mat3_rotate_y`  C-PASS C+PASS Rust-PASS
- [x] `132` `wcn_math_Mat3_rotation`  C-PASS C+PASS Rust-PASS
- [x] `133` `wcn_math_Mat3_rotation_x`  C-PASS C+PASS Rust-PASS
- [x] `134` `wcn_math_Mat3_rotation_y`  C-PASS C+PASS Rust-PASS
- [x] `135` `wcn_math_Mat3_get_axis`  C-PASS C+PASS Rust-PASS
- [x] `136` `wcn_math_Mat3_set_axis`  C-PASS C+PASS Rust-PASS
- [x] `137` `wcn_math_Mat3_get_scaling`  C-PASS C+PASS Rust-PASS
- [x] `138` `wcn_math_Mat3_get_3D_scaling`  C-PASS C+PASS Rust-PASS
- [x] `139` `wcn_math_Mat3_get_translation`  C-PASS C+PASS Rust-PASS
- [x] `140` `wcn_math_Mat3_set_translation`  C-PASS C+PASS Rust-PASS
- [x] `141` `wcn_math_Mat3_translation`  C-PASS C+PASS Rust-PASS
- [x] `142` `wcn_math_Mat3_translate`  C-PASS C+PASS Rust-PASS
- [x] `143` `wcn_math_Mat3_scale`  C-PASS C+PASS Rust-PASS
- [x] `144` `wcn_math_Mat3_scale3D`  C-PASS C+PASS Rust-PASS
- [x] `145` `wcn_math_Mat3_scaling`  C-PASS C+PASS Rust-PASS
- [x] `146` `wcn_math_Mat3_scaling3D`  C-PASS C+PASS Rust-PASS
- [x] `147` `wcn_math_Mat3_uniform_scale`  C-PASS C+PASS Rust-PASS
- [x] `148` `wcn_math_Mat3_uniform_scale_3D`  C-PASS C+PASS Rust-PASS
- [x] `149` `wcn_math_Mat3_uniform_scaling`  C-PASS C+PASS Rust-PASS
- [x] `150` `wcn_math_Mat3_uniform_scaling_3D`  C-PASS C+PASS Rust-PASS

## Mat4

- [x] `151` `wcn_math_Mat4_identity`  C-PASS C+PASS Rust-PASS
- [x] `152` `wcn_math_Mat4_zero`  C-PASS C+PASS Rust-PASS
- [x] `153` `wcn_math_Mat4_create`  C-PASS C+PASS Rust-PASS
- [x] `154` `wcn_math_Mat4_copy`  C-PASS C+PASS Rust-PASS
- [x] `155` `wcn_math_Mat4_set`  C-PASS C+PASS Rust-PASS
- [x] `156` `wcn_math_Mat4_negate`  C-PASS C+PASS Rust-PASS
- [x] `157` `wcn_math_Mat4_equals`  C-PASS C+PASS Rust-PASS
- [x] `158` `wcn_math_Mat4_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `159` `wcn_math_Mat4_add`  C-PASS C+PASS Rust-PASS
- [x] `160` `wcn_math_Mat4_sub`  C-PASS C+PASS Rust-PASS
- [x] `161` `wcn_math_Mat4_multiply_scalar`  C-PASS C+PASS Rust-PASS
- [x] `162` `wcn_math_Mat4_multiply`  C-PASS C+PASS Rust-PASS
- [x] `163` `wcn_math_Mat4_inverse`  C-PASS C+PASS Rust-PASS
- [x] `164` `wcn_math_Mat4_transpose`  C-PASS C+PASS Rust-PASS
- [x] `165` `wcn_math_Mat4_determinant`  C-PASS C+PASS Rust-PASS
- [x] `166` `wcn_math_Mat4_aim`  C-PASS C+PASS Rust-PASS
- [x] `167` `wcn_math_Mat4_look_at`  C-PASS C+PASS Rust-PASS
- [x] `168` `wcn_math_Mat4_ortho`  C-PASS C+PASS Rust-PASS
- [x] `169` `wcn_math_Mat4_from_mat3`  C-PASS C+PASS Rust-PASS
- [x] `170` `wcn_math_Mat4_from_quat`  C-PASS C+PASS Rust-PASS
- [x] `171` `wcn_math_Mat4_axis_rotate`  C-PASS C+PASS Rust-PASS
- [x] `172` `wcn_math_Mat4_axis_rotation`  C-PASS C+PASS Rust-PASS
- [x] `173` `wcn_math_Mat4_camera_aim`  C-PASS C+PASS Rust-PASS
- [x] `174` `wcn_math_Mat4_frustum`  C-PASS C+PASS Rust-PASS
- [x] `175` `wcn_math_Mat4_frustum_reverse_z`  C-PASS C+PASS Rust-PASS
- [x] `176` `wcn_math_Mat4_set_axis`  C-PASS C+PASS Rust-PASS
- [x] `177` `wcn_math_Mat4_set_translation`  C-PASS C+PASS Rust-PASS
- [x] `178` `wcn_math_Mat4_translation`  C-PASS C+PASS Rust-PASS
- [x] `179` `wcn_math_Mat4_perspective`  C-PASS C+PASS Rust-PASS
- [x] `180` `wcn_math_Mat4_perspective_reverse_z`  C-PASS C+PASS Rust-PASS
- [x] `181` `wcn_math_Mat4_translate`  C-PASS C+PASS Rust-PASS
- [x] `182` `wcn_math_Mat4_rotate_x`  C-PASS C+PASS Rust-PASS
- [x] `183` `wcn_math_Mat4_rotate_y`  C-PASS C+PASS Rust-PASS
- [x] `184` `wcn_math_Mat4_rotate_z`  C-PASS C+PASS Rust-PASS
- [x] `185` `wcn_math_Mat4_rotation_x`  C-PASS C+PASS Rust-PASS
- [x] `186` `wcn_math_Mat4_rotation_y`  C-PASS C+PASS Rust-PASS
- [x] `187` `wcn_math_Mat4_rotation_z`  C-PASS C+PASS Rust-PASS
- [x] `188` `wcn_math_Mat4_scale`  C-PASS C+PASS Rust-PASS
- [x] `189` `wcn_math_Mat4_get_scaling`  C-PASS C+PASS Rust-PASS
- [x] `190` `wcn_math_Mat4_scaling`  C-PASS C+PASS Rust-PASS
- [x] `191` `wcn_math_Mat4_uniform_scale`  C-PASS C+PASS Rust-PASS
- [x] `192` `wcn_math_Mat4_uniform_scaling`  C-PASS C+PASS Rust-PASS

## Quat

- [x] `193` `wcn_math_Quat_zero`  C-PASS C+PASS Rust-PASS
- [x] `194` `wcn_math_Quat_identity`  C-PASS C+PASS Rust-PASS
- [x] `195` `wcn_math_Quat_create`  C-PASS C+PASS Rust-PASS
- [x] `196` `wcn_math_Quat_copy`  C-PASS C+PASS Rust-PASS
- [x] `197` `wcn_math_Quat_dot`  C-PASS C+PASS Rust-PASS
- [x] `198` `wcn_math_Quat_slerp`  C-PASS C+PASS Rust-PASS
- [x] `199` `wcn_math_Quat_sqlerp`  C-PASS C+PASS Rust-PASS
- [x] `200` `wcn_math_Quat_normalize`  C-PASS C+PASS Rust-PASS
- [x] `201` `wcn_math_Quat_equals_approximately`  C-PASS C+PASS Rust-PASS
- [x] `202` `wcn_math_Quat_equals`  C-PASS C+PASS Rust-PASS
- [x] `203` `wcn_math_Quat_angle`  C-PASS C+PASS Rust-PASS
- [x] `204` `wcn_math_Quat_rotation_to`  C-PASS C+PASS Rust-PASS
- [x] `205` `wcn_math_Quat_multiply`  C-PASS C+PASS Rust-PASS
- [x] `206` `wcn_math_Quat_multiply_scalar`  C-PASS C+PASS Rust-PASS
- [x] `207` `wcn_math_Quat_sub`  C-PASS C+PASS Rust-PASS
- [x] `208` `wcn_math_Quat_add`  C-PASS C+PASS Rust-PASS
- [x] `209` `wcn_math_Quat_inverse`  C-PASS C+PASS Rust-PASS
- [x] `210` `wcn_math_Quat_conjugate`  C-PASS C+PASS Rust-PASS
- [x] `211` `wcn_math_Quat_div_scalar`  C-PASS C+PASS Rust-PASS
- [x] `212` `wcn_math_Quat_from_axis_angle`  C-PASS C+PASS Rust-PASS
- [x] `213` `wcn_math_Quat_to_axis_angle`  C-PASS C+PASS Rust-PASS
- [x] `214` `wcn_math_Quat_from_mat4`  C-PASS C+PASS Rust-PASS
- [x] `215` `wcn_math_Quat_from_mat3`  C-PASS C+PASS Rust-PASS
- [x] `216` `wcn_math_Quat_from_euler`  C-PASS C+PASS Rust-PASS
- [x] `217` `wcn_math_Quat_rotate_x`  C-PASS C+PASS Rust-PASS
- [x] `218` `wcn_math_Quat_rotate_y`  C-PASS C+PASS Rust-PASS
- [x] `219` `wcn_math_Quat_rotate_z`  C-PASS C+PASS Rust-PASS

---

## 统计

| 后端 | 通过 | 失败 |
|---|---|---|
| C no-SIMD | 219 | 0 |
| C SIMD | 219 | 0 |
| Rust | 219 | 0 |
| **三后端全通** | **219** | 0 |

- Task 总数：**219**
- 三后端均已实现并运行：**219 / 219**
- 三后端全部通过：**219 / 219**
- 至少一个后端失败：**0**
