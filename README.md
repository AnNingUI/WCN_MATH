# WCN_MATH

A C math library for 3D graphics, originally ported from [wgpu-matrix](https://github.com/greggman/wgpu-matrix).

## Overview

Designed to provide SIMD-accelerated math operations for CPU-based computations. Independent of the main WCN project and GPU rendering pipeline.

## Features

- **Types**: Vec2/Vec3/Vec4, Mat3/Mat4, Quat
- **SIMD**: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, AVX, AVX2, FMA, NEON, RISC-V Vector, LoongArch LSX
- **WASM**: WebAssembly compilation support
- **API**: destination-out-param (the only API; see [docs/MIGRATION.md](docs/MIGRATION.md))

## API shape

Every operation takes its destination as the first argument and writes into it.
Functions whose result is a scalar (`dot`, `length`, `angle`, `equals`, …) return it
directly instead.

```c
T$(Vec3) out;
WMATH_ADD(Vec3)(&out, a, b);        // struct result -> written into `out`
float d = WMATH_DOT(Vec3)(a, b);    // scalar result -> returned
```

**Why not return-by-value:** this needs no implicit copies, keeps every operation on the
stack, and maps directly onto SIMD. A second return-by-value API used to exist alongside
it; [docs/MIGRATION.md](docs/MIGRATION.md) explains why it was removed and how to rewrite
its call sites.

## Building

```bash
# Native
cmake -B build && cmake --build build

# WebAssembly
cmake -B build-wasm -DWCN_MATH_BUILD_WASM=ON && cmake --build build-wasm

# Building examples
python make.example.py -name B.c -cc gcc
python make.example.py -name DST_MOD_CPP.cpp -cc g++

# run examples
./build/examples/B
# in windows
.\build\examples\B.exe
```

## Usage

Include `WCN/WCN_Math.h` (a thin alias for `WCN/WCN_MATH_DST.h`) and link against `libwcn_math.a`.

> **One API only.** There used to be a second, return-by-value API declared inline by
> `WCN_Math.h`, selected by `WCN_USE_DST_MODE`. Both libraries implemented the same math
> under the same symbol names with different calling conventions, so a mismatched build
> linked cleanly and then corrupted the stack. Only the destination-out-param API is
> maintained now; see [docs/MIGRATION.md](docs/MIGRATION.md) for the one-line rewrite rules.

**C Example:**
```c
#include "WCN/WCN_Math.h"
#include <stdio.h>

int main() {
    T$(Vec3) v1 = INIT$(Vec3, .v_x = 1.0f, .v_y = 2.0f, .v_z = 3.0f);
    T$(Vec3) v2 = INIT$(Vec3, .v_x = 4.0f, .v_y = 5.0f, .v_z = 6.0f);

    // Scalar-returning ops read the same as before
    float dot = WMATH_DOT(Vec3)(v1, v2);

    // Destination-out-param ops write into a value you own
    T$(Vec3) cross;      WMATH_CROSS(Vec3)(&cross, v1, v2);
    T$(Vec3) normalized; WMATH_NORMALIZE(Vec3)(&normalized, v1);

    printf("dot = %f\n", dot);
    return 0;
}
```

**C++ Wrapper:**
```cpp
#include "WCN/WCN_MATH_DST_CPP.hpp"
using namespace WCN::Math;

Mat4 m1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
Mat4 m2 = Mat4::scaling({2.0f, 2.0f, 2.0f});
Mat4 result = m1 * m2;
```

See `examples/` directory for more. `examples/R.c` is the regression test — it exits
non-zero when any check fails.

## Relationship with WCN

Like [WCN_SIMD](https://github.com/AnNingUI/WCN_SIMD), it is independent with no direct dependency on the WCN main project, but remains part of the WCN family.

## TODO

- [ ] DST Mode CPP wrapper

- [ ] SoA (Structure of Arrays) types for batch processing
  - [ ] Vec2xN
    - [x] WMATH_ADD
    - [ ] WMATH_CREATE, WMATH_SET, WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY
    - [ ] WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND, WMATH_CLAMP
    - [x] WMATH_SUB, WMATH_ADD_SCALED
    - [ ] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_DIV, WMATH_DIV_SCALAR
    - [ ] WMATH_DOT, WMATH_CROSS, WMATH_LENGTH, WMATH_LENGTH_SQ
    - [ ] WMATH_DISTANCE, WMATH_DISTANCE_SQ, WMATH_NORMALIZE
    - [ ] WMATH_NEGATE, WMATH_INVERSE, WMATH_LERP, WMATH_LERP_V
    - [ ] WMATH_FMAX, WMATH_FMIN, WMATH_ANGLE
    - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
    - [ ] WMATH_RANDOM, WMATH_ROTATE, WMATH_SET_LENGTH, WMATH_TRUNCATE, WMATH_MIDPOINT
  - [ ] Vec3xN
    - [ ] WMATH_CREATE, WMATH_COPY, WMATH_SET, WMATH_ZERO
    - [ ] WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND, WMATH_CLAMP
    - [ ] WMATH_ADD, WMATH_ADD_SCALED, WMATH_SUB
    - [ ] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_DIV, WMATH_DIV_SCALAR
    - [ ] WMATH_DOT, WMATH_CROSS, WMATH_LENGTH, WMATH_LENGTH_SQ
    - [ ] WMATH_DISTANCE, WMATH_DISTANCE_SQ, WMATH_NORMALIZE
    - [ ] WMATH_NEGATE, WMATH_INVERSE, WMATH_LERP, WMATH_LERP_V
    - [ ] WMATH_FMAX, WMATH_FMIN, WMATH_ANGLE
    - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
    - [ ] WMATH_RANDOM, WMATH_SET_LENGTH, WMATH_TRUNCATE, WMATH_MIDPOINT
  - [ ] Vec4xN
    - [ ] WMATH_CREATE, WMATH_SET, WMATH_COPY, WMATH_ZERO, WMATH_IDENTITY
    - [ ] WMATH_CEIL, WMATH_FLOOR, WMATH_ROUND, WMATH_CLAMP
    - [ ] WMATH_ADD, WMATH_ADD_SCALED, WMATH_SUB
    - [ ] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_DIV, WMATH_DIV_SCALAR
    - [ ] WMATH_DOT, WMATH_LENGTH, WMATH_LENGTH_SQ
    - [ ] WMATH_DISTANCE, WMATH_DISTANCE_SQ, WMATH_NORMALIZE
    - [ ] WMATH_NEGATE, WMATH_INVERSE, WMATH_LERP, WMATH_LERP_V
    - [ ] WMATH_FMAX, WMATH_FMIN
    - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
    - [ ] WMATH_SET_LENGTH, WMATH_TRUNCATE, WMATH_MIDPOINT
  - [ ] QuatxN
    - [ ] WMATH_ZERO, WMATH_IDENTITY, WMATH_CREATE, WMATH_SET, WMATH_COPY
    - [ ] WMATH_DOT, WMATH_LERP, WMATH_SLERP, WMATH_SQLERP
    - [ ] WMATH_LENGTH, WMATH_LENGTH_SQ, WMATH_NORMALIZE
    - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY, WMATH_ANGLE
    - [ ] WMATH_ROTATION_TO
    - [ ] WMATH_MULTIPLY, WMATH_MULTIPLY_SCALAR, WMATH_SUB, WMATH_ADD
    - [ ] WMATH_INVERSE, WMATH_CONJUGATE, WMATH_DIV_SCALAR
    - [ ] WMATH_FROM_AXIS_ANGLE, WMATH_TO_AXIS_ANGLE
    - [ ] WMATH_FROM_EULER, WMATH_FROM_MAT3, WMATH_FROM_MAT4
  - [ ] Mat3xN
    - [ ] WMATH_IDENTITY, WMATH_ZERO, WMATH_CREATE, WMATH_COPY
    - [ ] WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY, WMATH_SET
    - [ ] WMATH_NEGATE, WMATH_TRANSPOSE
    - [ ] WMATH_ADD, WMATH_SUB, WMATH_MULTIPLY_SCALAR
    - [ ] WMATH_INVERSE, WMATH_MULTIPLY
    - [ ] WMATH_DETERMINANT
    - [ ] WMATH_FROM_MAT4, WMATH_FROM_QUAT
    - [ ] WMATH_ROTATE, WMATH_ROTATE_X, WMATH_ROTATE_Y, WMATH_ROTATE_Z
    - [ ] WMATH_ROTATION, WMATH_ROTATION_X, WMATH_ROTATION_Y, WMATH_ROTATION_Z
    - [ ] WMATH_GET_AXIS, WMATH_SET_AXIS
    - [ ] WMATH_GET_SCALING, WMATH_GET_3D_SCALING
    - [ ] WMATH_GET_TRANSLATION, WMATH_SET_TRANSLATION
    - [ ] WMATH_TRANSLATION, WMATH_TRANSLATE
    - [ ] WMATH_SCALE, WMATH_SCALE3D
    - [ ] WMATH_SCALING, WMATH_SCALING3D
    - [ ] WMATH_UNIFORM_SCALE, WMATH_UNIFORM_SCALE_3D
    - [ ] WMATH_UNIFORM_SCALING, WMATH_UNIFORM_SCALING_3D
  - [ ] Mat4xN
    - [ ] WMATH_IDENTITY, WMATH_ZERO, WMATH_CREATE, WMATH_COPY, WMATH_SET
    - [ ] WMATH_NEGATE, WMATH_EQUALS, WMATH_EQUALS_APPROXIMATELY
    - [ ] WMATH_ADD, WMATH_SUB, WMATH_MULTIPLY_SCALAR, WMATH_MULTIPLY
    - [ ] WMATH_INVERSE, WMATH_TRANSPOSE
    - [ ] WMATH_DETERMINANT
    - [ ] WMATH_AIM, WMATH_LOOK_AT
    - [ ] WMATH_ORTHO, WMATH_PERSPECTIVE, WMATH_PERSPECTIVE_REVERSE_Z
    - [ ] WMATH_FRUSTUM, WMATH_FRUSTUM_REVERSE_Z
    - [ ] WMATH_GET_AXIS, WMATH_SET_AXIS
    - [ ] WMATH_GET_TRANSLATION, WMATH_SET_TRANSLATION
    - [ ] WMATH_TRANSLATION, WMATH_TRANSLATE
    - [ ] WMATH_ROTATE, WMATH_ROTATE_X, WMATH_ROTATE_Y, WMATH_ROTATE_Z
    - [ ] WMATH_ROTATION, WMATH_ROTATION_X, WMATH_ROTATION_Y, WMATH_ROTATION_Z
    - [ ] WMATH_AXIS_ROTATE, WMATH_AXIS_ROTATION
    - [ ] WMATH_CAMERA_AIM
    - [ ] WMATH_SCALE, WMATH_UNIFORM_SCALE
    - [ ] WMATH_SCALING, WMATH_UNIFORM_SCALING
    - [ ] WMATH_GET_SCALING
    - [ ] WMATH_FROM_MAT3, WMATH_FROM_QUAT

