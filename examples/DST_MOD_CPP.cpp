#include "WCN/WCN_MATH_DST_CPP.hpp"
#include <iostream>
using namespace WCN::Math;
using namespace WCN::Math::Ops; 

// 辅助打印函数 (假设 Vec3 结构体有 x, y, z 成员)
void print_vec3(const char* label, const Vec3& v) {
    std::cout << "[" << label << "]: " << v.v[0] << ", " << v.v[1] << ", " << v.v[2] << "\n";
}

void print_mat4(const char* label, const Mat4& m) {
    printf("[%s] (Translation only):\n", label);
    // 获取矩阵的位移部分用于演示
    Vec3 t;
    pipe(t) |= zero(); // 初始化
    WMATH_GET_TRANSLATION(Mat4)(&t, m); // 调用底层 C API 获取位移
    print_vec3("  Pos", t);
}
int main() {
    std::cout << "--- 1. Vector Chain ---\n";

    Vec3 position;
    Vec3 velocity = {10.0f, 0.0f, 0.0f};
    Vec3 gravity  = {0.0f, -9.8f, 0.0f};
    Vec2 t1 = {1.0f, 2.0f};
    Vec2 t3 = {3.0f, 4.0f};
    wcn_math_Vec2_add(&t1, t1, t3);
    std::cout << "t1: (" << t1.v[0] << ", " << t1.v[1] << ")\n";

    float dt = 0.5f;

    // 本库传统 C 写法 (DST 模式):
    // wcn_math_Vec3_add(&position, position, velocity);
    // wcn_math_Vec3_multiply_scalar(&position, position, dt);
    // wcn_math_vec3_add(&position, position, gravity);

    // 本库 C++ 链式写法:
    pipe(position) 
        |= zero()                            // 1. 初始化为零向量
        |  add(velocity)                  // 2. 加上速度
        |  mul(dt)                        // 3. 乘以时间 (缩放)
        |  add(gravity);                  // 4. 加上重力

    print_vec3("Result Position", position);

    printf("\n--- 2. Matrix Chain ---\n");

    Mat4 modelMatrix;

    // 构建一个模型矩阵：重置 -> 平移 -> 旋转 -> 缩放
    // 这种写法非常直观，阅读顺序即执行顺序
    pipe(modelMatrix)
        |= identity()                                // 重置为单位矩阵
        |  translate({5.0f, 0.0f, 0.0f})             // 平移 (5, 0, 0)
        |  rotate({0.0f, 1.0f, 0.0f}, PI / 2.0f)     // 绕 Y 轴旋转 90 度
        |  scale({2.0f, 2.0f, 2.0f});                // 整体放大 2 倍

    print_mat4("Model Matrix", modelMatrix);

    printf("\n--- 3. Custom Extensions ---\n");
    
    // 假设你想在链式调用中插入一个自定义的逻辑，或者调用一个尚未封装的 C 函数
    // 比如：打印当前值的调试操作
    auto log_debug = [](std::string tag) {
        return WCN::Internal::make_op([=](Vec3* d) {
            // printf("  [Debug %s] Current: %.2f, %.2f, %.2f\n", tag, d->x, d->y, d->z);
            std::cout << "  [Debug " << tag << "] Current: " << d->v[0] << ", " << d->v[1] << ", " << d->v[2] << "\n";
        });
    };

    Vec3 heroPos;
    pipe(heroPos)
        |= make_vec3(1.0f, 1.0f, 1.0f)
        |  log_debug("Init")            // 插入自定义操作
        |  normalize()                  // 归一化
        |  log_debug("Normalized")
        |  mul(10.0f);                  // 长度设为 10

    print_vec3("Hero Position", heroPos);
    std::cout << "--- 4. Quaternion ---\n";

    Quat rotation;
    Quat deltaRot;
    
    // 创建一个绕 Z 轴旋转的增量
    pipe(deltaRot) |= make_quat(0, 0, 0, 1); // 安全起见先初始化
    WMATH_CALL(Quat, from_axis_angle)(&deltaRot, {0, 0, 1}, PI); // 调用底层 C API

    pipe(rotation)
        |= identity()      // 初始无旋转
        |  mul(deltaRot)   // 应用增量旋转
        |  normalize();    // 规范化防止误差积累

    // printf("Rotation applied.\n");
    std::cout << "Rotation applied.\n";

    Mat4 aaa{
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    Quat bbb;
    pipe(bbb) |= to_quat(aaa);
    using namespace WCN::Index;
    M4x4<0, 0> ccc;
    pipe(aaa) | set_index(ccc, 5.0f);
    printf("Quat: %f, %f, %f, %f\n", bbb | x, bbb | y, bbb | z, bbb | w);
    printf("Mat4.(0, 0): %f\n", aaa | ccc);
    return 0;
}