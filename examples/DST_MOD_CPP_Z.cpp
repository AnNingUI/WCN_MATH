#include "WCN/WCN_MATH_DST_CLASS.hpp"
#include <iostream>
int main() {
    using WCN::z_vec2;
    z_vec2 v(1.0f, 2.0f);
    const z_vec2 u(3.0f, 4.0f);
    v+=u;
    std::cout << "v: (" << v.x() << ", " << v.y() << ")" << std::endl;
    v+=5.0f;          // 9 11
    std::cout << "v: (" << v.x() << ", " << v.y() << ")" << std::endl;
    v.add_scaled_self(z_vec2{0.5f, 0.5f}, 2.0f);
    std::cout << "v: (" << v.x() << ", " << v.y() << ")" << std::endl;
    auto vv = v + u;
    // printf("v: (%f, %f)\n", v.x(), v.y());
    std::cout << "vv: (" << vv.x() << ", " << vv.y() << ")" << std::endl;
}