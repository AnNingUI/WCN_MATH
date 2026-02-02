#include "WCN/WCN_MATH_CPP_COMMON.hpp"
#include "WCN/WCN_MATH_DST.h"
#include <iostream>
int main() {
    // in c
    WCN::Vec2 v1;
    WCN::Vec2 v2 = {1, 2};
    WCN::Vec2 v3 = {2, 3};
    WMATH_ADD(Vec2)(&v1, v2, v3);
    std::cout << "v1.x = " << v1.v[0] << "v1.y = " << v1.v[1] << "\n";
}