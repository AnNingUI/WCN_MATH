mkdir build_no_dst
cd build_no_dst
cmake .. -DWCN_MATH_IS_DST=0 -G "Ninja"
ninja -j12
cd ..