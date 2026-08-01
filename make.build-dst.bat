mkdir build
cd build
cmake .. -DWCN_MATH_IS_DST=1 -G "Ninja"
ninja -j12
cd ..