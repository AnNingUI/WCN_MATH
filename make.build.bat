@echo off
REM 构建本库。只有一套实现（destination-out-param），
REM 旧的 -DWCN_MATH_IS_DST=0 分支已删除，无需再传该选项。
cd /d "%~dp0"
if not exist build mkdir build
cd build
cmake .. -G "Ninja"
if errorlevel 1 exit /b 1
ninja -j12
cd ..