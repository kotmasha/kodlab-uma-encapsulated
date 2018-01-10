Please use CMAKE to compile the project

---------------------------------------------WINDOWS-------------------------------------------
1 make sure you have vcpkg installed in your environment, check this link https://github.com/Microsoft/vcpkg
2 make sure you have cpprestsdk(CASABLANCA) installed, use "vcpkg install cpprestsdk cpprestsdk:x64-windows" to install the package, check https://github.com/Microsoft/cpprestsdk,
installing the corresponding package may require 30~60min, take around 5GB disk
3 make sure you have CUDA 8.0 or higher in your environment, check https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
4 make sure you have CMAKE installed in your pc, check here https://cmake.org/download/ please use binary distribution
5 clone the project
6 create a build folder where you want the visual studio project to be
7 find cmake bin folder, open cmake-gui.exe, and choose "where is the source code" to be src, and "where to build the binaries" to be build folder
8 click on Configure, resolve the variables you want(RELEASE/X64 ...)
9 if no error, click generate, then open the project
10 build the project, the runnable project will be under build/bin/$build_type
11 be sure to copy "ini" folder to where you run the binary