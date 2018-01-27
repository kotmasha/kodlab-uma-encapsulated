Please use CMAKE to compile the project

---------------------------------------------WINDOWS-------------------------------------------
1 make sure you have vcpkg installed in your environment, check this link https://github.com/Microsoft/vcpkg
2 make sure you have cpprestsdk(CASABLANCA) installed, use "vcpkg install cpprestsdk cpprestsdk:x64-windows" to install the package, check https://github.com/Microsoft/cpprestsdk,
installing the corresponding package may require 30~60min, take around 5GB disk
3 make sure you have CUDA 8.0 or higher in your environment, check https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
4 make sure you have CMAKE installed in your pc, check here https://cmake.org/download/ please use binary distribution
5 clone the project
6 run build_dependency.bat, to build gtest
7 run build_vs.bat, for the project visual studio solution.
8* if needed, find cmake bin folder, open cmake-gui.exe, and choose "where is the source code" to be src, and "where to build the binaries" to be build folder
9* if needed click on Configure, resolve the variables you want(RELEASE/X64 ...)
10* if no error, click generate, then open the project
11 build the project, the runnable project will be under build/bin/$build_type
12 be sure to copy "ini" folder to where you run the binary
note 7 can be done manually by 8-10

---------------------------------------------LINUX--------------------------------------------
1 make sure you have cpprestsdk(CASABLANCA) installed, check https://github.com/Microsoft/cpprestsdk for install steps
2 make sure you have CMAKE installed in your pc, check here https://cmake.org/download/ please use binary distribution
3 make sure you have CUDA 8.0 or higher in your environment, check https://developer.nvidia.com/cuda-downloads?target_os=Linux
4 clone the project
5 run build_kernel.sh to build the UMAKernel(under src/kernel) separately
6 run build_dependency.sh to build gtest
6 mkdir build
7 cd build; cmake ../src
8 make