cd dependency
mkdir build
cd build
cmake ../googletest-release-1.8.0 -DCMAKE_CONFIGURATION_TYPES="Release" -DCMAKE_GENERATOR_PLATFORM=x64 -DBUILD_GMOCK="OFF" -DBUILD_GTEST="ON" -DCMAKE_CXX_FLAGS_RELEASE="/Md"
cmake --build ./ --config Release
