cd dependency
mkdir build
cd build
cmake ../googletest-release-1.8.0 -DBUILD_GMOCK="OFF" -DBUILD_GTEST="ON" -DCMAKE_CXX_FLAGS_RELEASE="/Md"
cmake --build ./ --config Release
