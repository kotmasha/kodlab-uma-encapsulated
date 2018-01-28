#This shell is used to build the kernel code on LINUX system only, on WINDOWS, the build is still handled by CMAKE

UMA_PATH=$PWD
OUTPUT_PATH=$UMA_PATH/lib #The output
KERNEL_PATH=$UMA_PATH/src/kernel
COMMON_PATH=$UMA_PATH/src/common
UTILITY_PATH=$UMA_PATH/src/utility
NVCC_FLAG="-std=c++11 -Xcompiler -fPIC"
TARGET=libUMAKernel.a

cd $KERNEL_PATH
nvcc $NVCC_FLAG *.cu -dc -I $COMMON_PATH -I $UTILITY_PATH
echo "Device code compiled"
nvcc $NVCC_FLAG *.o -dlink -o device_link.o
echo "Generate tmp link obj"
ar rcs $TARGET *.o
echo "Generate the static library"
rm *.o
echo "Cleaning up obj files"
mv $TARGET $OUTPUT_PATH
echo Target moved to $OUTPUT_PATH
