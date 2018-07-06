#ifndef _DEVICE_UTIL_
#define _DEVICE_UTIL_

__host__ __device__ int compi(int x);
__host__ __device__ int ind(int row, int col);
__host__ __device__ int npdirInd(int row, int col);
__host__ __device__ bool qless(double d1, double d2);

#endif