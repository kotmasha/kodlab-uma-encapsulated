#ifndef _KERNEL_
#define _KERNEL_

#include <cuda.h>
#include <cuda_runtime.h>

//marco for GPU kernel function, change those value according to your own GPU
#define THREAD1D 256
#define THREAD2D 16
#define GRID1D(X) dim3((X + THREAD1D - 1) / THREAD1D)
#define BLOCK1D dim3(THREAD1D)
#define GRID2D(X, Y) dim3((X + THREAD2D - 1) / THREAD2D, (Y + THREAD2D - 1) / THREAD2D)
#define BLOCK2D dim3(THREAD2D, THREAD2D)

/*
----------------------MARCO-----------------------
*/

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
			system("pause");\
			exit(1); \
		        } \
	    } while (0)

/*
----------------------MARCO-----------------------
*/

#endif