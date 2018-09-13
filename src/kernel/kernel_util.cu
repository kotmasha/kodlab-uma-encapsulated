#include "kernel.h"
#include "kernel_util.cuh"
#include "device_util.h"
#include "Logger.h"

static Logger kernelUtilLogger("KernelUtil", "log/device.log");

/*
---------------------HELPER FUNCTION-----------------------
*/

/*
set the input vector to be all true
Input: bool array, and size of it
*/
__global__ void alltrue_kernel(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = true;
	}
}

/*
set the input array to be all false
Input: bool array, and size of it
*/
__global__ void allfalse_kernel(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = false;
	}
}

/*
convert the input bool to int, false->0, true->1
Input: a bool source array, and int dest array, and size of both array
*/
__global__ void bool2int_kernel(bool *b, int *i, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (b[index]) i[index] = 1;
		else i[index] = 0;
	}
}

/*
convert the input bool to double, false->0, true->1
Input: a bool source array, and double dest array, and size of both array
*/
__global__ void bool2double_kernel(bool *b, double *d, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (b[index]) d[index] = 1.0;
		else d[index] = 0.0;
	}
}

/*
This function does the conjunction for two lists
Input: two bool lists, and size of both
Output: None
*/
__global__ void conjunction_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] && b2[index];
	}
}

/*
This function computes the difference of two lists
Input: two bool lists (same length)
Output: None
*/
__global__ void subtraction_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] && !b2[index];
	}
}

/*
This function does the disjunction for two lists
Input: two bool lists
Output: None
*/
__global__ void disjunction_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] || b2[index];
	}
}

/*
This function does compi, conjunction together
Input: two boolean lists
Output: None
*/
__global__ void negateConjunctionStar_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] && !b2[compi(index)];
	}
}

/*
Conjuncate the start of array, store the result in first array
Input: two bool array, and size of both
*/
__global__ void conjunctionStar_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] && b2[compi(index)];
	}
}

/*
Changing the signal from up to down
Input: two bool array, and both size
*/
__global__ void up2down_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b2[index] = !b1[compi(index)];
	}
}

__global__ void sum_kernel(double *d, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = 1;
	int j = index;
	while (true) {
		j = index;
		while (j < size) {
			if (j % (2 * offset) == 0 && j + offset < size) d[j] += d[j + offset];
			j += THREAD1D;
		}
		offset = offset * 2;
		__syncthreads();
		if (offset >= size) break;
	}
}

__global__ void initMaskSignal_kernel(bool *b, int initSize, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (index < 2 * initSize) {
			b[index] = true;
		}
		else {
			b[index] = false;
		}
	}
}

void kernel_util::alltrue(bool *b, int size) {
	cudaCheckErrors("check alltrue error");
	alltrue_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	kernelUtilLogger.debug("alltrue_kernel invoked");
	cudaCheckErrors("check alltrue error");
}

void kernel_util::allfalse(bool *b, int size) {
	allfalse_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	kernelUtilLogger.debug("allfalse_kernel invoked");
	cudaCheckErrors("check allfalse error");
}

void kernel_util::bool2int(bool *b, int *i, int size) {
	bool2int_kernel << <GRID1D(size), BLOCK1D >> > (b, i, size);
	kernelUtilLogger.debug("bool2int_kernel invoked");
	cudaCheckErrors("check bool2int error");
}

void kernel_util::bool2double(bool *b, double *d, int size) {
	bool2double_kernel << <GRID1D(size), BLOCK1D >> > (b, d, size);
	kernelUtilLogger.debug("bool2double_kernel invoked");
	cudaCheckErrors("check bool2double error");
}

void kernel_util::conjunction(bool *b1, bool *b2, int size) {
	conjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("conjunction_kernel invoked");
	cudaCheckErrors("check conjunction error");
}

void kernel_util::disjunction(bool *b1, bool *b2, int size) {
	disjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("disjunction_kernel invoked");
	cudaCheckErrors("check disjunction error");
}

void kernel_util::subtraction(bool *b1, bool *b2, int size) {
	subtraction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("subtraction_kernel invoked");
	cudaCheckErrors("check subtraction error");
}

void kernel_util::negateConjunctionStar(bool *b1, bool *b2, int size) {
	negateConjunctionStar_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("negate_conjunction_kernel invoked");
	cudaCheckErrors("check negateConjunctionStar error");
}

void kernel_util::ConjunctionStar(bool *b1, bool *b2, int size) {
	conjunctionStar_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("conjunctionStar_kernel invoked");
	cudaCheckErrors("check ConjunctionStar error");
}

void kernel_util::up2down(bool *b1, bool *b2, int size) {
	up2down_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("up2down_kernel invoked");
	cudaCheckErrors("check up2down error");
}

double kernel_util::sum(double *d, int size) {
	sum_kernel << <1, BLOCK1D >> > (d, size);
	double r;
	cudaMemcpy(&r, d, sizeof(double), cudaMemcpyDeviceToHost);
	kernelUtilLogger.debug("sum_kernel invoked");
	cudaCheckErrors("check sum error");
	return r;
}

void kernel_util::initMaskSignal(bool *b, int initSize, int size) {
	initMaskSignal_kernel << <GRID1D(size), BLOCK1D >> > (b, initSize, size);
	kernelUtilLogger.debug("initMaskSignal_kernel invoked");
	cudaCheckErrors("check initMaskSignal error");
}