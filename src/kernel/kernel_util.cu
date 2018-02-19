#include "kernel.h"
#include "kernel_util.cuh"
#include "device_util.h"
#include "Logger.h"

static Logger kernelUtilLogger("DataUtil", "log/device.log");

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
__global__ void negate_conjunction_star_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b1[index] = b1[index] && !b2[compi(index)];
	}
}

/*
Conjuncate the start of array, store the result in first array
Input: two bool array, and size of both
*/
__global__ void conjunction_star_kernel(bool *b1, bool *b2, int size) {
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

void kernel_util::alltrue(bool *b, int size) {
	alltrue_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	kernelUtilLogger.debug("alltrue_kernel invoked");
}

void kernel_util::allfalse(bool *b, int size) {
	allfalse_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	kernelUtilLogger.debug("allfalse_kernel invoked");
}

void kernel_util::bool2int(bool *b, int *i, int size) {
	bool2int_kernel << <GRID1D(size), BLOCK1D >> > (b, i, size);
	kernelUtilLogger.debug("bool2int_kernel invoked");
}

void kernel_util::conjunction(bool *b1, bool *b2, int size) {
	conjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("conjunction_kernel invoked");
}

void kernel_util::disjunction(bool *b1, bool *b2, int size) {
	disjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("disjunction_kernel invoked");
}

void kernel_util::subtraction(bool *b1, bool *b2, int size) {
	subtraction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("subtraction_kernel invoked");
}

void kernel_util::negate_conjunction_star(bool *b1, bool *b2, int size) {
	negate_conjunction_star_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("negate_conjunction_kernel invoked");
}

void kernel_util::conjunction_star(bool *b1, bool *b2, int size) {
	conjunction_star_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("conjunction_star_kernel invoked");
}

void kernel_util::up2down(bool *b1, bool *b2, int size) {
	up2down_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	kernelUtilLogger.debug("up2down_kernel invoked");
}