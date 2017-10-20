#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "device_util.h"

/*
---------------------HELPER FUNCTION-----------------------
*/
__global__ void alltrue_kernel(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = true;
	}
}

__global__ void allfalse_kernel(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = false;
	}
}

__global__ void bool2int_kernel(bool *b, int *i, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (b[index]) i[index] = 1;
		else i[index] = 0;
	}
}

/*
This function does the conjunction for two lists
Input: two bool lists
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
		bool m1 = b1[2 * index];
		bool m2 = b1[2 * index + 1];
		b1[2 * index] = m1 && !b2[2 * index + 1];
		b1[2 * index + 1] = m2 && !b2[2 * index];
	}
}

__global__ void conjunction_star_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (index % 2 == 0) {
			b1[index] = b1[index] && b2[index + 1];
		}
		else {
			b1[index] = b1[index] && b2[index - 1];
		}
	}
}

__global__ void up2down_kernel(bool *b1, bool *b2, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b2[index] = !b1[compi(index)];
	}
}

namespace kernel_util {
	void alltrue(bool *b, int size) {
		alltrue_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	}

	void allfalse(bool *b, int size) {
		allfalse_kernel << <GRID1D(size), BLOCK1D >> > (b, size);
	}

	void bool2int(bool *b, int *i, int size) {
		bool2int_kernel << <GRID1D(size), BLOCK1D >> > (b, i, size);
	}

	void conjunction(bool *b1, bool *b2, int size) {
		conjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}

	void disjunction(bool *b1, bool *b2, int size) {
		disjunction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}

	void subtraction(bool *b1, bool *b2, int size) {
		subtraction_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}

	void negate_conjunction_star(bool *b1, bool *b2, int size) {
		negate_conjunction_star_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}

	void conjunction_star(bool *b1, bool *b2, int size) {
		conjunction_star_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}

	void up2down(bool *b1, bool *b2, int size) {
		up2down_kernel << <GRID1D(size), BLOCK1D >> > (b1, b2, size);
	}
}