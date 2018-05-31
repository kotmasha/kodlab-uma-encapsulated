#include "kernel.h"
#include "device_util.h"


/*
----------------------HOST DEVICE---------------------
*/

/*
Complimentary function
*/
__host__ __device__ int compi(int x) {
	if (x % 2 == 0) return x + 1;
	else return x - 1;
}

/*
This function is for extracting the 1d index from a triangelized 2d structure
The function requires that the 'row' has to be not small than the 'col'
*/
__host__ __device__ int ind(int row, int col) {
	if (row >= col)
		return row * (row + 1) / 2 + col;
	else if (row + 1 == col && row % 2 == 0) {
		return col * (col + 1) / 2 + row;
	}
	else {
		return compi(col) * (compi(col) + 1) / 2 + compi(row);
	}
}

/*
This function is calculating the npdir index in npdir matrix ONLY
input same as ind, row and col
*/
__host__ __device__ int npdir_ind(int row, int col) {
	int offset = (row + 1) / 2;
	if(row >= col) return offset + ind(row, col);
	else if (row + 1 == col && row % 2 == 0) return offset + 1 + ind(row, row);
	else return npdir_ind(compi(col), compi(row));
}

/*
This function is checking whether d1 is smaller than d2
-1 means infinity
*/
__host__ __device__ bool qless(double d1, double d2) {
	if (d1 < -0.5) return false;
	if (d2 < -0.5) return true;
	return d1 - d2 < -1e-12;
}

/*
----------------------HOST DEVICE---------------------
*/
