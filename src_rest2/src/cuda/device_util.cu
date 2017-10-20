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
	else if (col == row + 1) {
		return col * (col + 1) / 2 + row;
	}
	else {
		return compi(col) * (compi(col) + 1) / 2 + compi(row);
	}
}

/*
----------------------HOST DEVICE---------------------
*/
