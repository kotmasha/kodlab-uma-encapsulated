#include <cuda.h>
#include <cuda_runtime.h>

#include "DataManager.h"

#define THREAD1D 256
#define THREAD2D 16
#define GRID1D(X) dim3((X + THREAD1D - 1) / THREAD1D)
#define BLOCK1D dim3(THREAD1D)
#define GRID2D(X, Y) dim3((X + THREAD2D - 1) / THREAD2D, (Y + THREAD2D - 1) / THREAD2D)
#define BLOCK2D dim3(THREAD2D, THREAD2D)

__host__ __device__ int ind(int row, int col) {
	if (row >= col)
		return row * (row + 1) / 2 + col;
	else if (col == row + 1) {
		return col * (col + 1) / 2 + row;
	}
}


__global__ void union_init_GPU(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

__global__ void check_dist_GPU(double *data, float delta, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (data[index] < delta) data[index] = 1;
		else data[index] = 0;
	}
}


__global__ void union_GPU(int *data, int *root, int size) {
	int index = threadIdx.x;
	for (int i = 0; i < size; ++i) {
		__syncthreads();
		if (root[i] != i) continue;
		int j = index;
		while (j < size) {
			if (data[i * size + j] == 1) root[j] = i;
			j += THREAD1D;
		}
	}
}

__global__ void dioid_square_GPU(double* Md, int Width) {
	const int TILE_WIDTH = THREAD2D;
	__shared__ double Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ double Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	double Pvalue = 0;
	if (Row < Width && Col < Width) {
		Pvalue = Md[Row * Width + Col];
	}
	for (int i = 0; i < Width / TILE_WIDTH + 1; ++i) {
		if (Row < Width && Col < Width) {//to prevent array index error, the cell does not exist
			if (i * TILE_WIDTH + tx < Width && Row < Width) {//check if Tile cell index overflow
				int y = Row;
				int x = i * TILE_WIDTH + tx;
				Mds[ty][tx] = Md[y * Width + x];
			}
			if (Col < Width && i * TILE_WIDTH + ty < Width) {//check if Tile cell index overflow
				int y = i * TILE_WIDTH + ty;
				int x = Col;
				Nds[ty][tx] = Md[y * Width + x];
			}
		}
		__syncthreads();

		if (Row < Width && Col < Width) {//to prevent array index error, the cell does not exist
			for (int k = 0; k < TILE_WIDTH; ++k) {
				if (i * TILE_WIDTH + k >= Width) break; //cell exist, but index overflow
				Pvalue = min(Pvalue, max(Mds[ty][k], Nds[k][tx]));
			}
		}
		__syncthreads();
	}
	if (Row < Width && Col < Width) {
		Md[Row * Width + Col] = Pvalue;
	}
}

void DataManager::calculate_slhc() {
	int t = floor(log(_point_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		dioid_square_GPU << <GRID2D(_point_size, _point_size), BLOCK2D >> >(dev_dists, _point_size);
	}

	/*
	cudaMemcpy(h_dists, dev_dists, _sensor_size * _sensor_size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _sensor_size; ++i) {
		for (int j = 0; j < _sensor_size; ++j) cout << h_dists[i * _sensor_size + j] << ",";
		cout << endl;
	}
	*/

	//union_init_GPU << <GRID1D(_point_size), BLOCK1D >> >(dev_union_root, _point_size);
	//check_dist_GPU << <GRID1D(_pair_size), BLOCK1D >> >(dev_dists, delta, _pair_size);

	//union_GPU << <GRID1D(1), BLOCK1D >> >(dev_dists, dev_union_root, _point_size);
	//cudaMemcpy(h_union_root, dev_union_root, _point_size * sizeof(int), cudaMemcpyDeviceToHost);
}
