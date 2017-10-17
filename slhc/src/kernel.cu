#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "DataManager.h"

#define THREAD1D 256
#define THREAD2D 16
#define GRID1D(X) dim3((X + THREAD1D - 1) / THREAD1D)
#define BLOCK1D dim3(THREAD1D)
#define GRID2D(X, Y) dim3((X + THREAD2D - 1) / THREAD2D, (Y + THREAD2D - 1) / THREAD2D)
#define BLOCK2D dim3(THREAD2D, THREAD2D)

//index function, getting value in a lower triangle matrix.
//if col > row, just do symetric operation(used in dioid_square_GPU)
__host__ __device__ int ind(int row, int col) {
	if (row >= col)
		return row * (row + 1) / 2 + col;
	else {
		return col * (col + 1) / 2 + row;
	}
}

// setup the union find init value
__global__ void union_init_GPU(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

// compare dist with delta, and get 0 or 1
__global__ void check_dist_GPU(double *data, int *result, float delta, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (data[index] <= delta) result[index] = 1;
		else result[index] = 0;
	}
}

// do union find 
__global__ void union_GPU(int *data, int *root, int size) {
	int index = threadIdx.x;
	for (int i = 0; i < size; ++i) {
		__syncthreads();
		if (root[i] != i) continue;
		int j = index;
		while (j < size) {
			if (data[ind(i, j)] == 1) root[j] = i;
			j += THREAD1D;
		}
	}
}

// dioid operation, to get slhc value
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
		Pvalue = Md[ind(Row, Col)];
	}
	for (int i = 0; i < Width / TILE_WIDTH + 1; ++i) {
		if (Row < Width && Col < Width) {//to prevent array index error, the cell does not exist
			if (i * TILE_WIDTH + tx < Width && Row < Width) {//check if Tile cell index overflow
				int y = Row;
				int x = i * TILE_WIDTH + tx;
				Mds[ty][tx] = Md[ind(y, x)];
			}
			if (Col < Width && i * TILE_WIDTH + ty < Width) {//check if Tile cell index overflow
				int y = i * TILE_WIDTH + ty;
				int x = Col;
				Nds[ty][tx] = Md[ind(y, x)];
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
		Md[ind(Row, Col)] = Pvalue;
	}
}

// calculate slhc
void DataManager::calculate_slhc() {
	int t = floor(log(_point_size) / log(2)) + 1;
	cudaMemcpy(dev_slhc, dev_dists, _pair_size * sizeof(double), cudaMemcpyDeviceToDevice);
	for (int i = 0; i < t; ++i) {
		dioid_square_GPU << <GRID2D(_point_size, _point_size), BLOCK2D >> > (dev_slhc, _point_size);
	}
}

// calculate blocks, will use the slhc value as input
vector<vector<int> > DataManager::blocks(double &delta){
	union_init_GPU << <GRID1D(_point_size), BLOCK1D >> >(dev_union_root, _point_size);
	check_dist_GPU << <GRID1D(_pair_size), BLOCK1D >> >(dev_slhc, dev_slhc_, delta, _pair_size);

	union_GPU << <GRID1D(1), BLOCK1D >> >(dev_slhc_, dev_union_root, _point_size);
	cudaMemcpy(h_union_root, dev_union_root, _point_size * sizeof(int), cudaMemcpyDeviceToHost);

	map<int, int> m;
	vector<vector<int> > result;
	for (int i = 0; i < _point_size; ++i) {
		if (m.find(h_union_root[i]) == m.end()) {
			m[h_union_root[i]] = result.size();
			vector<int> tmp;
			tmp.push_back(i);
			result.push_back(tmp);
		}
		else {
			result[m[h_union_root[i]]].push_back(i);
		}
	}
	//_log->debug() << "Block get " + to_string(result.size()) + " groups";

	return result;
}

// get the height value, do sort on slhc first, then do dedup
vector<double> DataManager::getHeight() {
	vector<double> result;

	cudaMemcpy(h_height, dev_slhc, _pair_size * sizeof(double), cudaMemcpyDeviceToHost);
	thrust::sort(h_height, h_height + _pair_size, thrust::greater<double>());
	double *new_end = thrust::unique(h_height, h_height + _pair_size);
	for (int i = 0; h_height + i != new_end; ++i) result.push_back(h_height[i]);

	return result;
}
