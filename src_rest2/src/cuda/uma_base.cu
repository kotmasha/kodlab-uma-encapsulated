#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "kernel.h"
#include "kernel_util.cuh"
#include "device_util.h"

using namespace std;


/*
---------------------DEVICE------------------------
*/
__device__ double lower_threshold(double q, double total, double phi, double T){
	if (total <= phi || T <= (1 - q) / total) {
		return T;
	}
	else {
		return q * T;
	}
		
}

__device__ double raise_threshold(double q, double total, double phi, double T){
	return ( 1.0 / q ) * T;
}

/*
This function does implies on GPU, using non-worker solution
Input: row, col info, measurable size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
__device__ bool implies_GPU(int row, int col, double *weights, double total, double threshold){//implies
	double rc = weights[ind(row, col)];//0.4
	double r_c = weights[ind(compi(row), col)];//0
	double rc_ = weights[ind(row, compi(col))];//0
	double r_c_ = weights[ind(compi(row), compi(col))];//0.6
	return rc_ < min(total * threshold, min(rc, min(r_c, r_c_))) || (rc_ == 0 && r_c == 0 && rc > 0 && r_c_ > 0);
}

/*
This function does equivalent on GPU, using non-worker solution
Input: row, col info, measurable size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
/*
__device__ bool equivalent_GPU(int row, int col, double *weights){//equivalent
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	return rc_ == 0 && r_c == 0 && rc > 0 && r_c_ > 0;
}
*/

/* SIQI: I believe the thresholds will have to be updated separately, and BEFORE the orientation matrix can be updated. This is a new method, please edit and wrap it into the Agent class.
GPU method: updates learning thresholds.
Input: direction, weight, threshold matrix, last_total, q, phi, xy location
Output: None
*/
__device__ void update_thresholds_GPU(bool *dir, double *thresholds, double last_total, double q, double phi, int x, int y){
	//LATEST THRESHOLD FOR THIS SQUARE:
	double threshold = thresholds[ind(y, x)];
	//CHECK CONDITION FOR LOWERING THE THRESHOLD: (depends on previous entries of the orientation matrix)
	bool condition_to_lower = (dir[ind(2 * y, 2 * x)] || dir[ind(compi(2 * y), 2 * x)] || dir[ind(2 * y, compi(2 * x))] || dir[ind(compi(2 * y), compi(2 * x))]);
	//CHECK CONDITION FOR RAISING THE THRESHOLD: (raising is currently disabled)
	bool condition_to_raise = false;

	//UPDATE THE THRESHOLD(S) FOR THIS SQUARE:
	// lowering the thresholds:
	if (condition_to_lower) {
		thresholds[ind(y, x)] = lower_threshold(q, last_total, phi, threshold);
	}
	//raising the thresholds:
	if (condition_to_raise) {
		thresholds[(ind(y, x))] = raise_threshold(q, last_total, phi, threshold);
	}
}

/* SIQI: I've updated this method to only use implies_GPU (see above)
GPU method: updates a "square" in the orientation matrix
Input: dir matrix, weights matrix, thresholds matrix, xy location in dir matrix
Output: None
*/
__device__ void orient_square_GPU(bool *dir, double *weights, double *thresholds, double total, int x, int y) {//orient_square
	//OBTAIN THE LOCAL THRESHOLD
	double threshold = thresholds[ind(y / 2, x / 2)];
	//UPDATE THE ORIENTATION MATRIX
	if(y >= x)
		dir[ind(y, x)] = implies_GPU(y, x, weights, total, threshold);
	else
		dir[ind(compi(x), compi(y))] = implies_GPU(compi(x), compi(y), weights, total, threshold);
	if (compi(y) >= x)
		dir[ind(compi(y), x)] = implies_GPU(compi(y), x, weights, total, threshold);
	else
		dir[ind(compi(x), y)] = implies_GPU(compi(x), y, weights, total, threshold);
	if (y >= compi(x))
		dir[ind(y, compi(x))] = implies_GPU(y, compi(x), weights, total, threshold);
	else
		dir[ind(x, compi(y))] = implies_GPU(x, compi(y), weights, total, threshold);
	if (compi(y) >= compi(x))
		dir[ind(compi(y), compi(x))] = implies_GPU(compi(y), compi(x), weights, total, threshold);
	else
		dir[ind(x, y)] = implies_GPU(x, y, weights, total, threshold);
}

/*
---------------------DEVICE------------------------
*/

/*
---------------------GLOBAL---------------------
*/

__global__ void init_mask_kernel(bool *mask, int initial_size, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index < initial_size * 2) mask[index] = false;
		else mask[index] = true;
	}
}

__global__ void init_diag_kernel(double *diag, double *diag_, double total, double last_total, int measurable_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < measurable_size){
		diag[index] = total / 2.0;
		diag_[index] = last_total / 2.0;
	}
}

/*
This function is update weights for discounted agent, using non-worker solution
Input: weight matrix, observe bool value from python side and measurable size
Output: None
*/
__global__ void update_weights_kernel_stationary(double *weights, bool *observe, int measurable_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < measurable_size){
		if(activity){
			weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * phi;
		}
	}
}

__global__ void update_weights_kernel_forgetful(double *weights, bool *observe, int measurable_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < measurable_size){
	    weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * activity * phi;
	}
}

__global__ void update_weights_kernel(double *weights, bool *observe, int measurable_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < measurable_size){
		weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * phi;
	}
}

__global__ void get_weights_diag_kernel(double *weights, double *diags, double *diags_, int measurable_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < measurable_size){
		int idx = ind(index, index);
		diags_[index] = diags[index];
		diags[index] = weights[idx];
	}
}

//TODO: change 1e-12 to $total*\min_{z}{\tau_{index,z}}$ think whether this is the right quantity
//      perhaps max instead of min? could also use average...
__global__ void calculate_target_kernel(double *measurable, bool *target, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(measurable[2 * index] - measurable[2 * index + 1] > 1e-12){
			target[2 * index] = true;
			target[2 * index + 1] = false;
		}
		else if(measurable[2 * index] - measurable[2 * index + 1] < 1e-12){
			target[2 * index] = false;
			target[2 * index + 1] = true;
		}
		else{
			target[2 * index] = false;
			target[2 * index + 1] = false;
		}
	}
}

/* SIQI: I've added this kernel to do the threshold update
GPU method: updating the thresholds prior to orientation update
Input: dir weights, thresholds, last_total, q, phi, measurable_size
Output: None
*/
__global__ void update_thresholds_kernel(bool *dir, double *thresholds, double last_total, double q, double phi, int sensor_size) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexY < sensor_size && indexX < indexY) {
		update_thresholds_GPU(dir, thresholds, last_total, q, phi, indexX, indexY);
	}
}

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and measurable size
Output: None
*/
__global__ void orient_all_kernel(bool *dir, double *weights, double *thresholds, double total, int sensor_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexY < sensor_size && indexX < indexY){
		orient_square_GPU(dir, weights, thresholds, total, indexX * 2, indexY * 2);
	}
}

/*
This function is dfs on GPU, using non-worker solution
This function use shared memory and thus has only one GPU block, the default threshold number is 1024
Input: bool list of data to be dfsed, direction matrix and measurable size
Output: None
*/
__global__ void multiply_kernel(bool *x, bool *dir, double *thresholds, bool is_stable, double lowest, int size){//dfs
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ bool shared[];
	bool *xs = &shared[0];
	bool *ys = &shared[size];
	__shared__ bool flag[1];
	int j = index;
	// matrix multiplication variable
	while(j < size) {
		xs[j] = x[j];
		ys[j] = x[j];
		j += 512;
	}
	flag[0] = true;
	__syncthreads();
	while(flag[0]){
		flag[0] = false;
		__syncthreads();
		j = index;
		while(j < size){
			if(xs[j] == 1){
				j += 512;
				continue;
			}
			for(int i = 0; i < size; ++i){
				if(i / 2 == j / 2) continue;
				//make sure do not imply itself
				if(i >= j && dir[ind(i, j)] && xs[i] == 1){
					ys[j] = 1;
					flag[0] = true;
					break;
				}
				else if(i < j && dir[ind(compi(j),compi(i))] && xs[i] == 1){
					ys[j] = 1;
					flag[0] = true;
					break;
				}
			}
			j += 512;
		}
		__syncthreads();
		j = index;
		while(j < size){
			xs[j] = ys[j];
			j += 512;
		}
		__syncthreads();
	}
	j = index;
	while(j < size){
		x[j] = ys[j];
		j += 512;
	}
}

__global__ void floyd_kernel(bool *dir, int measurable_size) {
	int indexX = threadIdx.x;
	int indexY = threadIdx.y;
	for (int i = 0; i < measurable_size; ++i) {
		int x = indexX, y = indexY;
		while (y < measurable_size) {
			if (y < x) {
				y += 16;
				x = indexX;
				continue;
			}
			dir[ind(y, x)] = dir[ind(y, x)] || (dir[ind(y, i)] && dir[ind(i, x)]);
			x += 16;
		}
		__syncthreads();
	}
}


__global__ void dioid_square_kernel(int* Md, int Width) {
	const int TILE_WIDTH = 16;
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	int Pvalue = 0;
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


__global__ void transpose_multiply_kernel(bool* Md, bool *Nd, int Width, int Height) {
	const int TILE_WIDTH = THREAD2D;
	__shared__ bool Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ bool Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	bool Pvalue = false;
	if (Row < Width && Col < Height) {
		Pvalue = Nd[Col * Width + Row];
	}
	for (int i = 0; i < Width / TILE_WIDTH + 1; ++i) {
		if (i * TILE_WIDTH + tx < Width && Row < Width) {//check if Tile cell index overflow
			//int y = Row;
			//int x = i * TILE_WIDTH + tx;
			int y = i * TILE_WIDTH + tx;
			int x = Row;
			if (y >= x) Mds[ty][tx] = Md[ind(y, x)];
			else Mds[ty][tx] = Md[ind(compi(x), compi(y))];
		}
		if (i * TILE_WIDTH + ty < Width && Col < Height) {
			//Nds[ty][tx] = Nd[Col + (i * TILE_WIDTH + ty) * Width];
			Nds[ty][tx] = Nd[Col * Width + i * TILE_WIDTH + ty];
		}
		__syncthreads();
		
		if (Row < Width && Col < Height) {//to prevent array index error, the cell does not exist
			for (int k = 0; k < TILE_WIDTH; ++k) {
				if (i * TILE_WIDTH + k >= Width) break; //cell exist, but index overflow
				Pvalue = Pvalue || (Mds[ty][k] && Nds[k][tx]);
			}
		}
		
		__syncthreads();
	}
	if (Row < Width && Col < Height) {
		Nd[Col * Width + Row] = Pvalue;
	}
}


/*
This function is the GPU version of python function mask, it is designed to get mask signal
Deprecated in new version
Input: destination mask address, action list and mask size
Output: None
*/
__global__ void mask_kernel(bool *mask_amper, bool *mask, bool *current, int sensor_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < sensor_size && (mask_amper[2 * ind(indexY, indexX)] || mask_amper[2 * ind(indexY, indexX) + 1])){//need trick in the future to improve performance
		if(mask[2 * indexY]){//means still need check
			if(mask_amper[2 * ind(indexY, indexX)]){//check pure position
				if(!current[2 * indexX]) mask[2 * indexY] = false;
			}
			else{//check '*' position
				if(!current[2 * indexX + 1]) mask[2 * indexY] = false;
			}
		}
	}
}

__global__ void check_mask_kernel(bool *mask, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(mask[2 * index]) mask[2 * index + 1] = false;
	}
}

__global__ void delta_weight_sum_kernel(double *measurable, bool *signal, float *result, int measurable_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < measurable_size){
		atomicAdd(result, signal[index] * (measurable[index] - measurable[compi(index)]));
	}
}

__global__ void union_init_kernel(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

__global__ void check_dist_kernel(int *data, float delta, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (data[index] < delta) data[index] = 1;
		else data[index] = 0;
	}
}

__global__ void union_GPU_kernel(int *data, int *root, int size) {
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

/*
---------------------AGENT---------------------
*/


namespace uma_base {
	void init_mask(bool *mask, int initial_size, int measurable_size) {
		init_mask_kernel << <GRID1D(measurable_size), BLOCK1D >> > (mask, initial_size, measurable_size);
	}

	void init_diag(double *diag, double *diag_, double total, double total_, int measurable_size_max) {
		init_diag_kernel << <GRID1D(measurable_size_max), BLOCK1D >> > (diag, diag_, total, total_, measurable_size_max);
	}

	void update_weights(double *weights, bool *observe, int measurable_size, double q, double phi, bool active) {
		update_weights_kernel << <GRID2D(measurable_size, measurable_size), BLOCK2D >> > (weights, observe, measurable_size, q, phi, active);
	}

	void get_weights_diag(double *weights, double *diag, double *diag_, int measurable_size) {
		get_weights_diag_kernel << <GRID1D(measurable_size), BLOCK1D >> > (weights, diag, diag_, measurable_size);
	}

	void calculate_target(double *diag, bool *target, int sensor_size) {
		calculate_target_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	}

	void update_thresholds(bool *dirs, double *thresholds, double total_, double q, double phi, int &sensor_size) {
		update_thresholds_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> > (dirs, thresholds, total_, q, phi, sensor_size);
	}

	void orient_all(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
		orient_all_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	}

	void multiply(bool *signal, bool *dirs, double *thresholds, double q, int measurable_size) {
		multiply_kernel << <1, BLOCK1D, 2 * measurable_size * sizeof(bool) >> >(signal, dirs, thresholds, false, 1 - q, measurable_size);
	}

	void floyd(bool *npdirs, int measurable_size) {
		floyd_kernel << <GRID2D(1, 1), BLOCK2D >> >(npdirs, measurable_size);
	}

	void dioid_square(int *dists, int sensor_size) {
		dioid_square_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dists, sensor_size);
	}

	void transpose_multiply(bool *npdirs, bool *signals, int measurable_size, int sig_count) {
		transpose_multiply_kernel << <GRID2D(sig_count, measurable_size), BLOCK2D >> >(npdirs, signals, measurable_size, sig_count);
	}

	void mask(bool *mask_amper, bool *mask, bool *current, int sensor_size){
		mask_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(mask_amper, mask, current, sensor_size);
	}

	void check_mask(bool *mask, int sensor_size) {
		check_mask_kernel << <GRID1D(sensor_size), BLOCK1D >> > (mask, sensor_size);
	}

	void delta_weight_sum(double *diag, bool *d1, float *result, int measurable_size) {
		delta_weight_sum_kernel << <GRID1D(measurable_size), BLOCK1D >> > (diag, d1, result, measurable_size);
	}

	void union_init(int *union_root, int sensor_size){
		union_init_kernel << <GRID1D(sensor_size), BLOCK1D >> >(union_root, sensor_size);
	}

	void check_dist(int *dists, float delta, int sensor_size) {
		check_dist_kernel << <GRID1D(sensor_size * sensor_size), BLOCK1D >> >(dists, delta, sensor_size * sensor_size);
	}

	void union_GPU(int *dists, int *union_root, int sensor_size) {
		union_GPU_kernel << <GRID1D(1), BLOCK1D >> > (dists, union_root, sensor_size);
	}
}