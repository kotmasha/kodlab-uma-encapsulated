#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "Snapshot.h"
#include "Sensor.h"
#include "UMATest.h"
#include "DataManager.h"

using namespace std;

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

/*
----------------------HOST DEVICE---------------------
*/

__host__ __device__ int compi(int x) {
	if (x % 2 == 0) return x + 1;
	else return x - 1;
}

/*
This function is for extracting the 1d index from a triangelized 2d structure
The function requires that the 'row' has to be not small than the 'col'
*/
__host__ __device__ int ind(int row, int col){
	if(row >= col)
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

/*
---------------------HELPER FUNCTION-----------------------
*/

__global__ void alltrue(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = true;
	}
}

__global__ void allfalse(bool *b, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		b[index] = false;
	}
}

__global__ void bool2int(bool *b, int *i, int size) {
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
__global__ void conjunction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] && b2[index];
	}
}

/*
This function computes the difference of two lists
Input: two bool lists (same length)
Output: None
*/
__global__ void subtraction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] && !b2[index];
	}
}

/*
This function does the disjunction for two lists
Input: two bool lists
Output: None
*/
__global__ void disjunction_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b1[index] = b1[index] || b2[index];
	}
}

/*
This function does compi, conjunction together
Input: two boolean lists
Output: None
*/
__global__ void negate_conjunction_star_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		bool m1 = b1[2 * index];
		bool m2 = b1[2 * index + 1];
		b1[2 * index] = m1 && !b2[2 * index + 1];
		b1[2 * index + 1] = m2 && !b2[2 * index];
	}
}

__global__ void conjunction_star_kernel(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index%2 == 0){
			b1[index] = b1[index] && b2[index+1];
		}
		else{
			b1[index] = b1[index] && b2[index-1];
		}
	}
}

__global__ void up2down(bool *b1, bool *b2, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		b2[index] = !b1[compi(index)];
	}
}
/*
---------------------HELPER FUNCTION-----------------------
*/

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


__global__ void dioid_square_GPU(int* Md, int Width) {
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


__global__ void transpose_multiply_GPU(bool* Md, bool *Nd, int Width, int Height) {
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

__global__ void check_mask(bool *mask, int sensor_size){
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

__global__ void union_init_GPU(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

__global__ void check_dist_GPU(int *data, float delta, int size) {
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

/*
----------------------GLOBAL--------------------
*/


/*
---------------------AGENT---------------------
*/

void DataManager::init_other_parameter(double total){
	for(int i = 0; i < _measurable_size_max; ++i){
		h_observe[i] = false;
		h_signal[i] = false;
		h_load[i] = false;
		h_mask[i] = false;
		h_current[i] = false;
		h_target[i] = false;
		h_prediction[i] = false;
		h_up[i] = false;
		h_down[i] = false;
		h_diag[i] = total / 2.0;
		h_diag_[i] = total / 2.0;
	}
	for (int i = 0; i < _sensor_size_max; ++i) {
		h_union_root[i] = 0;
	}

	cudaMemset(dev_observe, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_signal, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_load, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_mask, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_current, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_target, 0, _measurable_size_max * sizeof(bool));
	
	cudaMemset(dev_d1, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_d2, 0, _measurable_size_max * sizeof(bool));

	cudaMemset(dev_union_root, 0, _sensor_size_max * sizeof(bool));
	//cudaMemset(dev_npdirs, 0, _measurable_size_max * sizeof(bool));
	init_diag_kernel<<<(_measurable_size_max + 255) / 256, 256>>>(dev_diag, dev_diag_, total, total, _measurable_size_max);
}
/*
This function is an independent up function on GPU
It only use signal to do dfs, result is stored in Gsignal after using the function
Input: signal to be dfsed
Output: None
*/
void DataManager::up_GPU(vector<bool> &signal, double q, bool is_stable){
	for(int i = 0; i < _measurable_size; ++i) h_signal[i] = signal[i];
	cudaMemcpy(dev_signal, h_signal, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	
	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(dev_signal, dev_dirs, dev_thresholds, is_stable, 1 - q, _measurable_size);

	cudaMemcpy(h_up, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaCheckErrors("kernel fails");
}

/*
Call this function only when dev_load is ready!
*/
void DataManager::setLSignals(vector<vector<bool> > &signals) {
	int sig_count = signals.size();
	for (int i = 0; i < sig_count; ++i) {
		for (int j = 0; j < signals[i].size(); ++j) {
			h_lsignals[i * _measurable_size + j] = signals[i][j];
		}
	}
	cudaMemcpy(dev_lsignals, h_lsignals, sig_count * _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	for (int i = 0; i < sig_count; ++i) {
		disjunction_kernel<<<GRID1D(_measurable_size), BLOCK1D>>>(dev_lsignals + i * _measurable_size, dev_load, _measurable_size);
	}
}

void DataManager::ups_GPU(int sig_count) {
	transpose_multiply_GPU << <GRID2D(sig_count, _measurable_size), BLOCK2D >> >(dev_npdirs, dev_signals, _measurable_size, sig_count);
}

void DataManager::propagate_mask() {
	allfalse<<<GRID1D(_measurable_size * _sensor_size), BLOCK1D>>>(dev_npdir_mask, _sensor_size * _measurable_size);
	for (int i = 0; i < _sensor_size; ++i) {
		cudaMemcpy(dev_npdir_mask + _measurable_size * i, dev_mask_amper + ind(i, 0) * 2, (ind(i + 1, 0) - ind(i, 0)) * 2 * sizeof(bool), cudaMemcpyDeviceToDevice);
	}

	transpose_multiply_GPU << <GRID2D(_sensor_size, _measurable_size), BLOCK2D>> >(dev_npdirs, dev_npdir_mask, _measurable_size, _sensor_size);

	//cudaMemcpy(h_npdir_mask, dev_npdir_mask, _sensor_size * _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void DataManager::floyd_GPU() {
	cudaMemcpy(dev_npdirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	floyd_kernel<<<GRID2D(1, 1), BLOCK2D>>>(dev_npdirs, _measurable_size);
	cudaMemcpy(h_npdirs, dev_npdirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaCheckErrors("kernel fails");
}

vector<vector<vector<bool> > > DataManager::abduction(vector<vector<bool> > &signals) {
	vector<vector<vector<bool> > > results;
	vector<vector<bool> > even_results, odd_results;
	for (int i = 0; i < signals.size(); ++i) {
		vector<int> even_idx, odd_idx;
		for (int j = 0; j < signals[0].size() / 2; ++j) {
			if (signals[i][2 * j]) even_idx.push_back(j);
			if (signals[i][2 * j + 1]) odd_idx.push_back(j);
		}
		if (even_idx.empty()) {
			vector<bool> tmp(_measurable_size, false);
			even_results.push_back(tmp);
		}
		else {
			alltrue<<<GRID1D(_measurable_size), BLOCK1D>>>(dev_signal, _measurable_size);
			for (int j = 0; j < even_idx.size(); ++j) {
				conjunction_kernel<<<GRID1D(_measurable_size), BLOCK1D>>>(dev_signal, dev_npdir_mask + even_idx[j] * _measurable_size, _measurable_size);
			}
			cudaMemcpy(h_signal, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
			vector<bool> tmp;
			for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signal[j]);
			even_results.push_back(tmp);
		}
		if (odd_idx.empty()) {
			vector<bool> tmp(_measurable_size, false);
			odd_results.push_back(tmp);
		}
		else {
			allfalse << <GRID1D(_measurable_size), BLOCK1D >> >(dev_signal, _measurable_size);
			for (int j = 0; j < odd_idx.size(); ++j) {
				disjunction_kernel << <GRID1D(_measurable_size), BLOCK1D >> >(dev_signal, dev_npdir_mask + odd_idx[j] * _measurable_size, _measurable_size);
			}
			cudaMemcpy(dev_signals, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_lsignals, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
			cudaMemset(dev_load, false, _measurable_size * sizeof(bool));
			propagates_GPU(1);
			cudaMemcpy(h_signals, dev_lsignals, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
			vector<bool> tmp;
			for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signals[j]);
			odd_results.push_back(tmp);
		}
	}
	results.push_back(even_results);
	results.push_back(odd_results);
	return results;
}

vector<vector<int> > DataManager::blocks_GPU(float delta) {
	int t = floor(log(_sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		dioid_square_GPU << <GRID2D(_sensor_size, _sensor_size), BLOCK2D>> >(dev_dists, _sensor_size);
	}

	cudaMemcpy(h_dists, dev_dists, _sensor_size * _sensor_size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _sensor_size; ++i) {
		for (int j = 0; j < _sensor_size; ++j) cout << h_dists[i * _sensor_size + j] << ",";
		cout << endl;
	}

	union_init_GPU << <GRID1D(_sensor_size), BLOCK1D>> >(dev_union_root, _sensor_size);
	check_dist_GPU << <GRID1D(_sensor_size * _sensor_size), BLOCK1D >> >(dev_dists, delta, _sensor_size * _sensor_size);

	union_GPU << <GRID1D(1), BLOCK1D>> >(dev_dists, dev_union_root, _sensor_size);
	cudaMemcpy(h_union_root, dev_union_root, _sensor_size * sizeof(int), cudaMemcpyDeviceToHost);

	map<int, int> m;
	vector<vector<int> > result;
	for (int i = 0; i < _sensor_size; ++i) {
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

	return result;
}

void DataManager::gen_mask(int initial_size){
	init_mask_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_mask, initial_size, _measurable_size);

	dim3 dimGrid((_sensor_size + 15) / 16,(_sensor_size + 15) / 16);
	dim3 dimBlock(16, 16);

	mask_kernel<<<dimGrid, dimBlock>>>(dev_mask_amper, dev_mask, dev_current, _sensor_size);
	check_mask<<<(_sensor_size + 255) / 256, 256>>>(dev_mask, _sensor_size);
}

void DataManager::propagates_GPU(int sig_count) {
	transpose_multiply_GPU << <GRID2D(sig_count, _measurable_size), BLOCK2D >> >(dev_npdirs, dev_lsignals, _measurable_size, sig_count);
	transpose_multiply_GPU << <GRID2D(sig_count, _measurable_size), BLOCK2D >> >(dev_npdirs, dev_signals, _measurable_size, sig_count);

	negate_conjunction_star_kernel << <GRID1D(sig_count * _measurable_size), BLOCK1D >> >(dev_lsignals, dev_signals, _sensor_size);
}

/*
This function do propagate on GPU
//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
Result is stored in Gload
Ask Kotomasha for mathematic questions
Input: signal and load
Output: None
*/
void DataManager::propagate_GPU(bool *signal){//propagate
	cudaMemcpy(dev_signal, signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, _measurable_size * sizeof(bool));

	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(dev_load, dev_dirs, dev_thresholds, false, 0, _measurable_size);
	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(dev_signal, dev_dirs, dev_thresholds, false, 0, _measurable_size);

	// standard operations
	disjunction_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, _measurable_size);
	negate_conjunction_star_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, _sensor_size);
	
	cudaMemcpy(h_load, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}


float DataManager::distance(bool *signal1, bool *signal2) {
	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	cudaMemcpy(dev_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	propagate_GPU(signal1);
	cudaMemcpy(signal1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	cudaMemcpy(dev_signal, signal2, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	propagate_GPU(signal2);
	cudaMemcpy(signal2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	conjunction_star_kernel << <(_measurable_size + 255) / 256, 256 >> >(signal1, signal2, _measurable_size);
	//apply weights to the computed distance signal and output the result:
	/*
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
	*/
	//weights are w_{xx}-w_{x*x*}:
	/*
	delta_weight_sum_kernel << <(_measurable_size + 255) / 256, 256 >> > (dev_diag, signal1, dev_result, _measurable_size);
	cudaMemcpy(tmp_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	float result = tmp_result[0];
	delete[] tmp_result;
	cudaFree(dev_result);
	return result;
	*/
	//weights are all 1:

	cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
}

float DataManager::divergence() {
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_d2, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(dev_d1);
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(dev_d2);
	cudaMemcpy(dev_d2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	subtraction_kernel << <(_measurable_size + 255) / 256, 256 >> >(dev_d1, dev_d2, _measurable_size);
	//apply weights to the computed divergence signal and output the result:
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
	//weights are w_{xx}-w_{x*x*}:
	
	delta_weight_sum_kernel << <(_measurable_size + 255) / 256, 256 >> > (dev_diag, dev_d1, dev_result, _measurable_size);
	cudaMemcpy(tmp_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	float result = tmp_result[0];
	delete[] tmp_result;
	cudaFree(dev_result);
	return result;
	
	//weights are all 1:
	/*
	cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
	*/
}

void DataManager::update_diag(){
	get_weights_diag_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_weights, dev_diag, dev_diag_, _measurable_size);
}

void DataManager::calculate_target(){
	calculate_target_kernel<<<(_sensor_size + 255) / 256, 256>>>(dev_diag, dev_target, _sensor_size);
	cudaMemcpy(h_target, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void DataManager::update_weights(double q, double phi, bool active){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, q, phi, active);
}

void DataManager::orient_all(double total){
	orient_all_kernel<<<GRID2D(_sensor_size, _sensor_size), BLOCK2D>>>(dev_dirs, dev_weights, dev_thresholds, total, _sensor_size);
}

void DataManager::update_thresholds(double q, double phi, double total_) {
	dim3 dimGrid1((_sensor_size + 15) / 16, (_sensor_size + 15) / 16);
	dim3 dimBlock1(16, 16);
	update_thresholds_kernel << <dimGrid1, dimBlock1 >> >(dev_dirs, dev_thresholds, total_, q, phi, _sensor_size);
}
/*
----------------------------SNAPSHOT------------------------
*/

/*
---------------------------SNAPSHOT_STATIONARY---------------------
*/

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
/*
void Snapshot_Stationary::update_weights(bool activity){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_stationary<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, _q, _phi, activity);
}
*/

/* SIQI: This is the method calling the threshold updating kernel from Agent_Stationary
This function realizes the update of learning thresholds on GPU.
See update_thresholds_kernel for more details
Input: None
Output: None
*/
/*
void Snapshot_Stationary::update_thresholds() {
	Snapshot::update_thresholds();
}
*/

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
/*
void Snapshot_Stationary::orient_all(){
	Snapshot::orient_all();
}
*/

/*
------------------------SNAPSHOT_STATIONARY----------------------------
*/


/*
---------------------------SNAPSHOT_FORGETFUL---------------------
*/
/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
/*
void Snapshot_Forgetful::update_weights(bool active){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_forgetful<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, _q, _phi, active);
}
*/

/* SIQI: This is the method calling the threshold update kernel for Agent_Forgetful
This function realizes the update of learning thresholds on GPU.
See update_thresholds_kernel for more details
Input: None
Output: None
*/
/*
void Snapshot_Forgetful::update_thresholds() {
	Snapshot::update_thresholds();
}
*/

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
/*
void Snapshot_Forgetful::orient_all(){
	Snapshot::orient_all();
}
*/

/*
------------------------SNAPSHOT_FORGETFUL----------------------------
*/