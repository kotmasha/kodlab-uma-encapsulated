#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "kernel.h"
#include "uma_base.cuh"
#include "kernel_util.cuh"
#include "device_util.h"
#include "Logger.h"

using namespace std;

static Logger umaBaseLogger("UMABase", "log/device.log");

/*
---------------------DEVICE------------------------
*/

/*
lower the threshold value
*/
__device__ double lower_threshold(double q, double total, double phi, double T){
	if (total <= phi || T <= (1 - q) / total) {
		return T;
	}
	else {
		return q * T;
	}
		
}

/*
raise the threshold value
*/
__device__ double raise_threshold(double q, double total, double phi, double T){
	return ( 1.0 / q ) * T;
}

/*
This function does implies on GPU, using non-worker solution
Input: row, col info, attr_sensor size, weight matrix and threshold
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
Input: row, col info, attr_sensor size, weight matrix and threshold
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

/*
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

/*
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
/*
initiate the mask kernel, filling all initial sensor to false, the other to true
Input: mask signal, initial size and attr_sensor size
*/
__global__ void init_mask_kernel(bool *mask, int initial_size, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index < initial_size * 2) mask[index] = false;
		else mask[index] = true;
	}
}

/*
initiate diag value, the value should be half of total value
Input: diag and old diag list, total, last total, and attr_sensor size
*/
__global__ void init_diag_kernel(double *diag, double *diag_, double total, double last_total, int attr_sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attr_sensor_size){
		diag[index] = total / 2.0;
		diag_[index] = last_total / 2.0;
	}
}

/*
This function is update weights for discounted agent, using non-worker solution
Input: weight matrix, observe bool value from python side and attr_sensor size
Output: None
*/
__global__ void update_weights_kernel_stationary(double *weights, bool *observe, int attr_sensor_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < attr_sensor_size){
		if(activity){
			weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * phi;
		}
	}
}

/*
Deprecated
*/
__global__ void update_weights_kernel_forgetful(double *weights, bool *observe, int attr_sensor_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < attr_sensor_size){
	    weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * activity * phi;
	}
}

/*
update the weight matrix based on observe signal
Input: weight, observe signal, attr_sensor size, q, phi value, and activity(indicating whether the snapshot is active)
*/
__global__ void update_weights_kernel(double *weights, bool *observe, int attr_sensor_size, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexX <= indexY && indexY < attr_sensor_size) {
		if (activity) {
			weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * phi;
		}
	}
}

/*
get the weight matrix's diag value, store in diag and update old diag
Input: weight matrix, diag, old diag, attr_sensor size
*/
__global__ void get_weights_diag_kernel(double *weights, double *diags, double *diags_, int attr_sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attr_sensor_size){
		int idx = ind(index, index);
		diags_[index] = diags[index];
		diags[index] = weights[idx];
	}
}

/*
calculate the target value based on the attr_sensor vector value
Input: attr_sensor list, target signal, sensor size
*/
__global__ void calculate_target_kernel(double *attr_sensor, bool *target, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(attr_sensor[2 * index] - attr_sensor[2 * index + 1] > 1e-12){
			target[2 * index] = true;
			target[2 * index + 1] = false;
		}
		else if(attr_sensor[2 * index] - attr_sensor[2 * index + 1] < 1e-12){
			target[2 * index] = false;
			target[2 * index + 1] = true;
		}
		else{
			target[2 * index] = false;
			target[2 * index + 1] = false;
		}
	}
}

/* 
GPU method: updating the thresholds prior to orientation update
Input: dir weights, thresholds, last_total, q, phi, attr_sensor_size
Output: None
*/
__global__ void update_thresholds_kernel(bool *dir, double *thresholds, double last_total, double q, double phi, int sensor_size) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexY < sensor_size && indexX <= indexY) {
		update_thresholds_GPU(dir, thresholds, last_total, q, phi, indexX, indexY);
	}
}

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and attr_sensor size
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
Input: bool list of data to be dfsed, direction matrix and attr_sensor size
Output: None
****************************No long usaged in latest version, but functionality remain***************************
*/
__global__ void dfs_kernel(bool *x, bool *dir, double *thresholds, bool is_stable, double lowest, int size){//dfs
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
		j += THREAD1D;
	}
	flag[0] = true;
	__syncthreads();
	while(flag[0]){
		flag[0] = false;
		__syncthreads();
		j = index;
		while(j < size){
			if(xs[j] == 1){
				j += THREAD1D;
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
			j += THREAD1D;
		}
		__syncthreads();
		j = index;
		while(j < size){
			xs[j] = ys[j];
			j += THREAD1D;
		}
		__syncthreads();
	}
	j = index;
	while(j < size){
		x[j] = ys[j];
		j += THREAD1D;
	}
}

/*
copy the npdir value from dir value, but need to take care of x=2i+1, y=2i case, set the value to false
Input: npdir matrix, dir matrix, attr_sensor size
*/
__global__ void copy_npdir_kernel(bool *npdir, bool *dir, int attr_sensor_size) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexX < attr_sensor_size && indexY < attr_sensor_size) {
		if (indexX <= indexY) {//normal case
			npdir[npdir_ind(indexY, indexX)] = dir[ind(indexY, indexX)];
		}
		else if (indexY % 2 == 0 && indexX == indexY + 1) {
			npdir[npdir_ind(indexY, indexX)] = false;
		}
	}
}

/*
floyd kernel, calculating the npdir
Input: dir matrix, attr_sensor size
*/
__global__ void floyd_kernel(bool *dir, int attr_sensor_size) {
	int indexX = threadIdx.x;
	int indexY = threadIdx.y;
	for (int i = 0; i < attr_sensor_size; ++i) {
		int x = indexX, y = indexY;
		while (y < attr_sensor_size) {
			if (y < x && (x % 2 != 0 || x != y + 1)) {
				y += THREAD2D;
				x = indexX;
				continue;
			}
			dir[npdir_ind(y, x)] = dir[npdir_ind(y, x)] || (dir[npdir_ind(y, i)] && dir[npdir_ind(i, x)]);
			x += THREAD2D;
		}
		__syncthreads();
	}
}

/*
dioid square for slhc, using tile-based algorithm
Input: int type of distance, width of matrix(square matrix)
*/
__global__ void dioid_square_kernel(int* Md, int Width) {
	const int TILE_WIDTH = THREAD2D;
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

/*
doing matrix multiplication, but only get 1/0
Input: matrix m(Width*Width), n(Width*Height), Width, Height
*/
__global__ void multiply_kernel(bool* Md, bool *Nd, int Width, int Height) {
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
			int y = Row;
			int x = i * TILE_WIDTH + tx;
			//if (y >= x) Mds[ty][tx] = Md[ind(y, x)];
			//else Mds[ty][tx] = Md[ind(compi(x), compi(y))];
			Mds[ty][tx] = Md[npdir_ind(y, x)];
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
doing transpose matrix multiplication, but only get 1/0
Input: matrix m(Width*Width), n(Width*Height), Width, Height
*/
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
			int y = i * TILE_WIDTH + tx;
			int x = Row;
			//if (y >= x) Mds[ty][tx] = Md[ind(y, x)];
			//else Mds[ty][tx] = Md[ind(compi(x), compi(y))];
			Mds[ty][tx] = Md[npdir_ind(y, x)];
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
Input: mask amper, destination mask address, current signal and sensor size
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

/*
after running mask kernel function, also need to turn 2i+1 to false when 2i is true(true, true case exist)
Input: mask signal, sensor size
*/
__global__ void check_mask_kernel(bool *mask, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(mask[2 * index]) mask[2 * index + 1] = false;
	}
}

/*
get the delta weight sum
Input: attr_sensor(diag) value, current signal, tmp variable for result, and attr_sensor size
*/
__global__ void delta_weight_sum_kernel(double *attr_sensor, bool *signal, float *result, int attr_sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attr_sensor_size){
		atomicAdd(result, signal[index] * (attr_sensor[index] - attr_sensor[compi(index)]));
	}
}

/*
init union find, set the root of a index to be the index it self
Input: root list, sensor size
*/
__global__ void union_init_kernel(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

/*
check the dist kernel, if small than delta, mark 1(same group), otherwise mark 0
*/
__global__ void check_dist_kernel(int *data, float delta, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		if (data[index] < delta) data[index] = 1;
		else data[index] = 0;
	}
}

/*
union find kernel
Input: data used for union find, root list, sensor size
*/
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
make negligible value
Input: npdir matrix, negligible list, sensor size
*/
__global__ void negligible_kernel(bool *npdir, bool *negligible, int sensor_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < sensor_size) {
		negligible[2 * index] = false;
		negligible[2 * index + 1] = false;
		if (npdir[npdir_ind(2 * index, 2 * index + 1)]) negligible[2 * index] = true;
		if (npdir[npdir_ind(2 * index + 1, 2 * index)]) negligible[2 * index + 1] = true;
	}
}

__global__ void new_episode_kernel(bool *current, int initial_sensor_size, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		if (index >= initial_sensor_size * 2) current[index] = false;
	}
}

/*
---------------------AGENT---------------------
*/


void uma_base::init_mask(bool *mask, int initial_size, int attr_sensor_size) {
	init_mask_kernel << <GRID1D(attr_sensor_size), BLOCK1D >> > (mask, initial_size, attr_sensor_size);
	umaBaseLogger.debug("init_mask_kernel invoked!");
}

void uma_base::init_diag(double *diag, double *diag_, double total, double total_, int attr_sensor_size_max) {
	init_diag_kernel << <GRID1D(attr_sensor_size_max), BLOCK1D >> > (diag, diag_, total, total_, attr_sensor_size_max);
	umaBaseLogger.debug("init_diag_kernel invoked!");
}

void uma_base::update_weights(double *weights, bool *observe, int attr_sensor_size, double q, double phi, bool active) {
	update_weights_kernel << <GRID2D(attr_sensor_size, attr_sensor_size), BLOCK2D >> > (weights, observe, attr_sensor_size, q, phi, active);
	umaBaseLogger.debug("update_weights invoked!");
}

void uma_base::get_weights_diag(double *weights, double *diag, double *diag_, int attr_sensor_size) {
	get_weights_diag_kernel << <GRID1D(attr_sensor_size), BLOCK1D >> > (weights, diag, diag_, attr_sensor_size);
	umaBaseLogger.debug("get_weights_diag_kernel invoked!");
}

void uma_base::calculate_target(double *diag, bool *target, int sensor_size) {
	calculate_target_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	umaBaseLogger.debug("calculate_target_kernel invoked!");
}

void uma_base::update_thresholds(bool *dirs, double *thresholds, double total_, double q, double phi, int sensor_size) {
	update_thresholds_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> > (dirs, thresholds, total_, q, phi, sensor_size);
	umaBaseLogger.debug("update_thresholds_kernel invoked!");
}

void uma_base::orient_all(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
	orient_all_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	umaBaseLogger.debug("orient_all_kernel invoked!");
}

void uma_base::dfs(bool *signal, bool *dirs, double *thresholds, double q, int attr_sensor_size) {
	dfs_kernel << <1, BLOCK1D, 2 * attr_sensor_size * sizeof(bool) >> >(signal, dirs, thresholds, false, 1 - q, attr_sensor_size);
	umaBaseLogger.debug("dfs_kernel invoked!");
}

void uma_base::floyd(bool *npdirs, int attr_sensor_size) {
	floyd_kernel << <GRID2D(1, 1), BLOCK2D >> >(npdirs, attr_sensor_size);
	umaBaseLogger.debug("floyd_kernel invoked!");
}

void uma_base::dioid_square(int *dists, int sensor_size) {
	dioid_square_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dists, sensor_size);
	umaBaseLogger.debug("diod_square_kernel invoked!");
}

void uma_base::transpose_multiply(bool *npdirs, bool *signals, int attr_sensor_size, int sig_count) {
	transpose_multiply_kernel << <GRID2D(sig_count, attr_sensor_size), BLOCK2D >> >(npdirs, signals, attr_sensor_size, sig_count);
	umaBaseLogger.debug("transpose_multiply_kernel invoked!");
}

void uma_base::multiply(bool *npdirs, bool *signals, int attr_sensor_size, int sig_count) {
	multiply_kernel << <GRID2D(sig_count, attr_sensor_size), BLOCK2D >> >(npdirs, signals, attr_sensor_size, sig_count);
	umaBaseLogger.debug("multiply_kernel invoked!");
}

void uma_base::mask(bool *mask_amper, bool *mask, bool *current, int sensor_size){
	mask_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(mask_amper, mask, current, sensor_size);
	umaBaseLogger.debug("mask_kernel invoked!");
}

void uma_base::check_mask(bool *mask, int sensor_size) {
	check_mask_kernel << <GRID1D(sensor_size), BLOCK1D >> > (mask, sensor_size);
	umaBaseLogger.debug("check_mask_kernel invoked!");
}

void uma_base::delta_weight_sum(double *diag, bool *d1, float *result, int attr_sensor_size) {
	delta_weight_sum_kernel << <GRID1D(attr_sensor_size), BLOCK1D >> > (diag, d1, result, attr_sensor_size);
	umaBaseLogger.debug("delta_weight_sum_kernel invoked!");
}

void uma_base::union_init(int *union_root, int sensor_size){
	union_init_kernel << <GRID1D(sensor_size), BLOCK1D >> >(union_root, sensor_size);
	umaBaseLogger.debug("union_init_kernel invoked!");
}

void uma_base::check_dist(int *dists, float delta, int sensor_size) {
	check_dist_kernel << <GRID1D(sensor_size * sensor_size), BLOCK1D >> >(dists, delta, sensor_size * sensor_size);
	umaBaseLogger.debug("check_dist_kernel invoked!");
}

void uma_base::union_GPU(int *dists, int *union_root, int sensor_size) {
	union_GPU_kernel << <GRID1D(1), BLOCK1D >> > (dists, union_root, sensor_size);
	umaBaseLogger.debug("union_GPU_kernel invoked!");
}

void uma_base::copy_npdir(bool *npdir, bool *dir, int attr_sensor_size) {
	copy_npdir_kernel << <GRID2D(attr_sensor_size, attr_sensor_size), BLOCK2D >> > (npdir, dir, attr_sensor_size);
	umaBaseLogger.debug("copy_npdir_kernel invoked!");
}

void uma_base::negligible(bool *npdir, bool *negligible, int sensor_size) {
	negligible_kernel << <GRID1D(sensor_size), BLOCK1D >> > (npdir, negligible, sensor_size);
	umaBaseLogger.debug("negligible_kernel invoked!");
}

void uma_base::new_episode(bool *current, int initial_sensor_size, int attr_sensor_size) {
	new_episode_kernel << <GRID1D(attr_sensor_size), BLOCK1D >> > (current, initial_sensor_size, attr_sensor_size);
}