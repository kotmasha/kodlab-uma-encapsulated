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
__device__ double lowerThreshold(double q, double total, double phi, double T){
	if (total <= phi || T * total <= (1 - q) ) {
		return T;
	}
	else {
		return q * T;
	}
		
}

/*
raise the threshold value
*/
__device__ double raiseThreshold(double q, double total, double phi, double T){
	return ( 1.0 / q ) * T;
}

/*
This function does implies on GPU, using non-worker solution
Input: row, col info, attr_sensor size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
__device__ bool impliesGPU(int row, int col, double *weights, double total, double threshold){//implies
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	return rc_ < min(total * threshold, min(rc, min(r_c, r_c_))) || (rc_ == 0 && r_c == 0);// && rc > 0 && r_c_ > 0);
}

/*
This function does implies on GPU, using qualitative way
Input: row, col info, attr_sensor size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotmasha)
*/
__device__ bool impliesGPUQualitative(int row, int col, double *weights, double total, double threshold) {//implies
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	return qless(qmax(rc, r_c_), rc_);
	//return qless(qadd(qmax(rc,r_c_),threshold),rc_);
}

/*

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
Input: direction, weight, threshold matrix, lastTotal, q, phi, xy location
Output: None
*/
__device__ void updateThresholdsGPU(bool *dir, double *thresholds, double lastTotal, double q, double phi, int x, int y){
	//LATEST THRESHOLD FOR THIS SQUARE:
	double threshold = thresholds[ind(y, x)];
	//CHECK CONDITION FOR LOWERING THE THRESHOLD: (depends on previous entries of the orientation matrix)
	bool conditionToLower = (dir[ind(2 * y, 2 * x)] || dir[ind(compi(2 * y), 2 * x)] || dir[ind(2 * y, compi(2 * x))] || dir[ind(compi(2 * y), compi(2 * x))]);
	//CHECK CONDITION FOR RAISING THE THRESHOLD: (raising is currently disabled)
	bool conditionToRaise = false;

	//UPDATE THE THRESHOLD(S) FOR THIS SQUARE:
	// lowering the thresholds:
	if (conditionToLower) {
		thresholds[ind(y, x)] = lowerThreshold(q, lastTotal, phi, threshold);
	}
	//raising the thresholds:
	if (conditionToRaise) {
		thresholds[(ind(y, x))] = raiseThreshold(q, lastTotal, phi, threshold);
	}
}

/*
GPU method: updates a "square" in the orientation matrix
Input: dir matrix, weights matrix, thresholds matrix, xy location in dir matrix
Output: None
*/
__device__ void orientSquareGPU(bool *dir, double *weights, double *thresholds, double total, int x, int y) {//orient_square
	//OBTAIN THE LOCAL THRESHOLD
	double threshold = thresholds[ind(y / 2, x / 2)];
	//UPDATE THE ORIENTATION MATRIX
	if(y >= x)
		dir[ind(y, x)] = impliesGPU(y, x, weights, total, threshold);
	else
		dir[ind(compi(x), compi(y))] = impliesGPU(compi(x), compi(y), weights, total, threshold);
	if (compi(y) >= x)
		dir[ind(compi(y), x)] = impliesGPU(compi(y), x, weights, total, threshold);
	else
		dir[ind(compi(x), y)] = impliesGPU(compi(x), y, weights, total, threshold);
	if (y >= compi(x))
		dir[ind(y, compi(x))] = impliesGPU(y, compi(x), weights, total, threshold);
	else
		dir[ind(x, compi(y))] = impliesGPU(x, compi(y), weights, total, threshold);
	if (compi(y) >= compi(x))
		dir[ind(compi(y), compi(x))] = impliesGPU(compi(y), compi(x), weights, total, threshold);
	else
		dir[ind(x, y)] = impliesGPU(x, y, weights, total, threshold);
}

/*
GPU method: updates a "square" in the orientation matrix
Input: dir matrix, weights matrix, thresholds matrix, xy location in dir matrix
Output: None
*/
__device__ void orientSquareGPUQualitative(bool *dir, double *weights, double *thresholds, double total, int x, int y) {//orient_square
	double threshold = thresholds[ind(y / 2, x / 2)];//OBTAIN THE LOCAL THRESHOLD
	//UPDATE THE ORIENTATION MATRIX
	if (y >= x)
		dir[ind(y, x)] = impliesGPUQualitative(y, x, weights, total, threshold);
	else
		dir[ind(compi(x), compi(y))] = impliesGPUQualitative(compi(x), compi(y), weights, total, threshold);
	if (compi(y) >= x)
		dir[ind(compi(y), x)] = impliesGPUQualitative(compi(y), x, weights, total, threshold);
	else
		dir[ind(compi(x), y)] = impliesGPUQualitative(compi(x), y, weights, total, threshold);
	if (y >= compi(x))
		dir[ind(y, compi(x))] = impliesGPUQualitative(y, compi(x), weights, total, threshold);
	else
		dir[ind(x, compi(y))] = impliesGPUQualitative(x, compi(y), weights, total, threshold);
	if (compi(y) >= compi(x))
		dir[ind(compi(y), compi(x))] = impliesGPUQualitative(compi(y), compi(x), weights, total, threshold);
	else
		dir[ind(x, y)] = impliesGPUQualitative(x, y, weights, total, threshold);
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
__global__ void initMask_kernel(bool *mask, int initialSize, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index < initialSize * 2) mask[index] = false;
		else mask[index] = true;
	}
}

/*
initiate diag value, the value should be half of total value
Input: diag and old diag list, total, last total, and attr_sensor size
*/
__global__ void initDiag_kernel(double *diag, double *diag_, double total, double lastTotal, int attrSensorSize){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attrSensorSize){
		diag[index] = total / 2.0;
		diag_[index] = lastTotal / 2.0;
	}
}

/*
This function is update weights for discounted agent, using non-worker solution
Input: weight matrix, observe bool value from python side and attr_sensor size
Output: None
*/
__global__ void updateWeightsStationary_kernel(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < attrSensorSize){
		if(activity){
			weights[ind(indexY, indexX)] = q * weights[ind(indexY, indexX)] + (1 - q) * observe[indexX] * observe[indexY] * phi;
		}
	}
}

/*
Deprecated
*/
__global__ void updateWeightsForgetful_kernel(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexX <= indexY && indexY < attrSensorSize){
	    weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * activity * phi;
	}
}

/*
update the weight matrix based on observe signal
Input: weight, observe signal, attr_sensor size, q, phi value, and activity(indicating whether the snapshot is active)
*/
__global__ void updateWeights_kernel(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool activity){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	// since $activity$ is independent of position in the matrix, better to update the weight this way:
	if (activity) {
		if (indexX <= indexY && indexY < attrSensorSize) {
			weights[ind(indexY, indexX)] = q * weights[ind(indexY, indexX)] + (1.0 - q) * observe[indexX] * observe[indexY] * phi;
		}
	}
	/* OLDER UPDATE
	if (indexX <= indexY && indexY < attrSensorSize) {
		if (activity) {
			weights[ind(indexY, indexX)] = weights[ind(indexY, indexX)] * q + (1 - q) * observe[indexX] * observe[indexY] * phi;
			}
		}
	*/
}

/*
update the weight matrix based on observe signal, for trial purpose only
*/
__global__ void updateWeightsQualitative_kernel(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool activity) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	// since $activity$ is position-independent, better put it in front of all other decisions (no activity=no weight update needed)
	if (activity) {
		if (indexX <= indexY && indexY < attrSensorSize) {
			if (observe[indexX] && observe[indexY]) {
				if (weights[ind(indexY, indexX)] < -0.5) {// do it in a hack way, using == may be dangerous since it is double
					weights[ind(indexY, indexX)] = phi;
				}
				else {
					weights[ind(indexY, indexX)] = min(weights[ind(indexY, indexX)], phi);
				}
			}
		}
	}
}

/*
get the weight matrix's diag value, store in diag and update old diag
Input: weight matrix, diag, old diag, attr_sensor size
*/
__global__ void getWeightsDiag_kernel(double *weights, double *diags, double *diags_, int attrSensorSize){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attrSensorSize){
		int idx = ind(index, index);
		diags_[index] = diags[index];
		diags[index] = weights[idx];
	}
}

/*
calculate the target value based on the attr_sensor vector value
Input: attr_sensor list, target signal, sensor size
*/
__global__ void calculateTarget_kernel(double *attr_sensor, bool *target, int sensor_size){
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
calculate the target value based on the attr_sensor vector value
Input: attr_sensor list, target signal, sensor size
*/
__global__ void calculateTargetQualitative_kernel(double *attr_sensor, bool *target, int sensor_size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < sensor_size) {
		if (qless(attr_sensor[2 * index], attr_sensor[2 * index + 1])) {
			target[2 * index] = true;
			target[2 * index + 1] = false;
		}
		else if (qless(attr_sensor[2 * index + 1], attr_sensor[2 * index])) {
			target[2 * index] = false;
			target[2 * index + 1] = true;
		}
		else {
			target[2 * index] = false;
			target[2 * index + 1] = false;
		}
	}
}

/* 
GPU method: updating the thresholds prior to orientation update
Input: dir weights, thresholds, lastTotal, q, phi, attrSensorSize
Output: None
*/
__global__ void updateThreshold_kernel(bool *dir, double *thresholds, double lastTotal, double q, double phi, int sensor_size) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexY < sensor_size && indexX <= indexY) {
		updateThresholdsGPU(dir, thresholds, lastTotal, q, phi, indexX, indexY);
	}
}

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and attr_sensor size
Output: None
*/
__global__ void orientAll_kernel(bool *dir, double *weights, double *thresholds, double total, int sensor_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexY < sensor_size && indexX < indexY){
		orientSquareGPU(dir, weights, thresholds, total, indexX * 2, indexY * 2);
	}
}

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and attr_sensor size
Output: None
*/
__global__ void orientAllQualitative_kernel(bool *dir, double *weights, double *thresholds, double total, int sensor_size) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexY < sensor_size && indexX < indexY) {
		orientSquareGPUQualitative(dir, weights, thresholds, total, indexX * 2, indexY * 2);
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
__global__ void copyNpdir_kernel(bool *npdir, bool *dir, int attrSensorSize) {
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if (indexX < attrSensorSize && indexY < attrSensorSize) {
		if (indexX <= indexY) {//normal case
			npdir[npdirInd(indexY, indexX)] = dir[ind(indexY, indexX)];
		}
		else if (indexY % 2 == 0 && indexX == indexY + 1) {
			npdir[npdirInd(indexY, indexX)] = false;
		}
	}
}

/*
floyd kernel, calculating the npdir
Input: dir matrix, attr_sensor size
*/
__global__ void floyd_kernel(bool *dir, int attrSensorSize) {
	int indexX = threadIdx.x;
	int indexY = threadIdx.y;
	for (int i = 0; i < attrSensorSize; ++i) {
		int x = indexX, y = indexY;
		while (y < attrSensorSize) {
			if (y < x && (x % 2 != 0 || x != y + 1)) {
				y += THREAD2D;
				x = indexX;
				continue;
			}
			dir[npdirInd(y, x)] = dir[npdirInd(y, x)] || (dir[npdirInd(y, i)] && dir[npdirInd(i, x)]);
			x += THREAD2D;
		}
		__syncthreads();
	}
}

/*
dioid square for slhc, using tile-based algorithm
Input: int type of distance, width of matrix(square matrix)
*/
__global__ void dioidSquare_kernel(int* Md, int Width) {
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
			Mds[ty][tx] = Md[npdirInd(y, x)];
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
__global__ void transposeMultiply_kernel(bool* Md, bool *Nd, int Width, int Height) {
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
			Mds[ty][tx] = Md[npdirInd(y, x)];
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
__global__ void checkMask_kernel(bool *mask, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		if(mask[2 * index]) mask[2 * index + 1] = false;
	}
}

/*
get the delta weight sum
Input: attr_sensor(diag) value, current signal, tmp variable for result, and attr_sensor size
*/
__global__ void deltaWeightSum_kernel(double *attr_sensor, bool *signal, float *result, int attrSensorSize){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < attrSensorSize){
		atomicAdd(result, signal[index] * (attr_sensor[index] - attr_sensor[compi(index)]));
	}
}

/*
init union find, set the root of a index to be the index itself
Input: root list, sensor size
*/
__global__ void unionInit_kernel(int *root, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		root[index] = index;
	}
}

/*
check the dist kernel, if small than delta, mark 1(same group), otherwise mark 0
*/
__global__ void checkDist_kernel(int *data, float delta, int size) {
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
__global__ void unionGPU_kernel(int *data, int *root, int size) {
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
		if (npdir[npdirInd(2 * index, 2 * index + 1)]) negligible[2 * index] = true;
		if (npdir[npdirInd(2 * index + 1, 2 * index)]) negligible[2 * index + 1] = true;
	}
}

/*
---------------------AGENT---------------------
*/


void uma_base::initMask(bool *mask, int initialSize, int attrSensorSize) {
	initMask_kernel << <GRID1D(attrSensorSize), BLOCK1D >> > (mask, initialSize, attrSensorSize);
	umaBaseLogger.debug("initMask_kernal invoked!");
	cudaCheckErrors("check initMask error");
}

void uma_base::initDiag(double *diag, double *diag_, double total, double total_, int attrSensorSize_max) {
	initDiag_kernel << <GRID1D(attrSensorSize_max), BLOCK1D >> > (diag, diag_, total, total_, attrSensorSize_max);
	umaBaseLogger.debug("initDiag_kernel invoked!");
	cudaCheckErrors("check initDiag error");
}

void uma_base::updateWeights(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool active) {
	updateWeights_kernel << <GRID2D(attrSensorSize, attrSensorSize), BLOCK2D >> > (weights, observe, attrSensorSize, q, phi, active);
	umaBaseLogger.debug("updateWeights invoked!");
	cudaCheckErrors("check_updateWeights error");
}

void uma_base::getWeightsDiag(double *weights, double *diag, double *diag_, int attrSensorSize) {
	getWeightsDiag_kernel << <GRID1D(attrSensorSize), BLOCK1D >> > (weights, diag, diag_, attrSensorSize);
	umaBaseLogger.debug("getWeightsDiag_kernel invoked!");
	cudaCheckErrors("check getWeightsDiag error");
}

void uma_base::calculateTarget(double *diag, bool *target, int sensor_size) {
	calculateTarget_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	umaBaseLogger.debug("calculateTarget_kernel invoked!");
	cudaCheckErrors("check calculateTarget error");
}

void uma_base::updateThresholds(bool *dirs, double *thresholds, double total_, double q, double phi, int sensor_size) {
	updateThreshold_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> > (dirs, thresholds, total_, q, phi, sensor_size);
	umaBaseLogger.debug("updateThreshold_kernel invoked!");
	cudaCheckErrors("check updateThresholds error");
}

void uma_base::orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
	orientAll_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	umaBaseLogger.debug("orientAll_kernel invoked!");
	cudaCheckErrors("check orientAll error");
}

void uma_base::dfs(bool *signal, bool *dirs, double *thresholds, double q, int attrSensorSize) {
	dfs_kernel << <1, BLOCK1D, 2 * attrSensorSize * sizeof(bool) >> >(signal, dirs, thresholds, false, 1 - q, attrSensorSize);
	umaBaseLogger.debug("dfs_kernel invoked!");
	cudaCheckErrors("check dfs error");
}

void uma_base::floyd(bool *npdirs, int attrSensorSize) {
	floyd_kernel << <GRID2D(1, 1), BLOCK2D >> >(npdirs, attrSensorSize);
	umaBaseLogger.debug("floyd_kernel invoked!");
	cudaCheckErrors("check floyd error");
}

void uma_base::dioidSquare(int *dists, int sensor_size) {
	dioidSquare_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dists, sensor_size);
	umaBaseLogger.debug("diod_square_kernel invoked!");
	cudaCheckErrors("check diod_square error");
}

void uma_base::transposeMultiply(bool *npdirs, bool *signals, int attrSensorSize, int sig_count) {
	transposeMultiply_kernel << <GRID2D(sig_count, attrSensorSize), BLOCK2D >> >(npdirs, signals, attrSensorSize, sig_count);
	umaBaseLogger.debug("transposeMultiply_kernel invoked!");
	cudaCheckErrors("check transposeMultiply error");
}

void uma_base::multiply(bool *npdirs, bool *signals, int attrSensorSize, int sig_count) {
	multiply_kernel << <GRID2D(sig_count, attrSensorSize), BLOCK2D >> >(npdirs, signals, attrSensorSize, sig_count);
	umaBaseLogger.debug("multiply_kernel invoked!");
	cudaCheckErrors("check multiply error");
}

void uma_base::mask(bool *mask_amper, bool *mask, bool *current, int sensor_size){
	mask_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(mask_amper, mask, current, sensor_size);
	umaBaseLogger.debug("mask_kernel invoked!");
	cudaCheckErrors("check mask error");
}

void uma_base::checkMask(bool *mask, int sensor_size) {
	checkMask_kernel << <GRID1D(sensor_size), BLOCK1D >> > (mask, sensor_size);
	umaBaseLogger.debug("checkMask_kernel invoked!");
	cudaCheckErrors("check checkMask error");
}

void uma_base::deltaWeightSum(double *diag, bool *d1, float *result, int attrSensorSize) {
	deltaWeightSum_kernel << <GRID1D(attrSensorSize), BLOCK1D >> > (diag, d1, result, attrSensorSize);
	umaBaseLogger.debug("deltaWeightSum_kernel invoked!");
	cudaCheckErrors("check deltaWeightSum error");
}

void uma_base::unionInit(int *union_root, int sensor_size){
	unionInit_kernel << <GRID1D(sensor_size), BLOCK1D >> >(union_root, sensor_size);
	umaBaseLogger.debug("unionInit_kernel invoked!");
	cudaCheckErrors("check unionInit error");
}

void uma_base::checkDist(int *dists, float delta, int sensor_size) {
	checkDist_kernel << <GRID1D(sensor_size * sensor_size), BLOCK1D >> >(dists, delta, sensor_size * sensor_size);
	umaBaseLogger.debug("checkDist_kernel invoked!");
	cudaCheckErrors("check checkDist error");
}

void uma_base::unionGPU(int *dists, int *union_root, int sensor_size) {
	unionGPU_kernel << <GRID1D(1), BLOCK1D >> > (dists, union_root, sensor_size);
	umaBaseLogger.debug("unionGPU_kernel invoked!");
	cudaCheckErrors("check unionGPU error");
}

void uma_base::copyNpdir(bool *npdir, bool *dir, int attrSensorSize) {
	copyNpdir_kernel << <GRID2D(attrSensorSize, attrSensorSize), BLOCK2D >> > (npdir, dir, attrSensorSize);
	umaBaseLogger.debug("copyNpdir_kernel invoked!");
	cudaCheckErrors("check copyNpdir error");
}

void uma_base::negligible(bool *npdir, bool *negligible, int sensor_size) {
	negligible_kernel << <GRID1D(sensor_size), BLOCK1D >> > (npdir, negligible, sensor_size);
	umaBaseLogger.debug("negligible_kernel invoked!");
	cudaCheckErrors("check neligible error");
}

void uma_base_qualitative::updateWeights(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool active) {
	updateWeightsQualitative_kernel << <GRID2D(attrSensorSize, attrSensorSize), BLOCK2D >> > (weights, observe, attrSensorSize, q, phi, active);
	umaBaseLogger.debug("updateWeights_qualitative invoked!");
}

void uma_base_qualitative::orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
	orientAllQualitative_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	umaBaseLogger.debug("orientAllQualitative_kernel invoked!");
}

void uma_base_qualitative::calculateTarget(double *diag, bool *target, int sensor_size) {
	calculateTargetQualitative_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	umaBaseLogger.debug("calculateTargetQualitative_kernel invoked!");
}

//for now reuse the stationary kernel, as there is nothing different
void uma_base_discounted::updateWeights(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool active) {
	updateWeights_kernel << <GRID2D(attrSensorSize, attrSensorSize), BLOCK2D >> > (weights, observe, attrSensorSize, q, phi, active);
	umaBaseLogger.debug("updateWeights_discounted invoked!");
}

void uma_base_discounted::orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
	orientAll_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	umaBaseLogger.debug("orientAllDiscounted_kernel invoked!");
}

void uma_base_discounted::calculateTarget(double *diag, bool *target, int sensor_size) {
	calculateTarget_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	umaBaseLogger.debug("calculateTargetDiscounted_kernel invoked!");
}

//for now reuse the stationary kernel, as there is nothing different
void uma_base_empirical::updateWeights(double *weights, bool *observe, int attrSensorSize, double q, double phi, bool active) {
	updateWeights_kernel << <GRID2D(attrSensorSize, attrSensorSize), BLOCK2D >> > (weights, observe, attrSensorSize, q, phi, active);
	umaBaseLogger.debug("updateWeights_empirical invoked!");
}

void uma_base_empirical::orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size) {
	orientAll_kernel << <GRID2D(sensor_size, sensor_size), BLOCK2D >> >(dirs, weights, thresholds, total, sensor_size);
	umaBaseLogger.debug("orientAllEmpirical_kernel invoked!");
}

void uma_base_empirical::calculateTarget(double *diag, bool *target, int sensor_size) {
	calculateTarget_kernel << <GRID1D(sensor_size), BLOCK1D >> > (diag, target, sensor_size);
	umaBaseLogger.debug("calculateTargetEmpirical_kernel invoked!");
}