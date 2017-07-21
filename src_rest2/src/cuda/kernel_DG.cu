#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "Global.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "UMATest.h"

using namespace std;

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

/*
This function is for extracting the 1d index from a triangelized 2d structure
The function requires that the 'row' has to be not small than the 'col'
*/
__host__ __device__ int ind(int row, int col){
	if(row >= col)
		return row * (row + 1) / 2 + col;
	else
		return col * (col + 1) / 2 + row;
}

__host__ __device__ int compi(int x){
	if(x % 2 == 0) return x + 1;
	else return x - 1;
}

/*
----------------------HOST DEVICE---------------------
*/

/*
---------------------HELPER FUNCTION-----------------------
*/

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
		if(index%2 == 0){
			b1[index] = b1[index] && !b2[index+1];
		}
		else{
			b1[index] = b1[index] && !b2[index-1];
		}
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

__global__ void init_mask_kernel(bool *mask, int base_sensor_size, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		if(index < base_sensor_size * 2) mask[index] = false;
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
/*
----------------------GLOBAL--------------------
*/


/*
---------------------AGENT---------------------
*/

void Snapshot::init_other_parameter(){
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
		h_diag[i] = _total / 2.0;
		h_diag_[i] = _total_ / 2.0;
	}

	cudaMemset(dev_observe, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_observe_, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_signal, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_load, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_mask, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_current, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_target, 0, _measurable_size_max * sizeof(bool));
	
	cudaMemset(dev_d1, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_d2, 0, _measurable_size_max * sizeof(bool));
	init_diag_kernel<<<(_measurable_size_max + 255) / 256, 256>>>(dev_diag, dev_diag_, _total, _total_, _measurable_size_max);
}
/*
This function is an independent up function on GPU
It only use signal to do dfs, result is stored in Gsignal after using the function
Input: signal to be dfsed
Output: None
*/
void Snapshot::up_GPU(vector<bool> signal, bool is_stable){
	for(int i = 0; i < _measurable_size; ++i) h_signal[i] = signal[i];
	cudaMemcpy(dev_signal, h_signal, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	
	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(dev_signal, dev_dirs, dev_thresholds, is_stable, 1 - _q, _measurable_size);

	cudaMemcpy(h_up, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaCheckErrors("kernel fails");
}

void Snapshot::gen_mask(){
	init_mask_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_mask, _base_sensor_size, _measurable_size);

	dim3 dimGrid((_sensor_size + 15) / 16,(_sensor_size + 15) / 16);
	dim3 dimBlock(16, 16);

	mask_kernel<<<dimGrid, dimBlock>>>(dev_mask_amper, dev_mask, dev_current, _sensor_size);
	check_mask<<<(_sensor_size + 255) / 256, 256>>>(dev_mask, _sensor_size);
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
void Snapshot::propagate_GPU(bool *signal, bool *load){//propagate
	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(load, dev_dirs, dev_thresholds, false, 0, _measurable_size);
	multiply_kernel<<<1, 512, 2 * _measurable_size * sizeof(bool)>>>(signal, dev_dirs, dev_thresholds, false, 0, _measurable_size);

	// standard operations
	disjunction_kernel<<<(_measurable_size + 255) / 256, 256>>>(load, signal, _measurable_size);
	negate_conjunction_star_kernel<<<(_measurable_size + 255) / 256, 256>>>(load, signal, _measurable_size);
	
	//cudaMemcpy(h_load, load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

float Snapshot::distance(bool *signal1, bool *signal2) {
	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(signal1, dev_load);
	cudaMemcpy(signal1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(signal2, dev_load);
	cudaMemcpy(signal2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	conjunction_star_kernel << <(_measurable_size + 255) / 256, 256 >> >(signal1, signal2, _measurable_size);
	//apply weights to the computed divergence signal and output the result:
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

float Snapshot::divergence(bool *signal1, bool *signal2) {
	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(signal1, dev_load);
	cudaMemcpy(signal1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(signal2, dev_load);
	cudaMemcpy(signal2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	subtraction_kernel << <(_measurable_size + 255) / 256, 256 >> >(signal1, signal2, _measurable_size);
	//apply weights to the computed divergence signal and output the result:
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
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

void Snapshot::calculate_total(bool active){
	get_weights_diag_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_weights, dev_diag, dev_diag_, _measurable_size);
	_total_ = _total;
	_total = _q * _total + (1 - _q) * _phi;
}

void Snapshot::calculate_target(){
	calculate_target_kernel<<<(_sensor_size + 255) / 256, 256>>>(dev_diag, dev_target, _sensor_size);
	cudaMemcpy(h_target, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void Snapshot::update_weights(bool active){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, _q, _phi, active);
}

void Snapshot::orient_all(){
	dim3 dimGrid1((_sensor_size + 15) / 16,(_sensor_size + 15) / 16);
	dim3 dimBlock1(16, 16);
	orient_all_kernel<<<dimGrid1, dimBlock1>>>(dev_dirs, dev_weights, dev_thresholds, _total, _sensor_size);
}

void Snapshot::update_thresholds() {
	dim3 dimGrid1((_sensor_size + 15) / 16, (_sensor_size + 15) / 16);
	dim3 dimBlock1(16, 16);
	update_thresholds_kernel << <dimGrid1, dimBlock1 >> >(dev_dirs, dev_thresholds, _total_, _q, _phi, _sensor_size);
}
/*
----------------------------SNAPSHOT------------------------
*/

/*
---------------------------SNAPSHOT_STATIONARY---------------------
*/
void Snapshot_Stationary::calculate_total(bool activity){
	get_weights_diag_kernel<<<(_measurable_size + 255) / 256, 256>>>(dev_weights, dev_diag, dev_diag_, _measurable_size);
	_total_ = _total;
	if(activity){
		_total = _q * _total + (1 - _q) * _phi;
	}
}

void Snapshot_Stationary::calculate_target(){
	Snapshot::calculate_target();
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
void Snapshot_Stationary::update_weights(bool activity){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_stationary<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, _q, _phi, activity);
}

/* SIQI: This is the method calling the threshold updating kernel from Agent_Stationary
This function realizes the update of learning thresholds on GPU.
See update_thresholds_kernel for more details
Input: None
Output: None
*/
void Snapshot_Stationary::update_thresholds() {
	Snapshot::update_thresholds();
}

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Snapshot_Stationary::orient_all(){
	Snapshot::orient_all();
}

/*
------------------------SNAPSHOT_STATIONARY----------------------------
*/


/*
---------------------------SNAPSHOT_FORGETFUL---------------------
*/
void Snapshot_Forgetful::calculate_total(bool active){
	Snapshot::calculate_total(active);
}

void Snapshot_Forgetful::calculate_target(){
	Snapshot::calculate_target();
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
void Snapshot_Forgetful::update_weights(bool active){
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_forgetful<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, _measurable_size, _q, _phi, active);
}

/* SIQI: This is the method calling the threshold update kernel for Agent_Forgetful
This function realizes the update of learning thresholds on GPU.
See update_thresholds_kernel for more details
Input: None
Output: None
*/
void Snapshot_Forgetful::update_thresholds() {
	Snapshot::update_thresholds();
}

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Snapshot_Forgetful::orient_all(){
	Snapshot::orient_all();
}

/*
------------------------SNAPSHOT_FORGETFUL----------------------------
*/

/*
------------------------SNAPSHOT_UNITTEST----------------------------
*/

void Snapshot_UnitTest::calculate_total(bool active){
	Snapshot::calculate_total(active);
}

void Snapshot_UnitTest::calculate_target(){
	Snapshot::calculate_target();
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
void Snapshot_UnitTest::update_weights(bool active){
	Snapshot::update_weights(active);
}

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Snapshot_UnitTest::orient_all(){
	Snapshot::orient_all();
}

/*
------------------------SNAPSHOT_UNITTEST----------------------------
*/

/*
The below code is unit test code for kernel.cu
*/

/*
-----------------------GPU TEST------------------------
*/

__global__ void unit_test_kernel_ind(int *row, int *col, int *result, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		result[index] = ind(row[index], col[index]);
	}
}
int GPUTest::TEST_ind_device(int row, int col){
	int *dev_row, *dev_col, *dev_result;
	int *Grow = new int[1];
	int *Gcol = new int[1];
	int *Gresult = new int[1];
	Grow[0] = row;
	Gcol[0] = col;

	cudaMalloc(&dev_row, sizeof(int));
	cudaMalloc(&dev_col, sizeof(int));
	cudaMalloc(&dev_result, sizeof(int));
	cudaMemcpy(dev_row, Grow, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_col, Gcol, sizeof(int), cudaMemcpyHostToDevice);
	
	unit_test_kernel_ind<<<1, 1>>>(dev_row, dev_col, dev_result, 1);
	cudaMemcpy(Gresult, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
	int tmp_result = Gresult[0];

	cudaFree(dev_row);
	cudaFree(dev_col);
	cudaFree(dev_result);
	delete[] Gresult;
	delete[] Grow;
	delete[] Gcol;

	return tmp_result;
}
//ind device test

__global__ void unit_test_kernel_compi(int *x, int *result, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		result[index] = compi(x[index]);
	}
}
int GPUTest::TEST_compi_device(int x){
	int *dev_x, *dev_result;
	int *Gx = new int[1];
	int *Gresult = new int[1];
	Gx[0] = x;

	cudaMalloc(&dev_x, sizeof(int));
	cudaMalloc(&dev_result, sizeof(int));
	cudaMemcpy(dev_x, Gx, sizeof(int), cudaMemcpyHostToDevice);
	
	unit_test_kernel_compi<<<1, 1>>>(dev_x, dev_result, 1);
	cudaMemcpy(Gresult, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
	int tmp_result = Gresult[0];

	cudaFree(dev_x);
	cudaFree(dev_result);
	delete[] Gresult;
	delete[] Gx;

	return tmp_result;
}
//compi device test

vector<bool> GPUTest::TEST_subtraction_kernel(vector<bool> b1, vector<bool> b2, int size){
	bool *dev_b1, *dev_b2;
	bool *Gb1 = new bool[size];
	bool *Gb2 = new bool[size];
	for(int i = 0; i < size; ++i){
		Gb1[i] = b1[i];
		Gb2[i] = b2[i];
	}

	cudaMalloc(&dev_b1, size * sizeof(bool));
	cudaMalloc(&dev_b2, size * sizeof(bool));
	cudaMemcpy(dev_b1, Gb1, sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b2, Gb2, sizeof(bool), cudaMemcpyHostToDevice);
	
	subtraction_kernel<<<(size + 255) / 256, 256>>>(dev_b1, dev_b2, size);
	cudaMemcpy(Gb1, dev_b1, size * sizeof(bool), cudaMemcpyDeviceToHost);
	
	vector<bool> results;
	for(int i = 0; i < size; ++i) results.push_back(Gb1[i]);

	cudaFree(dev_b1);
	cudaFree(dev_b2);
	delete[] Gb1;
	delete[] Gb2;

	return results;
}
//subtraction kernel test

__global__ void unit_test_kernel_implies_GPU(int row, int col, double *weights, double total, double threshold, bool *results, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		results[index] = implies_GPU(row, col, weights, total, threshold);
	}
}
bool GPUTest::TEST_implies_GPU(int row, int col, vector<double> weights, double total, double threshold){
	double *Gweights = new double[weights.size()];
	double *dev_weights;
	bool *Gresult, *dev_result;
	for(int i = 0; i < weights.size(); ++i) Gweights[i] = weights[i];

	Gresult = new bool[1];
	cudaMalloc(&dev_weights, weights.size() * sizeof(double));
	cudaMemcpy(dev_weights, Gweights, weights.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_result, sizeof(bool));
	unit_test_kernel_implies_GPU<<<1, 1>>>(row, col, dev_weights, total, threshold, dev_result, 1);
	cudaMemcpy(Gresult, dev_result, sizeof(bool), cudaMemcpyDeviceToHost);

	int result = Gresult[0];

	delete[] Gresult;
	delete[] Gweights;
	cudaFree(dev_weights);
	cudaFree(dev_result);

	return result;
}
//implies GPU test

/*
__global__ void unit_test_kernel_equivalent_GPU(int row, int col, double *weights, bool *results, int size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < size){
		results[index] = equivalent_GPU(row, col, weights);
	}
}

bool GPUTest::TEST_equivalent_GPU(int row, int col, vector<double> weights){
	double *Gweights = new double[weights.size()];
	double *dev_weights;
	bool *Gresult, *dev_result;
	for(int i = 0; i < weights.size(); ++i) Gweights[i] = weights[i];

	Gresult = new bool[1];
	cudaMalloc(&dev_weights, weights.size() * sizeof(double));
	cudaMemcpy(dev_weights, Gweights, weights.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc(&dev_result, sizeof(bool));
	unit_test_kernel_equivalent_GPU<<<1, 1>>>(row, col, dev_weights, dev_result, 1);
	cudaMemcpy(Gresult, dev_result, sizeof(bool), cudaMemcpyDeviceToHost);

	int result = Gresult[0];

	delete[] Gresult;
	delete[] Gweights;
	cudaFree(dev_weights);
	cudaFree(dev_result);

	return result;
}
//equivalent GPU test
*/

vector<bool> GPUTest::TEST_multiply_kernel(vector<bool> x, vector<bool> dir){
	bool *Gdir = new bool[dir.size()];
	bool *Gx = new bool[x.size()];
	bool *dev_dir, *dev_x;
	for(int i = 0; i < dir.size(); ++i) Gdir[i] = dir[i];
	for(int i = 0; i < x.size(); ++i) Gx[i] = x[i];
	cudaMalloc(&dev_dir, dir.size() * sizeof(bool));
	cudaMalloc(&dev_x, x.size() * sizeof(bool));
	cudaMemcpy(dev_dir, Gdir, dir.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_x, Gx, x.size() * sizeof(bool), cudaMemcpyHostToDevice);

	multiply_kernel<<<(x.size() + 255) / 256, 256>>>(dev_x, dev_dir, NULL, false, 0, x.size());

	cudaMemcpy(Gx, dev_x, x.size() * sizeof(bool), cudaMemcpyDeviceToHost);

	vector<bool> results;
	for(int i = 0; i < x.size(); ++i) results.push_back(Gx[i]);

	delete[] Gdir;
	delete[] Gx;
	cudaFree(dev_dir);
	cudaFree(dev_x);
	return results;
}

vector<bool> GPUTest::TEST_check_mask(vector<bool> mask){
	bool *Gmask = new bool[mask.size()];
	bool *dev_mask;
	cudaMalloc(&dev_mask, mask.size() * sizeof(bool));

	for(int i = 0; i < mask.size(); ++i) Gmask[i] = mask[i];
	cudaMemcpy(dev_mask, Gmask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);

	check_mask<<<(mask.size() / 2 + 255) / 256, 256>>>(dev_mask, mask.size() / 2);

	cudaMemcpy(Gmask, dev_mask, mask.size() * sizeof(bool), cudaMemcpyDeviceToHost);

	vector<bool> result;
	for(int i = 0; i < mask.size(); ++i) result.push_back(Gmask[i]);

	delete[] Gmask;
	cudaFree(dev_mask);

	return result;
}

vector<bool> GPUTest::TEST_mask_kernel(vector<bool> mask_amper, vector<bool> mask, vector<bool> current){
	bool *Gmask_amper = new bool[mask_amper.size()];
	bool *Gmask = new bool[mask.size()];
	bool *Gcurrent = new bool[current.size()];
	for(int i = 0; i < mask_amper.size(); ++i) Gmask_amper[i] = mask_amper[i];
	for(int i = 0; i < mask.size(); ++i) Gmask[i] = mask[i];
	for(int i = 0; i < current.size(); ++i) Gcurrent[i] = current[i];
	bool *dev_mask_amper, *dev_mask, *dev_current;

	cudaMalloc(&dev_mask_amper, mask_amper.size() * sizeof(bool));
	cudaMalloc(&dev_mask, mask.size() * sizeof(bool));
	cudaMalloc(&dev_current, current.size() * sizeof(bool));
	cudaMemcpy(dev_mask_amper, Gmask_amper, mask_amper.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mask, Gmask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_current, Gcurrent, current.size() * sizeof(bool), cudaMemcpyHostToDevice);

	dim3 dimGrid((current.size() / 2 + 15) / 16,(current.size() / 2 + 15) / 16);
	dim3 dimBlock(16, 16);
	mask_kernel<<<dimGrid, dimBlock>>>(dev_mask_amper, dev_mask, dev_current, current.size() / 2);

	cudaMemcpy(Gmask, dev_mask, mask.size() * sizeof(bool), cudaMemcpyDeviceToHost);

	vector<bool> result;
	for(int i = 0; i < mask.size(); ++i) result.push_back(Gmask[i]);

	delete[] Gmask;
	delete[] Gmask_amper;
	delete[] Gcurrent;
	cudaFree(dev_mask_amper);
	cudaFree(dev_mask);
	cudaFree(dev_current);

	return result;
}
/*
vector<double> GPUTest::TEST_update_weights_forgetful(vector<bool> signal, vector<double> weights, bool activity, double phi, double q, int sensor_size){
	bool *Gsignals, *dev_signals;
	double *Gweights, *dev_weights;
	Gsignals = new bool[signal.size()];
	Gweights = new double[weights.size()];

	for(int i = 0; i < signal.size(); ++i) Gsignals[i] = signal[i];
	for(int i = 0; i < weights.size(); ++i) Gweights[i] = weights[i];

	cudaMalloc(&dev_signals, signal.size() * sizeof(bool));
	cudaMalloc(&dev_weights, weights.size() * sizeof(double));

	cudaMemcpy(dev_signals, Gsignals, signal.size() * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights, Gweights, weights.size() * sizeof(double), cudaMemcpyHostToDevice);

	int measurable_size = 2 * sensor_size;
	dim3 dimGrid((measurable_size + 15) / 16, (measurable_size + 15) / 16);
	dim3 dimBlock(16, 16);
	update_weights_kernel_forgetful<<<dimGrid, dimBlock>>>(dev_weights, dev_signals, measurable_size, q, phi, activity);

	cudaMemcpy(Gweights, dev_weights, weights.size() * sizeof(double), cudaMemcpyDeviceToHost);

	vector<double> results;
	for(int i = 0; i < weights.size(); ++i) results.push_back(Gweights[i]);

	delete[] Gweights;
	delete[] Gsignals;
	cudaFree(dev_weights);
	cudaFree(dev_signals);

	return results;
}

vector<bool> GPUTest::TEST_orient_all(vector<double> weights, double q, double threshold, double total, int sensor_size){
	double *Gweights, *dev_weights;
	bool *Gdir, *dev_dir;
	int measurable_size = 2 * sensor_size;
	Gweights = new double[weights.size()];
	Gdir = new bool[weights.size()];
	
	for(int i = 0; i < weights.size(); ++i) Gweights[i] = weights[i];

	int array_size = measurable_size * (measurable_size + 1) / 2;
	int x = 0, y = 0;
	for(int i = 0; i < array_size; ++i){
		Gdir[i] = (x == y);
		x++;
		if(x > y){
			y++;
			x = 0;
		}
	}
	
	cudaMalloc(&dev_weights, weights.size() * sizeof(double));
	cudaMalloc(&dev_dir, weights.size() * sizeof(bool));
	cudaMemcpy(dev_weights, Gweights, weights.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dir, Gdir, array_size * sizeof(bool), cudaMemcpyHostToDevice);

	dim3 dimGrid((measurable_size / 2 + 15) / 16,(measurable_size / 2 + 15) / 16);
	dim3 dimBlock(16, 16);
	//threshold is hard coded for now
	orient_all_kernel<<<dimGrid, dimBlock>>>(dev_dir, dev_weights, NULL, total, q, measurable_size);

	cudaMemcpy(Gdir, dev_dir, weights.size() * sizeof(bool), cudaMemcpyDeviceToHost);

	vector<bool> results;

	for(int i = 0; i < weights.size(); ++i) results.push_back(Gdir[i]);

	delete[] Gweights;
	delete[] Gdir;
	cudaFree(dev_weights);
	cudaFree(dev_dir);

	return results;
}

/*
--------------------------GPU TEST------------------------
*/