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
#include "Agent.h"
#include "UMATest.h"
#include "logging.h"

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
__device__ double lower_threshold(double q, double T){
	return q * T;
}

__device__ double raise_threshold(double q, double T){
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
	double epsilon = total * threshold;
	double m = min(epsilon,min(rc, min(r_c, r_c_)));
	return rc_ < m;
}

/*
This function does equivalent on GPU, using non-worker solution
Input: row, col info, measurable size, weight matrix and threshold
Output: bool value(mathematical info please inquiry Kotomasha)
*/
__device__ bool equivalent_GPU(int row, int col, double *weights){//equivalent
	double rc = weights[ind(row, col)];
	double r_c = weights[ind(compi(row), col)];
	double rc_ = weights[ind(row, compi(col))];
	double r_c_ = weights[ind(compi(row), compi(col))];
	return rc_ == 0 && r_c == 0 && rc > 0 && r_c_ > 0;
}

/*
This function does orient_square on GPU, using non-worker solution
Input: direction, weight, threshold matrix, xy location, and measurable size
Output: None
*/
__device__ void orient_square_GPU(bool *dir, double *weights, double *thresholds, double total, double q, int x, int y, int width){//orient_square
	double threshold = 1.0 / 125.0;
	//double threshold = 1.0;
	if(y >= x)
		dir[ind(y, x)] = implies_GPU(y, x, weights, total, threshold) || equivalent_GPU(y, x, weights);
	else
		dir[ind(compi(x), compi(y))] = implies_GPU(y, x, weights, total, threshold) || equivalent_GPU(y, x, weights);
	if(compi(y) >= x)
		dir[ind(compi(y), x)] = implies_GPU(compi(y), x, weights, total, threshold) || equivalent_GPU(compi(y), x, weights);
	else
		dir[ind(compi(x), y)] = implies_GPU(compi(y), x, weights, total, threshold) || equivalent_GPU(compi(y), x, weights);
	if(y >= compi(x))
		dir[ind(y, compi(x))] = implies_GPU(y, compi(x), weights, total, threshold) || equivalent_GPU(y, compi(x), weights);
	else
		dir[ind(x, compi(y))] = implies_GPU(y, compi(x), weights, total, threshold) || equivalent_GPU(y, compi(x), weights);
	if(compi(y) >= compi(x))
		dir[ind(compi(y), compi(x))] = implies_GPU(compi(y), compi(x), weights, total, threshold) || equivalent_GPU(compi(y), compi(x), weights);
	else
		dir[ind(x, y)] = implies_GPU(compi(y), compi(x), weights, total, threshold) || equivalent_GPU(compi(y), compi(x), weights);
	/*
	bool changed = false;
	//SIQI:if(!old_y_x && dir[ind(y, x)] && threshold >= 1 - q){
	if(dir[ind(y, x)] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_cy_x && dir[ind(compi(y), x)] && threshold >= 1 - q){
	if(!changed && dir[ind(compi(y), x)] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_y_cx && dir[ind(y, compi(x))] && threshold >= 1 - q){
	if(!changed && dir[ind(y, compi(x))] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = lower_threshold(q, threshold);
	}
	//SIQI:if(!changed && !old_cy_cx && dir[ind(compi(y), compi(x))] && threshold >= 1-q){
	if(!changed && dir[ind(compi(y), compi(x))] && threshold >= 1 - q){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = lower_threshold(q, threshold);
	}

	if(!changed && old_y_x && !dir[ind(y, x)]){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = raise_threshold(q, threshold);
	}
	if(!changed && old_cy_x && !dir[ind(compi(y), x)]){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = raise_threshold(q, threshold);
	}
	if(!changed && old_y_cx && !dir[ind(y, compi(x))]){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = raise_threshold(q, threshold);
	}
	if(!changed && old_cy_cx && !dir[ind(compi(y), compi(x))]){
		changed = true;
		thresholds[ind(y - y % 2, x - x % 2)] = raise_threshold(q, threshold);
	}
	*/
}

/*
---------------------DEVICE------------------------
*/

/*
---------------------GLOBAL---------------------
*/
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

__global__ void get_measurable_kernel(double *weights, double *measurable, double *measurable_old, int measurable_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < measurable_size){
		int idx = ind(index, index);
		measurable_old[index] = measurable[index];
		measurable[index] = weights[idx];
	}
}

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

/*
This function is orient all on GPU, using non-worker solution
Input: direction, weight, threshold matrix, and measurable size
Output: None
*/
__global__ void orient_all_kernel(bool *dir, double *weights, double *thresholds, double total, double q, int measurable_size){
	int indexX = blockDim.x * blockIdx.x + threadIdx.x;
	int indexY = blockDim.y * blockIdx.y + threadIdx.y;
	if(indexY < measurable_size / 2){
		orient_square_GPU(dir, weights, thresholds, total, q, indexX * 2, indexY * 2, measurable_size);
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
		j += 1024;
	}
	flag[0] = true;
	__syncthreads();
	while(flag[0]){
		flag[0] = false;
		__syncthreads();
		j = index;
		while(j < size){
			if(xs[j] == 1){
				j += 1024;
				continue;
			}
			for(int i = 0; i < size; ++i){
				if(dir[ind(i,j)] && xs[i] == 1 && (!is_stable || thresholds[ind(i - i % 2, j - j % 2)] < lowest)){
					ys[j] = 1;
					flag[0] = true;
					break;
				}
			}
			j += 1024;
		}
		__syncthreads();
		j = index;
		while(j < size){
			xs[j] = ys[j];
			j += 1024;
		}
		__syncthreads();
	}
	j = index;
	while(j < size){
		x[j] = ys[j];
		j += 1024;
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

__global__ void delta_weight_sum_kernel(double *measurable, bool *signal, float *result, int sensor_size){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < sensor_size){
		atomicAdd(result, signal[2 * index] * (measurable[2 * index] - measurable[2 * index + 1]) + signal[2 * index + 1] * (measurable[2 * index + 1] - measurable[2 * index]));
	}
}
/*
----------------------GLOBAL--------------------
*/


/*
---------------------AGENT---------------------
*/
/*
This function is an independent up function on GPU
It only use signal to do dfs, result is stored in Gsignal after using the function
Input: signal to be dfsed
Output: None
*/
void Agent::up_GPU(vector<bool> signal, bool is_stable){
	//logging_info->append_log(logging::UP, "[UP]:\n");
	//logging_info->add_indent();
	for(int i = 0; i < measurable_size; ++i) Gsignal[i] = signal[i];
	cudaMemcpy(dev_signal, Gsignal, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	//logging_info->append_log(logging::UP, "Separate Up Process is Invoked by Agent: "+name+"\n");
	//logging_info->append_log(logging::UP, "Size of Signal: "+to_string(whole_size)+"\n");
	
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_signal, dev_dir, dev_thresholds, is_stable, 1-q, measurable_size);

	cudaMemcpy(Gup, dev_signal, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	//cudaMemcpy(Gdir, dev_dir, array_size * sizeof(bool), cudaMemcpyDeviceToHost);
	//int result = 0;
	//for(int i = 0; i < measurable_size; ++i) result += Gup[i];
	//cout<<result<<endl;
	//cout<<result<<","<<measurable_size<<endl;
	//exit(0);
	
	up2down<<<(measurable_size + 255) / 256, 256>>>(dev_signal, dev_load, measurable_size);

	cudaMemcpy(Gdown, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaCheckErrors("kernel fails");
	//logging_info->append_process(logging::UP, logging::PROCESS);
	//logging_info->reduce_indent();
}

void Agent::gen_mask(){
	cudaMemset(dev_mask, false, 2 * this->base_sensor_size * sizeof(bool));
	cudaMemset(dev_mask + 2 * this->base_sensor_size, true, (measurable_size - 2 * this->base_sensor_size) * sizeof(bool));

	dim3 dimGrid((sensor_size + 15) / 16,(sensor_size + 15) / 16);
	dim3 dimBlock(16, 16);

	mask_kernel<<<dimGrid, dimBlock>>>(dev_mask_amper, dev_mask, dev_current, this->sensor_size);
	check_mask<<<(this->sensor_size + 255) / 256, 256>>>(dev_mask, this->sensor_size);
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
void Agent::propagate_GPU(vector<bool> signal, vector<bool> load, bool t){//propagate
	//logging_info->record_start();

	if(!signal.empty()){
		for(int i = 0; i < signal.size(); ++i){
			Gsignal[i] = signal[i];
			Gload[i] = load[i];
		}
		cudaMemcpy(dev_load, Gload, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_signal, Gsignal, measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	}

	//don't forget to add logging system
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_load, dev_dir, dev_thresholds, false, 0, measurable_size);
	multiply_kernel<<<1, 1024, 2 * measurable_size * sizeof(bool)>>>(dev_signal, dev_dir, dev_thresholds, false, 0, measurable_size);

	// standard operations
	disjunction_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, measurable_size);
	negate_conjunction_star_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_load, dev_signal, measurable_size);
	
	cudaMemcpy(Gload, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

double Agent::get_delta_weight_sum(vector<bool> signal){
	float result = 0;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemset(dev_result, 0, sizeof(float));
	
	for(int i = 0; i < signal.size(); ++i) Gsignal[i] = signal[i];

	delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, Gsignal, dev_result, sensor_size);

	cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);

	return result;
}

int Agent::distance(bool *signal1, bool *signal2){
	cudaMemcpy(dev_signal, signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load,false);
	cudaMemcpy(dev_signal1, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemcpy(dev_signal, signal2, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	propagate_GPU(tmp_signal, tmp_load, true);
	cudaMemcpy(dev_signal2, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	conjunction_star_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_signal1, dev_signal2, measurable_size);
	cudaMemcpy(Gsignal, dev_signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for(int i = 0; i < measurable_size; ++i) sum += Gsignal[i];
	
	return sum;
}

float Agent::distance_big(bool *signal1, bool *signal2){
	float result = 0;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemset(dev_result, 0, sizeof(float));

    cudaMemcpy(dev_signal, signal1, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	vector<bool> tmp_signal, tmp_load;
	propagate_GPU(tmp_signal, tmp_load,false);
	cudaMemcpy(dev_signal1, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemcpy(dev_signal, signal2, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, measurable_size * sizeof(bool));
	propagate_GPU(tmp_signal, tmp_load, true);
	cudaMemcpy(dev_signal2, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	subtraction_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_signal1, dev_signal2, measurable_size);

	/*
	This generates an all-ones weight vector
	*/
	double *allonesvec;
	cudaMalloc(&allonesvec, measurable_size * sizeof(double));
	cudaMemset(allonesvec,1.0,measurable_size * sizeof(double));

	/*
	Raw bit-count option:
	*/
	delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(allonesvec, dev_signal1, dev_result, sensor_size);

	/*
	Weighted bit-count option:
	/*
	//delta_weight_sum_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, dev_signal1, dev_result, sensor_size);

	cudaMemcpy(&result, dev_result, sizeof(double), cudaMemcpyDeviceToHost);
	/*
	To be removed if allones vector is not needed:
	*/
	cudaFree(allonesvec);

	cudaFree(dev_result);

	return result;
}

/*
----------------------------AGENT------------------------
*/

/*
---------------------------AGENT_STATIONARY---------------------
*/
void Agent_Stationary::calculate_total(bool activity){
	get_measurable_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_weights, dev_measurable, dev_measurable_old, measurable_size);
	//thrust::device_ptr<double> cptr = thrust::device_pointer_cast(dev_diagonal);
	//total = thrust::reduce(cptr, cptr + whole_size) / measurable_size;
	last_total = total;
	if(activity){
		total = q * total  +  (1-q) * phi;
	}
}

void Agent_Stationary::calculate_target(){
	calculate_target_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, dev_target, sensor_size);
	cudaMemcpy(Gtarget, dev_target, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
void Agent_Stationary::update_weights(bool activity){
	//logging_info->record_start();
	dim3 dimGrid2((measurable_size + 15) / 16, (measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_stationary<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, measurable_size, q, phi, activity);
	//logging_info->record_stop(logging::UPDATE_WEIGHT);
}

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Agent_Stationary::orient_all(){
	//logging_info->record_start();
	dim3 dimGrid1((measurable_size / 2 + 15) / 16,(measurable_size / 2 + 15) / 16);
	dim3 dimBlock1(16, 16);
	orient_all_kernel<<<dimGrid1, dimBlock1>>>(dev_dir, dev_weights, dev_thresholds, total, q, measurable_size);
}

/*
------------------------AGENT_STATIONARY----------------------------
*/


/*
---------------------------AGENT_FORGETFUL---------------------
*/
void Agent_Forgetful::calculate_total(bool activity){
	get_measurable_kernel<<<(measurable_size + 255) / 256, 256>>>(dev_weights, dev_measurable, dev_measurable_old, measurable_size);
	//thrust::device_ptr<double> cptr = thrust::device_pointer_cast(dev_diagonal);
	//total = thrust::reduce(cptr, cptr + whole_size) / measurable_size;
	last_total = total;
	total = q * total  +  (1-q) * phi * activity;
}

void Agent_Forgetful::calculate_target(){
	calculate_target_kernel<<<(sensor_size + 255) / 256, 256>>>(dev_measurable, dev_target, sensor_size);
	cudaMemcpy(Gtarget, dev_target, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function is update weights on GPU for discounted agent
It uses the update_weights_kernel_discounted, see that for detail
Input: None
Output: None
*/
void Agent_Forgetful::update_weights(bool activity){
	//logging_info->record_start();
	dim3 dimGrid2((measurable_size + 15) / 16, (measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	update_weights_kernel_forgetful<<<dimGrid2, dimBlock2>>>(dev_weights, dev_observe, measurable_size, q, phi, activity);
	//logging_info->record_stop(logging::UPDATE_WEIGHT);
}

/*
This function is orient all on GPU
See orient_all_kernel for more detail
Input: None
Output: None
*/
void Agent_Forgetful::orient_all(){
	//logging_info->record_start();
	dim3 dimGrid1((measurable_size / 2 + 15) / 16,(measurable_size / 2 + 15) / 16);
	dim3 dimBlock1(16, 16);
	orient_all_kernel<<<dimGrid1, dimBlock1>>>(dev_dir, dev_weights, dev_thresholds, total, q, measurable_size);
}

/*
------------------------AGENT_FORGETFUL----------------------------
*/






/*
The below code is unit test code for kernel.cu
*/





/*
-----------------CPU TEST----------------------
*/

int CPUTest::TEST_ind_host(int row, int col){
	return ind(row, col);
}
//ind host test

int CPUTest::TEST_compi_host(int x){
	return compi(x);
}
//compi host test

vector<bool> CPUTest::TEST_up_GPU(vector<bool> signal, vector<bool> dir){
	Agent *agent = new Agent(0, true);
	vector<string> names;
	agent->init_data("test", signal.size() / 2, names, "");
	agent->copy_dir(NULL, dir);

	agent->up_GPU(signal, false);

	vector<bool> results = agent->getUp();

	delete agent;

	return results;
}

vector<bool> CPUTest::TEST_gen_mask(vector<bool> mask_amper, vector<bool> current, int base_sensor_size){
	Agent *agent = new Agent(0, true);
	vector<string> names;
	agent->init_data("test", current.size() / 2, names, "");
	
	agent->copy_mask_amper(NULL, mask_amper);
	agent->copy_current(NULL, current);
	agent->reset_base_sensor_size(base_sensor_size, NULL);

	agent->gen_mask();

	vector<bool> result = agent->getMask();

	delete agent;

	return result;
}

vector<bool> CPUTest::TEST_set_signal(vector<bool> signal){
	Agent *agent = new Agent(0, true);
	vector<string> names;
	agent->init_data("test", signal.size() / 2, names, "");
	agent->setSignal(signal);
	
	vector<bool> result = agent->getObserve();

	delete agent;

	return result;
}

vector<double> CPUTest::TEST_init_weight(int sensor_size){
	Agent *agent = new Agent(0, true);
	vector<string> names;
	agent->init_data("test", sensor_size, names, "");

	vector<vector<double> > results = agent->getWeight();
	vector<double> result;

	for(int i = 0; i < results.size(); ++i){
		for(int j = 0; j < results[i].size(); ++j)
			result.push_back(results[i][j]);
	}

	delete agent;
	return result;
}

vector<bool> CPUTest::TEST_init_direction(int sensor_size){
	Agent *agent = new Agent(0, true);
	vector<string> names;
	agent->init_data("test", sensor_size, names, "");
	
	vector<vector<bool> > results = agent->getDir();
	vector<bool> result;

	for(int i = 0; i < results.size(); ++i){
		for(int j = 0; j < results[i].size(); ++j)
			result.push_back(results[i][j]);
	}

	delete agent;
	return result;
}

/*
-----------------------CPU TEST---------------------
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
	
	cudaMalloc(&dev_weights, weights.size() * sizeof(double));
	cudaMalloc(&dev_dir, weights.size() * sizeof(bool));
	cudaMemcpy(dev_weights, Gweights, weights.size() * sizeof(double), cudaMemcpyHostToDevice);

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