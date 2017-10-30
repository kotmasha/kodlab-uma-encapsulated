#include "data_util.h"
#include <cuda.h>
#include <cuda_runtime.h>

void data_util::boolD2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToDevice);
}

void data_util::boolH2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyHostToDevice);
}

void data_util::boolD2H(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void data_util::intD2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToDevice);
}

void data_util::intH2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyHostToDevice);
}

void data_util::intD2H(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToHost);
}

void data_util::doubleD2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToDevice);
}

void data_util::doubleH2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyHostToDevice);
}

void data_util::doubleD2H(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToHost);
}