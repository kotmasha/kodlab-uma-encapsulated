#include "kernel.h"
#include "data_util.h"
#include "Logger.h"

static Logger dataUtilLogger("DataUtil", "log/device.log");

void data_util::boolD2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(bool)) + " Bytes");
	cudaCheckErrors("check boolD2D error");
}

void data_util::boolH2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(bool)) + " Bytes");
	cudaCheckErrors("check boolH2D error");
}

void data_util::boolD2H(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(bool)) + " Bytes");
	cudaCheckErrors("check boolD2H error");
}

void data_util::boolH2H(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyHostToHost);
	dataUtilLogger.debug("Host to Host Copy, " + to_string(size * sizeof(bool)) + " Bytes");
	cudaCheckErrors("check boolH2H error");
}

void data_util::intD2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(int)) + " Bytes");
	cudaCheckErrors("check intD2D error");
}

void data_util::intH2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(int)) + " Bytes");
	cudaCheckErrors("check intH2D error");
}

void data_util::intD2H(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(int)) + " Bytes");
	cudaCheckErrors("check intD2H error");
}

void data_util::intH2H(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyHostToHost);
	dataUtilLogger.debug("Host to Host Copy, " + to_string(size * sizeof(int)) + " Bytes");
	cudaCheckErrors("check intH2H error");
}

void data_util::doubleD2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(double)) + " Bytes");
	cudaCheckErrors("check doubleD2D error");
}

void data_util::doubleH2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(double)) + " Bytes");
	cudaCheckErrors("check doubleH2D error");
}

void data_util::doubleD2H(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(double)) + " Bytes");
	cudaCheckErrors("check doubleD2H error");
}

void data_util::doubleH2H(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyHostToHost);
	dataUtilLogger.debug("Host to Host Copy, " + to_string(size * sizeof(double)) + " Bytes");
	cudaCheckErrors("check doubleH2H error");
}

void data_util::floatD2D(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(float)) + " Bytes");
	cudaCheckErrors("check floatD2D error");
}

void data_util::floatH2D(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(float)) + " Bytes");
	cudaCheckErrors("check floatH2D error");
}

void data_util::floatD2H(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(float)) + " Bytes");
	cudaCheckErrors("check floatD2H error");
}

void data_util::floatH2H(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyHostToHost);
	dataUtilLogger.debug("Host to Host Copy, " + to_string(size * sizeof(float)) + " Bytes");
	cudaCheckErrors("check floatH2H error");
}

void data_util::dev_bool(bool *&dst, int size) {
	cudaMalloc(&dst, size * sizeof(bool));
	cudaCheckErrors("check dev_bool error");
}

void data_util::dev_int(int *&dst, int size) {
	cudaMalloc(&dst, size * sizeof(int));
	cudaCheckErrors("check dev_int error");
}

void data_util::dev_double(double *&dst, int size) {
	cudaMalloc(&dst, size * sizeof(double));
	cudaCheckErrors("check dev_double error");
}

void data_util::dev_float(float *&dst, int size) {
	cudaMalloc(&dst, size * sizeof(float));
	cudaCheckErrors("check dev_float error");
}

void data_util::dev_init(bool *ptr, int size) {
	cudaMemset(ptr, 0, size * sizeof(bool));
	cudaCheckErrors("check dev_init_bool error");
}

void data_util::dev_init(double *ptr, int size) {
	cudaMemset(ptr, 0, size * sizeof(double));
	cudaCheckErrors("check dev_init_double error");
}

void data_util::dev_init(int *ptr, int size) {
	cudaMemset(ptr, 0, size * sizeof(int));
	cudaCheckErrors("check dev_init_int error");
}

void data_util::dev_init(float *ptr, int size) {
	cudaMemset(ptr, 0, size * sizeof(float));
	cudaCheckErrors("check dev_init_float error");
}

void data_util::dev_free(bool *ptr) {
	cudaFree(ptr);
	cudaCheckErrors("check dev_free_bool error");
}

void data_util::dev_free(double *ptr) {
	cudaFree(ptr);
	cudaCheckErrors("check dev_free_double error");
}

void data_util::dev_free(int *ptr) {
	cudaFree(ptr);
	cudaCheckErrors("check dev_free_int error");
}

void data_util::dev_free(float *ptr) {
	cudaFree(ptr);
	cudaCheckErrors("check dev_free_float error");
}