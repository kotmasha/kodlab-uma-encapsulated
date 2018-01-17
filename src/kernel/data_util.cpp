#include "kernel.h"
#include "data_util.h"
#include "Logger.h"

static Logger dataUtilLogger("DataUtil", "log/device.log");

void data_util::boolD2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(bool)) + " Bytes");
}

void data_util::boolH2D(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(bool)) + " Bytes");
}

void data_util::boolD2H(bool *from, bool *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(bool), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(bool)) + " Bytes");
}

void data_util::intD2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(int)) + " Bytes");
}

void data_util::intH2D(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(int)) + " Bytes");
}

void data_util::intD2H(int *from, int *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(int), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(int)) + " Bytes");
}

void data_util::doubleD2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(double)) + " Bytes");
}

void data_util::doubleH2D(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(double)) + " Bytes");
}

void data_util::doubleD2H(double *from, double *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(double), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(double)) + " Bytes");
}

void data_util::floatD2D(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyDeviceToDevice);
	dataUtilLogger.debug("Device to Device Copy, " + to_string(size * sizeof(float)) + " Bytes");
}

void data_util::floatH2D(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyHostToDevice);
	dataUtilLogger.debug("Host to Device Copy, " + to_string(size * sizeof(float)) + " Bytes");
}

void data_util::floatD2H(float *from, float *to, int size, int from_offset, int to_offset) {
	cudaMemcpy(to + to_offset, from + from_offset, size * sizeof(float), cudaMemcpyDeviceToHost);
	dataUtilLogger.debug("Device to Host Copy, " + to_string(size * sizeof(float)) + " Bytes");
}