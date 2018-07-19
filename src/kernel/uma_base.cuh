#ifndef _UMA_BASE_
#define _UMA_BASE_

namespace uma_base {
	void initMask(bool *mask, int initial_size, int attr_sensorSize);
	void initDiag(double *diag, double *diag_, double total, double total_, int attr_sensorSize_max);
	void updateWeights(double *weights, bool *observe, int attr_sensorSize, double q, double phi, bool active);
	void getWeightsDiag(double *weights, double *diag, double *diag_, int attr_sensorSize);
	void calculateTarget(double *diag, bool *target, int sensor_size);
	void updateThresholds(bool *dirs, double *thresholds, double total_, double q, double phi, int sensor_size);
	void orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size);
	void dfs(bool *signal, bool *dirs, double *thresholds, double q, int attr_sensorSize);
	void floyd(bool *npdirs, int attr_sensorSize);
	void dioidSquare(int *dists, int sensor_size);
	void transposeMultiply(bool *npdirs, bool *signals, int attr_sensorSize, int sig_count);
	void multiply(bool *npdirs, bool *signals, int attr_sensorSize, int sig_count);
	void mask(bool *mask_amper, bool *mask, bool *current, int sensor_size);
	void checkMask(bool *mask, int sensor_size);
	void deltaWeightSum(double *diag, bool *d1, float *result, int attr_sensorSize);
	void unionInit(int *union_root, int sensor_size);
	void checkDist(int *dists, float delta, int sensor_size);
	void unionGPU(int *dists, int *union_root, int sensor_size);
	void copyNpdir(bool *npdir, bool *dir, int attr_sensorSize);
	void negligible(bool *npdir, bool *negligible, int sensor_size);
}

namespace uma_base_qualitative {
	void updateWeights(double *weights, bool *observe, int attr_sensorSize, double q, double phi, bool active);
	void orientAll(bool *dirs, double *weights, double *thresholds, double total, int sensor_size);
	void calculateTarget(double *diag, bool *target, int sensor_size);
}
#endif