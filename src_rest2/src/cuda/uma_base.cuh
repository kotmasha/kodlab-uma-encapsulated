#ifndef _UMA_BASE_
#define _UMA_BASE_

#include <cuda.h>
#include <cuda_runtime.h>

namespace uma_base {
	void init_mask(bool *mask, int initial_size, int measurable_size);
	void init_diag(double *diag, double *diag_, double total, double total_, int measurable_size_max);
	void update_weights(double *weights, bool *observe, int measurable_size, double q, double phi, bool active);
	void get_weights_diag(double *weights, double *diag, double *diag_, int measurable_size);
	void calculate_target(double *diag, bool *target, int sensor_size);
	void update_thresholds(bool *dirs, double *thresholds, double total_, double q, double phi, int &sensor_size);
	void orient_all(bool *dirs, double *weights, double *thresholds, double total, int sensor_size);
	void dfs(bool *signal, bool *dirs, double *thresholds, double q, int measurable_size);
	void floyd(bool *npdirs, int measurable_size);
	void dioid_square(int *dists, int sensor_size);
	void transpose_multiply(bool *npdirs, bool *signals, int measurable_size, int sig_count);
	void multiply(bool *npdirs, bool *signals, int measurable_size, int sig_count);
	void mask(bool *mask_amper, bool *mask, bool *current, int sensor_size);
	void check_mask(bool *mask, int sensor_size);
	void delta_weight_sum(double *diag, bool *d1, float *result, int measurable_size);
	void union_init(int *union_root, int sensor_size);
	void check_dist(int *dists, float delta, int sensor_size);
	void union_GPU(int *dists, int *union_root, int sensor_size);
	void copy_npdir(bool *npdir, bool *dir, int measurable_size);
	void negligible(bool *npdir, bool *negligible, int sensor_size);
}
#endif