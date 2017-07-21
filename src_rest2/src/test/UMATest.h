#ifndef _UMATEST_
#define _UMATEST_

#include <vector>
using namespace std;

class CPUTest{
public:
	CPUTest();
	~CPUTest();
	int TEST_ind_host(int row, int col);
	int TEST_compi_host(int x);
	/*
	vector<bool> TEST_up_GPU(vector<bool> signal, vector<bool> dir);
	vector<bool> TEST_gen_mask(vector<bool> mask_amper, vector<bool> current, int initial_size);
	vector<bool> TEST_set_signal(vector<bool> signal);
	vector<double> TEST_init_weight(int sensor_size);
	vector<bool> TEST_init_direction(int sensor_size);
	vector<bool> TEST_init_mask_amper(int sensor_size);
	vector<double> TEST_delay(vector<vector<double> > weights, vector<double> measurable, vector<double> measurable_old, double last_total, int measurable_id);
	*/
};

class GPUTest{
public:
	GPUTest();
	~GPUTest();
	int TEST_ind_device(int row, int col);
	int TEST_compi_device(int x);
	vector<bool> TEST_subtraction_kernel(vector<bool> b1, vector<bool> b2, int size);
	bool TEST_implies_GPU(int row, int col, vector<double> weights, double total, double threshold);
	//bool TEST_equivalent_GPU(int row, int col, vector<double> weights);
	vector<bool> TEST_multiply_kernel(vector<bool> x, vector<bool> dir);
	vector<bool> TEST_check_mask(vector<bool> mask);
	vector<bool> TEST_mask_kernel(vector<bool> mask_amper, vector<bool> mask, vector<bool> current);
	/*
	vector<double> TEST_update_weights_forgetful(vector<bool> signal, vector<double> weights, bool activity,
		double phi, double q, int sensor_size);
	vector<bool> TEST_orient_all(vector<double> weights, double q, double threshold, double total, int sensor_size);
	*/
};

#endif