#include <iostream>
#include "gtest/gtest.h"
#include "UMAutil.h"
#include "UMAException.h"
#include "kernel.h"
#include "data_util.h"
#include "device_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"

//--------------------------data_util test----------------------------------
TEST(data_util_test, bool_cp) {
	bool *h_b1, *h_b2, *h_b3, *h_b4, *dev_b1, *dev_b2;
	vector<bool> v1, v2, v3, v4;
	h_b1 = new bool[10];
	h_b2 = new bool[10];
	h_b3 = new bool[10];
	h_b4 = new bool[10];
	data_util::dev_bool(dev_b1, 10);
	data_util::dev_bool(dev_b2, 10);
	
	for (int i = 0; i < 10; ++i) {
		h_b1[i] = i % 2;
		h_b3[i] = !(i % 2);
	}

	//host to device, then device to host
	for (int i = 0; i < 10; ++i) v1.push_back(h_b1[i]);
	data_util::boolH2D(h_b1, dev_b1, 10);
	data_util::boolD2H(dev_b1, h_b2, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_b2[i]);
	EXPECT_EQ(v1, v2);

	//device to device
	data_util::boolD2D(dev_b1, dev_b2, 10);
	data_util::boolD2H(dev_b2, h_b2, 10);
	v2.clear();
	for (int i = 0; i < 10; ++i) v2.push_back(h_b2[i]);
	EXPECT_EQ(v1, v2);

	//host to host
	data_util::boolH2H(h_b3, h_b4, 10);
	for (int i = 0; i < 10; ++i) v3.push_back(h_b3[i]);
	for (int i = 0; i < 10; ++i) v4.push_back(h_b4[i]);
	EXPECT_EQ(v3, v4);

	//host to device, then device to host with offset
	data_util::boolH2D(h_b1, dev_b1, 6, 2 ,4);
	data_util::boolD2H(dev_b1, h_b2, 6, 4, 0);
	v3.clear();
	for (int i = 2; i < 8; ++i) v3.push_back(v1[i]);
	v2.clear();
	for (int i = 0; i < 6; ++i) v2.push_back(h_b2[i]);
	EXPECT_EQ(v3, v2);

	delete[] h_b1, h_b2, h_b3, h_b4;
	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
}

TEST(data_util_test, memset) {
	//bool test
	bool *h_b, *dev_b;
	vector<bool> vb1, vb2(10, 0);
	h_b = new bool[10];
	data_util::dev_bool(dev_b, 10);
	data_util::dev_init(dev_b, 10);
	data_util::boolD2H(dev_b, h_b, 10);
	for (int i = 0; i < 10; ++i) vb1.push_back(h_b[i]);
	EXPECT_EQ(vb1, vb2);
	data_util::dev_free(dev_b);
	delete[] h_b;

	//int test
	int *h_i, *dev_i;
	vector<int> vi1, vi2(10, 0);
	h_i = new int[10];
	data_util::dev_int(dev_i, 10);
	data_util::dev_init(dev_i, 10);
	data_util::intD2H(dev_i, h_i, 10);
	for (int i = 0; i < 10; ++i) vi1.push_back(h_i[i]);
	EXPECT_EQ(vi1, vi2);
	data_util::dev_free(dev_i);
	delete[] h_i;

	//double test
	double *h_d, *dev_d;
	vector<double> vd1, vd2(10, 0.0);
	h_d = new double[10];
	data_util::dev_double(dev_d, 10);
	data_util::dev_init(dev_d, 10);
	data_util::doubleD2H(dev_d, h_d, 10);
	for (int i = 0; i < 10; ++i) vd1.push_back(h_d[i]);
	EXPECT_EQ(vd1, vd2);
	data_util::dev_free(dev_d);
	delete[] h_d;

	//float test
	float *h_f, *dev_f;
	vector<float> vf1, vf2(10, 0.0);
	h_f = new float[10];
	data_util::dev_float(dev_f, 10);
	data_util::dev_init(dev_f, 10);
	data_util::floatD2H(dev_f, h_f, 10);
	for (int i = 0; i < 10; ++i) vf1.push_back(h_f[i]);
	EXPECT_EQ(vf1, vf2);
	data_util::dev_free(dev_f);
	delete[] h_f;
}

TEST(data_util_test, double_cp) {
	double *h_d1, *h_d2, *h_d3, *h_d4, *dev_d1, *dev_d2;
	vector<double> v1, v2, v3, v4;
	h_d1 = new double[10];
	h_d2 = new double[10];
	h_d3 = new double[10];
	h_d4 = new double[10];
	data_util::dev_double(dev_d1, 10);
	data_util::dev_double(dev_d2, 10);

	for (int i = 0; i < 10; ++i) {
		h_d1[i] = 1.1 * i;
		h_d3[i] = 100 - 1.2 * i;
	}

	//host to device, then device to host
	for (int i = 0; i < 10; ++i) v1.push_back(h_d1[i]);
	data_util::doubleH2D(h_d1, dev_d1, 10);
	data_util::doubleD2H(dev_d1, h_d2, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_d2[i]);
	EXPECT_EQ(v1, v2);

	//device to device
	data_util::doubleD2D(dev_d1, dev_d2, 10);
	data_util::doubleD2H(dev_d2, h_d2, 10);
	v2.clear();
	for (int i = 0; i < 10; ++i) v2.push_back(h_d2[i]);
	EXPECT_EQ(v1, v2);

	//host to host
	data_util::doubleH2H(h_d3, h_d4, 10);
	for (int i = 0; i < 10; ++i) v3.push_back(h_d3[i]);
	for (int i = 0; i < 10; ++i) v4.push_back(h_d4[i]);
	EXPECT_EQ(v3, v4);

	//host to device, then device to host with offset
	data_util::doubleH2D(h_d1, dev_d1, 6, 2, 4);
	data_util::doubleD2H(dev_d1, h_d2, 6, 4, 0);
	v3.clear();
	for (int i = 2; i < 8; ++i) v3.push_back(v1[i]);
	v2.clear();
	for (int i = 0; i < 6; ++i) v2.push_back(h_d2[i]);
	EXPECT_EQ(v3, v2);

	delete[] h_d1, h_d2, h_d3, h_d4;
	data_util::dev_free(dev_d1);
	data_util::dev_free(dev_d2);
}

TEST(data_util_test, int_cp) {
	int *h_i1, *h_i2, *h_i3, *h_i4, *dev_i1, *dev_i2;
	vector<int> v1, v2, v3, v4;
	h_i1 = new int[10];
	h_i2 = new int[10];
	h_i3 = new int[10];
	h_i4 = new int[10];
	data_util::dev_int(dev_i1, 10);
	data_util::dev_int(dev_i2, 10);

	for (int i = 0; i < 10; ++i) {
		h_i1[i] = 2 * i;
		h_i3[i] = 100 - 2 * i;
	}

	//host to device, then device to host
	for (int i = 0; i < 10; ++i) v1.push_back(h_i1[i]);
	data_util::intH2D(h_i1, dev_i1, 10);
	data_util::intD2H(dev_i1, h_i2, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_i2[i]);
	EXPECT_EQ(v1, v2);

	//device to device
	data_util::intD2D(dev_i1, dev_i2, 10);
	data_util::intD2H(dev_i2, h_i2, 10);
	v2.clear();
	for (int i = 0; i < 10; ++i) v2.push_back(h_i2[i]);
	EXPECT_EQ(v1, v2);

	//host to host
	data_util::intH2H(h_i3, h_i4, 10);
	for (int i = 0; i < 10; ++i) v3.push_back(h_i3[i]);
	for (int i = 0; i < 10; ++i) v4.push_back(h_i4[i]);
	EXPECT_EQ(v3, v4);

	//host to device, then device to host with offset
	data_util::intH2D(h_i1, dev_i1, 6, 2, 4);
	data_util::intD2H(dev_i1, h_i2, 6, 4, 0);
	v3.clear();
	for (int i = 2; i < 8; ++i) v3.push_back(v1[i]);
	v2.clear();
	for (int i = 0; i < 6; ++i) v2.push_back(h_i2[i]);
	EXPECT_EQ(v3, v2);

	delete[] h_i1, h_i2, h_i3, h_i4;
	data_util::dev_free(dev_i1);
	data_util::dev_free(dev_i2);
}

TEST(data_util_test, float_cp) {
	float *h_f1, *h_f2, *h_f3, *h_f4, *dev_f1, *dev_f2;
	vector<float> v1, v2, v3, v4;
	h_f1 = new float[10];
	h_f2 = new float[10];
	h_f3 = new float[10];
	h_f4 = new float[10];
	data_util::dev_float(dev_f1, 10);
	data_util::dev_float(dev_f2, 10);

	for (int i = 0; i < 10; ++i) {
		h_f1[i] = 0.7 * i;
		h_f3[i] = 10 - 0.5 * i;
	}

	//host to device, then device to host
	for (int i = 0; i < 10; ++i) v1.push_back(h_f1[i]);
	data_util::floatH2D(h_f1, dev_f1, 10);
	data_util::floatD2H(dev_f1, h_f2, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_f2[i]);
	EXPECT_EQ(v1, v2);

	//device to device
	data_util::floatD2D(dev_f1, dev_f2, 10);
	data_util::floatD2H(dev_f2, h_f2, 10);
	v2.clear();
	for (int i = 0; i < 10; ++i) v2.push_back(h_f2[i]);
	EXPECT_EQ(v1, v2);

	//host to host
	data_util::floatH2H(h_f3, h_f4, 10);
	for (int i = 0; i < 10; ++i) v3.push_back(h_f3[i]);
	for (int i = 0; i < 10; ++i) v4.push_back(h_f4[i]);
	EXPECT_EQ(v3, v4);

	//host to device, then device to host with offset
	data_util::floatH2D(h_f1, dev_f1, 6, 2, 4);
	data_util::floatD2H(dev_f1, h_f2, 6, 4, 0);
	v3.clear();
	for (int i = 2; i < 8; ++i) v3.push_back(v1[i]);
	v2.clear();
	for (int i = 0; i < 6; ++i) v2.push_back(h_f2[i]);
	EXPECT_EQ(v3, v2);

	delete[] h_f1, h_f2, h_f3, h_f4;
	data_util::dev_free(dev_f1);
	data_util::dev_free(dev_f2);
}
//--------------------------data_util test----------------------------------

//--------------------------device_util test----------------------------------
extern int compi(int x);
extern int ind(int row, int col);
extern int npdir_ind(int row, int col);

TEST(device_util_test, compi_host) {
	vector<int> input = { 0, 3, 5, 8, 10 ,13 };
	vector<int> target = { 1, 2 ,4, 9, 11, 12 };
	vector<int> output;
	for (int i = 0; i < input.size(); ++i) output.push_back(compi(input[i]));
	EXPECT_EQ(output, target);
}

TEST(device_util_test, ind_host) {
	vector<int> input_y = { 0, 3, 5, 8, 11 ,13, 16, 18, 19 };
	vector<int> input_x = { 0, 2, 1, 8, 11, 14, 17, 24, 25 };
	vector<int> target = { 0, 8, 16, 44, 77, 132, 169, 344, 318 };
	vector<int> output;
	for (int i = 0; i < input_x.size(); ++i) output.push_back(ind(input_y[i], input_x[i]));
	EXPECT_EQ(output, target);
}

TEST(device_util_test, npdir_ind_host) {
	vector<int> input_y = { 0, 3, 5, 8, 11 ,13, 16, 18, 19 };
	vector<int> input_x = { 0, 2, 1, 8, 11, 14, 17, 24, 25 };
	vector<int> target = { 0, 10, 19, 48, 83, 140, 161, 357, 330 };
	vector<int> output;
	for (int i = 0; i < input_x.size(); ++i) output.push_back(npdir_ind(input_y[i], input_x[i]));
	EXPECT_EQ(output, target);
}

//--------------------------device_util test----------------------------------

//--------------------------kernel_util test----------------------------------

TEST(kernel_util_test, all_true) {
	bool *h_b, *dev_b;
	vector<bool> v1, v2;
	h_b = new bool[20];
	data_util::dev_bool(dev_b, 20);

	for (int i = 0; i < 20; ++i) v1.push_back(true);
	kernel_util::alltrue(dev_b, 20);
	data_util::boolD2H(dev_b, h_b, 20);
	for (int i = 0; i < 20; ++i) v2.push_back(h_b[i]);

	delete h_b;
	data_util::dev_free(dev_b);

	EXPECT_EQ(v1, v2);
}

TEST(kernel_util_test, all_false) {
	bool *h_b, *dev_b;
	vector<bool> v1, v2;
	h_b = new bool[20];
	data_util::dev_bool(dev_b, 20);

	for (int i = 0; i < 20; ++i) v1.push_back(false);
	kernel_util::allfalse(dev_b, 20);
	data_util::boolD2H(dev_b, h_b, 20);
	for (int i = 0; i < 20; ++i) v2.push_back(h_b[i]);

	delete h_b;
	data_util::dev_free(dev_b);

	EXPECT_EQ(v1, v2);
}

TEST(kernel_util_tets, bool2int) {
	bool *h_b, *dev_b;
	int *h_i, *dev_i;
	vector<int> v1 = {1, 0 ,0 ,1 ,1 ,0, 1, 0, 1, 0};
	vector<int> v2;
	h_b = new bool[10];
	h_i = new int[10];
	h_b[0] = true; h_b[1] = false; h_b[2] = false; h_b[3] = true; h_b[4] = true;
	h_b[5] = false; h_b[6] = true; h_b[7] = false; h_b[8] = true; h_b[9] = false;
	data_util::dev_bool(dev_b, 10);
	data_util::dev_int(dev_i, 10);

	data_util::boolH2D(h_b, dev_b, 10);
	kernel_util::bool2int(dev_b, dev_i, 10);
	data_util::intD2H(dev_i, h_i, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_i[i]);

	data_util::dev_free(dev_b);
	data_util::dev_free(dev_i);
	delete[] h_i, h_b;

	EXPECT_EQ(v1, v2);
}

TEST(kernel_util_tets, bool2double) {
	bool *h_b, *dev_b;
	double *h_d, *dev_d;
	vector<int> v1 = { 1, 0 ,0 ,1 ,1 ,0, 1, 0, 1, 0 };
	vector<int> v2;
	h_b = new bool[10];
	h_d = new double[10];
	h_b[0] = true; h_b[1] = false; h_b[2] = false; h_b[3] = true; h_b[4] = true;
	h_b[5] = false; h_b[6] = true; h_b[7] = false; h_b[8] = true; h_b[9] = false;
	data_util::dev_bool(dev_b, 10);
	data_util::dev_double(dev_d, 10);

	data_util::boolH2D(h_b, dev_b, 10);
	kernel_util::bool2double(dev_b, dev_d, 10);
	data_util::doubleD2H(dev_d, h_d, 10);
	for (int i = 0; i < 10; ++i) v2.push_back(h_d[i]);

	data_util::dev_free(dev_b);
	data_util::dev_free(dev_d);
	delete[] h_d, h_b;

	for(int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(v1[i], v2[i]);
}

TEST(kernel_util_test, conjunction) {
	bool *h_b1, *h_b2;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { false, false, true, false };
	vector<bool> v2;
	vector<bool> v3 = { true, false, true, false };
	vector<bool> v4;
	h_b1 = new bool[4];
	h_b2 = new bool[4];
	data_util::dev_bool(dev_b1, 4);
	data_util::dev_bool(dev_b2, 4);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true;
	h_b2[0] = true; h_b2[1] = false; h_b2[2] = true; h_b2[3] = false;

	data_util::boolH2D(h_b1, dev_b1, 4);
	data_util::boolH2D(h_b2, dev_b2, 4);
	kernel_util::conjunction(dev_b1, dev_b2, 4);
	data_util::boolD2H(dev_b1, h_b1, 4);
	data_util::boolD2H(dev_b2, h_b2, 4);

	for (int i = 0; i < 4; ++i) v2.push_back(h_b1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1, h_b2;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, disjunction) {
	bool *h_b1, *h_b2;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { true, false, true, true };
	vector<bool> v2;
	vector<bool> v3 = { true, false, true, false };
	vector<bool> v4;
	h_b1 = new bool[4];
	h_b2 = new bool[4];
	data_util::dev_bool(dev_b1, 4);
	data_util::dev_bool(dev_b2, 4);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true;
	h_b2[0] = true; h_b2[1] = false; h_b2[2] = true; h_b2[3] = false;

	data_util::boolH2D(h_b1, dev_b1, 4);
	data_util::boolH2D(h_b2, dev_b2, 4);
	kernel_util::disjunction(dev_b1, dev_b2, 4);
	data_util::boolD2H(dev_b1, h_b1, 4);
	data_util::boolD2H(dev_b2, h_b2, 4);

	for (int i = 0; i < 4; ++i) v2.push_back(h_b1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1, h_b2;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, subtraction) {
	bool *h_b1, *h_b2;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { false, false, false, true };
	vector<bool> v2;
	vector<bool> v3 = { true, false, true, false };
	vector<bool> v4;
	h_b1 = new bool[4];
	h_b2 = new bool[4];
	data_util::dev_bool(dev_b1, 4);
	data_util::dev_bool(dev_b2, 4);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true;
	h_b2[0] = true; h_b2[1] = false; h_b2[2] = true; h_b2[3] = false;

	data_util::boolH2D(h_b1, dev_b1, 4);
	data_util::boolH2D(h_b2, dev_b2, 4);
	kernel_util::subtraction(dev_b1, dev_b2, 4);
	data_util::boolD2H(dev_b1, h_b1, 4);
	data_util::boolD2H(dev_b2, h_b2, 4);

	for (int i = 0; i < 4; ++i) v2.push_back(h_b1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1, h_b2;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, negate_conjunction_star) {
	bool *h_b1, *h_b2;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { false, false, true, false };
	vector<bool> v2;
	vector<bool> v3 = { true, false, true, false };
	vector<bool> v4;
	h_b1 = new bool[4];
	h_b2 = new bool[4];
	data_util::dev_bool(dev_b1, 4);
	data_util::dev_bool(dev_b2, 4);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true;
	h_b2[0] = true; h_b2[1] = false; h_b2[2] = true; h_b2[3] = false;

	data_util::boolH2D(h_b1, dev_b1, 4);
	data_util::boolH2D(h_b2, dev_b2, 4);
	kernel_util::negate_conjunction_star(dev_b1, dev_b2, 4);
	data_util::boolD2H(dev_b1, h_b1, 4);
	data_util::boolD2H(dev_b2, h_b2, 4);

	for (int i = 0; i < 4; ++i) v2.push_back(h_b1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1, h_b2;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, conjunction_star) {
	bool *h_b1, *h_b2;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { false, false, false, true };
	vector<bool> v2;
	vector<bool> v3 = { true, false, true, false };
	vector<bool> v4;
	h_b1 = new bool[4];
	h_b2 = new bool[4];
	data_util::dev_bool(dev_b1, 4);
	data_util::dev_bool(dev_b2, 4);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true;
	h_b2[0] = true; h_b2[1] = false; h_b2[2] = true; h_b2[3] = false;

	data_util::boolH2D(h_b1, dev_b1, 4);
	data_util::boolH2D(h_b2, dev_b2, 4);
	kernel_util::conjunction_star(dev_b1, dev_b2, 4);
	data_util::boolD2H(dev_b1, h_b1, 4);
	data_util::boolD2H(dev_b2, h_b2, 4);

	for (int i = 0; i < 4; ++i) v2.push_back(h_b1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1, h_b2;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, up2down) {
	bool *h_b1;
	bool *dev_b1, *dev_b2;
	vector<bool> v1 = { true, true, false, false, false, true, true, false };
	vector<bool> v2;
	vector<bool> v3 = { false, false, true, true, false, true, true, false };
	vector<bool> v4;
	h_b1 = new bool[8];
	data_util::dev_bool(dev_b1, 8);
	data_util::dev_bool(dev_b2, 8);

	h_b1[0] = false; h_b1[1] = false; h_b1[2] = true; h_b1[3] = true; h_b1[4] = false; h_b1[5] = true, h_b1[6] = true; h_b1[7] = false;

	data_util::boolH2D(h_b1, dev_b1, 8);
	kernel_util::up2down(dev_b1, dev_b2, 8);
	data_util::boolD2H(dev_b2, h_b1, 8);

	for (int i = 0; i < 8; ++i) v2.push_back(h_b1[i]);
	data_util::boolD2H(dev_b1, h_b1, 8);
	for (int i = 0; i < 8; ++i) v4.push_back(h_b1[i]);

	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
	delete[] h_b1;
	EXPECT_EQ(v1, v2);
	EXPECT_EQ(v3, v4);
}

TEST(kernel_util_test, sum) {
	double *h_d = new double[20];
	double *dev_d;
	data_util::dev_double(dev_d, 20);

	h_d[0] = 1.0; h_d[1] = 3.0; h_d[2] = 5.0; h_d[3] = 7.0; h_d[4] = 9.0; h_d[5] = 11.0;
	h_d[6] = -1.0; h_d[7] = -3.0; h_d[8] = -5.0; h_d[9] = -7.0; h_d[10] = -9.0; h_d[11] = -11.0;
	h_d[12] = 1.0; h_d[13] = 3.0; h_d[14] = 5.0; h_d[15] = 7.0; h_d[16] = 9.0; h_d[17] = 11.0;

	data_util::doubleH2D(h_d, dev_d, 18);
	EXPECT_DOUBLE_EQ(kernel_util::sum(dev_d, 5), 25.0);
	data_util::doubleH2D(h_d, dev_d, 18);
	EXPECT_DOUBLE_EQ(kernel_util::sum(dev_d, 10), 20);
	data_util::doubleH2D(h_d, dev_d, 18);
	EXPECT_DOUBLE_EQ(kernel_util::sum(dev_d, 15), 9.0);

	delete[] h_d;
	data_util::dev_free(dev_d);
}

//--------------------------kernel_util test----------------------------------

//--------------------------uma_base test----------------------------------

TEST(uma_base_test, init_mask) {
	vector<bool> v1 = { false, false, false, false };
	vector<bool> v2 = { false, false, false, false, false, false, true, true, true, true };
	vector<bool> v3, v4;

	bool *h_b1, *h_b2;
	bool *d_b1, *d_b2;
	h_b1 = new bool[4];
	h_b2 = new bool[10];
	data_util::dev_bool(d_b1, 4);
	data_util::dev_bool(d_b2, 10);

	uma_base::init_mask(d_b1, 2, 4);
	uma_base::init_mask(d_b2, 3, 10);

	data_util::boolD2H(d_b1, h_b1, 4);
	data_util::boolD2H(d_b2, h_b2, 10);

	for (int i = 0; i < 4; ++i) v3.push_back(h_b1[i]);
	for (int i = 0; i < 10; ++i) v4.push_back(h_b2[i]);

	data_util::dev_free(d_b1);
	data_util::dev_free(d_b2);
	delete[] h_b1, h_b2;

	EXPECT_EQ(v1, v3);
	EXPECT_EQ(v2, v4);
}

TEST(uma_base_test, init_diag) {
	double *h_d1, *h_d2;
	double *dev_d1, *dev_d2;
	vector<double> v1 = {0.8, 0.8, 0.8, 0.8};
	vector<double> v2 = { 0.6, 0.6, 0.6, 0.6 };
	vector<double> v3, v4;

	h_d1 = new double[4];
	h_d2 = new double[4];
	data_util::dev_double(dev_d1, 4);
	data_util::dev_double(dev_d2, 4);

	uma_base::init_diag(dev_d1, dev_d2, 1.6, 1.2, 4);
	
	data_util::doubleD2H(dev_d1, h_d1, 4);
	data_util::doubleD2H(dev_d2, h_d2, 4);

	for (int i = 0; i < 4; ++i) v3.push_back(h_d1[i]);
	for (int i = 0; i < 4; ++i) v4.push_back(h_d2[i]);

	data_util::dev_free(dev_d1);
	data_util::dev_free(dev_d2);
	delete[] h_d1, h_d2;

	EXPECT_EQ(v1, v3);
	EXPECT_EQ(v2, v4);
}

TEST(uma_base_test, update_weights) {
	double *h_weights = new double[21];
	double *dev_weights;
	bool *h_observe = new bool[6];
	bool *dev_observe;
	vector<double> d0;

	data_util::dev_double(dev_weights, 21);
	data_util::dev_init(dev_weights, 21);
	data_util::dev_bool(dev_observe, 6);

	//1st round
	h_observe[0] = true; h_observe[1] = true; h_observe[2] = true; h_observe[3] = true; h_observe[4] = true; h_observe[5] = true;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0, 1, true);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d1(21, 1);
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d1[i]);
	d0.clear();

	//2nd round
	h_observe[0] = true; h_observe[1] = false; h_observe[2] = true; h_observe[3] = false; h_observe[4] = true; h_observe[5] = false;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0.5, 1.0, true);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d2 = {1, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d2[i]);
	d0.clear();

	//3nd round
	h_observe[0] = true; h_observe[1] = false; h_observe[2] = true; h_observe[3] = false; h_observe[4] = true; h_observe[5] = false;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0.5, 1.0, false);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d3 = { 1, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d3[i]);
	d0.clear();

	//4th round
	h_observe[0] = false; h_observe[1] = true; h_observe[2] = false; h_observe[3] = true; h_observe[4] = false; h_observe[5] = true;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0.8, 1.2, true);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d4 = { 0.8, 0.4, 0.64, 0.8, 0.4, 0.8, 0.4, 0.64, 0.4, 0.64, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.64, 0.4, 0.64, 0.4, 0.64 };
	//EXPECT_EQ(d0, d4);
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d4[i]);
	d0.clear();

	//5th round
	h_observe[0] = true; h_observe[1] = true; h_observe[2] = false; h_observe[3] = true; h_observe[4] = true; h_observe[5] = false;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0.7, 1, true);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d5 = { 0.86, 0.58, 0.748, 0.56, 0.28, 0.56, 0.58, 0.748, 0.28, 0.748, 0.86, 0.58, 0.56, 0.58, 0.86, 0.28, 0.448, 0.28, 0.448, 0.28, 0.448 };
	//EXPECT_EQ(d0, d4);
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d5[i]);
	d0.clear();

	//6th round
	h_observe[0] = true; h_observe[1] = false; h_observe[2] = false; h_observe[3] = false; h_observe[4] = false; h_observe[5] = true;
	data_util::boolH2D(h_observe, dev_observe, 6);
	uma_base::update_weights(dev_weights, dev_observe, 21, 0.9, 1, true);
	data_util::doubleD2H(dev_weights, h_weights, 21);
	for (int i = 0; i < 21; ++i) d0.push_back(h_weights[i]);
	vector<double> d6 = { 0.874, 0.522, 0.6732, 0.504, 0.252, 0.504, 0.522, 0.6732, 0.252, 0.6732, 0.774, 0.522, 0.504, 0.522, 0.774, 0.352, 0.4032, 0.252, 0.4032, 0.252, 0.5032 };
	for (int i = 0; i < 21; ++i) ASSERT_DOUBLE_EQ(d0[i], d6[i]);
	d0.clear();

	data_util::dev_free(dev_weights);
	data_util::dev_free(dev_observe);
	delete[] h_weights;
	delete[] h_observe;
}

TEST(uma_base_test, get_weights_diag) {
	double *h_diag, *h_diag_;
	double *dev_diag, *dev_diag_;
	double *h_weights;
	double *dev_weights;

	h_diag = new double[6];
	h_diag_ = new double[6];
	h_weights = new double[21];

	data_util::dev_double(dev_diag, 6);
	data_util::dev_double(dev_diag_, 6);
	data_util::dev_double(dev_weights, 21);
	data_util::dev_init(dev_diag, 6);
	data_util::dev_init(dev_diag_, 6);
	data_util::dev_init(dev_weights, 21);

	//1st round
	vector<double> w1(21, 1);
	vector<double> diag1(6, 1);
	vector<double> diag_1(6, 0);
	for (int i = 0; i < 21; ++i) h_weights[i] = w1[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag1[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_1[i], h_diag_[i]);

	//2nd round
	vector<double> w2 = { 1, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	vector<double> diag2 = {1, 0.5, 1, 0.5, 1, 0.5};
	vector<double> diag_2(6, 1);
	for (int i = 0; i < 21; ++i) h_weights[i] = w2[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag2[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_2[i], h_diag_[i]);

	//3rd round
	vector<double> w3 = { 1, 0.5, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 1, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
	vector<double> diag3 = { 1, 0.5, 1, 0.5, 1, 0.5 };
	vector<double> diag_3 = { 1, 0.5, 1, 0.5, 1, 0.5 };
	for (int i = 0; i < 21; ++i) h_weights[i] = w3[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag3[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_3[i], h_diag_[i]);

	//4th round
	vector<double> w4 = { 0.8, 0.4, 0.64, 0.8, 0.4, 0.8, 0.4, 0.64, 0.4, 0.64, 0.8, 0.4, 0.8, 0.4, 0.8, 0.4, 0.64, 0.4, 0.64, 0.4, 0.64 };
	vector<double> diag4 = { 0.8, 0.64, 0.8, 0.64, 0.8, 0.64 };
	vector<double> diag_4 = { 1, 0.5, 1, 0.5, 1, 0.5 };
	for (int i = 0; i < 21; ++i) h_weights[i] = w4[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag4[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_4[i], h_diag_[i]);

	//5th round
	vector<double> w5 = { 0.86, 0.58, 0.748, 0.56, 0.28, 0.56, 0.58, 0.748, 0.28, 0.748, 0.86, 0.58, 0.56, 0.58, 0.86, 0.28, 0.448, 0.28, 0.448, 0.28, 0.448 };
	vector<double> diag5 = { 0.86, 0.748, 0.56, 0.748, 0.86, 0.448 };
	vector<double> diag_5 = { 0.8, 0.64, 0.8, 0.64, 0.8, 0.64 };
	for (int i = 0; i < 21; ++i) h_weights[i] = w5[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag5[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_5[i], h_diag_[i]);

	//6th round
	vector<double> w6 = { 0.874, 0.522, 0.6732, 0.504, 0.252, 0.504, 0.522, 0.6732, 0.252, 0.6732, 0.774, 0.522, 0.504, 0.522, 0.774, 0.352, 0.4032, 0.252, 0.4032, 0.252, 0.5032 };
	vector<double> diag6 = { 0.874, 0.6732, 0.504, 0.6732, 0.774, 0.5032 };
	vector<double> diag_6 = { 0.86, 0.748, 0.56, 0.748, 0.86, 0.448 };
	for (int i = 0; i < 21; ++i) h_weights[i] = w6[i];
	data_util::doubleH2D(h_weights, dev_weights, 21);
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, 21);
	data_util::doubleD2H(dev_diag, h_diag, 6);
	data_util::doubleD2H(dev_diag_, h_diag_, 6);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag6[i], h_diag[i]);
	for (int i = 0; i < 6; ++i) ASSERT_DOUBLE_EQ(diag_6[i], h_diag_[i]);

	delete[] h_diag, h_diag_, h_weights;
	data_util::dev_free(dev_diag);
	data_util::dev_free(dev_diag_);
	data_util::dev_free(dev_weights);
}

TEST(uma_base_test, calculate_target) {
	double *h_attr_sensor, *dev_attr_sensor;
	bool *h_target, *dev_target;

	h_attr_sensor = new double[10];
	h_target = new bool[10];
	data_util::dev_double(dev_attr_sensor, 10);
	data_util::dev_bool(dev_target, 10);

	h_attr_sensor[0] = 1.02; h_attr_sensor[2] = -2.21; h_attr_sensor[4] = 10000000; h_attr_sensor[6] = 0.0003; h_attr_sensor[8] = 0.0;
	h_attr_sensor[1] = 1.01; h_attr_sensor[3] = -2.22; h_attr_sensor[5] = -10000000; h_attr_sensor[7] = 0.0004; h_attr_sensor[9] = -0.01;
	vector<bool> v_target = { true, false, true, false, true, false, false, true, true, false };

	data_util::doubleH2D(h_attr_sensor, dev_attr_sensor, 10);
	uma_base::calculate_target(dev_attr_sensor, dev_target, 5);
	data_util::boolD2H(dev_target, h_target, 10);

	for (int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(h_target[i], v_target[i]);

	delete[] h_attr_sensor, h_target;
	data_util::dev_free(dev_attr_sensor);
	data_util::dev_free(dev_target);
}

TEST(uma_base_test, update_thresholds) {
	bool *h_dirs, *dev_dirs;
	double *h_thresholds, *dev_thresholds;

	h_dirs = new bool[21];
	h_thresholds = new double[6];
	data_util::dev_bool(dev_dirs, 21);
	data_util::dev_double(dev_thresholds, 6);

	h_dirs[0] = true;
	h_dirs[1] = false; h_dirs[2] = true;
	h_dirs[3] = false; h_dirs[4] = false; h_dirs[5] = true;
	h_dirs[6] = false; h_dirs[7] = false; h_dirs[8] = false; h_dirs[9] = true;
	h_dirs[10] = false; h_dirs[11] = true; h_dirs[12] = false; h_dirs[13] = false; h_dirs[14] = true;
	h_dirs[15] = true; h_dirs[16] = false; h_dirs[17] = false; h_dirs[18] = true; h_dirs[19] = true; h_dirs[20] = true;

	//1st round
	h_thresholds[0] = 0.05; h_thresholds[1] = 1.0; h_thresholds[2] = 1.2;
	h_thresholds[3] = 0.9; h_thresholds[4] = 0.125; h_thresholds[5] = 0.25;

	data_util::boolH2D(h_dirs, dev_dirs, 21);
	data_util::doubleH2D(h_thresholds, dev_thresholds, 6);
	uma_base::update_thresholds(dev_dirs, dev_thresholds, 1.0, 0.9, 0.1, 3);
	data_util::doubleD2H(dev_thresholds, h_thresholds, 6);

	vector<double> v1 = {0.05, 1.0, 1.08, 0.81, 0.1125, 0.225};
	for (int i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(v1[i], h_thresholds[i]);
	//2nd round
	h_thresholds[0] = 0.05; h_thresholds[1] = 1.0; h_thresholds[2] = 1.2;
	h_thresholds[3] = 0.91; h_thresholds[4] = 0.125; h_thresholds[5] = 0.25;

	data_util::boolH2D(h_dirs, dev_dirs, 21);
	data_util::doubleH2D(h_thresholds, dev_thresholds, 6);
	uma_base::update_thresholds(dev_dirs, dev_thresholds, 1.0, 0.1, 0.1, 3);
	data_util::doubleD2H(dev_thresholds, h_thresholds, 6);

	vector<double> v2 = { 0.05, 1.0, 0.12, 0.091, 0.125, 0.25 };
	for (int i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(v2[i], h_thresholds[i]);

	delete[] h_dirs, h_thresholds;
	data_util::dev_free(dev_dirs);
	data_util::dev_free(dev_thresholds);
}

TEST(uma_base_test, orient_all) {
	double *h_weights, *dev_weights;
	bool *h_dirs, *dev_dirs;
	double *h_thresholds, *dev_thresholds;
	h_weights = new double[36];
	h_dirs = new bool[36];
	h_thresholds = new double[10];
	data_util::dev_double(dev_weights, 36);
	data_util::dev_bool(dev_dirs, 36);
	data_util::dev_double(dev_thresholds, 10);

	data_util::dev_init(dev_dirs, 36);

	h_weights[0] = 0.2;
	h_weights[1] = 0; h_weights[2] = 0.8;
	h_weights[3] = 0.2; h_weights[4] = 0.2; h_weights[5] = 0.4;
	h_weights[6] = 0; h_weights[7] = 0.6; h_weights[8] = 0; h_weights[9] = 0.6;
	h_weights[10] = 0.2; h_weights[11] = 0.4; h_weights[12] = 0.4; h_weights[13] = 0.2; h_weights[14] = 0.6;
	h_weights[15] = 0; h_weights[16] = 0.4; h_weights[17] = 0; h_weights[18] = 0.4; h_weights[19] = 0; h_weights[20] = 0.4;
	h_weights[21] = 0.2; h_weights[22] = 0.5; h_weights[23] = 0.4; h_weights[24] = 0.3; h_weights[25] = 0.6; h_weights[26] = 0.0; h_weights[27] = 0.8;
	h_weights[28] = 0.1; h_weights[29] = 0.2; h_weights[30] = 0.1; h_weights[31] = 0.2; h_weights[32] = 0; h_weights[33] = 0.4; h_weights[34] = 0; h_weights[35] = 0.2;
	data_util::doubleH2D(h_weights, dev_weights, 36);
	h_thresholds[0] = 0.1;
	h_thresholds[1] = 0.0; h_thresholds[2] = 0.2;
	h_thresholds[3] = 0.1; h_thresholds[4] = 0.1; h_thresholds[5] = 0.2;
	h_thresholds[6] = 0.2; h_thresholds[7] = 0.05; h_thresholds[8] = 0.1; h_thresholds[9] = 0.2;
	data_util::doubleH2D(h_thresholds, dev_thresholds, 10);

	uma_base::orient_all(dev_dirs, dev_weights, dev_thresholds, 1, 4);
	data_util::boolD2H(dev_dirs, h_dirs, 36);
	vector<bool> v1 = { false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
	false, true, false, true, false, false, false, false, false, false, true, false, false, false, true, false, false, false, true, false, false
	};
	vector<bool> v2;
	for (int i = 0; i < 36; ++i) v2.push_back(h_dirs[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_weights, h_dirs, h_thresholds;
	data_util::dev_free(dev_weights);
	data_util::dev_free(dev_dirs);
	data_util::dev_free(dev_thresholds);
}

TEST(uma_base_test, floyd) {
	bool *h_npdirs, *dev_npdirs;

	h_npdirs = new bool[60];

	data_util::dev_bool(dev_npdirs, 60);

	h_npdirs[0] = 1;
	h_npdirs[2] = 0; h_npdirs[3] = 1;
	h_npdirs[4] = 0; h_npdirs[5] = 0; h_npdirs[6] = 1;
	h_npdirs[8] = 0; h_npdirs[9] = 1; h_npdirs[10] = 0; h_npdirs[11] = 1;
	h_npdirs[12] = 0; h_npdirs[13] = 0; h_npdirs[14] = 0; h_npdirs[15] = 0; h_npdirs[16] = 1;
	h_npdirs[18] = 0; h_npdirs[19] = 0; h_npdirs[20] = 0; h_npdirs[21] = 1; h_npdirs[22] = 0; h_npdirs[23] = 1;
	h_npdirs[24] = 0; h_npdirs[25] = 0; h_npdirs[26] = 0; h_npdirs[27] = 0; h_npdirs[28] = 0; h_npdirs[29] = 0; h_npdirs[30] = 1;
	h_npdirs[32] = 0; h_npdirs[33] = 1; h_npdirs[34] = 0; h_npdirs[35] = 0; h_npdirs[36] = 0; h_npdirs[37] = 0; h_npdirs[38] = 0; h_npdirs[39] = 1;
	h_npdirs[40] = 0; h_npdirs[41] = 0; h_npdirs[42] = 0; h_npdirs[43] = 0; h_npdirs[44] = 1; h_npdirs[45] = 0; h_npdirs[46] = 1; h_npdirs[47] = 0; h_npdirs[48] = 1;
	h_npdirs[50] = 0; h_npdirs[51] = 0; h_npdirs[52] = 0; h_npdirs[53] = 1; h_npdirs[54] = 0; h_npdirs[55] = 0; h_npdirs[56] = 0; h_npdirs[57] = 0; h_npdirs[58] = 0; h_npdirs[59] = 1;

	vector<bool> v1 = { 
		1, 0,
		0, 1,
		0, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 0, 0, 1, 0,
		0, 1, 0, 1, 0, 1,
		0, 0, 0, 0, 0, 0, 1, 0,
		0, 1, 0, 1, 0, 0, 0, 1,
		0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 0, 0, 0, 0, 1 };
	vector<bool> v2;

	data_util::boolH2D(h_npdirs, dev_npdirs, 60);
	uma_base::floyd(dev_npdirs, 10);
	data_util::boolD2H(dev_npdirs, h_npdirs, 60);
	for (int i = 0; i < 60; ++i) v2.push_back(h_npdirs[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs;
	data_util::dev_free(dev_npdirs);
}

TEST(uma_base_test, dioid_square) {
	int *h_dists, *dev_dists;
	h_dists = new int[25];
	data_util::dev_int(dev_dists, 25);

	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 3; h_dists[3] = 4; h_dists[4] = 5;
	h_dists[5] = 1; h_dists[6] = 0; h_dists[7] = 3; h_dists[8] = 4; h_dists[9] = 5;
	h_dists[10] = 3; h_dists[11] = 3; h_dists[12] = 0; h_dists[13] = 2; h_dists[14] = 3;
	h_dists[15] = 4; h_dists[16] = 4; h_dists[17] = 2; h_dists[18] = 0; h_dists[19] = 4;
	h_dists[20] = 5; h_dists[21] = 5; h_dists[22] = 3; h_dists[23] = 4; h_dists[24] = 0;

	data_util::intH2D(h_dists, dev_dists, 25);
	uma_base::dioid_square(dev_dists, 5);
	data_util::intD2H(dev_dists, h_dists, 25);

	vector<int> d1 = {
	0, 1, 3, 3, 3,
	1, 0, 3, 3, 3,
	3, 3, 0, 2, 3,
	3, 3, 2, 0 ,3,
	3, 3, 3, 3, 0
	};
	vector<int> d2;
	for (int i = 0; i < 25; ++i) d2.push_back(h_dists[i]);
	EXPECT_EQ(d1, d2);

	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 2; h_dists[3] = 3;
	h_dists[4] = 1; h_dists[5] = 0; h_dists[6] = 3; h_dists[7] = 2;
	h_dists[8] = 2; h_dists[9] = 3; h_dists[10] = 0; h_dists[11] = 1;
	h_dists[12] = 3; h_dists[13] = 2; h_dists[14] = 1; h_dists[15] = 0;

	data_util::intH2D(h_dists, dev_dists, 16);
	uma_base::dioid_square(dev_dists, 4);
	data_util::intD2H(dev_dists, h_dists, 16);
	vector<int> d3 = {
		0, 1, 2, 2,
		1, 0, 2, 2,
		2, 2, 0, 1,
		2, 2, 1, 0
	};
	vector<int> d4;
	for (int i = 0; i < 16; ++i) d4.push_back(h_dists[i]);
	EXPECT_EQ(d3, d4);

	delete[] h_dists;
	data_util::dev_free(dev_dists);
}

TEST(uma_base_test, transpose_multiply) {
	bool *h_npdirs, *dev_npdirs;
	bool *h_signals, *dev_signals;
	h_npdirs = new bool[60];
	h_signals = new bool[100];
	data_util::dev_bool(dev_npdirs, 60);
	data_util::dev_bool(dev_signals, 100);

	h_npdirs[0] = 1; h_npdirs[1] = 0;
	h_npdirs[2] = 0; h_npdirs[3] = 1;
	h_npdirs[4] = 0; h_npdirs[5] = 0; h_npdirs[6] = 1; h_npdirs[7] = 0;
	h_npdirs[8] = 0; h_npdirs[9] = 1; h_npdirs[10] = 0; h_npdirs[11] = 1;
	h_npdirs[12] = 0; h_npdirs[13] = 0; h_npdirs[14] = 0; h_npdirs[15] = 0; h_npdirs[16] = 1; h_npdirs[17] = 0;
	h_npdirs[18] = 0; h_npdirs[19] = 1; h_npdirs[20] = 0; h_npdirs[21] = 1; h_npdirs[22] = 0; h_npdirs[23] = 1;
	h_npdirs[24] = 0; h_npdirs[25] = 0; h_npdirs[26] = 0; h_npdirs[27] = 0; h_npdirs[28] = 0; h_npdirs[29] = 0; h_npdirs[30] = 1; h_npdirs[31] = 0;
	h_npdirs[32] = 0; h_npdirs[33] = 1; h_npdirs[34] = 0; h_npdirs[35] = 1; h_npdirs[36] = 0; h_npdirs[37] = 0; h_npdirs[38] = 0; h_npdirs[39] = 1;
	h_npdirs[40] = 0; h_npdirs[41] = 0; h_npdirs[42] = 0; h_npdirs[43] = 0; h_npdirs[44] = 1; h_npdirs[45] = 0; h_npdirs[46] = 1; h_npdirs[47] = 0; h_npdirs[48] = 1; h_npdirs[49] = 0;
	h_npdirs[50] = 0; h_npdirs[51] = 1; h_npdirs[52] = 0; h_npdirs[53] = 1; h_npdirs[54] = 0; h_npdirs[55] = 0; h_npdirs[56] = 0; h_npdirs[57] = 0; h_npdirs[58] = 0; h_npdirs[59] = 1;
	data_util::boolH2D(h_npdirs, dev_npdirs, 60);

	h_signals[0] = 1; h_signals[1] = 0; h_signals[2] = 0; h_signals[3] = 0; h_signals[4] = 0; h_signals[5] = 0; h_signals[6] = 0; h_signals[7] = 0; h_signals[8] = 0; h_signals[9] = 0;
	h_signals[10] = 0; h_signals[11] = 0; h_signals[12] = 1; h_signals[13] = 0; h_signals[14] = 0; h_signals[15] = 0; h_signals[16] = 0; h_signals[17] = 0; h_signals[18] = 0; h_signals[19] = 0;
	h_signals[20] = 0; h_signals[21] = 0; h_signals[22] = 0; h_signals[23] = 0; h_signals[24] = 1; h_signals[25] = 0; h_signals[26] = 0; h_signals[27] = 0; h_signals[28] = 0; h_signals[29] = 0;
	h_signals[30] = 0; h_signals[31] = 0; h_signals[32] = 0; h_signals[33] = 0; h_signals[34] = 0; h_signals[35] = 0; h_signals[36] = 1; h_signals[37] = 0; h_signals[38] = 0; h_signals[39] = 0;	
	h_signals[40] = 0; h_signals[41] = 0; h_signals[42] = 0; h_signals[43] = 0; h_signals[44] = 0; h_signals[45] = 0; h_signals[46] = 0; h_signals[47] = 0; h_signals[48] = 1; h_signals[49] = 0;
	h_signals[50] = 1; h_signals[51] = 0; h_signals[52] = 0; h_signals[53] = 0; h_signals[54] = 0; h_signals[55] = 0; h_signals[56] = 0; h_signals[57] = 0; h_signals[58] = 1; h_signals[59] = 0;
	data_util::boolH2D(h_signals, dev_signals, 100);
	uma_base::transpose_multiply(dev_npdirs, dev_signals, 10, 6);
	data_util::boolD2H(dev_signals, h_signals, 60);

	vector<bool> v1 = {
		1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
		0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
		0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
	};
	vector<bool> v2;
	for (int i = 0; i < 60; ++i) v2.push_back(h_signals[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs, h_signals;
	data_util::dev_free(dev_npdirs);
	data_util::dev_free(dev_signals);
}

TEST(uma_base_test, multiply) {
	bool *h_npdirs, *dev_npdirs;
	bool *h_signals, *dev_signals;
	h_npdirs = new bool[60];
	h_signals = new bool[100];
	data_util::dev_bool(dev_npdirs, 60);
	data_util::dev_bool(dev_signals, 100);

	h_npdirs[0] = 1; h_npdirs[1] = 0;
	h_npdirs[2] = 0; h_npdirs[3] = 1;
	h_npdirs[4] = 0; h_npdirs[5] = 0; h_npdirs[6] = 1; h_npdirs[7] = 0;
	h_npdirs[8] = 0; h_npdirs[9] = 1; h_npdirs[10] = 0; h_npdirs[11] = 1;
	h_npdirs[12] = 0; h_npdirs[13] = 0; h_npdirs[14] = 0; h_npdirs[15] = 0; h_npdirs[16] = 1; h_npdirs[17] = 0;
	h_npdirs[18] = 0; h_npdirs[19] = 1; h_npdirs[20] = 0; h_npdirs[21] = 1; h_npdirs[22] = 0; h_npdirs[23] = 1;
	h_npdirs[24] = 0; h_npdirs[25] = 0; h_npdirs[26] = 0; h_npdirs[27] = 0; h_npdirs[28] = 0; h_npdirs[29] = 0; h_npdirs[30] = 1; h_npdirs[31] = 0;
	h_npdirs[32] = 0; h_npdirs[33] = 1; h_npdirs[34] = 0; h_npdirs[35] = 1; h_npdirs[36] = 0; h_npdirs[37] = 0; h_npdirs[38] = 0; h_npdirs[39] = 1;
	h_npdirs[40] = 0; h_npdirs[41] = 0; h_npdirs[42] = 0; h_npdirs[43] = 0; h_npdirs[44] = 1; h_npdirs[45] = 0; h_npdirs[46] = 1; h_npdirs[47] = 0; h_npdirs[48] = 1; h_npdirs[49] = 0;
	h_npdirs[50] = 0; h_npdirs[51] = 1; h_npdirs[52] = 0; h_npdirs[53] = 1; h_npdirs[54] = 0; h_npdirs[55] = 0; h_npdirs[56] = 0; h_npdirs[57] = 0; h_npdirs[58] = 0; h_npdirs[59] = 1;
	data_util::boolH2D(h_npdirs, dev_npdirs, 60);

	h_signals[0] = 1; h_signals[1] = 0; h_signals[2] = 0; h_signals[3] = 0; h_signals[4] = 0; h_signals[5] = 0; h_signals[6] = 0; h_signals[7] = 0; h_signals[8] = 0; h_signals[9] = 0;
	h_signals[10] = 0; h_signals[11] = 0; h_signals[12] = 1; h_signals[13] = 0; h_signals[14] = 0; h_signals[15] = 0; h_signals[16] = 0; h_signals[17] = 0; h_signals[18] = 0; h_signals[19] = 0;
	h_signals[20] = 0; h_signals[21] = 0; h_signals[22] = 0; h_signals[23] = 0; h_signals[24] = 1; h_signals[25] = 0; h_signals[26] = 0; h_signals[27] = 0; h_signals[28] = 0; h_signals[29] = 0;
	h_signals[30] = 0; h_signals[31] = 0; h_signals[32] = 0; h_signals[33] = 0; h_signals[34] = 0; h_signals[35] = 0; h_signals[36] = 1; h_signals[37] = 0; h_signals[38] = 0; h_signals[39] = 0;
	h_signals[40] = 0; h_signals[41] = 0; h_signals[42] = 0; h_signals[43] = 0; h_signals[44] = 0; h_signals[45] = 0; h_signals[46] = 0; h_signals[47] = 0; h_signals[48] = 1; h_signals[49] = 0;
	h_signals[50] = 1; h_signals[51] = 0; h_signals[52] = 0; h_signals[53] = 0; h_signals[54] = 0; h_signals[55] = 0; h_signals[56] = 0; h_signals[57] = 0; h_signals[58] = 1; h_signals[59] = 0;
	data_util::boolH2D(h_signals, dev_signals, 100);
	uma_base::multiply(dev_npdirs, dev_signals, 10, 6);
	data_util::boolD2H(dev_signals, h_signals, 60);

	vector<bool> v1 = {
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
		1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
		1, 0, 1, 0, 0, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
		1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
	};
	vector<bool> v2;
	for (int i = 0; i < 60; ++i) v2.push_back(h_signals[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs, h_signals;
	data_util::dev_free(dev_npdirs);
	data_util::dev_free(dev_signals);
}

TEST(uma_base_test, check_mask) {
	bool *h_mask, *dev_mask;
	h_mask = new bool[8];
	data_util::dev_bool(dev_mask, 8);

	h_mask[0] = 0; h_mask[1] = 0; h_mask[2] = 0; h_mask[3] = 1;
	h_mask[4] = 1; h_mask[5] = 0; h_mask[6] = 1; h_mask[7] = 1;

	data_util::boolH2D(h_mask, dev_mask, 8);
	uma_base::check_mask(dev_mask, 4);
	data_util::boolD2H(dev_mask, h_mask, 8);

	vector<bool> v1 = {0, 0, 0, 1, 1, 0, 1, 0};
	vector<bool> v2;
	for (int i = 0; i < 8; ++i) v2.push_back(h_mask[i]);
	EXPECT_EQ(v1, v2);
	
	delete[] h_mask;
	data_util::dev_free(dev_mask);
}

TEST(uma_base_test, mask) {
	bool *h_mask_amper, *dev_mask_amper;
	bool *h_current, *dev_current;
	bool *h_mask, *dev_mask;
	vector<bool> v;

	h_mask_amper = new bool[72];
	h_mask_amper[0] = 0; h_mask_amper[1] = 0;
	h_mask_amper[2] = 0; h_mask_amper[3] = 0; h_mask_amper[4] = 0; h_mask_amper[5] = 0;
	h_mask_amper[6] = 0; h_mask_amper[7] = 0; h_mask_amper[8] = 0; h_mask_amper[9] = 0; h_mask_amper[10] = 0; h_mask_amper[11] = 0;
	h_mask_amper[12] = 0; h_mask_amper[13] = 0; h_mask_amper[14] = 0; h_mask_amper[15] = 0; h_mask_amper[16] = 0; h_mask_amper[17] = 0; h_mask_amper[18] = 0; h_mask_amper[19] = 0;
	h_mask_amper[20] = 0; h_mask_amper[21] = 1; h_mask_amper[22] = 0; h_mask_amper[23] = 0; h_mask_amper[24] = 0; h_mask_amper[25] = 0; h_mask_amper[26] = 0; h_mask_amper[27] = 0; h_mask_amper[28] = 0; h_mask_amper[29] = 0;
	h_mask_amper[30] = 0; h_mask_amper[31] = 0; h_mask_amper[32] = 1; h_mask_amper[33] = 0; h_mask_amper[34] = 0; h_mask_amper[35] = 1; h_mask_amper[36] = 0; h_mask_amper[37] = 0; h_mask_amper[38] = 0; h_mask_amper[39] = 0; h_mask_amper[40] = 0; h_mask_amper[41] = 0;
	h_mask_amper[42] = 0; h_mask_amper[43] = 0; h_mask_amper[44] = 0; h_mask_amper[45] = 0; h_mask_amper[46] = 1; h_mask_amper[47] = 0; h_mask_amper[48] = 0; h_mask_amper[49] = 1; h_mask_amper[50] = 0; h_mask_amper[51] = 0; h_mask_amper[52] = 0; h_mask_amper[53] = 0; h_mask_amper[54] = 0; h_mask_amper[55] = 0;
	h_mask_amper[56] = 0; h_mask_amper[57] = 0; h_mask_amper[58] = 0; h_mask_amper[59] = 0; h_mask_amper[60] = 0; h_mask_amper[61] = 0; h_mask_amper[62] = 0; h_mask_amper[63] = 0; h_mask_amper[64] = 0; h_mask_amper[65] = 0; h_mask_amper[66] = 0; h_mask_amper[67] = 1; h_mask_amper[68] = 1; h_mask_amper[69] = 0; h_mask_amper[70] = 0; h_mask_amper[71] = 0;
	data_util::dev_bool(dev_mask_amper, 72);
	data_util::boolH2D(h_mask_amper, dev_mask_amper, 72);

	h_current = new bool[16];
	h_mask = new bool[16];
	data_util::dev_bool(dev_current, 16);
	data_util::dev_bool(dev_mask, 16);

	//1st round
	h_current[0] = 1; h_current[1] = 0; h_current[2] = 1; h_current[3] = 0; h_current[4] = 1; h_current[5] = 0; h_current[6] = 1; h_current[7] = 0;
	h_current[8] = 1; h_current[9] = 0; h_current[10] = 1; h_current[11] = 0; h_current[12] = 1; h_current[13] = 0; h_current[14] = 1; h_current[15] = 0;
	h_mask[0] = 0; h_mask[1] = 0; h_mask[2] = 0; h_mask[3] = 0; h_mask[4] = 0; h_mask[5] = 0; h_mask[6] = 0; h_mask[7] = 0;
	h_mask[8] = 1; h_mask[9] = 1; h_mask[10] = 1; h_mask[11] = 1; h_mask[12] = 1; h_mask[13] = 1; h_mask[14] = 1; h_mask[15] = 1;
	data_util::boolH2D(h_current, dev_current, 16);
	data_util::boolH2D(h_mask, dev_mask, 16);
	uma_base::mask(dev_mask_amper, dev_mask, dev_current, 8);
	data_util::boolD2H(dev_mask, h_mask, 16);
	vector<bool> v1 = { 0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1 };
	for (int i = 0; i < 16; ++i) v.push_back(h_mask[i]);
	EXPECT_EQ(v, v1);
	v.clear();

	//2nd round
	h_current[0] = 0; h_current[1] = 1; h_current[2] = 0; h_current[3] = 1; h_current[4] = 0; h_current[5] = 1; h_current[6] = 0; h_current[7] = 1;
	h_current[8] = 0; h_current[9] = 1; h_current[10] = 0; h_current[11] = 1; h_current[12] = 0; h_current[13] = 1; h_current[14] = 0; h_current[15] = 1;
	h_mask[0] = 0; h_mask[1] = 0; h_mask[2] = 0; h_mask[3] = 0; h_mask[4] = 0; h_mask[5] = 0; h_mask[6] = 0; h_mask[7] = 0;
	h_mask[8] = 1; h_mask[9] = 1; h_mask[10] = 1; h_mask[11] = 1; h_mask[12] = 1; h_mask[13] = 1; h_mask[14] = 1; h_mask[15] = 1;
	data_util::boolH2D(h_current, dev_current, 16);
	data_util::boolH2D(h_mask, dev_mask, 16);
	uma_base::mask(dev_mask_amper, dev_mask, dev_current, 8);
	data_util::boolD2H(dev_mask, h_mask, 16);
	vector<bool> v2 = { 0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1 };
	for (int i = 0; i < 16; ++i) v.push_back(h_mask[i]);
	EXPECT_EQ(v, v2);
	v.clear();

	//3rd round
	h_current[0] = 1; h_current[1] = 0; h_current[2] = 1; h_current[3] = 0; h_current[4] = 0; h_current[5] = 1; h_current[6] = 0; h_current[7] = 1;
	h_current[8] = 0; h_current[9] = 1; h_current[10] = 0; h_current[11] = 1; h_current[12] = 1; h_current[13] = 0; h_current[14] = 1; h_current[15] = 0;
	h_mask[0] = 0; h_mask[1] = 0; h_mask[2] = 0; h_mask[3] = 0; h_mask[4] = 0; h_mask[5] = 0; h_mask[6] = 0; h_mask[7] = 0;
	h_mask[8] = 1; h_mask[9] = 1; h_mask[10] = 1; h_mask[11] = 1; h_mask[12] = 1; h_mask[13] = 1; h_mask[14] = 1; h_mask[15] = 1;
	data_util::boolH2D(h_current, dev_current, 16);
	data_util::boolH2D(h_mask, dev_mask, 16);
	uma_base::mask(dev_mask_amper, dev_mask, dev_current, 8);
	data_util::boolD2H(dev_mask, h_mask, 16);
	vector<bool> v3 = { 0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1 };
	for (int i = 0; i < 16; ++i) v.push_back(h_mask[i]);
	EXPECT_EQ(v, v3);
	v.clear();

	//4th
	h_current[0] = 0; h_current[1] = 1; h_current[2] = 0; h_current[3] = 1; h_current[4] = 1; h_current[5] = 0; h_current[6] = 0; h_current[7] = 1;
	h_current[8] = 1; h_current[9] = 0; h_current[10] = 1; h_current[11] = 0; h_current[12] = 1; h_current[13] = 0; h_current[14] = 0; h_current[15] = 1;
	h_mask[0] = 0; h_mask[1] = 0; h_mask[2] = 0; h_mask[3] = 0; h_mask[4] = 0; h_mask[5] = 0; h_mask[6] = 0; h_mask[7] = 0;
	h_mask[8] = 1; h_mask[9] = 1; h_mask[10] = 1; h_mask[11] = 1; h_mask[12] = 1; h_mask[13] = 1; h_mask[14] = 1; h_mask[15] = 1;
	data_util::boolH2D(h_current, dev_current, 16);
	data_util::boolH2D(h_mask, dev_mask, 16);
	uma_base::mask(dev_mask_amper, dev_mask, dev_current, 8);
	data_util::boolD2H(dev_mask, h_mask, 16);
	vector<bool> v4 = { 0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,1 };
	for (int i = 0; i < 16; ++i) v.push_back(h_mask[i]);
	EXPECT_EQ(v, v4);
	v.clear();

	delete[] h_current, h_mask, h_mask_amper;
	data_util::dev_free(dev_current);
	data_util::dev_free(dev_mask);
	data_util::dev_free(dev_mask_amper);
}

TEST(uma_base_test, union_init) {
	int *h_union_root, *dev_union_root;
	h_union_root = new int[8];
	for (int i = 0; i < 8; ++i) h_union_root[i] = -1;

	data_util::dev_int(dev_union_root, 8);
	uma_base::union_init(dev_union_root, 8);
	data_util::intD2H(dev_union_root, h_union_root, 8);

	vector<int> v1 = { 0, 1, 2, 3, 4, 5, 6, 7 };
	vector<int> v2;
	for (int i = 0; i < 8; ++i) v2.push_back(h_union_root[i]);
	EXPECT_EQ(v1, v2);
	
	delete[] h_union_root;
	data_util::dev_free(dev_union_root);
}

TEST(uma_base_test, check_dist) {
	int *h_dists, *dev_dists;
	h_dists = new int[25];
	data_util::dev_int(dev_dists, 25);
	vector<int> v;

	//1st round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 3; h_dists[3] = 3; h_dists[4] = 3;
	h_dists[5] = 1; h_dists[6] = 0; h_dists[7] = 3; h_dists[8] = 3; h_dists[9] = 3;
	h_dists[10] = 3; h_dists[11] = 3; h_dists[12] = 0; h_dists[13] = 2; h_dists[14] = 3;
	h_dists[15] = 3; h_dists[16] = 3; h_dists[17] = 2; h_dists[18] = 0; h_dists[19] = 3;
	h_dists[20] = 3; h_dists[21] = 3; h_dists[22] = 3; h_dists[23] = 3; h_dists[24] = 0;
	data_util::intH2D(h_dists, dev_dists, 25);
	uma_base::check_dist(dev_dists, 0.7, 5);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v1 = {
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1
	};
	for (int i = 0; i < 25; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v1);
	v.clear();

	//2nd round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 3; h_dists[3] = 3; h_dists[4] = 3;
	h_dists[5] = 1; h_dists[6] = 0; h_dists[7] = 3; h_dists[8] = 3; h_dists[9] = 3;
	h_dists[10] = 3; h_dists[11] = 3; h_dists[12] = 0; h_dists[13] = 2; h_dists[14] = 3;
	h_dists[15] = 3; h_dists[16] = 3; h_dists[17] = 2; h_dists[18] = 0; h_dists[19] = 3;
	h_dists[20] = 3; h_dists[21] = 3; h_dists[22] = 3; h_dists[23] = 3; h_dists[24] = 0;
	data_util::intH2D(h_dists, dev_dists, 25);
	uma_base::check_dist(dev_dists, 1.5, 5);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v2 = {
		1, 1, 0, 0, 0,
		1, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1
	};
	for (int i = 0; i < 25; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v2);
	v.clear();

	//3nd round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 3; h_dists[3] = 3; h_dists[4] = 3;
	h_dists[5] = 1; h_dists[6] = 0; h_dists[7] = 3; h_dists[8] = 3; h_dists[9] = 3;
	h_dists[10] = 3; h_dists[11] = 3; h_dists[12] = 0; h_dists[13] = 2; h_dists[14] = 3;
	h_dists[15] = 3; h_dists[16] = 3; h_dists[17] = 2; h_dists[18] = 0; h_dists[19] = 3;
	h_dists[20] = 3; h_dists[21] = 3; h_dists[22] = 3; h_dists[23] = 3; h_dists[24] = 0;
	data_util::intH2D(h_dists, dev_dists, 25);
	uma_base::check_dist(dev_dists, 2.2, 5);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v3 = {
		1, 1, 0, 0, 0,
		1, 1, 0, 0, 0,
		0, 0, 1, 1, 0,
		0, 0, 1, 1, 0,
		0, 0, 0, 0, 1
	};
	for (int i = 0; i < 25; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v3);
	v.clear();

	//4th round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 3; h_dists[3] = 3; h_dists[4] = 3;
	h_dists[5] = 1; h_dists[6] = 0; h_dists[7] = 3; h_dists[8] = 3; h_dists[9] = 3;
	h_dists[10] = 3; h_dists[11] = 3; h_dists[12] = 0; h_dists[13] = 2; h_dists[14] = 3;
	h_dists[15] = 3; h_dists[16] = 3; h_dists[17] = 2; h_dists[18] = 0; h_dists[19] = 3;
	h_dists[20] = 3; h_dists[21] = 3; h_dists[22] = 3; h_dists[23] = 3; h_dists[24] = 0;
	data_util::intH2D(h_dists, dev_dists, 25);
	uma_base::check_dist(dev_dists, 3.1, 5);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v4 = {
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1
	};
	for (int i = 0; i < 25; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v4);
	v.clear();

	//5th round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 2; h_dists[3] = 2;
	h_dists[4] = 1; h_dists[5] = 0; h_dists[6] = 2; h_dists[7] = 2;
	h_dists[8] = 2; h_dists[9] = 2; h_dists[10] = 0; h_dists[11] = 1;
	h_dists[12] = 2; h_dists[13] = 2; h_dists[14] = 1; h_dists[15] = 0;
	data_util::intH2D(h_dists, dev_dists, 16);
	uma_base::check_dist(dev_dists, 0.5, 4);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v5 = {
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	};
	for (int i = 0; i < 16; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v5);
	v.clear();

	//6th round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 2; h_dists[3] = 2;
	h_dists[4] = 1; h_dists[5] = 0; h_dists[6] = 2; h_dists[7] = 2;
	h_dists[8] = 2; h_dists[9] = 2; h_dists[10] = 0; h_dists[11] = 1;
	h_dists[12] = 2; h_dists[13] = 2; h_dists[14] = 1; h_dists[15] = 0;
	data_util::intH2D(h_dists, dev_dists, 16);
	uma_base::check_dist(dev_dists, 1.5, 4);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v6 = {
		1, 1, 0, 0,
		1, 1, 0, 0,
		0, 0, 1, 1,
		0, 0, 1, 1,
	};
	for (int i = 0; i < 16; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v6);
	v.clear();

	//7th round
	h_dists[0] = 0; h_dists[1] = 1; h_dists[2] = 2; h_dists[3] = 2;
	h_dists[4] = 1; h_dists[5] = 0; h_dists[6] = 2; h_dists[7] = 2;
	h_dists[8] = 2; h_dists[9] = 2; h_dists[10] = 0; h_dists[11] = 1;
	h_dists[12] = 2; h_dists[13] = 2; h_dists[14] = 1; h_dists[15] = 0;
	data_util::intH2D(h_dists, dev_dists, 16);
	uma_base::check_dist(dev_dists, 2.01, 4);
	data_util::intD2H(dev_dists, h_dists, 25);
	vector<int> v7 = {
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
		1, 1, 1, 1,
	};
	for (int i = 0; i < 16; ++i) v.push_back(h_dists[i]);
	EXPECT_EQ(v, v7);
	v.clear();

	delete[] h_dists;
	data_util::dev_free(dev_dists);
}

TEST(uma_base_test, union_GPU) {
	int *h_dists, *dev_dists;
	int *h_union_root, *dev_union_root;
	h_dists = new int[25];
	h_union_root = new int[5];
	data_util::dev_int(dev_dists, 25);
	data_util::dev_int(dev_union_root, 5);
	vector<int> v;

	//1st round
	h_dists[0] = 1; h_dists[1] = 0; h_dists[2] = 0; h_dists[3] = 0; h_dists[4] = 0;
	h_dists[5] = 0; h_dists[6] = 1; h_dists[7] = 0; h_dists[8] = 0; h_dists[9] = 0;
	h_dists[10] = 0; h_dists[11] = 0; h_dists[12] = 1; h_dists[13] = 0; h_dists[14] = 0;
	h_dists[15] = 0; h_dists[16] = 0; h_dists[17] = 0; h_dists[18] = 1; h_dists[19] = 0;
	h_dists[20] = 0; h_dists[21] = 0; h_dists[22] = 0; h_dists[23] = 0; h_dists[24] = 1;
	for (int i = 0; i < 5; ++i) h_union_root[i] = i;
	data_util::intH2D(h_dists, dev_dists, 25);
	data_util::intH2D(h_union_root, dev_union_root, 5);
	uma_base::union_GPU(dev_dists, dev_union_root, 5);
	data_util::intD2H(dev_union_root, h_union_root, 5);
	vector<int> v1 = { 0, 1, 2, 3, 4 };
	for (int i = 0; i < 5; ++i) v.push_back(h_union_root[i]);
	EXPECT_EQ(v, v1);
	v.clear();

	//2nd round
	h_dists[0] = 1; h_dists[1] = 1; h_dists[2] = 0; h_dists[3] = 0; h_dists[4] = 0;
	h_dists[5] = 1; h_dists[6] = 1; h_dists[7] = 0; h_dists[8] = 0; h_dists[9] = 0;
	h_dists[10] = 0; h_dists[11] = 0; h_dists[12] = 1; h_dists[13] = 0; h_dists[14] = 0;
	h_dists[15] = 0; h_dists[16] = 0; h_dists[17] = 0; h_dists[18] = 1; h_dists[19] = 0;
	h_dists[20] = 0; h_dists[21] = 0; h_dists[22] = 0; h_dists[23] = 0; h_dists[24] = 1;
	for (int i = 0; i < 5; ++i) h_union_root[i] = i;
	data_util::intH2D(h_dists, dev_dists, 25);
	data_util::intH2D(h_union_root, dev_union_root, 5);
	uma_base::union_GPU(dev_dists, dev_union_root, 5);
	data_util::intD2H(dev_union_root, h_union_root, 5);
	vector<int> v2 = { 0, 0, 2, 3, 4 };
	for (int i = 0; i < 5; ++i) v.push_back(h_union_root[i]);
	EXPECT_EQ(v, v2);
	v.clear();

	//3rd round
	h_dists[0] = 1; h_dists[1] = 1; h_dists[2] = 0; h_dists[3] = 0; h_dists[4] = 0;
	h_dists[5] = 1; h_dists[6] = 1; h_dists[7] = 0; h_dists[8] = 0; h_dists[9] = 0;
	h_dists[10] = 0; h_dists[11] = 0; h_dists[12] = 1; h_dists[13] = 1; h_dists[14] = 0;
	h_dists[15] = 0; h_dists[16] = 0; h_dists[17] = 1; h_dists[18] = 1; h_dists[19] = 0;
	h_dists[20] = 0; h_dists[21] = 0; h_dists[22] = 0; h_dists[23] = 0; h_dists[24] = 1;
	for (int i = 0; i < 5; ++i) h_union_root[i] = i;
	data_util::intH2D(h_dists, dev_dists, 25);
	data_util::intH2D(h_union_root, dev_union_root, 5);
	uma_base::union_GPU(dev_dists, dev_union_root, 5);
	data_util::intD2H(dev_union_root, h_union_root, 5);
	vector<int> v3 = { 0, 0, 2, 2, 4 };
	for (int i = 0; i < 5; ++i) v.push_back(h_union_root[i]);
	EXPECT_EQ(v, v3);
	v.clear();

	//4th round
	h_dists[0] = 1; h_dists[1] = 1; h_dists[2] = 1; h_dists[3] = 1; h_dists[4] = 1;
	h_dists[5] = 1; h_dists[6] = 1; h_dists[7] = 1; h_dists[8] = 1; h_dists[9] = 1;
	h_dists[10] = 1; h_dists[11] = 1; h_dists[12] = 1; h_dists[13] = 1; h_dists[14] = 1;
	h_dists[15] = 1; h_dists[16] = 1; h_dists[17] = 1; h_dists[18] = 1; h_dists[19] = 1;
	h_dists[20] = 1; h_dists[21] = 1; h_dists[22] = 1; h_dists[23] = 1; h_dists[24] = 1;
	for (int i = 0; i < 5; ++i) h_union_root[i] = i;
	data_util::intH2D(h_dists, dev_dists, 25);
	data_util::intH2D(h_union_root, dev_union_root, 5);
	uma_base::union_GPU(dev_dists, dev_union_root, 5);
	data_util::intD2H(dev_union_root, h_union_root, 5);
	vector<int> v4 = { 0, 0, 0, 0, 0 };
	for (int i = 0; i < 5; ++i) v.push_back(h_union_root[i]);
	EXPECT_EQ(v, v4);
	v.clear();

	delete[] h_union_root, h_dists;
	data_util::dev_free(dev_union_root);
	data_util::dev_free(dev_dists);
}

TEST(uma_base_test, copy_npdir) {
	bool *h_npdirs, *dev_npdirs;
	bool *h_dirs, *dev_dirs;
	h_npdirs = new bool[60];
	h_dirs = new bool[55];
	data_util::dev_bool(dev_npdirs, 60);
	data_util::dev_bool(dev_dirs, 55);
	data_util::dev_init(dev_dirs, 55);
	data_util::dev_init(dev_npdirs, 60);

	h_dirs[0] = 1;
	h_dirs[1] = 0; h_dirs[2] = 1;
	h_dirs[3] = 0; h_dirs[4] = 0; h_dirs[5] = 1;
	h_dirs[6] = 0; h_dirs[7] = 1; h_dirs[8] = 0; h_dirs[9] = 1;
	h_dirs[10] = 0; h_dirs[11] = 0; h_dirs[12] = 0; h_dirs[13] = 0; h_dirs[14] = 1; 
	h_dirs[15] = 0; h_dirs[16] = 1; h_dirs[17] = 0; h_dirs[18] = 1; h_dirs[19] = 0; h_dirs[20] = 1;
	h_dirs[21] = 0; h_dirs[22] = 0; h_dirs[23] = 0; h_dirs[24] = 0; h_dirs[25] = 0; h_dirs[26] = 0; h_dirs[27] = 1; 
	h_dirs[28] = 0; h_dirs[29] = 1; h_dirs[30] = 0; h_dirs[31] = 1; h_dirs[32] = 0; h_dirs[33] = 0; h_dirs[34] = 0; h_dirs[35] = 1;
	h_dirs[36] = 0; h_dirs[37] = 0; h_dirs[38] = 0; h_dirs[39] = 0; h_dirs[40] = 1; h_dirs[41] = 0; h_dirs[42] = 1; h_dirs[43] = 0; h_dirs[44] = 1;
	h_dirs[45] = 0; h_dirs[46] = 1; h_dirs[47] = 0; h_dirs[48] = 1; h_dirs[49] = 0; h_dirs[50] = 0; h_dirs[51] = 0; h_dirs[52] = 0; h_dirs[53] = 0; h_dirs[54] = 1;
	data_util::boolH2D(h_dirs, dev_dirs, 55);

	uma_base::copy_npdir(dev_npdirs, dev_dirs, 55);

	h_npdirs[0] = 1; h_npdirs[1] = 0;
	h_npdirs[2] = 0; h_npdirs[3] = 1;
	h_npdirs[4] = 0; h_npdirs[5] = 0; h_npdirs[6] = 1; h_npdirs[7] = 0;
	h_npdirs[8] = 0; h_npdirs[9] = 1; h_npdirs[10] = 0; h_npdirs[11] = 1;
	h_npdirs[12] = 0; h_npdirs[13] = 0; h_npdirs[14] = 0; h_npdirs[15] = 0; h_npdirs[16] = 1; h_npdirs[17] = 0;
	h_npdirs[18] = 0; h_npdirs[19] = 1; h_npdirs[20] = 0; h_npdirs[21] = 1; h_npdirs[22] = 0; h_npdirs[23] = 1;
	h_npdirs[24] = 0; h_npdirs[25] = 0; h_npdirs[26] = 0; h_npdirs[27] = 0; h_npdirs[28] = 0; h_npdirs[29] = 0; h_npdirs[30] = 1; h_npdirs[31] = 0;
	h_npdirs[32] = 0; h_npdirs[33] = 1; h_npdirs[34] = 0; h_npdirs[35] = 1; h_npdirs[36] = 0; h_npdirs[37] = 0; h_npdirs[38] = 0; h_npdirs[39] = 1;
	h_npdirs[40] = 0; h_npdirs[41] = 0; h_npdirs[42] = 0; h_npdirs[43] = 0; h_npdirs[44] = 1; h_npdirs[45] = 0; h_npdirs[46] = 1; h_npdirs[47] = 0; h_npdirs[48] = 1; h_npdirs[49] = 0;
	h_npdirs[50] = 0; h_npdirs[51] = 1; h_npdirs[52] = 0; h_npdirs[53] = 1; h_npdirs[54] = 0; h_npdirs[55] = 0; h_npdirs[56] = 0; h_npdirs[57] = 0; h_npdirs[58] = 0; h_npdirs[59] = 1;
	
	vector<bool> v1, v2;
	for (int i = 0; i < 60; ++i) v1.push_back(h_npdirs[i]);
	data_util::boolD2H(dev_npdirs, h_npdirs, 60);
	for (int i = 0; i < 60; ++i) v2.push_back(h_npdirs[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs, h_dirs;
	data_util::dev_free(dev_npdirs);
	data_util::dev_free(dev_dirs);
}

TEST(uma_base_test, negligible) {
	bool *h_negligible, *dev_negligible;
	bool *h_npdirs, *dev_npdirs;
	h_npdirs = new bool[60];
	h_negligible = new bool[10];
	data_util::dev_bool(dev_npdirs, 60);
	data_util::dev_bool(dev_negligible, 10);

	h_npdirs[0] = 1; h_npdirs[1] = 1;
	h_npdirs[2] = 0; h_npdirs[3] = 1;
	h_npdirs[4] = 0; h_npdirs[5] = 0; h_npdirs[6] = 1; h_npdirs[7] = 0;
	h_npdirs[8] = 0; h_npdirs[9] = 1; h_npdirs[10] = 1; h_npdirs[11] = 1;
	h_npdirs[12] = 0; h_npdirs[13] = 0; h_npdirs[14] = 0; h_npdirs[15] = 0; h_npdirs[16] = 1; h_npdirs[17] = 0;
	h_npdirs[18] = 0; h_npdirs[19] = 1; h_npdirs[20] = 0; h_npdirs[21] = 1; h_npdirs[22] = 0; h_npdirs[23] = 1;
	h_npdirs[24] = 0; h_npdirs[25] = 0; h_npdirs[26] = 0; h_npdirs[27] = 0; h_npdirs[28] = 0; h_npdirs[29] = 0; h_npdirs[30] = 1; h_npdirs[31] = 1;
	h_npdirs[32] = 0; h_npdirs[33] = 1; h_npdirs[34] = 0; h_npdirs[35] = 1; h_npdirs[36] = 0; h_npdirs[37] = 0; h_npdirs[38] = 1; h_npdirs[39] = 1;
	h_npdirs[40] = 0; h_npdirs[41] = 0; h_npdirs[42] = 0; h_npdirs[43] = 0; h_npdirs[44] = 1; h_npdirs[45] = 0; h_npdirs[46] = 1; h_npdirs[47] = 0; h_npdirs[48] = 1; h_npdirs[49] = 0;
	h_npdirs[50] = 0; h_npdirs[51] = 1; h_npdirs[52] = 0; h_npdirs[53] = 1; h_npdirs[54] = 0; h_npdirs[55] = 0; h_npdirs[56] = 0; h_npdirs[57] = 0; h_npdirs[58] = 1; h_npdirs[59] = 1;
	data_util::boolH2D(h_npdirs, dev_npdirs, 60);
	uma_base::negligible(dev_npdirs, dev_negligible, 5);
	data_util::boolD2H(dev_negligible, h_negligible, 10);

	vector<bool> v1 = { 1, 0, 0, 1, 0, 0, 1, 1, 0, 1 };
	vector<bool> v2;
	for (int i = 0; i < 10; ++i) v2.push_back(h_negligible[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs, h_negligible;
	data_util::dev_free(dev_npdirs);
	data_util::dev_free(dev_negligible);
}

TEST(uma_base_test, delta_weight_sum) {
	double *h_attr_sensor, *dev_attr_sensor;
	bool *h_signal, *dev_signal;
	float *h_result, *dev_result;
	h_signal = new bool[16];
	h_attr_sensor = new double[16];
	h_result = new float;

	data_util::dev_bool(dev_signal, 16);
	data_util::dev_double(dev_attr_sensor, 16);
	data_util::dev_float(dev_result, 1);
	data_util::dev_init(dev_result, 1);

	h_attr_sensor[0] = 0; h_attr_sensor[1] = 3; h_attr_sensor[2] = 6; h_attr_sensor[3] = 2;
	h_attr_sensor[4] = 9; h_attr_sensor[5] = 7; h_attr_sensor[6] = 11; h_attr_sensor[7] = 19;
	h_attr_sensor[8] = 16; h_attr_sensor[9] = 3; h_attr_sensor[10] = 14; h_attr_sensor[11] = 22;
	h_attr_sensor[12] = 28; h_attr_sensor[13] = 6; h_attr_sensor[14] = 31; h_attr_sensor[15] = 8;
	data_util::doubleH2D(h_attr_sensor, dev_attr_sensor, 16);

	//1st round
	h_signal[0] = 1; h_signal[1] = 0; h_signal[2] = 1; h_signal[3] = 0;
	h_signal[4] = 0; h_signal[5] = 1; h_signal[6] = 0; h_signal[7] = 1;
	data_util::boolH2D(h_signal, dev_signal, 8);
	uma_base::delta_weight_sum(dev_attr_sensor, dev_signal, dev_result, 8);
	data_util::floatD2H(dev_result, h_result, 1);
	EXPECT_FLOAT_EQ(*h_result, 7.0);
	data_util::dev_init(dev_result, 1);

	//2st round
	h_signal[0] = 0; h_signal[1] = 1; h_signal[2] = 0; h_signal[3] = 0;
	h_signal[4] = 1; h_signal[5] = 0; h_signal[6] = 0; h_signal[7] = 1;
	h_signal[8] = 1; h_signal[9] = 0; h_signal[10] = 0; h_signal[11] = 0;
	data_util::boolH2D(h_signal, dev_signal, 12);
	uma_base::delta_weight_sum(dev_attr_sensor, dev_signal, dev_result, 12);
	data_util::floatD2H(dev_result, h_result, 1);
	EXPECT_FLOAT_EQ(*h_result, 26.0);
	data_util::dev_init(dev_result, 1);

	//3rd round
	h_signal[0] = 0; h_signal[1] = 1; h_signal[2] = 1; h_signal[3] = 0;
	h_signal[4] = 0; h_signal[5] = 0; h_signal[6] = 0; h_signal[7] = 1;
	h_signal[8] = 0; h_signal[9] = 0; h_signal[10] = 1; h_signal[11] = 0;
	h_signal[12] = 0; h_signal[13] = 1; h_signal[14] = 1; h_signal[15] = 0;
	data_util::boolH2D(h_signal, dev_signal, 16);
	uma_base::delta_weight_sum(dev_attr_sensor, dev_signal, dev_result, 16);
	data_util::floatD2H(dev_result, h_result, 1);
	EXPECT_FLOAT_EQ(*h_result, 8.0);
	data_util::dev_init(dev_result, 1);

	//4th round
	h_signal[0] = 1; h_signal[1] = 0; h_signal[2] = 0; h_signal[3] = 0;
	h_signal[4] = 0; h_signal[5] = 1; h_signal[6] = 0; h_signal[7] = 0;
	h_signal[8] = 1; h_signal[9] = 0; h_signal[10] = 1; h_signal[11] = 0;
	h_signal[12] = 0; h_signal[13] = 1; h_signal[14] = 0; h_signal[15] = 0;
	data_util::boolH2D(h_signal, dev_signal, 16);
	uma_base::delta_weight_sum(dev_attr_sensor, dev_signal, dev_result, 16);
	data_util::floatD2H(dev_result, h_result, 1);
	EXPECT_FLOAT_EQ(*h_result, -22.0);
	data_util::dev_init(dev_result, 1);

	delete[] h_attr_sensor, h_result, h_signal;
	data_util::dev_free(dev_attr_sensor);
	data_util::dev_free(dev_result);
	data_util::dev_free(dev_signal);
}

TEST(uma_base_test, new_episode) {
	bool *h_current = new bool[10];
	bool *dev_current;

	data_util::dev_bool(dev_current, 10);

	//1st
	h_current[0] = 0; h_current[1] = 0; h_current[2] = 1; h_current[3] = 1;
	h_current[4] = 1; h_current[5] = 1; h_current[6] = 1; h_current[7] = 1;
	h_current[8] = 0; h_current[9] = 0;
	data_util::boolH2D(h_current, dev_current, 10);
	uma_base::new_episode(dev_current, 3, 10);
	data_util::boolD2H(dev_current, h_current, 10);
	vector<bool> v1 = { 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 };
	vector<bool> v2;
	for (int i = 0; i < 10; ++i) v2.push_back(h_current[i]);
	EXPECT_EQ(v1, v2);

	//2nd
	h_current[0] = 0; h_current[1] = 0; h_current[2] = 1; h_current[3] = 1;
	h_current[4] = 1; h_current[5] = 1; h_current[6] = 1; h_current[7] = 1;
	h_current[8] = 0; h_current[9] = 0;
	data_util::boolH2D(h_current, dev_current, 10);
	uma_base::new_episode(dev_current, 4, 10);
	data_util::boolD2H(dev_current, h_current, 10);
	v1 = { 0, 0, 1, 1, 1, 1, 1, 1, 0, 0 };
	v2.clear();
	for (int i = 0; i < 10; ++i) v2.push_back(h_current[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_current;
	data_util::dev_free(dev_current);
}

//--------------------------uma_base test----------------------------------

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}
