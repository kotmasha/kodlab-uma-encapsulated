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
TEST(bool_cp, data_util_test) {
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

TEST(memset, data_util_test) {
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

TEST(double_cp, data_util_test) {
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

TEST(int_cp, data_util_test) {
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

TEST(float_cp, data_util_test) {
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

TEST(compi_host, device_util_test) {
	vector<int> input = { 0, 3, 5, 8, 10 ,13 };
	vector<int> target = { 1, 2 ,4, 9, 11, 12 };
	vector<int> output;
	for (int i = 0; i < input.size(); ++i) output.push_back(compi(input[i]));
	EXPECT_EQ(output, target);
}

TEST(ind_host, device_util_test) {
	vector<int> input_y = { 0, 3, 5, 8, 11 ,13, 16, 18, 19 };
	vector<int> input_x = { 0, 2, 1, 8, 11, 14, 17, 24, 25 };
	vector<int> target = { 0, 8, 16, 44, 77, 132, 169, 344, 318 };
	vector<int> output;
	for (int i = 0; i < input_x.size(); ++i) output.push_back(ind(input_y[i], input_x[i]));
	EXPECT_EQ(output, target);
}

TEST(npdir_ind_host, device_util_test) {
	vector<int> input_y = { 0, 3, 5, 8, 11 ,13, 16, 18, 19 };
	vector<int> input_x = { 0, 2, 1, 8, 11, 14, 17, 24, 25 };
	vector<int> target = { 0, 10, 19, 48, 83, 140, 161, 357, 330 };
	vector<int> output;
	for (int i = 0; i < input_x.size(); ++i) output.push_back(npdir_ind(input_y[i], input_x[i]));
	EXPECT_EQ(output, target);
}

//--------------------------device_util test----------------------------------

//--------------------------kernel_util test----------------------------------

TEST(all_true, kernel_util_test) {
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

TEST(all_false, kernel_util_test) {
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

TEST(bool2int, kernel_util_tets) {
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

TEST(conjunction, kernel_util_test) {
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

TEST(disjunction, kernel_util_test) {
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

TEST(subtraction, kernel_util_test) {
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

TEST(negate_conjunction_star, kernel_util_test) {
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

TEST(conjunction_star, kernel_util_test) {
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

TEST(up2down, kernel_util_test) {
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

//--------------------------kernel_util test----------------------------------

//--------------------------uma_base test----------------------------------

TEST(init_mask, uma_base_test) {
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

TEST(init_diag, uma_base_test) {
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

TEST(update_weights, uma_base_test) {
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

TEST(get_weights_diag, uma_base_test) {
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

TEST(calculate_target, uma_base_test) {
	double *h_measurable, *dev_measurable;
	bool *h_target, *dev_target;

	h_measurable = new double[10];
	h_target = new bool[10];
	data_util::dev_double(dev_measurable, 10);
	data_util::dev_bool(dev_target, 10);

	h_measurable[0] = 1.02; h_measurable[2] = -2.21; h_measurable[4] = 10000000; h_measurable[6] = 0.0003; h_measurable[8] = 0.0;
	h_measurable[1] = 1.01; h_measurable[3] = -2.22; h_measurable[5] = -10000000; h_measurable[7] = 0.0004; h_measurable[9] = -0.01;
	vector<bool> v_target = { true, false, true, false, true, false, false, true, true, false };

	data_util::doubleH2D(h_measurable, dev_measurable, 10);
	uma_base::calculate_target(dev_measurable, dev_target, 5);
	data_util::boolD2H(dev_target, h_target, 10);

	for (int i = 0; i < 10; ++i) EXPECT_DOUBLE_EQ(h_target[i], v_target[i]);

	delete[] h_measurable, h_target;
	data_util::dev_free(dev_measurable);
	data_util::dev_free(dev_target);
}

TEST(update_thresholds, uma_base_test) {
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

TEST(orient_all, uma_base_test) {
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

TEST(floyd, uma_base_test) {
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
	uma_base::floyd(dev_npdirs, 55);
	data_util::boolD2H(dev_npdirs, h_npdirs, 60);
	for (int i = 0; i < 60; ++i) v2.push_back(h_npdirs[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_npdirs;
	data_util::dev_free(dev_npdirs);
}

TEST(dioid_square, uma_base_test) {
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

//--------------------------uma_base test----------------------------------

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}
