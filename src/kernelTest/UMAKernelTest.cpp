#include <iostream>
#include "gtest/gtest.h"
#include "UMAutil.h"
#include "UMAException.h"
#include "kernel.h"
#include "data_util.h"
#include "device_util.h"
#include "kernel_util.cuh"
#include "uma_base.cuh"

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
	for (int i = 0; i < 4; ++i) v1.pop_back();
	v2.clear();
	for (int i = 0; i < 6; ++i) v2.push_back(h_b2[i]);
	EXPECT_EQ(v1, v2);

	delete[] h_b1, h_b2, h_b3, h_b4;
	data_util::dev_free(dev_b1);
	data_util::dev_free(dev_b2);
}

TEST(memset, data_util_test) {
	//bool test
	bool *h_d, *dev_d;
	vector<bool> v1, v2(10, 0);
	h_d = new bool[10];
	data_util::dev_bool(dev_d, 10);
	data_util::dev_init(dev_d, 10);
	data_util::boolD2H(dev_d, h_d, 10);
	for (int i = 0; i < 10; ++i) v1.push_back(h_d[i]);
	EXPECT_EQ(v1, v2);

	//TODO
	//int test
	//double test
	//float test
}

TEST(double_cp, data_util_test) {
	//TODO
	EXPECT_EQ(1, 1);
}

TEST(int_cp, data_util_test) {
	//TODO
	EXPECT_EQ(1, 1);
}

TEST(float_cp, data_util_test) {
	//TODO
	EXPECT_EQ(1, 1);
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}