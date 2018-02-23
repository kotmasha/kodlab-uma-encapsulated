#include <iostream>
#include "gtest/gtest.h"
#include "UMAutil.h"
#include "UMAException.h"

//test string to log level
TEST(string_to_log_level, string_to_log_level_test) {
	const string error = "ERROR";
	const string warn = "WARN";
	const string info = "INFO";
	const string debug = "DEBUG";
	const string verbose = "VERBOSE";
	EXPECT_EQ(StrUtil::string_to_log_level(error), 0);
	EXPECT_EQ(StrUtil::string_to_log_level(warn), 1);
	EXPECT_EQ(StrUtil::string_to_log_level(info), 2);
	EXPECT_EQ(StrUtil::string_to_log_level(debug), 3);
	EXPECT_EQ(StrUtil::string_to_log_level(verbose), 4);
}

TEST(string2d_to_string1d_pair, string2d_to_string1d_pair_test) {
	vector<vector<string>> inputs1, inputs2, inputs3;
	vector<pair<string, string>> outputs1, outputs2, outputs3;
	vector<string> v1 = { "test1", "test2" };
	vector<string> v2 = { "test2", "test3"};
	vector<string> v3 = { "test1", "test2", "test3" };
	pair<string, string> p1 = { "test1", "test2" };
	pair<string, string> p2 = { "test2", "test3" };
	pair<string, string> p3 = { "test2", "test4" };
	inputs1.push_back(v1); inputs1.push_back(v2);
	outputs1.push_back(p1); outputs1.push_back(p2);
	inputs2.push_back(v3);
	outputs2.push_back(p3);
	inputs3.push_back(v1); inputs3.push_back(v2);
	outputs3.push_back(p1); outputs3.push_back(p3);
	EXPECT_EQ(StrUtil::string2d_to_string1d_pair(inputs1), outputs1);
	EXPECT_THROW(StrUtil::string2d_to_string1d_pair(inputs2), UMAException);
	EXPECT_NE(StrUtil::string2d_to_string1d_pair(inputs3), outputs3);
}

TEST(bool_signal_to_int_idx, bool_signal_to_int_idx_test) {
	vector<bool> signal1 = { 1, 0, 0, 1, 0, 0, 1 };
	vector<bool> signal2 = { 0, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
	vector<bool> signal3 = { 0, 0, 1, 1 };
	vector<int> list1 = { 0, 3, 6 };
	vector<int> list2 = { 2, 3, 5, 6, 7 };
	vector<int> list3 = { 2 };
	EXPECT_EQ(SignalUtil::bool_signal_to_int_idx(signal1), list1);
	EXPECT_EQ(SignalUtil::bool_signal_to_int_idx(signal2), list2);
	EXPECT_NE(SignalUtil::bool_signal_to_int_idx(signal3), list3);
}

TEST(attr_sensor_signal_to_sensor_signal, attr_sensor_signal_to_sensor_signal_test) {
	vector<bool> signal1 = { 1, 0, 0, 1, 0, 0, 1, 1 };
	vector<bool> signal2 = { 0, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
	vector<bool> signal3 = { 0, 1, 0, 1, 1, 1 };
	vector<bool> list1 = { 1, 1, 0, 1 };
	vector<bool> list2 = { 0, 1, 1, 1, 0 };
	vector<bool> list3 = { 1, 1, 1, 1, 1 };
	EXPECT_EQ(SignalUtil::attr_sensor_signal_to_sensor_signal(signal1), list1);
	EXPECT_EQ(SignalUtil::attr_sensor_signal_to_sensor_signal(signal2), list2);
	EXPECT_NE(SignalUtil::attr_sensor_signal_to_sensor_signal(signal3), list3);
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}
