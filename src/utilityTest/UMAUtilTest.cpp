#include <iostream>
#include "gtest/gtest.h"
#include "UMAutil.h"
#include "Logger.h"
#include "UMAException.h"
#include "PropertyMap.h"
#include "PropertyPage.h"

//test string to log level
TEST(Logger_test, stringToLogLevel) {
	const string error = "ERROR";
	const string warn = "WARN";
	const string info = "INFO";
	const string debug = "DEBUG";
	const string verbose = "VERBOSE";
	EXPECT_EQ(Logger::stringToLogLevel(error), 0);
	EXPECT_EQ(Logger::stringToLogLevel(warn), 1);
	EXPECT_EQ(Logger::stringToLogLevel(info), 2);
	EXPECT_EQ(Logger::stringToLogLevel(debug), 3);
	EXPECT_EQ(Logger::stringToLogLevel(verbose), 4);
}

TEST(Logger_test, logLevelGetSet) {
	Logger log = Logger("Server", "log/UMA_server.log");
	log.setLogLevel(Logger::LOG_VERBOSE);
	EXPECT_EQ(Logger::LOG_VERBOSE, log.getLogLevel());
}

TEST(PropertyMap_test, PropertyMap_test) {
	PropertyMap *ppm = new PropertyMap();
	EXPECT_TRUE(ppm->empty());

	ppm->add("key", "value");
	EXPECT_TRUE(ppm->exist("key"));
	EXPECT_FALSE(ppm->exist("none"));
	EXPECT_EQ(ppm->get("key"), "value");
	EXPECT_EQ(ppm->get("k1"), "");

	PropertyMap *other = new PropertyMap();
	other->add("k1", "v1");
	ppm->extend(other);
	EXPECT_TRUE(ppm->exist("k1"));
	EXPECT_EQ(ppm->get("k1"), "v1");

	EXPECT_FALSE(ppm->empty());
	ppm->remove("k1");
	EXPECT_NO_THROW(ppm->remove("k2"));
	ppm->remove("key");
	EXPECT_FALSE(ppm->exist("key"));
	EXPECT_TRUE(ppm->empty());

	ppm->add("key", "value");
	ppm->clear();
	EXPECT_TRUE(ppm->empty());

	delete ppm, other;
}

TEST(PropertyPage_test, PropertyPage_test) {
	PropertyMap *ppm = new PropertyMap();
	PropertyPage *pp = new PropertyPage();
	EXPECT_TRUE(pp->empty());

	pp->add("key", ppm);
	ppm->add("a", "b");
	EXPECT_TRUE(pp->exist("key"));
	EXPECT_FALSE(pp->exist("none"));
	EXPECT_EQ(pp->get("key"), ppm);
	EXPECT_EQ(pp->get("k1"), nullptr);

	PropertyMap *other = new PropertyMap();
	PropertyPage *otherPage = new PropertyPage();
	otherPage->add("k1", other);
	pp->extend(otherPage);
	EXPECT_TRUE(pp->exist("k1"));
	EXPECT_EQ(pp->get("k1"), other);

	EXPECT_FALSE(pp->empty());
	pp->remove("k1");
	EXPECT_NO_THROW(pp->remove("k2"));
	pp->remove("key");
	EXPECT_FALSE(pp->exist("key"));
	EXPECT_TRUE(pp->empty());

	delete pp, otherPage;
}

TEST(StrUtil_test, string2dToString1dPair) {
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
	EXPECT_EQ(StrUtil::string2dToString1dPair(inputs1), outputs1);
	EXPECT_THROW(StrUtil::string2dToString1dPair(inputs2), UMAException);
	EXPECT_NE(StrUtil::string2dToString1dPair(inputs3), outputs3);
}

TEST(StrUtil_test, isEmpty) {
	string s1 = "";
	string s2 = "abc";
	EXPECT_TRUE(StrUtil::isEmpty(s1));
	EXPECT_FALSE(StrUtil::isEmpty(s2));
}

TEST(SignalUtil_test, boolSignalToIntIdx) {
	vector<bool> signal1 = { 1, 0, 0, 1, 0, 0, 1 };
	vector<bool> signal2 = { 0, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
	vector<bool> signal3 = { 0, 0, 1, 1 };
	vector<int> list1 = { 0, 3, 6 };
	vector<int> list2 = { 2, 3, 5, 6, 7 };
	vector<int> list3 = { 2 };
	EXPECT_EQ(SignalUtil::boolSignalToIntIdx(signal1), list1);
	EXPECT_EQ(SignalUtil::boolSignalToIntIdx(signal2), list2);
	EXPECT_NE(SignalUtil::boolSignalToIntIdx(signal3), list3);
}

TEST(SignalUtil_test, attrSensorToSensorSignal) {
	vector<bool> signal1 = { 1, 0, 0, 1, 0, 0, 1, 1 };
	vector<bool> signal2 = { 0, 0, 1, 1, 0, 1, 1, 1, 0, 0 };
	vector<bool> signal3 = { 0, 1, 0, 1, 1, 1 };
	vector<bool> list1 = { 1, 1, 0, 1 };
	vector<bool> list2 = { 0, 1, 1, 1, 0 };
	vector<bool> list3 = { 1, 1, 1, 1, 1 };
	EXPECT_EQ(SignalUtil::attrSensorToSensorSignal(signal1), list1);
	EXPECT_EQ(SignalUtil::attrSensorToSensorSignal(signal2), list2);
	EXPECT_NE(SignalUtil::attrSensorToSensorSignal(signal3), list3);
}

TEST(SignalUtil_test, trimSignal) {
	vector<bool> signal1 = { 0, 1, 0, 1, 1, 0, 0 };
	vector<bool> signal2 = { 0, 1, 0 ,1, 1 };
	vector<bool> signal3 = { 1, 1, 0, 0, 1 };
	vector<bool> signal4 = { 1, 1, 0, 0, 1 };
	vector<bool> signal5 = { 1, 0, 1, 0, 0, 1, 0, 0, 0, 0 };
	vector<bool> signal6 = { 1, 0, 1, 0, 0, 1 };

	signal1 = SignalUtil::trimSignal(signal1);
	signal2 = SignalUtil::trimSignal(signal2);
	signal3 = SignalUtil::trimSignal(signal3);
	signal4 = SignalUtil::trimSignal(signal4);
	signal5 = SignalUtil::trimSignal(signal5);
	signal6 = SignalUtil::trimSignal(signal6);
	EXPECT_EQ(signal1, signal2);
	EXPECT_EQ(signal3, signal4);
	EXPECT_EQ(signal5, signal6);
}

TEST(ArrayUtil_test, findIdxInSortedArray) {
	vector<int> input = { 1, 3, 5, 6, 10, 11 };
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 0), -1);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 1), 0);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 2), 0);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 3), 1);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 4), 1);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 5), 2);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 6), 3);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 7), 3);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 8), 3);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 9), 3);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 10), 4);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 11), 5);
	EXPECT_EQ(ArrayUtil::findIdxInSortedArray(input, 12), 5);
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}
