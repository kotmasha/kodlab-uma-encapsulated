#include <iostream>
#include "gtest/gtest.h"
#include "RestUtil.h"
#include "UMAutil.h"
#include "UMAException.h"
#include "UMARestTestFixture.h"
#include <thread>

TEST(RestUtil_test, string2string_t) {
	string_t x = U("This is a simple test");
	string_t y = RestUtil::string2string_t("This is a simple test");
	EXPECT_EQ(x, y);
}

TEST(RestUtil_test, string_t2string) {
	string x = "This is a simple test";
	string y = RestUtil::string_t2string(U("This is a simple test"));
	EXPECT_EQ(x, y);
}

TEST(RestUtil_test, string_t2bool) {
	string_t t1 = U("True");
	string_t t2 = U("true");
	string_t t3 = U("test");
	string_t t4 = U("1");
	string_t t5 = U("0");

	EXPECT_EQ(true, RestUtil::string_t2bool(t1));
	EXPECT_EQ(true, RestUtil::string_t2bool(t2));
	EXPECT_EQ(true, RestUtil::string_t2bool(t3));
	EXPECT_EQ(true, RestUtil::string_t2bool(t4));
	EXPECT_EQ(false , RestUtil::string_t2bool(t5));
}

TEST(RestUtil_test, error_type_to_status_code) {
	//No test now
}

TEST(RestUtil_test, status_code2string) {
	EXPECT_EQ(RestUtil::status_code2string(status_codes::Created), "201");
	EXPECT_EQ(RestUtil::status_code2string(status_codes::OK), "200");
	EXPECT_EQ(RestUtil::status_code2string(status_codes::BadRequest), "400");
	EXPECT_EQ(RestUtil::status_code2string(status_codes::NotFound), "404");
	EXPECT_EQ(RestUtil::status_code2string(status_codes::Conflict), "409");
}

TEST(RestUtil_test, checkField_data) {
	json::value d;
	d[U("a")] = json::value(1);
	d[U("b")] = json::value("test");
	string_t A = U("a");
	string_t B = U("b");

	EXPECT_EQ(RestUtil::checkField(d, U("a"), false), true);
	EXPECT_EQ(RestUtil::checkField(d, U("b"), false), true);
	EXPECT_EQ(RestUtil::checkField(d, U("a"), true), true);
	EXPECT_EQ(RestUtil::checkField(d, U("b"), true), true);
	EXPECT_EQ(RestUtil::checkField(d, U("c"), false), false);
	EXPECT_THROW(RestUtil::checkField(d, U("c"), true), UMAException);
}

TEST(RestUtil_test, checkField_query) {
	map<string_t, string_t> m;
	m[U("a")] = U("1");
	m[U("b")] = U("test");
	string_t A = U("a");
	string_t B = U("b");

	EXPECT_EQ(RestUtil::checkField(m, U("a"), false), true);
	EXPECT_EQ(RestUtil::checkField(m, U("b"), false), true);
	EXPECT_EQ(RestUtil::checkField(m, U("a"), true), true);
	EXPECT_EQ(RestUtil::checkField(m, U("b"), true), true);
	EXPECT_EQ(RestUtil::checkField(m, U("c"), false), false);
	EXPECT_THROW(RestUtil::checkField(m, U("c"), true), UMAException);
}

TEST(RestUtil_test, vectorInt2Json) {
	vector<int> input = { 0, 1, 2, 3, 4, 5 };
	json::value output = json::value::array();
	output[0] = json::value::number(0);
	output[1] = json::value::number(1);
	output[2] = json::value::number(2);
	output[3] = json::value::number(3);
	output[4] = json::value::number(4);
	output[5] = json::value::number(5);
	EXPECT_EQ(output, RestUtil::vectorInt2Json(input));
}

TEST(RestUtil_test, vectorDouble2Json) {
	vector<double> input = { .0, .1, .2, .3, .4, .5 };
	json::value output = json::value::array();
	output[0] = json::value::number(.0);
	output[1] = json::value::number(.1);
	output[2] = json::value::number(.2);
	output[3] = json::value::number(.3);
	output[4] = json::value::number(.4);
	output[5] = json::value::number(.5);
	EXPECT_EQ(output, RestUtil::vectorDouble2Json(input));
}

TEST(RestUtil_test, vectorBool2Json) {
	vector<bool> input = { 0, 1, 0, 1, 0, 1 };
	json::value output = json::value::array();
	//!!!!! DO NOT EVER ASSIGN BASIC TYPE TO json::value, IT WILL PRODUCE UNPREDICTIBLE VALUE, USE json::value::type(basic_type_value) FOR INITIALIZATION
	output[0] = json::value::boolean(0);
	output[1] = json::value::boolean(1);
	output[2] = json::value::boolean(0);
	output[3] = json::value::boolean(1);
	output[4] = json::value::boolean(0);
	output[5] = json::value::boolean(1);
	EXPECT_EQ(RestUtil::vectorBool2Json(input), output);
}

TEST(RestUtil_test, vectorString2Json) {
	vector<string> input = { "This", "is", "a", "test" };
	json::value output = json::value::array();
	output[0] = json::value::string(U("This"));
	output[1] = json::value::string(U("is"));
	output[2] = json::value::string(U("a"));
	output[3] = json::value::string(U("test"));
	EXPECT_EQ(RestUtil::vectorString2Json(input), output);
}

TEST(RestUtil_test, vectorBool2d2Json) {
	vector<vector<bool>> input = { { 0, 1, 0, 1, 0, 1 }, {1, 0, 1, 0}, {0, 1, 0, 1, 0} };
	json::value output = json::value::array();
	json::value v1 = json::value::array();
	json::value v2 = json::value::array();
	json::value v3 = json::value::array();
	//!!!!! DO NOT EVER ASSIGN BASIC TYPE TO json::value, IT WILL PRODUCE UNPREDICTIBLE VALUE, USE json::value::type(basic_type_value) FOR INITIALIZATION
	v1[0] = json::value::boolean(0); v1[1] = json::value::boolean(1); v1[2] = json::value::boolean(0);
	v1[3] = json::value::boolean(1); v1[4] = json::value::boolean(0); v1[5] = json::value::boolean(1);
	v2[0] = json::value::boolean(1); v2[1] = json::value::boolean(0); v2[2] = json::value::boolean(1); v2[3] = json::value::boolean(0);
	v3[0] = json::value::boolean(0); v3[1] = json::value::boolean(1); v3[2] = json::value::boolean(0);
	v3[3] = json::value::boolean(1); v3[4] = json::value::boolean(0);
	output[0] = v1;
	output[1] = v2;
	output[2] = v3;

	EXPECT_EQ(RestUtil::vectorBool2d2Json(input), output);
}

TEST(RestUtil_test, vectorString2d2Json) {
	vector<vector<string>> input = { { "This", "is", "a", "test" },{ "This", "is", "test", "for", "UMA" },{ "UMA", "test", "is", "here" } };
	json::value output = json::value::array();
	json::value v1 = json::value::array();
	json::value v2 = json::value::array();
	json::value v3 = json::value::array();
	//!!!!! DO NOT EVER ASSIGN BASIC TYPE TO json::value, IT WILL PRODUCE UNPREDICTIBLE VALUE, USE json::value::type(basic_type_value) FOR INITIALIZATION
	v1[0] = json::value::string(U("This")); v1[1] = json::value::string(U("is"));
	v1[2] = json::value::string(U("a")); v1[3] = json::value::string(U("test"));
	v2[0] = json::value::string(U("This")); v2[1] = json::value::string(U("is")); v2[2] = json::value::string(U("test"));
	v2[3] = json::value::string(U("for")); v2[4] = json::value::string(U("UMA"));
	v3[0] = json::value::string(U("UMA")); v3[1] = json::value::string(U("test"));
	v3[2] = json::value::string(U("is")); v3[3] = json::value::string(U("here"));
	output[0] = v1;
	output[1] = v2;
	output[2] = v3;

	EXPECT_EQ(RestUtil::vectorString2d2Json(input), output);
}

TEST(RestUtil_test, vectorInt2d2Json) {
	vector<vector<int>> input = { { 0, 1, 2, 3, 4, 5 },{ 7, 6, 5, 4 },{ 0, 10, 20, 100, 30 } };
	json::value output = json::value::array();
	json::value v1 = json::value::array();
	json::value v2 = json::value::array();
	json::value v3 = json::value::array();
	//!!!!! DO NOT EVER ASSIGN BASIC TYPE TO json::value, IT WILL PRODUCE UNPREDICTIBLE VALUE, USE json::value::type(basic_type_value) FOR INITIALIZATION
	v1[0] = json::value::number(0); v1[1] = json::value::number(1); v1[2] = json::value::number(2);
	v1[3] = json::value::number(3); v1[4] = json::value::number(4); v1[5] = json::value::number(5);
	v2[0] = json::value::number(7); v2[1] = json::value::number(6); v2[2] = json::value::number(5); v2[3] = json::value::number(4);
	v3[0] = json::value::number(0); v3[1] = json::value::number(10); v3[2] = json::value::number(20);
	v3[3] = json::value::number(100); v3[4] = json::value::number(30);
	output[0] = v1;
	output[1] = v2;
	output[2] = v3;

	EXPECT_EQ(RestUtil::vectorInt2d2Json(input), output);
}

TEST(RestUtil_test, vectorDouble2d2Json) {
	vector<vector<double>> input = { { .0, .1, .8, 2.3, 0.4, 3.1 },{ 1.1, 3.0, 11.1, 0.32 },{ 0.34, 10.02, 0.3, 1.1, 0.005 } };
	json::value output = json::value::array();
	json::value v1 = json::value::array();
	json::value v2 = json::value::array();
	json::value v3 = json::value::array();
	//!!!!! DO NOT EVER ASSIGN BASIC TYPE TO json::value, IT WILL PRODUCE UNPREDICTIBLE VALUE, USE json::value::type(basic_type_value) FOR INITIALIZATION
	v1[0] = json::value::number(.0); v1[1] = json::value::number(.1); v1[2] = json::value::number(.8);
	v1[3] = json::value::number(2.3); v1[4] = json::value::number(0.4); v1[5] = json::value::number(3.1);
	v2[0] = json::value::number(1.1); v2[1] = json::value::number(3.0); v2[2] = json::value::number(11.1); v2[3] = json::value::number(0.32);
	v3[0] = json::value::number(0.34); v3[1] = json::value::number(10.02); v3[2] = json::value::number(0.3);
	v3[3] = json::value::number(1.1); v3[4] = json::value::number(0.005);
	output[0] = v1;
	output[1] = v2;
	output[2] = v3;

	EXPECT_EQ(RestUtil::vectorDouble2d2Json(input), output);
}


TEST_F(UMARestTestFixture, UMARestRequest_getData) {
	//get string data
	clientData[U("type")] = json::value::string(U("get_string_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("this is"));
	clientString = "a test";
	clientData[U("this is")] = json::value::string(RestUtil::string2string_t(clientString));
	testAction("Post");
	clientData.erase(U("this is"));
	EXPECT_EQ(serverString, clientString);
	//2nd
	clientData[U("p_name")] = json::value::string(U("UMA"));
	clientString = "is an machine-learning architecture";
	clientData[U("UMA")] = json::value::string(RestUtil::string2string_t(clientString));
	testAction("Post");
	clientData.erase(U("UMA"));
	EXPECT_EQ(serverString, clientString);

	//get int data
	clientData[U("type")] = json::value::string(U("get_int_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("agent_id"));
	clientInt = 12223;
	clientData[U("agent_id")] = json::value::number(clientInt);
	testAction("Post");
	clientData.erase(U("agent_id"));
	EXPECT_EQ(serverInt, clientInt);

	//get double data
	clientData[U("type")] = json::value::string(U("get_double_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("threshold"));
	clientDouble = 0.12524;
	clientData[U("threshold")] = json::value::number(clientDouble);
	testAction("Post");
	clientData.erase(U("threshold"));
	EXPECT_EQ(serverDouble, clientDouble);
	//2nd
	clientData[U("p_name")] = json::value::string(U("phi"));
	clientDouble = -0.01;
	clientData[U("phi")] = json::value::number(clientDouble);
	testAction("Post");
	clientData.erase(U("phi"));
	EXPECT_EQ(serverDouble, clientDouble);

	//get bool data
	clientData[U("type")] = json::value::string(U("get_bool_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("signal"));
	clientBool = true;
	clientData[U("signal")] = json::value::boolean(clientBool);
	testAction("Post");
	clientData.erase(U("signal"));
	EXPECT_EQ(serverBool, clientBool);
	//2nd
	clientData[U("p_name")] = json::value::string(U("signal"));
	clientBool = false;
	clientData[U("signal")] = json::value::boolean(clientBool);
	testAction("Post");
	clientData.erase(U("signal"));
	EXPECT_EQ(serverBool, clientBool);

	//get vector<int> data
	clientData[U("type")] = json::value::string(U("get_int1d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("dists"));
	clientInt1d = {1, 2, 3, 4, 5, 6, 7};
	clientData[U("dists")] = RestUtil::vectorInt2Json(clientInt1d);
	testAction("Post");
	clientData.erase(U("dists"));
	EXPECT_EQ(serverInt1d, clientInt1d);

	//get vector<bool> data
	clientData[U("type")] = json::value::string(U("get_bool1d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("signals"));
	clientBool1d = { false, true, false, true, true, false, false, false, true, true };
	clientData[U("signals")] = RestUtil::vectorBool2Json(clientBool1d);
	testAction("Post");
	clientData.erase(U("signals"));
	EXPECT_EQ(serverBool1d, clientBool1d);

	//get vector<string> data
	clientData[U("type")] = json::value::string(U("get_string1d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("s_ids"));
	clientString1d = { "sensor0", "sensor1", "sensor2", "sensor3", "sensor4", "sensor5"};
	clientData[U("s_ids")] = RestUtil::vectorString2Json(clientString1d);
	testAction("Post");
	clientData.erase(U("s_ids"));
	EXPECT_EQ(serverString1d, clientString1d);

	//get vector<double> data
	clientData[U("type")] = json::value::string(U("get_double1d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("diags"));
	clientDouble1d = { 0.2, 0.4, 0.1, 0.6, 0.02, 0.09, 0, 1.02, 10.33, 9.46, 1000.0001 };
	clientData[U("diags")] = RestUtil::vectorDouble2Json(clientDouble1d);
	testAction("Post");
	clientData.erase(U("diags"));
	EXPECT_EQ(serverDouble1d, clientDouble1d);

	//get vector<vector<int>> data
	clientData[U("type")] = json::value::string(U("get_int2d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("dists"));
	clientInt2d = { {}, {1, 2, 3}, {100, 200}, {5, 9, 10, 20,}, {8} };
	clientData[U("dists")] = RestUtil::vectorInt2d2Json(clientInt2d);
	testAction("Post");
	clientData.erase(U("dists"));
	EXPECT_EQ(serverInt2d, clientInt2d);

	//get vector<vector<bool>> data
	clientData[U("type")] = json::value::string(U("get_bool2d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("signals"));
	clientBool2d = { {true, false, true, false}, {}, {false, false, false}, {true, true, true, true} };
	clientData[U("signals")] = RestUtil::vectorBool2d2Json(clientBool2d);
	testAction("Post");
	clientData.erase(U("signals"));
	EXPECT_EQ(serverBool2d, clientBool2d);

	//get vector<vector<string>> data
	clientData[U("type")] = json::value::string(U("get_string2d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("sids"));
	clientString2d = { {},{"sid00", "sid01", "sid02"}, {"sid10"}, {"sid20"}, {"sid30", "sid31", "sid32", "sid33", "sid34"} };
	clientData[U("sids")] = RestUtil::vectorString2d2Json(clientString2d);
	testAction("Post");
	clientData.erase(U("sids"));
	EXPECT_EQ(serverString2d, clientString2d);

	//get vector<vector<double>> data
	clientData[U("type")] = json::value::string(U("get_double2d_data"));
	//1st
	clientData[U("p_name")] = json::value::string(U("weights"));
	clientDouble2d = { {},{ .1, .2, .3 },{ 10.0, 20.0 },{ 5.2, 9.4, 1.0, .20, },{ 80.9 } };
	clientData[U("weights")] = RestUtil::vectorDouble2d2Json(clientDouble2d);
	testAction("Post");
	clientData.erase(U("weights"));
	EXPECT_EQ(serverDouble2d, clientDouble2d);
}

TEST_F(UMARestTestFixture, UMARestRequest_getQuery) {
	//get string query
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//BE VERY CAREFUL, WHEN APPEND_QUERY USING STRING VALUE, DO NOT USE json::value, OR SERVER SIDE CANNOT PARSE STRING CORRECTLY!
	//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	//1st
	clientQuery.append_query<string_t>(U("type"), U("get_string_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("this is"));
	clientString = "a test";
	clientQuery.append_query<string_t>(U("this is"), RestUtil::string2string_t(clientString));
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverString, clientString);
	//2nd
	clientQuery.append_query<string_t>(U("type"), U("get_string_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("UMA"));
	clientString = "is an machine-learning architecture";
	clientQuery.append_query<string_t>(U("UMA"), RestUtil::string2string_t(clientString));
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverString, clientString);

	//get int query
	//1st
	clientQuery.append_query<string_t>(U("type"), U("get_int_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("agent_count"));
	clientInt = 3123;
	clientQuery.append_query<int>(U("agent_count"), clientInt);
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverInt, clientInt);

	//get double query
	//1st
	clientQuery.append_query<string_t>(U("type"), U("get_double_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("threshold"));
	clientDouble = 0.1253;
	clientQuery.append_query<double>(U("threshold"), clientDouble);
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverDouble, clientDouble);

	//get int query
	//1st
	clientQuery.append_query<string_t>(U("type"), U("get_bool_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("active"));
	clientBool = true;
	clientQuery.append_query<bool>(U("active"), clientBool);
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverBool, clientBool);
	//2nd
	clientQuery.append_query<string_t>(U("type"), U("get_bool_query"));
	clientQuery.append_query<string_t>(U("p_name"), U("active"));
	clientBool = false;
	clientQuery.append_query<bool>(U("active"), clientBool);
	testAction("Get");
	clientQuery.clear();
	EXPECT_EQ(serverBool, clientBool);
}

TEST_F(UMARestTestFixture, UMARestRequest_setMessage) {
	clientData[U("type")] = json::value::string(U("set_message"));
	clientData[U("p_name")] = json::value::string(U("set_message"));
	serverString = "This is a test string";
	testAction("Receiving");
	EXPECT_EQ(serverString, clientString);
}

TEST_F(UMARestTestFixture, UMARestRequest_setData) {
	//int
	clientData[U("type")] = json::value::string(U("set_data_int"));
	clientData[U("p_name")] = json::value::string(U("set_data_int"));
	serverInt = 123;
	testAction("Receiving");
	EXPECT_EQ(serverInt, clientInt);
	
	//double
	clientData[U("type")] = json::value::string(U("set_data_double"));
	clientData[U("p_name")] = json::value::string(U("set_data_double"));
	serverDouble = 0.1267;
	testAction("Receiving");
	EXPECT_EQ(serverDouble, clientDouble);
	
	//bool
	clientData[U("type")] = json::value::string(U("set_data_bool"));
	clientData[U("p_name")] = json::value::string(U("set_data_bool"));
	serverBool = true;
	testAction("Receiving");
	EXPECT_EQ(serverBool, clientBool);
	
	serverBool = false;
	testAction("Receiving");
	EXPECT_EQ(serverBool, clientBool);
	
	//string
	clientData[U("type")] = json::value::string(U("set_data_string"));
	clientData[U("p_name")] = json::value::string(U("set_data_string"));
	serverString = "UMA is a good project";
	testAction("Receiving");
	EXPECT_EQ(serverString, clientString);
	
	//int1d
	clientData[U("type")] = json::value::string(U("set_data_int1d"));
	clientData[U("p_name")] = json::value::string(U("set_data_int1d"));
	serverInt1d = { 1, 3, 2, 0, -1, 1000, 10 };
	testAction("Receiving");
	EXPECT_EQ(serverInt1d, clientInt1d);
	
	//double1d
	clientData[U("type")] = json::value::string(U("set_data_double1d"));
	clientData[U("p_name")] = json::value::string(U("set_data_double1d"));
	serverDouble1d = { 0.1, 3.33, 2.145, -0.245, 100.0001, 132.1, 10.32 };
	testAction("Receiving");
	EXPECT_EQ(serverDouble1d, clientDouble1d);

	//string1d
	clientData[U("type")] = json::value::string(U("set_data_string1d"));
	clientData[U("p_name")] = json::value::string(U("set_data_string1d"));
	serverString1d = { "this", " ", "a", "", "simple", "UMA", "-", " ", "_", "TEST" };
	testAction("Receiving");
	EXPECT_EQ(serverString1d, clientString1d);
	
	//bool1d
	clientData[U("type")] = json::value::string(U("set_data_bool1d"));
	clientData[U("p_name")] = json::value::string(U("set_data_bool1d"));
	serverBool1d = { true, false, false, false, false, true, true, true };
	testAction("Receiving");
	EXPECT_EQ(serverBool1d, clientBool1d);

	//int2d
	clientData[U("type")] = json::value::string(U("set_data_int2d"));
	clientData[U("p_name")] = json::value::string(U("set_data_int2d"));
	serverInt2d = { {}, {0, 1, 2}, {2, -1, 3, 5}, {5} };
	testAction("Receiving");
	EXPECT_EQ(serverInt2d, clientInt2d);
	
	//double2d
	clientData[U("type")] = json::value::string(U("set_data_double2d"));
	clientData[U("p_name")] = json::value::string(U("set_data_double2d"));
	serverDouble2d = { {0.001},{ 0.12, 30.1, -2 },{ 2.01, -10},{ 5.1, 8.09, 3.04 }, {} };
	testAction("Receiving");
	EXPECT_EQ(serverDouble2d, clientDouble2d);

	//string2d
	clientData[U("type")] = json::value::string(U("set_data_string2d"));
	clientData[U("p_name")] = json::value::string(U("set_data_string2d"));
	serverString2d = { {}, {"UMA", " ", "is", "-", "_", " ", "an", "architecture"}, {"this", "is", " ", "a", "test"}, {"unit_test"} };
	testAction("Receiving");
	EXPECT_EQ(serverString2d, clientString2d);

	//bool2d
	clientData[U("type")] = json::value::string(U("set_data_bool2d"));
	clientData[U("p_name")] = json::value::string(U("set_data_bool2d"));
	serverBool2d = { {}, {true, false, false}, {false, true, true}, {false, false}, {true, true, true, true, true} };
	testAction("Receiving");
	EXPECT_EQ(serverBool2d, clientBool2d);

	//map string string
	clientData[U("type")] = json::value::string(U("set_data_map_string_string"));
	clientData[U("p_name")] = json::value::string(U("set_data_map_string_string"));
	serverMapStringString["This is"] = "a test";
	serverMapStringString["UMA"] = "is an architecture";
	testAction("Receiving");
	EXPECT_EQ(serverMapStringString, clientMapStringString);

	//map string int
	clientData[U("type")] = json::value::string(U("set_data_map_string_int"));
	clientData[U("p_name")] = json::value::string(U("set_data_map_string_int"));
	serverMapStringInt["value1"] = 1;
	serverMapStringInt["value100"] = 100;
	serverMapStringInt["value0"] = 0;
	serverMapStringInt["value-100"] = -100;
	testAction("Receiving");
	EXPECT_EQ(serverMapStringInt, clientMapStringInt);

	//map string double
	clientData[U("type")] = json::value::string(U("set_data_map_string_double"));
	clientData[U("p_name")] = json::value::string(U("set_data_map_string_double"));
	serverMapStringDouble["value2.5"] = 2.5;
	serverMapStringDouble["value13.0003"] = 13.0003;
	testAction("Receiving");
	EXPECT_EQ(serverMapStringDouble, clientMapStringDouble);

	//map string bool
	clientData[U("type")] = json::value::string(U("set_data_map_string_bool"));
	clientData[U("p_name")] = json::value::string(U("set_data_map_string_bool"));
	serverMapStringBool["true"] = true;
	serverMapStringBool["false"] = false;
	testAction("Receiving");
	EXPECT_EQ(serverMapStringBool, clientMapStringBool);
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}