#include "RestUtil.h"
#include "UMAException.h"

/*
transfering the string to string_t
input: string
output: string_t
*/
string_t RestUtil::string_to_string_t(const string &s) {
	string_t ss(s.begin(), s.end());
	return ss;
}

/*
transfering the string_t to string
input: string_t
output: string
*/
string RestUtil::string_t_to_string(const string_t &s) {
	string ss(s.begin(), s.end());
	return ss;
}

/*
This function is transfering a string to bool value
input: string_t
output: bool value
*/
bool RestUtil::string_t_to_bool(const string_t &s) {
	return !(s == U("false") || s == U("False") || s == U("0"));
}


status_code RestUtil::UMAExceptionToStatusCode(UMAException *ex){
	const UMAException::UMAExceptionType type = ex->getType();
	switch (type) {
	case UMAException::UMAExceptionType::UMA_BAD_OPERATION:
	case UMAException::UMAExceptionType::UMA_INVALID_ARGS:
		return status_codes::BadRequest;
	case UMAException::UMAExceptionType::UMA_INTERNAL:
		return status_codes::InternalError;
	case UMAException::UMAExceptionType::UMA_DUPLICATION:
		return status_codes::Conflict;
	case UMAException::UMAExceptionType::UMA_NO_RESOURCE:
	default:
		return status_codes::NotFound;
	}
}

string RestUtil::status_code_to_string(const status_code s) {
	string ss = to_string(s);
	return ss;
}

/*
The function to check whether a specified field exist in a json::value, if hard check is true, will throw exception when the field not exist
Input: data from request, field name s, and bool value for hard_check
Output: whether the field exist
*/
bool RestUtil::check_field(const json::value &data, const string_t &s, bool hard_check) {
	if (!data.has_field(s)) {
		if (hard_check) {
			throw UMAInvalidArgsException("Coming request is missing necessary fields");
		}
		return false;
	}
	return true;
}

/*
The function to check whether a specified field exist in a map, if hard check is true, will throw exception when the field not exist
Input: data from request, field name s, and bool value for hard_check
Output: whether the field exist
*/
bool RestUtil::check_field(const map<string_t, string_t> &query, const string_t &s, bool hard_check) {
	if (query.find(s) == query.end()) {
		if (hard_check) {
			throw UMAInvalidArgsException("Coming request is missing necessary fields");
		}
		return false;
	}
	return true;
}

/*
The function is transfering a vector of int to a json::value
Input: vector int
Output: vector json::value
*/
json::value RestUtil::vector_int_to_json(const std::vector<int> &list) {
	vector<json::value> results;
	for (int i = 0; i < list.size(); ++i) {
		results.push_back(json::value::number(list[i]));
	}
	return json::value::array(results);
}

/*
The function is transfering a vector of double to a json::value
Input: vector double
Output: vector json::value
*/
json::value RestUtil::vector_double_to_json(const std::vector<double> &list) {
	std::vector<json::value> results;
	for (int i = 0; i < list.size(); ++i) {
		results.push_back(json::value::number(list[i]));
	}
	return json::value::array(results);
}

/*
The function is transfering a vector of bool to a json::value
Input: vector bool
Output: vector json::value
*/
json::value RestUtil::vector_bool_to_json(const std::vector<bool> &list) {
	std::vector<json::value> results;
	for (int i = 0; i < list.size(); ++i) {
		results.push_back(json::value::boolean(list[i]));
	}
	return json::value::array(results);
}

/*
The function is transfering a vector of string to a json::value
Input: vector string
Output: vector json::value
*/
json::value RestUtil::vector_string_to_json(const std::vector<string> &list) {
	std::vector<json::value> results;
	for (int i = 0; i < list.size(); ++i) {
		results.push_back(json::value::string(string_to_string_t(list[i])));
	}
	return json::value::array(results);
}

/*
The function is transfering a 2d vector of bool to a json::value
Input: 2d vector bool
Output: vector json::value
*/
json::value RestUtil::vector_bool2d_to_json(const std::vector<vector<bool> > &lists) {
	std::vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		const json::value result = vector_bool_to_json(lists[i]);
		results.push_back(result);
	}
	return json::value::array(results);
}

/*
The function is transfering a 2d vector of int to a json::value
Input: 2d vector int
Output: vector json::value
*/
json::value RestUtil::vector_int2d_to_json(const std::vector<vector<int> > &lists) {
	std::vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		const json::value result = vector_int_to_json(lists[i]);
		results.push_back(result);
	}
	return json::value::array(results);
}

/*
The function is transfering a 2d vector of double to a json::value
Input: 2d vector double
Output: vector json::value
*/
json::value RestUtil::vector_double2d_to_json(const std::vector<vector<double> > &lists) {
	std::vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		const json::value result = vector_double_to_json(lists[i]);
		results.push_back(result);
	}
	return json::value::array(results);
}

/*
The function is transfering a 2d vector of string to a vector of json::value
Input: 2d vector string
Output: vector json::value
*/
json::value RestUtil::vector_string2d_to_json(const std::vector<vector<string> > &lists) {
	std::vector<json::value> results;
	for (int i = 0; i < lists.size(); ++i) {
		const json::value result = vector_string_to_json(lists[i]);
		results.push_back(result);
	}
	return json::value::array(results);
}