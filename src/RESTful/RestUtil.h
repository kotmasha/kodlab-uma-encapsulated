#ifndef _RESTUTIL_
#define _RESTUTIL_

#include "Global.h"
#include "UMARest.h"

namespace RestUtil {
	string_t string_to_string_t(const string &s);
	string string_t_to_string(const string_t &s);
	bool string_t_to_bool(const string_t &s);
	status_code error_type_to_status_code(const int error_type);
	string status_code_to_string(const status_code s);

	bool check_field(const json::value &data, const string_t &s, bool hard_check=true);
	bool check_field(const map<string_t, string_t> &query, const string_t &s, bool hard_check=true);
	json::value vector_int_to_json(const std::vector<int> &list);
	json::value vector_double_to_json(const std::vector<double> &list);
	json::value vector_bool_to_json(const std::vector<bool> &list);
	json::value vector_string_to_json(const std::vector<string> &list);
	json::value vector_bool2d_to_json(const std::vector<vector<bool> > &lists);
	json::value vector_int2d_to_json(const std::vector<vector<int> > &lists);
	json::value vector_double2d_to_json(const std::vector<vector<double> > &lists);
	json::value vector_string2d_to_json(const std::vector<vector<string> > &lists);
}

#endif
