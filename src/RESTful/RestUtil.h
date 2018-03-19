#ifndef _RESTUTIL_
#define _RESTUTIL_

#include "Global.h"
#include "UMARest.h"

namespace RestUtil {
	DLL_PUBLIC string_t string_to_string_t(const string &s);
	DLL_PUBLIC string string_t_to_string(const string_t &s);
	DLL_PUBLIC bool string_t_to_bool(const string_t &s);
	DLL_PUBLIC status_code error_type_to_status_code(const int error_type);
	DLL_PUBLIC string status_code_to_string(const status_code s);

	DLL_PUBLIC bool check_field(const json::value &data, const string_t &s, bool hard_check=true);
	DLL_PUBLIC bool check_field(const map<string_t, string_t> &query, const string_t &s, bool hard_check=true);
	DLL_PUBLIC json::value vector_int_to_json(const std::vector<int> &list);
	DLL_PUBLIC json::value vector_double_to_json(const std::vector<double> &list);
	DLL_PUBLIC json::value vector_bool_to_json(const std::vector<bool> &list);
	DLL_PUBLIC json::value vector_string_to_json(const std::vector<string> &list);
	DLL_PUBLIC json::value vector_bool2d_to_json(const std::vector<vector<bool> > &lists);
	DLL_PUBLIC json::value vector_int2d_to_json(const std::vector<vector<int> > &lists);
	DLL_PUBLIC json::value vector_double2d_to_json(const std::vector<vector<double> > &lists);
	DLL_PUBLIC json::value vector_string2d_to_json(const std::vector<vector<string> > &lists);
}

#endif
