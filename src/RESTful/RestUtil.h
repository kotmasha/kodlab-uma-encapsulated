#ifndef _RESTUTIL_
#define _RESTUTIL_

#include "Global.h"
#include "UMARest.h"

class UMAException;

enum REQUEST_TYPE {GET, POST, PUT, DEL};

namespace RestUtil {
	DLL_PUBLIC string_t string2string_t(const string &s);
	DLL_PUBLIC string string_t2string(const string_t &s);
	DLL_PUBLIC bool string_t2bool(const string_t &s);
	DLL_PUBLIC status_code UMAExceptionToStatusCode(UMAException *ex);
	DLL_PUBLIC string status_code2string(const status_code s);

	DLL_PUBLIC bool checkField(const json::value &data, const string_t &s, bool hardCheck=true);
	DLL_PUBLIC bool checkField(const map<string_t, string_t> &query, const string_t &s, bool hardCheck=true);
	DLL_PUBLIC json::value vectorInt2Json(const std::vector<int> &list);
	DLL_PUBLIC json::value vectorDouble2Json(const std::vector<double> &list);
	DLL_PUBLIC json::value vectorBool2Json(const std::vector<bool> &list);
	DLL_PUBLIC json::value vectorString2Json(const std::vector<string> &list);
	DLL_PUBLIC json::value vectorBool2d2Json(const std::vector<vector<bool> > &lists);
	DLL_PUBLIC json::value vectorInt2d2Json(const std::vector<vector<int> > &lists);
	DLL_PUBLIC json::value vectorDouble2d2Json(const std::vector<vector<double> > &lists);
	DLL_PUBLIC json::value vectorString2d2Json(const std::vector<vector<string> > &lists);
}

#endif
