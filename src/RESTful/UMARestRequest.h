#ifndef _UMA_REST_REQUEST_
#define _UMA_REST_REQUEST_

#include "Global.h"
#include "UMARest.h"

class UMARestHandler;
class UMARestListener;

class DLL_PUBLIC UMARestRequest {
protected:
	http_request _request;
	http_response _response;
	json::value _data;
	std::map<string_t, string_t> _query;
	json::value _response_body;

	friend class UMARestHandler;
	friend class UMARestListener;

public:
	UMARestRequest(const http_request &request);
	~UMARestRequest();

	string get_string_data(const string &name);
	string get_string_query(const string &name);
	int get_int_data(const string &name);
	int get_int_query(const string &name);
	double get_double_data(const string &name);
	double get_double_query(const string &name);
	bool get_bool_data(const string &name);
	bool get_bool_query(const string &name);
	vector<int> get_int1d_data(const string &name);
	vector<bool> get_bool1d_data(const string &name);
	vector<double> get_double1d_data(const string &name);
	vector<string> get_string1d_data(const string &name);
	vector<vector<int> > get_int2d_data(const string &name);
	vector<vector<bool> > get_bool2d_data(const string &name);
	vector<vector<double> > get_double2d_data(const string &name);
	vector<vector<string> > get_string2d_data(const string &name);

	const string get_request_url() const;
	const string get_absolute_url() const;
	void set_message(const string message);

	void set_data(const string &name, int value);
	void set_data(const string &name, double value);
	void set_data(const string &name, bool value);
	void set_data(const string &name, string value);
	void set_data(const string &name, const std::vector<int> &list);
	void set_data(const string &name, const std::vector<double> &list);
	void set_data(const string &name, const std::vector<bool> &list);
	void set_data(const string &name, const std::vector<string> &list);
	void set_data(const string &name, const std::vector<vector<int>> &lists);
	void set_data(const string &name, const std::vector<vector<double>> &lists);
	void set_data(const string &name, const std::vector<vector<bool>> &lists);
	void set_data(const string &name, const std::vector<vector<string>> &lists);
	void set_data(const string &name, const std::map<string, string> &map);
	void set_data(const string &name, const std::map<string, int> &map);
	void set_data(const string &name, const std::map<string, double> &map);
	void set_data(const string &name, const std::map<string, bool> &map);

	bool check_data_field(const string &name);
	bool check_query_field(const string &name);

	void reply();

private:
	void set_status_code(const status_code code);
};

#endif