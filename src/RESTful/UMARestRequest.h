#ifndef _UMA_REST_REQUEST_
#define _UMA_REST_REQUEST_

#include "Global.h"
#include "UMARest.h"
#include "PropertyMap.h"

class UMARestHandler;
class UMARestListener;

class DLL_PUBLIC UMARestRequest {
protected:
	http_request _request;
	http_response _response;
	json::value _data;
	std::map<string_t, string_t> _query;
	json::value _body;

	friend class UMARestHandler;
	friend class UMARestListener;
	friend class UMARestClient;

public:
	UMARestRequest(const http_request &request);
	UMARestRequest(const web::uri &u, const http::method m);
	~UMARestRequest();

	string getStringData(const string &name);
	string getStringQuery(const string &name);
	int getIntData(const string &name);
	int getIntQuery(const string &name);
	double getDoubleData(const string &name);
	double getDoubleQuery(const string &name);
	bool getBoolData(const string &name);
	bool getBoolQuery(const string &name);
	vector<int> getInt1dData(const string &name);
	vector<bool> getBool1dData(const string &name);
	vector<double> getDouble1dData(const string &name);
	vector<string> getString1dData(const string &name);
	vector<vector<int> > getInt2dData(const string &name);
	vector<vector<bool> > getBool2dData(const string &name);
	vector<vector<double> > getDouble2dData(const string &name);
	vector<vector<string> > getString2dData(const string &name);
	void getValueInKeys(vector<string> &keys, PropertyMap &ppm);
	

	const string getRequestUrl() const;
	const string getAbsoluteUrl() const;
	void setMessage(const string message);

	void setData(const string &name, int value);
	void setData(const string &name, double value);
	void setData(const string &name, bool value);
	void setData(const string &name, string value);
	void setData(const string &name, const std::vector<int> &list);
	void setData(const string &name, const std::vector<double> &list);
	void setData(const string &name, const std::vector<bool> &list);
	void setData(const string &name, const std::vector<string> &list);
	void setData(const string &name, const std::vector<vector<int>> &lists);
	void setData(const string &name, const std::vector<vector<double>> &lists);
	void setData(const string &name, const std::vector<vector<bool>> &lists);
	void setData(const string &name, const std::vector<vector<string>> &lists);
	void setData(const string &name, const std::map<string, string> &map);
	void setData(const string &name, const std::map<string, int> &map);
	void setData(const string &name, const std::map<string, double> &map);
	void setData(const string &name, const std::map<string, bool> &map);

	bool checkDataField(const string &name);
	bool checkQueryField(const string &name);

	void reply();

private:
	void setStatusCode(const status_code code);
};

#endif