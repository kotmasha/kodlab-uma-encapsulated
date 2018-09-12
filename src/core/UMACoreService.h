#ifndef _UMACORESERVICE_
#define _UMACORESERVICE_

#include "Global.h"

class UMACoreService{
private:
	static UMACoreService *_coreService;
	std::map<string, std::map<string, string>> _coreInfo;

public:
	UMACoreService();
	~UMACoreService();
	static UMACoreService *instance();
	std::map<string, string> getPropertyMap(const string &objName);
	string getPropertyValue(const string &objName, const string &key);
};

#endif
