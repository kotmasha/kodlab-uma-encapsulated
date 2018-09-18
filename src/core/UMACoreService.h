#ifndef _UMACORESERVICE_
#define _UMACORESERVICE_

#include "Global.h"

class PropertyMap;
class PropertyPage;

class UMACoreService{
private:
	static UMACoreService *_coreService;
	PropertyPage *_coreInfo;

public:
	UMACoreService();
	~UMACoreService();
	static UMACoreService *instance();
	PropertyMap *getPropertyMap(const string &objName);
	string getPropertyValue(const string &objName, const string &key);
};

#endif
