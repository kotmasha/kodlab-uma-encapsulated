#ifndef _CORESERVICE_
#define _CORESERVICE_

#include "Global.h"

class PropertyPage;
class PropertyMap;

class DLL_PUBLIC CoreService {
private:
	static CoreService *_coreService;
	PropertyPage *_coreInfo;
	static std::mutex _lock;

public:
	CoreService();
	~CoreService();
	static CoreService *instance();
	static void reset();
	PropertyPage *getPropertyPage() const;
	PropertyMap *getPropertyMap(const string &objName);
	string getValue(const string &objName, const string &key);
};

#endif