#include "CoreService.h"
#include "ConfReader.h"
#include "PropertyMap.h"
#include "PropertyPage.h"
#include "UMAException.h"

CoreService *CoreService::_coreService = nullptr;
std::mutex CoreService::_lock;

CoreService::CoreService() {
	_coreInfo = ConfReader::readConf("core.ini");
	cout << "core service is initiated" << endl;
}

CoreService::~CoreService() {
	delete _coreInfo;
}

CoreService *CoreService::instance() {
	if (!_coreService) {
		_lock.lock();
		if (!_coreService) {
			_coreService = new CoreService();
		}
		_lock.unlock();
	}
	return _coreService;
}

void CoreService::reset() {
	delete CoreService::instance();
	CoreService::_coreService = nullptr;
}

PropertyPage *CoreService::getPropertyPage() const{
	return _coreInfo;
}

PropertyMap *CoreService::getPropertyMap(const string &objName){
	if (_coreInfo->end() == _coreInfo->find(objName)) {
		return nullptr;
	}
	return _coreInfo->get(objName);
}

string CoreService::getValue(const string &objName, const string &key){
	PropertyMap *pm = getPropertyMap(objName);
	if (!pm || pm->end() == pm->find(key)) {
		throw UMAInternalException("Cannot find key=" + key + " in PropertyMap objName=" + objName, true);
	}
	return pm->get(key);
}