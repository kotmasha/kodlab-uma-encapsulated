#include "UMACoreService.h"
#include "ConfReader.h"
#include "PropertyMap.h"
#include "UMAException.h"

UMACoreService *UMACoreService::_coreService = nullptr;

UMACoreService::UMACoreService() {
	_coreInfo = ConfReader::readConf("core.ini");
	cout << "CoreService is initiated!" << endl;
}

UMACoreService::~UMACoreService() {
	delete _coreInfo;
}

UMACoreService *UMACoreService::instance() {
	if (!_coreService) {
		_coreService = new UMACoreService();
	}
	return _coreService;
}

PropertyMap *UMACoreService::getPropertyMap(const string &objName) {
	if (!_coreInfo->exist(objName)) {
		string err = "The property map " + objName + " does not exist";
		throw UMAInternalException(err, false);
	}
	return _coreInfo->get(objName);
}

string UMACoreService::getPropertyValue(const string &objName, const string &key) {
	PropertyMap *propertyMap = this->getPropertyMap(objName);
	if (!propertyMap->exist(key)) {
		string err = "The property key " + key + " does not exist";
		throw UMAInternalException(err, false);
	}
	return propertyMap->get(key);
}