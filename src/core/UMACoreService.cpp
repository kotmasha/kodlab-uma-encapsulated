#include "UMACoreService.h"
#include "ConfReader.h"
#include "UMAException.h"

UMACoreService *UMACoreService::_coreService = nullptr;

UMACoreService::UMACoreService() {
	_coreInfo = ConfReader::readConf("core.ini");
	cout << "CoreService is initiated!" << endl;
}

UMACoreService::~UMACoreService() {}

UMACoreService *UMACoreService::instance() {
	if (!_coreService) {
		_coreService = new UMACoreService();
	}
	return _coreService;
}

std::map<string, string> UMACoreService::getPropertyMap(const string &objName) {
	if (_coreInfo.end() == _coreInfo.find(objName)) {
		string err = "The property map " + objName + " does not exist";
		throw UMAInternalException(err, false);
	}
	return _coreInfo[objName];
}

string UMACoreService::getPropertyValue(const string &objName, const string &key) {
	std::map<string, string> propertyMap = this->getPropertyMap(objName);
	if (propertyMap.end() == propertyMap.find(key)) {
		string err = "The property key " + key + " does not exist";
		throw UMAInternalException(err, false);
	}
	return propertyMap[key];
}