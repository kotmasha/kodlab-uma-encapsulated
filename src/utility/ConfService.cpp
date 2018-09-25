#include "ConfService.h"
#include "ConfReader.h"
#include "PropertyMap.h"
#include "UMAUtil.h"
#include "UMAException.h"

ConfService *ConfService::_confService = nullptr;

ConfService::ConfService() {
	logInit();
	coreInit();
	serverInit();
	restmapInit();

	cout << "conf service is initiated" << endl;
}

ConfService::~ConfService() {
	for (auto it = _confInfo.begin(); it != _confInfo.end(); ++it) {
		delete it->second;
		it->second = nullptr;
	}
}

ConfService *ConfService::instance() {
	if (!_confService) {
		_confService = new ConfService();
	}
	return _confService;
}

void ConfService::addPropertyPage(const string name, PropertyPage *ppm) {
	if (_confInfo.end() != _confInfo.find(name)) {
		throw UMAInternalException("Cannot add the PropertyPage=" + name + " because it already exists!");
	}
	_confInfo[name] = ppm;
}

PropertyPage *ConfService::getPropertyPage(const string name) const{
	if (_confInfo.end() == _confInfo.find(name)) {
		throw UMAInternalException("Cannot find the PropertyPage=" + name);
	}
	return _confInfo.at(name);
}

PropertyPage *ConfService::getLogPage() const{
	return getPropertyPage("log");
}

PropertyPage *ConfService::getServerPage() const {
	return getPropertyPage("server");
}

PropertyPage *ConfService::getCorePage() const {
	return getPropertyPage("core");
}

PropertyPage *ConfService::getRestmapPage() const {
	return getPropertyPage("restmap");
}

void ConfService::logInit() {
	try {
		string logFolder = "log";
		SysUtil::UMAMkdir(logFolder);
	}
	catch (exception &ex) {
		cout << "Cannot make a log folder, error=" + string(ex.what()) << endl;
	}

	PropertyPage *logInfo = ConfReader::readConf("log.ini");
	_confInfo["log"] = logInfo;
}

void ConfService::coreInit() {
	PropertyPage *coreInfo = ConfReader::readConf("core.ini");
	_confInfo["core"] = coreInfo;
}

void ConfService::serverInit() {
	PropertyPage *serverInfo = ConfReader::readConf("server.ini");
	_confInfo["server"] = serverInfo;
}

void ConfService::restmapInit() {
	PropertyPage *restmapInfo = ConfReader::readRestmap();
	_confInfo["restmap"] = restmapInfo;
}