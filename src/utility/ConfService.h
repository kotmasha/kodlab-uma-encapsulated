#ifndef _CONFSERVICE_
#define _CONFWERVICE_

#include "Global.h"

class PropertyPage;

class ConfService{
private:
	static ConfService *_confService;
	std::map<string, PropertyPage*> _confInfo;

public:
	ConfService();
	~ConfService();
	static ConfService *instance();
	void addPropertyPage(const string name, PropertyPage *ppm);
	PropertyPage *getPropertyPage(const string name) const;
	PropertyPage *getLogPage() const;
	PropertyPage *getServerPage() const;
	PropertyPage *getCorePage() const;
	PropertyPage *getRestmapPage() const;

private:
	void logInit();
	void coreInit();
	void restmapInit();
	void serverInit();
};

#endif
