#ifndef _PROPERTYPAGE_
#define _PROPERTYPAGE_

#include "Global.h"

class PropertyMap;
#include "PropertyMap.h"

class PropertyPage: public std::map<string, PropertyMap*> {
public:
	PropertyPage();
	~PropertyPage();
	void add(const string &key, PropertyMap * const ppm);
	bool exist(const string &key);
	void remove(const string &key);
	PropertyMap *get(const string &key);
	void extend(PropertyPage *other);
};

#endif
