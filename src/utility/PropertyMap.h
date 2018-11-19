#ifndef _PROPERTYMAP_
#define _PROPERTYMAP_

#include "Global.h"

using namespace std;

class PropertyMap : public std::map<string, string> {
public:
	PropertyMap();
	~PropertyMap();
	void add(const string &key, const string &value);
	bool exist(const string &key);
	void remove(const string &key);
	string get(const string &key);
	void extend(PropertyMap *other);
	vector<string> getKeys();
	void save(ofstream &file);
	void load(ifstream &file);
	//void copy(PropertyMap *other);
};

#endif