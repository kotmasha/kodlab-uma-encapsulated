#include "PropertyMap.h"

PropertyMap::PropertyMap(){}

PropertyMap::~PropertyMap() {}

void PropertyMap::add(const string &key, const string &value){
	(*this)[key] = value;
}

bool PropertyMap::exist(const string &key) {
	return this->end() != this->find(key);
}

void PropertyMap::remove(const string &key) {
	if (this->exist(key)) {
		this->erase(key);
	}
}

string PropertyMap::get(const string &key) {
	if (this->exist(key)) {
		return (*this)[key];
	}
	return "";
}

void PropertyMap::extend(PropertyMap *other) {
	if (!other) return;
	for (auto it = other->begin(); it != other->end(); ++it) {
		this->add(it->first, it->second);
	}
}

vector<string> PropertyMap::getKeys() {
	vector<string> keys;
	for (auto it = this->begin(); it != this->end(); ++it) {
		keys.push_back(it->first);
	}
	return keys;
}

/*
//come back later
template <class T>
void PropertyMap<T>::copy(PropertyMap *other) {
	PropertyMap tmp(*other);
}
*/