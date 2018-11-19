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

void PropertyMap::save(ofstream &file) {
	size_t s = this->size();
	file.write(reinterpret_cast<const char *>(&s), sizeof(size_t));
	for (auto it = this->begin(); it != this->end(); ++it) {
		const string key = it->first;
		const string value = it->second;
		const size_t keySize = key.size();
		const size_t valueSize = value.size();
		file.write(reinterpret_cast<const char *>(&keySize), sizeof(size_t));
		file.write(reinterpret_cast<const char *>(key.c_str()), key.size() * sizeof(char));
		
		file.write(reinterpret_cast<const char *>(&valueSize), sizeof(size_t));
		file.write(reinterpret_cast<const char *>(value.c_str()), value.size() * sizeof(char));
	}
}

void PropertyMap::load(ifstream &file) {
	size_t s = 0;
	file.read((char *)(&s), sizeof(size_t));
	for (int i = 0; i < s; ++i) {
		size_t keySize = 0;
		file.read((char *)(&keySize), sizeof(size_t));
		string key = string(keySize, ' ');
		file.read(&key[0], keySize * sizeof(char));

		size_t valueSize = 0;
		file.read((char *)(&valueSize), sizeof(size_t));
		string value = string(valueSize, ' ');
		file.read(&value[0], valueSize * sizeof(char));

		this->add(key, value);
	}
}

/*
//come back later
template <class T>
void PropertyMap<T>::copy(PropertyMap *other) {
	PropertyMap tmp(*other);
}
*/