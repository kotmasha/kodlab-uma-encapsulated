#include "PropertyPage.h"
#include "PropertyMap.h"

PropertyPage::PropertyPage(){}

PropertyPage::~PropertyPage() {
	for (auto it = this->begin(); it != this->end(); ++it) {
		delete it->second;
		it->second = nullptr;
	}
}

void PropertyPage::add(const string &key, PropertyMap * const ppm) {
	(*this)[key] = ppm;
}

bool PropertyPage::exist(const string &key) {
	return this->end() != this->find(key);
}

void PropertyPage::remove(const string &key) {
	if (this->exist(key)) {
		this->erase(key);
	}
}

PropertyMap *PropertyPage::get(const string &key) {
	if (this->exist(key)) {
		return (*this)[key];
	}
	return nullptr;
}

void PropertyPage::extend(PropertyPage *other) {
	if (!other) return;
	for (auto it = other->begin(); it != other->end(); ++it) {
		this->add(it->first, it->second);
	}
}

/*
//come back later
template <class T>
void PropertyMap<T>::copy(PropertyMap *other) {
PropertyMap tmp(*other);
}
*/