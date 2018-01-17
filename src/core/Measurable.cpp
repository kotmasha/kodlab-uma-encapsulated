#include "Measurable.h"

extern int ind(int row, int col);
extern int compi(int x);

/*
Measurable::Measurable(ifstream &file) {
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	if (uuid_length > 0) {
		_uuid = string(uuid_length, ' ');
		file.read(&_uuid[0], uuid_length * sizeof(char));
	}
	else _uuid = "";
	file.read((char *)&_idx, sizeof(int));
	file.read((char *)&_isOriginPure, sizeof(bool));
	pointers_to_null();
}
*/

Measurable::Measurable(const string &uuid, int idx, bool isOriginPure, double diag): _uuid(uuid){
	_idx = idx;
	_isOriginPure = isOriginPure;
	pointers_to_null();
	_vdiag = diag;
	_vdiag_ = diag;
	_vobserve = false;
	_vobserve_ = false;
}

/*
The function is setting all pointers to NULL
*/
void Measurable::pointers_to_null(){
	_diag = NULL;
	_diag_ = NULL;
	_observe = NULL;
	_observe_ = NULL;
	_current = NULL;
}

/*
The function is copying the pointer value to values in the object
*/
void Measurable::pointers_to_values(){
	_vdiag = *_diag;
	_vdiag_ = *_diag_;
	_vobserve = *_observe;
	_vobserve_ = *_observe_;
}

/*
The function is copying values to pointer in the object
*/
void Measurable::values_to_pointers(){
	*_diag = _vdiag;
	*_diag_ = _vdiag_;
	*_observe = _vobserve;
	*_observe_ = _vobserve_;
}

/*
This is setting the diagonal value pointer
*/
void Measurable::setDiagPointers(double *_diags, double *_diags_){
	_diag = _diags + _idx;
	_diag_ = _diags_ + _idx;
}

/*
This is setting the status/current value pointer
*/
void Measurable::setObservePointers(bool *observe, bool *observe_){
	_observe = observe + _idx;
	_observe_ = observe_ + _idx;
}

void Measurable::setCurrentPointers(bool *current) {
	_current = current + _idx;
}

/*
This function is setting the measurable idx
Input: new idx
*/
void Measurable::setIdx(int idx){
	_idx = idx;
}

/*
This function is save measurable
Saving order MUST FOLLOW:
1 measurable idx
2 whether the measurable is originally pure
*/
/*
void Measurable::save_measurable(ofstream &file){
	int sid_length = _uuid.length();
	file.write(reinterpret_cast<const char *>(&sid_length), sizeof(int));
	file.write(_uuid.c_str(), sid_length * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	file.write(reinterpret_cast<const char *>(&_isOriginPure), sizeof(bool));
}
*/

/*
void Measurable::copy_data(Measurable *m) {
	_isOriginPure = m->_isOriginPure;
	_vdiag = m->_vdiag;
	_vdiag_ = m->_vdiag_;
}
*/

const double &Measurable::getDiag() const{
	return *_diag;
}

const double &Measurable::getOldDiag() const{
	return *_diag_;
}

const bool &Measurable::getIsOriginPure() const{
	return _isOriginPure;
}

const bool &Measurable::getObserve() const{
	return *_observe;
}

const bool &Measurable::getOldObserve() const{
	return *_observe_;
}

const bool &Measurable::getCurrent() const{
	return *_current;
}

void Measurable::setDiag(const double &diag) {
	*_diag = diag;
}

void Measurable::setOldDiag(const double &diag_) {
	*_diag_ = diag_;
}

void Measurable::setIsOriginPure(const bool &isOriginPure) {
	_isOriginPure = isOriginPure;
}

Measurable::~Measurable(){
	pointers_to_null();
}