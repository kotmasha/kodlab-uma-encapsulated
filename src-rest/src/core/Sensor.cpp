#include "Sensor.h"
#include "Measurable.h"

extern int ind(int row, int col);
extern int compi(int x);

Sensor::Sensor(ifstream &file) {
	int sid_length = -1;
	file.read((char *)(&sid_length), sizeof(int));
	if (sid_length > 0) {
		_sid = string(sid_length, ' ');
		file.read(&_sid[0], sid_length * sizeof(char));
	}
	else _sid = "";
	_sname = _sid;

	file.read((char *)(&_idx), sizeof(int));
	//write the amper list
	int amper_size = -1;
	file.read((char *)(&amper_size), sizeof(int));
	for (int i = 0; i < amper_size; ++i) {
		int tmp_value = -1;
		file.read((char *)(&tmp_value), sizeof(int));
		_amper.push_back(tmp_value);
	}
	_m = new Measurable(file);
	_cm = new Measurable(file);
}

/*
Init function
Input: _sid is sensor id, const int, and _sname, sensor name
*/
Sensor::Sensor(string sid, string sname, int idx):_sid(sid.c_str()){
	_sname = sname;
	_idx = idx;
	_m = new Measurable(2 * idx, true);
	_cm = new Measurable(2 * idx + 1, false);
}

/*
This is setting the amper list
Input: ampers, the amper list
*/
void Sensor::init_amper_list(bool *ampers){
	int start = 2 * ind(_idx, 0);
	int end = 2 * ind(_idx + 1, 0);
	setAmperList(ampers, start, end);
}

/*
This is setting the amper list, used when the ampers to be copied is temporary
Input: ampers, the amper list, start and end of the index
*/
void Sensor::setAmperList(bool *ampers, int start, int end){
	for(int i = start; i < end; ++i){
		if(ampers[i]) _amper.push_back(i);
	}
}

/*
This is setting the amper list, used when the ampers to be copied is permanant
Input: idx to be added in the amper
*/
void Sensor::setAmperList(int idx){
	_amper.push_back(idx);
}

/*
This is setting the amper list, using a new sensor pointer, just copy what is inside the _amper to the current Sensor
Input: pointer of another sensor
*/
void Sensor::setAmperList(Sensor *sensor){
	_amper.insert(_amper.end(), sensor->_amper.begin(), sensor->_amper.end());
}

/*
This is copying amper list into the amper array
Input: the ampers array pointer
*/
void Sensor::copyAmperList(bool *ampers){
	int offset = 2 * ind(_idx, 0);
	for(int i = 2 * ind(_idx, 0); i < 2 * ind(_idx + 1, 0); ++i){
		//by default, the copy is happening to the row of the sid
		ampers[i] = false;
	}
	for(int i = 0; i < _amper.size(); ++i){
		ampers[_amper[i] + offset] = true;
	}
}

/*
This function is setting the diag pointers of _m and _cm through the sensor object
Input: weight matrix diagonal value of this and last iteration
*/
void Sensor::setMeasurableDiagPointers(double *_diags, double *_diags_){
	_m->setDiagPointers(_diags, _diags_);
	_cm->setDiagPointers(_diags, _diags_);
}

/*
This function is setting the status pointers of _m and _cm through the sensor object
Input: weight matrix diagonal value of this and last iteration
*/
void Sensor::setMeasurableStatusPointers(bool *status){
	_m->setStatusPointers(status);
	_cm->setStatusPointers(status);
}

/*
This function is copying the diag value of _m and _cm to the pointer
*/
void Sensor::values_to_pointers(){
	_m->values_to_pointers();
	_cm->values_to_pointers();
}

/*
This function is copying the pointer of _m and _cm to value
*/
void Sensor::pointers_to_values(){
	_m->pointers_to_values();
	_cm->pointers_to_values();
}

/*
This function is changing the idx when a pruning happens, the corresponding measurable idx also need to change
Input: the new idx of the sensor
*/
void Sensor::setIdx(int idx){
	_idx = idx;
	_m->setIdx(2 * _idx);
	_cm->setIdx(2 * _idx + 1);
}

/*
This function is using the amper and observe array to get the delayed sensor value
Input: observe array
*/
bool Sensor::amper_and_signals(bool *observe){
	//if(_amper.size() == 0) return observe[2 * _sid];
	//probably need to give an warning when the amper list is empty
	for(int i = 0; i < _amper.size(); ++i){
		int j = _amper[i];
		if(!observe[j]) return false;
	}
	return true;
}

/*
This function is checking whether the current sensor is active or not
*/
bool Sensor::isSensorActive(){
	return *(_m->_status);
}

/*
This function is setting pointers to NULL
*/
void Sensor::pointers_to_null(){

}

/*
This is the save sensor function
Saving order MUST FOLLOW:
1 sid of the sensor
2 sensor name
3 sensor idx
4 amper list
    4.1 the size of the amper list
	4.2 all amper list value based on the size
5 all measurables
Input: ofstream file
*/
void Sensor::save_sensor(ofstream &file){
	//write sid and idx
	int sid_length = _sid.length();
	file.write(reinterpret_cast<const char *>(&sid_length), sizeof(int));
	file.write(_sid.c_str(), sid_length * sizeof(char));
	file.write(reinterpret_cast<const char *>(&_idx), sizeof(int));
	//write the amper list
	int amper_size = _amper.size();
	file.write(reinterpret_cast<const char *>(&amper_size), sizeof(int));
	for(int i = 0; i < amper_size; ++i){
		file.write(reinterpret_cast<const char *>(&_amper[i]), sizeof(int));
	}
	_m->save_measurable(file);
	_cm->save_measurable(file);
}

/*
destruct the sensor, sensorpair destruction is a separate process
*/
Sensor::~Sensor(){
	//destruct measurable first
	delete _m;
	delete _cm;
	_m = NULL;
	_cm = NULL;
	_amper.clear();
}