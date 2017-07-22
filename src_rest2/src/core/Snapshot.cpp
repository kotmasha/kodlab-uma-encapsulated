#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "Measurable.h"
#include "MeasurablePair.h"
#include "logManager.h"
/*
----------------Snapshot Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);

Snapshot::Snapshot(ifstream &file, string &log_dir) {
	//write uuid
	int uuid_length = -1;
	file.read((char *)(&uuid_length), sizeof(int));
	_uuid = string(uuid_length, ' ');
	file.read(&_uuid[0], uuid_length * sizeof(char));
	_log_dir = log_dir + _uuid;
	_log = new logManager(logging::VERBOSE, _log_dir, _uuid + ".txt", typeid(*this).name());
	_memory_expansion = .5;

	_log->info() << "Default memory expansion rate is " + to_string(_memory_expansion);

	//set the init value for 0.5 for now, ideally should read such value from conf file
	//TBD: read conf file for _memory_expansion
	_sensor_num = 0;
	_total = 1.0;
	_total_ = 1.0;
	_q = 0.9;
	_threshold = 0.125;
	_auto_target = false;
	_log->debug() << "Setting init total value to " + to_string(_total);

	init_pointers();

	int sensor_size = -1;
	int sensor_pair_size = -1;
	file.read((char *)(&sensor_size), sizeof(int));
	file.read((char *)(&sensor_pair_size), sizeof(int));
	_sensor_num = sensor_size;
	for (int i = 0; i < sensor_size; ++i) {
		Sensor *sensor = new Sensor(file);
		_sensor_idx[sensor->_m->_uuid] = sensor;
		_sensor_idx[sensor->_cm->_uuid] = sensor;
		_sensors.push_back(sensor);
	}
	for (int i = 0; i < sensor_pair_size; ++i) {
		SensorPair *sensor_pair = new SensorPair(file, _sensors);
		_sensor_pairs.push_back(sensor_pair);
	}

	_log->info() << "A Snapshot " + _uuid + " is loaded";
}

Snapshot::Snapshot(string uuid, string log_dir){
	_uuid = uuid;
	_log_dir = log_dir;
	_log = new logManager(logging::VERBOSE, log_dir, uuid + ".txt", typeid(*this).name());
	_memory_expansion = .5;

	_log->info() << "Default memory expansion rate is " + to_string(_memory_expansion);

	//set the init value for 0.5 for now, ideally should read such value from conf file
	//TBD: read conf file for _memory_expansion
	_sensor_num = 0;
	_total = 1.0;
	_total_ = 1.0;
	_q = 0.9;
	_threshold = 0.125;
	_auto_target = false;
	_log->debug() << "Setting init total value to " + to_string(_total);

	init_pointers();
	_log->info() << "A Snapshot " + _uuid + " is created";
}


Snapshot::~Snapshot(){

}

void Snapshot::init_pointers(){
	h_dirs = NULL;
	h_weights = NULL;
	h_thresholds = NULL;
	h_mask_amper = NULL;
	h_observe = NULL;
	h_signal = NULL;
	h_load = NULL;
	h_current = NULL;
	h_mask = NULL;
	h_target = NULL;
	h_prediction = NULL;
	h_diag = NULL;
	h_diag_ = NULL;
	h_up = NULL;
	h_down = NULL;

	dev_dirs = NULL;
	dev_weights = NULL;
	dev_thresholds = NULL;
	dev_mask_amper = NULL;
	dev_observe = NULL;
	dev_observe_ = NULL;
	dev_signal = NULL;
	dev_load = NULL;
	dev_current = NULL;
	dev_mask = NULL;
	dev_target = NULL;
	dev_diag = NULL;
	dev_diag_ = NULL;
	dev_d1 = NULL;
	dev_d2 = NULL;

	_log->debug() << "Setting all pointers to NULL";
}

float Snapshot::decide(vector<bool> &signal, double phi, bool active){//the decide function
	_phi = phi;
	setObserve(signal);
	if(t<200) active = true;
	update_state_GPU(active);
	halucinate_GPU();
	t++;
	if(t<200) return 0;
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_d2, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	_log->debug() << "finished the " + to_string(t) + " decision";
	return distance(dev_d1, dev_d2);
}

bool Snapshot::add_sensor(std::pair<string, string> &id_pair) {
	if (_sensor_idx.find(id_pair.first) != _sensor_idx.end() && _sensor_idx.find(id_pair.second) != _sensor_idx.end()) {
		_log->error() << "Cannot create a duplicate sensor!";
		return false;
	}
	Sensor *sensor = new Sensor(id_pair, _sensor_num++);
	_sensor_idx[id_pair.first] = sensor;
	_sensor_idx[id_pair.second] = sensor;
	_sensors.push_back(sensor);
	_log->debug() << "A Sensor " + id_pair.first + " is created with idx " + to_string(sensor->_idx);
	//creating sensor pairs
	for (int i = 0; i < _sensor_num; ++i) {
		SensorPair *sensor_pair = new SensorPair(sensor, _sensors[i], _threshold, _total);
		_log->debug() << "A sensor pair with sensor1=" + sensor->_uuid + " sensor2=" + _sensors[i]->_uuid + " is created";
		_sensor_pairs.push_back(sensor_pair);
	}
	_log->info() << to_string(_sensor_num) + " Sensor Pairs are created, total is " + to_string(ind(_sensor_num, 0));

	//TBD, resize after add sensor
	return true;
}

/*
This function is doing validation after adding sensors
*/
bool Snapshot::validate(int base_sensor_size) {
	_log->info() << "start data validation";
	free_all_parameters();
	init_size(_sensor_num);

	_base_sensor_size = base_sensor_size;
	_log->info() << "base sensor size is: " + to_string(_base_sensor_size);

	gen_weight();
	gen_direction();
	gen_thresholds();
	gen_mask_amper();

	gen_other_parameters();
	init_other_parameter();

	create_sensors_to_arrays_index(0, _sensor_num);
	create_sensor_pairs_to_arrays_index(0, _sensor_num);
	copy_sensors_to_arrays(0, _sensor_num);
	copy_sensor_pairs_to_arrays(0, _sensor_num);

	try {
		init_sensors();
		init_sensor_pairs();
	}
	catch (exception &e) {
		_log->error() << "Errors init sensors " + string(e.what());
		return false;
	}
	_log->info() << "Snapshot data validation succeed";
	return true;
}

void Snapshot::init_sensors(){
	for(int i = 0; i < _sensor_size; ++i){
		try{
			Sensor *sensor = _sensors[i];
			sensor->setMeasurableDiagPointers(h_diag, h_diag_);
			sensor->setMeasurableStatusPointers(h_current);
		}
		catch(exception &e){
			cout<<e.what()<<endl;
			throw CORE_EXCEPTION::CORE_FATAL;
			//this is propabaly because the sensor name vector is not long enough, regard as fatal error and will abort test
		}
		//store the sensor pointer
	}
}

void Snapshot::init_sensor_pairs(){
	for(int i = 0; i < _sensor_size; ++i){
		for(int j = 0; j <= i; ++j){
			try{
				SensorPair *sensor_pair = _sensor_pairs[ind(i, j)];
				sensor_pair->setWeightPointers(h_weights);
				sensor_pair->setDirPointers(h_dirs);
				sensor_pair->setThresholdPointers(h_thresholds);
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw CORE_EXCEPTION::CORE_FATAL;
				//probably because sensors is not init correctly, regarded as fatal error
			}
		}
	}
}

void Snapshot::init_size(int sensor_size){
	_sensor_size = sensor_size;
	_measurable_size = 2 * _sensor_size;
	_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
	_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
	_mask_amper_size = _sensor_size * (_sensor_size + 1);

	_log->info() << "Setting sensor size to " + to_string(_sensor_size);
	_log->debug() << "Setting measurable size to " + to_string(_measurable_size);
	_log->debug() << "Setting sensor size 2D to " + to_string(_sensor2d_size);
	_log->debug() << "Setting measurable size 2D to " + to_string(_measurable2d_size);
	_log->debug() << "Setting mask amper size to " + to_string(_mask_amper_size);

	_sensor_size_max = (int)(_sensor_size * (1 + _memory_expansion));
	_measurable_size_max = 2 * _sensor_size_max;
	_sensor2d_size_max = _sensor_size_max * (_sensor_size_max + 1) / 2;
	_measurable2d_size_max = _measurable_size_max * (_measurable_size_max + 1) / 2;
	_mask_amper_size_max = _sensor_size_max * (_sensor_size_max + 1);

	_log->debug() << "Setting max sensor size to " + to_string(_sensor_size_max);
	_log->debug() << "Setting max measurable size to " + to_string(_measurable_size_max);
	_log->debug() << "Setting max sensor size 2D to " + to_string(_sensor2d_size_max);
	_log->debug() << "Setting max measurable size 2D to " + to_string(_measurable2d_size_max);
	_log->debug() << "Setting max mask amper size to " + to_string(_mask_amper_size_max);
}

void Snapshot::reallocate_memory(int sensor_size){
	free_all_parameters();

	init_size(sensor_size);
	gen_weight();
	gen_direction();
	gen_thresholds();
	gen_mask_amper();

	gen_other_parameters();
	init_other_parameter();
}

/*
This function is copying the snapshot data to the current one, but only copying the sensor/sensorpair that already exist in current snapshot
*/
void Snapshot::copy_test_data(Snapshot *snapshot) {
	for (int i = 0; i < _sensor_num; ++i) {
		//the current sensor to look at
		Sensor *sensor = _sensors[i];
		//the current sensor id
		string sensor_id = sensor->_uuid;
		if (snapshot->_sensor_idx.find(sensor_id) != snapshot->_sensor_idx.end()) {
			//if the current sensor id also in the other snapshot

			//copy sensors
			//get the 'same' sensor in the other snapshot
			Sensor *c_sensor = snapshot->_sensor_idx[sensor_id];
			//copy the value
			sensor->copy_data(c_sensor);
			//reconstruct the amper list
			vector<int> amper_list;
			for (int i = 0; i < c_sensor->_amper.size(); ++i) {
				int idx = c_sensor->_amper[i];
				bool isOriginPure = (idx % 2 == 0);
				Sensor *a_sensor = _sensors[idx / 2];
				if (_sensor_idx.find(a_sensor->_uuid) != _sensor_idx.end()) {
					if (isOriginPure) {
						amper_list.push_back(_sensor_idx[a_sensor->_uuid]->_m->_idx);
						_log->debug() << "found the amper sensor(" + a_sensor->_uuid + ") in Pure position in new test";
					}
					else {
						amper_list.push_back(_sensor_idx[a_sensor->_uuid]->_cm->_idx);
						_log->debug() << "found the amper sensor(" + a_sensor->_uuid + ") in Complementary position in new test";
					}
				}
				else {
					_log->debug() << "Cannot found the amper sensor(" + a_sensor->_uuid + ") in new test";
				}
			}
			//set the amper list
			sensor->setAmperList(amper_list);
			_log->debug() << "Sensor(" + sensor_id + ") data are copied";

			//copy sensor pairs
			for (int j = 0; j <= i; ++j) {
				//go through all the sensor pair list of the current sensor
				//the current other sensor
				Sensor *s = _sensors[j];
				if (snapshot->_sensor_idx.find(s->_uuid) != snapshot->_sensor_idx.end()) {
					//if the other sensor id also in the other snapshot
					//get the current sensor pair
					SensorPair *sensor_pair = _sensor_pairs[ind(i, j)];
					//get the other 'same' sensor
					Sensor *c_s = snapshot->_sensor_idx[s->_uuid];
					//get the corresponding sensorpair in the other snapshot, the idx of the sensorpair in the other one maybe different
					SensorPair *c_sensor_pair = snapshot->getSensorPair(c_sensor, c_s);
					//copy the data
					sensor_pair->copy_data(c_sensor_pair);
					_log->debug() << "Sensor Pair(sensor1=" + sensor_id + ", sensor2=" + s->_uuid + ") are copied";
				}
				else {
					_log->debug() << "Cannot find the sensor(" + s->_uuid + ") while copying sensor pair, must be a new sensor";
				}
			}//j
		}//if
		else {
			_log->debug() << "Cannot find the sensor(" + sensor_id + ") while copying sensor, must be a new sensor";
		}
	}//i
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/
void Snapshot::copy_arrays_to_sensors(int start_idx, int end_idx) {
	//copy necessary info back from cpu array to GPU array first
	cudaMemcpy(h_diag + start_idx, dev_diag + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_diag_ + start_idx, dev_diag_ + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyDeviceToHost);
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from GPU arrays to CPU arrays";
	for (int i = start_idx; i < end_idx; ++i) {
		//bring all sensor and measurable info into the object
		_sensors[i]->pointers_to_values();
	}
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from cpu arrays to sensor";
	_log->info() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied back from arrays";
}

void Snapshot::copy_sensors_to_arrays(int start_idx, int end_idx) {
	//copy necessary info from sensor object to cpu arrays
	for (int i = start_idx; i < end_idx; ++i) {
		_sensors[i]->copyAmperList(h_mask_amper);
		_sensors[i]->values_to_pointers();
	}
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from sensor to CPU arrays";
	//copy data from cpu array to GPU array
	cudaMemcpy(dev_mask_amper + 2 * ind(start_idx, 0), h_mask_amper + 2 * ind(start_idx, 0), 2 * (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag + start_idx, h_diag + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag_ + start_idx, h_diag_ + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from CPU arrays to GPU arrays";
	_log->info() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied to arrays";
}

void Snapshot::create_sensors_to_arrays_index(int start_idx, int end_idx) {
	//create idx for diag and current
	for (int i = start_idx; i < end_idx; ++i) {
		_sensors[i]->setMeasurableDiagPointers(h_diag, h_diag_);
		_sensors[i]->setMeasurableStatusPointers(h_current);
	}
	_log->info() << "Sensor from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " have created idx to arrays";
}
/*
--------------------Sensor Object And Array Data Transfer---------------------
*/

/*
--------------------SensorPair Object And Array Data Transfer---------------------
*/
void Snapshot::copy_sensor_pairs_to_arrays(int start_idx, int end_idx){
	//copy data from sensor pair object to CPU arrays
	for(int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i){
		_sensor_pairs[i]->values_to_pointers();
	}
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from sensor pairs to CPU arrays";
	//copy data from CPU arrays to GPU arrays
	cudaMemcpy(dev_weights + ind(2 * start_idx, 0), h_weights + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dirs + ind(2 * start_idx, 0), h_dirs + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_thresholds + ind(start_idx, 0), h_thresholds + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to GPU arrays";
	_log->info() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied to arrays";
}

void Snapshot::copy_arrays_to_sensor_pairs(int start_idx, int end_idx) {
	//copy data from GPU arrays to CPU arrays
	cudaMemcpy(h_weights + ind(2 * start_idx, 0), dev_weights + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dirs + ind(2 * start_idx, 0), dev_dirs + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_thresholds + ind(start_idx, 0), dev_thresholds + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from GPU arrays to CPU arrays";
	//copy data from CPU arrays to sensor pairs
	for (int i = 0; i < _sensor_pairs.size(); ++i) {
		//bring all sensor pair info into the object
		_sensor_pairs[i]->pointers_to_values();
	}
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to sensor pairs";
	_log->info() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied back from arrays";
}

void Snapshot::create_sensor_pairs_to_arrays_index(int start_idx, int end_idx) {
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		_sensor_pairs[i]->setAllPointers(h_weights, h_dirs, h_thresholds);
	}
	_log->info() << "Sensor pairs from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " have created idx to arrays";
}
/*
--------------------SensorPair Object And Array Data Transfer---------------------
*/

void Snapshot::copy_mask(vector<bool> mask){
	for(int i = 0; i < mask.size(); ++i) h_mask[i] = mask[i];
	cudaMemcpy(dev_mask, h_mask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
The function is pruning the sensor and sensor pair list, also adjust their corresponding idx
Input: the signal of all measurable
*/
void Snapshot::pruning(vector<bool> signal){
	copy_arrays_to_sensors(0, _sensor_num);
	copy_arrays_to_sensor_pairs(0, _sensor_num);
	//get converted sensor list, from measurable signal
	vector<bool> sensor_list = convert_signal_to_sensor(signal);
	int row_escape = 0;
	int total_escape = 0;
	//destruct the corresponding sensors and sensor pairs
	for(int i = 0; i < _sensors.size(); ++i){
		if(sensor_list[i]){
			//delete the sensor if necessary
			delete _sensors[i];
			_sensors[i] = NULL;
			row_escape++;
			_sensor_num--;
		}
		else{
			//or just adjust the idx of the sensor, and change the position
			_sensors[i]->setIdx(i - row_escape);
			_sensors[i - row_escape] = _sensors[i];
		}
		//delete the row of sensor, or the col in a row, where ther other sensor is deleted
		for(int j = 0; j <= i; ++j){
			if(sensor_list[i] || sensor_list[j]){
				//delete the sensor pair if necessary
				delete _sensor_pairs[ind(i, j)];
				_sensor_pairs[ind(i, j)] = NULL;
				total_escape++;
			}
			else{
				//or just change the position
				_sensor_pairs[ind(i, j) - total_escape] = _sensor_pairs[ind(i, j)];
			}
		}
	}

	//earse the additional space
	_sensors.erase(_sensors.end() - row_escape, _sensors.end());
	_sensor_pairs.erase(_sensor_pairs.end() - total_escape, _sensor_pairs.end());
	//adjust the size variables
	_sensor_size = _sensor_num;
	_measurable_size = 2 * _sensor_size;
	_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
	_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
	_mask_amper_size = _sensor_size * (_sensor_size + 1);
	//copy just the sensors and sensor pairs

	create_sensors_to_arrays_index(0, _sensor_num);
	create_sensor_pairs_to_arrays_index(0, _sensor_num);
	copy_sensors_to_arrays(0, _sensor_num);
	copy_sensor_pairs_to_arrays(0, _sensor_num);

	_log->info() << "Pruning done successful";
}

vector<bool> Snapshot::convert_signal_to_sensor(vector<bool> &signal){
	vector<bool> result;
	for(int i = 0; i < signal.size() / 2; ++i){
		if(signal[2 * i] || signal[2 * i + 1]) result.push_back(true);
		else result.push_back(false);
	}
	return result;
}

/*
This function is converting the list, from bool to int
*/
vector<int> Snapshot::convert_list(vector<bool> &list){
	vector<int> converted_list;
	for(int i = 0; i < list.size(); ++i){
		if(list[i]) converted_list.push_back(i);
	}
	return converted_list;
}

void Snapshot::ampers(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs){
	copy_arrays_to_sensors(0, _sensor_num);
	copy_arrays_to_sensor_pairs(0, _sensor_num);
	int success_amper = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		amper(list, id_pairs[i]);
		success_amper++;
	}

	if(_sensor_num > _sensor_size_max){
		//if need to reallocate
		reallocate_memory(_sensor_num);
		//copy every sensor back, since the memory is new
		create_sensors_to_arrays_index(0, _sensor_num);
		create_sensor_pairs_to_arrays_index(0, _sensor_num);
		copy_sensors_to_arrays(0, _sensor_num);
		copy_sensor_pairs_to_arrays(0, _sensor_num);
	}
	else{
		//else just update the actual size
		_sensor_size = _sensor_num;
		_measurable_size = 2 * _sensor_size;
		_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
		_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
		_mask_amper_size = _sensor_size * (_sensor_size + 1);
		//copy just the new added sensors and sensor pairs
		create_sensors_to_arrays_index(_sensor_num - success_amper, _sensor_num);
		create_sensor_pairs_to_arrays_index(_sensor_num - success_amper, _sensor_num);
		copy_sensors_to_arrays(_sensor_num - success_amper, _sensor_num);
		copy_sensor_pairs_to_arrays(_sensor_num - success_amper, _sensor_num);
	}
}

void Snapshot::amper(vector<int> &list, std::pair<string, string> &uuid) {
	if(list.size() < 2) return;
	try{
		amperand(list[1], list[0], true, uuid);
		_sensor_num++;
		for(int j = 2; j < list.size(); ++j){
			amperand(_sensors.back()->_m->_idx, list[j], false, uuid);
		}
	}
	catch(exception &e){
		cout<<e.what()<<endl;
		throw CORE_EXCEPTION::CORE_ERROR;
	}
}

void Snapshot::delays(vector<vector<bool> > &lists, vector<std::pair<string, string> > &id_pairs) {
	copy_arrays_to_sensors(0, _sensor_num);
	copy_arrays_to_sensor_pairs(0, _sensor_num);
	int success_delay = 0;
	//record how many delay are successful
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		amper(list, id_pairs[i]);
		if(list.size() == 0) continue;
		if(list.size() == 1){
			try{
				generate_delayed_weights(list[0], true, id_pairs[i]);
				_sensor_num++;
				success_delay++;
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw CORE_EXCEPTION::CORE_ERROR;
			}
		}
		else{
			try{
				generate_delayed_weights(_sensors.back()->_m->_idx, false, id_pairs[i]);
				success_delay++;
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw CORE_EXCEPTION::CORE_ERROR;
			}
		}
		_log->info() << "A delayed sensor is generated " + id_pairs[i].first;
		string delay_list="";
		for(int j = 0; j < list.size(); ++j) delay_list += (to_string(list[j]) + ",");
		_log->verbose() << "The delayed sensor generated from " + delay_list;
	}
	if(_sensor_num > _sensor_size_max){
		//if need to reallocate
		reallocate_memory(_sensor_num);
		//copy every sensor back, since the memory is new
		create_sensors_to_arrays_index(0, _sensor_num);
		create_sensor_pairs_to_arrays_index(0, _sensor_num);
		copy_sensors_to_arrays(0, _sensor_num);
		copy_sensor_pairs_to_arrays(0, _sensor_num);
	}
	else{
		//else just update the actual size
		_sensor_size = _sensor_num;
		_measurable_size = 2 * _sensor_size;
		_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
		_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
		_mask_amper_size = _sensor_size * (_sensor_size + 1);
		//copy just the new added sensors and sensor pairs
		create_sensors_to_arrays_index(_sensor_num - success_delay, _sensor_num);
		create_sensor_pairs_to_arrays_index(_sensor_num - success_delay, _sensor_num);
		copy_sensors_to_arrays(_sensor_num - success_delay, _sensor_num);
		copy_sensor_pairs_to_arrays(_sensor_num - success_delay, _sensor_num);
	}
}

/*
This function is generating dir matrix memory both on host and device
*/
void Snapshot::gen_direction(){
	//malloc the space
	h_dirs = new bool[_measurable2d_size_max];
	cudaMalloc(&dev_dirs, _measurable2d_size_max * sizeof(bool));

	//fill in all 0
	memset(h_dirs, 0, _measurable2d_size_max * sizeof(bool));
	cudaMemset(dev_dirs, 0, _measurable2d_size_max * sizeof(bool));

	_log->debug() << "Dir matrix generated with size " + to_string(_measurable2d_size_max);
}

/*
This function is generating weight matrix memory both on host and device
*/
void Snapshot::gen_weight(){
	//malloc the space
	h_weights = new double[_measurable2d_size_max];
	cudaMalloc(&dev_weights, _measurable2d_size_max * sizeof(double));

	//fill in all 0
	memset(h_weights, 0, _measurable2d_size_max * sizeof(double));
	cudaMemset(dev_weights, 0, _measurable2d_size_max * sizeof(double));

	_log->debug() << "Weight matrix generated with size " + to_string(_measurable2d_size_max);
}

/*
This function is generating threshold matrix memory both on host and device
*/
void Snapshot::gen_thresholds(){
	//malloc the space
	h_thresholds = new double[_sensor2d_size_max];
	cudaMalloc(&dev_thresholds, _sensor2d_size_max * sizeof(double));

	//fill in all 0
	memset(h_thresholds, 0, _sensor2d_size_max * sizeof(double));
	cudaMemset(dev_thresholds, 0, _sensor2d_size_max * sizeof(double));

	_log->debug() << "Threshold matrix generated with the size " + to_string(_sensor2d_size_max);
}

/*
This function is generating mask amper matrix memory both on host and device
*/
void Snapshot::gen_mask_amper(){
	//malloc the space
	h_mask_amper = new bool[_mask_amper_size_max];
	cudaMalloc(&dev_mask_amper, _mask_amper_size_max * sizeof(bool));

	//fill in all 0
	memset(h_mask_amper, 0, _mask_amper_size_max * sizeof(bool));
	cudaMemset(dev_mask_amper, 0, _mask_amper_size_max * sizeof(bool));

	_log->debug() << "Mask amper matrix generated with size " + to_string(_mask_amper_size_max);
}

/*
This function generate other parameter
*/
void Snapshot::gen_other_parameters() {
	h_observe = new bool[_measurable_size_max];
	h_signal = new bool[_measurable_size_max];
	h_load = new bool[_measurable_size_max];
	h_mask = new bool[_measurable_size_max];
	h_current = new bool[_measurable_size_max];
	h_target = new bool[_measurable_size_max];
	h_diag = new double[_measurable_size_max];
	h_diag_ = new double[_measurable_size_max];
	h_prediction = new bool[_measurable_size_max];
	h_up = new bool[_measurable_size_max];
	h_down = new bool[_measurable_size_max];

	cudaMalloc(&dev_observe, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_observe_, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_load, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_mask, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_current, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_target, _measurable_size_max * sizeof(bool));

	cudaMalloc(&dev_d1, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_d2, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_diag, _measurable_size_max * sizeof(double));
	cudaMalloc(&dev_diag_, _measurable_size_max * sizeof(double));

	_log->debug() << "Other parameter generated, with size " + to_string(_measurable_size_max);
}

/*
This function init all dir matrix value to be 1 on diagonal, 0 otherwise
*/
void Snapshot::init_direction(){
	int x = 0, y = 0;
	for(int i = 0; i < _measurable2d_size_max; ++i){
		h_dirs[i] = (x == y);
		x++;
		if(x > y){
			y++;
			x = 0;
		}
	}
	_log->debug() << "Dir matrix initiated with value 1 on diagonal and 0 otherwise";
	cudaMemcpy(dev_dirs, h_dirs, _measurable2d_size_max * sizeof(bool), cudaMemcpyHostToDevice);
	_log->debug() << "Dir matrix value copied to GPU";
}

/*
This function init all weight matrix value to be 0.0, and copy it to device
*/
void Snapshot::init_weight(){
	for(int i = 0; i < _measurable2d_size_max; ++i){
		h_weights[i] = _total / 4.0;
	}
	_log->debug() << "Weight matrix initiated with value " + to_string(_total / 4.0);
	cudaMemcpy(dev_weights, h_weights, _measurable2d_size_max * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Weight matrix value copied to GPU";
}

/*
This function init all threshold matrix value to be the threshold input
*/
void Snapshot::init_thresholds(){
	for(int i = 0; i < _sensor2d_size_max; ++i){
		h_thresholds[i] = _threshold;
	}
	_log->debug() << "Threshold matrix initiated with value " + to_string(_threshold);
	cudaMemcpy(dev_thresholds, h_thresholds, _sensor2d_size_max * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Threshold matrix value copied to GPU";
}

/*
This function init all mask amper matrix value to be false
*/
void Snapshot::init_mask_amper(){
	for(int i = 0; i < _mask_amper_size_max; ++i) h_mask_amper[i] = false;
	_log->debug() << "Mask amper matrix initiated with value 0";
	cudaMemcpy(dev_mask_amper, h_mask_amper, _mask_amper_size_max * sizeof(bool), cudaMemcpyHostToDevice);
	_log->debug() << "Mask amper matrix value copied to GPU";
}


//*********************************The folloing get/set function may change under REST Call infra***********************

/*
------------------------------------SET FUNCTION------------------------------------
*/
void Snapshot::setThreshold(double &threshold) {
	_threshold = threshold;
	_log->info() << "snapshot threshold changed to " + to_string(threshold);
}

void Snapshot::setQ(double &q) {
	_q = q;
	_log->info() << "snapshot q changed to " + to_string(q);
}

void Snapshot::setTarget(vector<bool> &signal){
	for(int i = 0; i < _measurable_size; ++i){
		h_target[i] = signal[i];
	}
	cudaMemcpy(dev_target, h_target, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void Snapshot::setAutoTarget(bool auto_target) {
	_auto_target = auto_target;
}

/*
This function set observe signal from python side
*/
void Snapshot::setObserve(vector<bool> &observe){//this is where data comes in in every frame
	_log->debug() << "Setting observe signal";
	for(int i = 0; i < observe.size(); ++i){
		h_observe[i] = observe[i];
	}
	cudaMemcpy(dev_observe_, dev_observe, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_observe, h_observe, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}
/*
------------------------------------SET FUNCTION------------------------------------
*/

/*
------------------------------------GET FUNCTION------------------------------------
*/
//get the current(observe value through propagation) value from GPU
vector<bool> Snapshot::getCurrent(){
	vector<bool> result;
	cudaMemcpy(h_current, dev_current, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _measurable_size; ++i){	
		result.push_back(h_current[i]);
	}
	return result;
}

//get the prediction(next iteration prediction) value from GPU
vector<bool> Snapshot::getPrediction(){
	vector<bool> result;
	//no corresponding dev variable to copy from, should be copied after the halucinate
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_prediction[i]);
	}
	return result;
}

/*
The function is getting the current target
*/
vector<bool> Snapshot::getTarget(){
	vector<bool> result;
	cudaMemcpy(h_target, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_target[i]);
	}
	return result;
}

/*
The function is getting weight matrix in 2-dimensional form
*/
vector<vector<double> > Snapshot::getWeight2D(){
	cudaMemcpy(h_weights, dev_weights, _measurable2d_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for(int i = 0; i < _measurable_size; ++i){
		vector<double> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(h_weights[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the dir matrix in 2-dimensional form
*/
vector<vector<bool> > Snapshot::getDir2D(){
	cudaMemcpy(h_dirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<bool> > result;
	int n = 0;
	for(int i = 0; i < _measurable_size; ++i){
		vector<bool> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(h_dirs[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the threshold matrix in 2-dimensional form
*/
vector<vector<double> > Snapshot::getThreshold2D(){
	cudaMemcpy(h_thresholds, dev_thresholds, _sensor2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for(int i = 0; i < _sensor_size; ++i){
		vector<double> tmp;
		for(int j = 0; j <= i; ++j)
			tmp.push_back(h_thresholds[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the mask amper in 2-dimension form
*/
vector<vector<bool> > Snapshot::getMask_amper2D(){
	vector<vector<bool> > result;
	cudaMemcpy(h_mask_amper, dev_mask_amper, _mask_amper_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int n = 0;
	for(int i = 0; i < _sensor_size; ++i){
		vector<bool> tmp;
		for(int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(h_mask_amper[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting mask amper as an list
*/
vector<bool> Snapshot::getMask_amper(){
	vector<vector<bool> > tmp = getMask_amper2D();
	vector<bool> results;
	for(int i = 0; i < tmp.size(); ++i){
		for(int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the weight matrix as a list
*/
vector<double> Snapshot::getWeight(){
	vector<vector<double > > tmp = getWeight2D();
	vector<double> results;
	for(int i = 0; i < tmp.size(); ++i){
		for(int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the dir matrix as a list
*/
vector<bool> Snapshot::getDir(){
	vector<vector<bool> > tmp = this->getDir2D();
	vector<bool> results;
	for(int i = 0; i < tmp.size(); ++i){
		for(int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the threshold matrix as a list
*/
vector<double> Snapshot::getThreshold(){
	vector<vector<double> > tmp = getThreshold2D();
	vector<double> results;
	for(int i = 0; i < tmp.size(); ++i){
		for(int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
This function is getting the diagonal value of the weight matrix
*/
vector<double> Snapshot::getDiag(){
	cudaMemcpy(h_diag, dev_diag, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_diag[i]);
	}
	return result;
}

/*
This function is getting the diagonal value of the weight matrix of last iteration
*/
vector<double> Snapshot::getDiagOld(){
	cudaMemcpy(h_diag_, dev_diag_, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_diag_[i]);
	}
	return result;
}

/*
This function is getting the current mask value used in halucination
*/
vector<bool> Snapshot::getMask(){
	vector<bool> result;
	cudaMemcpy(h_mask, dev_mask, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < this->_measurable_size; ++i) result.push_back(h_mask[i]);
	return result;
}

/*
This function is getting the obersev matrix
*/
vector<bool> Snapshot::getObserve(){
	vector<bool> result;
	cudaMemcpy(h_observe, dev_observe, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < this->_measurable_size; ++i) result.push_back(h_observe[i]);
	return result;
}

/*
This function is getting the obersev matrix of last iteration
*/
vector<bool> Snapshot::getObserveOld(){
	vector<bool> result;
	cudaMemcpy(h_observe, dev_observe_, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _measurable_size; ++i) result.push_back(h_observe[i]);
	return result;
}

/*
The function is getting the up matrix
*/
vector<bool> Snapshot::getUp(){
	vector<bool> result;
	//up have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_up[i]);
	}
	return result;
}

/*
The function is getting the down matrix
*/
vector<bool> Snapshot::getDown(){
	vector<bool> result;
	//down have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for(int i = 0; i < _measurable_size; ++i){
		result.push_back(h_down[i]);
	}
	return result;
}

/*
this function is getting the measurable, from the sensor list
*/
Measurable *Snapshot::getMeasurable(int idx){
	int s_idx = idx / 2;
	if(idx % 2 == 0){
		return _sensors[s_idx]->_m;
	}
	else{
		return _sensors[s_idx]->_cm;
	}
}

SensorPair *Snapshot::getSensorPair(Sensor *sensor1, Sensor *sensor2) {
	int idx1 = sensor1->_idx > sensor2->_idx ? sensor1->_idx : sensor2->_idx;
	int idx2 = sensor1->_idx > sensor2->_idx ? sensor2->_idx : sensor1->_idx;
	return _sensor_pairs[ind(idx1, idx2)];
}

/*
This function is getting the measurable pair from the sensor pair list
Input: m_idx1, m_idx2 are index of the measurable, m_idx1 > m_idx2
*/
MeasurablePair *Snapshot::getMeasurablePair(int m_idx1, int m_idx2){
	int s_idx1 = m_idx1 / 2;
	int s_idx2 = m_idx2 / 2;
	Measurable *m1 = getMeasurable(m_idx1);
	Measurable *m2 = getMeasurable(m_idx2);
	return _sensor_pairs[ind(s_idx1, s_idx2)]->getMeasurablePair(m2->_isOriginPure, m1->_isOriginPure);
}

vector<bool> Snapshot::getAmperList(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	Sensor *sensor = _sensor_idx[sensor_id];
	vector<bool> result(_measurable_size, false);
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		result[sensor->_amper[i]] = true;
	}
	return result;
}

vector<string> Snapshot::getAmperListID(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	Sensor *sensor = _sensor_idx[sensor_id];
	vector<string> result;
	for (int i = 0; i < sensor->_amper.size(); ++i) {
		int idx = sensor->_amper[i];
		Sensor *s = _sensors[idx / 2];
		if (idx % 2 == 0) result.push_back(s->_m->_uuid);
		else result.push_back(s->_cm->_uuid);
	}
	return result;
}

Sensor *Snapshot::getSensor(string &sensor_id) {
	if (_sensor_idx.find(sensor_id) == _sensor_idx.end()) {
		throw CLIENT_EXCEPTION::CLIENT_ERROR;
	}
	return _sensor_idx[sensor_id];
}
/*
------------------------------------GET FUNCTION------------------------------------
*/

/*
This is the amper and function, used in amper and delay
Input: m_idx1, m_idx2 are the measurable idx that need to be amperand, m_idx1 > m_idx2, merge is indicating whether merge or replace the last sensor/row of sensor pair
*/
void Snapshot::amperand(int m_idx1, int m_idx2, bool merge, std::pair<string, string> &id_pair) {
	vector<SensorPair*> amper_and_sensor_pairs;
	Sensor *amper_and_sensor = new Sensor(id_pair, _sensor_num);
	_sensor_idx[id_pair.first] = amper_and_sensor;
	_sensor_idx[id_pair.second] = amper_and_sensor;

	double f = 1.0;
	if(_total > 0 && _total < 1e-5)
		f = getMeasurablePair(m_idx1, m_idx2)->v_w / _total;
	for(int i = 0; i < _sensors.size(); ++i){
		SensorPair *sensor_pair = NULL;
		sensor_pair = new SensorPair(amper_and_sensor, _sensors[i], _threshold, _total);
		Measurable *m1 = _sensors[i]->_m;
		Measurable *m2 = _sensors[i]->_cm;
		//mij
		if(_sensors[i]->_m->_idx == m_idx1 || _sensors[i]->_m->_idx == m_idx2){
			sensor_pair->mij->v_w = getMeasurablePair(m_idx1, m_idx2)->v_w;
		}
		else{
			sensor_pair->mij->v_w = f * getMeasurablePair(m1->_idx, m1->_idx)->v_w;
		}
		//mi_j
		if(_sensors[i]->_cm->_idx == m_idx1 || _sensors[i]->_cm->_idx == m_idx2){
			sensor_pair->mi_j->v_w = 0.0;
		}
		else{
			sensor_pair->mi_j->v_w = f * getMeasurablePair(m2->_idx, m2->_idx)->v_w;
		}
		//m_ij
		if(_sensors[i]->_m->_idx == m_idx1){
			sensor_pair->m_ij->v_w = getMeasurablePair(m_idx1, compi(m_idx2))->v_w;
		}
		else if(_sensors[i]->_m->_idx == m_idx2){
			sensor_pair->m_ij->v_w = getMeasurablePair(compi(m_idx1), m_idx2)->v_w;
		}
		else{
			sensor_pair->m_ij->v_w = (1.0 - f) * getMeasurablePair(m1->_idx, m1->_idx)->v_w;
		}
		//m_i_j
		if(_sensors[i]->_cm->_idx == m_idx1){
			sensor_pair->m_i_j->v_w = getMeasurablePair(compi(m_idx1), compi(m_idx1))->v_w;
		}
		else if(_sensors[i]->_cm->_idx == m_idx2){
			sensor_pair->m_i_j->v_w = getMeasurablePair(compi(m_idx2), compi(m_idx2))->v_w;
		}
		else{
			sensor_pair->m_i_j->v_w = (1.0 - f) * getMeasurablePair(m2->_idx, m2->_idx)->v_w;
		}
		amper_and_sensor_pairs.push_back(sensor_pair);
	}
	SensorPair *self_pair = new SensorPair(amper_and_sensor, amper_and_sensor, _threshold, _total);
	self_pair->mij->v_w = amper_and_sensor_pairs[0]->mij->v_w + amper_and_sensor_pairs[0]->mi_j->v_w;
	self_pair->mi_j->v_w = 0.0;
	self_pair->m_ij->v_w = 0.0;
	self_pair->m_i_j->v_w = amper_and_sensor_pairs[0]->m_ij->v_w + amper_and_sensor_pairs[0]->m_i_j->v_w;
	amper_and_sensor_pairs.push_back(self_pair);
	if(!merge){
		Sensor *old_sensor = _sensors.back();
		//restore the old sensor amper list
		amper_and_sensor->setAmperList(old_sensor);
		//also append the m_idx2 as the new amper list value
		amper_and_sensor->setAmperList(m_idx2);
		//take the idx of the old one, copy it to the new one 
		amper_and_sensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensor_num - 1, 0); i < _sensor_pairs.size(); ++i){
			delete _sensor_pairs[i];
		}
		_sensor_pairs.erase(_sensor_pairs.begin() + ind(_sensor_num - 1, 0), _sensor_pairs.end());
		//also need to remove the n-1 position of amper_and_sensor_pairs
		delete amper_and_sensor_pairs[_sensor_num - 1];
		amper_and_sensor_pairs.erase(amper_and_sensor_pairs.end() - 2, amper_and_sensor_pairs.end() - 1);
	}
	else{
		//set the amper list, if append a new sensor, append the smaller idx first
		amper_and_sensor->setAmperList(m_idx2);
		amper_and_sensor->setAmperList(m_idx1);
	}
	_sensors.push_back(amper_and_sensor);
	_sensor_pairs.insert(_sensor_pairs.end(), amper_and_sensor_pairs.begin(), amper_and_sensor_pairs.end());
}

/*
This function is generating the delayed weights
Before this function is called, have to make sure, _sensors and _sensor_pairs have valid info in vwxx;
Input: mid of the measurable doing the delay, and whether to merge after the operation
*/
void Snapshot::generate_delayed_weights(int mid, bool merge, std::pair<string, string> &id_pair){
	//create a new delayed sensor
	Sensor *delayed_sensor = new Sensor(id_pair, _sensor_num);
	_sensor_idx[id_pair.first] = delayed_sensor;
	_sensor_idx[id_pair.second] = delayed_sensor;
	vector<SensorPair *> delayed_sensor_pairs;
	//the sensor name is TBD, need python input
	int delay_mid1 = mid, delay_mid2 = compi(mid);
	int sid = mid / 2;
	//get mid and sid

	bool is_sensor_active;
	if(merge){
		//if need to merge, means just a single sensor delay
		is_sensor_active = _sensors[sid]->isSensorActive();
	}
	else{
		//means not a single sensor delay
		cudaMemcpy(h_observe, dev_observe_, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
		is_sensor_active = _sensors[sid]->amper_and_signals(h_observe);
	}

	for(int i = 0; i < _sensor_num + 1; ++i){
		SensorPair *sensor_pair = NULL;
		if(i == _sensor_num){
			//if this is the last sensor pair, and it is the pair of the delayed sensor itself
			sensor_pair = new SensorPair(delayed_sensor, delayed_sensor, _threshold, _total);
			//copy all those diag values first
			delayed_sensor->_m->_vdiag = delayed_sensor_pairs[0]->mij->v_w + delayed_sensor_pairs[0]->mi_j->v_w;
			delayed_sensor->_cm->_vdiag = delayed_sensor_pairs[0]->m_ij->v_w + delayed_sensor_pairs[0]->m_i_j->v_w;
			delayed_sensor->_m->_vdiag_ = delayed_sensor->_m->_vdiag;
			delayed_sensor->_cm->_vdiag_ = delayed_sensor->_cm->_vdiag;
			//then assign the value to sensor pair
			sensor_pair->mij->v_w = delayed_sensor->_m->_vdiag;
			sensor_pair->mi_j->v_w = 0;
			sensor_pair->m_ij->v_w = 0;
			sensor_pair->m_i_j->v_w = delayed_sensor->_cm->_vdiag;
			delayed_sensor_pairs.push_back(sensor_pair);
		}
		else{
			sensor_pair = new SensorPair(delayed_sensor, _sensors[i], _threshold, _total);
			sensor_pair->mij->v_w = is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->mi_j->v_w = is_sensor_active * _sensors[i]->_cm->_vdiag;
			sensor_pair->m_ij->v_w = !is_sensor_active * _sensors[i]->_m->_vdiag;
			sensor_pair->m_i_j->v_w = !is_sensor_active * _sensors[i]->_cm->_vdiag;
			delayed_sensor_pairs.push_back(sensor_pair);
		}
	}

	if(!merge){
		//replace the last one
		Sensor *old_sensor = _sensors[sid];
		//set the mask amper of the delayed sensor to be the same as the one that need to be delayed
		delayed_sensor->setAmperList(old_sensor);
		//take the idx of the old one, copy it to the new one 
		delayed_sensor->setIdx(old_sensor->_idx);
		//delete the old one
		delete old_sensor;
		//if no mrege needed, means the last existing sensor and row of sensor_pair need to be removed
		_sensors.pop_back();
		//destruct the last row of sensor pair
		for(int i = ind(_sensor_num - 1, 0); i < _sensor_pairs.size(); ++i){
			delete _sensor_pairs[i];
		}
		_sensor_pairs.erase(_sensor_pairs.begin() + ind(_sensor_num - 1, 0), _sensor_pairs.end());
		//also need to remove the n-1 position of delayed_sensor_pair
		delete delayed_sensor_pairs[_sensor_num - 1];
		delayed_sensor_pairs.erase(delayed_sensor_pairs.end() - 2, delayed_sensor_pairs.end() - 1);
	}
	else{
		delayed_sensor->setAmperList(mid);
	}
	_sensors.push_back(delayed_sensor);
	_sensor_pairs.insert(_sensor_pairs.end(), delayed_sensor_pairs.begin(), delayed_sensor_pairs.end());
}

/*
This function update state on GPU, it is the main function for simulation on C++
It contains three main parts: update weight, orient all, propagation, result will be stored in Gload(propagate_GPU)
Input: mode to use
Output: None
*/
void Snapshot::update_state_GPU(bool activity){//true for decide	
    // udpate the snapshot weights and total count:
	_log->debug() << "start updating data on GPU";

    update_weights(activity);

	calculate_total(activity);

	// SIQI: update the thresholds:
	update_thresholds();
	// compute the derived orientation matrix and update the thresholds:
	orient_all();

	//SIQI:here I have intervened to disconnect the automatic computation of a target. Instead, I will be setting the target externally (from the Python side) at the beginning of each state-update cycle.
	// compute the target state:
	if(_auto_target) calculate_target();

	cudaMemcpy(dev_signal, dev_observe, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, _measurable_size * sizeof(bool));
	propagate_GPU(dev_signal, dev_load);

	cudaMemcpy(h_current, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(Gdir, dev_dir, whole_size * whole_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function is halucinate on GPU, it use several propagate_GPU
It first get the mask to use and then use propagate
The output will be stored in Gload(propagate_GPU)
Input: action list to be halucinated
Output: None
*/
void Snapshot::halucinate_GPU(){
	gen_mask();

	cudaMemcpy(dev_signal, dev_mask, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, _measurable_size * sizeof(bool));
	propagate_GPU(dev_signal, dev_load);
	//cudaMemcpy(Gload, dev_load, measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_prediction, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void Snapshot::free_all_parameters(){//free data in case of memory leak
	delete[] h_dirs;
	delete[] h_weights;
	delete[] h_thresholds;
	delete[] h_mask_amper;
	delete[] h_observe;
	delete[] h_signal;
	delete[] h_load;
	delete[] h_mask;
	delete[] h_current;
	delete[] h_target;
	delete[] h_diag;
	delete[] h_diag_;
	delete[] h_prediction;
	delete[] h_up;
	delete[] h_down;

	cudaFree(dev_dirs);
	cudaFree(dev_weights);
	cudaFree(dev_thresholds);
	cudaFree(dev_mask_amper);

	cudaFree(dev_mask);
	cudaFree(dev_current);
	cudaFree(dev_target);

	cudaFree(dev_observe);
	cudaFree(dev_observe_);
	cudaFree(dev_signal);
	cudaFree(dev_load);

	cudaFree(dev_d1);
	cudaFree(dev_d2);
	cudaFree(dev_diag);
	cudaFree(dev_diag_);

	_log->info() << "All memory released";
}

void Snapshot::save_snapshot(ofstream &file) {
	//write uuid
	int uuid_size = _uuid.length();
	file.write(reinterpret_cast<const char *>(&uuid_size), sizeof(int));
	file.write(_uuid.c_str(), uuid_size * sizeof(char));
	int sensor_size = _sensors.size();
	int sensor_pair_size = _sensor_pairs.size();
	file.write(reinterpret_cast<const char *>(&sensor_size), sizeof(int));
	file.write(reinterpret_cast<const char *>(&sensor_pair_size), sizeof(int));
	copy_arrays_to_sensors(0, sensor_size);
	copy_arrays_to_sensor_pairs(0, sensor_size);
	for (int i = 0; i < sensor_size; ++i) {
		_sensors[i]->save_sensor(file);
	}
	for (int i = 0; i < sensor_pair_size; ++i) {
		_sensor_pairs[i]->save_sensor_pair(file);
	}
}
/*
----------------Snapshot Base Class-------------------
*/

/*
----------------Snapshot_Stationary Class-------------------
*/
Snapshot_Stationary::Snapshot_Stationary(ifstream &file, string &log_dir):Snapshot(file, log_dir) {}

Snapshot_Stationary::Snapshot_Stationary(string uuid, string log_dir)
	:Snapshot(uuid, log_dir){
}

Snapshot_Stationary::~Snapshot_Stationary(){}

/*
----------------Snapshot_Stationary Class-------------------
*/


/*
----------------Snapshot_Forgetful Class-------------------
*/

Snapshot_Forgetful::Snapshot_Forgetful(string uuid, string log_dir)
	:Snapshot(uuid, log_dir){
}

Snapshot_Forgetful::~Snapshot_Forgetful(){}

/*
----------------Snapshot_Forgetful Class-------------------
*/

/*
----------------Snapshot_UnitTest Class--------------------
*/

Snapshot_UnitTest::Snapshot_UnitTest(string uuid, string log_dir)
	:Snapshot(uuid, log_dir){
}

Snapshot_UnitTest::~Snapshot_UnitTest(){}
/*
----------------Snapshot_UnitTest Class--------------------
*/
