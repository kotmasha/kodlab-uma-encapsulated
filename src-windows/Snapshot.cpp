#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "Measurable.h"
#include "MeasurablePair.h"
/*
----------------Snapshot Base Class-------------------
*/
extern int ind(int row, int col);
extern int compi(int x);

Snapshot::Snapshot(int type, int base_sensor_size, double threshold, string name, vector<string> sensor_names,bool cal_target):
	_type(type),_base_sensor_size(base_sensor_size){
	_threshold = threshold;
	_memory_expansion = .5;
	//set the init value for 0.5 for now, ideally should read such value from conf file
	//TBD: read conf file for _memory_expansion
	_name = name;
	_sensor_num = 0;
	_total = 1.0;
	_total_ = 1.0;
	t = 0;
	this->cal_target = cal_target;
	init_pointers();
	init_data(sensor_names);
}


Snapshot::~Snapshot(){
	/*
	for(int i = 0; i < _sensors.size(); ++i){
		delete _sensors[i];
		_sensors[i] = NULL;
	}
	/*
	for(int i = 0; i < _sensor_pairs.size(); ++i){
		delete _sensor_pairs[i];
		_sensor_pairs[i] = NULL;
	}
	free_all_parameters();
	*/
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
}

float Snapshot::decide(vector<bool> signal, double phi, bool active){//the decide function
	_phi = phi;
	setObserve(signal);
	//if(t<200) active = true;
	update_state_GPU(active);
	halucinate_GPU();
	t++;
	//if(t<200) return 0;
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_d2, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	return distance(dev_d1, dev_d2);
}

void Snapshot::init_sensors(vector<string> uuid, vector<string> sensor_names){
	for(int i = 0; i < _sensor_size; ++i){
		try{
			Sensor *sensor = new Sensor(uuid[i], sensor_names[i], i);
			sensor->setMeasurableDiagPointers(h_diag, h_diag_);
			sensor->setMeasurableStatusPointers(h_current);
			_sensors.push_back(sensor);
		}
		catch(exception &e){
			cout<<e.what()<<endl;
			throw UMA_EXCEPTION::CORE_FATAL;
			//this is propabaly because the sensor name vector is not long enough, regard as fatal error and will abort test
		}
		//store the sensor pointer
	}
	_sensor_num = _sensor_size;
}

void Snapshot::init_sensor_pairs(){
	for(int i = 0; i < _sensor_size; ++i){
		for(int j = 0; j <= i; ++j){
			try{
				SensorPair *sensor_pair = new SensorPair(_sensors[i], _sensors[j], _threshold);
				sensor_pair->setWeightPointers(h_weights);
				sensor_pair->setDirPointers(h_dirs);
				sensor_pair->setThresholdPointers(h_thresholds);
				_sensor_pairs.push_back(sensor_pair);
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw UMA_EXCEPTION::CORE_FATAL;
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

	_sensor_size_max = (int)(_sensor_size * (1 + _memory_expansion));
	_measurable_size_max = 2 * _sensor_size_max;
	_sensor2d_size_max = _sensor_size_max * (_sensor_size_max + 1) / 2;
	_measurable2d_size_max = _measurable_size_max * (_measurable_size_max + 1) / 2;
	_mask_amper_size_max = _sensor_size_max * (_sensor_size_max + 1);
}

void Snapshot::init_data(vector<string> sensor_names){
	init_size(_base_sensor_size);
	//init the sensor size

	gen_weight();
	gen_direction();
	gen_thresholds();
	gen_mask_amper();
	gen_other_parameters();

	init_weight();
	init_direction();
	init_thresholds();
	init_mask_amper();
	init_other_parameter();

	init_sensors(vector<string>(sensor_names.size(), ""), sensor_names);
	init_sensor_pairs();
}

void Snapshot::reallocate_memory(int sensor_size){
	free_all_parameters();

	init_size(sensor_size);
	gen_weight();
	gen_direction();
	gen_thresholds();
	gen_mask_amper();
	gen_other_parameters();

	init_weight();
	init_direction();
	init_thresholds();
	init_mask_amper();
	init_other_parameter();
}

void Snapshot::copy_sensor_pair(int start_idx, int end_idx){
	for(int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i){
		_sensor_pairs[i]->setAllPointers(h_weights, h_dirs, h_thresholds);
		_sensor_pairs[i]->values_to_pointers();
	}
	cudaMemcpy(dev_weights + ind(2 * start_idx, 0), h_weights + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dirs + ind(2 * start_idx, 0), h_dirs + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_thresholds + ind(start_idx, 0), h_thresholds + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
}

void Snapshot::copy_sensor(int start_idx, int end_idx){
	for(int i = start_idx; i < end_idx; ++i){
		_sensors[i]->copyAmperList(h_mask_amper);
		_sensors[i]->setMeasurableDiagPointers(h_diag, h_diag_);
		_sensors[i]->setMeasurableStatusPointers(h_current);
		_sensors[i]->setMeasurableValuestoPointers();
	}
	cudaMemcpy(dev_mask_amper + 2 * ind(start_idx, 0), h_mask_amper + 2 * ind(start_idx, 0), 2 * (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag + start_idx, h_diag + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag_ + start_idx, h_diag_ + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
}

void Snapshot::copy_mask(vector<bool> mask){
	for(int i = 0; i < mask.size(); ++i) h_mask[i] = mask[i];
	cudaMemcpy(dev_mask, h_mask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
The function is pruning the sensor and sensor pair list, also adjust their corresponding idx
Input: the signal of all measurable
*/
void Snapshot::pruning(vector<bool> signal){
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
				_sensor_pairs[i - total_escape] = _sensor_pairs[i];
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
	copy_sensor(0, _sensor_num);
	copy_sensor_pair(0, _sensor_num);
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

void Snapshot::ampers(vector<vector<bool> > lists){
	cudaMemcpy(h_diag, dev_diag, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_diag_, dev_diag_, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _sensors.size(); ++i){
		//bring all sensor and measurable info into the object
		_sensors[i]->setMeasurablePointerstoValues();
	}
	cudaMemcpy(h_weights, dev_weights, _measurable2d_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_thresholds, dev_thresholds, _sensor2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _sensor_pairs.size(); ++i){
		//bring all sensor pair info into the object
		_sensor_pairs[i]->pointers_to_values();
	}
	int success_amper = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		amper(list);
		success_amper++;
	}

	if(_sensor_num > _sensor_size_max){
		//if need to reallocate
		reallocate_memory(_sensor_num);
		//copy every sensor back, since the memory is new
		copy_sensor(0, _sensor_num);
		copy_sensor_pair(0, _sensor_num);
	}
	else{
		//else just update the actual size
		_sensor_size = _sensor_num;
		_measurable_size = 2 * _sensor_size;
		_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
		_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
		_mask_amper_size = _sensor_size * (_sensor_size + 1);
		//copy just the new added sensors and sensor pairs
		copy_sensor(_sensor_num - success_amper, _sensor_num);
		copy_sensor_pair(_sensor_num - success_amper, _sensor_num);
	}
}

void Snapshot::amper(vector<int> &list){
	if(list.size() < 2) return;
	try{
		amperand(list[1], list[0], true);
		_sensor_num++;
		for(int j = 2; j < list.size(); ++j){
			amperand(_sensors.back()->_m->_idx, list[j], false);
		}
	}
	catch(exception &e){
		cout<<e.what()<<endl;
		throw UMA_EXCEPTION::CORE_ERROR;
	}
}

void Snapshot::delays(vector<vector<bool> > lists){
	cudaMemcpy(h_diag, dev_diag, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_diag_, dev_diag_, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _sensors.size(); ++i){
		//bring all sensor and measurable info into the object
		_sensors[i]->setMeasurablePointerstoValues();
	}
	cudaMemcpy(h_weights, dev_weights, _measurable2d_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_thresholds, dev_thresholds, _sensor2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for(int i = 0; i < _sensor_pairs.size(); ++i){
		//bring all sensor pair info into the object
		_sensor_pairs[i]->pointers_to_values();
	}
	int success_delay = 0;
	//record how many delay are successful
	
	for(int i = 0; i < lists.size(); ++i){
		vector<int> list = convert_list(lists[i]);
		amper(list);
		if(list.size() == 0) continue;
		if(list.size() == 1){
			try{
				generate_delayed_weights(list[0], true);
				_sensor_num++;
				success_delay++;
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw UMA_EXCEPTION::CORE_ERROR;
			}
		}
		else{
			try{
				generate_delayed_weights(_sensors.back()->_m->_idx, false);
				success_delay++;
			}
			catch(exception &e){
				cout<<e.what()<<endl;
				throw UMA_EXCEPTION::CORE_ERROR;
			}
		}
	}
	//cout<<_sensor_num<<","<<_sensor_size_max<<_name<<","<<success_delay<<endl;
	if(_sensor_num > _sensor_size_max){
		//if need to reallocate
		reallocate_memory(_sensor_num);
		//copy every sensor back, since the memory is new
		copy_sensor(0, _sensor_num);
		copy_sensor_pair(0, _sensor_num);
	}
	else{
		//else just update the actual size
		_sensor_size = _sensor_num;
		_measurable_size = 2 * _sensor_size;
		_sensor2d_size = _sensor_size * (_sensor_size + 1) / 2;
		_measurable2d_size = _measurable_size * (_measurable_size + 1) / 2;
		_mask_amper_size = _sensor_size * (_sensor_size + 1);
		//copy just the new added sensors and sensor pairs
		copy_sensor(_sensor_num - success_delay, _sensor_num);
		copy_sensor_pair(_sensor_num - success_delay, _sensor_num);
	}
}

/*
This function is generating dir matrix memory both on host and device
*/
void Snapshot::gen_direction(){
	h_dirs = new bool[_measurable2d_size_max];
	
	cudaMalloc(&dev_dirs, _measurable2d_size_max * sizeof(bool));
}

/*
This function is generating weight matrix memory both on host and device
*/
void Snapshot::gen_weight(){
	h_weights = new double[_measurable2d_size_max];
	
	cudaMalloc(&dev_weights, _measurable2d_size_max * sizeof(double));
}

/*
This function is generating threshold matrix memory both on host and device
*/
void Snapshot::gen_thresholds(){
	h_thresholds = new double[_sensor2d_size_max];
	cudaMalloc(&dev_thresholds, _sensor2d_size_max * sizeof(double));
}

/*
This function is generating mask amper matrix memory both on host and device
*/
void Snapshot::gen_mask_amper(){
	h_mask_amper = new bool[_mask_amper_size_max];

	cudaMalloc(&dev_mask_amper, _mask_amper_size_max * sizeof(bool));
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
	cudaMemcpy(dev_dirs, h_dirs, _measurable2d_size_max * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
This function init all weight matrix value to be 0.0, and copy it to device
*/
void Snapshot::init_weight(){
	for(int i = 0; i < _measurable2d_size_max; ++i){
		h_weights[i] = _total / 4.0;
	}
	cudaMemcpy(dev_weights, h_weights, _measurable2d_size_max * sizeof(double), cudaMemcpyHostToDevice);
}

/*
This function init all threshold matrix value to be the threshold input
*/
void Snapshot::init_thresholds(){
	for(int i = 0; i < _sensor2d_size_max; ++i){
		h_thresholds[i] = _threshold;
	}
	cudaMemcpy(dev_thresholds, h_thresholds, _sensor2d_size_max * sizeof(double), cudaMemcpyHostToDevice);
}

/*
This function init all mask amper matrix value to be false
*/
void Snapshot::init_mask_amper(){
	for(int i = 0; i < _mask_amper_size_max; ++i) h_mask_amper[i] = false;
	cudaMemcpy(dev_mask_amper, h_mask_amper, _mask_amper_size_max * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
This function generate other parameter
*/
void Snapshot::gen_other_parameters(){
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
}

//*********************************The folloing get/set function may change under REST Call infra***********************

/*
------------------------------------SET FUNCTION------------------------------------
*/
void Snapshot::setTarget(vector<bool> signal){
	for(int i = 0; i < _measurable_size; ++i){
		h_target[i] = signal[i];
	}
	cudaMemcpy(dev_target, h_target, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
This function set observe signal from python side
*/
void Snapshot::setObserve(vector<bool> observe){//this is where data comes in in every frame
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

string Snapshot::getName(){
	return _name;
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
/*
------------------------------------GET FUNCTION------------------------------------
*/

/*
This is the amper and function, used in amper and delay
Input: m_idx1, m_idx2 are the measurable idx that need to be amperand, m_idx1 > m_idx2, merge is indicating whether merge or replace the last sensor/row of sensor pair
*/
void Snapshot::amperand(int m_idx1, int m_idx2, bool merge){
	vector<SensorPair*> amper_and_sensor_pairs;
	Sensor *amper_and_sensor = new Sensor("", "", _sensor_num);
	double f = 1.0;
	if(_total > 0 && _total < 1e-5)
		f = getMeasurablePair(m_idx1, m_idx2)->v_w / _total;
	for(int i = 0; i < _sensors.size(); ++i){
		SensorPair *sensor_pair = NULL;
		sensor_pair = new SensorPair(amper_and_sensor, _sensors[i], _threshold);
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
	SensorPair *self_pair = new SensorPair(amper_and_sensor, amper_and_sensor, _threshold);
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
void Snapshot::generate_delayed_weights(int mid, bool merge){
	//create a new delayed sensor
	Sensor *delayed_sensor = new Sensor("", "", _sensor_num);
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
			sensor_pair = new SensorPair(delayed_sensor, delayed_sensor, _threshold);
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
			sensor_pair = new SensorPair(delayed_sensor, _sensors[i], _threshold);
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
    update_weights(activity);
	calculate_total(activity);

	// SIQI: update the thresholds:
	update_thresholds();
	// compute the derived orientation matrix and update the thresholds:
	orient_all();

	//SIQI:here I have intervened to disconnect the automatic computation of a target. Instead, I will be setting the target externally (from the Python side) at the beginning of each state-update cycle.
	// compute the target state:
	if(cal_target) calculate_target();

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
}
/*
----------------Snapshot Base Class-------------------
*/

/*
----------------Snapshot_Stationary Class-------------------
*/

Snapshot_Stationary::Snapshot_Stationary(int base_sensor_size, double threshold, string name, vector<string> sensor_names, double q, bool cal_target)
	:Snapshot(STATIONARY, base_sensor_size, threshold, name, sensor_names, cal_target){
	_q = q;
}

Snapshot_Stationary::~Snapshot_Stationary(){}

/*
----------------Snapshot_Stationary Class-------------------
*/


/*
----------------Snapshot_Forgetful Class-------------------
*/

Snapshot_Forgetful::Snapshot_Forgetful(int base_sensor_size, double threshold, string name, vector<string> sensor_names, double q, bool cal_target)
	:Snapshot(FORGETFUL, base_sensor_size, threshold, name, sensor_names, cal_target){
	_q = q;
}

Snapshot_Forgetful::~Snapshot_Forgetful(){}

/*
----------------Snapshot_Forgetful Class-------------------
*/

/*
----------------Snapshot_UnitTest Class--------------------
*/

Snapshot_UnitTest::Snapshot_UnitTest(int base_sensor_size, double threshold, double q)
	:Snapshot(UNITTEST, base_sensor_size, threshold, "test", vector<string>(base_sensor_size, "test_name"), false){
	_q = q;
}

Snapshot_UnitTest::~Snapshot_UnitTest(){}
/*
----------------Snapshot_UnitTest Class--------------------
*/
