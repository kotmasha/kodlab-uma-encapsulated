#include "DataManager.h"
#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "UMAException.h"
#include "logManager.h"
#include "uma_base.cuh"
#include "kernel_util.cuh"

extern int ind(int row, int col);
extern int compi(int x);

DataManager::DataManager(string &log_dir) {
	_log_dir = log_dir;
	_log = new logManager(logging::VERBOSE, log_dir, "DataManager.txt", typeid(*this).name());

	_memory_expansion = 0.5;
	_log->info() << "Setting the memory expansion rate to " + to_string(_memory_expansion);
}

void DataManager::init_pointers() {
	h_dirs = NULL;
	h_weights = NULL;
	h_thresholds = NULL;
	h_mask_amper = NULL;
	h_npdirs = NULL;
	h_observe = NULL;
	h_observe_ = NULL;
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
	h_npdir_mask = NULL;
	h_signals = NULL;
	h_dists = NULL;

	dev_dirs = NULL;
	dev_weights = NULL;
	dev_thresholds = NULL;
	dev_mask_amper = NULL;
	dev_npdirs = NULL;
	dev_observe = NULL;
	dev_signal = NULL;
	dev_load = NULL;
	dev_current = NULL;
	dev_mask = NULL;
	dev_target = NULL;
	dev_diag = NULL;
	dev_diag_ = NULL;
	dev_d1 = NULL;
	dev_d2 = NULL;
	dev_npdir_mask = NULL;
	dev_signals = NULL;
	dev_dists = NULL;

	_log->debug() << "Setting all pointers to NULL";
}

void DataManager::reallocate_memory(int sensor_size) {
	_log->info() << "Starting reallocating memory";
	free_all_parameters();

	set_size(sensor_size, true);

	try {
		gen_weight();
		gen_direction();
		gen_thresholds();
		gen_mask_amper();
		gen_np_direction();
		gen_signals();
		gen_npdir_mask();
		gen_dists();
		gen_other_parameters();
	}
	catch (CoreException &e) {
		_log->error() << "Fatal error in reallocate_memory when doing memory allocation";
		throw CoreException("Fatal error in reallocate_memory when doing memory allocation", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
	}

	try {
		init_other_parameter(1.0);
	}
	catch (CoreException &e) {
		_log->error() << "Fatal error in reallocate_memory when doing parameters init";
		throw CoreException("Fatal error in reallocate_memory when doing parameters ini", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
	}
	_log->info() << "Memory reallocated!";
}


void DataManager::set_size(int sensor_size, bool change_max = true) {
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

	if (change_max) {
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
	else _log->info() << "All size max value remain the same";
}


/*
This function free data together
Deprecated in new version
Input: None
Output: None
*/
void DataManager::free_all_parameters() {//free data in case of memory leak
	_log->info() << "Starting releasing memory";
	try {
		delete[] h_dirs;
		delete[] h_weights;
		delete[] h_thresholds;
		delete[] h_mask_amper;
		delete[] h_observe;
		delete[] h_observe_;
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
		delete[] h_npdirs;
		delete[] h_npdir_mask;
		delete[] h_signals;
		delete[] h_dists;
		delete[] h_union_root;
	}
	catch (CoreException &e) {
		_log->error() << "Fatal error in free_all_parameters, when doing cpu array release";
		throw CoreException("Fatal error in free_all_parameters, when doing cpu array release", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
	}

	try {
		cudaFree(dev_dirs);
		cudaFree(dev_weights);
		cudaFree(dev_thresholds);
		cudaFree(dev_mask_amper);

		cudaFree(dev_mask);
		cudaFree(dev_current);
		cudaFree(dev_target);

		cudaFree(dev_observe);
		cudaFree(dev_signal);
		cudaFree(dev_load);

		cudaFree(dev_d1);
		cudaFree(dev_d2);
		cudaFree(dev_diag);
		cudaFree(dev_diag_);

		cudaFree(dev_npdir_mask);
		cudaFree(dev_npdirs);
		cudaFree(dev_signals);
		cudaFree(dev_dists);
		cudaFree(dev_union_root);
	}
	catch (CoreException &e) {
		_log->error() << "Fatal error in free_all_parameters, when doing gpu array release";
		throw CoreException("Fatal error in free_all_parameters, when doing gpu array release", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
	}
	_log->info() << "All memory released";
}


/*
This function is generating dir matrix memory both on host and device
*/
void DataManager::gen_direction() {
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
void DataManager::gen_weight() {
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
void DataManager::gen_thresholds() {
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
void DataManager::gen_mask_amper() {
	//malloc the space
	h_mask_amper = new bool[_mask_amper_size_max];
	cudaMalloc(&dev_mask_amper, _mask_amper_size_max * sizeof(bool));

	//fill in all 0
	memset(h_mask_amper, 0, _mask_amper_size_max * sizeof(bool));
	cudaMemset(dev_mask_amper, 0, _mask_amper_size_max * sizeof(bool));

	_log->debug() << "Mask amper matrix generated with size " + to_string(_mask_amper_size_max);
}

/*
This function is generating n power of dir matrix both on host and device
*/
void DataManager::gen_np_direction() {
	//malloc the space
	h_npdirs = new bool[_measurable2d_size_max];
	cudaMalloc(&dev_npdirs, _measurable2d_size_max * sizeof(bool));

	//fill in all 0
	memset(h_npdirs, 0, _measurable2d_size_max * sizeof(bool));
	cudaMemset(dev_npdirs, 0, _measurable2d_size_max * sizeof(bool));

	_log->debug() << "NPDir matrix generated with size " + to_string(_measurable2d_size_max);
}

void DataManager::gen_signals() {
	//malloc the space
	h_signals = new bool[_sensor_size_max * _measurable_size_max];
	h_lsignals = new bool[_sensor_size_max * _measurable_size_max];
	cudaMalloc(&dev_signals, _sensor_size_max * _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_lsignals, _sensor_size_max * _measurable_size_max * sizeof(bool));

	//init with all false
	memset(h_signals, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));
	memset(h_lsignals, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));
	cudaMemset(dev_signals, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));
	cudaMemset(dev_lsignals, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));

	_log->debug() << to_string(_sensor_size_max) + " num of signals with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_sensor_size_max * _measurable_size_max);
	_log->debug() << to_string(_sensor_size_max) + " num of loaded signals with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_sensor_size_max * _measurable_size_max);
}

void DataManager::gen_npdir_mask() {
	//malloc the space
	h_npdir_mask = new bool[_sensor_size_max * _measurable_size_max];
	cudaMalloc(&dev_npdir_mask, _sensor_size_max * _measurable_size_max * sizeof(bool));

	//init with all false
	memset(h_npdir_mask, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));
	cudaMemset(dev_npdir_mask, 0, _sensor_size_max * _measurable_size_max * sizeof(bool));

	_log->debug() << to_string(_sensor_size_max) + " num of npdir mask with length " + to_string(_measurable_size_max) + " are generated, total size " + to_string(_sensor_size_max * _measurable_size_max);
}

void DataManager::gen_dists() {
	//malloc the space
	h_dists = new int[_measurable_size_max * _measurable_size_max];
	cudaMalloc(&dev_dists, _measurable_size_max * _measurable_size_max * sizeof(int));

	//init with all 0
	memset(h_dists, 0, _measurable_size_max * _measurable_size_max * sizeof(int));
	cudaMemset(dev_dists, 0, _measurable_size_max * _measurable_size_max * sizeof(int));

	_log->debug() << to_string(_measurable_size_max) + "*" + to_string(_measurable_size_max) + "=" + to_string(_measurable_size_max * _measurable_size_max) + " num of space allocated for dists, used for block GPU";
}

/*
This function generate other parameter
*/
void DataManager::gen_other_parameters() {
	h_observe = new bool[_measurable_size_max];
	h_observe_ = new bool[_measurable_size_max];
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
	h_union_root = new int[_sensor_size_max];

	cudaMalloc(&dev_observe, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_signal, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_load, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_mask, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_current, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_target, _measurable_size_max * sizeof(bool));

	cudaMalloc(&dev_d1, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_d2, _measurable_size_max * sizeof(bool));
	cudaMalloc(&dev_diag, _measurable_size_max * sizeof(double));
	cudaMalloc(&dev_diag_, _measurable_size_max * sizeof(double));
	cudaMalloc(&dev_union_root, _sensor_size_max * sizeof(int));

	_log->debug() << "Other parameter generated, with size " + to_string(_measurable_size_max);
}

void DataManager::create_sensors_to_arrays_index(int start_idx, int end_idx, vector<Sensor*> &sensors) {
	//create idx for diag and current
	for (int i = start_idx; i < end_idx; ++i) {
		try {
			sensors[i]->setMeasurableDiagPointers(h_diag, h_diag_);
			sensors[i]->setMeasurableObservePointers(h_observe, h_observe_);
			sensors[i]->setMeasurableCurrentPointers(h_current);
		}
		catch (exception &e) {
			_log->error() << "Fatal error while doing create_sensor_to_arrays_index";
			throw CoreException("Fatal error happen in create_sensors_to_arrays_index", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->info() << "Sensor from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " have created idx to arrays";
}

void DataManager::create_sensor_pairs_to_arrays_index(int start_idx, int end_idx, vector<SensorPair*> &sensor_pairs) {
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		try {
			sensor_pairs[i]->setAllPointers(h_weights, h_dirs, h_thresholds);
		}
		catch (exception &e) {
			_log->error() << "Fatal error while doing create_sensor_pairs_to_arrays_index";
			throw CoreException("Fatal error happen in create_sensor_pairs_to_arrays_index", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->info() << "Sensor pairs from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " have created idx to arrays";
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/
void DataManager::copy_arrays_to_sensors(int start_idx, int end_idx, vector<Sensor*> &_sensors) {
	//copy necessary info back from cpu array to GPU array first
	cudaMemcpy(h_current + start_idx, dev_current + start_idx, (end_idx - start_idx) * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_diag + start_idx, dev_diag + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_diag_ + start_idx, dev_diag_ + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyDeviceToHost);
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from GPU arrays to CPU arrays";
	for (int i = start_idx; i < end_idx; ++i) {
		//bring all sensor and measurable info into the object
		try {
			_sensors[i]->pointers_to_values();
		}
		catch (CoreException &e) {
			_log->error() << "Fatal error while doing copy_arrays_to_sensors";
			throw CoreException("Fatal error in function copy_arrays_to_sensors", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from cpu arrays to sensor";
	_log->info() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied back from arrays";
}

void DataManager::copy_sensors_to_arrays(int start_idx, int end_idx, vector<Sensor*> &_sensors) {
	//copy necessary info from sensor object to cpu arrays
	for (int i = start_idx; i < end_idx; ++i) {
		try {
			_sensors[i]->copyAmperList(h_mask_amper);
			_sensors[i]->values_to_pointers();
		}
		catch (exception &e) {
			_log->error() << "Fatal error while doing copy_sensors_to_arrays";
			throw CoreException("Fatal error in function copy_sensors_to_arrays", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from sensor to CPU arrays";
	//copy data from cpu array to GPU array
	cudaMemcpy(dev_mask_amper + 2 * ind(start_idx, 0), h_mask_amper + 2 * ind(start_idx, 0), 2 * (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_current + start_idx, h_current + start_idx, (end_idx - start_idx) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag + start_idx, h_diag + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_diag_ + start_idx, h_diag_ + start_idx, (end_idx - start_idx) * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied from CPU arrays to GPU arrays";
	_log->info() << "Sensor data from idx " + to_string(start_idx) + " to " + to_string(end_idx) + " are copied to arrays";
}

/*
--------------------Sensor Object And Array Data Transfer---------------------
*/

/*
--------------------SensorPair Object And Array Data Transfer---------------------
*/
void DataManager::copy_sensor_pairs_to_arrays(int start_idx, int end_idx, vector<SensorPair*> &_sensor_pairs) {
	//copy data from sensor pair object to CPU arrays
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		try {
			_sensor_pairs[i]->values_to_pointers();
		}
		catch (exception &e) {
			_log->error() << "Fatal error while doing copy_sensor_pairs_to_arrays";
			throw CoreException("Fatal error in function copy_sensor_pairs_to_arrays", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from sensor pairs to CPU arrays";
	//copy data from CPU arrays to GPU arrays
	cudaMemcpy(dev_weights + ind(2 * start_idx, 0), h_weights + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dirs + ind(2 * start_idx, 0), h_dirs + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_thresholds + ind(start_idx, 0), h_thresholds + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to GPU arrays";
	_log->info() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied to arrays";
}

void DataManager::copy_arrays_to_sensor_pairs(int start_idx, int end_idx, vector<SensorPair*> &_sensor_pairs) {
	//copy data from GPU arrays to CPU arrays
	cudaMemcpy(h_weights + ind(2 * start_idx, 0), dev_weights + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_dirs + ind(2 * start_idx, 0), dev_dirs + ind(2 * start_idx, 0), (ind(2 * end_idx, 0) - ind(2 * start_idx, 0)) * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_thresholds + ind(start_idx, 0), dev_thresholds + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from GPU arrays to CPU arrays";
	//copy data from CPU arrays to sensor pairs
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		//bring all sensor pair info into the object
		try {
			_sensor_pairs[i]->pointers_to_values();
		}
		catch (exception &e) {
			_log->error() << "Fatal error while doing copy_arrays_to_sensor_pairs";
			throw CoreException("Fatal error in function copy_arrays_to_sensor_pairs", CoreException::CORE_FATAL, status_codes::ServiceUnavailable);
		}
	}
	_log->debug() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied from CPU arrays to sensor pairs";
	_log->info() << "Sensor pairs data from idx " + to_string(ind(start_idx, 0)) + " to " + to_string(ind(end_idx, 0)) + " are copied back from arrays";
}

/*
Set the mask used in halucination, for testing purpose, usual workflow, mask is get from gen_mask
input: mask vector, size of mask have to be the _measurable_size
*/
void DataManager::setMask(vector<bool> &mask) {
	if (mask.size() != _measurable_size) {
		throw CoreException("Input mask size not matching measurable size!", CoreException::CORE_ERROR, status_codes::BadRequest);
	}
	for (int i = 0; i < mask.size(); ++i) h_mask[i] = mask[i];
	cudaMemcpy(dev_mask, h_mask, mask.size() * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
This function set observe signal from python side
*/
void DataManager::setObserve(vector<bool> &observe) {//this is where data comes in in every frame
	_log->debug() << "Setting observe signal";
	cudaMemcpy(h_observe_, h_observe, _measurable_size * sizeof(bool), cudaMemcpyHostToHost);
	for (int i = 0; i < observe.size(); ++i) {
		h_observe[i] = observe[i];
	}
	cudaMemcpy(dev_observe, h_observe, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void DataManager::setCurrent(vector<bool> &current) {//this is where data comes in in every frame
	_log->debug() << "Setting current signal for customized purpose";
	for (int i = 0; i < current.size(); ++i) {
		h_current[i] = current[i];
	}
	cudaMemcpy(dev_current, h_current, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void DataManager::setTarget(vector<bool> &signal) {
	for (int i = 0; i < _measurable_size; ++i) {
		h_target[i] = signal[i];
	}
	cudaMemcpy(dev_target, h_target, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
set Signal
input: list of signal in bool, input size has to be measurable size
*/
void DataManager::setSignal(vector<bool> &signal) {
	if (signal.size() != _measurable_size) {
		throw CoreException("The input signal size is not matching measurable size!", CoreException::CORE_ERROR, status_codes::BadRequest);
	}
	for (int i = 0; i < _measurable_size; ++i) {
		h_signal[i] = signal[i];
	}
	cudaMemcpy(dev_signal, h_signal, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
set Signals
input: 2d vector of signals, first dimension should not exceed sensor size, second one must be measurable size
*/
void DataManager::setSignals(vector<vector<bool> > &signals) {
	int sig_count = signals.size();
	if (sig_count > _sensor_size) {
		throw CoreException("The input sensor size is larger than current sensor size!", CoreException::CORE_ERROR, status_codes::BadRequest);
	}
	for (int i = 0; i < sig_count; ++i) {
		if (signals[i].size() != _measurable_size) {
			throw CoreException("The " + to_string(i) + "th input string size is not matching measurable size!", CoreException::CORE_ERROR, status_codes::BadRequest);
		}
		for (int j = 0; j < signals[i].size(); ++j) {
			h_signals[i * _measurable_size + j] = signals[i][j];
		}
	}
	cudaMemcpy(dev_signals, h_signals, sig_count * _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

/*
set load
input: list of load in bool, list size has to be measurable size
*/
void DataManager::setLoad(vector<bool> &load) {
	if (load.size() != _measurable_size) {
		throw CoreException("The input load size is not matching the measurable size!", CoreException::CORE_ERROR, status_codes::BadRequest);
	}
	for (int i = 0; i < _measurable_size; ++i) {
		h_load[i] = load[i];
	}
	cudaMemcpy(dev_load, h_load, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
}

void DataManager::setDists(vector<vector<int> > &dists) {
	if (dists.size() != _sensor_size) {
		throw CoreException("The input dists size is not matching the measurable size!", CoreException::CORE_ERROR, status_codes::BadRequest);
	}
	for (int i = 0; i < dists.size(); ++i) {
		for (int j = 0; j < dists[0].size(); ++j) h_dists[i * _sensor_size + j] = dists[i][j];
	}
	cudaMemcpy(dev_dists, h_dists, _sensor_size * _sensor_size * sizeof(int), cudaMemcpyHostToDevice);
}

vector<vector<bool> > DataManager::getSignals(int sig_count) {
	vector<vector<bool> > results;
	cudaMemcpy(h_signals, dev_signals, sig_count * _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sig_count; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signals[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	return results;
}

vector<vector<bool> > DataManager::getLSignals(int sig_count) {
	vector<vector<bool> > results;
	cudaMemcpy(h_lsignals, dev_lsignals, sig_count * _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < sig_count; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_lsignals[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	return results;
}

vector<vector<bool> > DataManager::getNpdirMasks() {
	vector<vector<bool> > results;
	cudaMemcpy(h_npdir_mask, dev_npdir_mask, _sensor_size * _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _sensor_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_npdir_mask[i * _measurable_size + j]);
		results.push_back(tmp);
	}
	return results;
}

//get the current(observe value through propagation) value from GPU
vector<bool> DataManager::getCurrent() {
	vector<bool> result;
	cudaMemcpy(h_current, dev_current, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_current[i]);
	}
	return result;
}

//get the prediction(next iteration prediction) value from GPU
vector<bool> DataManager::getPrediction() {
	vector<bool> result;
	//no corresponding dev variable to copy from, should be copied after the halucinate
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_prediction[i]);
	}
	return result;
}

/*
The function is getting the current target
*/
vector<bool> DataManager::getTarget() {
	vector<bool> result;
	cudaMemcpy(h_target, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_target[i]);
	}
	return result;
}

/*
The function is getting weight matrix in 2-dimensional form
*/
vector<vector<double> > DataManager::getWeight2D() {
	cudaMemcpy(h_weights, dev_weights, _measurable2d_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_weights[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the dir matrix in 2-dimensional form
*/
vector<vector<bool> > DataManager::getDir2D() {
	cudaMemcpy(h_dirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_dirs[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the n power of dir matrix in 2-dimensional form
*/
vector<vector<bool> > DataManager::getNPDir2D() {
	cudaMemcpy(h_npdirs, dev_npdirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<bool> > result;
	int n = 0;
	for (int i = 0; i < _measurable_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_npdirs[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the threshold matrix in 2-dimensional form
*/
vector<vector<double> > DataManager::getThreshold2D() {
	cudaMemcpy(h_thresholds, dev_thresholds, _sensor2d_size * sizeof(bool), cudaMemcpyDeviceToHost);
	vector<vector<double> > result;
	int n = 0;
	for (int i = 0; i < _sensor_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j)
			tmp.push_back(h_thresholds[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting the mask amper in 2-dimension form
*/
vector<vector<bool> > DataManager::getMask_amper2D() {
	vector<vector<bool> > result;
	cudaMemcpy(h_mask_amper, dev_mask_amper, _mask_amper_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int n = 0;
	for (int i = 0; i < _sensor_size; ++i) {
		vector<bool> tmp;
		for (int j = 0; j <= 2 * i + 1; ++j)
			tmp.push_back(h_mask_amper[n++]);
		result.push_back(tmp);
	}
	return result;
}

/*
The function is getting mask amper as an list
*/
vector<bool> DataManager::getMask_amper() {
	vector<vector<bool> > tmp = getMask_amper2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the weight matrix as a list
*/
vector<double> DataManager::getWeight() {
	vector<vector<double > > tmp = getWeight2D();
	vector<double> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the dir matrix as a list
*/
vector<bool> DataManager::getDir() {
	vector<vector<bool> > tmp = this->getDir2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the n power of dir matrix as a list
*/
vector<bool> DataManager::getNPDir() {
	vector<vector<bool> > tmp = this->getNPDir2D();
	vector<bool> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
The function is getting the threshold matrix as a list
*/
vector<double> DataManager::getThresholds() {
	vector<vector<double> > tmp = getThreshold2D();
	vector<double> results;
	for (int i = 0; i < tmp.size(); ++i) {
		for (int j = 0; j < tmp[i].size(); ++j) results.push_back(tmp[i][j]);
	}
	return results;
}

/*
This function is getting the diagonal value of the weight matrix
*/
vector<double> DataManager::getDiag() {
	cudaMemcpy(h_diag, dev_diag, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_diag[i]);
	}
	return result;
}

/*
This function is getting the diagonal value of the weight matrix of last iteration
*/
vector<double> DataManager::getDiagOld() {
	cudaMemcpy(h_diag_, dev_diag_, _measurable_size * sizeof(double), cudaMemcpyDeviceToHost);
	vector<double> result;
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_diag_[i]);
	}
	return result;
}

/*
This function is getting the current mask value used in halucination
*/
vector<bool> DataManager::getMask() {
	vector<bool> result;
	cudaMemcpy(h_mask, dev_mask, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->_measurable_size; ++i) result.push_back(h_mask[i]);
	return result;
}

/*
This function is getting the obersev matrix
*/
vector<bool> DataManager::getObserve() {
	vector<bool> result;
	cudaMemcpy(h_observe, dev_observe, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->_measurable_size; ++i) result.push_back(h_observe[i]);
	return result;
}

vector<bool> DataManager::getLoad() {
	vector<bool> result;
	cudaMemcpy(h_load, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _measurable_size; ++i) result.push_back(h_load[i]);
	return result;
}

/*
The function is getting the up matrix
*/
vector<bool> DataManager::getUp() {
	vector<bool> result;
	//up have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_up[i]);
	}
	return result;
}

/*
The function is getting the down matrix
*/
vector<bool> DataManager::getDown() {
	vector<bool> result;
	//down have no corresponding dev variable, when doing up_GPU, the value should be stored and copied back to host
	for (int i = 0; i < _measurable_size; ++i) {
		result.push_back(h_down[i]);
	}
	return result;
}

std::map<string, int> DataManager::getSizeInfo() {
	std::map<string, int> s;

	s["_sensor_size"] = _sensor_size;
	s["_sensor_size_max"] = _sensor_size_max;
	s["_sensor2d_size"] = _sensor2d_size;
	s["_sensor2d_size_max"] = _sensor2d_size_max;
	s["_measurable_size"] = _measurable_size;
	s["_measurable_size_max"] = _measurable_size_max;
	s["_measurable2d_size"] = _measurable2d_size;
	s["_measurable2d_size_max"] = _measurable2d_size_max;
	s["_mask_amper_size"] = _mask_amper_size;
	s["_mask_amper_size_max"] = _mask_amper_size_max;

	return s;
}


/*
This function is halucinate on GPU, it use several propagate_GPU
It first get the mask to use and then use propagate
The output will be stored in Gload(propagate_GPU)
Input: action list to be halucinated
Output: None
*/

void DataManager::halucinate(int initial_size) {
	gen_mask(initial_size);

	propagate_GPU(dev_mask);
	
	cudaMemcpy(h_prediction, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void DataManager::propagate_observe() {
	propagate_GPU(dev_observe);

	cudaMemcpy(h_current, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
}


void DataManager::update_state(vector<bool> &signal, double q, double phi, double total, double total_, bool active) {
	_log->debug() << "start updating data on GPU";

	setObserve(signal);

	update_weights(q, phi, active);
	//update_diag();
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, _measurable_size);

	update_thresholds(q, phi, total_);

	orient_all(total);
	floyd_GPU();

	propagate_observe();
}

void DataManager::set_implication(bool value, int idx1, int idx2) {
	h_dirs[ind(idx1, idx2)] = value;
	cudaMemcpy(dev_dirs + ind(idx1, idx2), h_dirs + ind(idx1, idx2), sizeof(bool), cudaMemcpyHostToDevice);
}

bool DataManager::get_implication(int idx1, int idx2) {
	cudaMemcpy(h_dirs + ind(idx1, idx2), dev_dirs + ind(idx1, idx2), sizeof(bool), cudaMemcpyDeviceToHost);
	return h_dirs[ind(idx1, idx2)];
}

void DataManager::init_other_parameter(double total) {
	for (int i = 0; i < _measurable_size_max; ++i) {
		h_observe[i] = false;
		h_signal[i] = false;
		h_load[i] = false;
		h_mask[i] = false;
		h_current[i] = false;
		h_target[i] = false;
		h_prediction[i] = false;
		h_up[i] = false;
		h_down[i] = false;
		h_diag[i] = total / 2.0;
		h_diag_[i] = total / 2.0;
	}
	for (int i = 0; i < _sensor_size_max; ++i) {
		h_union_root[i] = 0;
	}

	cudaMemset(dev_observe, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_signal, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_load, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_mask, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_current, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_target, 0, _measurable_size_max * sizeof(bool));

	cudaMemset(dev_d1, 0, _measurable_size_max * sizeof(bool));
	cudaMemset(dev_d2, 0, _measurable_size_max * sizeof(bool));

	cudaMemset(dev_union_root, 0, _sensor_size_max * sizeof(bool));
	cudaMemset(dev_npdirs, 0, _measurable_size_max * sizeof(bool));
	uma_base::init_diag(dev_diag, dev_diag_, total, total, _measurable_size_max);
}


/*
This function is an independent up function on GPU
It only use signal to do dfs, result is stored in Gsignal after using the function
Input: signal to be dfsed
Output: None
*/
void DataManager::up_GPU(vector<bool> &signal, double q, bool is_stable) {
	for (int i = 0; i < _measurable_size; ++i) h_signal[i] = signal[i];
	cudaMemcpy(dev_signal, h_signal, _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);

	uma_base::multiply(dev_signal, dev_dirs, dev_thresholds, q, _measurable_size);
	
	cudaMemcpy(h_up, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

/*
set signals with load
have to make sure load is set before calling this function
input: 2d signals vector
*/
void DataManager::setLSignals(vector<vector<bool> > &signals) {
	int sig_count = signals.size();
	for (int i = 0; i < sig_count; ++i) {
		for (int j = 0; j < signals[i].size(); ++j) {
			h_lsignals[i * _measurable_size + j] = signals[i][j];
		}
	}
	cudaMemcpy(dev_lsignals, h_lsignals, sig_count * _measurable_size * sizeof(bool), cudaMemcpyHostToDevice);
	for (int i = 0; i < sig_count; ++i) {
		kernel_util::disjunction(dev_lsignals + i * _measurable_size, dev_load, _measurable_size);
	}
}


void DataManager::ups_GPU(int sig_count) {
	uma_base::transpose_multiply(dev_npdirs, dev_signals, _measurable_size, sig_count);
}

void DataManager::propagate_mask() {
	kernel_util::allfalse(dev_npdir_mask, _sensor_size * _measurable_size);
	for (int i = 0; i < _sensor_size; ++i) {
		cudaMemcpy(dev_npdir_mask + _measurable_size * i, dev_mask_amper + ind(i, 0) * 2, (ind(i + 1, 0) - ind(i, 0)) * 2 * sizeof(bool), cudaMemcpyDeviceToDevice);
	}

	uma_base::transpose_multiply(dev_npdirs, dev_npdir_mask, _measurable_size, _sensor_size);

	//cudaMemcpy(h_npdir_mask, dev_npdir_mask, _sensor_size * _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void DataManager::floyd_GPU() {
	cudaMemcpy(dev_npdirs, dev_dirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	uma_base::floyd(dev_npdirs, _measurable_size);
	cudaMemcpy(h_npdirs, dev_npdirs, _measurable2d_size * sizeof(bool), cudaMemcpyDeviceToHost);

	cudaCheckErrors("kernel fails");
}

vector<vector<vector<bool> > > DataManager::abduction(vector<vector<bool> > &signals) {
	vector<vector<vector<bool> > > results;
	vector<vector<bool> > even_results, odd_results;
	for (int i = 0; i < signals.size(); ++i) {
		vector<int> even_idx, odd_idx;
		for (int j = 0; j < signals[0].size() / 2; ++j) {
			if (signals[i][2 * j]) even_idx.push_back(j);
			if (signals[i][2 * j + 1]) odd_idx.push_back(j);
		}
		if (even_idx.empty()) {
			vector<bool> tmp(_measurable_size, false);
			even_results.push_back(tmp);
		}
		else {
			kernel_util::alltrue(dev_signal, _measurable_size);
			for (int j = 0; j < even_idx.size(); ++j) {
				kernel_util::conjunction(dev_signal, dev_npdir_mask + even_idx[j] * _measurable_size, _measurable_size);
			}
			cudaMemcpy(h_signal, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
			vector<bool> tmp;
			for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signal[j]);
			even_results.push_back(tmp);
		}
		if (odd_idx.empty()) {
			vector<bool> tmp(_measurable_size, false);
			odd_results.push_back(tmp);
		}
		else {
			kernel_util::allfalse(dev_signal, _measurable_size);
			for (int j = 0; j < odd_idx.size(); ++j) {
				kernel_util::disjunction(dev_signal, dev_npdir_mask + odd_idx[j] * _measurable_size, _measurable_size);
			}
			cudaMemcpy(dev_signals, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_lsignals, dev_signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
			cudaMemset(dev_load, false, _measurable_size * sizeof(bool));
			propagates_GPU(1);
			cudaMemcpy(h_signals, dev_lsignals, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
			vector<bool> tmp;
			for (int j = 0; j < _measurable_size; ++j) tmp.push_back(h_signals[j]);
			odd_results.push_back(tmp);
		}
	}
	results.push_back(even_results);
	results.push_back(odd_results);
	return results;
}

vector<vector<int> > DataManager::blocks_GPU(float delta) {
	int t = floor(log(_sensor_size) / log(2)) + 1;
	for (int i = 0; i < t; ++i) {
		uma_base::dioid_square(dev_dists, _sensor_size);
	}

	uma_base::union_init(dev_union_root, _sensor_size);
	uma_base::check_dist(dev_dists, delta, _sensor_size * _sensor_size);

	uma_base::union_GPU(dev_dists, dev_union_root, _sensor_size);
	cudaMemcpy(h_union_root, dev_union_root, _sensor_size * sizeof(int), cudaMemcpyDeviceToHost);

	map<int, int> m;
	vector<vector<int> > result;
	for (int i = 0; i < _sensor_size; ++i) {
		if (m.find(h_union_root[i]) == m.end()) {
			m[h_union_root[i]] = result.size();
			vector<int> tmp;
			tmp.push_back(i);
			result.push_back(tmp);
		}
		else {
			result[m[h_union_root[i]]].push_back(i);
		}
	}

	return result;
}

void DataManager::gen_mask(int initial_size) {
	uma_base::init_mask(dev_mask, initial_size, _measurable_size);

	dim3 dimGrid((_sensor_size + 15) / 16, (_sensor_size + 15) / 16);
	dim3 dimBlock(16, 16);

	uma_base::mask(dev_mask_amper, dev_mask, dev_current, _sensor_size);
	uma_base::check_mask(dev_mask, _sensor_size);
}

void DataManager::propagates_GPU(int sig_count) {
	uma_base::transpose_multiply(dev_npdirs, dev_lsignals, _measurable_size, sig_count);
	uma_base::transpose_multiply(dev_npdirs, dev_signals, _measurable_size, sig_count);

	kernel_util::negate_conjunction_star(dev_lsignals, dev_signals, _sensor_size);
}

/*
This function do propagate on GPU
//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
Result is stored in Gload
Ask Kotomasha for mathematic questions
Input: signal and load
Output: None
*/
void DataManager::propagate_GPU(bool *signal) {//propagate
	cudaMemcpy(dev_signal, signal, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load, false, _measurable_size * sizeof(bool));

	uma_base::multiply(dev_load, dev_dirs, dev_thresholds, 0, _measurable_size);
	uma_base::multiply(dev_signal, dev_dirs, dev_thresholds, 0, _measurable_size);

	// standard operations
	kernel_util::disjunction(dev_load, dev_signal, _measurable_size);
	kernel_util::negate_conjunction_star(dev_load, dev_signal, _sensor_size);

	cudaMemcpy(h_load, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}


float DataManager::distance(bool *signal1, bool *signal2) {
	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	cudaMemcpy(dev_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	propagate_GPU(signal1);
	cudaMemcpy(signal1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	cudaMemcpy(dev_signal, signal2, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	propagate_GPU(signal2);
	cudaMemcpy(signal2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	kernel_util::conjunction_star(signal1, signal2, _measurable_size);
	//apply weights to the computed distance signal and output the result:
	/*
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
	*/
	//weights are w_{xx}-w_{x*x*}:
	/*
	delta_weight_sum_kernel << <(_measurable_size + 255) / 256, 256 >> > (dev_diag, signal1, dev_result, _measurable_size);
	cudaMemcpy(tmp_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	float result = tmp_result[0];
	delete[] tmp_result;
	cudaFree(dev_result);
	return result;
	*/
	//weights are all 1:

	cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
}

float DataManager::divergence() {
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_d2, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(dev_d1);
	cudaMemcpy(dev_d1, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	cudaMemset(dev_load, 0, _measurable_size * sizeof(bool));
	propagate_GPU(dev_d2);
	cudaMemcpy(dev_d2, dev_load, _measurable_size * sizeof(bool), cudaMemcpyDeviceToDevice);

	kernel_util::subtraction(dev_d1, dev_d2, _measurable_size);
	//apply weights to the computed divergence signal and output the result:
	float *tmp_result = new float[1];
	tmp_result[0] = 0.0f;
	float *dev_result;
	cudaMalloc(&dev_result, sizeof(float));
	cudaMemcpy(dev_result, tmp_result, sizeof(float), cudaMemcpyHostToDevice);
	//weights are w_{xx}-w_{x*x*}:

	uma_base::delta_weight_sum(dev_diag, dev_d1, dev_result, _measurable_size);
	cudaMemcpy(tmp_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);
	float result = tmp_result[0];
	delete[] tmp_result;
	cudaFree(dev_result);
	return result;

	//weights are all 1:
	/*
	cudaMemcpy(h_signal, signal1, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
	int sum = 0;
	for (int i = 0; i < _measurable_size; ++i) sum += h_signal[i];
	return sum;
	*/
}

void DataManager::update_diag() {
	uma_base::get_weights_diag(dev_weights, dev_diag, dev_diag_, _measurable_size);
}

void DataManager::calculate_target() {
	uma_base::calculate_target(dev_diag, dev_target, _sensor_size);
	cudaMemcpy(h_target, dev_target, _measurable_size * sizeof(bool), cudaMemcpyDeviceToHost);
}

void DataManager::update_weights(double q, double phi, bool active) {
	dim3 dimGrid2((_measurable_size + 15) / 16, (_measurable_size + 15) / 16);
	dim3 dimBlock2(16, 16);
	uma_base::update_weights(dev_weights, dev_observe, _measurable_size, q, phi, active);
}

void DataManager::orient_all(double total) {
	uma_base::orient_all(dev_dirs, dev_weights, dev_thresholds, total, _sensor_size);
}

void DataManager::update_thresholds(double q, double phi, double total_) {
	dim3 dimGrid1((_sensor_size + 15) / 16, (_sensor_size + 15) / 16);
	dim3 dimBlock1(16, 16);
	uma_base::update_thresholds(dev_dirs, dev_thresholds, total_, q, phi, _sensor_size);
}