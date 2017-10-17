#include "DataManager.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "logging.h"
#include "logManager.h"

extern int ind(int row, int col);

DataManager::DataManager() {
	// log object init, if need to change log level, goto Header.h, and rebuild after changing
	_log = new logManager(DATA_LOG_LEVEL, "./", "dataManager.txt", "DataManager");
	// memory expansion rate depend the additional space an array will get based on a given point size
	_mem_expansion_rate = MEMORY_EXP;
	_log->info() << "memory expansion rate is: " + to_string(_mem_expansion_rate);
	init_pointers();
	// set size to 0, change all max value
	set_size(0, true);
}

void DataManager::create_pair_to_array_indexes(int start_idx, int end_idx, vector<Pair*> &pairs) {
	// create the index from pair object to array, input idx are based on POINT IDX
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		pairs[i]->setPairPointers(h_dists, h_slhc);
	}
	_log->debug() << "Created pair index on pair idx from: " + to_string(ind(start_idx, 0)) + "-" + to_string(ind(end_idx, 0));
}

void DataManager::copy_arrays_to_pairs(int start_idx, int end_idx, vector<Pair*> &pairs) {
	// copy value from array to pair object, input idx are based on POINT IDX
	cudaMemcpy(h_dists + ind(start_idx, 0), dev_dists + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_slhc + ind(start_idx, 0), dev_slhc + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	_log->debug() << "Copied value from gpu array to cpu array on idx from: " + to_string(ind(start_idx, 0)) + "-" + to_string(ind(end_idx, 0));

	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		pairs[i]->pointers_to_values();
	}
	_log->debug() << "Copied value from cpu array to pair on idx from: " + to_string(ind(start_idx, 0)) + "-" + to_string(ind(end_idx, 0));
}

void DataManager::copy_pairs_to_arrays(int start_idx, int end_idx, vector<Pair*> &pairs) {
	// copy value from pair object to array, input idx are based on POINT IDX
	for (int i = ind(start_idx, 0); i < ind(end_idx, 0); ++i) {
		pairs[i]->values_to_pointers();
	}
	_log->debug() << "Copied value from pair to cpu array on idx from: " + to_string(ind(start_idx, 0)) + "-" + to_string(ind(end_idx, 0));

	cudaMemcpy(dev_dists + ind(start_idx, 0), h_dists + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_slhc + ind(start_idx, 0), h_slhc + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);

	_log->debug() << "Copied value from cpu array to gpu array on idx from: " + to_string(ind(start_idx, 0)) + "-" + to_string(ind(end_idx, 0));
}

void DataManager::init_pointers() {
	// init all pointers
	_log->debug() << "Initiate all pointers to be Null";
	h_dists = NULL;
	h_slhc = NULL;
	h_slhc_ = NULL;
	h_height = NULL;
	h_union_root = NULL;

	dev_dists = NULL;
	dev_slhc = NULL;
	dev_slhc_ = NULL;
	dev_height = NULL;
	dev_union_root = NULL;
}

void DataManager::free_all_parameters() {
	// delete all array
	delete[] h_dists;
	delete[] h_slhc;
	delete[] h_slhc_;
	delete[] h_height;
	delete[] h_union_root;
	_log->debug() << "All host memory released";

	cudaFree(dev_dists);
	cudaFree(dev_slhc);
	cudaFree(dev_slhc_);
	cudaFree(dev_height);
	cudaFree(dev_union_root);
	_log->debug() << "All device memory released";
}

void DataManager::realloc_memory(int point_size) {
	// free all array first
	free_all_parameters();
	// set the new size
	set_size(point_size, true);
	// generate corresponding array
	gen_dists();
	gen_slhc();
	gen_height();
	gen_other_parameters();
}

void DataManager::set_size(int point_size, bool change_max = true) {
	// calculate all size value
	_point_size = point_size;
	_pair_size = point_size * (point_size + 1) / 2;

	_log->info() << "Resizing point_size: " + to_string(_point_size) + ", pair_size: " + to_string(_pair_size);

	if (change_max) {
		_point_size_max = (int)(_point_size * (1 + _mem_expansion_rate));
		_pair_size_max = _point_size_max * (_point_size_max + 1) / 2;
		_log->info() << "Resizing point_size_max: " + to_string(_point_size_max) + ", pair_size_max: " + to_string(_pair_size_max);
	}
}

void DataManager::gen_dists() {
	// generate dists matrix
	h_dists = new double[_pair_size_max];

	cudaMalloc(&dev_dists, _pair_size_max * sizeof(double));

	_log->debug() << "Generate " + to_string(_pair_size_max) + " size of dists on host and device";
}

void DataManager::gen_slhc() {
	// generate slhc matrix
	h_slhc = new double[_pair_size_max];
	h_slhc_ = new int[_pair_size_max];

	cudaMalloc(&dev_slhc, _pair_size_max * sizeof(double));
	cudaMalloc(&dev_slhc_, _pair_size_max * sizeof(int));

	_log->debug() << "Generate " + to_string(_pair_size_max) + " size of slhc and slhc_ on host and device";
}

void DataManager::gen_height() {
	// generate height matrix
	h_height = new double[_pair_size_max];

	cudaMalloc(&dev_height, _pair_size_max * sizeof(double));

	_log->debug() << "Generate " + to_string(_pair_size_max) + " size of height on host and device";
}


void DataManager::gen_other_parameters() {
	// generate other parameter
	h_union_root = new int[_point_size_max];

	cudaMalloc(&dev_union_root, _point_size_max * sizeof(int));

	_log->debug() << "Generate " + to_string(_point_size_max) + " size of union root on host and device";
}

/*
get functions
*/
vector<vector<double> > DataManager::getDist2D() {
	//get dist in 2d format
	vector<vector<double> > results;
	cudaMemcpy(h_dists, dev_dists, _pair_size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _point_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			tmp.push_back(h_dists[ind(i, j)]);
		}
		results.push_back(tmp);
	}
	_log->info() << "Get dist 2D info";
	return results;
}

vector<vector<double> > DataManager::getSlhc2D() {
	//get slhc in 2d format
	vector<vector<double> > results;
	cudaMemcpy(h_slhc, dev_slhc, _pair_size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _point_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			tmp.push_back(h_slhc[ind(i, j)]);
		}
		results.push_back(tmp);
	}
	return results;
}

vector<double> DataManager::getDist() {
	//get dist in 1d format
	vector<double> result;
	vector<vector<double> > results = getDist2D();
	for (int i = 0; i < results.size(); ++i) {
		for (int j = 0; j < results[i].size(); ++j) result.push_back(results[i][j]);
	}
	_log->info() << "Get dist info";
	return result;
}

vector<double> DataManager::getSlhc() {
	//get slhc in 1d format
	vector<double> result;
	vector<vector<double> > results = getSlhc2D();
	for (int i = 0; i < results.size(); ++i) {
		for (int j = 0; j < results[i].size(); ++j) result.push_back(results[i][j]);
	}
	return result;
}

/*
get functions
*/