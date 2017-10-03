#include "DataManager.h"
#include "cuda.h"
#include "cuda_runtime.h"

extern int ind(int row, int col);

DataManager::DataManager() {
	_mem_expansion_rate = 0.5;
}

void DataManager::create_pair_to_array_indexes(int start_idx, int end_idx, vector<Pair*> &pairs) {
	for (int i = start_idx; i < end_idx; ++i) {
		pairs[i]->setPairPointers(h_dists, h_slhc);
	}
}

void DataManager::copy_arrays_to_pairs(int start_idx, int end_idx, vector<Pair*> &pairs) {
	cudaMemcpy(h_dists + ind(start_idx, 0), dev_dists + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_slhc + ind(start_idx, 0), dev_slhc + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < pairs.size(); ++i) {
		pairs[i]->pointers_to_values();
	}
}

void DataManager::copy_pairs_to_arrays(int start_idx, int end_idx, vector<Pair*> &pairs) {
	for (int i = 0; i < pairs.size(); ++i) {
		pairs[i]->values_to_pointers();
	}

	cudaMemcpy(dev_dists + ind(start_idx, 0), h_dists + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_slhc + ind(start_idx, 0), h_slhc + ind(start_idx, 0), (ind(end_idx, 0) - ind(start_idx, 0)) * sizeof(double), cudaMemcpyHostToDevice);
}

void DataManager::init_pointers() {
	h_dists = NULL;
	h_slhc = NULL;
	h_union_root = NULL;

	dev_dists = NULL;
	dev_slhc = NULL;
	dev_union_root = NULL;
}

void DataManager::free_all_parameters() {
	delete[] h_dists;
	delete[] h_slhc;
	delete[] h_union_root;

	cudaFree(dev_dists);
	cudaFree(dev_slhc);
	cudaFree(dev_union_root);
}

void DataManager::realloc_memory(int point_size) {
	free_all_parameters();

	set_size(point_size, true);

	gen_dists();
	gen_slhc();
	gen_other_parameters();
}

void DataManager::set_size(int point_size, bool change_max = true) {
	_point_size = point_size;
	_pair_size = point_size * (point_size + 1) / 2;

	if (change_max) {
		_point_size_max = (int)(_point_size * (1 + _mem_expansion_rate));
		_pair_size_max = (int)(_pair_size * (1 + _mem_expansion_rate));
	}
}

void DataManager::gen_dists() {
	h_dists = new double[_pair_size_max];

	cudaMalloc(&dev_dists, _pair_size_max * sizeof(double));
}

void DataManager::gen_slhc() {
	h_slhc = new double[_pair_size_max];

	cudaMalloc(&dev_slhc, _pair_size_max * sizeof(double));
}

void DataManager::gen_other_parameters() {
	h_union_root = new int[_point_size_max];

	cudaMalloc(&dev_union_root, _point_size_max * sizeof(int));
}

/*
get functions
*/

vector<vector<double> > DataManager::getDist2D() {
	vector<vector<double> > results;
	cudaMemcpy(h_dists, dev_dists, _pair_size * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < _point_size; ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			tmp.push_back(h_dists[ind(i, j)]);
		}
		results.push_back(tmp);
	}
	return results;
}

vector<vector<double> > DataManager::getSlhc2D() {
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
	vector<double> result;
	vector<vector<double> > results = getDist2D();
	for (int i = 0; i < results.size(); ++i) {
		for (int j = 0; j < results[i].size(); ++j) result.push_back(results[i][j]);
	}
	return result;
}

vector<double> DataManager::getSlhc() {
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