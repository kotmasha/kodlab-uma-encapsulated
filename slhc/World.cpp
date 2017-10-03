#include "World.h"
#include "Pair.h"
#include "DataManager.h"

extern int ind(int row, int col);

World::World() {
	_dm = new DataManager();
}

void World::insert(vector<double> &row) {
	//TBD check input row size, row.size() == _point_num + 1
	int row_idx = row.size();
	for (int i = 0; i < row.size(); ++i) {
		Pair *pair = new Pair(row_idx, i, row[i]);
		_pairs.push_back(pair);
	}
	_point_num++;

	if (_point_num > _dm->_point_size_max) {
		_dm->realloc_memory(_point_num);
		_dm->create_pair_to_array_indexes(ind(0, 0), ind(_point_num + 1, 0), _pairs);
		_dm->copy_pairs_to_arrays(ind(0, 0), ind(_point_num + 1, 0), _pairs);
		//need to copy all pairs since remalloc
	}
	else {
		_dm->set_size(_point_num, false);
		_dm->create_pair_to_array_indexes(ind(_point_num, 0), ind(_point_num + 1, 0), _pairs);
		_dm->copy_pairs_to_arrays(ind(_point_num, 0), ind(_point_num + 1, 0), _pairs);
	}
}

vector<bool> World::convert_int_list(vector<int> &list) {
	vector<bool> results(_pairs.size(), false);
	for (int i = 0; i < list.size(); ++i) {
		results[list[i]] = true;
	}
	return results;
}

void World::remove_rows(vector<int> &idx) {
	_dm->copy_arrays_to_pairs(0, _pairs.size(), _pairs);
	int row_escape = 0;
	int total_escape = 0;
	vector<bool> list = convert_int_list(idx);
	for (int i = 0; i < _pairs.size(); ++i) {
		if (list[i]) _point_num--;
		for (int j = 0; j <= i; ++j) {
			if (list[i] || list[j]) {
				delete _pairs[ind(i, j)];
				_pairs[ind(i, j)] = NULL;
				total_escape++;
			}
			else {
				//or just change the position
				_pairs[ind(i, j) - total_escape] = _pairs[ind(i, j)];
			}
		}
	}

	_pairs.erase(_pairs.end() - total_escape, _pairs.end());
	_dm->set_size(_point_num, false);

	_dm->create_pair_to_array_indexes(0, _pairs.size(), _pairs);
	_dm->copy_pairs_to_arrays(0, _pairs.size(), _pairs);
}

void World::calculate(string &key) {
	if (key == "slhc") {

	}
}

vector<double> World::report(string &key) {
	if (key == "dist") {
		return _dm->getDist();
	}
	else if(key == "slhc"){
		return _dm->getSlhc();
	}
}

void World::blocks(double &delta) {

}


World::~World() {}