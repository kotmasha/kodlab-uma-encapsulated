#include "World.h"
#include "Pair.h"
#include "DataManager.h"
#include "logManager.h"
#include "logging.h"

//defined in kernel.cu
extern int ind(int row, int col);

World::World() {
	//init the world, create the data manager, set point num to 0, and init the log object
	_dm = new DataManager();
	_point_num = 0;
	//for the log level, if need to change, goto Header.h and change to corresponding level. need to rebuild after changing
	_log = new logManager(WORLD_LOG_LEVEL, "./", "world.txt", "World");
}

void World::insert(vector<double> row) {
	//insert a row the the end of the current row
	int row_idx = row.size() - 1;
	for (int i = 0; i < row.size(); ++i) {
		//add the pair one by one
		Pair *pair = new Pair(row_idx, i, row[i]);
		_log->debug() << "Create a pair with y=" + to_string(row_idx) + ", x=" + to_string(i);
		_pairs.push_back(pair);
	}
	// point num plus
	_point_num++;
	_log->debug() << "Current point num: " + to_string(_point_num);
	_log->debug() << "Current pair num: " + to_string(_pairs.size());

	_log->verbose()<< "point_num: " + to_string(_point_num) + ", point_size_max : " + to_string(_dm->_point_size_max);
	//if after insertion CPU/GPU array is not enough to hold all data, will do remalloc
	if (_point_num > _dm->_point_size_max) {
		_log->info() << "Not enough space, need remalloc";
		//remalloc
		_dm->realloc_memory(_point_num);
		//create the index between world object and array
		_dm->create_pair_to_array_indexes(0, _point_num, _pairs);
		//copy data from world object to array on cpu/gpu
		_dm->copy_pairs_to_arrays(0, _point_num, _pairs);
		//need to copy all pairs since remalloc
	}
	//have enough space to hold all points
	else {
		_log->info() << "Have enough space, no remalloc";
		//just set the size, but do not change the max size
		_dm->set_size(_point_num, false);
		//create the index between world object and array only on newly added data
		_dm->create_pair_to_array_indexes(_point_num - 1, _point_num, _pairs);
		//copy data
		_dm->copy_pairs_to_arrays(_point_num - 1, _point_num, _pairs);
	}
}

vector<bool> World::convert_int_list(vector<int> &list) {
	//converting the int list to bool list, ie [1,5] length=6, will get [F, T, F, F, F, T]
	vector<bool> results(_point_num, false);
	for (int i = 0; i < list.size(); ++i) {
		results[list[i]] = true;
	}
	return results;
}

void World::remove_rows(vector<int> idx) {
	//remove the row from given idx list
	// before removing anything, copy all necessary fields back to object
	_dm->copy_arrays_to_pairs(0, _point_num, _pairs);
	// init value for in place change
	int row_escape = 0;
	int total_escape = 0;
	// get the bool list
	vector<bool> list = convert_int_list(idx);
	for (int i = 0; i < list.size(); ++i) {
		if (list[i]) {
			//if the current list is to be removed, minus point count
			_point_num--;
			row_escape++;
		}
		int col_escape = 0;
		for (int j = 0; j <= i; ++j) {
			if (list[i] || list[j]) {
				// if the pair will be removed, delete it, set the pointer to null
				col_escape++;
				delete _pairs[ind(i, j)];
				_pairs[ind(i, j)] = NULL;
				total_escape++;
			}
			else {
				// change the index of the current pair
				_pairs[ind(i, j)]->setIdx(i - row_escape, j - col_escape);
				// move the current pair to new place
				_pairs[ind(i, j) - total_escape] = _pairs[ind(i, j)];
			}
		}
		if (list[i]) _log->debug() << "Removed the row " + to_string(i);
	}

	// clean the pair vector, reset the size
	_pairs.erase(_pairs.end() - total_escape, _pairs.end());
	_dm->set_size(_point_num, false);

	// copy the data back to array
	_dm->create_pair_to_array_indexes(0, _point_num, _pairs);
	_dm->copy_pairs_to_arrays(0, _point_num, _pairs);
}

void World::calculate(string key) {
	if (key == "slhc") {
		// do slhc calculation
		_dm->calculate_slhc();
		_log->info() << "Slhc value calculated, point size: " + to_string(_point_num) + ", pair size: " + to_string(_pairs.size());
	}
}

vector<double> World::report(string key) {
	if (key == "dist") {
		_log->info() << "Get dist value";
		// return dist value
		return _dm->getDist();
	}
	else if(key == "slhc"){
		_log->info() << "Get slhc value";
		// return slhc value
		return _dm->getSlhc();
	}
	else if (key == "height") {
		_log->info() << "Get height value";
		// return height value
		return _dm->getHeight();
	}
}

vector<vector<int> > World::blocks(double delta) {
	// return different blocks based on input delta
	return _dm->blocks(delta);
}


World::~World() {}