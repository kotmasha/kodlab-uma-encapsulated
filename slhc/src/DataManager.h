#ifndef _DATAMANAGER_
#define _DATAMANAGER_

#include "Pair.h"
#include "Header.h"
#include "World.h"

using namespace std;

class logManager;

class DataManager {
protected:
	// the original distance matrix
	double *h_dists, *dev_dists;
	// the computed slhc matrix
	double *h_slhc, *dev_slhc;
	// the height matrix
	double *h_height, *dev_height;
	// the slhc value after comparing the delta, store 1 or 0
	int *h_slhc_, *dev_slhc_;
	// the array to store root value for union-find algorithm
	int *h_union_root, *dev_union_root;

protected:
	// point size and its max
	int _point_size, _point_size_max;
	// pair size and its max
	int _pair_size, _pair_size_max;
	// the memory expansion rate
	double _mem_expansion_rate;
	// log object
	logManager *_log;
	friend class World;

public:
	DataManager();
	void create_pair_to_array_indexes(int start_idx, int end_idx, vector<Pair*> &pairs);
	void copy_arrays_to_pairs(int start_idx, int end_idx, vector<Pair*> &pairs);
	void copy_pairs_to_arrays(int start_idx, int end_idx, vector<Pair*> &pairs);
	void init_pointers();
	void free_all_parameters();
	void realloc_memory(int point_size);
	void set_size(int point_size, bool change_max);
	void gen_dists();
	void gen_slhc();
	void gen_height();
	void gen_other_parameters();
	void calculate_slhc();
	vector<vector<int> > blocks(double &delta);

	vector<vector<double> > getDist2D();
	vector<vector<double> > getSlhc2D();
	vector<double> getDist();
	vector<double> getSlhc();
	vector<double> getHeight();
};

#endif