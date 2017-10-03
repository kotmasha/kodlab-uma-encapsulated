#ifndef _DATAMANAGER_
#define _DATAMANAGER_

#include "Pair.h"
#include "Header.h"
#include "World.h"

class DataManager {
protected:
	double *h_dists, *dev_dists;
	double *h_slhc, *dev_slhc;
	int *h_union_root, *dev_union_root;

protected:
	int _point_size, _point_size_max;
	int _pair_size, _pair_size_max;
	double _mem_expansion_rate;
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
	void gen_other_parameters();
	void calculate_slhc();

	vector<vector<double> > getDist2D();
	vector<vector<double> > getSlhc2D();
	vector<double> getDist();
	vector<double> getSlhc();
};

#endif