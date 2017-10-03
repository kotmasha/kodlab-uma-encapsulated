#ifndef _WORLD_
#define _WORLD_

#include "Header.h"

class Pair;
class DataManager;

class World {
protected:
	vector<Pair*> _pairs;
	int _point_num;
	DataManager *_dm;

public:
	World();
	void insert(vector<double> &row);
	void remove_rows(vector<int> &idx);
	void calculate(string &key);
	vector<double> report(string &key);
	void blocks(double &delta);
	vector<bool> convert_int_list(vector<int> &list);

	~World();
};

#endif