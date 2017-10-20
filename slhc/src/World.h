#ifndef _WORLD_
#define _WORLD_
/*
World class, to held the test environment
*/
#include "Header.h"
using namespace std;

class Pair;
class DataManager;
class logManager;

class World {
protected:
	// pairs vector, holding all point pairs
	vector<Pair*> _pairs;
	// the current point number
	int _point_num;
	// the data manager for the world, world is only deal with object-leveltion operation, but dm will do operation on cpu/gpu array directly
	DataManager *_dm;
	// log object
	logManager *_log;

public:
	// world is initiated empty
	World();
	// insert the row to the end
	void insert(vector<double> row);
	// remove the given row
	void remove_rows(vector<int> idx);
	// calculate the array based on key
	void calculate(std::string key);
	// return array value based on key
	vector<double> report(std::string key);
	// do block on the given data, return groups
	vector<vector<int> > blocks(double delta);
	// convert the input int list into bool list value
	vector<bool> convert_int_list(vector<int> &list);

	~World();

public:
	enum{DIST, SLHC};
};

#endif