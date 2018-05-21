#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include "Global.h"
using namespace std;

class AttrSensor;
class Sensor;
class AttrSensorPair;
class SensorPair;
class Agent;
class DataManager;
/*
----------------Snapshot Base Class-------------------
*/

/*
This is Snapshot class, it is owed by Agent class.
Snapshot is a base class, based on needs, virtual function can have different implementation
The majority of GPU operation start on the Snapshot object, but this object SHOULD NOT be accessed directly by python code directly
All means of operation on Snapshot should go through Agent class
*/
class DLL_PUBLIC Snapshot{
public:
	//the total sum(a square block sum, wij+wi_j+w_ij+w_i_j), and the value in last iteration
	double _total, _total_;

protected:
	//thershold value, the current value will be written to sensor pairs, changable
	double _threshold;
	//q value, used in simulation
	double _q;
	//the initial sensor size, initial sensor is the basic sensor without amper/delay, the value is initiated
	int _initial_size;
	//the snapshot id, not changable
	const string _uuid;
	//the sensor pointer vector
	vector<Sensor*> _sensors;
	//the sensor pair pointer vector
	vector<SensorPair*> _sensor_pairs;
	//the map of sensor id to sensor
	std::map<string, Sensor*> _sensor_idx;
	//the set of existing delay sensor hash
	std::set<size_t> _delay_sensor_hash;
	//record current sensor num in the current Snapshot
	int t;
	//the bool indicating whether auto-target is on or off
	bool _auto_target;
	//the bool indicating whether mask will be propagated automatically
	bool _propagate_mask;
	//the dataManager of the snapshot
	DataManager *_dm;
	//the snapshot's dependency chain
	const string _dependency;
	//the delay hash
	hash<vector<bool>> delay_hash;
	//snapshot type
	const int _type;
	//delay count
	int _delay_count;
	friend class Agent;

	friend class AmperAndTestFixture;
	friend class GenerateDelayedWeightsTestFixture;
	friend class AmperTestFixture;
	friend class AmperAndSignalsTestFixture;

public:
	//Snapshot(ifstream &file, string &log_dir);
	Snapshot(const string &uuid, const string &dependency, const int type = AGENT_TYPE::STATIONARY);
	Sensor *add_sensor(const std::pair<string, string> &id_pair, const vector<double> &diag, const vector<vector<double> > &w, const vector<vector< bool> > &b);
	void delete_sensor(const string &sensor_id);
	vector<vector<string> > getSensorInfo() const;

	void generateObserve(vector<bool> &observe);
	vector<bool> generateSignal(const vector<string> &mids);
	vector<bool> generateSignal(const vector<AttrSensor*> &m);

	/*
	self-enrichment and derichment function
	*/
	virtual void ampers(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &id_pairs);
	virtual void delays(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &id_pairs);
	virtual void pruning(const vector<bool> &signal);
	virtual void update_total(double phi, bool active);

	/*
	---------------------GET FUNCTION----------------------
	*/
	SensorPair *getSensorPair(const Sensor *sensor1, const Sensor *sensor2) const;
	AttrSensor *getAttrSensor(int idx) const;
	AttrSensor *getAttrSensor(const string &measurable_id) const;
	AttrSensorPair *getAttrSensorPair(int m_idx1, int m_idx2) const;
	AttrSensorPair *getAttrSensorPair(const string &mid1, const string &mid2) const;
	vector<bool> getAmperList(const string &sensor_id) const;
	vector<string> getAmperListID(const string &sensor_id) const;
	Sensor *getSensor(const string &sensor_id) const;
	DataManager *getDM() const;

	const double &getTotal() const;
	const double &getOldTotal() const;
	const double &getQ() const;
	const double &getThreshold() const;
	const bool &getAutoTarget() const;
	const bool &getPropagateMask() const;
	const int &getInitialSize() const;
	const int &getType() const;
	const int getDealyCount();
	/*
	---------------------GET FUNCTION----------------------
	*/

	/*
	---------------------SET FUNCTION----------------------
	*/
	void setThreshold(const double &threshold);
	void setQ(const double &q);
	void setAutoTarget(const bool &auto_target);
	void setPropagateMask(const bool &propagate_mask);
	void setInitialSize(const int &initial_size);
	void setInitialSize();
	void setTotal(const double &total);
	void setOldTotal(const double &total_);
	/*
	---------------------SET FUNCTION----------------------
	*/

	/*
	---------------------COPY FUNCTION---------------------
	before using the copy function, have to make sure all necessary variable are in place
	*/
	//void copy_test_data(Snapshot *snapshot);
	/*
	---------------------COPY FUNCTION---------------------
	*/
	//void save_snapshot(ofstream &file);
	virtual ~Snapshot();

protected:
	void amper(const vector<int> &list, const std::pair<string, string> &uuid);
	void amperand(int mid1, int mid2, bool merge, const std::pair<string, string> &id_pair);
	virtual void generate_delayed_weights(int mid, bool merge, const std::pair<string, string> &id_pair);
};

/*
Qualitative Snapshot
*/
class DLL_PUBLIC Snapshot_qualitative : public Snapshot {
public:
	Snapshot_qualitative(string uuid, string dependency);
	virtual ~Snapshot_qualitative();
	virtual void update_total(double phi, bool active);
	virtual void generate_delayed_weights(int mid, bool merge, const std::pair<string, string> &id_pair);

protected:
	//double q;
};
#endif
