#ifndef _SNAPSHOT_
#define _SNAPSHOT_

#include "Global.h"
#include "UMACoreObject.h"
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
class DLL_PUBLIC Snapshot: public UMACoreObject{
public:
	//the total sum(a square block sum, wij+wi_j+w_ij+w_i_j), and the value in last iteration
	double _total, _total_;

protected:
	//thershold value, the current value will be written to sensor pairs, changable
	double _threshold;
	//q value, used in simulation
	double _q;
	//the initial sensor size, initial sensor is the basic sensor without amper/delay, the value is initiated
	int _initialSize;
	//the sensor pointer vector
	vector<Sensor*> _sensors;
	//the sensor pair pointer vector
	vector<SensorPair*> _sensorPairs;
	//the map of sensor id to sensor
	std::map<string, Sensor*> _sensorIdx;
	//the set of existing delay sensor hash
	std::set<size_t> _delaySensorHash;
	//the bool indicating whether auto-target is on or off
	bool _autoTarget;
	//the bool indicating whether mask will be propagated automatically
	bool _propagateMask;
	//the dataManager of the snapshot
	DataManager *_dm;
	//the delay hash
	hash<vector<bool>> delayHash;
	//snapshot type
	UMA_SNAPSHOT _type;
	//delay count
	int _delayCount;
	friend class Agent;

	friend class AmperAndTestFixture;
	friend class GenerateDelayedWeightsTestFixture;
	friend class AmperTestFixture;
	friend class AmperAndSignalsTestFixture;

public:
	//Snapshot(ifstream &file, string &log_dir);
	Snapshot(const string &uuid, UMACoreObject *parent, UMA_SNAPSHOT type = UMA_SNAPSHOT::SNAPSHOT_STATIONARY);
	Sensor *createSensor(const std::pair<string, string> &idPair, const vector<double> &diag, const vector<vector<double> > &w, const vector<vector< bool> > &b);
	void deleteSensor(const string &sensorId);
	vector<vector<string> > getSensorInfo() const;

	void generateObserve(vector<bool> &observe);
	vector<bool> generateSignal(const vector<string> &mids);
	vector<bool> generateSignal(const vector<AttrSensor*> &m);

	/*
	self-enrichment and derichment function
	*/
	virtual void ampers(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &idPairs);
	virtual void delays(const vector<vector<bool> > &lists, const vector<std::pair<string, string> > &idPairs);
	virtual void pruning(const vector<bool> &signal);
	virtual void updateTotal(double phi, bool active);

	/*
	---------------------GET FUNCTION----------------------
	*/
	SensorPair *getSensorPair(const Sensor *sensor1, const Sensor *sensor2);
	AttrSensor *getAttrSensor(int idx);
	AttrSensor *getAttrSensor(const string &measurableId);
	AttrSensorPair *getAttrSensorPair(int mIdx1, int mIdx2);
	AttrSensorPair *getAttrSensorPair(const string &mid1, const string &mid2);
	vector<bool> getAmperList(const string &sensorId);
	vector<string> getAmperListID(const string &sensorId);
	Sensor *getSensor(const string &sensorId);
	DataManager *getDM() const;

	const double &getTotal() const;
	const double &getOldTotal() const;
	const double &getQ() const;
	const double &getThreshold() const;
	const bool &getAutoTarget() const;
	const bool &getPropagateMask() const;
	const int &getInitialSize() const;
	const UMA_SNAPSHOT &getType() const;
	const int getDealyCount();
	/*
	---------------------GET FUNCTION----------------------
	*/

	/*
	---------------------SET FUNCTION----------------------
	*/
	void setThreshold(const double &threshold);
	void setQ(const double &q);
	void setAutoTarget(const bool &autoTarget);
	void setPropagateMask(const bool &propagateMask);
	void setInitialSize(const int &initialSize);
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
	void amperand(int mid1, int mid2, bool merge, const std::pair<string, string> &idPair);
	virtual void generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair);
};

/*
Qualitative Snapshot
*/
class DLL_PUBLIC SnapshotQualitative : public Snapshot {
public:
	SnapshotQualitative(const string &uuid, UMACoreObject *parent);
	virtual ~SnapshotQualitative();
	virtual void updateTotal(double phi, bool active);
	virtual void generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair);
};

/*
Discounted Snapshot
*/
class DLL_PUBLIC SnapshotDiscounted : public Snapshot {
public:
	SnapshotDiscounted(const string &uuid, UMACoreObject *parent);
	virtual ~SnapshotDiscounted();
	virtual void updateTotal(double phi, bool active);
	virtual void generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair);
};

/*
Empirical Snapshot
*/
class DLL_PUBLIC SnapshotEmpirical : public Snapshot {
public:
	SnapshotEmpirical(const string &uuid, UMACoreObject *parent);
	virtual ~SnapshotEmpirical();
	virtual void updateTotal(double phi, bool active);
	virtual void generateDelayedWeights(int mid, bool merge, const std::pair<string, string> &idPair);
	void updateQ();
	void addT();

protected:
	//this value indicate how many times the snapshot is active
	int _t;
};
#endif
