#ifndef _UMACORETESTFIXTURE_
#define _UMACORETESTFIXTURE_

#include "Snapshot.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"
#include "DataManager.h"
#include "uma_base.cuh"
#include "gtest/gtest.h"

class WorldTestFixture {
public:
	WorldTestFixture();
	~WorldTestFixture();
};

class AgentTestFixture {

};

class AmperAndTestFixture : public ::testing::Test {
public:
	AmperAndTestFixture();
	~AmperAndTestFixture();

	vector<vector<double>> test_amper_and(int mid1, int mid2, bool merge, std::pair<string, string> &p);

protected:
	Snapshot *snapshot;
};

class GenerateDelayedWeightsTestFixture : public::testing::Test {
public:
	GenerateDelayedWeightsTestFixture();
	~GenerateDelayedWeightsTestFixture();

	vector<vector<double>> test_generate_delayed_weights(int mid, bool merge, const std::pair<string, string> &id_pair, vector<bool> &observe);

protected:
	Snapshot *snapshot;
};

class AmperTestFixture : public::testing::Test {
public:
	AmperTestFixture();
	~AmperTestFixture();

	vector<vector<double>> test_amper(const vector<int> &list, const std::pair<string, string> &uuid);

protected:
	Snapshot *snapshot;
};

class AmperAndSignalsTestFixture : public::testing::Test {
public:
	AmperAndSignalsTestFixture();
	~AmperAndSignalsTestFixture();

	bool test_amper_and_signals(const string &sid, vector<bool> &observe);

protected:
	Snapshot *snapshot;
};

class UMACoreDataFlowTestFixture : public::testing::Test {
public:
	UMACoreDataFlowTestFixture();
	~UMACoreDataFlowTestFixture();

	void test_uma_core_dataflow(int start_idx, int end_idx);

protected:
	DataManager *dm;
	vector<Sensor *> sensors;
	vector<SensorPair *> sensorPairs;
	vector<double> weights;
	vector<double> thresholds;
	vector<bool> dirs;
	vector<double> diag;
	vector<bool> observe;
};

#endif
