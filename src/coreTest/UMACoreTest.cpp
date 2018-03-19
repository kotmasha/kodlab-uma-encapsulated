#include <iostream>
#include "gtest/gtest.h"
#include "UMACoreTestFixture.h"
#include "UMAException.h"
#include "World.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "Sensor.h"
#include "SensorPair.h"
#include "AttrSensor.h"
#include "AttrSensorPair.h"

TEST(world_test, world_agent_test) {
	World::add_agent("test_agent1");
	World::add_agent("test_agent2");
	World::add_agent("test_agent3");
	World::add_agent("test_agent4");

	EXPECT_NO_THROW(World::getAgent("test_agent1"));
	EXPECT_THROW(World::getAgent("test_agent0"), UMAException);
	vector<string> s = { "test_agent1", "test_agent2", "test_agent3", "test_agent4" };
	EXPECT_EQ(s, World::getAgentInfo());

	World::delete_agent("test_agent1");
	s = {"test_agent2", "test_agent3", "test_agent4" };
	EXPECT_EQ(s, World::getAgentInfo());

	World::delete_agent("test_agent3");
	s = { "test_agent2", "test_agent4" };
	EXPECT_EQ(s, World::getAgentInfo());

	EXPECT_THROW(World::getAgent("test_agent3"), UMAException);
	EXPECT_THROW(World::delete_agent("test_agent3"), UMAException);
	
	World::delete_agent("test_agent2");
	World::delete_agent("test_agent4");
}

TEST(agent_test, agent_snapshot_test) {
	Agent *agent = new Agent("agent", "");

	agent->add_snapshot_stationary("snapshot1");
	agent->add_snapshot_stationary("snapshot2");
	agent->add_snapshot_stationary("snapshot3");
	agent->add_snapshot_stationary("snapshot4");

	EXPECT_NO_THROW(agent->getSnapshot("snapshot2"));
	EXPECT_THROW(agent->getSnapshot("snapshot0"), UMAException);
	vector<string> s = { "snapshot1", "snapshot2", "snapshot3", "snapshot4" };
	EXPECT_EQ(agent->getSnapshotInfo(), s);

	agent->delete_snapshot("snapshot2");
	s = { "snapshot1", "snapshot3", "snapshot4" };
	EXPECT_EQ(agent->getSnapshotInfo(), s);

	agent->delete_snapshot("snapshot4");
	s = { "snapshot1", "snapshot3"};
	EXPECT_EQ(agent->getSnapshotInfo(), s);

	EXPECT_THROW(agent->getSnapshot("snapshot2"), UMAException);
	EXPECT_THROW(agent->delete_snapshot("snapshot2"), UMAException);

	EXPECT_EQ(agent->getT(), 0);
	agent->setT(100);
	EXPECT_EQ(agent->getT(), 100);

	delete agent;
}

TEST(snapshot_test, snapshot_create_sensor_test) {
	Snapshot *snapshot = new Snapshot("snapshot", "");

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	snapshot->add_sensor(sensor1, diag, w, b);
	snapshot->add_sensor(sensor2, diag, w, b);
	snapshot->add_sensor(sensor3, diag, w, b);
	snapshot->add_sensor(sensor4, diag, w, b);

	EXPECT_NO_THROW(snapshot->getSensor("s1"));
	EXPECT_NO_THROW(snapshot->getSensor("cs2"));
	EXPECT_THROW(snapshot->getSensor("s0"), UMAException);

	snapshot->delete_sensor("s1");
	vector<vector<string>> s = { {"s2", "cs2"}, {"s3", "cs3"}, {"s4", "cs4"} };
	EXPECT_EQ(s, snapshot->getSensorInfo());
	snapshot->delete_sensor("cs3");
	s = { { "s2", "cs2" }, { "s4", "cs4" } };
	EXPECT_EQ(s, snapshot->getSensorInfo());

	EXPECT_THROW(snapshot->getSensor("s1"), UMAException);
	EXPECT_THROW(snapshot->getSensor("cs3"), UMAException);
	EXPECT_THROW(snapshot->delete_sensor("s1"), UMAException);
	EXPECT_THROW(snapshot->delete_sensor("cs3"), UMAException);

	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("s2")->getDiag(), 0.5);

	diag = { 0.32, 0.68 };
	std::pair<string, string> sensor5 = { "s5", "cs5" };
	snapshot->add_sensor(sensor5, diag, w, b);
	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("s5")->getDiag(), 0.32);
	EXPECT_DOUBLE_EQ(snapshot->getAttrSensor("cs5")->getDiag(), 0.68);

	delete snapshot;
}

TEST(snapshot_test, snapshot_create_sensor_pair_test) {
	Snapshot *snapshot = new Snapshot("snapshot", "");

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w1, w3;
	vector<vector<double>> w0 = { {0.2, 0.8, 0.457, 0.543} };
	vector<vector<double>> w2 = { {0.2, 0.8, 0.4, 0.6}, {0.1, 0.9, 0, 1}, {0.3, 0.7, 0.8, 0.2} };
	vector<vector<bool>> b1, b3;
	vector<vector<bool>> b0 = { {true, false, true, false} };
	vector<vector<bool>> b2 = { { true, false, false, true }, {false, false, false, false}, {true, true, true, true} };
	Sensor *s1 = snapshot->add_sensor(sensor1, diag, w0, b0);
	Sensor *s2 = snapshot->add_sensor(sensor2, diag, w1, b1);
	Sensor *s3 = snapshot->add_sensor(sensor3, diag, w2, b2);
	snapshot->setThreshold(0.501);
	Sensor *s4 = snapshot->add_sensor(sensor4, diag, w3, b3);

	EXPECT_NO_THROW(snapshot->getSensorPair(snapshot->getSensor("s1"), snapshot->getSensor("cs3")));
	EXPECT_THROW(snapshot->getSensorPair(snapshot->getSensor("s0"), snapshot->getSensor("cs3")), UMAException);

	EXPECT_EQ(snapshot->getSensorPair(s1, s3), snapshot->getSensorPair(s3, s1));
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, true)->getW(), 0.2);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, false)->getW(), 0.457);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, true)->getW(), 0.457);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, false)->getW(), 0.543);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s3)->getAttrSensorPair(true, true)->getW(), 0.1);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s3, s2)->getAttrSensorPair(true, false)->getW(), 0.9);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s3, s2)->getAttrSensorPair(false, true)->getW(), 0.0);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s3)->getAttrSensorPair(false, false)->getW(), 1.0);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s2)->getAttrSensorPair(true, true)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s4)->getAttrSensorPair(true, false)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s2)->getAttrSensorPair(false, true)->getW(), 0.25);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s2, s4)->getAttrSensorPair(false, false)->getW(), 0.25);

	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(true, false)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s1, s1)->getAttrSensorPair(false, false)->getD(), false);

	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(true, false)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(false, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s3, s3)->getAttrSensorPair(false, false)->getD(), true);

	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(true, true)->getD(), true);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(true, false)->getD(), false);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(false, true)->getD(), false);
	EXPECT_EQ(snapshot->getSensorPair(s4, s4)->getAttrSensorPair(false, false)->getD(), true);

	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s1, s1)->getThreshold(), 0.125);
	EXPECT_DOUBLE_EQ(snapshot->getSensorPair(s4, s1)->getThreshold(), 0.501);

	snapshot->delete_sensor("s2");
	EXPECT_THROW(snapshot->getSensorPair(snapshot->getSensor("s2"), snapshot->getSensor("s4")), UMAException);
	EXPECT_NO_THROW(snapshot->getSensorPair(snapshot->getSensor("s1"), snapshot->getSensor("s3")), UMAException);

	delete snapshot;
}

TEST(snapshot_test, snapshot_get_set_attribute_test, ) {
	Snapshot_Stationary *snapshot = new Snapshot_Stationary("snapshot", "");
	
	EXPECT_DOUBLE_EQ(snapshot->getThreshold(), 0.125);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.0);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.0);
	EXPECT_EQ(snapshot->getAutoTarget(), false);
	EXPECT_EQ(snapshot->getPropagateMask(), false);
	EXPECT_DOUBLE_EQ(snapshot->getQ(), 0.9);

	snapshot->setThreshold(0.501);
	snapshot->setTotal(2.0);
	snapshot->setOldTotal(0.31);
	snapshot->setAutoTarget(true);
	snapshot->setPropagateMask(true);
	snapshot->setQ(0.8);

	EXPECT_DOUBLE_EQ(snapshot->getThreshold(), 0.501);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 2.0);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 0.31);
	EXPECT_EQ(snapshot->getAutoTarget(), true);
	EXPECT_EQ(snapshot->getPropagateMask(), true);
	EXPECT_DOUBLE_EQ(snapshot->getQ(), 0.8);

	snapshot->update_total(1.1, true);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.82);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 2.0);

	snapshot->update_total(1.2, false);
	EXPECT_DOUBLE_EQ(snapshot->getTotal(), 1.82);
	EXPECT_DOUBLE_EQ(snapshot->getOldTotal(), 1.82);

	delete snapshot;
}

TEST(snapshot_test, snapshot_get_entity) {
	Snapshot *snapshot = new Snapshot("snapshot", "");
	
	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<vector<double>> w;
	vector<vector<bool>> b;
	Sensor *s1 = snapshot->add_sensor(sensor1, diag, w, b);
	Sensor *s2 = snapshot->add_sensor(sensor2, diag, w, b);
	Sensor *s3 = snapshot->add_sensor(sensor3, diag, w, b);
	Sensor *s4 = snapshot->add_sensor(sensor4, diag, w, b);

	EXPECT_EQ(s1, snapshot->getSensor("s1"));
	EXPECT_EQ(s2, snapshot->getSensor("cs2"));
	EXPECT_NE(s3, snapshot->getSensor("s4"));

	EXPECT_EQ(snapshot->getAttrSensor("cs3"), snapshot->getAttrSensor(5));
	EXPECT_EQ(snapshot->getAttrSensor(2), snapshot->getAttrSensor("s2"));
	EXPECT_NE(snapshot->getAttrSensor("s3"), snapshot->getAttrSensor("cs3"));
	EXPECT_NE(snapshot->getAttrSensor(0), snapshot->getAttrSensor(1));

	EXPECT_EQ(snapshot->getAttrSensorPair("s3", "cs2"), snapshot->getAttrSensorPair(3, 4));
	EXPECT_EQ(snapshot->getAttrSensorPair("s1", "cs4"), snapshot->getAttrSensorPair(7, 0));
	EXPECT_NE(snapshot->getAttrSensorPair("s4", "s3"), snapshot->getAttrSensorPair(0, 1));

	delete snapshot;
}

TEST_F(AmperAndTestFixture, snapshot_amper_and_test1) {
	std::pair<string, string> p = {"s5", "cs5"};
	vector<vector<double>> target = test_amper_and(0, 2, true, p);

	vector<vector<double>> w = {
		{0.2},
		{0.0, 0.8},
		{0.2, 0.2, 0.4},
		{0.0, 0.6, 0.0, 0.6},
		{0.2, 0.4, 0.4, 0.2, 0.6},
		{0.0, 0.4, 0.0, 0.4, 0.0, 0.4},
		{0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8},
		{0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2},
		{0.2, 0.0, 0.2, 0.0, 0.12, 0.08, 0.16, 0.04, 0.2},
		{0.0, 0.8, 0.2, 0.6, 0.48, 0.32, 0.64, 0.16, 0.0, 0.8}
	};
	
	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}

	target = test_amper_and(4, 8, false, p);
	p = { "s6", "cs6" };

	w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.024, 0.096, 0.048, 0.072, 0.12, 0.0, 0.096, 0.024, 0.12 },
		{ 0.176, 0.704, 0.352, 0.528, 0.48, 0.4, 0.704, 0.176, 0.0, 0.88 }
	};
	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(GenerateDelayedWeightsTestFixture, snapshot_generate_delayed_weights_test1) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<bool> observe = { true, false, false, false, false, false, false, false };
	vector<vector<double>> target = test_generate_delayed_weights(0, true, p, observe);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2, 1.0 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(GenerateDelayedWeightsTestFixture, snapshot_generate_delayed_weights_test2) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<bool> observe = { false, true, false, false, false, false, false, false };
	vector<vector<double>> target = test_generate_delayed_weights(0, true, p, observe);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
		{ 0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.8, 0.2, 0.0, 1.0 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(AmperTestFixture, amper_test) {
	std::pair<string, string> p = { "s5", "cs5" };
	vector<int> list = { 3, 5, 7 };

	vector<vector<double>> target = test_amper(list, p);

	vector<vector<double>> w = {
		{ 0.2 },
		{ 0.0, 0.8 },
		{ 0.2, 0.2, 0.4 },
		{ 0.0, 0.6, 0.0, 0.6 },
		{ 0.2, 0.4, 0.4, 0.2, 0.6 },
		{ 0.0, 0.4, 0.0, 0.4, 0.0, 0.4 },
		{ 0.2, 0.6, 0.4, 0.4, 0.6, 0.2, 0.8 },
		{ 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2 },
		{ 0.016, 0.064, 0.032, 0.048, 0.048, 0.032, 0.0, 0.08, 0.08 },
		{ 0.184, 0.736, 0.368, 0.552, 0.552, 0.368, 0.8, 0.12, 0.0, 0.92 }
	};

	for (int i = 0; i < w.size(); ++i) {
		for (int j = 0; j < w[i].size(); ++j)
			EXPECT_DOUBLE_EQ(w[i][j], target[i][j]);
	}
}

TEST_F(AmperAndSignalsTestFixture, amper_and_signals_test) {
	vector<bool> observe1 = { true, false, true, false, false, true, false, true, true, false, true, false };
	vector<bool> observe2 = { true, false, false, true, false, true, false, true, false, true, true, false };

	EXPECT_EQ(test_amper_and_signals("s4", observe1), false);
	EXPECT_EQ(test_amper_and_signals("s5", observe1), true);
	EXPECT_EQ(test_amper_and_signals("cs4", observe2), true);
	EXPECT_EQ(test_amper_and_signals("cs5", observe2), false);
}

TEST(dataManager_test, get_set_test) {
	Snapshot *snapshot = new Snapshot("snapshot", "");

	std::pair<string, string> sensor1 = { "s1", "cs1" };
	std::pair<string, string> sensor2 = { "s2", "cs2" };
	std::pair<string, string> sensor3 = { "s3", "cs3" };
	std::pair<string, string> sensor4 = { "s4", "cs4" };
	vector<double> diag;
	vector<double> diag1 = { 0.2, 0.8 };
	vector<vector<double>> w1, w3;
	vector<vector<double>> w0 = { { 0.2, 0.8, 0.457, 0.543 } };
	vector<vector<double>> w2 = { { 0.2, 0.8, 0.4, 0.6 },{ 0.1, 0.9, 0, 1 },{ 0.3, 0.7, 0.8, 0.2 } };
	vector<vector<bool>> b1, b3;
	vector<vector<bool>> b0 = { { true, false, true, false } };
	vector<vector<bool>> b2 = { { true, false, false, true },{ false, false, false, false },{ true, true, true, true } };
	Sensor *s1 = snapshot->add_sensor(sensor1, diag, w0, b0);
	Sensor *s2 = snapshot->add_sensor(sensor2, diag1, w1, b1);
	Sensor *s3 = snapshot->add_sensor(sensor3, diag, w2, b2);
	snapshot->setThreshold(0.501);
	Sensor *s4 = snapshot->add_sensor(sensor4, diag, w3, b3);

	DataManager *dm = snapshot->getDM();
	//mask test
	vector<bool> mask = { false, false, false, true, true, true };
	EXPECT_THROW(dm->setMask(mask), UMAException);
	mask.push_back(false); mask.push_back(false);
	dm->setMask(mask);
	EXPECT_EQ(mask, dm->getMask());
	//load test
	vector<bool> load = { true, false, false, true, false, false };
	EXPECT_THROW(dm->setLoad(load), UMAException);
	load.push_back(false); load.push_back(false);
	dm->setLoad(load);
	EXPECT_EQ(load, dm->getLoad());
	//signals test
	vector<vector<bool>> signals = {
		{true},{false},{true},{false},{true}
	};
	EXPECT_THROW(dm->setSignals(signals), UMAException);
	signals = { {true, false, true, false, false, false},
	{false, true, false, true, true, true, false, false}
	};
	EXPECT_THROW(dm->setSignals(signals), UMAException);
	signals[0].push_back(false); signals[0].push_back(true);
	dm->setSignals(signals);
	EXPECT_EQ(signals, dm->getSignals(2));
	//lsignals test
	vector<vector<bool>> lsignals = {
		{ true },{ false },{ true },{ false },{ true }
	};
	dm->setLoad(load);
	EXPECT_THROW(dm->setLSignals(lsignals), UMAException);
	lsignals = { { true, false, true, false, false, false },
	{ false, true, false, true, true, true, false, false }
	};
	EXPECT_THROW(dm->setLSignals(lsignals), UMAException);
	lsignals[0].push_back(false); lsignals[0].push_back(true);
	dm->setLSignals(lsignals);
	lsignals = { { true, false, true, true, false, false, false, true },
	{ true, true, false, true, true, true, false, false }
	};
	EXPECT_EQ(lsignals, dm->getLSignals(2));
	//dists test
	vector<vector<int>> dists = { {1}, {2}, {3} };
	EXPECT_THROW(dm->setDists(dists), UMAException);
	dists = {
		{1, 1, 1},
		{2, 2, 2, 2},
		{3, 3, 3, 3},
		{4, 4, 4, 4}
	};
	EXPECT_THROW(dm->setDists(dists), UMAException);
	dists[0].push_back(1);
	EXPECT_NO_THROW(dm->setDists(dists));
	//current test
	vector<bool> current = { true, false, false, true, true, true };
	EXPECT_THROW(dm->setCurrent(current), UMAException);
	current.push_back(false); current.push_back(false);
	dm->setCurrent(current);
	EXPECT_EQ(current, dm->getCurrent());
	//target test
	vector<bool> target = { false, true, false, true, true, true };
	EXPECT_THROW(dm->setTarget(target), UMAException);
	target.push_back(false); target.push_back(true);
	dm->setTarget(target);
	EXPECT_EQ(target, dm->getTarget());
	//observe test
	vector<bool> observe = { true, false, false, true, true, false };
	EXPECT_THROW(dm->setObserve(observe), UMAException);
	observe.push_back(true); observe.push_back(false);
	dm->setObserve(observe);
	EXPECT_EQ(observe, dm->getObserve());

	//weights test
	vector<vector<double>> t_weight2d = {
		{0.2},
		{0.457, 0.543},
		{0.25, 0.25, 0.25},
		{0.25, 0.25, 0.25, 0.25},
		{0.2, 0.8, 0.1, 0.9, 0.3},
		{0.4, 0.6, 0.0, 1.0, 0.8, 0.2},
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 },
		{0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 },
	};
	vector<double> t_weight1d = { 0.2, 0.457, 0.543, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
	0.2, 0.8, 0.1, 0.9, 0.3, 0.4, 0.6, 0.0, 1.0, 0.8, 0.2, 0.25, 0.25, 0.25, 0.25,0.25,
	0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 };
	vector<vector<double>> r_weight2d = dm->getWeight2D();
	vector<double> r_weight1d = dm->getWeight();
	for (int i = 0; i < t_weight2d.size(); ++i) {
		for (int j = 0; j < t_weight2d[i].size(); ++j)
			EXPECT_DOUBLE_EQ(t_weight2d[i][j], r_weight2d[i][j]);
	}
	for (int i = 0; i < t_weight1d.size(); ++i) {
		EXPECT_DOUBLE_EQ(t_weight1d[i], r_weight1d[i]);
	}

	//dirs test
	vector<vector<bool>> t_dirs2d = {
		{ true },
		{ true, false },
		{ false, false, true },
		{ false, false, false, true },
		{ true, false, false, false, true },
		{ false, true, false, false, true, true },
		{ false, false, false, false, false, false, true },
		{ false, false, false, false, false, false, false, true },
	};
	vector<bool> t_dirs1d = { true, true, false, false, false, true, false, false, false, true,
		true, false, false, false, true, false, true, false, false, true, true, false, false, false, false,
		false, false, true, false, false, false, false, false, false, false, true };
	vector<vector<bool>> r_dirs2d = dm->getDir2D();
	vector<bool> r_dirs1d = dm->getDir();
	EXPECT_EQ(t_dirs2d, r_dirs2d);
	EXPECT_EQ(t_dirs1d, r_dirs1d);

	//threshold test
	vector<vector<double>> t_threshold2d = {
		{0.125},
		{0.125, 0.125},
		{0.125, 0.125, 0.125},
		{0.501, 0.501, 0.501, 0.501}
	};
	vector<double> t_threshold1d = { 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.501, 0.501, 0.501, 0.501 };
	vector<vector<double>> r_threshold2d = dm->getThreshold2D();
	for (int i = 0; i < t_threshold2d.size(); ++i) {
		for (int j = 0; j < t_threshold2d[i].size(); ++j)
			EXPECT_DOUBLE_EQ(r_threshold2d[i][j], t_threshold2d[i][j]);
	}

	vector<double> r_threshold1d = dm->getThresholds();
	for (int i = 0; i < r_threshold1d.size(); ++i)
		EXPECT_DOUBLE_EQ(r_threshold1d[i], t_threshold1d[i]);

	//diag test
	vector<double> t_diag_ = { 0.5, 0.5, 0.2, 0.8, 0.5, 0.5 };
	vector<double> r_diag_ = dm->getDiagOld();
	for (int i = 0; i < t_diag_.size(); ++i)
		EXPECT_DOUBLE_EQ(t_diag_[i], r_diag_[i]);

	vector<double> t_diag = {0.5, 0.5, 0.2, 0.8, 0.5, 0.5};
	vector<double> r_diag = dm->getDiag();
	for (int i = 0; i < t_diag.size(); ++i)
		EXPECT_DOUBLE_EQ(t_diag[i], r_diag[i]);

	delete snapshot;
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test1) {
	EXPECT_THROW(test_uma_core_dataflow(0, 5), UMAException);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test2) {
	test_uma_core_dataflow(0, 3);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test3) {
	test_uma_core_dataflow(2, 4);
}

TEST_F(UMACoreDataFlowTestFixture, uma_core_dataflow_test4) {
	test_uma_core_dataflow(0, 4);
}

TEST(sensor_test, set_amper_list) {
	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	Sensor *s1 = new Sensor(p1, 1.0, 0);
	Sensor *s2 = new Sensor(p2, 1.0, 1);
	s1->setAmperList(0);
	s1->setAmperList(1);
	s2->setAmperList(2);
	s2->setAmperList(3);
	s2->setAmperList(4);
	vector<int> v1 = { 0, 1 };
	vector<int> v2 = { 2, 3, 4 };
	vector<int> v3 = { 0, 1, 2, 3, 4 };
	EXPECT_EQ(s1->getAmperList(), v1);
	EXPECT_EQ(s2->getAmperList(), v2);
	s1->setAmperList(s2);
	EXPECT_EQ(s1->getAmperList(), v3);

	delete s1;
	delete s2;
}

TEST(sensor_test, get_set_idx) {
	pair<string, string> p1 = { "s1", "cs1" };
	Sensor *s1 = new Sensor(p1, 1.0, 0);
	EXPECT_EQ(s1->getIdx(), 0);
	s1->setIdx(2);
	EXPECT_EQ(s1->getIdx(), 2);

	delete s1;
}

TEST(sensor_test, copy_amper_list_test) {
	pair<string, string> p1 = { "s1", "cs1" };
	pair<string, string> p2 = { "s2", "cs2" };
	pair<string, string> p3 = { "s3", "cs3" };
	pair<string, string> p4 = { "s4", "cs4" };
	pair<string, string> p5 = { "s5", "cs5" };

	Sensor *s1 = new Sensor(p1, 1.0, 0);
	Sensor *s2 = new Sensor(p2, 1.0, 1);
	Sensor *s3 = new Sensor(p3, 1.0, 2);
	Sensor *s4 = new Sensor(p4, 1.0, 3);
	Sensor *s5 = new Sensor(p5, 1.0, 4);

	bool *ampers = new bool[30];
	ampers[0] = 0; ampers[1] = 0;
	ampers[2] = 0; ampers[3] = 0; ampers[4] = 0; ampers[5] = 0;
	ampers[6] = 0; ampers[7] = 0; ampers[8] = 0; ampers[9] = 0; ampers[10] = 0; ampers[11] = 0;
	ampers[12] = 0; ampers[13] = 0; ampers[14] = 0; ampers[15] = 0; ampers[16] = 0; ampers[17] = 0; ampers[18] = 0; ampers[19] = 0;
	ampers[20] = 0; ampers[21] = 0; ampers[22] = 0; ampers[23] = 0; ampers[24] = 0; ampers[25] = 0; ampers[26] = 0; ampers[27] = 0; ampers[28] = 0; ampers[29] = 0;

	s4->setAmperList(0);
	s4->setAmperList(3);
	s4->setAmperList(5);
	s4->setAmperList(6);

	s4->copyAmperList(ampers);

	EXPECT_EQ(ampers[12], 1);
	EXPECT_EQ(ampers[13], 0);
	EXPECT_EQ(ampers[14], 0);
	EXPECT_EQ(ampers[15], 1);
	EXPECT_EQ(ampers[16], 0);
	EXPECT_EQ(ampers[17], 1);
	EXPECT_EQ(ampers[18], 1);
	EXPECT_EQ(ampers[19], 0);

	delete s1;
	delete s2;
	delete s3;
	delete s4;
	delete s5;
}

TEST(sensor_pair_test, sensor_pair_test) {
	pair<string, string> p1 = { "s1", "cs1" };

	Sensor *s1 = new Sensor(p1, 1.0, 0);

	double *thresholds = new double;
	double *weights = new double[4];
	bool *dirs = new bool[4];

	SensorPair *sp = new SensorPair(s1, s1, 0.25, 1.0);

	sp->setAllPointers(weights, dirs, thresholds);
	sp->values_to_pointers();

	EXPECT_DOUBLE_EQ(sp->getThreshold(), 0.25);
	sp->setThreshold(0.5);
	EXPECT_DOUBLE_EQ(sp->getThreshold(), 0.5);

	AttrSensorPair *asp1 = sp->getAttrSensorPair(true, true);
	AttrSensorPair *asp2 = sp->getAttrSensorPair(true, false);
	EXPECT_DOUBLE_EQ(asp1->getW(), 0.25);
	EXPECT_DOUBLE_EQ(asp2->getW(), 0.25);
	EXPECT_DOUBLE_EQ(asp1->getD(), true);
	EXPECT_DOUBLE_EQ(asp2->getD(), false);
	asp1->setD(false);
	asp1->setW(0.8);
	EXPECT_DOUBLE_EQ(asp1->getW(), 0.8);
	EXPECT_DOUBLE_EQ(asp1->getD(), false);

	delete s1;
	delete sp;
	delete thresholds;
	delete dirs;
	delete weights;
}

TEST(attr_sensor_test, attr_sensor_test) {
	AttrSensor *as = new AttrSensor("attr_sensor", 0, true, 0.5);
	double *diag = new double[2];
	double *diag_ = new double[2];
	bool *observe = new bool[2];
	bool *observe_ = new bool[2];

	EXPECT_EQ(as->getIdx(), 0);
	as->setIdx(1);
	EXPECT_EQ(as->getIdx(), 1);

	EXPECT_EQ(as->getIsOriginPure(), true);
	as->setIsOriginPure(false);
	EXPECT_EQ(as->getIsOriginPure(), false);

	as->setDiagPointers(diag, diag_);
	as->setObservePointers(observe, observe_);
	as->values_to_pointers();

	EXPECT_EQ(as->getDiag(), 0.5);
	EXPECT_EQ(as->getOldDiag(), 0.5);

	as->setDiag(0.6);
	as->setOldDiag(0.7);
	EXPECT_EQ(as->getDiag(), 0.6);
	EXPECT_EQ(as->getOldDiag(), 0.7);

	delete as;
	delete[] diag, diag_;
	delete[] observe, observe_;
}

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}