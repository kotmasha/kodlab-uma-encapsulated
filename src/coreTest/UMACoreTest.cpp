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

int main(int argc, char** argv)
{
	testing::InitGoogleTest(&argc, argv);
	(void)RUN_ALL_TESTS();
	std::getchar(); // keep console window open until Return keystroke
}