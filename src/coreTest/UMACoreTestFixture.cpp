#include "UMACoreTestFixture.h"

WorldTestFixture::WorldTestFixture() {}

WorldTestFixture::~WorldTestFixture() {}

AmperAndTestFixture::AmperAndTestFixture(){
	snapshot = new Snapshot("snapshot", "");
	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	snapshot->add_sensor(p0, diag, w0, b0);
	snapshot->add_sensor(p1, diag, w1, b1);
	snapshot->add_sensor(p2, diag, w2, b2);
	snapshot->add_sensor(p3, diag, w3, b3);
}

AmperAndTestFixture::~AmperAndTestFixture() {
	delete snapshot;
}

vector<vector<double>> AmperAndTestFixture::test_amper_and(int mid1, int mid2, bool merge, std::pair<string, string> &p) {
	snapshot->amperand(mid1, mid2, merge, p);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = snapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->v_w);
		}
		w.push_back(tmp);
	}
	
	return w;
}

GenerateDelayedWeightsTestFixture::GenerateDelayedWeightsTestFixture() {
	snapshot = new Snapshot("snapshot", "");
	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	Sensor *s0 = snapshot->add_sensor(p0, diag, w0, b0);
	Sensor *s1 = snapshot->add_sensor(p1, diag, w1, b1);
	Sensor *s2 = snapshot->add_sensor(p2, diag, w2, b2);
	Sensor *s3 = snapshot->add_sensor(p3, diag, w3, b3);
	
	snapshot->getAttrSensor(0)->_vdiag = 0.2;
	snapshot->getAttrSensor(1)->_vdiag = 0.8;
	snapshot->getAttrSensor(2)->_vdiag = 0.4;
	snapshot->getAttrSensor(3)->_vdiag = 0.6;
	snapshot->getAttrSensor(4)->_vdiag = 0.6;
	snapshot->getAttrSensor(5)->_vdiag = 0.4;
	snapshot->getAttrSensor(6)->_vdiag = 0.8;
	snapshot->getAttrSensor(7)->_vdiag = 0.2;
}

GenerateDelayedWeightsTestFixture::~GenerateDelayedWeightsTestFixture() {
	delete snapshot;
}

vector<vector<double>> GenerateDelayedWeightsTestFixture::test_generate_delayed_weights(int mid, bool merge, const std::pair<string, string> &id_pair, vector<bool> &observe) {
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		*(snapshot->getAttrSensor(i)->_observe) = observe[i];
		*(snapshot->getAttrSensor(i)->_observe_) = observe[i];
	}
	snapshot->generate_delayed_weights(mid, merge, id_pair);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = snapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->v_w);
		}
		w.push_back(tmp);
	}

	return w;
}

AmperTestFixture::AmperTestFixture() {
	snapshot = new Snapshot("snapshot", "");
	vector<vector<double>> w0 = { { 0.2, 0.0, 0.0, 0.8 } };
	vector<vector<double>> w1 = { { 0.2, 0.2, 0.0, 0.6 },{ 0.4, 0.0, 0.0, 0.6 } };
	vector<vector<double>> w2 = { { 0.2, 0.4, 0.0, 0.4 },{ 0.4, 0.2, 0.0, 0.4 },{ 0.6, 0.0, 0.0, 0.4 } };
	vector<vector<double>> w3 = { { 0.2, 0.6, 0.0, 0.2 },{ 0.4, 0.4, 0.0, 0.2 },{ 0.6, 0.2, 0.0, 0.2 },{ 0.8, 0.0, 0.0, 0.2 } };
	vector<vector<bool>> b0 = { { true, false, false, true } };
	vector<vector<bool>> b1 = { { false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b2 = { { false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	vector<vector<bool>> b3 = { { false, false, false, false },{ false, false, false, false },{ false, false, false, false },{ true, false, false, true } };
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<double> diag;
	snapshot->add_sensor(p0, diag, w0, b0);
	snapshot->add_sensor(p1, diag, w1, b1);
	snapshot->add_sensor(p2, diag, w2, b2);
	snapshot->add_sensor(p3, diag, w3, b3);
}

AmperTestFixture::~AmperTestFixture() {
	delete snapshot;
}

vector<vector<double>> AmperTestFixture::test_amper(const vector<int> &list, const std::pair<string, string> &uuid) {
	snapshot->amper(list, uuid);

	vector<vector<double>> w;
	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		vector<double> tmp;
		for (int j = 0; j <= i; ++j) {
			AttrSensorPair *asp = snapshot->getAttrSensorPair(i, j);
			tmp.push_back(asp->v_w);
		}
		w.push_back(tmp);
	}

	return w;
}

AmperAndSignalsTestFixture::AmperAndSignalsTestFixture() {
	snapshot = new Snapshot("snapshot", "");
	vector<vector<double>> w;
	vector<vector<bool>> b;
	std::pair<string, string> p0 = { "s0", "cs0" };
	std::pair<string, string> p1 = { "s1", "cs1" };
	std::pair<string, string> p2 = { "s2", "cs2" };
	std::pair<string, string> p3 = { "s3", "cs3" };
	vector<std::pair<string, string>> p4 = { { "s4", "cs4" } };
	vector<std::pair<string, string>> p5 = { { "s5", "cs5" } };
	vector<double> diag;
	snapshot->add_sensor(p0, diag, w, b);
	snapshot->add_sensor(p1, diag, w, b);
	snapshot->add_sensor(p2, diag, w, b);
	snapshot->add_sensor(p3, diag, w, b);

	vector<vector<bool>> list1 = { {0, 0, 0, 1, 0, 1, 0, 1} };
	vector<vector<bool>> list2 = { {1, 0, 0, 0, 0, 0, 0, 0, 1, 0} };
	snapshot->ampers(list1, p4);
	snapshot->delays(list2, p5);
}

AmperAndSignalsTestFixture::~AmperAndSignalsTestFixture() {
	delete snapshot;
}

bool AmperAndSignalsTestFixture::test_amper_and_signals(const string &sid, vector<bool> &observe) {
	Sensor *s = snapshot->getSensor(sid);

	for (int i = 0; i < 2 * snapshot->_sensors.size(); ++i) {
		*(snapshot->getAttrSensor(i)->_observe) = observe[i];
		*(snapshot->getAttrSensor(i)->_observe_) = observe[i];
	}

	return snapshot->amper_and_signals(s);
}