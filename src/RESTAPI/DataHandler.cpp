#include "DataHandler.h"
#include "World.h"
#include "Experiment.h"
#include "Agent.h"
#include "Snapshot.h"
#include "DataManager.h"
#include "UMAException.h"
#include "Logger.h"

static Logger serverLogger("Server", "log/UMA_server.log");
DataHandler::DataHandler(const string &handlerName): UMARestHandler(handlerName) {
}

void DataHandler::handleCreate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle POST " + requestUrl, false, &serverLogger);
}

void DataHandler::handleUpdate(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();

	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();

	if (requestUrl == "/UMA/data/observe" && request.checkDataField("observe")) {
		vector<bool> observe = request.getBool1dData("observe");
		dm->setObserve(observe);

		request.setMessage("observe value set");
		return;
	}
	else if (requestUrl == "/UMA/data/current" && request.checkDataField("current")) {
		vector<bool> current = request.getBool1dData("current");
		dm->setCurrent(current);

		request.setMessage("Customized current value set");
		return;
	}
	else if (requestUrl == "/UMA/data/target" && request.checkDataField("target")) {
		vector<bool> target = request.getBool1dData("target");
		dm->setTarget(target);

		request.setMessage("Customized target value set");
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to update", false, &serverLogger);
}

void DataHandler::handleRead(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();

	const string experimentId = request.getStringQuery("experiment_id");
	const string agentId = request.getStringQuery("agent_id");
	const string snapshotId = request.getStringQuery("snapshot_id");

	Experiment *experiment = World::instance()->getExperiment(experimentId);
	Agent *agent = experiment->getAgent(agentId);
	Snapshot *snapshot = agent->getSnapshot(snapshotId);
	DataManager *dm = snapshot->getDM();

	if (requestUrl == "/UMA/data/current") {
		vector<bool> current = dm->getCurrent();
		request.setMessage("get current value");
		request.setData("current", current);
		return;
	}
	else if(requestUrl == "/UMA/data/prediction"){
		vector<bool> prediction = dm->getPrediction();
		request.setMessage("get prediction value");
		request.setData("prediction", prediction);
		return;
	}
	else if (requestUrl == "/UMA/data/target") {
		vector<bool> target = dm->getTarget();
		request.setMessage("get target value");
		request.setData("target", target);
		return;
	}
	else if (requestUrl == "/UMA/data/observe") {
		vector<bool> observe = dm->getObserve();
		request.setMessage("get observe value");
		request.setData("observe", observe);
		return;
	}
	else if (requestUrl == "/UMA/data/weights") {
		vector<vector<double> > weights = dm->getWeight2D();
		request.setMessage("get weights value");
		request.setData("weights", weights);
		return;
	}
	else if (requestUrl == "/UMA/data/dirs") {
		vector<vector<bool> > dirs = dm->getDir2D();
		request.setMessage("get dirs value");
		request.setData("dirs", dirs);
		return;
	}
	else if (requestUrl == "/UMA/data/thresholds") {
		vector<vector<double> > thresholds = dm->getThreshold2D();
		request.setMessage("get thresholds value");
		request.setData("thresholds", thresholds);
		return;
	}
	else if (requestUrl == "/UMA/data/negligible") {
		vector<bool> negligible = dm->getNegligible();
		request.setMessage("get negligible value");
		request.setData("negligible", negligible);
		return;
	}
	else if (requestUrl == "/UMA/data/dataSize") {
		std::map<string, int> sizeInfo = dm->getSizeInfo();
		std::map<string, int> convertedSizeInfo = dm->convertSizeInfo(sizeInfo);
		request.setMessage("get size info");
		request.setData("sizes", convertedSizeInfo);
		return;
	}
	else if (requestUrl == "/UMA/data/npdirs") {
		vector<vector<bool>> npdirs = dm->getNPDir2D();
		request.setMessage("get npdirs info");
		request.setData("npdirs", npdirs);
		return;
	}
	else if (requestUrl == "/UMA/data/propagateMasks") {
		vector<vector<bool>> propagateMasks= dm->getNpdirMasks();
		request.setMessage("get propagate masks info");
		request.setData("propagate_masks", propagateMasks);
		return;
	}
	else if (requestUrl == "/UMA/data/maskAmper") {
		vector<vector<bool>> maskAmper = dm->getMaskAmper2D();
		request.setMessage("get mask amper info");
		request.setData("mask_amper", maskAmper);
		return;
	}
	else if (requestUrl == "/UMA/data/all") {
		vector<bool> current = dm->getCurrent();
		request.setMessage("get current value");
		request.setData("current", current);

		vector<bool> prediction = dm->getPrediction();
		request.setMessage("get prediction value");
		request.setData("prediction", prediction);

		vector<bool> target = dm->getTarget();
		request.setMessage("get target value");
		request.setData("target", target);

		vector<vector<double> > weights = dm->getWeight2D();
		request.setMessage("get weights value");
		request.setData("weights", weights);

		vector<vector<bool> > dirs = dm->getDir2D();
		request.setMessage("get dirs value");
		request.setData("dirs", dirs);

		vector<vector<double> > thresholds = dm->getThreshold2D();
		request.setMessage("get thresholds value");
		request.setData("thresholds", thresholds);

		vector<vector<bool>> npdirs = dm->getNPDir2D();
		request.setMessage("get npdirs info");
		request.setData("npdirs", npdirs);
		
		return;
	}

	throw UMABadOperationException("The coming put request has nothing to get", false, &serverLogger);
}

void DataHandler::handleDelete(UMARestRequest &request) {
	const string requestUrl = request.getRequestUrl();
	throw UMABadOperationException("Cannot handle DELETE " + requestUrl, false, &serverLogger);
}

DataHandler::~DataHandler() {}