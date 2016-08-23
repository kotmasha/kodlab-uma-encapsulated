#include "Snapshot.h"

Snapshot::Snapshot(){
	Gdir=NULL;
}

Snapshot::Snapshot(int type){
	Gdir=NULL;
	this->type=type;
}

Snapshot::Snapshot(double threshold,int type){
	Gdir=NULL;
	this->threshold=threshold;
	this->type=type;
}

vector<bool> Snapshot::initMask(vector<int> actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> result;
	for(int i=0;i<size;++i){
		bool flag=false;
		for(int j=0;j<actions_list.size();++j){
			if(i==actions_list[j]){
				flag=true;
				break;
			}
		}
		result.push_back(flag);
	}
	return result;
}

Snapshot::~Snapshot(){}

vector<bool> Snapshot::halucinate(vector<int> action_list){//halucinate
	halucinate_GPU(action_list);
	return this->getLoad();
}