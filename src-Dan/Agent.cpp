#include "Agent.h"

/*
----------------Agent Base Class-------------------
*/

Agent::Agent(int type,double threshold,bool using_worker,bool using_log){
	Gdir=NULL;
	this->type=type;
	this->threshold=threshold;
	this->is_worker_solution=using_worker;
	this->is_log_on=using_log;
	this->logging_info=new logging(using_log);
	test=1;
}

Agent::~Agent(){}

static int randInt(int n){
	return rand()%n;
}

void Agent::decide(string mode,vector<int> param1,string param2){//the decide function
	update_state_GPU(mode=="decide");
	
	projected_signal.clear();
	touched_workers.clear();
	if(mode=="execute"){
		if(checkParam(param1)){
			decision=translate(param1);
			message="Executing [";
			for(int i=0;i<param1.size();++i){
				message+=(" "+std::to_string(param1[i]));
			}
			message+="]";
		}
		else{
			cout<<"Illegal input for execution by "<<name<<" --- Aborting!"<<endl;
			exit(0);
		}
	}
	else if(mode=="decide"){
		if(param2=="ordered"){
			vector<vector<bool> > responses;
			for(int i=0;i<generalized_actions.size();++i){
				responses.push_back(halucinate(generalized_actions[i]));
			}
			vector<int> best_responses;
			for(int i=0;i<evals_names.size();++i){
				for(int j=0;j<responses.size();++j){
					if(responses[j][name_to_num[evals_names[i]]]){
						best_responses.push_back(j);
					}
				}
				if(!best_responses.empty()){
					int randIndex=randInt(best_responses.size());
					decision=translate(generalized_actions[best_responses[randIndex]]);
					message=evals_names[i]+", ";
					
					selected_projected_signal=projected_signal[randIndex];
					selected_touched_workers=touched_workers[randIndex];

					for(int i=0;i<decision.size();++i) message+=(decision[i]+" ");	
					break;
				}
			}
			if(best_responses.empty()){
				int randIndex=randInt(generalized_actions.size());
				decision=translate(generalized_actions[randIndex]);
				message="random";

				selected_projected_signal=projected_signal[randIndex];
				selected_touched_workers=touched_workers[randIndex];
			}
		}
		else if(checkParam(param2)){
			vector<vector<bool> > responses;
			for(int i=0;i<generalized_actions.size();++i){
				responses.push_back(halucinate(generalized_actions[i]));
			}
			vector<int> best_responses;
			for(int i=0;i<responses.size();++i){
				if(responses[i][name_to_num[param2]]){
					best_responses.push_back(i);
				}
			}
			
			if(!best_responses.empty()){
				int randIndex=randInt(best_responses.size());
				decision=translate(generalized_actions[best_responses[randIndex]]);
				message=param2+", ";
				for(int i=0;i<decision.size();++i) message+=(decision[i]+" ");

				selected_projected_signal=projected_signal[randIndex];
				selected_touched_workers=touched_workers[randIndex];
			}
			else{
				int randIndex=randInt(generalized_actions.size());
				decision=translate(generalized_actions[randIndex]);
				message=param2+", random";

				selected_projected_signal=projected_signal[randIndex];
				selected_touched_workers=touched_workers[randIndex];
			}
		}
		else{
			cout<<"Invalid decision criterion "<<param2<<" --- Aborting!"<<endl;
		}
	}
	else{
		cout<<"Invalid operation mode for agent "<<name<<" --- Aborting!"<<endl;
	}
}

vector<string> Agent::translate(vector<int> index_list){
	vector<string> name;
	for(int i=0;i<index_list.size();++i){
		name.push_back(sensors_names[index_list[i]]);
	}
	return name;
}

bool Agent::checkParam(vector<int> param){
	for(int i=0;i<generalized_actions.size();++i){
		if(param==generalized_actions[i]) return true;	
	}
	return false;
}

bool Agent::checkParam(string param){
	for(int i=0;i<evals_names.size();++i){
		if(param==evals_names[i]) return true;
	}
	return false;
}

vector<string> Agent::getDecision(){
	return decision;
}

string Agent::getMessage(){
	return message;
}

vector<bool> Agent::initMask(vector<int> &actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> result;
	for(int i=0;i<measurableSize;++i){
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

vector<bool> Agent::halucinate(vector<int> &action_list){//halucinate
	halucinate_GPU(action_list);
	projected_signal.push_back(this->getLoad());
	touched_workers.push_back(this->getAffectedWorkers());
	return this->getLoad();
}

void Agent::appendSensor(int id1, int id2, vector<vector<double> > &data, bool merge){
	vector<double> tmp,tmp_c;
	double w12=data[2*id2][2*id1],w_12=data[2*id2][2*id1+1];
	double w1_2=data[2*id2+1][2*id1],w_1_2=data[2*id2+1][2*id1+1];
	for(int i=0;i<data.back().size();++i){
		if(i==id1*2||i==id2*2) tmp.push_back(w12);
		else if(i==id1*2+1||i==id2*2+1) tmp.push_back(0);
		else if(i%2==0) tmp.push_back(w12*data[i][i]);
		else tmp.push_back(w12*(1-data[i-1][i-1]));
	}
	tmp.push_back(tmp[0]+tmp[1]);
	data.push_back(tmp);

	for(int i=0;i<data.back().size();++i){
		if(i==id1*2) tmp_c.push_back(w1_2);
		else if(i==id2*2) tmp_c.push_back(w_12);
		else if(i==id1*2+1) tmp_c.push_back(w_12+w_1_2);
		else if(i==id2*2+1) tmp_c.push_back(w1_2+w_1_2);
		else if(i%2==0) tmp_c.push_back((1-w12)*data[i][i]);
		else tmp_c.push_back((1-w12)*(1-data[i-1][i-1]));
	}
	tmp_c.push_back(tmp_c[0]+tmp_c[1]);
	data.push_back(tmp_c);

	if(!merge){
		data.pop_back();
		data.pop_back();
		for(int i=0;i<tmp.size()-3;++i){
			data[data.size()-2][i]=tmp[i];
			data.back()[i]=tmp_c[i];
		}
		data[data.size()-2].back()=tmp.back();
		data.back().back()=tmp_c.back();
	}
}

void Agent::addSensor(vector<int> &list, vector<vector<double> > &data){
	if(list.size()<2) return;//if list size is not larger than 1, return
	appendSensor(list[0],list[1],data,true);
	//end for two
	for(int i=2;i<list.size();++i){
		appendSensor(data.size()/2,list[i],data,false);
	}
}

vector<vector<double> > Agent::addSensors(vector<vector<int> > &list){
	vector<vector<double> > data=getVectorWeight();
	for(int i=0;i<list.size();++i){
		addSensor(list[i],data);
	}
	return data;
}

//those three functions down there are get functions for the variable in C++
vector<bool> Agent::getCurrent(){
	vector<bool> result;
	for(int i=0;i<measurableSize;++i){
		result.push_back(Gcurrent[i]);
	}
	return result;
}

vector<bool> Agent::getSignal(){
	vector<bool> result;
	for(int i=0;i<measurableSize;++i){
		result.push_back(Gsignal[i]);
	}
	return result;
}

vector<bool> Agent::getLoad(){
	vector<bool> result;
	for(int i=0;i<measurableSize;++i){
		result.push_back(Gload[i]);
	}
	return result;
}

vector<bool> Agent::getAffectedWorkers(){
	vector<bool> result;
	for(int i=0;i<workerSize;++i){
		result.push_back(Gaffected_worker[i]);
	}
	return result;
}

vector<vector<bool> > Agent::getDir(){
	vector<vector<bool> > result;
	for(int i=0;i<measurableSize;++i){
		vector<bool> tmp;
		for(int j=0;j<measurableSize;++j){
			tmp.push_back(Gdir[i*measurableSize+j]);
		}
		result.push_back(tmp);
	}
	return result;
}

logging Agent::get_log(){
	logging_info->finalize_log();
	return *logging_info;
}

/*
----------------Agent Base Class-------------------
*/

/*
----------------Agent_Empirical Class-------------------
*/

Agent_Empirical::Agent_Empirical(double threshold,bool using_worker,bool using_log)
	:Agent(EMPIRICAL,threshold,using_worker,using_log){
}

Agent_Empirical::~Agent_Empirical(){
}

/*
----------------Agent_Empirical Class-------------------
*/

/*
----------------Agent_Distributed Class-------------------
*/

Agent_Distributed::Agent_Distributed(double threshold,bool using_worker,bool using_log)
	:Agent(DISTRIBUTED,threshold,using_worker,using_log){

}

Agent_Distributed::~Agent_Distributed(){}



/*
----------------Agent_Distributed Class-------------------
*/

/*
----------------Agent_Discounted Class-------------------
*/

Agent_Discounted::Agent_Discounted(double threshold,double q,bool using_worker,bool using_log)
	:Agent(DISCOUNTED,threshold,using_worker,using_log){
	this->q=q;
}

Agent_Discounted::~Agent_Discounted(){}

/*
----------------Agent_Discounted Class-------------------
*/