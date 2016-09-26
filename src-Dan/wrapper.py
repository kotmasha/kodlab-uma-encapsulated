import snapshot_platform_new
import atexit
import UMA_NEW

def getLog(wrapper):
    logging_info=wrapper.get_log()

    file=open("log.txt",'w')
    file.write("#This is the testing log file\n\n")
    file.write("-----------performance stats--------------\n")
    file.write("UPDATE_WEIGHT:\n")
    write_stats_info(file,logging_info.STAT_UPDATE_WEIGHT)
    file.write("ORIENT_ALL:\n")
    write_stats_info(file,logging_info.STAT_ORIENT_ALL)
    file.write("PROPAGATION:\n")
    write_stats_info(file,logging_info.STAT_PROPAGATION)
    file.write("-----------performance stats--------------\n\n")
    file.write(logging_info.str_init)
    file.close()

def write_stats_info(file, perf_stats):
    file.write("num:   "+str(perf_stats.n)+"\n")
    file.write("time  :"+str(round(perf_stats.acc_t,3))+"\n")
    file.write("avg_t :"+str(round(perf_stats.avg_t,3))+"\n")
    file.write("\n")

class wrapper:
    def __init__(self,agent,type,threshold,using_worker,using_log,q=0):
        self.agent=agent
        if type=='EMPIRICAL':
            self.brain=UMA_NEW.Agent_Empirical(threshold,using_worker,using_log)
        elif type=='DISTRIBUTED':
            self.brain=UMA_NEW.Agent_Distributed(threshold,using_worker,using_log)
        elif type=='DISCOUNTED':
            self.brain=UMA_NEW.Agent_Discounted(threshold,q,using_worker,using_log)
        if using_log:
            atexit.register(getLog,self)
    
    ### The function called from Agent.cpp to extend the parts of the 
    ### snapshot on GPU with a sensor defined as the conjunction of 
    ### the sensors that are marked $True$ in $signal$.
    def amper(self,signal):
        self.brain.conjunction(signal) 

    def sendSignal(self):
        self.brain.setSignal(self.agent._OBSERVE._VAL.tolist())

    def decide(self,mode,param):
        if type(param) is list:
            self.brain.decide(mode,param,'')
        else:
            self.brain.decide(mode,[],param)

    def up(self,sig):
        self.brain.up_GPU(sig._VAL.tolist())
        sig_up=self.brain.getSignal()
        #affected_workers=self.brain.getAffectedWorkers()
        return snapshot_platform_new.Signal(sig_up)#,snapshot_platform_new.Signal(affected_workers)

    def getValue(self):
        self.agent._CURRENT=snapshot_platform_new.Signal(self.brain.getCurrent())
        decision=self.brain.getDecision()
        message=self.brain.getMessage()
        projected_signal=self.brain.selected_projected_signal
        #touched_workers=self.brain.selected_touched_workers
        return decision,projected_signal,message#,touched_workers,message 

    def get_log_update_weight(self):
        n=self.brain.get_n_update_weight()
        t=self.brain.get_t_update_weight()
        return n,t,t/n

    def get_log_orient_all(self):
        n=self.brain.get_n_orient_all()
        t=self.brain.get_t_orient_all()
        return n,t,t/n

    def get_log_propagation(self):
        n=self.brain.get_n_propagation()
        t=self.brain.get_t_propagation()
        return n,t,t/n

    def init(self):
        tmp=[sen._NAME for sen in self.agent._SENSORS]
        self.brain.initData(self.agent._NAME,self.agent._SIZE/2,self.agent._CONTEXT.keys(),self.agent._CONTEXT.values(),tmp,self.agent._EVALS,self.agent._GENERALIZED_ACTIONS)

    def amper(self,signals):
        self.brain.initNewSensor(signals.to_list())

    def get_log(self):
        return self.brain.get_log()
        
        