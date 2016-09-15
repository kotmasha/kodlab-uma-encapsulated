import snapshot_platform_new
import atexit
import UMA_NEW

def getLog(wrapper):
    file=open("log.txt",'w')
    file.write("#This is the testing log file\n\n")
    file.write("UPDATE_WEIGHT:\n")
    n,t,d=wrapper.get_log_update_weight()
    writeInfo(n,t,d,file)
    file.write("ORIENT_ALL:\n")
    n,t,d=wrapper.get_log_orient_all()
    writeInfo(n,t,d,file)
    file.write("PROPAGATION:\n")
    n,t,d=wrapper.get_log_propagation()
    writeInfo(n,t,d,file)
    file.close()

def writeInfo(n,t,d,file):
    file.write("num:   "+str(n)+"\n")
    file.write("time  :"+str(round(t,3))+"\n")
    file.write("avg_t :"+str(round(d,3))+"\n")
    file.write("\n")

class wrapper:
    def __init__(self,agent,type,threshold,q=0):
        self.agent=agent
        if type=='EMPIRICAL':
            self.brain=UMA_NEW.Agent_Empirical(threshold)
        elif type=='DISTRIBUTED':
            self.brain=UMA_NEW.Agent_Distributed(threshold)
        elif type=='DISCOUNTED':
            self.brain=UMA_NEW.Agent_Discounted(threshold,q)
        atexit.register(getLog,self)
    
    def sendSignal(self):
        self.brain.setSignal(self.agent._OBSERVE._VAL.tolist())

    def decide(self,mode,param):
        if type(param) is list:
            self.brain.decide(mode,param,'')
        else:
            self.brain.decide(mode,[],param)

    def up(self,sig):
        self.brain.up_GPU(sig._VAL.tolist())
        SIGB=self.brain.getSignal()
        discard=self.brain.getAffectedWorkers()
        return snapshot_platform_new.Signal(SIGB),snapshot_platform_new.Signal(discard)

    def getValue(self):
        self.agent._CURRENT=self.brain.getCurrent()
        decision=self.brain.getDecision()
        message=self.brain.getMessage()
        projected_signal=self.brain.selected_projected_signal
        touched_workers=self.brain.selected_touched_workers
        return decision,projected_signal,touched_workers,message 

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
