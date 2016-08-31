from snapshot_platform_new import *
import UMA_NEW

class wrapper:
    def __init__(self,agent,type,threshold,q=0):
        self.agent=agent
        if type=='EMPIRICAL':
            self.brain=UMA_NEW.Agent_Empirical(threshold)
        elif type=='DISTRIBUTED':
            self.brain=UMA_NEW.Agent_Distributed(threshold)
        elif type=='DISCOUNTED':
            self.brain=UMA_NEW.Agent_Discounted(threshold,q)
    
    def sendSignal(self):
        self.brain.setSignal(self.agent._OBSERVE._VAL.tolist())

    def decide(self,mode,param):
        if type(param) is list:
            self.brain.decide(mode,param,'')
        else:
            self.brain.decide(mode,[],param)

    def getValue(self):
        decision=self.brain.getDecision()
        message=self.brain.getMessage()
        return decision,message

    def init(self):
        tmp=[sen._NAME for sen in self.agent._SENSORS]
        self.brain.initData(self.agent._NAME,self.agent._SIZE/2,self.agent._CONTEXT.keys(),self.agent._CONTEXT.values(),tmp,self.agent._EVALS,self.agent._GENERALIZED_ACTIONS)
