import som_platform
import atexit
import UMA
import numpy as np
from numpy.random import randint as rnd

class brain:
    def __init__(self, agent):
        self.agent = agent
        self._snapshots = {}

    def init(self):
        #initialize agent
        self.a = UMA.Agent("")
        self.add_snapshot('plus')
        self.add_snapshot('minus')

    def add_snapshot(self, name):
        discount = self.agent._PARAMS[0]
        cal_target = self.agent._PARAMS[1]
        #snapshot = UMA.Agent_Forgetful(discount)
        snapshot = UMA.Snapshot_Stationary(self.agent._SIZE / 2, 0.25, name, self.agent._SENSORS, discount, cal_target)
        self._snapshots[name] = snapshot
        #self._snapshots[name] = self.a.add_snapshot("stationary", self.agent._SIZE / 2, 0.125, name, self.agent._SENSORS, discount, cal_target)  

    def up(self, sig, stability):
        sig_new={}
        for token in ['plus','minus']:
            self._snapshots[token].up_GPU(sig.value_all().tolist(),stability)
            sig_new[token]=som_platform.Signal(self._snapshots[token].getUp())
            #print self._snapshots[token].getName()
        return sig_new


    #def propagate(self, signal, load):
    #    self.brain.propagate_GPU(signal._VAL.tolist(), load._VAL.tolist())
    #    result = self.brain.getLoad()
    #    return som_platform.Signal(result)

    def get_log_update_weight(self):
        n = self.brain.get_n_update_weight()
        t = self.brain.get_t_update_weight()
        return n, t, t / n

    def get_log_orient_all(self):
        n=self.brain.get_n_orient_all()
        t=self.brain.get_t_orient_all()
        return n, t, t / n

    def get_log_propagation(self):
        n = self.brain.get_n_propagation()
        t = self.brain.get_t_propagation()
        return n, t, t / n

            
    def amper(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.ampers([signal._VAL.tolist() for signal in signals])

    def delay(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.delays([signal._VAL.tolist() for signal in signals])

    def blocks(self, dists, delta):
        return self.brain.blocks_GPU(dists, delta)
        
    def saveData(self):
        self.brain.savingData('test_data')

    def get_delta_weight_sum(self, signal):
        return self.brain.get_delta_weight_sum(signal._VAL.tolist())
        
