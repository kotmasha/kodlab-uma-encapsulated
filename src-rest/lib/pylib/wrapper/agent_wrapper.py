/*
This is the wrapper class
This wrapper is only from py agent -> c++ agent
All operation on snapshot/sensor/sensorpair should be through agent
*/
import numpy as np
from numpy.random import randint as rnd

class brain:
    def __init__(self, agent):
        self.agent = agent
        self._world = agent._EXPERIMENT._world._world
        self._uuid = agent._MID

    def init(self):
        self.add_snapshot_stationary('plus')
        self.add_snapshot_stationary('minus')

    def add_snapshot_stationary(self, name):
        discount = self.agent._PARAMS[0]
        cal_target = self.agent._PARAMS[1]
        #snapshot = UMA.Snapshot_Stationary(self.agent._SIZE / 2, 0.25, name, self.agent._SENSORS, discount, cal_target)
        #self._snapshots[name] = snapshot
        self._world.add_snapshot(self.agent._SIZE / 2, 0.125, name, self.agent._SENSORS, self.agent._SENSORS, discount, cal_target)

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
            
    def amper(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.ampers([signal._VAL.tolist() for signal in signals])

    def delay(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.delays([signal._VAL.tolist() for signal in signals])
        
    def saveData(self):
        self.brain.savingData('test_data')

    def get_delta_weight_sum(self, signal):
        return self.brain.get_delta_weight_sum(signal._VAL.tolist())
        
