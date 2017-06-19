/*
This is the world wrapper class
It is used to manage all c++ side agent object
*/
import som_platform
import from agent_wrapper import *
import UMA
import numpy as np
from numpy.random import randint as rnd

class world:
    def __init__(self, experiment):
        self._experiment = experiment
        self._world = UMA.World()

    def add_agent(self, name, uuid):
        self._world.add_agent(name, uuid)

    def up(self, sig, stability):
        sig_new={}
        for token in ['plus','minus']:
            self._snapshots[token].up_GPU(sig.value_all().tolist(),stability)
            sig_new[token]=som_platform.Signal(self._snapshots[token].getUp())
            #print self._snapshots[token].getName()
        return sig_new
            
    def amper(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.ampers([signal._VAL.tolist() for signal in signals])

    def delay(self, signals):
        for snapshot_name, snapshot in self._snapshots.iteritems():
            snapshot.delays([signal._VAL.tolist() for signal in signals])

        
