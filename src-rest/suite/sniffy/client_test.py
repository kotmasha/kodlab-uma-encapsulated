from Service_World import *
from Service_Agent import *
from Service_Snapshot import *
from random import randint
import time

world = ServiceWorld()
agent = world.add_agent('agent', 'uuid_agent')
snapshot = agent.add_snapshot('plus')
for i in range(20):
    snapshot.add_sensor(str(i))
snapshot.validate()
for i in range(100):
    signals = [randint(0,1) is 1 for i in range(10)]
    phi = 1
    active = True
    snapshot.simulation(signals, phi, active)
