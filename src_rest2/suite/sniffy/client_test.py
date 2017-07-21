from Service_World import *
from Service_Agent import *
from Service_Snapshot import *
from random import randint
import time

service = UMA_service()
world = ServiceWorld(service)
world.load('C:/Users/siqiHuang/Desktop/UMA/UMA4.0/UMA/UMA/UMA/sniffy1d')
"""
agent1 = world.add_agent('agent1', 'uuid_agent1')
agent2 = world.add_agent('agent2', 'uuid_agent2')
snapshot11 = agent1.add_snapshot('plus')
snapshot12 = agent1.add_snapshot('minus')
snapshot21 = agent2.add_snapshot('plus')
snapshot22 = agent2.add_snapshot('minus')
for i in range(5):
    snapshot11.add_sensor("sensor" + str(i))
    snapshot12.add_sensor("sensor" + str(i))
    snapshot21.add_sensor("sensor" + str(i))
    snapshot22.add_sensor("sensor" + str(i))
snapshot11.validate()
snapshot12.validate()
snapshot21.validate()
snapshot22.validate()
for i in range(10):
    signals = [False, True, True, False, False, True, True, False, True, False]
    agent1.make_decision(signals, 1.0, True)
    agent2.make_decision(signals, 0.5, True)
world.save('test')
"""
