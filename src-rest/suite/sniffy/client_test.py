from Service_World import *
from Service_Agent import *
from Service_Snapshot import *
from random import randint
import time

service = UMA_service()
world = ServiceWorld(service)
"""
dicts = world.load('sniffy1d')
print dicts

"""
agent = world.add_agent('uuid_agent')
snapshot = agent.add_snapshot('plus')
for i in range(5):
    snapshot.add_sensor('sensor' + str(i), 'sensorc' + str(i))
signals = [False, True, True, False, False, True, True, False, True, False]
snapshot.delay([signals], ['delay_uuid', 'delayc_uuid'])
print ServiceSensor('uuid_agent', 'plus', 'delay_uuid', service).getAmperList()
print ServiceSensor('uuid_agent', 'plus', 'delayc_uuid', service).getAmperList()
agent1 = world.add_agent('uuid_agent1')
agent2 = world.add_agent('uuid_agent2')
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
signals = [False, True, True, False, False, True, True, False, True, False]
snapshot11.delay([signals], ['delay_uuid'])
snapshot12.delay([signals], ['delay_uuid'])
snapshot21.delay([signals], ['delay_uuid'])
snapshot22.delay([signals], ['delay_uuid'])
sensor = ServiceSensor('uuid_agent1', 'uuid_agent1_plus', 'delay_uuid', service)
sensor.getAmperList()
for i in range(10):
    signals = [False, True, True, False, False, True, True, False, True, False]
    agent1.make_decision(signals, 1.0, True)
    agent2.make_decision(signals, 0.5, True)
dicts = {}
#world.save('test', dicts)
