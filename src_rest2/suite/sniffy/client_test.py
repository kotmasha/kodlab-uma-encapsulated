from Service_World import *
from Service_Agent import *
from Service_Snapshot import *
from random import randint
import time

service = UMA_service()
world = ServiceWorld(service)
agent = world.add_agent('uuid_agent')
snapshot = agent.add_snapshot('plus')
for i in range(5):
    snapshot.add_sensor('sensor' + str(i), 'sensorc' + str(i))
snapshot.validate(5)
snapshot.add_implication('sensor0', 'sensor1')
snapshot.add_implication('sensor1', 'sensor2')
snapshot.add_implication('sensor2', 'sensor3')
snapshot.add_implication('sensor1', 'sensor4')
snapshot.add_implication('sensor2', 'sensor4')
snapshot.add_implication('sensor4', 'sensor3')
#print snapshot.make_up([True, False, False, False, False, False, False, False, False, False])
print snapshot.make_npdirs()