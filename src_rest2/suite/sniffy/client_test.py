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
