#This is the test script for testing existing sensor endpoint
#Test CRUD functionality
from test_wrapper import *

"""
This plan is:

0 create an agent, test_agent
1 create a snapshot, test_snapshot
2 create a sensor without id: 400 error
3 create a sensor without cid: 400 error
4 create a sensor with an invalid id(number): 400 error
5 create a sensor without agent_id: 400 error
6 create a sensor without snapshot_id: 400 error
7 create a sensor with an invalid agent_id: 400 error
8 create a sensor1 with an invalid snapshot_id: 400 error
9 create a sensor: 201 created
10 create the same sensor: 409 conflict

11 get the sensor1 info: 200 ok
12 get the c_sensor1 info: 200 ok

13 change the snapshot threshold to 0.9: 200 ok
14 create sensor2: 201 created
15-22 check 8 combination of sensor pair sensor1, sensor2 info: 200 ok

23 create sensor3,4,5: 201 ok
24 get c_sensor3, sensor5: 200 ok
25 snapshot init: 201 ok

26 delete sensor without id: 400 error
27 delete sensor sensor3: 200 ok
28 get c_sensor1, sensor2: 200 ok
29 get sensor3, c_sensor3: 404 error
30 get c_sensor4, sensor5: 200 ok
31 clean up agent: 200 ok
32 get agent after deletion: 404 error
"""

test = test_wrapper()
test.T(method='POST', url='/UMA/object/agent', status_code=201, data={'agent_id': 'test_agent'})
test.T(method='POST', url='/UMA/object/snapshot', status_code=201, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'})
test.T(method='POST', url='/UMA/object/sensor', status_code=400, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'c_sid': 'c_test_sensor'}, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/sensor', status_code=400, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'test_sensor'}, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 123.45}, status_code=400, message="Cannot parsing the field")
test.T(method='POST', url='/UMA/object/sensor', data={'snapshot_id': 'test_snapshot', 'sensor_id': 'test_sensor', 'c_sid': 'c_test_sensor'}, status_code=400,  message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'sensor_id': 'test_sensor', 'c_sid': 'c_test_sensor'}, status_code=400,  message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id':'abc', 'snapshot_id': 'test_snapshot', 'sensor_id': 'test_agent', 'c_sid': 'c_test_sensor'}, status_code=404,  message="Cannot find the agent id")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id':'test_agent', 'snapshot_id': 'abc', 'sensor_id': 'test_agent', 'c_sid': 'c_test_sensor'}, status_code=404,  message="Cannot find the snapshot id")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1', 'c_sid': 'c_sensor1'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1', 'c_sid': 'c_sensor1'}, status_code=409, message="Cannot create a duplicate sensor")

test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1'}, status_code=200, kwargs={'amper_list': [], 'idx': 0})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor1'}, status_code=200, kwargs={'amper_list': [], 'idx': 0})

test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor1', 'sensor2': 'sensor1'}, status_code=200, kwargs={'threshold': 0.125})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor1', 'sensor2': 'c_sensor1'}, status_code=200, kwargs={'threshold': 0.125})

test.T(method='PUT', url='/UMA/object/snapshot', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'threshold': 0.9})
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor2', 'c_sid': 'c_sensor2'}, status_code=201)

test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor1', 'sensor2': 'sensor2'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor1', 'sensor2': 'c_sensor2'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'c_sensor1', 'sensor2': 'c_sensor2'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'c_sensor1', 'sensor2': 'c_sensor2'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor2', 'sensor2': 'sensor1'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'sensor2', 'sensor2': 'c_sensor1'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'c_sensor2', 'sensor2': 'sensor1'}, status_code=200, kwargs={'threshold': 0.9})
test.T(method='GET', url='/UMA/object/sensor_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor1': 'c_sensor2', 'sensor2': 'c_sensor1'}, status_code=200, kwargs={'threshold': 0.9})

test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor3', 'c_sid': 'c_sensor3'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor4', 'c_sid': 'c_sensor4'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor5', 'c_sid': 'c_sensor5'}, status_code=201)
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor3'}, status_code=200, kwargs={'amper_list': [], 'idx': 2})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor5'}, status_code=200, kwargs={'amper_list': [], 'idx': 4})
test.T(method='POST', url='/UMA/object/snapshot/init', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'initial_sensor_size': 2}, status_code=201)

test.T(method='DELETE', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=400, message="Coming request is missing necessary fields")
test.T(method='DELETE', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor3'}, status_code=200)
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor1'}, status_code=200, kwargs={'amper_list': [], 'idx': 0})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor2'}, status_code=200, kwargs={'amper_list': [], 'idx': 1})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor4'}, status_code=200, kwargs={'amper_list': [], 'idx': 2})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor5'}, status_code=200, kwargs={'amper_list': [], 'idx': 3})

test.T(method="DELETE", url='/UMA/object/agent', status_code=200, data={'agent_id': 'test_agent'})
test.T(method='GET', url='/UMA/object/sensor', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor5'}, status_code=404, message="Cannot find the agent id!")
print "Sensor test passed"