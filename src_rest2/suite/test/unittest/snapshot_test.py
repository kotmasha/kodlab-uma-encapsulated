#This is the test script for testing existing snapshot endpoint
#Test CRUD functionality
from test_wrapper import *

"""
This plan is:

0 create an agent, test_agent
1 create an snapshot without id: 400 error
2 create an snapshot with an invalid id(number): 400 error
3 create an snapshot without agent_id: 400 error
4 create an snapshot with an invalid agent_id: 400 error
5 create an snapshot: 201 created
6 create the same snapshot: 409 conflict

7 create an implication, without sensors field: 400 error
8 create an implication, sensor1->sensor2: 400 error
9 create sensor1, sensor2, validate
10 create an implication, sensor1->sensor2: 201 created
11 get implication from sensor1->sensor2: true
12 get implication from sensor2->sensor1: false
13 delete implication from sensor1->sensor2
14 get implication from sensor1->sensor2: false

15 change q without giving snapshot id: 400 error
16 chagne q to 'abc': 400 error
17 change q to 0.123: 200 OK
18 change threshold to 1.222: 200 OK
19 change auto target to 2.2: 400 error
20 change auto target to True: 200 OK

21 delete snapshot without snapshot id: 400 error
22 delete snapshot with invalid id: 400 error
23 delete snapshot: 200 ok

24 put request after delete: 404 not found
25 get request after delete: 404 not found
26 clean up agent
"""

test = test_wrapper()
#Snapshot creation
test.T(method='POST', url='/UMA/object/agent', status_code=201, data={'agent_id': 'test_agent'})
test.T(method='POST', url='/UMA/object/snapshot', status_code=400, data={'agent_id': 'test_agent'}, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/snapshot', data={'agent_id': 'test_agent', 'snapshot_id': 123.45}, status_code=400, message="Cannot parsing the field")
test.T(method='POST', url='/UMA/object/snapshot', data={'snapshot_id': 'test_snapshot'}, status_code=400,  message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/snapshot', data={'agent_id':'abc', 'snapshot_id': 'test_snapshot'}, status_code=404,  message="Cannot find the agent id")
test.T(method='POST', url='/UMA/object/snapshot', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=201)
test.T(method='POST', url='/UMA/object/snapshot', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=409, message="Cannot create a duplicate snapshot")

#Implication test
test.T(method='POST', url='/UMA/object/snapshot/implication', status_code=400, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/snapshot/implication', status_code=400, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor1', 'to_sensor': 'sensor2'}, message="Cannot find the sensor")
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1', 'c_sid': 'c_sensor1'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor2', 'c_sid': 'c_sensor2'}, status_code=201)
test.T(method='POST', url='/UMA/object/snapshot/init', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'initial_sensor_size': 2}, status_code=201)
test.T(method='POST', url='/UMA/object/snapshot/implication', status_code=201, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor1', 'to_sensor': 'sensor2'}, message="Implication created")
test.T(method='GET', url='/UMA/object/snapshot/implication', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor1', 'to_sensor': 'sensor2'}, kwargs={'implication': True})
test.T(method='GET', url='/UMA/object/snapshot/implication', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor2', 'to_sensor': 'sensor1'}, kwargs={'implication': False})
test.T(method='DELETE', url='/UMA/object/snapshot/implication', status_code=200, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor1', 'to_sensor': 'sensor2'}, message="Implication deleted")
test.T(method='GET', url='/UMA/object/snapshot/implication', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'from_sensor': 'sensor1', 'to_sensor': 'sensor2'}, kwargs={'implication': False})

test.T(method='PUT', url='/UMA/object/snapshot', status_code=400, query={'agent_id': 'test_agent'}, data={'q': 0.123}, message="Coming request is missing necessary fields")
test.T(method='PUT', url='/UMA/object/snapshot', status_code=400, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'q': 'abc'}, message="Cannot parsing the field")
test.T(method='PUT', url='/UMA/object/snapshot', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'q': 0.123})
test.T(method='PUT', url='/UMA/object/snapshot', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'threshold': 1.222})
test.T(method='PUT', url='/UMA/object/snapshot', status_code=400, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'auto_target': 2.2}, message="Cannot parsing the field")
test.T(method='PUT', url='/UMA/object/snapshot', status_code=200, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'auto_target': False})

test.T(method='GET', url='/UMA/object/snapshot', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=200, message="Get snapshot info", kwargs={'sensors': [{'sensor': {'m': 'sensor1', 'cm': 'c_sensor1'}}, {'sensor': {'m': 'sensor2', 'cm': 'c_sensor2'}}],
    'sizes':{'sensor_count': 2, 'sensor_pair_count': 3, '_sensor_size': 2, '_sensor_size_max': 3, '_sensor2d_size': 3, '_sensor2d_size_max': 6, '_measurable_size': 4, '_measurable_size_max': 6, '_measurable2d_size': 10, '_measurable2d_size_max': 21, '_mask_amper_size':6, '_mask_amper_size_max': 12},
    'q': 0.123,'threshold': 1.222, 'auto_target': False})

test.T(method="DELETE", url='/UMA/object/snapshot', status_code=400, data={'agent_id': 'test_agent'}, message="Coming request is missing necessary fields")
test.T(method="DELETE", url='/UMA/object/snapshot', status_code=404, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot1'}, message="Cannot find the agent to delete")
test.T(method="DELETE", url='/UMA/object/snapshot', status_code=200, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'})

test.T(method='PUT', url='/UMA/object/snapshot', status_code=404, query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'auto_target': False}, message="Cannot find the snapshot id")
test.T(method='GET', url='/UMA/object/snapshot', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=404, message="Cannot find the snapshot id")
test.T(method="DELETE", url='/UMA/object/agent', status_code=200, data={'agent_id': 'test_agent'})
print "Snapshot test passed"