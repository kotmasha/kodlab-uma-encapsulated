#This is the test script for testing existing agent endpoint
#Test R functionality
from test_wrapper import *

"""
This plan is:

1 create agent/snapshot
2 create sensor1-5
3 read measurable without id: 400 error
4 read sensor1: 200 ok
5 read sensor1-5: 200/400 ok/error
6 read sensorpair: 200/400 ok/error
7 delete sensor3
8 read sensor1-5: 200/404 ok/error
9 read sensorpair: 200/404 ok/error
10 update measurable pair without id, 400 error
11 update measurable pair, 200 ok
12 read measurable pair, 200 ok
11 delete the agent, clean environment
"""

test = test_wrapper()
test.T(method='POST', url='/UMA/object/agent', status_code=201, data={'agent_id': 'test_agent'})
test.T(method='POST', url='/UMA/object/snapshot', status_code=201, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'})
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1', 'c_sid': 'c_sensor1'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor2', 'c_sid': 'c_sensor2'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor3', 'c_sid': 'c_sensor3'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor4', 'c_sid': 'c_sensor4'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor5', 'c_sid': 'c_sensor5'}, status_code=201)
test.T(method='POST', url='/UMA/object/snapshot/init', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'initial_sensor_size': 2}, status_code=201)

test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=400, message="Coming request is missing necessary fields")
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor1'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': True})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor1'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': False})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor3'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': True})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor3'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': False})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor5'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': True})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor5'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': False})

test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1'}, status_code=400, message="Coming request is missing necessary fields")
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor1'}, status_code=200, kwargs={'d': True, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'c_sensor1'}, status_code=200, kwargs={'d': True, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor2'}, status_code=200, kwargs={'d': False, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'sensor1'}, status_code=200, kwargs={'d': False, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor3'}, status_code=200, kwargs={'d': True, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor4', 'measurable2': 'c_sensor3'}, status_code=200, kwargs={'d': False, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, status_code=200, kwargs={'d': False, 'w': 0.25})

test.T(method='DELETE', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'c_sensor3'}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor1'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': True})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor1'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': False})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor5'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': True})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'c_sensor5'}, status_code=200, kwargs={'diag': 0.0, 'old_diag': 0.0, 'isOriginPure': False})

test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor1'}, status_code=200, kwargs={'d': True, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'c_sensor1'}, status_code=200, kwargs={'d': True, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor2'}, status_code=200, kwargs={'d': False, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'sensor1'}, status_code=200, kwargs={'d': False, 'w': 0.25})
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor4', 'measurable2': 'c_sensor3'}, status_code=404, message="Cannot find the sensor id")
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, status_code=200, kwargs={'d': False, 'w': 0.25})

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4'}, data={'w': 1}, status_code=400, message="Coming request is missing necessary fields")
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, status_code=406, message="The coming put request has nothing to update")
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, data={'w': 1}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, data={'d': True}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor4', 'measurable2': 'sensor5'}, status_code=200, kwargs={'d': True, 'w': 1})

test.T(method="DELETE", url='/UMA/object/agent', status_code=200, data={'agent_id': 'test_agent'})
test.T(method='GET', url='/UMA/object/measurable', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable_id': 'sensor1'}, status_code=404, message="Cannot find the agent id!")
print "Measurable test passed"