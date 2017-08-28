from test_wrapper import *

"""
1 create anget, snapshot, 4 sensor, init
2 amper 0&2, compare info
3 amper 3&7 compare info
4 amper 2&5 compare info
5 amper 1&6 compare info

"""

test = test_wrapper()
test.T(method='POST', url='/UMA/object/agent', status_code=201, data={'agent_id': 'test_agent'})
test.T(method='POST', url='/UMA/object/snapshot', status_code=201, data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'})
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor0', 'c_sid': 'c_sensor0'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor1', 'c_sid': 'c_sensor1'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor2', 'c_sid': 'c_sensor2'}, status_code=201)
test.T(method='POST', url='/UMA/object/sensor', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'sensor_id': 'sensor3', 'c_sid': 'c_sensor3'}, status_code=201)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor0', 'measurable2': 'sensor0'}, data={'w': .2}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor0', 'measurable2': 'sensor0'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor0', 'measurable2': 'c_sensor0'}, data={'w': .8}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor0'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'c_sensor0'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor1', 'measurable2': 'sensor1'}, data={'w': .4}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'sensor0'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'c_sensor0'}, data={'w': .6}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'sensor1'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor1', 'measurable2': 'c_sensor1'}, data={'w': .6}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'sensor0'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'c_sensor0'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'sensor1'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'c_sensor1'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor2', 'measurable2': 'sensor2'}, data={'w': .6}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'sensor0'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'c_sensor0'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'sensor1'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'c_sensor1'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'sensor2'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor2', 'measurable2': 'c_sensor2'}, data={'w': .4}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor0'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'c_sensor0'}, data={'w': .6}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor1'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'c_sensor1'}, data={'w': .4}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor2'}, data={'w': .6}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'c_sensor2'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensor3', 'measurable2': 'sensor3'}, data={'w': .8}, status_code=200)

test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'sensor0'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'c_sensor0'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'sensor1'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'c_sensor1'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'sensor2'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'c_sensor2'}, data={'w': .2}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'sensor3'}, data={'w': .0}, status_code=200)
test.T(method='PUT', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensor3', 'measurable2': 'c_sensor3'}, data={'w': .2}, status_code=200)

test.T(method='POST', url='/UMA/object/snapshot/init', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'initial_sensor_size': 4}, status_code=201)
test.T(method='PUT', url='/UMA/data/signals', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'signals': [True, False, False, False, False, False, False, False]}, status_code=200)
test.T(method='PUT', url='/UMA/data/signals', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, data={'signals': [True, False, False, False, False, False, False, False]}, status_code=200)
#0&2
test.T(method='POST', url='/UMA/simulation/delay', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'delay_list': [[True, False, False, False, False, False, False, False]], 'uuid_list': [['sensorX', 'c_sensorX']]}, status_code=201)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'sensor0'}, kwargs={'w': .2}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'c_sensor0'}, kwargs={'w': .8}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'sensor1'}, kwargs={'w': .4}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'c_sensor1'}, kwargs={'w': .6}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'sensor2'}, kwargs={'w': .6}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'c_sensor2'}, kwargs={'w': .4}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'sensor3'}, kwargs={'w': .8}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'c_sensor3'}, kwargs={'w': .2}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'sensorX', 'measurable2': 'sensorX'}, kwargs={'w': 1.0}, status_code=200)

test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'sensor0'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'c_sensor0'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'sensor1'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'c_sensor1'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'sensor2'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'c_sensor2'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'sensor3'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'c_sensor3'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'sensorX'}, kwargs={'w': .0}, status_code=200)
test.T(method='GET', url='/UMA/object/measurable_pair', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'measurable1': 'c_sensorX', 'measurable2': 'c_sensorX'}, kwargs={'w': .0}, status_code=200)

test.T(method='GET', url='/UMA/object/snapshot', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=200, message="Get snapshot info", kwargs={'sensors': [{'sensor': {'m': 'sensor0', 'cm': 'c_sensor0'}}, {'sensor': {'m': 'sensor1', 'cm': 'c_sensor1'}}, {'sensor': {'m': 'sensor2', 'cm': 'c_sensor2'}}, {'sensor': {'m': 'sensor3', 'cm': 'c_sensor3'}}, {'sensor': {'m': 'sensorX', 'cm': 'c_sensorX'}}],
    'sizes':{'sensor_count': 5, 'sensor_pair_count': 15, '_sensor_size': 5, '_sensor_size_max': 6, '_sensor2d_size': 15, '_sensor2d_size_max': 21, '_measurable_size': 10, '_measurable_size_max': 12, '_measurable2d_size': 55, '_measurable2d_size_max': 78, '_mask_amper_size':30, '_mask_amper_size_max': 42},
    'q': 0.9,'threshold': 0.125, 'auto_target': False})
test.T(method="POST", url='/UMA/simulation/pruning', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot', 'signals': [False, False, False, False, False, False, False, False, True, True]}, status_code=201)

test.T(method='GET', url='/UMA/object/snapshot', query={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=200, message="Get snapshot info", kwargs={'sensors': [{'sensor': {'m': 'sensor0', 'cm': 'c_sensor0'}}, {'sensor': {'m': 'sensor1', 'cm': 'c_sensor1'}}, {'sensor': {'m': 'sensor2', 'cm': 'c_sensor2'}}, {'sensor': {'m': 'sensor3', 'cm': 'c_sensor3'}}],
    'sizes':{'sensor_count': 4, 'sensor_pair_count': 10, '_sensor_size': 4, '_sensor_size_max': 6, '_sensor2d_size': 10, '_sensor2d_size_max': 21, '_measurable_size': 8, '_measurable_size_max': 12, '_measurable2d_size': 36, '_measurable2d_size_max': 78, '_mask_amper_size':20, '_mask_amper_size_max': 42},
    'q': 0.9,'threshold': 0.125, 'auto_target': False})

print "Delay test passed"