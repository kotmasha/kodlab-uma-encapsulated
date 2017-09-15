from test_wrapper import *

test = test_wrapper()

test.run_test('test_data/agent_test.yml')
print "Agent test passed"
test.run_test('test_data/snapshot_test.yml')
print "Snapshot test passed"
test.run_test('test_data/sensor_test.yml')
print "Sensor test passed"
test.run_test('test_data/measurable_test.yml')
print "Measurable test passed"
test.run_test('test_data/matrix_test.yml')
print "Matrix test passed"
test.run_test('test_data/amper_test.yml')
print "Amper test passed"
test.run_test('test_data/delay_test.yml')
print "Delay test passed"
test.run_test('test_data/pruning_test.yml')
print "Pruning test passed"