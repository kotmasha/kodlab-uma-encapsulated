#This is the test script for testing existing agent endpoint
#Test CRUD functionality
from test_wrapper import *

"""
This plan is:

1 create an agent without id: 400 error
2 create an agent with an invalid id(number): 400 error
3 create an agent: 201 created
4 create the same agent: 409 conflict

5 create an snapshot for get
6 get the snapshot info, without giving agent_id: 400 error
7 get the snapshot info: 200 OK

8 delete the agent, without id: 400 error
9 delete the agent with invalid id
10 delete the agent, 200 OK

11 get the agent info after deletion: 404 error
"""

test = test_wrapper()
test.T(method='POST', url='/UMA/object/agent', status_code=400, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 123.45}, status_code=400, message="Cannot parsing the field")
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 'test_agent'}, status_code=201)
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 'test_agent'}, status_code=409, message="Cannot create a duplicate agent")

test.T(method='POST', url='/UMA/object/snapshot', data={'agent_id': 'test_agent', 'snapshot_id': 'test_snapshot'}, status_code=201)
test.T(method='GET', url='/UMA/object/agent', status_code=400, message="Coming request is missing necessary fields")
test.T(method='GET', url='/UMA/object/agent', query={'agent_id': 'test_agent'}, status_code=200, message="Get agent info", kwargs={'snapshot_ids': ['test_snapshot'], 'snapshot_count': 1})

test.T(method="DELETE", url='/UMA/object/agent', status_code=400, message="Coming request is missing necessary fields")
test.T(method="DELETE", url='/UMA/object/agent', status_code=404, data={'agent_id': 'test_agent1'}, message="Cannot find the agent to delete")
test.T(method="DELETE", url='/UMA/object/agent', status_code=200, data={'agent_id': 'test_agent'})

test.T(method='GET', url='/UMA/object/agent', query={'agent_id': 'test_agent'}, status_code=404, message="Cannot find the agent id!")
print "Agent test passed"