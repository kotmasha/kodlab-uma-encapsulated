#This is the test script for testing existing agent endpoint
#Test CRUD functionality
from test_wrapper import *

"""
This plan is:

1 create an agent without id: 400 error
2 create an agent with an invalid id(number): 400 error
3 create an agent: 201 created
4 create the same agent: 409 conflict
"""

test = test_wrapper()
test.T(method='POST', url='/UMA/object/agent', status_code=400, message="Coming request is missing necessary fields")
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 123.45}, status_code=400, message="Cannot parsing the field")
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 'test_agent'}, status_code=201)
test.T(method='POST', url='/UMA/object/agent', data={'agent_id': 'test_agent'}, status_code=409, message="Cannot create a duplicate agent")

test.T(method='GET', url='/UMA/object/agent', data={'agent_id': 'test_agent'}, status_code=20, message="Get agent inf")
print "Agent test passed"