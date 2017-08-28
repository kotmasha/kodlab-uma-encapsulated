#This script test the general rest endpoint

from test_wrapper import *

"""
This plan is:

1 invalid call
"""

test = test_wrapper()
test.T(method='POST', url='/UMA/abc/def', status_code=400, message="cannot find coresponding handler")

print "general test passed!"