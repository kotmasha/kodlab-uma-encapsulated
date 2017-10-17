#this is the unittest for add and remove
from wrapper import *
import numpy as np

test = wrapper()

row0 = np.array([])
row1 = np.array([1.0])
row2 = np.array([3.0, 3.0])
row3 = np.array([4.0, 4.0, 2.0])
row4 = np.array([5.0, 5.0, 3.0, 4.0])

# insert incorrect number
try:
    test.insert(row1)
    print "incorrect number case failed"
except:
    print "Passed incorrect number case"

# insert delete1
test.insert(row0)
test.remove_one()
assert test.report(key="dist") == ()
print "Passed insert remove 1"

# delete invalid rows
test.insert(row0)
test.insert(row1)
test.insert(row2)
try:
    test.remove_rows([3])
    print "delete invalid rows case failed"
except:
    print "Passed delete invalid rows case"

# insert delete 2
test.remove_rows([1])
assert test.report(key="dist") == (0.0, 3.0, 0.0)
print "Passed insert delete 2"

# insert delete 3
test.remove_one()
test.remove_one()
print "Passed insert delete 3"

# insert delete 4
try:
    test.remove_one()
    print "insert delete 4 case failed"
except:
    print "Passed insert delete 4"

# insert delete 5
test.insert(row0)
test.insert(row1)
test.insert(row2)
test.insert(row3)
test.insert(row4)
assert test.report(key="dist") == (0.0, 1.0, 0.0, 3.0, 3.0, 0.0, 4.0, 4.0, 2.0, 0.0, 5.0, 5.0, 3.0, 4.0, 0.0)
print "Passed insert delete 5"

# slhc 1
test.calculate(key="slhc")
assert test.blocks(0.7) == ((0,),(1,),(2,),(3,),(4,))
assert test.blocks(1.5) == ((0,1,),(2,),(3,),(4,))
assert test.blocks(2.2) == ((0,1,),(2,3,),(4,))
assert test.blocks(3.1) == ((0,1,2,3,4,),)
assert test.report(key="height") == (3.0, 2.0, 1.0, 0.0)
print "Passed slhc 1"

# slhc 2
assert test.report(key="slhc") == (0.0, 1.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 2.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0)
test.remove_one()
assert test.report(key="dist") == (0.0, 1.0, 0.0, 3.0, 3.0, 0.0, 4.0, 4.0, 2.0, 0.0)
assert test.report(key="slhc") == (0.0, 1.0, 0.0, 3.0, 3.0, 0.0, 3.0, 3.0, 2.0, 0.0)
print "Passed slhc 2"

# insert delete 6
test.remove_rows([0,2])
assert test.report(key="dist") == (0.0, 4.0, 0.0)
assert test.report(key="slhc") == (0.0, 3.0, 0.0)
print "Passed insert delete 6"