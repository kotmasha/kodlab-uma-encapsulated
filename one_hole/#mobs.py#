### UMA tracker agents: data structures and their operators

import numpy as np

## Complex integers class
#  - possibly there is something better, but I couldn't find it
#
class icomplex(object):
    def __init__(self,x,y):
        self.real=int(np.floor(x))
        self.imag=int(np.floor(y))

    def __repr__(self):
        return str(complex(self.real,self.imag))

    def __add__(self,z):
        return icomplex(self.real+z.real,self.imag+z.imag)

    def __sub__(self,z):
        return icomplex(self.real-z.real,self.imag-z.imag)
        
    def __mul__(self,z):
        return icomplex(self.real*z.real-self.imag*z.imag,self.real*z.imag+self.imag*z.real)

    def conj(self):
        return icomplex(self.real,-self.imag)
    
    def __abs__(self):
        return (self*self.conj()).real

    def __complex__(self):
        return complex(self.real,self.imag)

    def __floordiv__(self,scale): # $scale$ must be a non-zero integer
        return icomplex(self.real / scale,self.imag / scale)

    def __mod__(self,scale): # $scale$ must be a non-zero integer
        return icomplex(self.real % scale,self.imag % scale)
    
    #def __coerce__(self,other):
    #    return complex(self.real,self.imag),complex(other)

up=icomplex(0,1)
down=icomplex(0,-1)
right=icomplex(1,0)
left=icomplex(-1,0)
#origin=icomplex(0,0)

    
## Tracker physical state (to be "pan-tilt-zoom" when this matures...)
#  - viewport is a grid of size $self._res$;
#  - viewport coords are come first ($self._state[0]$);
#  - $depth$ is the maximal zoom-in depth;
#  - viewports do not overlap except when containing each other.
#  - actions are:
#       zoomin, zoomout, pan by arbitrary vector in least-significant units

class ptz(object):
    def __init__(self,res,depth,state):
        # res,depth are positive integers
        # state is a list of complex integers in $range(self._res)$
        # Immutable content:
        self._res=res
        self._depth=depth
        # Mutable content:
        self._state=state[:depth]
        
    def __repr__(self):
        return str(self._state)

    ## Least significant position at the head of self._state
    def zoomin(self,pos):
        if len(self._state)<self._depth:
            self._state.insert(0,pos)
        return self
            
    def zoomout(self):
        if self._state!=[]:
            self._state.pop(0)
        return self
            
    ## Panning by $panvec$, experessed in units of current level
    #  - $panvec$ is assumed to be of type $icomplex$
    #  - input rejected (returns None, state remains unchanged) if $panvec$
    #    is too long.
    #
    def pan(self,panvec):
        depth=len(self._state)
        bound=pow(self._res,depth)
        pos=panvec
        for ind in xrange(depth):
            pos+=self._state[ind]*pow(self._res,ind)
        x=pos.real
        y=pos.imag
        if pos.real<bound and pos.imag<bound and pos.real>=0 and pos.imag>=0:
            new_state=[]
            for ind in xrange(depth):
                new_state.append(pos % self._res)
                pos //= self._res
            self._state=new_state
            return self
        else:
            return self

