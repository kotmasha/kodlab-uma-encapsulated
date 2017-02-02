#import sys as sys
import numpy as np
from random import randint
import curses
from snapshot_platform_new import *


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

    def __neg__(self):
        return icomplex(-self.real,-self.imag)

    def __mul__(self,z):
        return icomplex(self.real*z.real-self.imag*z.imag,self.real*z.imag+self.imag*z.real)

    def pow(self,x):
        if isinstance(x,int):
            if x==0:
                return 1
            elif x>0:
                return self*pow(self,x-1)
            else:
                raise Exception('Complex integer has only non-negative integer powers.')
        else:
            raise Exception('Complex integer has only non-negative integer powers.')
            
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

_nulik=icomplex(0,0)
_up=icomplex(0,1)
_down=icomplex(0,-1)
_right=icomplex(1,0)
_left=icomplex(-1,0)
#origin=icomplex(0,0)


def start_experiment(stdscr):
    ### log and output files
    #sniffylog=open('sniffy.log', 'w')
    #sys.stdout=sniffylog
    #sniffypic=open('sniffypic.dat', 'w')

    #
    ## grid definitions
    #
    
    X_BOUND=10 #grid size
    Y_BOUND=10
    THRESHOLD=1./(1+max(X_BOUND,Y_BOUND)) #learning threshold for ALICE and BOB
    in_bounds=lambda pos: pos.real>=0 and pos.real<=X_BOUND and pos.imag>=0 and pos.imag<=Y_BOUND
    #
    HOLE_LEFT=X_BOUND/4+randint(0,X_BOUND/4) #hole size and location
    HOLE_RIGHT=X_BOUND/2+randint(1,X_BOUND/4)
    HOLE_DOWN=Y_BOUND/4+randint(0,Y_BOUND/4)
    HOLE_UP=Y_BOUND/2+randint(1,Y_BOUND/4)
    #
    in_hole= lambda pos: pos.real>=HOLE_LEFT and pos.real<HOLE_RIGHT and pos.imag>=HOLE_DOWN and pos.imag<HOLE_UP
    dist2= lambda p,q: abs(p-q) #ell2 distance squared
    dist1=lambda p,q: abs(p.real-q.real)+abs(p.imag-q.imag) #ellone distance
    distm=lambda p,q: max(abs(p.real-q.real),abs(p.imag-q.imag)) #ellinfty distance
    
    #
    ### open and initialize a new experiment
    #
    EX=Experiment(1)
    #
    ## initialize experiment
    #
    # select starting position
    START=icomplex(X_BOUND/2,Y_BOUND/2)
    while in_hole(START):
        START=icomplex(randint(0,X_BOUND),randint(0,Y_BOUND))
    # select target position
    TARGET=START
    while dist1(TARGET,START)<(X_BOUND+Y_BOUND)/8 or in_hole(TARGET):
        TARGET=icomplex(randint(0,X_BOUND),randint(0,Y_BOUND))
    #
    ## add agents
    ALICE=EX.add_agent_empirical('Alice',THRESHOLD,False)
    BOB=EX.add_agent_empirical('Bob',THRESHOLD,False)

    ### introduce actions
    EX.new_sensor([ALICE,BOB],'rt')
    EX.new_sensor([ALICE,BOB],'lt')
    EX.new_sensor([ALICE,BOB],'up')
    EX.new_sensor([ALICE,BOB],'dn')
    
    # remove all but the pure actions and the "no-action"
    templist=ALICE._GENERALIZED_ACTIONS[:] # and currently is the same as Bob's
    for item in templist:
        if sum(x%2 for x in item)<3:
            ALICE._GENERALIZED_ACTIONS.remove(item)
            BOB._GENERALIZED_ACTIONS.remove(item)
            

    # naming the available generalized actions for future use
    RIGHT=[0,3,5,7]
    LEFT=[1,2,5,7]
    UP=[1,3,4,7]
    DOWN=[1,3,5,6]
    STAND_STILL=[1,3,5,7]

    
    #
    ### ``mapping'' system (ALICE)
    #

    ### introduce agent's position, 'pos' (an icomplex object):
    def motion(state):
        triggers={'rt':_right,'lt':_left,'up':_up,'dn':_down}
        diff=_nulik
        for t in triggers:
            diff+=triggers[t]*int(state[t][0])
        newpos=state['pos'][0]+diff
        if in_bounds(newpos) and not in_hole(newpos):
            return newpos
        else:
            return state['pos'][0]

    INIT=START
    EX.add_measurable('pos',[INIT,INIT],motion)

        
    # set up position sensors
    def xsensor(m):
        return lambda state: state['pos'][0].real<m+1
    #
    # setting up positional context for actions
    for ind in xrange(X_BOUND):
        tmp_name='x'+str(ind)
        EX.new_sensor([ALICE],tmp_name,xsensor(ind))
        EX.twedge([ALICE],'rt',tmp_name)
        EX.twedge([ALICE],'rt',name_comp(tmp_name))
        EX.twedge([ALICE],name_comp('rt'),tmp_name)
        EX.twedge([ALICE],name_comp('rt'),name_comp(tmp_name))
        EX.twedge([ALICE],'lt',tmp_name)
        EX.twedge([ALICE],'lt',name_comp(tmp_name))
        EX.twedge([ALICE],name_comp('lt'),tmp_name)
        EX.twedge([ALICE],name_comp('lt'),name_comp(tmp_name))

    def ysensor(m):
        return lambda state: state['pos'][0].imag<m+1
    #
    # setting up positional context for actions
    for ind in xrange(Y_BOUND):
        tmp_name='y'+str(ind)
        EX.new_sensor([ALICE],tmp_name,ysensor(ind))
        EX.twedge([ALICE],'up',tmp_name)
        EX.twedge([ALICE],'up',name_comp(tmp_name))
        EX.twedge([ALICE],name_comp('up'),tmp_name)
        EX.twedge([ALICE],name_comp('up'),name_comp(tmp_name))
        EX.twedge([ALICE],'dn',tmp_name)
        EX.twedge([ALICE],'dn',name_comp(tmp_name))
        EX.twedge([ALICE],name_comp('dn'),tmp_name)
        EX.twedge([ALICE],name_comp('dn'),name_comp(tmp_name))


    #
    ### motivational system (ALICE)
    #

    # normalized distance to playground (nav function #1)
    def navA(state):
        return dist2(state['pos'][0],TARGET)

    INIT=navA(EX.state_all())
    EX.add_measurable('navA',[INIT,INIT],navA)

    # value sensing: "am I closer to the target"?
    def clA_1(state):
        return state['navA'][1]-state['navA'][0]>=1


    def clA_2(state):
        return state['navA'][1]-state['navA'][0]>=2
    
    EX.new_sensor([ALICE],'T1',clA_1)
    #leave this for future testing
    EX.new_sensor([ALICE],'T2',clA_2)
    ALICE.add_eval('T2') # Alice gives higher priority to steeper descent
    ALICE.add_eval('T1') # This corresponds to diagonal moves, so won't happen until we allow more complex actions.

    ### Initialize ALICE on GPU
    ALICE.brain.init()

    #
    ### Censor (BOB)
    #
    ## add description of hole to BOB
    BOB_REPELLER=['x'+str(HOLE_RIGHT),
              name_comp('x'+str(HOLE_LEFT)),
              'y'+str(HOLE_UP),
              name_comp('y'+str(HOLE_DOWN)),
    ]
    #Update BOB_REPELLER by upwards-closing it under ALICE and turn
    #  BOB_REPELLER into an ALICE signal, denoted $SIGB$
    #  the contents of $discard$ will not be used at this point.
    SIGB=ALICE.brain.up(Signal([(item._NAME in BOB_REPELLER) for item in ALICE._SENSORS]))
    BOB_REPELLER=[ALICE._SENSORS[ind]._NAME for ind,tag in enumerate(SIGB.value_all()) if tag]
    
    ## add the relevant sensors to BOB...
    #    (had to add method "take_sensor" to platform file)
    for name in BOB_REPELLER:
        BOB.take_sensor(ALICE,name)
        ## Forming contextual sensors for BOB's actions
        for act in ['rt','rt*','lt','lt*','up','up*','dn','dn*']:
            try:
                BOB.take_sensor(ALICE,wedge(act,name))
            except:
                pass
            

    ## If Alice predicts a decrease of the distance from the current state
    #  to the hole...
    #  -  We want Bob to respond by countering ALICE's actions
    #  -  Therefore, we want BOB to learn to increase distance to the hole,
    #     provided this distance is small enough

    #ADDED now the following two functions are well-formed, which completes
    #   the construction of BOB.
    def dist_to_hole(state):
        return sum(SIGB.intersect(ALICE.current().star()).value_all())
        
    INIT=dist_to_hole(EX.state_all())
    EX.add_measurable('dist_to_hole',[INIT,INIT],dist_to_hole)
        
    def projected_dist_to_hole(state):
        return sum(SIGB.intersect(ALICE.projected().star()).value_all())
        # We assume here that ALICE has made her decision. Therefore, in
        #    the "moment" between ALICE having made hers and BOB beginning
        #    to work on his, information ABOUT THE PROJECTED CONSEQUENCES OF
        #    ALICE'S DECISION NEEDS TO BE AVAILABLE, represented in the
        #    experiment state.

    INIT=projected_dist_to_hole(EX.state_all())
    EX.add_measurable('pdist_to_hole',[INIT,INIT],projected_dist_to_hole)

    # define sensors for distance
    def distB(m):
        return lambda state: state['dist_to_hole'][0]<=m
    # define sensors for projected distance
    def pdistB(m):
        return lambda state: state['pdist_to_hole'][0]<=m
    # define the "BOB, you need to get worried" sensor
    BOB_worry=lambda state: state['pdist_to_hole']<state['dist_to_hole']

    # Construct BOB's target sensor
    # This is the simplest, most naive version of BOB
    EX.new_sensor([BOB],'evB',distB(0))
    BOB.add_eval(name_comp('evB'))
    
    ### Initialize BOB on GPU
    BOB.brain.init()
    
    #
    ### Run
    #
    # prepare windows for output
    curses.curs_set(0)
    stdscr.erase()
    WIN=curses.newwin(Y_BOUND+3,X_BOUND+3,8,7)
    stdscr.nodelay(1)
    WIN.border(int(35),int(35),int(35),int(35),int(35),int(35),int(35),int(35))
    WIN.bkgdset(int(46))
    WIN.overlay(stdscr)
    WIN.noutrefresh()

    # output subroutine
    def print_state(counter,text):
        if stdscr.getch()==int(32):
            raise('Aborting at your request...\n\n')
        stdscr.clear()
        stdscr.addstr('A-L-I-C-E  I-S  R-U-N-N-I-N-G    (press [space] to stop) ')
        stdscr.addstr(4,3,text)
        stdscr.clrtoeol()
        stdscr.noutrefresh()
        WIN.clear()
        WIN.addstr(0,0,str(counter))
        WIN.addch(Y_BOUND+1-TARGET.imag,1+TARGET.real,int(84)) # print target 
        WIN.addch(Y_BOUND+1-EX.state('pos')[0].imag,1+EX.state('pos')[0].real,int(83)) # print agent's position
        for x in xrange(HOLE_LEFT,HOLE_RIGHT): # print position of hole
            for y in xrange(HOLE_DOWN,HOLE_UP):
                WIN.addch(Y_BOUND+1-y,1+x,int(67))
        WIN.overlay(stdscr)
        WIN.noutrefresh()
        curses.doupdate()

    count=0
    while stdscr.getch()!=int(32):
        message='RUNNING:\n'+EX.tick('decide','ordered')
        print_state(count,message)
        #while stdscr.getch() not in {int(61),int(32)}:
        #    pass
        count+=1
    
curses.wrapper(start_experiment)
exit(0)