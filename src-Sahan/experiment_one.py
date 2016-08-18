import numpy as np 
import curses
from random import randint
from snapshot_platform_new import *

def start_experiment(stdscr):
	# world constraints
    X_BOUND=50 #length of the interval environment
    Y_BOUND=16
    THRESHOLD=1./((X_BOUND+1)*(Y_BOUND+1.)) #learning threshold for ALICE

    # agent's viewport constraints
    RADIUS = 10

    # new experiment
    EX = Experiment(1)

    # initialize basic experiment measurables 
    # initial positions
    X_START=rand(X_BOUND+1) 
    Y_START=rand(Y_BOUND+1)
    POS_START = complex(X_START, Y_START)

    # initial pose
    # make everything complex
    # use the standard orientation
    heading = randint(0, 3)
    if heading == 0:
        agent_heading = complex(0, 1) # up
    elif heading == 1:
        agent_heading = complex(0, -1) # down
    elif heading == 2:
        agent_heading = complex(1, 0) # right
    else:
        agent_heading = complex(-1, 0) # left

    POSE_START = agent_heading
    
    # FIX THIS LATER
    # for target's locations all we need
    # advised strongly AGAINST writing a simulation 
    # for a specific reason
    TARGET_LIST = []
    # pick initial position of  target
    # we can extend this to MULTIPLE targets

    NUMBER_OF_TARGETS = (X_BOUND+Y_BOUND)/20

    for index in xrange(NUMBER_OF_TARGETS):
        OK = False
        while not OK:
            X_PICK = rand(X_BOUND+1)
            Y_PICK = rand(Y_BOUND+1)
            OK = ((X_START-X_PICK)**2+(Y_START-Y_PICK)**2>0)
        TARGET_LIST.append(complex(X_PICK, Y_PICK))
    
    # add agents
    # let's change the name of agent
    ALICE = EX.add_agent('ALICE', THRESHOLD)

    # introduce actions
    EX.new_sensor([ALICE], 'fd')
    EX.new_sensor([ALICE], 'rt')
    EX.new_sensor([ALICE], 'lt')

    # i need to PURIFY the actions
    templist=ALICE._GENERALIZED_ACTIONS[:]
    for item in templist:
        if sum(x%2 for x in item)<3:
            ALICE._GENERALIZED_ACTIONS.remove(item)

    in_bounds = lambda x: x.real <= X_BOUND and x.imag <= Y_BOUND

    def pos(state):
        if state['fd'][0]:
            new_pos = state['pos'][0] + state['pose'][0]
            if in_bounds(new_pos):
                while new_pos in TARGET_LIST:
                    TARGET_LIST.remove(new_pos)
                return state['pos'][0] + state['pose'][0]
            else:
                return state['pos'][0]
        if state['lt'][0] or state['rt'][0]:
            return state['pos'][0]

    def pose_meas(state):
        if state['rt'][0]:
            return state['pose'][0] * complex(0, -1)
        elif state['lt'][0]:
            return state['pose'][0] * complex(0, 1)
        else:
            return state['pose'][1]

    EX.add_measurable('pos', [POS_START, POS_START], pos)
    EX.add_measurable('pose', [POSE_START, POSE_START], pose_meas)

    # motivational sensor
    # make this l1 distance
    # sum or max of those things
    # maybe min
    # l1_distance * (position of target - position of agent AND compare with the pose)
    # add a COST for turning
    # once a target is out of the viewport, the target signal will
    # be max THRESHOLD

    # target signal and competitor signal
    # z = target position, pos = agent position, dir = direction
    
    #prime = lambda x: complex(x.real, np.fabs(x.imag))

    def region(z):
        if z.real >= np.absolute(z.imag):
            return 0
        if z.real < -np.absolute(z.imag):
            return 2
        if z.real < np.absolute(z.imag) and z.real >= -np.absolute(z.imag):
            if z.imag > 0:
                return 1
            else:
                return -1

    '''
    cond = lambda z: z.real >= z.imag

    def c(complex_number):
        k = 0
        while not cond((complex(0, -1)**k) * complex_number):
            k += 1
        return k
    '''

    dist = lambda z, w: np.absolute(z - w) 
    plussed = lambda x: x if x >= 0 else 0

    target_signal = lambda z, pos, dir: np.amin([dist(z, pos), RADIUS + 1]) + np.abs(region((z-pos)*dir.conjugate()))
    #competitor_signal = lambda z, pos: plussed((RADIUS - dist(z, pos))) * sum([(dist(z, pos) <= RADIUS) for z in TARGET_LIST])

    sig = lambda state: sum([target_signal(z, state['pos'][0], state['pose'][0]) for z in TARGET_LIST])
    INIT = sig(EX.state_all())
    EX.add_measurable('sig', [INIT, INIT], sig)
    
    def radar(m, direction):
        return lambda state: any([
            dist(z, state['pos'][0]) < m and (region((z - state['pos'][0]) * state['pose'][0].conjugate()) % 4 == direction % 4) 
            for z in TARGET_LIST]) 

    for i in xrange(1, RADIUS+1):
        for j in xrange(4):
            tmp_name = 'ds'+str(i)+'dr'+str(j)
            EX.new_sensor([ALICE], tmp_name, radar(i, j))
            EX.twedge([ALICE],'fd',tmp_name)
            EX.twedge([ALICE],'fd',name_comp(tmp_name))
            EX.twedge([ALICE],'rt',tmp_name)
            EX.twedge([ALICE],'rt',name_comp(tmp_name))
            EX.twedge([ALICE],'lt',tmp_name)
            EX.twedge([ALICE],'lt',name_comp(tmp_name))

   # state['sig'] as a measurable
    def delta_signal(m):
        return lambda state: state['sig'][1] - state['sig'][0] > m if m >= 0 else state['sig'][1] - state['sig'][0] < m 

    # delta_signal sensors
    for ind in xrange(-RADIUS, RADIUS+1):
        tmp_name = 'delta'+str(ind)
        EX.new_sensor([ALICE], tmp_name, delta_signal(ind))

    for ind in xrange(RADIUS, -1, -1):
        tmp_name = 'delta_signal' + str(ind)
        EX.new_sensor([ALICE], tmp_name, delta_signal(ind))
        ALICE.add_eval(tmp_name)


    ### Run
    
    # prepare windows for output
    curses.curs_set(0)
    stdscr.erase()
    WIN=curses.newwin(Y_BOUND+3,X_BOUND+3,6,7)
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
        stdscr.addstr('S-N-I-F-F-Y  I-S  R-U-N-N-I-N-G    (press [space] to stop) ')
        stdscr.addstr(4,3,text)
        stdscr.clrtoeol()
        stdscr.noutrefresh()
        WIN.clear()
        WIN.addstr(0,0,str(counter))

        # output the targets
        for target in TARGET_LIST:
            WIN.addch(Y_BOUND+1-target.real,1+target.imag,int(84)) # print target (playground)

        WIN.addch(Y_BOUND+1-EX.state('pos')[0].real,1+EX.state('pos')[0].imag,int(83)) # print ALICE's position
        WIN.overlay(stdscr)
        WIN.noutrefresh()
        curses.doupdate()
    #print ALICE._SIZE
    
    #------------------------THIS IS THE DATA INIT--------------------------------------#
    #The tmp is a list for sensors names, every time you add some new sensors you have to recall the initData function, with new data given
    #And currently the method for initData is DELETE ALL OLD DATA, but in the future if you want to keep some old one I can adjust then
    #Function detail is in interface.txt
    tmp=[sen._NAME for sen in ALICE._SENSORS]
    acc.initData(ALICE._NAME,ALICE._SIZE,THRESHOLD,ALICE._CONTEXT.keys(),ALICE._CONTEXT.values(),tmp,ALICE._EVALS,ALICE._GENERALIZED_ACTIONS)
    #------------------------THIS IS THE DATA INIT--------------------------------------#
    
    # REAL RUN : GO TO TARGET
    while not TARGET_LIST:
        print_state(count,message)
        message='RUNNING: '+EX.tick('decide','ordered')
        count+=1
    
curses.wrapper(start_experiment)
exit(0)