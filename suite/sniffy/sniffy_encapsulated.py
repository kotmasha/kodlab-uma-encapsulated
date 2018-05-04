import numpy as np
from numpy.random import randint as rnd
from collections import deque
import curses
import time
from som2 import *


def start_experiment(stdscr,env_length,discount,burn_in,agent_to_examine,delay_string):
    # wait (=0) / don't wait (=1) for spacebar to execute next cycle:
    NODELAY=1 if delay_string=='nodelay' else 0
    # Number of decision cycles for burn-in period:
    BURN_IN=burn_in
    # experiment parameters and definitions
    X_BOUND=env_length #length

    def in_bounds(pos):
        return (pos>=0 and pos<=X_BOUND)

    # distance function
    def dist(p,q):
        return abs(p-q) 

    # "Discounted" agent parameters
    Q=1.-pow(2,-discount)
    MOTION_PARAMS={
        'type':'default',
        'discount':Q,
        'AutoTarg':False,
        }
    
    # "Qualitative" agent parameters
    #MOTION_PARAMS={
    #    'type':'qualitative',
    #    'AutoTarg':False,
    #    }
    
    # initialize a new experiment
    EX=Experiment()
    id_dec='decision'
    
    # register basic motion agents;
    # - $True$ tag means they will be marked as dependent (on other agents)
    id_rt, cid_rt = EX.register_sensor('rt')
    id_lt, cid_lt = EX.register_sensor('lt')

    # register motivation for motion agents
    # - this one is NOT dependent on agents except through the position, so
    #   it carries the default False tag.
    id_at_targ,cid_at_targ=EX.register_sensor('atT')
    id_dist=EX.register('dist')
    id_sig=EX.register('sig')
    id_nav, id_navc=EX.register_sensor('nav')

    # agent to be visualized
    id_lookat = agent_to_examine
    
    # register arbiter variable whose purpose is provide a hard-wired response to a conflict
    # between agents 'lt' and 'rt'.
    id_arbiter=EX.register('ar')

    # add a counter
    id_count=EX.register('count')
    def ex_counter(state):
        return 1+state[id_count][0]
    EX.construct_measurable(id_count,ex_counter,[0])
    
    #
    ### Arbitration
    #

    # arbitration state
    def arbiter(state):
        return bool(rnd(2))
    EX.construct_measurable(id_arbiter,arbiter,[bool(rnd(2))],0, decdep=True)

    # intention sensors
    id_toRT, id_toRTc=EX.register_sensor('toR')
    def intention_RT(state):
        return id_rt in state[id_dec][0]
    EX.construct_sensor(id_toRT,intention_RT, decdep=True)
    
    id_toLT, id_toLTc=EX.register_sensor('toL')
    def intention_LT(state):
        return id_lt in state[id_dec][0]
    EX.construct_sensor(id_toLT,intention_LT, decdep=True)

    # failure mode for action $lt^rt$
    id_toF, id_toFc=EX.register_sensor('toF')
    def about_to_enter_failure_mode(state):
        return state[id_toLT][0] and state[id_toRT][0]
    EX.construct_sensor(id_toF,about_to_enter_failure_mode, decdep=True)

    # add basic motion agents with arbitration
    def action_RT(state):
        rt_decided=(id_rt in state[id_dec][0])
        if state[id_toF][0]:
            #return not(rt_decided) if state[id_arbiter][0] else rt_decided
            return state[id_arbiter][0]
        else:
            return rt_decided
    RT=EX.construct_agent(id_rt,id_sig,action_RT,True,MOTION_PARAMS)

    def action_LT(state):
        lt_decided=(id_lt in state[id_dec][0])
        if state[id_toF][0]:
            #return lt_decided if state[id_arbiter][0] else not(lt_decided)
            return not(state[id_arbiter][0])
        else:
            return lt_decided
    LT=EX.construct_agent(id_lt,id_sig,action_LT,True,MOTION_PARAMS)
 
    #
    ### "mapping" system
    #

    ## introduce agent's position

    # select starting position
    START=rnd(X_BOUND+1)

    # effect of motion on position
    id_pos=EX.register('pos')
    def motion(state):
        triggers={id_rt:1,id_lt:-1}
        diff=0
        for t in triggers:
            diff+=triggers[t]*int(state[t][0])
        newpos=state[id_pos][0]+diff
        if in_bounds(newpos):
            return newpos
        else:
            return state[id_pos][0]
    EX.construct_measurable(id_pos,motion,[START,START])

    # generate target position
    TARGET=START
    while dist(TARGET,START)<X_BOUND/8:
        TARGET=rnd(X_BOUND+1)

    # staying-at-target sensor
    def stay_at_targetQ(state):
        return state[id_pos][0]==TARGET and state[id_pos][1]==TARGET
    INIT=False
    EX.construct_sensor(id_at_targ,stay_at_targetQ,[INIT,INIT])
    RT.add_sensor(id_at_targ)
    LT.add_sensor(id_at_targ)

    # set up position sensors
    def xsensor(m): # along x-axis
        return lambda state: state[id_pos][0]<m+1

    for ind in xrange(X_BOUND):
        tmp_name = 'x'+str(ind)
        id_tmp,id_tmpc=EX.register_sensor(tmp_name) #registers the sensor pairs
        EX.construct_sensor(id_tmp,xsensor(ind)) #constructs the measurables associated with the sensor
        RT.add_sensor(id_tmp)
        LT.add_sensor(id_tmp)

    # distance to target
    # - $id_distM$ has already been registerd
    def dist_to_target(state):
        return dist(state[id_pos][0],TARGET)
    INIT=dist(START,TARGET)
    EX.construct_measurable(id_dist,dist_to_target,[INIT,INIT])
         
    ## value signal for agents LT and RT
    #signal scales logarithmically with distance to target
    rescaling=lambda r: -(1./(1.-Q))*np.log2((1.+r)/(2*(1.+X_BOUND)))
    #signal scales linearly with distance to target
    #rescaling=lambda r: 1.+(1+X_BOUND-r)

    def sig(state):
        #return 1.
        return 1+state[id_sig][0] if state[id_at_targ][0] else rescaling(state[id_dist][0]) 
    #initial value for signal:
    #INIT=1.
    INIT=rescaling(dist(START,TARGET))
    #construct the motivational signal
    EX.construct_measurable(id_sig,sig,[INIT,INIT])

    # performance sensor ("am I better now than in the last cycle?")
    # - $id_nav$ has already been registered
    def better(state):
        return True if state[id_at_targ][0] else state[id_dist][0]<state[id_dist][1]
    EX.construct_sensor(id_nav,better)
    RT.add_sensor(id_nav)
    LT.add_sensor(id_nav)

    # STOPPED HERE

    RT.add_sensor(id_rt)
    RT.add_sensor(id_lt)
    LT.add_sensor(id_rt)
    LT.add_sensor(id_lt)

    #-------------------------------------init--------------------------------------------

    for agent_name in EX._AGENTS:
        EX._AGENTS[agent_name].init()
        #EX._AGENTS[agent_name].validate()

    # ONE UPDATE CYCLE (without action) TO "FILL" THE STATE DEQUES
    reported_data=EX.update_state([cid_rt,cid_lt])

    # INTRODUCE DELAYED GPS SENSORS:
    #for agent in [RT,LT]:
    #    for token in ['plus','minus']:
    #        delay_sigs=[agent.generate_signal(['x'+str(ind)]) for ind in xrange(X_BOUND)]
    #        agent.delay(delay_sigs,token)

    # SET ARTIFICIAL TARGET ONCE AND FOR ALL
    for agent in [RT,LT]:
        for token in ['plus','minus']:
            tmp_target=agent.generate_signal([id_nav]).value().tolist()
            service_snapshot = ServiceSnapshot(agent._ID, token, service)
            service_snapshot.setTarget(tmp_target)

    # ANOTHER UPDATE CYCLE (without action)
    reported_data=EX.update_state([cid_rt,cid_lt])

    #
    ### Run
    #

    # prepare windows for output
    curses.curs_set(0)
    stdscr.erase()

    # color definitions
    curses.init_color(0,0,0,0)    #black=0
    curses.init_color(1,1000,0,0) #red=1
    curses.init_color(2,0,1000,0) #green=2
    curses.init_color(3,1000,1000,0) #yellow=3
    curses.init_color(4,1000,1000,1000) #white=4
    curses.init_color(5,1000,1000,500)
    
    curses.init_pair(1,0,1) #black on red
    curses.init_pair(2,0,2) #green on black
    curses.init_pair(3,0,3) #black on yellow
    curses.init_pair(4,4,0) #white on black
    curses.init_pair(5,1,0) #red on black
    curses.init_pair(6,0,5)
        
    REG_BG=curses.color_pair(4) | curses.A_BOLD
    POS_BG=curses.color_pair(2) | curses.A_BOLD
    NEG_BG=curses.color_pair(1) | curses.A_BOLD
    OBS_BG=curses.color_pair(6) | curses.A_BOLD
    BG=curses.color_pair(5) | curses.A_BOLD
    FG=curses.color_pair(3) | curses.A_BOLD
    
    WIN=curses.newwin(9,2*X_BOUND+3,5,7)
    WINs=curses.newwin(16,140,16,7)
    stdscr.nodelay(NODELAY)

    WIN.bkgdset(ord('.'),REG_BG)
    
    WIN.overlay(stdscr)
    WINs.overlay(stdscr)

    def print_state(banner_text,reported_data,id_agent):
        stdscr.clear()
        stdscr.addstr('W-E  A-R-E  R-U-N-N-I-N-G    (press [space] to stop) ')
        stdscr.addstr(2,3,banner_text)
        stdscr.clrtoeol()
        stdscr.noutrefresh()
        WIN.clear()
        WIN.addstr(0,0,str(EX.this_state(id_count,1)))
        WIN.chgat(0,0,BG)
        
        ## Unpacking the output from the tested agent (RT/LT)
        
        #extract information from agent:
        agent=EX._AGENTS[id_agent]
        namelist = agent._SENSORS
        curr=agent._CURRENT
        targ=agent._TARGET
        pred=agent._PREDICTED

        #extract information from reported data:
        ex_report,agent_reports=reported_data
        elapsed_time=ex_report['exiting_update_cycle']-ex_report['entering_update_counter']

        ## information about experiment update:
        #'entering_update_cycle':           time stamp for beginning of update cycle
        #'exiting_update_cycle':            time stamp for end of update cycle
        #
        ## information about decision stage:
        #'activity':                        which snapshot was active last?
        #
        #'entering_decision_cycle':         time stamp for beginning of agent's decision cycle
        #'exiting_enrichment_and_pruning':  termination time stamp of agent's e&p stage
        #'exiting_decision_cycle':          time stamp for end of agent's decision cycle
        #
        #'decision':            agent's decision
        #'deliberateQ':         was the decision deliberate?
        #'override':            decision was overridden by... (if at all)
        #'final':               final decision to be executed (post arbitration)
        #'pred_too_general':    weight of Curr-Pred
        
        #construct on-screen report:
        text=''
        for aid in EX._AGENTS.keys():
            text +='\t'+(
                'deliberate ' if agent_reports[aid]['deliberateQ'] else 'randomized '
                )+agent_reports[aid]['decision']+',\t '
            if not (agent_reports[aid]['override'] is ''):
                text += 'overruled by instruction '+agent_reports[aid]['override']+',\t '
            text +='leading to final: '+agent_reports[aid]['final']+'.\n'

        stdscr.addstr(3,3,text)

        # extract "geographic" signals from state
        gps_list={'plus':[],'minus':[]}
        for token in ['plus','minus']:
            for pos in xrange(X_BOUND):
                ind=agent._SENSORS[token].index('x'+str(pos))
                gps_list[token].extend([ind,ind+1])
        
        #choose the signals to visualize (curr,targ or pred)
        for ind,lookat in enumerate([curr,pred,targ]): 
            #convert signals to region bounds
            bounds={'plus':[0,X_BOUND],'minus':[0,X_BOUND]}
            for token in ['plus','minus']:
                for x in xrange(0,X_BOUND):
                    if lookat[token].out(gps_list[token][2*x]):
                        bounds[token][1]=min(x,bounds[token][1]) #pushing down the upper bound
                        break
                    else:
                        continue
                for x in xrange(X_BOUND-1,-1,-1):
                    if lookat[token].out(gps_list[token][2*x+1]):
                        bounds[token][0]=max(x+1,bounds[token][0]) #pushing up the lower bound
                        break
                    else:
                        continue
            
            #display the results
            tok_BG={'plus':POS_BG,'minus':NEG_BG}
            tok_line={'plus':3-ind,'minus':6+ind}
            for token in ['plus','minus']:
                min_pos=bounds[token][0]
                max_pos=bounds[token][1]
                for x in xrange(0,X_BOUND):
                    ori=ord('<') if lookat[token].out(gps_list[token][2*x]) else (ord('>') if lookat[token].out(gps_list[token][2*x+1]) else ord('*'))
                    this_BG=tok_BG[token] if (x>=min_pos and x<max_pos) else BG
                    WIN.addch(tok_line[token],2+2*x,ori,this_BG)

                    WIN.chgat(tok_line[token],1+2*min_pos,1+2*(max_pos-min_pos),tok_BG[token])
        # display targets with FG attributes
        WIN.addch(5,1+2*TARGET,ord('T'),FG)
        # display agent's position with FG attributes
        WIN.addch(4,1+2*EX.this_state(id_pos,1),ord('S'),FG)

        
        ### Display info about internal state signals (length, weights...)
        WINs.clear()
        delta=lambda tok: 0 if tok=='plus' else 1
        for token in ['plus','minus']:
            WINs.addstr(0+delta(token),0,'Number of sensors in snapshot '+agent._ID+'['+ token +'] : '+str(agent._SIZE[token])+'\t'+str(agent_reports[id_agent]['pred_too_general']))
            WINs.addstr(2+delta(token),0,'Length,weight of CURRENT['+token+'] :'+str(agent.report_current(token).len())+',\t'+str(agent.report_current(token).weight()))
            WINs.addstr(4+delta(token),0,'Length,weight of TARGET['+token+'] :'+str(agent.report_target(token).len())+',\t'+str(agent.report_target(token).weight()))
            WINs.addstr(6+delta(token),0,'Length,weight of PREDICTED['+token+'] :'+str(agent.report_predicted(token).len())+',\t'+str(agent.report_predicted(token).weight()))
            WINs.addstr(8+delta(token),0,'Length,weight of LAST['+token+'] :'+str(agent.report_last(token).len())+',\t'+str(agent.report_last(token).weight()))
            WINs.addstr(10+delta(token),0,'Length,weight of INITMASK['+token+'] :'+str(agent._INITMASK[token].len())+',\t'+str(agent._INITMASK[token].weight()))
        
        """
        ### Display the filter generated by a signal
        WINs.clear()
        WINs.addstr(0,0,'Observation:')
        WINs.addstr(4,0,'Chosen signed signal:')
        
        tok_BG={'plus':POS_BG,'minus':NEG_BG}
        vpos={'plus':6,'minus':8}
        hpos=lambda tok,x: 0 if x==0 else 2+len('  '.join(namelist[tok][:x]))
        
        
        # CURRENT OBSERVATION
        OBS=agent._OBSERVE

        #SIGNED SIGNAL TO WATCH:
        #SIG=agent._CURRENT
        
        sig=agent.generate_signal(['x0'])
        #sig = Signal([True,False,False,True])
        #sig=agent.generate_signal([EX.nid('{x0*;x2}')])
        #sig=agent.generate_signal([EX.nid('#x2')])
        #sig=agent.generate_signal([EX.nid('#x0*')])
        #SIG=agent.brain.up(sig,False)

        SIG = {}
        for token in ['plus', 'minus']:
            snapshot = ServiceSnapshot(agent._ID, token, service)
            res = snapshot.make_up(sig.value().tolist())
            SIG[token] = Signal(res)

        for token in ['plus','minus']:
            for x,mid in enumerate(namelist[token]):
                this_BG=OBS_BG if OBS[token].out(x) else REG_BG
                WINs.addstr(2,hpos(token,x),mid,this_BG)
                this_BG=tok_BG[token] if SIG[token].out(x) else REG_BG
                WINs.addstr(vpos[token],hpos(token,x),mid,this_BG)
        """      

        # refresh the window
        WIN.overlay(stdscr)
        WIN.noutrefresh()
        WINs.noutrefresh()
        curses.doupdate()



    ## Random walk period
    while EX.this_state(id_count)<BURN_IN and stdscr.getch()!=ord(' '):
        # call output subroutine
        print_state('BURN-IN ENDS IN '+str(BURN_IN-EX.this_state(id_count))+' STEPS:',reported_data,id_lookat)
        # update the state
        instruction=[(id_lt if rnd(2) else cid_lt),(id_rt if rnd(2) else cid_rt)]
        reported_data=EX.update_state(instruction)
       
    ## Main loop
    while stdscr.getch()!=ord(' '):

        # call output subroutine
        print_state('RUNNING:',reported_data,id_lookat)

        # make decisions, update the state
        reported_data=EX.update_state()

    else:
        #EX.save_experiment('sniffy1d')
        #print "data saved"
        #raise Exception('Aborting at your request...\n\n')
        pass
       
#generate a run with parameters:
#   - length of environment
#   - discount parameter becomes 1.-pow(2,-input)
#   - number of cycles for burn-in period
#   - BUA to visualize
#   - 'nodelay' means no delay between cycles; 'delay' means [spacebar] is 
#       required to advance the clock.
#
curses.wrapper(start_experiment,20,5,1000,'rt','nodelay')
print "Aborting at your request...\n\n"
exit(0)
