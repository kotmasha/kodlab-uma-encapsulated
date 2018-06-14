# from multiprocessing import Pool
from collections import deque
from itertools import *
from numpy.random import randint as rnd
import numpy as np

from UMARest import *

import uuid
import time
import json

N_CORES = 8
###
### Randomized inequality
###

PRECISION = pow(2, -64)


# Return the token associated with the smaller value if precision threshold
# is met and report a strong inequality; otherwise return a random token and
# report a weak(=uncertain) inequality.
def rlessthan(x, y, prec=PRECISION):
    xval, xtok = x
    yval, ytok = y
    if yval - xval > abs(prec):
        return xtok, True
    elif xval - yval > abs(prec):
        return ytok, True
    else:
        return (xtok if bool(rnd(2)) else ytok), False


###
### Handling Boolean functions
###

def func_amper(experiment, mid_list):
    def f(state):
        return all([experiment._DEFS[mid](state) for mid in mid_list])

    return f


def func_not(func):
    def f(state):
        return not (func(state))

    return f


def func_delay(midlist):
    def f(state):
        return all([state[mid][1] for mid in midlist])

    return f


###
### Shortcuts to numpy Boolean Logic functions
###

def negate(x):
    return np.logical_not(x)


def conjunction(x, y):
    return np.logical_and(x, y)


def disjunction(x, y):
    return np.logical_or(x, y)


def symmetric(x, y):
    return np.logical_xor(x, y)


def alltrue(n):
    return np.array([True for x in xrange(n)])


def allfalse(n):
    return np.array([False for x in xrange(n)])


###
### Name-handling functions
###

def name_comp(name):
    ### return the name of the complementary sensor
    return name + '*' if name[-1:] != '*' else name[:-1]


def name_invert(names):
    ### return the set of complemented names in the list/set names
    return set(name_comp(name) for name in names)


def name_delay(name):
    ### delay
    return '#' + str(name)


def name_ampersand(name_list):
    ### conjunction
    L = len(name_list)
    if L == 0:
        raise Exception('\nEmpty conjunction not allowed.\n')
    elif L == 1:
        return name_list[0]
    else:
        return '{' + ';'.join(name_list) + '}'


###
### DATA STRUCTURES
###

class Signal(object):
    def __init__(self, value):
        if len(value) % 2 == 0:
            self._VAL = np.array(value, dtype=bool)
        else:
            raise Exception('Objects of class Signal must have even length -- Aborting!\n')

    def __repr__(self):
        return str(self._VAL)

    def len(self):
        return len(self._VAL)

    def weight(self):
        return self._VAL.sum()

    # set the signal
    def set(self, ind, value):
        self._VAL[ind] = value

    # inspect the signal
    def out(self, ind=None):
        if ind is None:
            return self._VAL.tolist()
        else:
            return self._VAL[ind]

    # report the signal
    def value(self):
        return self._VAL

    # extend the signal
    def extend(self, value):
        self._VAL = np.concatenate((self._VAL, value))

    ### negating a partial signal
    def star(self):
        return Signal([(self._VAL[i + 1] if i % 2 == 0 else self._VAL[i - 1]) for i in xrange(len(self._VAL))])

    ### full complement of a signal
    def negate(self):
        return Signal(negate(self._VAL))

    ### subtracting Signal "other" from Signal "self"
    def subtract(self, other):
        return Signal(conjunction(self._VAL, negate(other._VAL)))

    def add(self, other):
        return Signal(disjunction(self._VAL, other._VAL))

    def intersect(self, other):
        return Signal(conjunction(self._VAL, other._VAL))

    def contained_in(self, other):
        return self.subtract(other).weight() == 0


### Data type maintaining the "environment" state and its interactions
### with the agents

class Experiment(object):
    def __init__(self, experiment_id):
        # dictionary of agents in this experiment, by uuid
        self._AGENTS = {}
        self._EXPERIMENT_ID = experiment_id

        # registering the decision observable
        self._ID = set()
        self._ID.add('decision')
        # Dictionaries translating user-assigned names to
        # system-assigned uuids
        self._ID_TO_DEP = {'decision': False}

        # List of names of the experiment measurables:
        # - ordered to accommodate dependencies during the
        #   updating process;
        # - initialized to contain the trivial sensors.
        self._MID = ['decision']

        # ID-based representation of the experiment state
        # - each entry is $key:deque$;
        # - the trivial measurables initialized:
        self._STATE = {'decision': deque([[]], 1)}

        self._EXPERIMENT_SERVICE = UMAClientWorld().add_experiment(self._EXPERIMENT_ID)

        ### ID-based representation of the currently evolving decision

        ### Name-based representation $self._DEFS$ of the state update:
        ### - each entry is $key:function$;
        ### - the function accepts a dictionary of the same format as
        ###   $self._STATE$.
        def ex_decision(state):
            return state['decision'][0]

        self._DEFS = {'decision': ex_decision}

    # The register function, just do id register.
    # If id is None, generate a new one, if not just add it
    # The function is NOT supposed to be used in any test file directly
    def register(self, mid=None):
        """
            :param id: the id to be registered, if it is None, will generate an uuid
            :return: the id
            """
        if mid is None:  # means the sensor is a new one, uuid will be generated
            new_id = str(uuid.uuid4())
            self._ID.add(new_id)
            return new_id
        elif mid in self._ID:
            raise Exception('ID $' + str(mid) + '$ already registered -- Aborting.')
        else:  # need to add the id. if same name provided, just override
            self._ID.add(mid)
            return mid

    # The function for register sensor
    # If id is None, generate 2 new sensor ids
    # The function is NOT supposed to be used in any test file directly
    def register_sensor(self, id_string=None):
        """
            :param id: the id to be registered as sensor
            :return: the id and cid of the sensor
            """
        # generate the mid first
        mid = self.register(id_string)
        if id_string is None:
            # if the id_string is None, the midc will generate from the #new_id
            midc = self.register(name_comp(mid))
        else:
            # else the midc will generate from id_string*
            midc = self.register(name_comp(id_string))
        return mid, midc

    def dep(self, mid):
        try:
            return self._ID_TO_DEP[mid]
        except:
            raise Exception('This id is not registered.')

    ## Front-end query (name-based) of an experiment state variable
    def this_state(self, mid, delta=0):
        try:
            return self._STATE[mid][delta]
        except:
            pass

    ## Internal query of an experiment state variable
    def state_all(self):
        return self._STATE

    ## Set new state value
    def set_state(self, mid, value):
        self._STATE[mid].appendleft(value)
        return None

    # Construct the measurable using id
    def construct_measurable(self, mid, definition=None, init_value=None, depth=1, decdep=False):
        """
            :param mid:  the input id, if it is none, id will be generated
            :param definition: the definition function of the measurable
            :param init_value: the init value for the measurable
            :param depth: the depth of the measurable
            :param decdep: value indicating if the measurable is decision dependent
            :return: nothing
            """
        # check mid first
        #        construction requires prior registration
        if mid not in self._ID:
            raise Exception("the mid " + mid + " is not registered!")

        # add the mid into the MID list, ID_TO_DEP, and DEFS
        self._MID.append(mid)
        self._ID_TO_DEP[mid] = bool(decdep)
        self._DEFS[mid] = definition
        if init_value is None:  # will try to generate the init_value based on the definition
            self._STATE[mid] = deque([], depth + 1)
            # -------------------- Remove the old try/except block, because: -----------------------
            # 1 if definition is None, getting this by throwing an exception will reduce performance, an if else check will be faster
            # 2 if definition is not None, and there is an error generating values, then the exception will not be caught because those error are unexpected
            if definition is not None:  # if definition exist, calculate the init value, any error will throw to outer scope
                self.set_state(mid, definition(self._STATE))
        else:
            self._STATE[mid] = deque(init_value, depth + 1)
        return None

    # construct the agent using id
    def construct_sensor(self, mid, definition=None, init_value=None, depth=1, decdep=False):
        """
            :param mid:  the input id, if it is none, id will be generated
            :param definition: the definition function of the measurable
            :param init_value: the init value for the measurable
            :param depth: the depth of the measurable
            :param decdep: value indicating if the measurable is decision dependent
            :return: nothing
            """
        midc = name_comp(mid)
        # check mid/midc first
        if mid not in self._ID or midc not in self._ID:
            raise Exception("the mid $" + mid + "$ is not registered!")

        # compute initial value of sensor
        if definition is None:  # this is an action sensor, init value WILL be defaulted to False...
            self.construct_measurable(mid, None, [False for ind in xrange(depth + 1)], depth, decdep)
            self.construct_measurable(midc, None, [True for ind in xrange(depth + 1)], depth, decdep)
        else:
            self.construct_measurable(mid, definition, init_value, depth, decdep)
            self.construct_measurable(midc, func_not(definition),
                                      negate(init_value) if init_value is not None else None, depth, decdep)
        return None

    def construct_agent(self, id_agent, id_motivation, definition, decdep, params):
        """
            :param id_agent: the agent id, must provide, cannot be None
            :param id_motivation: id for phi value
            :param definition: definition of the agent
            :param decdep: bool indicating whether the agent is decision dependent
            :param params: other parameters for the agent
            :return: agent if succeed, None if fail
            """
        # -------------------- Remove the old try/except block, because: -----------------------
        # if id_agent is None, getting this by throwing an exception will reduce performance, an if else check will be faster
        if id_agent in self._ID:
            # construct and initialize new sensor
            self.construct_sensor(id_agent, definition=definition, decdep=decdep, init_value=[False,False])
            # construct new agent (Python and C++)
            try:
                #trying to pass along type specification to the UMA core
                agent_service = self._EXPERIMENT_SERVICE.add_agent(id_agent, type=params['type'])
            except KeyError:
                #if no type specification, use the default
                agent_service = self._EXPERIMENT_SERVICE.add_agent(id_agent)
            new_agent = Agent(self, id_agent, id_motivation, params, agent_service)
            self._AGENTS[id_agent] = new_agent
            return new_agent
        else:
            raise Exception("the agent " + id_agent + " is not registered!")

    ## Update the experiment state and output a summary of decisions
    def update_state(self, instruction='decide'):
        #additional date reported about the experiment update:
        ex_reports={}
        #additional data reported to the experiment, by agent id:
        agent_reports={}

        #update initiation time:
        ex_reports['entering_update_cycle']=time.clock()
        
        #purge the experiment's decision variable
        id_dec = 'decision'
        self.set_state(id_dec, [])
        
        ## Iterate over agents and decision-dependent measurables
        # - if id of an agent, then form a decision and append to id_dec
        # - if cid of an agent, then pass
        # - if id of a measurable, update its value
        for mid in (tmp_id for tmp_id in self._MID if self.dep(tmp_id)):
            midc = name_comp(mid)
            agentQ = mid in self._AGENTS or midc in self._AGENTS

            if agentQ:
                if mid in self._AGENTS:  # if mid is an agent...
                    agent_reports[mid]={} #prepare a dictionary for agent's report
                    agent_reports[mid]['entering_decision_cycle']=time.clock()
                    ## agent activity set to current reading
                    agent = self._AGENTS[mid]
                    agent._ACTIVE = self.this_state(mid)
                    activity = 'plus' if agent._ACTIVE else 'minus'
                    agent_reports[mid]['activity']=activity #report active snapshot

                    ## ENRICHMENT
                    # agent analyzes outcomes of preceding decision
                    # cycle; if the prediction was too broad, add a
                    # sensor corresponding to the episode last observed
                    # and acknowledged by the active snapshot of the agent:
                    agent_reports[mid]['pred_too_general']=agent.report_current().subtract(agent.report_predicted()).weight()
                    #if agent.report_current().subtract(agent.report_predicted()).weight() > 0:
                    #    new_episode = agent.report_last().intersect(agent.report_initmask())
                    #    sensors_to_be_added = [new_episode]
                    #else:
                    #    sensors_to_be_added = []
                    #
                    #sensors_not_to_be_removed = agent.ALL_FALSE()

                    ## PRUNING
                    # abduction over negligible sensors
                    # STEP 1. gather the negligible delayed sensors;
                    #         ($x$ is negligible if $x<x^*$)
                    # STEP 2. cluster them;
                    # STEP 3. perform abduction over the clusters.

                    # abduction over delayed sensors implying an initial sensor
                    # STEP 1. Gather them
                    #
                    #init_downs = agent.close_downwards([agent.generate_signal([sid]) for sid in agent.report_initial()])
                    #delayed_downs = [sig.subtract(agent.report_initmask()) for sig in init_downs]
                    # STEP 2. Perform abduction
                    #new_masks = agent.perform_abduction(delayed_downs)

                    # Restructuring
                    # STEP 1. Add new delayed sensors:
                    #sensors_to_be_added.extend(new_masks)
                    #ENRICHMENT DONE HERE
                    #agent.delay(sensors_to_be_added)
                    
                    # Step 2. Remove old sensors:
                    #sensors_to_be_removed = agent.ALL_FALSE()
                    #for sig in delayed_downs:
                    #    sensors_to_be_removed.add(sig)
                    #agent.pad_signal(sensors_to_be_removed)
                    #agent.prune(sensors_to_be_removed)

                    # RANDOM THOUGHTS:
                    # There needs to be a budget of delayed units, and we should merely
                    # be rewiring them....
                    # Increases in the budget should be prompted by a need for more units, that is:
                    # add more units only when "pruning" ends up demanding more resources than available.
                    #
                    ##THIS IS WHERE IT WOULD BE NICE TO HAVE AN ALTERNATIVE ARCHITECTURE
                    ##LEARNING THE SAME STRUCTURE OR AN APPROXIMATION THEREOF
                    
                    ## compute and record e&p duration:
                    agent_reports[mid]['exiting_enrichment_and_pruning']=time.clock()

                    ## agent enters observation-deliberation-decision stage and reports:
                    agent_reports[mid]['deliberateQ'] = agent.decide()
                    agent_reports[mid]['decision'] = mid if mid in self.this_state(id_dec) else midc

                    ## compute and report duration of decision cycle:
                    agent_reports[mid]['exiting_decision_cycle']=time.clock()
                    
                    ## report the agent size:
                    agent_reports[mid]['size']=max(agent._SNAPSHOTS['plus']._SIZE,agent._SNAPSHOTS['minus']._SIZE)
                    
                    ## possible overriding instruction is considered
                    if instruction != 'decide':
                        if mid in instruction or midc in instruction:
                            # append override information to message
                            agent_reports[mid]['override']=mid if mid in instruction else midc
                            # replace last decision with provided instruction
                            self.this_state(id_dec).pop()
                            self.this_state(id_dec).append(mid if mid in instruction else midc)
                    else:
                        agent_reports[mid]['override']=''

                else:  
                    # if midc is an agent, a decision has already been reached, so no action is required
                    pass
            else:  # neither mid nor midc is an agent, so perform the value update
                try:  # attempt update using definition
                    self.set_state(mid, self._DEFS[mid](self._STATE))
                except:  # if no definition available, do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                    pass

        # At this point, there is a complete decision vector
        action_signal = self.this_state(id_dec)
        for mid in self._MID:
            midc = name_comp(mid)
            agentQ = mid in self._AGENTS or midc in self._AGENTS
            depQ = self.dep(mid)

            if agentQ:  
                # if mid is an agent (or its star)...
                try:  
                    # try evaluating the definition for this mid
                    self.set_state(mid, (self._DEFS[mid](self._STATE)))
                    #report final decision for mid
                    if mid in self._AGENTS:
                        agent_reports[mid]['final'] = mid if self.this_state(mid) else midc
                except:  
                    # if no definition, set the value according to $action_signal$
                    self.set_state(mid, (mid in action_signal))
            else:
                try:  
                    # try updating using definition
                    self.set_state(mid, (self._DEFS[mid](self._STATE)))
                except:  
                    # if no definition available then do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                    pass

        ex_reports['exiting_update_cycle']=time.clock()
        return ex_reports,agent_reports

# The snapshot class that hold the snapshot shell on python side
class Snapshot(object):
    def __init__(self, agent, id_snapshot, snapshot_service):
        self._ID = id_snapshot
        self._AGENT = agent
        self._SENSORS = []
        self._SIZE = 0
        self._OBSERVE = Signal(np.array([], dtype=np.bool))
        self._LAST = Signal(np.array([], dtype=np.bool))
        self._CURRENT = Signal(np.array([], dtype=np.bool))
        self._PREDICTED = Signal(np.array([], dtype=np.bool))
        self._TARGET = Signal(np.array([], dtype=np.bool))
        self._INITMASK = Signal(np.array([], dtype=np.bool))
        self._SNAPSHOT_SERVICE = snapshot_service

    def add_sensor(self, mid, experiment):
        midc = name_comp(mid)
        if mid not in self._SENSORS:
            self._SIZE += 2
            self._SENSORS.extend([mid, midc])
            self._OBSERVE.extend(np.array([experiment.this_state(mid), experiment.this_state(midc)], dtype=bool))
            self._CURRENT.extend(np.array([False, False], dtype=bool))
            self._TARGET.extend(np.array([False, False], dtype=bool))
            self._PREDICTED.extend(np.array([False, False], dtype=bool))
            self._LAST.extend(np.array([False, False], dtype=bool))
            self._INITMASK.extend(np.array([True, True], dtype=bool))

    def collect_observation(self):
        self._OBSERVE = Signal([self._AGENT._EXPERIMENT.this_state(mid) for mid in self._SENSORS])
        return None

    def init(self, params):
        if params['type'] == 'default':
            self._SNAPSHOT_SERVICE.set_q(params['discount'])

        self._SNAPSHOT_SERVICE.set_auto_target(params['AutoTarg'])

        for i in xrange(self._AGENT._INITIAL_SIZE):
            self._SNAPSHOT_SERVICE.add_sensor(self._SENSORS[2 * i], self._SENSORS[2 * i + 1])
        self._SNAPSHOT_SERVICE.init()

class Agent(object):
    ### initialize an "empty" agent with prescribed learning parameters
    def __init__(self, experiment, id_agent, id_motivation, params, service_agent):
        # Agent's ID and complementary ID:
        self._ID = id_agent
        self._CID = name_comp(id_agent)

        # Agent's initialization parameters; expects a dictionary with keys
        #   'type'      -- 'default' / 'qualitative'
        #   'discount'  -- for type 'default', the value of the discount parameter,
        #                 in the range (0.5,1).
        #   'AutoTarg'  -- True if Auto-Targeting mode, False for externally 
        #                 specified target
        self._PARAMS = params

        # The experiment serving as the agent's environment
        self._EXPERIMENT = experiment

        # Agent's motivational signal:
        self._MOTIVATION = id_motivation

        # Flag denoting readiness state of the agent:
        # - $False$ means initial sensors may still be added, only Python side is activated;
        # - $True$ means agent has been constructed on CPP/GPU side, no initial sensors may be
        #          added beyond this point.
        self._READY = False

        self._AGENT_SERVICE = service_agent

        # initial size will be updated (+=2) with each added sensor prior to validation:
        self._INITIAL_SIZE = 0

        snapshot_plus_service = self._AGENT_SERVICE.add_snapshot('plus')
        snapshot_minus_service = self._AGENT_SERVICE.add_snapshot('minus')
        self._SNAPSHOTS = {"plus": Snapshot(self, "plus", snapshot_plus_service), "minus": Snapshot(self, "minus", snapshot_minus_service)}

        # agent is initialized as idle:
        self._ACTIVE = False

    # pad a short signal with $False$ values
    def pad_signal(self, sig, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        L = self._SIZE[token] - sig.len()
        if L > 0:
            sig.extend(allfalse(L))
        elif L < 0:
            raise Exception('\nSignal length mismatch: signal too long.\n')
        else:
            pass
            # return sig

    ## Activate the agent when the initial definitions stage is completed
    def init(self):
        # switch to work mode (no more initial sensors added)
        self._READY = True
        # make first observation
        self.collect_observation()
        # initialize structures

        for token in ['plus', 'minus']:
            self._SNAPSHOTS[token].init(self._PARAMS)

    #def validate(self):
    #    snapshot_plus = ServiceSnapshot(self._ID, 'plus', service)
    #    snapshot_minus = ServiceSnapshot(self._ID, 'minus', service)
    #    # snapshot_plus.setInitialSize(self._INITIAL_SIZE)
    #    # snapshot_minus.setInitialSize(self._INITIAL_SIZE)
    #    snapshot_plus.init()
    #    snapshot_minus.init()

    def ALL_FALSE(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return Signal(allfalse(self._SIZE[token]))

    def sig_product(self, siglist, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self.ALL_FALSE(token) if siglist == [] else Signal(np.array([sig._VAL for sig in siglist]).all(0))

    ## Make an observation of the current state
    def collect_observation(self):
        for token in ['plus', 'minus']:
            self._SNAPSHOTS[token].collect_observation()
        return None

    ## Report list of sensors
    def report_sensors(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SENSORS[token]

        ## Report the currently observed state

    def report_observed(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._OBSERVE

    ## Report the last state
    def report_last(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._LAST

    ## Report the current state for snapshot $token$, else for
    #  the active snapshot
    def report_current(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._CURRENT

    ## Report the target state for snapshot $token$, else for
    #  the active snapshot
    def report_target(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._TARGET

    ## Report the predicted state for snapshot $token$, else for
    #  the active snapshot
    def report_predicted(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._PREDICTED

    ## Report the initials mask for snapshot $token$, else for
    #  the active snapshot
    def report_initmask(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return self._SNAPSHOTS[token]._INITMASK

        ## Return an iterator containing the mids of initial sensors

    def report_initial(self, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return [sid for ind, sid in enumerate(self._SNAPSHOTS[token]._SENSORS) if self._SNAPSHOTS[token]._INITMASK.out(ind)]

    ## Report the delay masks (in the prescribed snapshot) for each of the sensors indicated in the input signal
    # def report_masks(self,signal,token=None):
    #      token=('plus' if self._ACTIVE else 'minus') if token is None else token
    #      return [(mid,Signal(ServiceSensor(self._ID,token,mid,service).getAmperList())) for ind,mid in enumerate(self._SENSORS[token]) if signal.out(ind)]

    ## Closure operators for a batch of sinals
    def close_upwards(self, siglist, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        snap = self._SNAPSHOTS[token]._SNAPSHOT_SERVICE
        return [Signal(item) for item in snap.make_ups([sig.out() for sig in siglist])]

    def close_downwards(self, siglist, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        snap = self._SNAPSHOTS[token]._SNAPSHOT_SERVICE
        return [Signal(item) for item in snap.make_downs([sig.out() for sig in siglist])]

    ## Propagation for a batch of signals (completely on GPU)
    def propagate(self, siglist, load=None, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        if siglist is []:
            return [self.ALL_FALSE(token)]
        else:
            if load is None:
                load = self.ALL_FALSE(token)
            snap = self._SNAPSHOTS[token]._SNAPSHOT_SERVICE
            return [Signal(item) for item in snap.make_propagation([sig.out() for sig in siglist], load)]

    ## Propagation for a batch of signals (closures on GPU, subtraction here)
    # def propagate(self, siglist, load=None, token=None):
    #      token=('plus' if self._ACTIVE else 'minus') if token is None else token
    #      if siglist is []:
    #          return [self.ALL_FALSE(token)]
    #      else:
    #          snap=ServiceSnapshot(self._ID,token,service)
    #          if load is None:
    #              up_sigs=[Signal(item) for item in snap.make_ups([sig.out() for sig in siglist])]
    #              return [sig.subtract(sig.star()) for sig in up_sigs]
    #          else:
    #              up_sigs=[Signal(item) for item in snap.make_ups([sig.add(load).out() for sig in siglist])]
    #              down_sigs=[Signal(item).star() for item in snap.make_ups([sig.out() for sig in siglist])]
    #              return [sig1.subtract(sig2) for sig1,sig2 in zip(up_sigs,down_sigs)]

    ## Perform abduction on a batch of signals, each indicating the collection of
    ## delayed queries to be generalized.
    def perform_abduction(self, sigs, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        if sigs is []:
            return [self.ALL_FALSE(token)]
        else:
            snap = self._SNAPSHOTS[token]._SNAPSHOT_SERVICE
            pos, neg = snap.make_abduction([sig.out() for sig in sigs])
            result = []
            for sig1, sig2 in zip(pos, neg):
                result.extend([Signal(sig1), Signal(sig2)])
            return list(set(result))

    ### Adding a sensor to the $token$ snapshot
    ### - assumes the sensors $mid$ and its complement are present
    ###   in the parent experiment
    def add_sensor(self, mid, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token == None else token
        if not self._READY:
            self._INITIAL_SIZE += 1
            for tok in ['plus', 'minus']:
                self._SNAPSHOTS[tok].add_sensor(mid, self._EXPERIMENT)
        else:
            self._SNAPSHOTS[token].add_sensor(mid, self._EXPERIMENT)

    ## Decide whether to act or not to act and write the decision into the experiment state
    def decide(self):
        dist = {}

        # mapping agent activity
        activity = {'plus': self._ACTIVE, 'minus': not (self._ACTIVE)}

        # acquire a new observation
        self.collect_observation()

        # move the latest record of the current state to "last state"
        for token in ['plus', 'minus']:
            self._SNAPSHOTS[token]._LAST = self.report_current(token)

        res = self._AGENT_SERVICE.make_decision(
            self._SNAPSHOTS['plus']._OBSERVE.out(),
            self._SNAPSHOTS['minus']._OBSERVE.out(),
            self._EXPERIMENT.this_state(self._MOTIVATION),
            self._ACTIVE
        )
        for token in ['plus', 'minus']:
            dist[token] = res[token]['res']
            self._SNAPSHOTS[token]._TARGET = Signal(res[token]['target'])
            self._SNAPSHOTS[token]._CURRENT = Signal(res[token]['current'])
            self._SNAPSHOTS[token]._PREDICTED = Signal(res[token]['prediction'])

        # make a decision
        token, qualityQ = rlessthan((dist['plus'], 'plus'), (dist['minus'], 'minus'))

        # Update the decision vector:
        # - only the decision vector gets updated -- not the activity
        #   status of the agent -- because the latter might be altered
        #   by other agents in the process of updating the experiment
        self._EXPERIMENT.this_state('decision').append(self._ID if token == 'plus' else self._CID)

        # return comment
        #return 'deliberate' if qualityQ else 'random'
        return qualityQ

    ## PICK OUT SENSORS CORRESPONDING TO A SIGNAL
    #  - returns a list of mids corresponding to the signal
    #  - signal must match $token$ snapshot
    def select(self, signal, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token == None else token
        return [mid for indicator, mid in zip(signal.value().tolist(), self._SNAPSHOTS[token]._SENSORS) if indicator]

    ## Provide signal encoding a list of sensors in the $token$ snapshot
    #  - listed sensors that are absent from the snapshot will not contribute anything to the signal
    def generate_signal(self, mid_list, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token is None else token
        return Signal([(tmp_mid in mid_list) for tmp_mid in self._SNAPSHOTS[token]._SENSORS])


    ### Form a delayed sensor
    ### - $signals$ is a list of signals, each describing a delayed
    ###   conjunction which must be added to the agent's snapshot $token$ / currently active snapshot
    def delay(self, signals, token=None):
        token = ('plus' if self._ACTIVE else 'minus') if token == None else token
        affected_snapshot = self._SNAPSHOTS[token]._SNAPSHOT_SERVICE
        uuids = []
        for i in range(len(signals)):
            uuids.append(['predelay' + str(i), 'c_delay' + str(i)])
        affected_snapshot.delay([signal.out() for signal in signals], uuids)

