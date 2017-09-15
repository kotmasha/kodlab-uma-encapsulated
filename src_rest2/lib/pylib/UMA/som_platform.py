#from multiprocessing import Pool
from collections import deque
from Service_World import *
from Service_Agent import *
from Service_Snapshot import *
from Service_Sensor import *
import numpy as np
import uuid
import time
from numpy.random import randint as rnd
import json

N_CORES=8
service = UMA_service()
###
### Randomized inequality
###

PRECISION=pow(2,-64)
# Return the token associated with the smaller value if precision threshold
# is met and report a strong inequality; otherwise return a random token and
# report a weak(=uncertain) inequality.
def rlessthan(x,y,prec=PRECISION):
      xval,xtok=x
      yval,ytok=y
      if yval-xval>abs(prec):
            return xtok,True
      elif xval-yval>abs(prec):
            return ytok,True
      else:
            return (xtok if bool(rnd(2)) else ytok),False

###
### Handling Boolean functions
###

def func_amper(experiment,mid_list):
      def f(state):
            return all([experiment._DEFS[mid](state) for mid in mid_list])
      return f

def func_not(func):
      def f(state):
            return not(func(state))
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

def conjunction(x,y):
      return np.logical_and(x,y)

def disjunction(x,y):
      return np.logical_or(x,y)

def symmetric(x,y):
      return np.logical_xor(x,y)

def alltrue(n):
      return np.array([True for x in xrange(n)])

def allfalse(n):
      return np.array([False for x in xrange(n)])

###
### Name-handling functions
###

def name_comp(name):
      ### return the name of the complementary sensor
      return name+'*' if name[-1:]!='*' else name[:-1]

def name_invert(names):
      ### return the set of complemented names in the list/set names
      return set(name_comp(name) for name in names)

def name_delay(name):
      ### delay
      return '#'+str(name)

def name_ampersand(name_list):
      ### conjunction
      L=len(name_list)
      if L==0:
            raise Exception('\nEmpty conjunction not allowed.\n')
      elif L==1:
            return name_list[0]
      else:
            return '{'+';'.join(name_list)+'}'


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
            print self._VAL

      def len(self):
            return len(self._VAL)

      # set the signal
      def set(self, ind, value):
            self._VAL[ind] = value

      # inspect the signal
      def value(self, ind):
            return self._VAL[ind]

      # report the signal
      def value_all(self):
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


ALL_FALSE = lambda agent: Signal(allfalse(agent._SIZE))

### Data type maintaining the "environment" state and its interactions
### with the agents

class Experiment(object):
      def __init__(self):
            # dictionary of agents in this experiment, by uuid
            self._AGENTS={}

            # registering the decision observable
            self._ID = set()
            self._ID.add('decision')
            # Dictionaries translating user-assigned names to
            # system-assigned uuids
            self._ID_TO_DEP={'decision':False}
            
            # List of names of the experiment measurables:
            # - ordered to accommodate dependencies during the
            #   updating process; 
            # - initialized to contain the trivial sensors.
            self._MID=['decision']

            # ID-based representation of the experiment state
            # - each entry is $key:deque$;
            # - the trivial measurables initialized:
            self._STATE={'decision':deque([[]],1)}
            ### ID-based representation of the currently evolving decision
            
            ### Name-based representation $self._DEFS$ of the state update:
            ### - each entry is $key:function$;
            ### - the function accepts a dictionary of the same format as
            ###   $self._STATE$.
            def ex_decision(state):
                  return state['decision'][0]
            self._DEFS={'decision':ex_decision}


      def load_experiment(self, filename):
            ServiceWorld(service).load(filename)
            f = open(filename + '.txt', 'r')
            self.basic_sensors = []
            self.complex_sensors = []
            self.agents = []
            self.measurables = []

            for line in f:
                  data = json.loads(line)
                  # get id and type
                  id = data['id']
                  type = data['type']
                  if type == 'measurable':
                        # save in measurable
                        self.measurables.append(id)
                  elif type == 'agent':
                        # save in agent
                        self.agents.append(id)
                  else:
                        # save in sensor
                        if data['is_basic']:
                              # save in basic sensor
                              self.basic_sensors.append(id)
                        else:
                              # save in complex sensor
                              if 'pure_id' not in data:
                                    # this is a pure sensor, must have amper list
                                    self.complex_sensors.append({'id': id, 'amper_list': data['amper_list']})
                              else:
                                    # this is a complementary sensor, must have a pure id
                                    self.complex_sensors.append({'id': id, 'pure_id': data['pure_id']})

      def save_experiment(self, filename):
            ServiceWorld(service).save(filename)
            f = open(filename + '.txt', 'w+')
            existing_sensors = {} # id to whether it is basic sensor
            for id in self._MID:
                  data = {}
                  # also save sensor values from observation vectors
                  data['id'] = id
                  if id in self._AGENTS:
                        # the type is agent
                        data['type'] = 'agent'
                  else:
                        tmp_agent = None
                        # may need to optimize in the future
                        for agent_name, agent in self._AGENTS.iteritems():
                              if id in agent._SENSORS:
                                    tmp_agent = agent
                                    break
                        if tmp_agent is None:
                              # not exist in any agent, just an experiment measurable
                              data['type'] = 'measurable'
                              # SIQI: update the taxonomy of types...
                              #       need to save a dict "which agent have this sensor?"
                        else:
                              # found an agent hold the measurable, then it is a sensor
                              data['type'] = 'sensor'
                              if name_comp(id) in existing_sensors:
                                    # if find the pure id, then this one must be a complementary
                                    pure_id = name_comp(id)
                                    data['pure_id'] = pure_id
                                    data['is_basic'] = existing_sensors[pure_id]
                              else:
                                    # it is a pure measurable
                                    amper_list_id = ServiceSensor(tmp_agent._MID, 'plus', id, service).getAmperListID()
                                    data['amper_list'] = amper_list_id
                                    if len(amper_list_id) == 0:
                                          # indicating if it is basic sensor or not
                                          data['is_basic'] = True
                                          existing_sensors[id] = True
                                    else:
                                          data['is_basic'] = False
                                          existing_sensors[id] = False
                  json.dump(data, f)
                  f.write('\n')
            f.close()

      def reconstruct(self):
            diff_measurables = []
            diff_basic_sensors = []

            #SIQI: new terminology --
            # "old/new scripted sensor" = sensor that was given in the old/new script
            # "initial sensor for agent X" = these are the sensors from which delayed sensors are built

            #for mid in self.measurables:
            #      if mid not in self._ID:
            #            diff_measurables.append(mid)
                        # maybe this will be useful later

            # record all basic sensors that is not used in this test
            for sensor in self.basic_sensors:
                  if sensor not in self._ID:
                        diff_basic_sensors.append(sensor)

            for data in self.complex_sensors:
                  id = data['id']
                  if 'amper_list' in data:
                        [data['amper_list'].remove(aid) for aid in data['amper_list'] if aid in diff_basic_sensors]
                        if len(data['amper_list']) > 0:
                              self.register(id)
                              self.construct_measurable(id, func_delay(data['amper_list']), init_value = [False, False])
                              #SIQI: something's wrong in the next line -- there are no 'rt', 'lt' in the
                              #      general experiment.
                        else:
                              pass
                  else:
                        if data['pure_id'] in self._ID:
                              self.register(id)
                              self.construct_measurable(id, func_not(data['pure_id']), init_value = [False, False])
                              self.assign_sensor(id, False, False, ['rt', 'lt'])

      # The register function, just do id register.
      # If id is None, generate a new one, if not just add it
      # The function is NOT supposed to be used in any test file directly
      def register(self, mid = None):
            """
            :param id: the id to be registered, if it is None, will generate an uuid
            :return: the id
            """
            if mid is None: # means the sensor is a new one, uuid will be generated
                  new_id = str(uuid.uuid4())
                  self._ID.add(new_id)
                  return new_id
            elif mid in self._ID:
                  raise Exception('ID $'+str(mid)+'$ already registered -- Aborting.')
            else: # need to add the id. if same name provided, just override
                  self._ID.add(mid)
                  return mid

      # The function for register sensor
      # If id is None, generate 2 new sensor ids
      # The function is NOT supposed to be used in any test file directly
      def register_sensor(self, id_string = None):
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

      def dep(self,mid):
            try:
                  return self._ID_TO_DEP[mid]
            except:
                  raise Exception('This id is not registered.')
      
      ## Front-end query (name-based) of an experiment state variable
      def this_state(self,mid,delta=0):
            try:
                  return self._STATE[mid][delta]
            except:
                  pass
                  
      ## Internal query of an experiment state variable
      def state_all(self):
            return self._STATE
 
      ## Set new state value
      def set_state(self,mid,value):
            self._STATE[mid].appendleft(value)
            return None

      # Construct the measurable using id
      def construct_measurable(self, mid, definition=None, init_value=None, depth=1, decdep = False):
            """
            :param mid:  the input id, if it is none, id will be generated
            :param definition: the definition function of the measurable
            :param init_value: the init value for the measurable
            :param depth: the depth of the measurable
            :param decdep: value indicating if the measurable is decision dependent
            :return: nothing
            """
            # check mid first
            if mid not in self._ID:
                  raise Exception("the mid " + mid + " is not registered!")

            ## SIQI: let's completely separate registration from construction;
            #        construction assumes registration

            # add the mid into the MID list, ID_TO_DEP, and DEFS
            self._MID.append(mid)
            self._ID_TO_DEP[mid] = bool(decdep)
            self._DEFS[mid] = definition
            if init_value is None: # will try to generate the init_value based on the definition
                  self._STATE[mid]=deque([],depth+1)
                  # -------------------- Remove the old try/except block, because: -----------------------
                  # 1 if definition is None, getting this by throwing an exception will reduce performance, an if else check will be faster
                  # 2 if definition is not None, and there is an error generating values, then the exception will not be caught because those error are unexpected
                  if definition is not None: # if definition exist, calculate the init value, any error will throw to outer scope
                        self.set_state(mid, definition(self._STATE))
            else:
                  self._STATE[mid]=deque(init_value,depth+1)
            return None

      #construct the agent using id
      def construct_sensor(self, mid, definition=None, init_value=None, depth=1, decdep = False):
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
                  raise Exception("the mid " + mid + " is not registered!")

            #compute initial value of sensor
            if definition==None: # this is an action sensor, init value WILL be defaulted to None...
                  self.construct_measurable(mid, None, [False for ind in xrange(depth + 1)], depth, decdep)
                  self.construct_measurable(midc, None, [True for ind in xrange(depth + 1)], depth, decdep)
            else:
                  self.construct_measurable(mid, definition, init_value, depth, decdep)
                  self.construct_measurable(midc, func_not(definition), negate(init_value) if init_value else None, depth, decdep)
            return None


      # Add the indicated sensor to the listed agents (by mid)
      def assign_sensor(self, mid, is_initial, is_base_sensor, agent_id_list):
            for id_agent in agent_id_list:
                  try:
                        self._AGENTS[id_agent].add_sensor(mid, is_initial, is_base_sensor)
                  except:
                        raise Exception('Attempted sensor assignment to unregistered agent.')
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
                  # construct new sensor and agent and append to agents list
                  self.construct_sensor(id_agent, definition, decdep = decdep)
                  new_agent = Agent(self, id_agent, id_motivation, params)
                  self._AGENTS[id_agent]=new_agent
                  ServiceWorld(service).add_agent(id_agent)
                  return new_agent
            else:
                  raise Exception("the agent " + id_agent + " is not registered!")

      ## Update the experiment state and output a summary of decisions
      def update_state(self, instruction='decide'):
            id_dec = 'decision'
            self.set_state(id_dec, [])
            #prepare empty dictionary for agent messages
            messages={}
            
            ## Iterate over agents and decision-dependent measurables
            # - if id of an agent, then form a decision and append to id_dec
            # - if id of a measurable, update its value
            for mid in (tmp_id for tmp_id in self._MID if self.dep(tmp_id)):
                  midc = name_comp(mid)
                  agentQ = mid in self._AGENTS or midc in self._AGENTS

                  if agentQ:
                        if mid in self._AGENTS:  # if mid is an agent...
                              ## agent activity set to current reading
                              agent = self._AGENTS[mid]
                              agent.active = self.this_state(mid)

                              ## ENRICHMENT
                              # agent analyzes outcomes of preceding decision
                              # cycle; if the prediction is too broad, add a
                              # sensor corresponding to the episode last observed
                              # and acknowledged by the active snapshot of the agent:
                              if sum(agent.report_current().subtract(agent.report_predicted()).value_all()) > 0:
                                    new_sensor_signals = [agent.report_last().intersect(agent._INITMASK)]
                              else:
                                    new_sensor_signals = []

                              #
                              # - add new sensors
                              #agent.delay(new_sensor_signals)

                              ## PRUNING
                              #sensors_to_be_removed=ALL_FALSE(agent)

                              # - gather the negligible delayed sensors, cluster them, and abduce;
                              #   NOT YET IMPLEMENTED
                              # - for each initial sensor $x$:
                              #   1. Gather the mask vectors of delayed sensors implying $x$ in the active
                              #      snapshot;
                              #   2. abduce to form $s_x$;
                              #   3. 
                              #service_active=ServiceSnapshot(agent._MID,'plus' if agent.active else 'minus',service)
                              #for ind,mid in enumerate(agent._SENSORS) if agent._INITMASK.value(ind):
                              #    sig=agent.generate_signal(mid)
                              #    sig_active=Signal(service_active.make_up(sig.star())).star().intersect(agent._INITMASK.negate())
                                  
                                  #STOPPED HERE

                              # - remove all old sensors involved in this process
                              #agent.prune(sensors_to_be_removed)

                              ## agent makes a decision
                              messages[mid] = agent.decide()
                              if instruction != 'decide' and (mid in instruction or midc in instruction):
                                    # append override information to message
                                    messages[mid] += '\tINSTRUCTION OVERRIDE: EXECUTING ' + mid if mid in instruction else midc
                                    # replace last decision with provided instruction
                                    self.this_state(id_dec).pop()
                                    self.this_state(id_dec).append(mid if mid in instruction else midc)

                        else:  # if midc is an agent, decision already reached
                              pass
                  else:  # neither mid nor midc is an agent, perform update
                        try:  # attempt update using definition
                              self.set_state(mid, self._DEFS[mid](self._STATE))
                        except:  # if no definition available, do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                              pass

            # At this point, there is a complete decision vector
            action_signal=self.this_state(id_dec)
            for mid in self._MID:
                  midc = name_comp(mid)
                  agentQ = mid in self._AGENTS or midc in self._AGENTS
                  depQ=self.dep(mid)
                  
                  if agentQ: # if mid is an agent (or its star)...
                        try: # try evaluating the definition for the agent
                              self.set_state(mid,(self._DEFS[mid](self._STATE)))
                              if mid in self._AGENTS and self.this_state(mid)!=(mid in action_signal): # if initial decision was changed
                                    messages[mid]+=', override by other (was '+(mid if mid in action_signal else name_comp(mid))+')'
                        except: # if no definition, set the value according to $action_signal$
                              self.set_state(mid,(mid in action_signal))
                  else:
                        try: # try updating using definition
                              self.set_state(mid,(self._DEFS[mid](self._STATE)))
                        except: # if no definition available then do nothing; this is a state variable evolving independently of the agent's actions, e.g., a pointer to a data structure.
                              pass

                        
            #aggregate and output the messages
            message_all = ""
            ordered_agents = [mid for mid in self._MID if mid in self._AGENTS]
            for mid in ordered_agents:
                  name = mid
                  outp='\t'+(name if self.this_state(mid) else name_comp(name))+' : '+messages[mid]+'\n'
                  message_all+=outp

            return message_all
      


class Agent(object):
      ### initialize an "empty" agent with prescribed learning parameters
      def __init__(self, experiment, id_agent, id_motivation, params):
            # a string naming the agent/action
            self._MID=id_agent
            self.id_agentc=name_comp(id_agent)
            self._MOTIVATION=id_motivation
            # the experiment serving as the agent's environment
            self._EXPERIMENT=experiment
            # the agent's parameters
            self._PARAMS=params
            # snapshot size is always even
            self._SIZE=0
            # ordered list of the Boolean measurables used by the agent:
            self._SENSORS=[]

            ## Boolean vectors ordered according to self._SENSORS

            # raw observation:
            self._OBSERVE=Signal(np.array([],dtype=np.bool))
            # previous state representation:
            self._LAST={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }

            # current state representation:
            self._CURRENT={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }
            # predicted state representation:
            self._PREDICTED={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }

            # target representation:
            self._TARGET={
                  'plus':Signal(np.array([],dtype=np.bool)),
                  'minus':Signal(np.array([],dtype=np.bool))
            }
            self._INITMASK=Signal(np.array([]))
            
            ## Calling the wrapper in charge of communication with CUDA side
            self.active = False
            self._base_sensor_size = 0

      ## Activate the agent when the initial definitions stage is completed
      def init(self):
            self.collect_observation()
            agent = ServiceAgent(self._MID, service)
            snapshot_plus = agent.add_snapshot('plus')
            snapshot_minus = agent.add_snapshot('minus')
            snapshot_plus.setQ(self._PARAMS[0])
            snapshot_plus.setAutoTarget(self._PARAMS[1])
            snapshot_minus.setQ(self._PARAMS[0])
            snapshot_minus.setAutoTarget(self._PARAMS[1])
            snapshot_plus.init_with_sensors([[self._SENSORS[2 * i], self._SENSORS[2 * i + 1]] for i in range(self._SIZE / 2)])
            snapshot_minus.init_with_sensors([[self._SENSORS[2 * i], self._SENSORS[2 * i + 1]] for i in range(self._SIZE / 2)])

      def prune(self,midsig):
            #SIQI: is this line really necessary? -->
            #agent = ServiceAgent(self._MID, service)

            snapshot_plus = ServiceSnapshot(self._MID, 'plus', service)
            snapshot_minus = ServiceSnapshot(self._MID, 'minus', service)
            # SIQI: I've made a change here....
            #       Can you make it so that I am able to just input a signal?
            snapshot_plus.pruning(midsig._VAL.tolist())
            snapshot_minus.pruning(midsig._VAL.tolist())

      def validate(self):
            #SIQI: is this line really necessary? -->
            #agent = ServiceAgent(self._MID, service)

            snapshot_plus = ServiceSnapshot(self._MID, 'plus', service)
            snapshot_minus = ServiceSnapshot(self._MID, 'minus', service)
            snapshot_plus.validate(self._base_sensor_size)
            snapshot_minus.validate(self._base_sensor_size)

      def collect_observation(self):
            self._OBSERVE=Signal([self._EXPERIMENT.this_state(mid) for mid in self._SENSORS])
            #for ind,mid in enumerate(self._SENSORS):
            #      try:
            #            value=self._EXPERIMENT.this_state(mid)
            #      except:
            #            #raise Exception('\nAHA!!!\n')
            #            value=False
            #self._OBSERVE.set(ind,value)
            return None
                              
      ### Adding a sensor to the agent
      ### - intended ONLY to be called by $experiment.new_sensor$
      ### - assumes the measurable $name$ and its complement are present
      ###   in the parent experiment
      def add_sensor(self,mid,is_initial, is_base_sensor):
            #verify proper registration of $mid$
            try:
                  midc = name_comp(mid)
            except:
                  raise Exception('Sensor id not properly registered.')
            ## Extending the sensor lists
            if mid not in self._SENSORS:
                  self._SIZE+=2
                  if is_base_sensor:
                        self._base_sensor_size += 1
                  self._SENSORS.extend([mid,midc])
                  ## update observed signal
                  self._OBSERVE.extend(np.array([self._EXPERIMENT.this_state(mid),self._EXPERIMENT.this_state(midc)],dtype=bool))
                  ## expand maintained signals
                  for token in ['plus','minus']:
                        self._CURRENT[token].extend(np.array([False,False],dtype=bool))
                        self._TARGET[token].extend(np.array([False,False],dtype=bool))
                        self._PREDICTED[token].extend(np.array([False,False],dtype=bool))
                        self._LAST[token].extend(np.array([False,False],dtype=bool))
                  self._INITMASK.extend(np.array([is_initial,is_initial],dtype=bool))


      ## Report the last state
      def report_last(self,token=None):
            try:
                  return self._LAST[token]
            except:
                  return self._LAST['plus' if self.active else 'minus']

      
      ## Report the current state for snapshot $token$, else for
      #  the active snapshot
      def report_current(self,token=None):
            try:
                  return self._CURRENT[token]
            except:
                  #print "@"
                  return self._CURRENT['plus' if self.active else 'minus']

      ## Report the target state for snapshot $token$, else for
      #  the active snapshot
      def report_target(self,token=None):
            try:
                  return self._TARGET[token]
            except:
                  return self._TARGET['plus' if self.active else 'minus']

      ## Report the current state for snapshot $token$, else for
      #  the active snapshot
      def report_predicted(self,token=None):
            try:
                  return self._PREDICTED[token]
            except:
                  return self._PREDICTED['plus' if self.active else 'minus']

      def decide(self):
            dist={}
            id_agent=self._MID
            id_agentc=name_comp(id_agent)
            id_dec = 'decision'

            # mapping agent activity
            activity={'plus':self.active,'minus':not(self.active)}

            # acquire a new observation
            self.collect_observation()

            # move the latest record of the current state to "last state"
            for token in ['plus','minus']:
                self._LAST[token]=self.report_current(token)

            agent = ServiceAgent(self._MID, service)
            res = agent.make_decision(self._OBSERVE._VAL.tolist(), self._EXPERIMENT.this_state(self._MOTIVATION), activity['plus'])
            for token in ['plus','minus']:
                  dist[token] = res[token]['res']
                  self._TARGET[token]=Signal(res[token]['target'])
                  self._CURRENT[token]=Signal(res[token]['current'])
                  self._PREDICTED[token]=Signal(res[token]['prediction'])

            # make a decision
            token,quality=rlessthan((dist['plus'],'plus'),(dist['minus'],'minus'))

            # Update the decision vector:
            # - only the decision vector gets updated -- not the activity
            #   status of the agent -- because the latter might be altered
            #   by other agents in the process of updating the experiment
            self._EXPERIMENT.this_state(id_dec).append(id_agent if token=='plus' else id_agentc)
            
            # return comment
            return 'deliberate' if quality else 'random'

            
            
      ## PICK OUT SENSORS CORRESPONDING TO A SIGNAL
      #  - returns a list of mids corresponding to the signal  
      def select(self,signal):
            return [mid for indicator,mid in zip(signal.value_all().tolist(),self._SENSORS) if indicator]

      def generate_signal(self,mid_list):
            return Signal([(tmp_mid in mid_list) for tmp_mid in self._SENSORS])
      
      ### FORM NEW CONJUNCTIONS
      ### - $signals$ is a list of signals, each describing a conjunction
      ###   that needs to be added to the snapshot
      def amper(self,signals):
            new_signals=[]
            for signal in signals:
                  #if signal is trivial (1 or no sensors), skip it
                  if sum(signal.value_all()) < 2: 
                        continue
                  #transform signal into a list of mids
                  mid_list=self.select(signal)
                  #new name as a means of verifying redundancy (CHANGE THIS!!)
                  new_name=name_ampersand([self._EXPERIMENT.din(mid) for mid in mid_list])
                  #construct definition for new sensor
                  new_def=func_amper(self._EXPERIMENT,mid_list)
                  #determine dependency on initial decisions
                  new_dep=any([self._EXPERIMENT._ID_TO_DEP[mid] for mid in mid_list])
                  #register/construct/assign new sensor to self
                  try: #in case the sensor is not even registered...
                        #register the new sensor
                        new_mid,new_midc=self._EXPERIMENT.register_sensor(new_name,new_dep)
                        #construct the new sensor
                        self._EXPERIMENT.construct_sensor(new_mid,new_def)
                        #add the new sensor to this agent as non-initial
                        self.add_sensor(new_mid,False, False)
                        new_signals.append(signal)
                  except: #if the sensor was registered
                        new_mid=self._EXPERIMENT.nid(new_name)
                        if new_mid in self._SENSORS:
                              pass
                        else:
                              self.add_sensor(new_mid,False, Fals)
                              new_signals.append(signal)
            if new_signals:
                  self.brain.amper(new_signals)
            else:
                  pass
            return None
     
      ### Form a delayed sensor
      ### - $signals$ is a list of signals, each describing a delayed 
      ###   conjunction which must be added to the agent's snapshots
      def delay(self,signals):
            new_signals=[]
            new_uuids = []
            for signal in signals:
                  #if signal is trivial (no sensors), skip it
                  if sum(signal.value_all()) < 1: 
                        continue
                  #transform signal into a list of mids
                  mid_list=self.select(signal)
                  #new name as a means of verifying redundancy (CHANGE THIS!!)
                  new_mid = name_delay(name_ampersand(mid_list))
                  new_midc = name_comp(new_mid)
                  #construct definition for new sensor
                  #def new_def(state):
                  #      return all([state[mid][1] for mid in mid_list])
                  new_def=func_delay(mid_list)
                  #determine dependency on initial decisions
                  new_dep=any([self._EXPERIMENT._ID_TO_DEP[mid] for mid in mid_list])
                  #register/construct/assign new sensor to self
                  if new_mid not in self._EXPERIMENT._ID: #in case the sensor is not even registered...
                        #construct the new sensor
                        self._EXPERIMENT.register_sensor(new_mid)
                        self._EXPERIMENT.construct_sensor(new_mid, new_def, decdep=new_dep)
                        #add the new sensor to this agent as non-initial
                        self.add_sensor(new_mid, False, False)
                        new_signals.append(signal)
                        new_uuids.append([new_mid, new_midc])
                  else: #if the sensor was registered
                        if new_mid not in self._SENSORS:
                              self.add_sensor(new_mid,False, False)
                              new_signals.append(signal)
                              new_uuids.append([new_mid, new_midc])
            if new_signals:
                  agent = ServiceAgent(self._MID, service)
                  snapshot_plus = ServiceSnapshot(self._MID, 'plus', service)
                  snapshot_minus = ServiceSnapshot(self._MID, 'minus', service)
                  snapshot_plus.delay([signal._VAL.tolist() for signal in new_signals], new_uuids)
                  snapshot_minus.delay([signal._VAL.tolist() for signal in new_signals], new_uuids)
            else:
                  pass
            return None

