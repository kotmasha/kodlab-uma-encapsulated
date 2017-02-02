### Experiment and Snapshot base classes

from multiprocessing import Pool
#import cPickle as nedosol
import numpy as np
from wrapper import *
from numpy.random import randint as rand

N_CORES=8

### EXPERIMENT:
### Need new mechanism for adding measurables while recording dependencies
###

### MEASURABLES SHOULD BE A CLASS? 
### SIGNALS SHOULD BE A CLASS?

### SNAPSHOT:
### Stop using sensor names in computations
### Parallelize weight computation

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
      return set(comp(name) for name in names)

def wedge(name0,name1):
      ### conjunction of two names
      return '('+str(name0)+'^'+str(name1)+')'

###
### Normalizing functions
###

def simplex_normalize(probs):
      s=sum([x for x in probs if x>0])
      return [((x+0.)/(s+0.) if x>0. else 0.) for x in probs]

###
### DATA STRUCTURES
###

class Signal(object):
      def __init__(self,value):
            if len(value)%2==0:
                  self._VAL=np.array(value,dtype=bool)
            else:
                  raise Exception('Objects of class Signal must have even length -- Aborting!\n')

      def __repr__(self):
            print self._VAL

      ### set the signal
      def set(self,ind,value):
            self._VAL[ind]=value

      def value(self,ind):
            return self._VAL[ind]

      def value_all(self):
            return self._VAL

      def extend(self,value):
            if len(value)%2==0 and type(value)==type(self._VAL):
                  self._VAL=np.array(list(self._VAL)+list(value))
            else:
                  raise Exception("Cannot extend a signal by something that's not a signal -- Aborting!\n")
                  
      ### negating a partial signal
      def star(self):
            return Signal([(self._VAL[i+1] if i%2==0 else self._VAL[i-1]) for i in xrange(len(self._VAL))])
      
      ### full complement of a signal
      def negate(self):
            return Signal(negate(self._VAL))

      ### subtracting Signal "other" from Signal "self"
      def subtract(self,other):
            return Signal(conjunction(self._VAL,negate(other._VAL)))

      def add(self,other):
            return Signal(disjunction(self._VAL,other._VAL))

      def intersect(self,other):
            return Signal(conjunction(self._VAL,other._VAL))

#ADDED the field self._PROJECTED to all measurables (default value is None),
#   the value of this fields is Boolean.
#ADDED methods self.set_projected(value) and self.projected() to set and report
#   the value of this field, respectively.
class Measurable(object):
      ### initialize a measurable
      def __init__(self,name,experiment,value,definition=None):

            ### name will remain unchanged
            self._NAME=str(name)

            ### experiment will remain unchanged
            self._EXPERIMENT=experiment

            ### initial value is a stack containing no more than $experiment._DEPTH+1$ values
            self._VAL=value

            ### a function of the experiment state (the latter is a dictionary indexed by [all] measurables)
            self._DEFN=definition
            #ADDED a dictionary of the form $agent:value$ reserved for Boolean-
            #   valued measurables, where $agent$ is [a pointer to] an agent
            #   and $value$ is the projected value (if any) for the given
            #   measurable by the agent $name$.
            self._PROJECTED={}
            
      def __repr__(self):
            return str(self._VAL)

      def val(self):
            return self._VAL

      ### Arbitrarily set the current value of a measurable:
      ### (pushes preceding values down the value stack)
      def set(self,value):
            self._VAL=[value]+list(self.val())[:self._EXPERIMENT._DEPTH]
            return self._VAL

      #ADDED
      def set_projected(self,agent,value):
            self._PROJECTED[agent]=value

      #ADDED
      def projected(self,agent):
            return self._PROJECTED[agent]

class Experiment(object):
      def __init__(self,depth):
            ### depth of data recording in the experiment:
            self._DEPTH=depth
            ### list of agents in this experiment:
            self._AGENTS=[]

            ### measurables corresponding to binary control signals
            self._CONTROL=[]
            ### list of measurables ordered to accommodate dependencies during the updating process:
            self._MEASURABLES=[]
            ### name-based look-up table (dictionary) for the unified list of  measurables, including control signals:
            self._MNAMES={}

      ### represent an experiment
      def __repr__(self):
            intro="Experiment of depth "+str(self._DEPTH)+" including the agents "+str([agent._NAME for agent in self._AGENTS])+".\nThe current state is:\n"
            state_data=""
            ### SWITCH TO _CONTROLS and _MEASURABLES HERE:
            for meas in self._CONTROL:
                  state_data+=meas._NAME+":  "+str(meas.val())+"\n"
            for meas in self._MEASURABLES:
                  state_data+=meas._NAME+":  "+str(meas.val())+"\n"

            return intro+state_data

      ### initialize an agent
      def add_agent_empirical(self,name,threshold):
          new_agent=Agent(name,self,threshold)
          new_agent.brain=wrapper(new_agent,'EMPIRICAL',threshold)
          self._AGENTS.append(new_agent)
          return new_agent

      def add_agent_distributed(self,name,threshold):
          new_agent=Agent(name,self,threshold)
          new_agent.brain=wrapper(new_agent,'DISTRIBUTED',threshold)
          self._AGENTS.append(new_agent)
          return new_agent

      def add_agent_discounted(self,name,threshold,q):
          new_agent=Agent(name,self,threshold)
          new_agent.brain=wrapper(new_agent,'DISCOUNTED',threshold,q)
          self._AGENTS.append(new_agent)
          return new_agent

      ### initialize a measurable -- measurables are observable quantities 
      ### of the experiment, init is a $self._DEPTH+1$-dimensional vector of
      ### values of the measurable $name$.
      ###
      ### returns the new measurable
      def add_measurable(self,name,init,definition=None):
            if name in self._MNAMES:
                  raise Exception("The name ["+name+"] is already in use as the name of a measurable in this experiment -- Aborting!\n\n")
            else:
                  # construct the new measurable
                  new_measurable=Measurable(name,self,init,definition)
                  #the case of a control signal
                  if definition==None:
                        self._CONTROL.append(new_measurable)
                  # the case of a dependent measurable
                  else:
                        self._MEASURABLES.append(new_measurable)

                  # add the new measurable to related dictionaries
                  self._MNAMES[name]=new_measurable

                  return new_measurable

      ### query the state of the experiment:
      def state(self,name,delta=0):
            return self._MNAMES[name].val()[delta:]

      def state_all(self):
            return {name:self._MNAMES[name].val() for name in self._MNAMES.keys()}

      ### SIMULATION ONLY: update the state of the experiment given a choice 
      ### of actions, $action_signal$.
      ###
      ### we assume $action_signal$ is a boolean list corresponding to 
      ### self._CONTROL
      def update_state(self,action_signal):
            last_state={name:self._MNAMES[name].val() for name in self._MNAMES}
            for ind,meas in enumerate(self._CONTROL):
                  last_state[meas._NAME]=meas.set(action_signal[ind])
                  #print meas._NAME,meas.val()

            for meas in self._MEASURABLES:
                  last_state[meas._NAME]=meas.set(meas._DEFN(last_state))
                  #print meas._NAME,meas.val()

      ### ADD A SENSOR TO THE EXPERIMENT
      ###
      def new_sensor(self,agent_list,name,definition=None):
            # SETTING UP NEW SENSOR IN THE EXPERIMENT:
            # compute the initial values of the new measurable:
            # put 'False' in both $name$ and $name*$ over full depth of experiment
            new_meas=self.add_measurable(name,allfalse(1+self._DEPTH),definition)

            # $definition==None$ is used to designate action sensors
            new_meas_comp=self.add_measurable(name_comp(name),alltrue(1+self._DEPTH) if definition==None else allfalse(1+self._DEPTH),None if definition==None else lambda state: negate(definition(state)))

            ### add the sensor to the agents
            for agent in agent_list:
                  agent.add_sensor(name,new_meas,new_meas_comp,definition==None)
                  new_meas.set_projected(agent,False)
                  new_meas_comp.set_projected(agent,False)

            ### return the new pair of measurables
            return new_meas,new_meas_comp

      ### FORM A DELAYED CONJUNCTION
      ### (name1 is the delayed sensor)
      def twedge(self,agent_list,name0,name1):
            if name0 in self._MNAMES and name1 in self._MNAMES:
                  def newdefn(state):
                        return conjunction(state[name0][0],state[name1][1])
                  self.new_sensor(agent_list,wedge(name0,name1),newdefn)
                  for agent in agent_list:
                        agent.set_context(name0,name1,wedge(name0,name1))
                        #agent._CONTEXT[(agent._NAME_TO_NUM[name0],agent._NAME_TO_NUM[name1])]=agent._NAME_TO_NUM[wedge(name0,name1)]
            else:
                  raise('One of the provided component names is undefined in agent '+str(self._NAME)+' -- Aborting.\n')
 
                              
      ### ONE TICK OF THE CLOCK: 
      ### have agents decide what to do, then collect their decisions and update the measurables.
      def tick(self,mode,param):
          decision=[]
          message_all=''
          for agent in self._AGENTS:
              for ind in xrange(agent._SIZE):
                  agent._OBSERVE.set(ind,agent._SENSORS[ind].val()[0])
		      
              agent.brain.sendSignal()
		 
              agent.brain.decide(mode,param)
              #ADDED to output of agent.brain.getValue():
              #-  $projected_signal$ is the signal obtained as the result
              #      of the run of $halucinate$ corresponding to the
              #      decision $dec$;
              #-  $touched_workers$ is the list of workers "touched" during
              #      the same propagation (halucination) run.
              dec,projected_signal,message=agent.brain.getValue()
              #dec,projected_signal,touched_workers,message=agent.brain.getValue()
              #ADDED now the agent updates its predicted values for the
              #   measurables associated with its sensors and a Boolean matric
              #   representing the workers which were involved in generating
              #   this prediction
              agent.set_projected(projected_signal)
              #agent.set_touched(touched_workers)
              
              decision.extend(dec)
              message_all+=('\t'+agent._NAME+':\t'+message+'\n')
              
          self.update_state([(meas._NAME in decision) for meas in self._CONTROL])
          return message_all


#ADDED method self.projected() returning the signal currently projected by the
#   agent.
class Agent(object):
      ### initialize an empty snapshot with a list of sensors and a learning threshold
      ###
      def __init__(self,name,experiment,threshold):
            self._NAME=str(name) ### a string naming the snapshot
            self._EXPERIMENT=experiment ### the experiment observed by the snapshot
            self._SIZE=0 ### snapshot size is always even
            ### an ordered list of Boolean measurables used by the agent:
            self._SENSORS=[] 
            ### LOOKUP TABLES TO SAVE TIME
            ### a list of the actions available to the agent
            ### (indices in the self._SENSORS array)
            self._ACTIONS=[]
            self._GENERALIZED_ACTIONS=[[]]
            ### a list of available evaluator sensors:
            ### (establishes the agent's priorities)
            self._EVALS=[]
            ### a dictionary of sensor indices
            self._NAME_TO_NUM={} # $name:number$ pairs
            self._NAME_TO_SENS={} # $name:sensor$ pairs
            
            ### Boolean vectors ordered according to self._SENSORS
            ### raw observation:
            self._OBSERVE=Signal(np.array([],dtype=np.bool))
            ### current state representation:
            self._CURRENT=Signal(np.array([],dtype=np.bool))

            ### poc set learning machinery:
            ### learning thresholds matrix
            self._THRESHOLDS=np.array([[threshold]])
            ### snapshot weight matrix
            self._WEIGHTS=np.array([[0.]])
            ### snapshot graph matrix
            self._DIR=np.array([[False]],dtype=np.bool)

            ### context is a dictionary of the form (i,j):k indicating that sensor number k was constructed as twedge(i,j).
            self._CONTEXT={}
            #ADDED this list will be updated each tick of the clock, and will
            #   contain [pointers to the] workers "touched" in the process
            #   of obtaining the most recent decision
            #self._TOUCHED=np.array([[False]],dtype=np.bool)
                  
      def __repr__(self):
            return 'The snapshot '+str(self._NAME)+' has '+str(self._SIZE/2)+' sensors:\n\n'+str([meas._NAME for ind,meas in enumerate(self._SENSORS) if ind%2==0])+'\nout of which the following are actions:\n'+str([self._SENSORS[ind]._NAME for ind in self._ACTIONS])+'\n\n'
      
      ### adding a sensor to the agent
      ###
      def add_sensor(self,name,new_meas,new_meas_comp,actionQ):
            ### EXTENDING THE SNAPSHOT:
            self._SIZE+=2
            # adding new sensors to lists
            self._SENSORS.extend([new_meas,new_meas_comp])
            if actionQ:
                  self._ACTIONS.append(self._SIZE-2)
                  temp_list=self._GENERALIZED_ACTIONS[:]
                  for item in self._GENERALIZED_ACTIONS:
                        temp_list.extend([item+[self._SIZE-2],item+[self._SIZE-1]])
                        temp_list.remove(item)
                  self._GENERALIZED_ACTIONS=temp_list
            # extending lookup tables
            self._NAME_TO_SENS[name]=new_meas
            self._NAME_TO_SENS[name+'*']=new_meas_comp
            self._NAME_TO_NUM[name]=self._SIZE-2
            self._NAME_TO_NUM[name+'*']=self._SIZE-1
            ### update current values of sensor according to definition in both the experiment and the snapshot:
            self._OBSERVE.extend(np.array([self._EXPERIMENT.state(name)[0],self._EXPERIMENT.state(name_comp(name))[0]]))
            self._CURRENT.extend(np.array([False,False]))
            ### preparing weight, threshold and direction matrices:
            self._WEIGHTS=np.array([[self._WEIGHTS[0][0] for col in range(self._SIZE)] for row in range(self._SIZE)])
            self._THRESHOLDS=np.array([[self._THRESHOLDS[0][0] for col in range(self._SIZE)] for row in range(self._SIZE)])
            self._DIR=np.array([[False for col in range(self._SIZE)] for row in range(self._SIZE)],dtype=np.bool)

      ## Adding a sensor from another agent
      #
      def take_sensor(self,other,name):
            self.add_sensor(name,other._NAME_TO_SENS[name],other._NAME_TO_SENS[name_comp(name)],False)
            
      def add_eval(self,name):
            # any non-action sensor may serve as an evaluation sensor
            if name in self._NAME_TO_NUM and self._NAME_TO_NUM[name] not in self._ACTIONS:
                  self._EVALS.append(name)

      def set_context(self,name0,name1,name_tc):
            self._CONTEXT[(self._NAME_TO_NUM[name0],self._NAME_TO_NUM[name1])]=self._NAME_TO_NUM[name_tc]
            return None

      #ADDED: sets the projected values of the agent's sensorium to
      #   a given signal
      def set_projected(self,signal):
            for ind,value in enumerate(signal):
                  self._SENSORS[ind].set_projected(self,value)
                            
      #ADDED: returns a signal formed of the projected sensor values currently
      #   stored in the measurables corresponding to the agent's sensorium
      def projected(self):
            return Signal([item.projected(self) for item in self._SENSORS])

      #ADDED set/return the list of most recently "touched" workers
      #def set_touched(self,touched_matrix):
      #      self._TOUCHED=np.copy(touched_matrix)
      #def touched(self):
      #      return self._TOUCHED

      #ADDED a function to return the value of self._CURRENT
      #---> Siqi, is self._CURRENT somehow updated on your ("brain") side of
      #     the code? If not, this needs to be taken care of: it should be
      #     updated to the result of propagating the raw observation of
      #     current sensor values through the agent's snapshot.
      def current(self):
            return self._CURRENT #return a signal corresponding to the agent's perceived current state


      