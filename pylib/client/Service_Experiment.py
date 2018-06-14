from UMA_service import *
from Service_Agent import *

class ServiceExperiment:
    def __init__(self, experiment_id, service):
        self._experiment_id = experiment_id
        self._service = service

    def add_agent(self, agent_id, type='default'):
        data = {'experiment_id': self.experiment_id, 'agent_id': agent_id, 'type': type}
        result = self._service.post('/UMA/object/agent', data)
        if not result:
            print "create agent=%s failed!" % agent_id
            return None
        else:
            return ServiceAgent(self._experiment_id, agent_id, self._service)

"""
    def save(self, filename):
        data = {'filename': filename}
        result = self._service.post('/UMA/simulation/saving', data)
        if not result:
            print "data saving failed!"
            return None
        else:
            return result

    def load(self, filename):
        data = {'filename': filename}
        result = self._service.post('/UMA/simulation/loading', data)
        if not result:
            print "data loading failed!"
            return None
        else:
            return result

    def merge(self):
        data =  {}
        result = self._service.post('/UMA/simulation/merge', data)
        if not result:
            return None
        else:
            return result
            
"""