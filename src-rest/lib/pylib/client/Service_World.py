from UMA_service import *
from Service_Agent import *

class ServiceWorld:
    def __init__(self, service):
        self._service = service

    def add_agent(self, uuid):
        data = {'uuid': uuid}
        result = self._service.post('/UMA/object/agent', data)
        if not result:
            print "add agent failed!"
            return None
        else:
            return ServiceAgent(uuid, self._service)

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

    def get_agent_count(self):
        return

    def merge(self):
        data =  {}
        result = self._service.post('/UMA/simulation/merge', data)
        if not result:
            return None
        else:
            return result