from UMA_service import *
from Service_Agent import *

class ServiceWorld:
    def __init__(self):
        self._service = UMA_service()

    def add_agent(self, name, uuid):
        data = {'name': name, 'uuid': uuid}
        status = self._service.post('/UMA/data/agent', data)
        if not status:
            print "add agent failed!"
            return None
        else:
            return ServiceAgent(uuid)

    def get_agent_count(self):
        return