from UMA_service import *
from Service_Snapshot import *

class ServiceAgent:
    def __init__(self, agent_id):
        self._agent_id = agent_id
        self._service = UMA_service()

    def add_snapshot(self, name):
        snapshot_id = self._agent_id + '_' + name
        data = {'name': name, 'uuid': snapshot_id, 'agent_id': self._agent_id}
        status =  self._service.post('/UMA/data/snapshot', data)
        if not status:
            print "add snapshot failed!"
            return None
        else:
            return ServiceSnapshot(self._agent_id, snapshot_id)

    def get_snapshot_count(self):
        return