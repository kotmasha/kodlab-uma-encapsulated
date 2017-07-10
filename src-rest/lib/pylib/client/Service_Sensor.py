from UMA_service import *

class ServiceSensor:
    def __init__(self, agent_id, snapshot_id, sensor_id, service):
        self._agent_id = agent_id
        self._snapshot_id = snapshot_id
        self._sensor_id = sensor_id
        self._service = service

    def getAmperList(self):
        result = self._service.get('/UMA/object/sensor', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'sensor_id': self._sensor_id})
        if not result:
            return None
        result = result['data']['amper_list']
        return result

    def getAmperListID(self):
        result = self._service.get('/UMA/object/sensor', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'sensor_id': self._sensor_id})
        if not result:
            return None
        result = result['data']['amper_list_id']
        return result

    def setAmperList(self, amper_list):
        result = self._service.post('/UMA/object/sensor', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'sensor_id': self._sensor_id, 'amper_list': amper_list})
        if not result:
            return None
        return result