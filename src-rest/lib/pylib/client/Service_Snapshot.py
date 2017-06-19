from UMA_service import *
from Service_Sensor import *

class ServiceSnapshot:
    def __init__(self, agent_id, snapshot_id):
        self._agend_id = agent_id
        self._snapshot_id = snapshot_id
        self._service = UMA_service()

    def add_sensor(self, sensor_id):
        data = {'agent_id': self._agend_id, 'snapshot_id': self._snapshot_id, 'uuid': sensor_id, 'name': sensor_id}
        status =  self._service.post('/UMA/data/sensor', data)
        if not status:
            print "add sensor failed!"
            return None
        else:
            return ServiceSensor(self._agend_id, self._snapshot_id, sensor_id)

    def validate(self):
        data = {'agent_id': self._agend_id, 'snapshot_id': self._snapshot_id}
        return self._service.post('/UMA/validation/snapshot', data)

    def simulation(self, signals, phi, active):
        data =  {'agent_id': self._agend_id, 'snapshot_id': self._snapshot_id, 'phi': phi, 'active': active, 'signals': signals}
        return self._service.post('/UMA/simulation/snapshot', data)

    def get_sensor_count(self):
        return

    def get_sensor_pair_count(self):
        return