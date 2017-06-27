from UMA_service import *
from Service_Sensor import *

class ServiceSnapshot:
    def __init__(self, agent_id, snapshot_id, service):
        self._agent_id = agent_id
        self._snapshot_id = snapshot_id
        self._service = service

    def add_sensor(self, sensor_id):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'uuid': sensor_id, 'name': sensor_id}
        result =  self._service.post('/UMA/object/sensor', data)
        if not result:
            print "add sensor failed!"
            return None
        else:
            return ServiceSensor(self._agent_id, self._snapshot_id, sensor_id, self._service)

    def validate(self):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/validation/snapshot', data)
        if not result:
            return False
        return True

    def init_with_sensors(self, sensors):
        for sensor in sensors:
            self.add_sensor(sensor)
        self.validate()

    def make_up(self, signals):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signals}
        result = self._service.post('/UMA/simulation/up', data)
        if not result:
            return None
        return list(result['data'])

    def make_decision(self, signals, phi, active):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'phi': phi, 'active': active, 'signals': signals}
        #print signals
        result = self._service.post('/UMA/simulation/decision', data)
        if not result:
            return None
        result = result['data']
        return float(result['res']), result['current'], result['prediction'], result['target']

    def delay(self, delay_list, uuid_list):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'delay_list': delay_list, 'uuid_list': uuid_list}
        result = self._service.post('/UMA/simulation/delay', data)
        if not result:
            return False
        return True

    def getCurrent(self):
        return self._service.get('/UMA/data/current', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getPrediction(self):
        return self._service.get('/UMA/data/prediction', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getTarget(self):
        return self._service.get('/UMA/data/target', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setName(self, name):
        return self._service.put('/UMA/object/snapshot', {'name': name}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setQ(self, q):
        return self._service.put('/UMA/object/snapshot', {'q': q}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setThreshold(self, threshold):
        return self._service.put('/UMA/object/snapshot', {'threshold': threshold}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setTarget(self, target_list):
        return self._service.put('/UMA/data/target', {'target_list': target_list}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def get_sensor_count(self):
        return

    def get_sensor_pair_count(self):
        return