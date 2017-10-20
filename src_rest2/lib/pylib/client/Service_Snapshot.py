from UMA_service import *
from Service_Sensor import *

class ServiceSnapshot:
    def __init__(self, agent_id, snapshot_id, service):
        self._agent_id = agent_id
        self._snapshot_id = snapshot_id
        self._service = service

    def add_sensor(self, sensor_id, c_sensor_id):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'sensor_id': sensor_id, 'c_sid': c_sensor_id, 'w': [], 'd': [], 'diag': []}
        result =  self._service.post('/UMA/object/sensor', data)
        if not result:
            print "add sensor failed!"
            return None
        else:
            return ServiceSensor(self._agent_id, self._snapshot_id, sensor_id, self._service)

    def init_with_sensors(self, sensors):
        for sensor in sensors:
            self.add_sensor(sensor[0], sensor[1])

    def init(self):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/object/snapshot/init', data)
        if not result:
            return None
        return result

    def make_up(self, signal):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signal': signal}
        result = self._service.post('/UMA/matrix/up', data)
        if not result:
            return None
        return list(result['data']['signal'])

    def make_abduction(self, signals):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signals}
        result = self._service.post('/UMA/matrix/abduction', data)
        if not result:
            return None
        return list(result['data']['abduction_even']), list(result['data']['abduction_odd'])

    def make_propagate_masks(self):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/matrix/propagate_masks', data)
        if not result:
            return None
        return list(result['data']['propagate_mask'])

    def make_ups(self, signals):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signals}
        result = self._service.post('/UMA/matrix/ups', data)
        if not result:
            return None
        return list(result['data']['signals'])

    def make_propagation(self, signals, load):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signals, 'load': load}
        result = self._service.post('/UMA/matrix/propagation', data)
        if not result:
            return None
        return list(result['data']['signals'])

    def make_blocks(self, dists, delta):
        data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'dists': dists, 'delta': delta}
        result = self._service.post('/UMA/matrix/blocks', data)
        if not result:
            return None
        return list(result['data']['blocks'])

    def make_npdirs(self):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/matrix/npdirs', data)
        if not result:
            return None
        return list(result['data']['npdirs'])

    def add_implication(self, from_sensor, to_sensor):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'sensors': [from_sensor, to_sensor]}
        result = self._service.post('/UMA/object/snapshot/implication', data)
        if not result:
            False
        return True

    #def make_decision(self, signals, phi, active):
    #    data =  {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'phi': phi, 'active': active, 'signals': signals}
    #    #print signals
    #    result = self._service.post('/UMA/simulation/decision', data)
    #    if not result:
    #        return None
    #    result = result['data']
    #    return float(result['res']), result['current'], result['prediction'], result['target']

    def delay(self, delay_list, uuid_list):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'delay_list': delay_list, 'uuid_list': uuid_list}
        #data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'delay_list': [signal._VAL.tolist() for signal in delay_list], 'uuid_list': uuid_list}
        result = self._service.post('/UMA/simulation/delay', data)
        if not result:
            return False
        return True

    def pruning(self, signal):
        data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signal}
        #data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signal._VAL.tolist()}
        result = self._service.post('/UMA/simulation/pruning', data)
        if not result:
            return False
        return True

    def getCurrent(self):
        return self._service.get('/UMA/data/current', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getPrediction(self):
        return self._service.get('/UMA/data/prediction', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getTarget(self):
        return self._service.get('/UMA/data/target', {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setQ(self, q):
        return self._service.put('/UMA/object/snapshot', {'q': q}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setThreshold(self, threshold):
        return self._service.put('/UMA/object/snapshot', {'threshold': threshold}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setTarget(self, target_list):
        return self._service.put('/UMA/data/target', {'target_list': target_list}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setAutoTarget(self, auto_target):
        return self._service.put('/UMA/object/snapshot', {'auto_target': auto_target}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setPropagateMask(self, propagate_mask):
        return self._service.put('/UMA/object/snapshot', {'propagate_mask': propagate_mask}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def setInitialSize(self, initial_size):
        return self._service.put('/UMA/object/snapshot', {'initial_size': initial_size}, {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def get_sensor_count(self):
        return

    def get_sensor_pair_count(self):
        return