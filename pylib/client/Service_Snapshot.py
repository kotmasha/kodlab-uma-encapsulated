from UMA_service import *
from Service_Sensor import *

class ServiceSnapshot:
    def __init__(self, experiment_id, agent_id, snapshot_id, service):
        self._experiment_id = experiment_id
        self._agent_id = agent_id
        self._snapshot_id = snapshot_id
        self._service = service

    def add_sensor(self, sensor_id, c_sensor_id):
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                'sensor_id': sensor_id, 'c_sid': c_sensor_id, 'w': [], 'd': [], 'diag': []}
        result =  self._service.post('/UMA/object/sensor', data)
        if not result:
            print "add sensor=%s failed!" % sensor_id
            return None
        else:
            return ServiceSensor(self._agent_id, self._snapshot_id, sensor_id, self._service)

    def init_with_sensors(self, sensors):
        for sensor in sensors:
            self.add_sensor(sensor[0], sensor[1])

    def init(self):
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/object/snapshot/init', data)
        if not result:
            return None
        return result

    def make_up(self, signal):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                 'signal': signal}
        result = self._service.post('/UMA/simulation/up', data)
        if not result:
            return None
        return list(result['data']['signal'])

    def make_abduction(self, signals):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                 'signals': signals}
        result = self._service.post('/UMA/simulation/abduction', data)
        if not result:
            return None
        return list(result['data']['abduction_even']), list(result['data']['abduction_odd'])

    def make_propagate_masks(self):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/simulation/propagateMasks', data)
        if not result:
            return None
        return list(result['data']['propagate_mask'])

    def make_ups(self, signals):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id,
                 'snapshot_id': self._snapshot_id, 'signals': signals}
        result = self._service.post('/UMA/simulation/ups', data)
        if not result:
            return None
        return list(result['data']['signals'])

    def make_downs(self, signals):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                 'signals': signals}
        result = self._service.post('/UMA/simulation/downs', data)
        if not result:
            return None
        return list(result['data']['signals'])

    def make_propagation(self, signals, load):
        data =  {'experimentId': self._experiment_id, 'agentId': self._agent_id, 'snapshotId': self._snapshot_id,
                 'signals': signals, 'load': load}
        result = self._service.post('/UMA/simulation/propagation', data)
        if not result:
            return None
        return list(result['data']['signals'])

    def make_blocks(self, dists, delta):
        data =  {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                 'dists': dists, 'delta': delta}
        result = self._service.post('/UMA/simulation/blocks', data)
        if not result:
            return None
        return list(result['data']['blocks'])

    def make_npdirs(self):
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id}
        result = self._service.post('/UMA/simulation/npdirs', data)
        if not result:
            return None
        return list(result['data']['npdirs'])

    def add_implication(self, from_sensor, to_sensor):
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                'sensors': [from_sensor, to_sensor]}
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
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                'delay_lists': delay_list, 'uuid_lists': uuid_list}
        #data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'delay_list': [signal._VAL.tolist() for signal in delay_list], 'uuid_list': uuid_list}
        result = self._service.post('/UMA/object/snapshot/delay', data)
        if not result:
            return False
        return True

    def pruning(self, signal):
        data = {'experiment_id': self._experiment_id, 'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,
                'signals': signal}
        #data = {'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'signals': signal._VAL.tolist()}
        result = self._service.post('/UMA/object/snapshot/pruning', data)
        if not result:
            return False
        return True

    def getCurrent(self):
        return self._service.get('/UMA/data/current', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getPrediction(self):
        return self._service.get('/UMA/data/prediction', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getTarget(self):
        return self._service.get('/UMA/data/target', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})

    def getNegligible(self):
        return self._service.get('/UMA/data/negligible', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id})['data']['negligible']

    def setQ(self, q):
        return self._service.put('/UMA/object/snapshot', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id,  'q': q,})

    def setThreshold(self, threshold):
        return self._service.put('/UMA/object/snapshot', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'threshold': threshold})

    def setTarget(self, target):
        return self._service.put('/UMA/data/target',{'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'target': target})

    def setAutoTarget(self, auto_target):
        return self._service.put('/UMA/object/snapshot', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'auto_target': auto_target})

    def setPropagateMask(self, propagate_mask):
        return self._service.put('/UMA/object/snapshot', {'experimentId': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'propagate_mask': propagate_mask})

    def setInitialSize(self, initial_size):
        return self._service.put('/UMA/object/snapshot', {'experiment_id': self._experiment_id,
                        'agent_id': self._agent_id, 'snapshot_id': self._snapshot_id, 'initial_size': initial_size})