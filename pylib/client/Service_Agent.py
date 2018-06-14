from UMA_service import *
from Service_Snapshot import *

class ServiceAgent:
    def __init__(self, experiment_id, agent_id, service):
        self._experiment_id = experiment_id
        self._agent_id = agent_id
        self._service = service

    def add_snapshot(self, snapshot_id):
        data = {'snapshotId': snapshot_id, 'agentId': self._agent_id, 'experimentId': self._experiment_id}
        result = self._service.post('/UMA/object/snapshot', data)
        if not result:
            print "create snapshot=%s failed!" % snapshot_id
            return None
        else:
            return ServiceSnapshot(self._agent_id, snapshot_id, self._service)

    def make_decision(self, obs_plus, obs_minus, phi, active):
        #post to service:
        data =  {'experimentId': self._experiment_id, 'agentId': self._agent_id, 'phi': phi,
                 'active': active, 'obsPlus': obs_plus, 'obsMinus': obs_minus}
        result = self._service.post('/UMA/simulation/decision', data)
        if not result:
            return None
        result = result['data']
        plus = {'res': float(result['resPlus']), 'current': result['currentPlus'],
                'prediction': result['predictionPlus'], 'target': result['targetPlus']}
        minus = {'res': float(result['resMinus']), 'current': result['currentMinus'],
                 'prediction': result['predictionMinus'], 'target': result['targetMinus']}
        return {'plus': plus, 'minus': minus}