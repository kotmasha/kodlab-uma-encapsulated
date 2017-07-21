from UMA_service import *
from Service_Snapshot import *

class ServiceAgent:
    def __init__(self, agent_id, service):
        self._agent_id = agent_id
        self._service = service

    def add_snapshot(self, snapshot_id):
        data = {'uuid': snapshot_id, 'agent_id': self._agent_id}
        result = self._service.post('/UMA/object/snapshot', data)
        if not result:
            print "add snapshot failed!"
            return None
        else:
            return ServiceSnapshot(self._agent_id, snapshot_id, self._service)

    def make_decision(self, signals, phi, active):
        #translate observation signals to a pair of Boolean arrays:
        obs_plus=signals['plus']._VAL.tolist()
        obs_minus=signals['minus']._VAL.tolist()
        #post to service:
        data =  {'agent_id': self._agent_id, 'phi': phi, 'active': active, 'obs_plus': obs_plus, 'obs_minus': obs_minus}
        result = self._service.post('/UMA/simulation/decision', data)
        if not result:
            return None
        result = result['data']
        plus = {'res': float(result['res_plus']), 'current': result['current_plus'], 'prediction': result['prediction_plus'], 'target': result['target_plus']}
        minus = {'res': float(result['res_minus']), 'current': result['current_minus'], 'prediction': result['prediction_minus'], 'target': result['target_minus']}
        return {'plus': plus, 'minus': minus}

    def get_snapshot_count(self):
        return