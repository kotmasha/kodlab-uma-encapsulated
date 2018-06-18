from UMA_service import *
from Service_Experiment import *

class ServiceWorld:
    def __init__(self, service):
        self._service = service

    def add_experiment(self, experiment_id):
        data = {'experiment_id': experiment_id}
        result = self._service.post('/UMA/object/experiment', data)
        if not result:
            print "create experiment=%s failed!" % experiment_id
            return None
        else:
            return ServiceExperiment(experiment_id, self._service)
