from UMA_service import *
from Service_Agent import *

class ServiceWorld:
    def __init__(self, service):
        self._service = service

    def add_agent(self, uuid):
        data = {'uuid': uuid}
        result = self._service.post('/UMA/object/agent', data)
        if not result:
            print "add agent failed!"
            return None
        else:
            return ServiceAgent(uuid, self._service)

    def save(self, filename, dicts, id_list):
        data = {'filename': filename}
        result = self._service.post('/UMA/simulation/saving', data)
        if not result:
            print "data saving failed!"
            return None
        else:
            f = open(filename + '.txt', 'w+')
            for id in id_list:
                f.write(id + ': ' + dicts[id] + '\n')
            f.close()
            return result

    def load(self, filename):
        data = {'filename': filename}
        result = self._service.post('/UMA/simulation/loading', data)
        if not result:
            print "data loading failed!"
            return None
        else:
            dicts = {}
            f = open(filename + '.txt', 'r')
            for line in f:
                kv_pair = line.split(':')
                dicts[kv_pair[0].strip()] =kv_pair[1].strip()
            f.close()
            return dicts

    def get_agent_count(self):
        return