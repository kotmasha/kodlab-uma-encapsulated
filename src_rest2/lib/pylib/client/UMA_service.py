import requests
import json

class UMA_service:
    def __init__(self, scheme = 'http', ip = 'localhost', port = '8000'):
        self._url = scheme + '://' + ip + ':' + port
        self._headers = {'Content-type': 'application/json', 'Accpet': 'text/plain'}
        self._log = open('./client.txt', 'w')

    def post(self, uri, data):
        uri = self._url + uri
        try:
            r = requests.post(uri, data = json.dumps(data), headers = self._headers)
        except Exception, e:
            self._log.write("Errors while doing post request " + uri + ': ' + str(e) + '\n')
            return None
        if r.status_code >= 400 and r.status_code < 500:
            self._log.write("Client Error(" + str(r.status_code) + "): " + str(r.json()['message'] + '\n'))
            return None
        if r.status_code >= 500:
            self._log.write("Server Error(" + str(r.status_code) + ") please check server log" + '\n')
            return None
        self._log.write("(" + str(r.status_code) + ") " + str(r.json()['message']) + '\n')
        return r.json()

    def get(self, uri, query):
        uri = self._url + uri
        retry = 0
        while retry < 5:
            try:
                r = requests.get(uri, params = query, headers = self._headers)
                break
            except:
                self._log.write("Errors while doing get request " + uri + '\n')
                retry += 1
                if retry == 5:
                    return None
                
        if r.status_code >= 400 and r.status_code < 500:
            self._log.write("Client Error(" + str(r.status_code) + "): " + str(r.json()['message'] + '\n'))
            return None
        if r.status_code >= 500:
            self._log.write("Server Error(" + str(r.status_code) + ") please check server log" + '\n')
            return None
        self._log.write("(" + str(r.status_code) + ") " + str(r.json()['message']) + '\n')
        return r.json()

    def put(self, uri, data, query):
        uri = self._url + uri
        try:
            r = requests.put(uri, data = json.dumps(data), params=query, headers = self._headers)
        except:
            self._log.write("Errors while doing put request " + uri + '\n')
            return None
        if r.status_code >= 400 and r.status_code < 500:
            self._log.write("Client Error(" + str(r.status_code) + "): " + str(r.json()['message'] + '\n'))
            return None
        if r.status_code >= 500:
            self._log.write("Server Error(" + str(r.status_code) + ") please check server log" + '\n')
            return None
        self._log.write("(" + str(r.status_code) + ") " + str(r.json()['message']) + '\n')
        return r.json()

    def delete(self):
        pass
