import requests
import json

class UMA_service:
    def __init__(self, scheme = 'http', ip = 'localhost', port = '8000'):
        self._url = scheme + '://' + ip + ':' + port
        self._headers = {'Content-type': 'application/json', 'Accpet': 'text/plain'}

    def post(self, uri, data):
        uri = self._url + uri
        try:
            r = requests.post(uri, data = json.dumps(data), headers = self._headers)
        except:
            print "Errors while doing post request " + uri
            return False
        if r.status_code >= 400 and r.status_code < 500:
            print "Client Error(" + str(r.status_code) + "): " + str(r.json())
            return False
        if r.status_code >= 500:
            print "Server Error(" + str(r.status_code) + ") please check server log"
            return False
        print  "(" + str(r.status_code) + ") " + str(r.json())
        return True
        