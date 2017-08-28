#This is the wrapper file for the unit test
import requests
import json

class test_wrapper:

    def __init__(self, scheme = 'http', ip = 'localhost', port = '8000'):
        self._url = scheme + '://' + ip + ':' + port
        self._headers = {'Content-type': 'application/json', 'Accpet': 'text/plain'}
        self._log = open('./client.txt', 'w')

    def T(self, method, url, status_code, data=None, query=None, message=None, kwargs=None):
        url = self._url + url
        if method is "POST":
            response = requests.post(url, json.dumps(data), headers = self._headers)
            if not response.status_code == status_code:
                print response.status_code, status_code
                assert response.status_code == status_code
            if message is not None:
                if message not in response.json()['message']:
                    print message, response.json()['message']
                    assert message in response.json()['message']
        elif method is "GET":
            response = requests.get(url, params = query, headers = self._headers)
            if not response.status_code == status_code:
                print response.status_code, status_code
                assert response.status_code == status_code
            if message is not None:
                if message not in response.json()['message']:
                    print message, response.json()['message']
                    assert message in response.json()['message']
            if kwargs is not None:
                for key, value in kwargs.iteritems():
                    if 'data' not in response.json() or key not in response.json()['data']:
                        print "No %s fields found in return value" % key
                        assert False
                    if not response.json()['data'][key] == value:
                        if type(value) is float and type(response.json()['data'][key]) is float:
                            if abs(response.json()['data'][key] - value) < 1e-6:
                                continue
                        print value, response.json()['data'][key]
                        assert False
        elif method is "DELETE":
            response = requests.delete(url, data=json.dumps(data), headers = self._headers)
            if not response.status_code == status_code:
                print response.status_code, status_code
                assert response.status_code == status_code
            if message is not None:
                if message not in response.json()['message']:
                    print message, response.json()['message']
                    assert message in response.json()['message']
        elif method is "PUT":
            response = requests.put(url, data = json.dumps(data), params=query, headers = self._headers)
            if not response.status_code == status_code:
                print response.status_code, status_code
                assert response.status_code == status_code
            if message is not None:
                if message not in response.json()['message']:
                    print message, response.json()['message']
                    assert message in response.json()['message']

    def run(self, test_file):
        pass