import os
import yaml
import sys
import shutil
from ConfigParser import SafeConfigParser
import shutil
import errno

DEPLOY_YML = 'deploy.yml'
UMA_HOME = os.path.dirname(os.getcwd())

def copy_folder(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def update_server_ini(ini_path, base_url, port):
    parser = SafeConfigParser()
    parser.read(ini_path)
    parser.set('Server', 'port', port)
    parser.set('Server', 'host', base_url)

    with open(ini_path, 'w') as f:
        parser.write(f)

if __name__ == "__main__":
    with open(os.path.join(UMA_HOME, 'deployment', DEPLOY_YML), 'r') as f:
        info = yaml.load(f)
        instance = info['Cluster']['instance']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])
        print 'Will deploy %d UMA instances' % instance
        print 'base_url read from conf is: %s' % base_url
        print 'port read from conf is: %s' % port

    for i in range(instance):
        src = os.path.join(UMA_HOME, 'bin')
        dst = os.path.join(UMA_HOME, 'deployment', 'cluster', 'UMA' + str(i))
        copy_folder(src, dst)

        try:
            ini_path = os.path.join(dst, 'ini', 'server.ini')
            update_server_ini(ini_path, base_url, str(port + i))
        except Exception as ex:
            print "Error when trying to updating the server.ini for %dth UMA deployment, error: %s" % (i, str(ex))

        print "%dth UMA copied" % i