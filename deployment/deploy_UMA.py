import yaml
import shutil
import errno
from ConfigParser import SafeConfigParser
from cluster_setting import *

# copy the folder from 1 to another
def copy_folder(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

# update the server ini file, including changing the url, and port
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
        instance = info['Cluster']['Ninstances']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])
        print 'Will deploy %d UMA instances' % instance
        print 'base_url read from conf is: %s' % base_url
        print 'port read from conf is: %s' % port

    cluster_path = os.path.join(UMA_HOME, 'deployment', 'cluster')
    try:
        shutil.rmtree(cluster_path)
    except Exception as ex:
        print "Error when trying to remove any existing cluster, error: %s" % str(ex)
    print "removed old UMA cluster successfully"

    for i in range(instance):
        src = os.path.join(UMA_HOME, 'bin')
        dst = os.path.join(UMA_HOME, 'deployment', 'cluster', 'UMA' + str(i))
        copy_folder(src, dst)

        try:
            ini_path = os.path.join(dst, 'ini', 'server.ini')
            update_server_ini(ini_path, base_url, str(port + i))
            print "%dth UMA copied" % i
        except Exception as ex:
            print "Error when trying to updating the server.ini for %dth UMA deployment, error: %s" % (i, str(ex))