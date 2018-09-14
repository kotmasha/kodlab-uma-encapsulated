import psutil
import yaml
from cluster_setting import *

def kill_process_pid(pid_path):
    try:
        with open(pid_path, 'r') as f:
            pid = int(f.read())

    except Exception as ex:
        print "Error when trying open file=%s, error=%s" % (pid_path, str(ex))

    try:
        p = psutil.Process(pid)
    except Exception as ex:
        print "cannot find the process by pid=%d" % pid
        return

    try:
        p.terminate()
        print "pid=%d is terminated successfully" % pid
    except Exception as ex:
        print "cannot terminate process pid=%d, error=%s" % (pid, str(ex))

if __name__ == "__main__":
    with open(os.path.join(UMA_HOME, 'deployment', DEPLOY_YML), 'r') as f:
        info = yaml.load(f)
        instance = info['Cluster']['Ninstances']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])

    cluster_path = os.path.join(UMA_HOME, 'deployment', 'cluster')

    for i in range(instance):
        path = os.path.join(cluster_path, 'UMA' + str(i), 'uma_pid.txt')
        if not os.path.exists(path):
            raise Exception("Cannot find the UMA path=%s" %path)

        kill_process_pid(path)