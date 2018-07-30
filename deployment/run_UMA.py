import os
import subprocess
import yaml
from cluster_setting import *

# save the pid to file, just in case process cannot be stopped normally.
def save_UMA_pid(path):
    os.chdir(path)
    p = subprocess.Popen("UMA.exe", shell=False)
    with open('uma_pid.txt', 'w') as f:
        try:
            f.write(str(p.pid))
            print "UMA process pid=%d is saved to file" % p.pid
        except Exception as ex:
            print "Error when saving pid=%d to file, error is: %s" % (p.pid, str(ex))
            try:
                p.kill()
            except:
                print "Failed to kill the current process, pid=%d" % p.pid


if __name__ == "__main__":
    with open(os.path.join(UMA_HOME, 'deployment', DEPLOY_YML), 'r') as f:
        info = yaml.load(f)
        instance = info['Cluster']['instance']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])

    cluster_path = os.path.join(UMA_HOME, 'deployment', 'cluster')

    for i in range(instance):
        path = os.path.join(cluster_path, 'UMA' + str(i))
        if not os.path.exists(path):
            raise Exception("Cannot find the UMA path=%s" %path)

        save_UMA_pid(path)