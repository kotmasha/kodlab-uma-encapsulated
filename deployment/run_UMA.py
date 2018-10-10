import os
import subprocess
import yaml
import platform
from cluster_setting import *

# for now just assume running on either linux or windows
def is_running_on_windows():
    return 'Windows' == platform.system()

# save the pid to file, just in case process cannot be stopped normally.
def save_UMA_pid(path):
    os.chdir(path)
    binary_name = ["UMA.exe" if is_running_on_windows() else "./UMA", "start"]
    p = subprocess.Popen(binary_name, shell=False)
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
        instance = info['Cluster']['Ninstances']
        base_url = info['Cluster']['base_url']
        port = int(info['Cluster']['port'])
        if info['Cluster']['cluster_path'] is None or info['Cluster']['cluster_path'] == "None" or info['Cluster']['cluster_path'] == "":
            cluster_path = os.path.join(UMA_HOME, 'deployment', 'cluster')
        else:
            cluster_path = info['Cluster']['cluster_path']

    for i in range(instance):
        path = os.path.join(cluster_path, 'UMA' + str(i))
        if not os.path.exists(path):
            raise Exception("Cannot find the UMA path=%s" %path)

        save_UMA_pid(path)