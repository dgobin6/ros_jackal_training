import os
from os.path import join
from multiprocessing import Pool
from spython.main import Client as client

def run_actor_in_container(id=0):
    BUFFER_PATH = "./local_buffer"
    out = client.execute(
        join("./local_buffer", "nav_benchmark.sif"),
        ['/bin/bash', '/jackal_ws/src/ros_jackal_training/entrypoint.sh', 'python3', 'actor_es.py', '--id=%d' %id],
        bind=['%s:%s' %(BUFFER_PATH, '/local_buffer'), '%s:%s' %(os.getcwd(), "/jackal_ws/src/ros_jackal_training")],
        options=["-i", "-n", "--network=none", "-p"], nv=True
    )
    return out

def collect(n_worlds, ids): 
    with Pool(n_worlds) as p:
        output = p.map(run_actor_in_container, ids)
        p.close()
        p.join()
    for o in output:
        print(o['message'])