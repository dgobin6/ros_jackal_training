#Modified training script for NSRA
#Taken examples from: https://github.com/goktug97/estorch/blob/master/examples/nsra_es.py
import argparse
import GPUtil
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil
import logging
import collections
import time
import uuid
import random
from pprint import pformat
import csv 

import torch
from estorch import NSRA_ES
from tensorboardX import SummaryWriter

from envs import registration
from envs.wrappers import StackFrame

# from multiprocessing import Pool, set_start_method, get_context
from parallelbar import progress_map
from spython.main import Client as client

def run_actor_in_container(id=0):
    BUFFER_PATH = "./local_buffer"
    out = client.execute(
        join(BUFFER_PATH, "nav_benchmark.sif"),
        ['/bin/bash', '/jackal_ws/src/ros_jackal/entrypoint.sh', 'python3', 'actor_es.py', '--id=%d' %id],
        bind=['%s:%s' %(BUFFER_PATH, '/local_buffer'), '%s:%s' %(os.getcwd(), "/jackal_ws/src/ros_jackal")],
        options=["-i", "-n", "--network=none", "-p"], nv=True
    )
    return out

class Agent(): 
    def __init__(self, worlds, config, device=torch.device('cpu')):
        self.device = device
        self.config = config
        self.worlds = worlds
        self.n_worlds = 10
        self.save_starter = "./local_buffer/policy"
        self.ids = range(10)
        self.rollout_count = 0

        # These two env variable ensure ROS running correctly in the container
        os.environ["ROS_HOSTNAME"] = "localhost"
        os.environ["ROS_MASTER_URI"] = "http://localhost:11311"

        shutil.copyfile(
            config["env_config"]["config_path"],
            join('./local_buffer', "config.yaml")
        )
    
    #I wanted to evaluate across each world, but loading and restarting policies isn't supported in the library I found
    #and I am running low on time. I will evaluate across 10 worlds at a time and return average reward for training and
    #policy selection. I can try more worlds, but don't want it to take too long
    def rollout(self, policy):
        # with torch.no_grad(): 
        #     for world in select_worlds:
        #         env = initialize_envs(self.config, world)
        #         observation = env.reset() #probably don't need the reset but need initial obs
        #         while not done:
        #             observation = (torch.from_numpy(observation).float().to(self.device))
        #             action = (policy(observation).data.detach().cpu().numpy())
        #             observation, reward, done, info = env.step(action)
        #             total_reward += reward

        print("###########################ROLL OUT #################################")
        st = time.time()
        #save a dict for each actor to prevent lock outs
        for id in self.ids:
            save_path = f'{self.save_starter}_{id}.pth'
            torch.save(policy.state_dict(), save_path)
        bc_vec = []
        total_reward = 0.0

        with Pool(self.num_actor) as p:
            output = p.map(run_actor_in_container, self.ids)
        for o in output:
            print(o[0])

        # try:
        #     output = progress_map(run_actor_in_container, self.ids, process_timeout=600)
        # except:
        #     print(output)
        #     raise Exception("CONTAINER FAILED")

    
        for id in self.ids: 
            file_path = f'./local_buffer/actor_{id}.csv'
            with open(file_path, newline='') as csvfile: 
                data = list(csv.reader(csvfile))
                #there should be one row
                assert len(data)==1
                assert len(data[0])==3
                for row in data: 
                    total_reward += float(row[0])
                    bc_vec.append([float(row[1]), float(row[2])])
            os.remove(file_path) #no more file
        
        #updated jackal_gazebo_envs to provide the current position of the robot
        #behaviorial characteristic is the final (x,y) positions of the robot a la uber
        average_reward = total_reward/self.n_worlds
        bc = np.asarray(bc_vec).flatten()
        dt = time.time() - st
        print(f'avg_reward: {average_reward}')
        self.rollout_count += 1
        print(f'###########################ROLL OUT DONE: {dt}, {self.rollout_count} #################################')
        for id in self.ids:
            os.remove(f'{self.save_starter}_{id}.pth')
        return average_reward, bc


class Policy(torch.nn.Module): 
    def __init__(self, n_input, n_output): 
        super(Policy,self).__init__()
        layers = []
        layers.append(torch.nn.Conv1d(in_channels=n_input, out_channels=256, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, padding=0))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Flatten(start_dim=0))
        layers.append(torch.nn.Linear(256, 256))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(256, n_output))
        layers.append(torch.nn.ReLU())

        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # x [seq_len, state_dim]
        x = x.permute(1, 0) # [state_dim, seq_len]
        a = self.net(x)
        return a


def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    string = now.strftime("%Y_%m_%d_%H_%M")

    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        string,
        uuid.uuid4().hex[:4]
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer

def initialize_envs(config, world_name):
    env_config = config["env_config"]
    if env_config["use_container"]:
        env_config["kwargs"]["init_sim"] = False

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    return env

    return env

def seed(config):
    env_config = config["env_config"]
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])


if __name__ == "__main__":
    # set_start_method("spawn")
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description = 'Start training')
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    CONFIG_PATH = "./configs/e2e_default_SAC.yaml"
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    seed(config)
    t = time.time()
    env = initialize_envs(config, 0)
    print(f'Time to initialize environment: {time.time()-t}')

    worlds = config["container_config"]["worlds"].copy()
    random.shuffle(worlds)

    action_dim = np.prod(env.action_space.shape)
    state_dim = env.observation_space.shape
    input_dim = state_dim[-1]

    # print('###################################################')
    # print(input_dim)
    # print(action_dim)
    # print('###################################################')

    #train policy on the various worlds

    agent_kwargs = {"worlds": worlds, 'config': config}
    policy_kwargs = {'n_input':input_dim, 'n_output':action_dim}
    optimizer_kwargs = {'lr': .01}
    population_size = 256 #honestly not 100% sure what this is for
    meta_population_size = 5 #policy population
    sigma = .02
    weight_t = 20 #from uber research paper, wish I could set starting weight to .5 as they did
    weight_delta = .05
    min_weight = 0.0
    knn = 10

    print("BEGINNING TRAINING")
    es = NSRA_ES(Policy, Agent, torch.optim.Adam, population_size=population_size, sigma=sigma,
                 weight_t=weight_t, k=knn, meta_population_size=meta_population_size,
                 device='cpu', min_weight = min_weight, policy_kwargs=policy_kwargs,
                 agent_kwargs=agent_kwargs, optimizer_kwargs=optimizer_kwargs)
    
    es.train(n_steps = 100, n_proc=1)

    #save best policy
    save_file_path = "./save_stuff/nsra-es-best-policy_22723_1.pth"

    torch.save(es.best_policy_dict, save_file_path)



