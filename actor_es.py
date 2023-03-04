import os
import yaml
from os.path import join, dirname, abspath, exists
import torch
import gym
import numpy as np
import random
import time
import rospy
import argparse
import csv

from envs import registration
from envs.wrappers import StackFrame

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
        y = x.unsqueeze(0)
        y = y.permute(0,2,1)
        # x = x.permute(1, 0) # [state_dim, seq_len]
        a = self.net(y)
        return a

BUFFER_PATH = os.getenv('BUFFER_PATH')
if not BUFFER_PATH:
    BUFFER_PATH = "./local_buffer"

# add path to the plugins to the GAZEBO_PLUGIN_PATH
gpp = os.getenv('GAZEBO_PLUGIN_PATH') if os.getenv('GAZEBO_PLUGIN_PATH') is not None else ""
wd = os.getcwd()
os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(wd, "jackal_helper/plugins/build") + ":" + gpp
rospy.logwarn(os.environ['GAZEBO_PLUGIN_PATH'])

def initialize_actor(id):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    # try
    #     # assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    # except
    #     raise Exception(f'COULD NOT FIND {BUFFER_PATH} in {os.getcwd()}')
    f = None
    c = 0
    while f is None and c < 10:
        c += 1
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_policy(id):
    file_path = f'/local_buffer/policy_{id}.pth'
    n_input = 724 #state dim
    n_output = 2 #action dim
    policy = Policy(n_input, n_output).to('cpu')
    policy.load_state_dict(torch.load(file_path))
    return policy

def write_buffer(total_reward, bc, id):
    file_path = f'/local_buffer/actor_{id}.csv'
    
    with open(file_path, 'w', newline='') as csvfile: 
        datawriter = csv.writer(csvfile)
        datawriter.writerow([total_reward, bc[0], bc[1]])

    return 

def get_world_name(config, id):
    if len(config["container_config"]["worlds"]) < config["container_config"]["num_actor"]:
        duplicate_time = config["container_config"]["num_actor"] // len(config["container_config"]["worlds"]) + 1
        worlds = config["container_config"]["worlds"] * duplicate_time
    else:  # if num_actors < num_worlds, then each actor will rollout in a random world
        worlds = config["container_config"]["worlds"].copy()
        random.shuffle(worlds)
        worlds = worlds[:config["container_config"]["num_actor"]]
    world_name = worlds[id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name

def _debug_print_robot_status(env, count, rew, actions):
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), Y position: %f(world_frame), rew: %f' %(count, p.x, p.y, rew))

def main(args):
    id = args.id
    config = initialize_actor(id)
    env_config = config['env_config']
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    num_ep = 0
    total_reward =0
    done = False

    policy = load_policy(id)

    observation = env.reset() #probably don't need the reset but need initial obs
    with torch.no_grad():
        while not done:
            observation = (torch.from_numpy(observation).float().to('cpu'))
            action = (policy(observation).data.detach().cpu().numpy())
            observation, reward, done, info = env.step(action)
            total_reward += reward
    
    #I wanted bc to be final positon (x,y), but that is not in jackal_gazebo_envs by default
    #so getting it in the container is a bit of a pain
    #but goal_position is world_frame_goal - (x,y), so it includes that information and should be good enough
    bc = info['goal_position']
    write_buffer(total_reward, bc, id)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>> actor_id: %d, world_idx: %s, num_episode: %d" %(id, world_name, num_ep))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='id', type = int, default = 1)

    args = parser.parse_args()
    main(args)
