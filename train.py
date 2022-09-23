'''
TODO: Initialise the graph
TODO: Learn the mapping for speaker and listener module 
TODO: 
'''

from collections import deque
from pickletools import optimize
import numpy as np
import torch.optim as optim
import torch
import os

from agent_network import MapNet
import agent_network

from graph_world import World
from policy import Policy

# TODO :getting the state and passing to the policy
# TODO : In case of the UTTER Action sample it 4 times
# TODO : applying policy gradient 


# First round of communication : 
    ## Speaker and listener will be at a location in beginning  
    ## Speaker will get a target location
    ## Action has to be chosen from the ACTION SPACE
    ## Get the reward 
    ## Policy Gradient has to be applied to learn the policy


GAMMA = 0.9

def reinfoce(n_episodes=1000, max_t = 1000, gamma = 1.0, print_every = 100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        pass


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []
    timesteps = len(rewards)
    for t in range(timesteps):
        Gt = 0 #discounted from timestep t
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw*r
            pw = pw+1
        discounted_rewards.append(Gt)
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards-discounted_rewards.mean())/(discounted_rewards.std() + 1e-9)
    
    policy_gradient = []

    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob*Gt)
    
    optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    optimizer.step()
    
    



if __name__ == '__main__':
    os.system('cls')
    # Create a environment with N vertices
    N_vertices = 10
    world = World(N_vertices)
    # choose a source location
    src_loc = world.locations[np.random.choice(N_vertices)]
    
    policy = Policy(2, 3, world)
    optimizer = optim.Adam(policy.parameters(), lr = 1e-2)

    max_episode_num = 10
    max_steps = 100
    num_steps = []
    avg_numsteps =[]
    all_rewards = []

    for episode in range(max_episode_num):
        log_probs=[]
        rewards = []
        for steps in range(max_steps):
            state = world.locations[np.random.choice(N_vertices)]
            action, log_prob, target_loc = policy.communicate(state)
            reward = policy.get_reward(action,target_loc, state)
            log_probs.append(log_prob)
            rewards.append(reward)

        # update policy
        print(f"reward sum = {np.sum(rewards)}")
        update_policy(policy, rewards, log_probs)
