
import torch
import torch.nn as nn
import numpy as np
import constants
"""
Agent will observe the state
Agent will 

Actions:
   0: utterance
   1: pointing
   2: guiding

observation:
    current location of agent (absolute location) (x, y)
    goal location (absolute location) (x,y)

    input to FC --> [current location, goal location]
    output of FC -->[probaility distribution over action space]
    
    2 layers with 256 hidden units  
"""
class Agent(nn.Module):
    def __init__(self,input_size,action_space_size,cur_loc, graph):
        super(Agent, self).__init__()
        self.cur_loc = cur_loc
        self.graph = graph

        self.FC = nn.Sequential(
                nn.Linear(input_size,256),
                nn.ReLU(),
                nn.Linear(256,256),
                nn.ReLU(),
                nn.Linear(256,action_space_size),
                nn.Softmax()

        )
    def forward(self, goal_location):
        # print(goal_location)
        x = self.FC(goal_location)
        return x
        # print(x)
    def get_reward(self, action, target_loc):
        if action == constants.ACTION_GUIDE:
            dist = np.linalg.norm(self.cur_loc-target_loc)
            reward = -10* dist
        if action == constants.ACTION_POINT:
            reward = -5
        if action == constants.ACTION_UTTER:
            # now we perform the utterance and listener has to interpr
            # or a single neural network taking two inputs as one
            pass 


    def communicate(self, listener):
        
        nonsrc_indices = np.where(self.graph.locations != self.cur_loc)[0]

        target = np.random.choice(nonsrc_indices)
        target_loc = self.graph.locations[target]
        quad, segment = self.graph.quadrant_circle_pair(target_loc, self.cur_loc)
        x = torch.cat(quad, segment)
        action_prob = self.forward(x)
        action = np.random.choice(constants.ACTION_SPACE, p = action_prob)
