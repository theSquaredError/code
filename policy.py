
import imp
import torch
import torch.nn as nn
import numpy as np
import constants
import agent_network
from graph_world import World
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
class Policy(nn.Module):
    def __init__(self,input_size,action_space_size, graph):
        super(Policy, self).__init__()
        self.graph = graph
        self.X_, self.Y_, self.conceptNet, self.vocabNet = agent_network.initialise()
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
    def get_reward(self, action, target_loc, cur_loc):
        if action == constants.ACTION_GUIDE:
            dist = np.linalg.norm(cur_loc-target_loc)
            reward = -10* dist
        if action == constants.ACTION_POINT:
            reward = -5
        if action == constants.ACTION_UTTER:
            # now we perform the utterance and listener has to interpret
            # or a single neural network taking two inputs as one
            
            octant, segment = self.graph.quadrant_circle_pair(target_loc, cur_loc)
            # finding vocab for octant and segment
            input_oct = torch.zeros(28)
            input_oct[self.X_.index(octant)] = 1

            input_seg = torch.zeros(28)
            # print(f"X_={self.X_}")
            # print(f"segment={segment}")
            input_seg[self.X_.index(segment)] = 1
            

            # This has to be done 4 times
            oct_vocab_probs,seg_vocab_probs = self.vocabNet(input_oct), self.vocabNet(input_seg)

            # above 2 contains probabilities for each word 
            # we will sample the vocab and send to listener

            # sampling vocabs
            oct_utter = np.random.choice(
                self.Y_, p=oct_vocab_probs.detach().numpy())
            seg_utter = np.random.choice(
                self.Y_, p=seg_vocab_probs.detach().numpy())
            
            vocab_oct = torch.zeros(28)
            vocab_oct[self.Y_.index(oct_utter)]

            vocab_seg = torch.zeros(28)
            vocab_seg[self.Y_.index(seg_utter)]

            # print(f"oct_utter={oct_utter}")
            ####### Listener ############
            oct_probs,seg_probs = self.conceptNet(vocab_oct), self.conceptNet(vocab_seg)

            pred_oct = np.random.choice(self.X_, p = oct_probs)
            pred_seg = np.random.choice(self.X_, p = seg_probs)

            if pred_oct == octant and pred_seg == segment:
                # Listener understood correctly
                reward = 10  # some higher reward
            else:
                reward = 0

             

        return reward


    def communicate(self, src_loc):
        
        nonsrc_indices = np.where(self.graph.locations != src_loc)[0]

        target = np.random.choice(nonsrc_indices)
        target_loc = self.graph.locations[target]
        quad, segment = World.quadrant_circle_pair(target_loc, src_loc)
        # quad, segment = torch.tensor(quad), torch.tensor(segment)
        x = torch.tensor([quad, segment]).float()
        # x = torch.cat((quad, segment))
        action_prob = self.forward(x)
        print(f"action_prob={action_prob}")
        action = np.random.choice(constants.ACTION_SPACE, p = action_prob.detach().numpy())
        log_prob = torch.log(action_prob)[action]
        return action, log_prob, target_loc
