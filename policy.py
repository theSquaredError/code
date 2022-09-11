from os import nice
import re
from resource import RLIM_INFINITY
from tkinter import Variable
from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions = 18):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 1, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(49*37, 32)
        self.fc2 = nn.Linear(32, self.num_actions)
        # self.optimiser = optim.Adam(self.parameters(), lr = learning_rate)
    
    def forward(self, state):
        x = torch.tensor(state)
        x = x.permute(2,0,1)
        # print(x.shape)
        x = x.reshape([1, 3, 210, 160])
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.pool1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)

        x = x.view(49*37)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.softmax(self.fc2(x))
        return x
    
    def get_action(self, state):
        probs = self.forward(state)
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob





'''
if __name__ == '__main__':
    state = torch.randn(210,160,3)
    net = PolicyNetwork()
    net(state)

'''

