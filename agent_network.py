from asyncio import constants
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from communication import Communication
import constants
'''
Both of the agent will learn their own mapping because each agent 
is a separate neural network
'''


'''
Neural network architecture:
    Input Layer: [octant,segment]
    Hidden Layer1: 5
    Hidden Layer2: 5
    Output Layer : size of vocabulary = num of octant * num of segment
'''
class MapNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MapNet,self).__init__()

        self.L1 = nn.Linear(input_size,10)
        self.L2 = nn.Linear(10,output_size)

    def forward(self, input):
        print(input)
        x = self.L1(input)
        print(x)
        x = F.relu(x)
        x = self.L2(x)
        print(x)
        x = F.softmax(x, dim=0)
        return x


if __name__ == '__main__':
    # Getting the mappings
    vocab_map = Communication.generate_vocabulary(constants.n_octants,constants.n_segments)
    # vocab_map = np.array(vocab_map)
    X = [i[0] for i in vocab_map]
    Y = [i[1] for i in vocab_map]
    # print(vocab_map)
    print(X)
    # print(len(vocab_map))
    # VocabNet = MapNet()
    # ConceptNet = MapNet()