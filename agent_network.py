from asyncio import constants
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from communication import Communication
import constants
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
'''
Both of the agent will learn their own mapping because each agent 
is a separate neural network
'''


'''
Neural network architecture:
    Input Layer: 28 [one_hot encoding]
    Hidden Layer1: 10
    Output Layer : size of vocabulary = 28 [one_hot encoding]
'''
class MapNet(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(MapNet,self).__init__()

        self.L1 = nn.Linear(input_size,10)
        self.L2 = nn.Linear(10,output_size)

    def forward(self, input):
        # print(input)
        x = self.L1(input)
        # print(x)
        x = F.relu(x)
        x = self.L2(x)
        # print(x)
        x = F.softmax(x, dim=0)
        return x




def print_loss(losses, learning_rate = 0.01):
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

def train_agent(net, X,Y, learning_rate = 0.01):
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(10):
        pred_y = net(X)
        loss = loss_fn(pred_y, Y)
        losses.append(loss.item())
        
        net.zero_grad()
        loss.backward()
        optimiser.step()
    print_loss(losses)

def one_hot_encoded(data):
    dim = len(data)
    temp = np.eye(dim)
    return temp
    

if __name__ == '__main__':
    # Getting the mappings
    vocab_map = Communication.generate_vocabulary(constants.n_octants,constants.n_segments)
    # vocab_map = np.array(vocab_map)
    X_ = [i[0] for i in vocab_map]
    Y_ = [i[1] for i in vocab_map]
    
    # Creating one hot encoding of each of the vector
    X = torch.tensor(one_hot_encoded(X_), dtype=torch.float)
    Y = torch.tensor(one_hot_encoded(Y_), dtype=torch.float)
    

    # VocabNet tries to learn the mapping from concept to vocab
    # input size = output size = 8+20  = constants.n_octants+ constants.n_segments
    vocabNet = MapNet(28,28)
    # print(vocabNet)

    train_agent(vocabNet,X,Y)


    # train_agent(vocabNet,)
    # ConceptNet = MapNet()
