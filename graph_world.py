"""

"""
from unicodedata import name
import torch
from torch import Tensor
import constants
import graphvisualisation
# from agent import Agent

class World: 
    def __init__(self, n_concepts) -> None:
        self.n_quadrants = 8
        self.n_circles = 5
        self.num_vertices = n_concepts
        self.locations = (constants.max_DIMENSIONALITY - constants.min_DIMENSIONALITY)*torch.rand(n_concepts, 2) 
        + constants.min_DIMENSIONALITY
        # creating the radiuses of the concentric circles
        self.radiuses = torch.linspace(0, constants.max_DIMENSIONALITY, steps=self.n_circles)

    @staticmethod
    def quadrant_circle_pair(self, pairs, source):
        co1 = pairs[0] - source[0]
        co2 = pairs[1] - source[1]

        perpendicular = pairs[1] - source[1]
        base = pairs[0]-source[1]

        quad = 0
        segment = 1
        if co1>0 and co2 > 0:
            quad = 1
        elif co1<0 and co2>0:
            quad = 2
        elif co1<0 and co2<0:
            quad = 3
        else:
            quad=4
        
        # finding the circle
        # c_x,c_y =source[0], source[1] #coordinates of the origin
        distance = torch.sqrt(torch.square(co1) + torch.square(co2))

        for i, s in enumerate(self.radiuses):
            if distance<=s.item():
                segment = i+1
                break

        return quad,segment
    


if __name__ == '__main__':
    world = World(10)
    #  graphvisualisation.graphVisualisation(world.locations)
    #  print(world.quadrant_circle_pair(world.locations[0]))
    input_size = 4
    action_space_size = 3
    print(world.locations[0], world.locations[3])
    # x = torch.cat((world.locations[0], world.locations[3]))
    # print(x.size())
    # agent = Agent(input_size, action_space_size, world)

    # agent(world.locations[0], world.locations[3])


# %%
import torch
point1 = [0,0]
point2 = [-1,0]
perpendicular = point2[0] - point1[0]
base = point2[1] - point1[1]
t = torch.atan(torch.tensor(perpendicular/base))
print(t)
deg = torch.rad2deg(t)
print(deg)

# %%
