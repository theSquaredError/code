
'''
Provides mappings from:
     vocabulary -> QS pair
     QS pair    -> vocabulary
     location   -> QS pair  
'''
import torch
import constants
from graph_world import World


class Communication:
    def __init__(self,n_locations) -> None:
        self.locations = n_locations
        self.octant_vocab = torch.rand(constants.QUADRANT_SIZE)
        self.segment_vocab = torch.rand(constants.SEGMENT_SIZE)
        self.qs_pair =[]
        self.qs_map = {}

        for location in (self.locations):
            octant, segment = World.quadrant_circle_pair(location)
            self.qs_pair.append([octant,segment])
            self.qs_map[location] = [octant, segment]





    def generate_vocab(size):
        return torch.rand(size)

    def QSPair(location):
        octant, segment = World.quadrant_circle_pair(location)
        return [octant, segment]

    def mappings(self, n_locations):
        '''
        Takes the concepts vector and 
        return a mapping for the corresponding vocabulary mapping and vice
        '''
        # vocab for each segment and quadrant is generated
        octants_vocab = self.generate_vocab(constants.QUADRANT_SIZE) 
        segment_vocab = self.generate_vocab(constants.SEGMENT_SIZE)

        # contains the Quadrant and segment of each location passed    
        mappings = map(self.QSPair, n_locations) 
        # We need Quadrant Segment pair for finding the vocabulary
        # TODO: getting vocabulary for each Q-S pair given
        v_quad = octants_vocab[]

# %%
s = 'Hello'
print(s)
print('hello again')
# %%