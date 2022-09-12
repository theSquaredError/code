
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
        self.qs_pairs =[]
        self.qs_map = {}
        self.qs_vocab = {}
        for location in (self.locations):
            octant, segment = World.quadrant_circle_pair(location)
            self.qs_pairs.append([octant,segment])
            self.qs_map[location] = [octant, segment]

        for qs_pair in self.qs_pairs:
            q_word = self.octant_vocab[qs_pair[0]]
            s_word = self.segment_vocab[qs_pair[1]]

            self.qs_vocab[qs_pair] = [q_word, s_word]


    def find_vocab(self, qs_pair):
        try:
            vocab = self.qs_vocab(qs_pair)
            return vocab
        except:
            return -1

    def find_QSPair(self,location):
        try:
            qs_pair = self.qs_map[location]
            return qs_pair
        except:
            return -1