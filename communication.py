import torch
import constants
# Communication module


def generate_vocab(size):
    return torch.rand(size)


def mappings(n_locations):
    '''
    Takes the concepts vector and 
    return a mapping for the corresponding vocabulary mapping and vice
    '''
    # vocab for each segment and quadrant is generated
    octants_vocab = generate_vocab(constants.QUADRANT_SIZE)
    segment_vocab = generate_vocab(constants.SEGMENT_SIZE)
