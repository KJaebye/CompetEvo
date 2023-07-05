# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class Memory (Reply Buffer).
#   @author: Modified from khrylib by Ye Yuan, modified by Kangyao Huang
#   @created date: 27.Oct.2022
# ------------------------------------------------------------------------------------------------------------------- #
from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)