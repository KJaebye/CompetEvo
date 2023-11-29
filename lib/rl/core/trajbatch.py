import numpy as np


class TrajBatch:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = np.stack(next(self.batch))
        self.actions = np.stack(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = np.stack(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))

class MaTrajBatch:
    def __init__(self, memories_list):
        memories_list = list(map(list, zip(*memories_list)))
        self.buffers = []
        for i in range(len(memories_list)):
            """ i means the ith agent's memory. """
            memory_i_list = memories_list[i]
            self.buffers.append(TrajBatch(memory_i_list))

class TrajBatchDisc:

    def __init__(self, memory_list):
        memory = memory_list[0]
        for x in memory_list[1:]:
            memory.append(x)
        self.batch = zip(*memory.sample())
        self.states = list(next(self.batch))
        self.actions = list(next(self.batch))
        self.masks = np.stack(next(self.batch))
        self.next_states = list(next(self.batch))
        self.rewards = np.stack(next(self.batch))
        self.exps = np.stack(next(self.batch))

class MaTrajBatchDisc:
    def __init__(self, memories_list):
        memories_list = list(map(list, zip(*memories_list)))
        self.buffers = []
        for i in range(len(memories_list)):
            """ i means the ith agent's memory. """
            memory_i_list = memories_list[i]
            self.buffers.append(TrajBatchDisc(memory_i_list)) 


        