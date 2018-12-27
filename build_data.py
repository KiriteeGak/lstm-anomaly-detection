import numpy as np
import random as rd


class SplitData(object):
    def __init__(self, x, split_ratios=(0.6, 0.3), axis=0):
        self.x = x
        rd.shuffle(np.array(self.x))
        self.axis = axis
        self.sections = tuple([int(x.shape[axis] *
                                   round(sum(split_ratios[:i+1]), 1)) for i, _ in enumerate(split_ratios)])

    def split_data(self):
        return np.split(self.x, indices_or_sections=self.sections, axis=self.axis)

    def create_data(self):
        pass
