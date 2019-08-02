import math

from collections.abc import Iterable

class StreamingMoments(object):
    from sys import float_info

    def __init__(self, count=0, min=float_info.max,
                 max=-float_info.max, m1=0.0, m2=0.0):
        (self.count, self.min, self.max) = (count, min, max)
        (self.m1, self.m2) = (m1, m2)

    def __lshift__(self, data):
        if isinstance(data, Iterable):
            # if data is a collection then insert each element of the collection
            for x in data:
                self << x
        else:
            # if data is a numeric value, update the moments according to Chan's formula
            (self.max, self.min) = (max(self.max, data), min(self.min, data))
            dev = data - self.m1
            self.m1 = self.m1 + (dev / (self.count + 1))
            self.m2 = self.m2 + (dev * dev) * self.count / (self.count + 1)
            self.count += 1
        return self

    def mean(self): 
        return self.m1

    def variance(self): 
        return self.m2 / self.count

    def stddev(self): 
        return math.sqrt(self.variance)
    
    def merge_from(self, other):
        if other.count == 0:
            return self
        if self.count == 0:
            (self.m1, self.m2) = (other.m1, other.m2)
            self.count = other.count
            (self.min, self.max) = (other.min, other.max)
            return self
        else:
            dev = other.m1 - self.m1
            new_count = other.count + self.count
            self.m1 = (self.count * self.m1 + other.count * other.m1) / new_count
            self.m2 = self.m2 + other.m2 + (dev * dev) * self.count * other.count / new_count
            self.count = new_count
            self.max = max(self.max, other.max)
            self.min = min(self.min, other.min)
            return self
