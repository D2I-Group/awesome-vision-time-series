#!/usr/bin/env python
from abc import ABC, abstractmethod

class TimeSeriesPlotter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def plot(self, x, **kwargs):
        pass