#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from .utils.plotter import TimeSeriesPlotter
from pyts.image import RecurrencePlot
from typing import Literal

class Line_Plotter(TimeSeriesPlotter):
    def __init__(self):
        super().__init__()

    def plot(self, x: np.ndarray, save_file: str='line.pdf', label=True, save=True, label_size=20, tick_size=15, color='tab:blue'):
        plt.figure(figsize=(10, 6))
        plt.plot(x, linewidth=4, color=color)
        
        if label:
            plt.xlabel("Timestep", size=label_size)
            plt.ylabel("Value", size=label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tight_layout()
        if save:
            plt.savefig(save_file, format="pdf", bbox_inches="tight")