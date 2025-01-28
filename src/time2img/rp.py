#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from .utils.plotter import TimeSeriesPlotter
from pyts.image import RecurrencePlot
from typing import Literal

class RP_plotter(TimeSeriesPlotter):
    '''
    Recurrence Plot (RP) Plotter: Plots the Recurrence Plot of a univariate time series signal.
    Usage:
    plotter = RP_plotter()
    plotter.plot(x, threshold='point', percentage=10, dimension=1, time_delay=1, save_file='rp.pdf', color_bar=False, label=True, save=True)

    Args:
    - x: np.ndarray
    - threshold: str
    - percentage: float
    - dimension: int
    - time_delay: int
    - save_file: str
    - label: bool
    - save: bool
    - color_bar: bool
    '''
    def __init__(self):
        super().__init__()

    def plot(self, x: np.ndarray, threshold: str='point', percentage: float=10, 
             dimension: int=1, time_delay: int=1,
             save_file: str='rp.pdf', color_bar=False, label=True, save=True,
             tick_size=15, label_size=20):
        x = x.reshape(1, -1)
        transformer = RecurrencePlot(threshold=threshold, percentage=percentage, 
                                     dimension=dimension, time_delay=time_delay)
        rp = transformer.fit_transform(x)
        plt.figure(figsize=(10, 6))
        plt.imshow(rp[0], cmap='gray', origin='lower')
        if color_bar:
            plt.colorbar()
        if label:
            plt.xlabel("Timestamp", size=label_size)
            plt.ylabel("Timestamp", size=label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tight_layout()
        if save:
            plt.savefig(save_file, format="pdf", bbox_inches="tight")