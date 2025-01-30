#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
from .utils.plotter import TimeSeriesPlotter
from pyts.image import GramianAngularField
from typing import Literal

class GAF_plotter(TimeSeriesPlotter):
    '''
    Gramian Angular Field (GAF) Plotter: Plots the Gramian Angular Field of a univariate time series signal.

    Usage:
    plotter = GAF_plotter()
    plotter.plot(x, method='summation', save_file='gaf.pdf', color_bar=False, label=True, save=True)

    Args:
    - x: np.ndarray
    - method: str
    - save_file: str
    - label: bool
    - save: bool
    - color_bar: bool
    '''
    def __init__(self):
        super().__init__()

    def plot(self, x: np.ndarray, method: Literal['summation', 'difference']='summation', 
             save_file: str='gaf.pdf', color_bar=False, label=True, save=True, tick_size=15, label_size=20):
        x = x.reshape(1, -1)
        transformer = GramianAngularField(method=method)
        gaf = transformer.fit_transform(x)
        plt.figure(figsize=(10, 6))
        plt.imshow(gaf[0], cmap='rainbow', origin='lower')
        if color_bar:
            plt.colorbar()
        if label:
            plt.xlabel("Timestep", size=label_size)
            plt.ylabel("Timestep", size=label_size)
        plt.tick_params(axis='both', which='major', labelsize=tick_size)
        plt.tight_layout()
        if save:
            plt.savefig(save_file, format="pdf", bbox_inches="tight")