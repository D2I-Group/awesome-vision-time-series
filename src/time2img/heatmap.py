#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from .utils.plotter import TimeSeriesPlotter

class UniHeatmap_Plotter(TimeSeriesPlotter):
    '''
    UniHeatmap Plotter: Plots the heatmap of a univariate time series signal.

    Usage:
    plotter = UniHeatmap_Plotter()
    plotter.plot(x, patch_size=24, label=True, save_file='heatmap.pdf')

    Args:
    - x: np.ndarray
    - patch_size: int
    - label: bool
    - save_file: str
    '''
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, patch_size: int, label: bool = False, save_file: str = 'heatmap.pdf', **kwargs):
        def heatmap_univariate(
            x: np.ndarray,
            patch_size: int,
            *,
            colorbar: bool = False,
            label: bool = False,
            title: bool = False,
            save: bool = True,
            save_file: str = 'heatmap.pdf',
            tick_size: int = 15,
            label_size: int = 20
        ):
            # Calculate number of patches
            N = len(x)
            patch_num = N // patch_size

            # Reshape signal into patches with time as first dimension
            patches = x[: patch_num * patch_size].reshape(patch_size, patch_num)

            # Create heatmap visualization
            plt.figure(figsize=(10, 6))
            plt.imshow(
                patches.T,
                aspect="auto",
                origin="lower",  # Make time flow downward
                cmap="viridis",
            )

            if colorbar:
                plt.colorbar(label="Amplitude")

            if label:
                plt.ylabel("Patch Number", size=label_size)
                plt.xlabel("Timestamp", size=label_size)

            if title:
                plt.title(f"Signal Heatmap ({patch_num} patches)")
                
            plt.tick_params(axis='both', which='major', labelsize=tick_size)
            plt.tight_layout()
            if save:
                if label:
                    plt.savefig(save_file, format="pdf", bbox_inches="tight")
                else:
                    plt.savefig(save_file, format="pdf", bbox_inches="tight")
            else:
                plt.show()
        heatmap_univariate(x, patch_size, label=label, save_file=save_file, **kwargs)


class MultiHeatmap_Plotter(TimeSeriesPlotter):
    '''
    MultiHeatmap Plotter: Plots the heatmap of a multivariate time series signal.

    Usage:
    plotter = MultiHeatmap_Plotter()
    plotter.plot(x, label=True, save_file='multivariate_heatmap.pdf')

    Args:
    - x : np.ndarray
      A 2D array with shape (n_samples, n_features) containing the multivariate time series data
    - label: bool
    - save_file: str
    '''
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, label: bool = False, save_file: str = 'multivariate_heatmap.pdf', **kwargs):
        def heatmap_multivariate(
            x: np.ndarray,
            *,
            colorbar: bool = False,
            label: bool = False,
            title: bool = False,
            save: bool = True,
            save_file: str = 'multivariate_heatmap.pdf',
            label_size: int = 20,
            tick_size: int = 15
        ):
            plt.figure(figsize=(10, 6))
            plt.imshow(
                x.T,
                aspect="auto",
                origin="lower",  # Make time flow downward
                cmap="viridis",
            )

            if colorbar:
                plt.colorbar(label="Value")

            if label:
                plt.ylabel("Variates", size=label_size)
                plt.xlabel("Timestamp", size=label_size)

            if title:
                plt.title("Multivariate Signal Heatmap")
            
            plt.tick_params(axis='both', which='major', labelsize=tick_size)
            plt.tight_layout()
            if save:
                if label:
                    plt.savefig(
                        save_file, format="pdf", bbox_inches="tight"
                    )
                else:
                    plt.savefig(save_file, format="pdf", bbox_inches="tight")
            else:
                plt.show()
        heatmap_multivariate(x, label=label, save_file=save_file, **kwargs)