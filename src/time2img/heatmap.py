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
                plt.xlabel("Timestep", size=label_size)

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


class AdaptiveHeatmap_Plotter(TimeSeriesPlotter):
    def __init__(self):
        super().__init__()
    
    def plot(self, x: np.ndarray, top_k: int = 1, 
             label: bool = True, 
             save_file: str = 'adaptive_heatmap.pdf', **kwargs):
        def heatmap_with_adaptive_period(
            x: np.ndarray,
            top_k: int = 2,
            *,
            show_raw: bool = False,
            show_fft: bool = False,
            show_2d: bool = True,
            colorbar: bool = False,
            label: bool = False,
            save: bool = True,
            title: bool = False,
            save_file: str = 'adaptive_heatmap.pdf',
            label_size: int = 20,
            tick_size: int = 15
        ):
            """Analyze and visualize periodic patterns in time series data

            Args:
                time_series: Input time series data
                top_k: Number of main frequencies to identify
                show_raw: Whether to display raw time series
                show_fft: Whether to display FFT spectrum
                show_2d: Whether to display 2D periodic pattern
                colorbar: Whether to show colorbar
                label: Whether to show axis labels
                save: Whether to save the plots
            """
            # Display raw time series
            if show_raw:
                plt.figure(figsize=(12, 5))
                plt.plot(x)
                if label:
                    plt.xlabel("Time")
                    plt.ylabel("Amplitude")
                    plt.title("Raw Time Series")
                if save:
                    plt.savefig("raw_series.pdf", format="pdf", bbox_inches="tight")
                else:
                    plt.show()

            # FFT analysis
            freqs = np.fft.rfftfreq(len(x))
            fft_vals = np.fft.rfft(x)
            amplitudes = np.abs(fft_vals)
            amplitudes[0] = 0  # Ignore DC component

            # Find top k peak frequencies
            top_freq_indices = np.argsort(amplitudes)[-top_k:]
            top_periods = (1 / freqs[top_freq_indices]).astype(int)

            # Display FFT spectrum
            if show_fft:
                plt.figure(figsize=(12, 6))
                plt.plot(freqs, amplitudes)
                if label:
                    plt.xlabel("Frequency")
                    plt.ylabel("Amplitude")
                if title:
                    plt.title("FFT Spectrum Analysis")
                if save:
                    plt.savefig("fft_spectrum.pdf", format="pdf", bbox_inches="tight")
                else:
                    plt.show()

            # Display 2D periodic pattern
            if show_2d:
                for period in top_periods:
                    # Reshape time series into 2D array with given period
                    n_periods = len(x) // period
                    x_2d = x[: n_periods * period].reshape(n_periods, period)
                    plt.figure(figsize=(10, 6))
                    plt.imshow(x_2d, aspect="auto", cmap="viridis", origin="lower")
                    if colorbar:
                        plt.colorbar(label="Magnitude")
                    if label:
                        plt.xlabel("Timestep", size=label_size)
                        plt.ylabel("Patch Number", size=label_size)
                        if title:
                            plt.title(f"Time Series Reshaped to 2D (Period = {period})")
                    plt.tick_params(axis='both', which='major', labelsize=tick_size)
                    plt.tight_layout()

                    if save:
                        plt.savefig(
                            save_file, format="pdf", bbox_inches="tight"
                        )
                    else:
                        plt.show()
        heatmap_with_adaptive_period(x, top_k=top_k, save_file=save_file, label=label, **kwargs)




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
            tick_size: int = 15,
            clip_percentage: float = 0.005
        ):
            plt.figure(figsize=(10, 6))

            y = x.astype(int)
            values, counts = np.unique(y, return_counts=True)
            tot = 0
            max_ind = 0
            low_ind = 0
            for i in counts:
                if tot < np.sum(counts) * clip_percentage:
                    tot += i
                    max_ind += 1
                else:
                    break   
            low_clip = values[max_ind] 
            tot = 0
            for i in counts[::-1]:
                if tot < np.sum(counts) * clip_percentage:
                    tot += i
                    low_ind += 1
                else:
                    break  
            max_clip = values[-low_ind]

            plt.imshow(
                x.clip(min=low_clip, max=max_clip).T,
                aspect="auto",
                origin="lower",  # Make time flow downward
                cmap="jet",
                interpolation="nearest"
            )

            if colorbar:
                plt.colorbar(label="Value")

            if label:
                plt.ylabel("Variates", size=label_size)
                plt.xlabel("Timestep", size=label_size)

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