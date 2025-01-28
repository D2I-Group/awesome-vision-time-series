#!/usr/bin/env python
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from data_factory import Dataset_ETT_hour
from utils.config_loader import get_config
from pyts.image import GramianAngularField
import random
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

## Expired Config
# [dataset]
# root: ./dataset/ETT-small

# [general_setting]
# tick_num = 5

# [recurrence_plot]
# dimension = 1
# time_delay = 1
# threshold: point
# percentage = 10


# [gramian_angular]
# method: summation

CONFIG_PATH = './config.ini'
Config = get_config(CONFIG_PATH)

class RP_plotter:
    def __init__(self):
        super().__init__()

    @staticmethod
    def plot_rp(Config):
        etth2 = Dataset_ETT_hour(Config['dataset']['root'], get_whole=False)
        etth2_whole = Dataset_ETT_hour(Config['dataset']['root'], get_whole=True)
        dimension = Config.getint('recurrence_plot', 'dimension')
        time_delay = Config.getint('recurrence_plot', 'time_delay')
        threshold = Config['recurrence_plot']['threshold']
        percentage = Config.getfloat('recurrence_plot', 'percentage')

        selected_idx = [0]
        for _ in range(2):
            selected_idx.append(random.randint(0, len(etth2)))

        whole_X = etth2_whole[0][0].reshape(1, -1)
        partial_X1, partial_X2, partial_X3 = etth2[selected_idx[0]][0].reshape(1, -1), \
                                             etth2[selected_idx[1]][0].reshape(1, -1), \
                                             etth2[selected_idx[2]][0].reshape(1, -1)

        rp = RecurrencePlot(threshold=threshold, percentage=percentage, dimension=dimension, time_delay=time_delay)
        whole_rp = rp.fit_transform(whole_X)
        partial_rp1 = rp.fit_transform(partial_X1)
        partial_rp2 = rp.fit_transform(partial_X2)
        partial_rp3 = rp.fit_transform(partial_X3)

        # rp plot w/o time series
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(whole_rp[0], cmap='binary', origin='lower')
        axs[0, 0].set_title('Whole Series')

        axs[0, 1].imshow(partial_rp1[0], cmap='binary', origin='lower')
        axs[0, 1].set_title(f'Series from Timestamp {selected_idx[0]}')

        axs[1, 0].imshow(partial_rp2[0], cmap='binary', origin='lower')
        axs[1, 0].set_title(f'Series from Timestamp {selected_idx[1]}')

        axs[1, 1].imshow(partial_rp3[0], cmap='binary', origin='lower')
        axs[1, 1].set_title(f'Series from Timestamp {selected_idx[2]}')

        plt.savefig('recurrence_plot.png')

    @staticmethod
    def plot_rp_series(Config):
        etth2 = Dataset_ETT_hour(Config['dataset']['root'], get_whole=False)
        etth2_whole = Dataset_ETT_hour(Config['dataset']['root'], get_whole=True)
        dimension = Config.getint('recurrence_plot', 'dimension')
        time_delay = Config.getint('recurrence_plot', 'time_delay')
        threshold = Config['recurrence_plot']['threshold']
        percentage = Config.getfloat('recurrence_plot', 'percentage')

        selected_idx = [0]
        for _ in range(2):
            selected_idx.append(random.randint(0, len(etth2)))

        whole_X = etth2_whole[0][0].reshape(1, -1)
        partial_X1, partial_X2, partial_X3 = etth2[selected_idx[0]][0].reshape(1, -1), \
                                             etth2[selected_idx[1]][0].reshape(1, -1), \
                                             etth2[selected_idx[2]][0].reshape(1, -1)

        rp = RecurrencePlot(threshold=threshold, percentage=percentage, dimension=dimension, time_delay=time_delay)
        whole_rp = rp.fit_transform(whole_X)
        partial_rp1 = rp.fit_transform(partial_X1)
        partial_rp2 = rp.fit_transform(partial_X2)
        partial_rp3 = rp.fit_transform(partial_X3)

        # rp plot w time series
        fig = plt.figure(figsize=(20, 20))

        gs = fig.add_gridspec(4, 4,  width_ratios=(1, 7, 1, 7), height_ratios=(1, 7, 1, 7),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.1, hspace=0.1)
        
        def sample_plot(seq, rp, x, y, seq_init=0, whole=False):
            tick_num = Config.getint('general_setting', 'tick_num')
            seq_len = len(seq)
            time_ticks = np.arange(0, seq_len, seq_len // tick_num)[:tick_num]
            time_ticklabels = [seq_init+i*(seq_len // tick_num) for i in range(tick_num)][:tick_num]
            value_ticks = []
            # Plot the time series on the top
            ax_top = fig.add_subplot(gs[x, y])
            ax_top.plot(seq)
            ax_top.set_xticks(time_ticks)
            ax_top.set_xticklabels(time_ticklabels)
            ax_top.set_yticks(value_ticks)
            ax_top.set_yticklabels(value_ticks)
            ax_top.xaxis.tick_bottom()
            if whole:
                ax_top.set_title(f'Whole Series')
            else:
                ax_top.set_title(f'Series from Timestamp {seq_init}')

            # Plot the recurrence plot on the bottom right
            ax_rp = fig.add_subplot(gs[x+1, y], sharex=ax_top)
            ax_rp.imshow(rp[0], cmap='binary', origin='lower')
            ax_rp.set_yticks([])
            ax_rp.xaxis.tick_top()
            plt.setp(ax_rp.get_xticklabels(), visible=False)
        
        sample_plot(etth2_whole[0][0], whole_rp, 0, 1, whole=True)
        sample_plot(etth2[selected_idx[0]][0], partial_rp1, 0, 3, selected_idx[0])
        sample_plot(etth2[selected_idx[1]][0], partial_rp2, 2, 1, selected_idx[1])
        sample_plot(etth2[selected_idx[2]][0], partial_rp3, 2, 3, selected_idx[2])

        plt.savefig('recurrence_series.png')


class GAR_plotter:
    def __init__(self):
        super().__init__()

    @staticmethod
    def plot_gar(Config):
        etth2 = Dataset_ETT_hour(Config['dataset']['root'], get_whole=False)
        etth2_whole = Dataset_ETT_hour(Config['dataset']['root'], get_whole=True)
        selected_idx = [0]
        for _ in range(2):
            selected_idx.append(random.randint(0, len(etth2)))

        whole_X = etth2_whole[0][0].reshape(1, -1)
        partial_X1, partial_X2, partial_X3 = etth2[selected_idx[0]][0].reshape(1, -1), \
                                             etth2[selected_idx[1]][0].reshape(1, -1), \
                                             etth2[selected_idx[2]][0].reshape(1, -1)
        print(partial_X1.shape)
        transformer = GramianAngularField()

        whole_gaf = transformer.fit_transform(whole_X)
        partial_gaf1 = transformer.fit_transform(partial_X1)
        partial_gaf2 = transformer.fit_transform(partial_X2)
        partial_gaf3 = transformer.fit_transform(partial_X3)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(whole_gaf[0], cmap='rainbow', origin='lower')
        axs[0, 0].set_title('Whole Series')
        axs[0, 1].imshow(partial_gaf1[0], cmap='rainbow', origin='lower')
        axs[0, 1].set_title(f'Series from Timestamp {selected_idx[0]}')
        axs[1, 0].imshow(partial_gaf2[0], cmap='rainbow', origin='lower')
        axs[1, 0].set_title(f'Series from Timestamp {selected_idx[1]}')

        axs[1, 1].imshow(partial_gaf3[0], cmap='rainbow', origin='lower')
        axs[1, 1].set_title(f'Series from Timestamp {selected_idx[2]}')
        plt.colorbar(axs[0, 0].imshow(whole_gaf[0], cmap='rainbow', origin='lower'), ax=axs[0, 0])
        plt.colorbar(axs[0, 1].imshow(partial_gaf1[0], cmap='rainbow', origin='lower'), ax=axs[0, 1])
        plt.colorbar(axs[1, 0].imshow(partial_gaf2[0], cmap='rainbow', origin='lower'), ax=axs[1, 0])
        plt.colorbar(axs[1, 1].imshow(partial_gaf3[0], cmap='rainbow', origin='lower'), ax=axs[1, 1])
        plt.savefig('gar_plot.png')
    
    @staticmethod
    def plot_gar_series(Config):
        etth2 = Dataset_ETT_hour(Config['dataset']['root'], get_whole=False)
        etth2_whole = Dataset_ETT_hour(Config['dataset']['root'], get_whole=True)
        selected_idx = [0]
        for _ in range(2):
            selected_idx.append(random.randint(0, len(etth2)))

        whole_X = etth2_whole[0][0].reshape(1, -1)
        partial_X1, partial_X2, partial_X3 = etth2[selected_idx[0]][0].reshape(1, -1), \
                                             etth2[selected_idx[1]][0].reshape(1, -1), \
                                             etth2[selected_idx[2]][0].reshape(1, -1)
        transformer = GramianAngularField(method=Config['gramian_angular']['method'])

        whole_gaf = transformer.fit_transform(whole_X)
        partial_gaf1 = transformer.fit_transform(partial_X1)
        partial_gaf2 = transformer.fit_transform(partial_X2)
        partial_gaf3 = transformer.fit_transform(partial_X3)

        fig = plt.figure(figsize=(20, 20))

        gs = fig.add_gridspec(4, 4,  width_ratios=(1, 7, 1, 7), height_ratios=(1, 7, 1, 7),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.1, hspace=0.1)
        
        def sample_plot(seq, gaf, x, y, seq_init=0, whole=False, last=False):
            tick_num = Config.getint('general_setting', 'tick_num')
            seq_len = len(seq)
            time_ticks = np.arange(0, seq_len, seq_len // tick_num)[:tick_num]
            time_ticklabels = [seq_init+i*(seq_len // tick_num) for i in range(tick_num)][:tick_num]
            value_ticks = []
            # Plot the time series on the top
            ax_top = fig.add_subplot(gs[x, y])
            ax_top.plot(seq)
            ax_top.set_xticks(time_ticks)
            ax_top.set_xticklabels(time_ticklabels)
            ax_top.set_yticks(value_ticks)
            ax_top.set_yticklabels(value_ticks)
            ax_top.xaxis.tick_bottom()
            if whole:
                ax_top.set_title(f'Whole Series')
            else:
                ax_top.set_title(f'Series from Timestamp {seq_init}')

            # Plot the recurrence plot on the bottom right
            ax_rp = fig.add_subplot(gs[x+1, y], sharex=ax_top)
            im = ax_rp.imshow(gaf[0], cmap='rainbow', origin='lower')
            if last:
                divider = make_axes_locatable(ax_rp)
                cax = divider.append_axes('bottom', size='1%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='horizontal')
            ax_rp.set_yticks([])
            ax_rp.xaxis.tick_top()
            plt.setp(ax_rp.get_xticklabels(), visible=False)
        
        sample_plot(etth2_whole[0][0], whole_gaf, 0, 1, whole=True)
        sample_plot(etth2[selected_idx[0]][0], partial_gaf1, 0, 3, selected_idx[0])
        sample_plot(etth2[selected_idx[1]][0], partial_gaf2, 2, 1, selected_idx[1])
        sample_plot(etth2[selected_idx[2]][0], partial_gaf3, 2, 3, selected_idx[2], last=True)

        plt.savefig('gar_series.png')


if __name__ == '__main__':
    # RP_plotter.plot_rp(Config)
    # RP_plotter.plot_rp_series(Config)
    # GAR_plotter.plot_gar_series(Config)
    GAR_plotter.plot_gar(Config)
