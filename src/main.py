from time2img import (
    GAF_plotter,
    STFT_Plotter,
    Wavelet_Plotter,
    Filterbank_Plotter,
    data_provider,
    UniHeatmap_Plotter,
    MultiHeatmap_Plotter,
    RP_plotter,
    Line_Plotter
)
import os


if __name__ == "__main__":
    # Univariate time series
    dataset = data_provider(data='electricity', root_path="./dataset", split='train', download=True, size=[336, 96, 96])
    seq_x, _, _, _ = dataset.__getitem__(0)
    X = seq_x[:, 0]

    if not os.path.exists('./image'):
        os.mkdir('./image')
        os.mkdir('./image/labeled')
        os.mkdir('./image/unlabeled')

    plotter = STFT_Plotter()
    plotter.plot(X,window_size=10,hop=1,T_x=3600, label=True, save_file='./image/labeled/stft.pdf', tick_size=15, label_size=20)
    plotter.plot(X,window_size=10,hop=1,T_x=3600, label=False, save_file='./image/unlabeled/stft.pdf')

    plotter = Wavelet_Plotter()
    plotter.plot(X, label=True, save_file='./image/labeled/wavelet.pdf')
    plotter.plot(X, label=False, save_file='./image/unlabeled/wavelet.pdf')

    plotter = Filterbank_Plotter()
    plotter.plot(X, window_size=10, hop=1, T_x=3600, num_filters=10, label=True, save_file='./image/labeled/filterbank.pdf')
    plotter.plot(X, window_size=10, hop=1, T_x=3600, num_filters=10, label=False, save_file='./image/unlabeled/filterbank.pdf')

    plotter = UniHeatmap_Plotter()
    plotter.plot(X, patch_size=24, label=True, save_file='./image/labeled/heatmap.pdf')
    plotter.plot(X, patch_size=24, label=False, save_file='./image/unlabeled/heatmap.pdf')

    plotter = GAF_plotter()
    plotter.plot(X, method='summation', label=True, save_file='./image/labeled/gaf.pdf')
    plotter.plot(X, method='summation', label=False, save_file='./image/unlabeled/gaf.pdf')

    plotter = RP_plotter()
    plotter.plot(X, threshold=0.1, label=False, save_file='./image/unlabeled/rp.pdf')
    plotter.plot(X, threshold=0.1, label=True, save_file='./image/labeled/rp.pdf')

    plotter = Line_Plotter()
    for color in ['tab:blue', 'tab:orange', 'tab:green', 'black']:
        plotter.plot(X, color=color, label=True, save_file=f'./image/labeled/line_{color}.pdf')
        plotter.plot(X, color=color, label=False, save_file=f'./image/unlabeled/line_{color}.pdf')


    # Multivariate time series
    dataset = data_provider(data='electricity', root_path="./dataset", split='train', features='M', download=False, size=[336, 96, 96])
    X, _, _, _ = dataset.__getitem__(0)
    plotter = MultiHeatmap_Plotter()
    plotter.plot(X, label=False, save_file='./image/unlabeled/multiheatmap.pdf')
    plotter.plot(X, label=True, save_file='./image/labeled/multiheatmap.pdf')