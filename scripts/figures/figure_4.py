"""
Figure 3. Reanalysis of electrophysiological recordings from Mishra et al., 2024.
"""

# Imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel
from matplotlib.gridspec import GridSpec

# Imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import FS
from settings import SPECPARAM_SETTINGS
from plots import beautify_ax

# settings
plt.style.use('mplstyle/default.mplstyle')
COLORS = sns.color_palette('colorblind')[:3]
NPERSEG = 512 # samples per segment for fft
SP_FIT_RANGE = [0, .5] # frequency range for spectral parameterization
AP_MODE = 'knee' # aperiodic mode for spectral parameterization
EPOCHS = {'low': (0, 1.5), 'high': (3.5, 5), 'control': (7, 8.5)}
fs = FS['fungi']


def main():

    # display progress
    print("\n\nGenerating Figure 3...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = "figures/main_figures"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load data
    try:
        fname = "data/raw/mishra_2024/Light stimulation test/Mycelium_light test_plate 1.txt"
        data = load_data(fname)
    except FileNotFoundError:
        print("Data not found. Please download the dataset from https://zenodo.org/records/12812074 and place in data/raw/mishra_2024/")
        return

    # Analysis #################################################################

    # epoch data
    data_0 = data[int(EPOCHS['control'][0]*3600*fs):int(EPOCHS['control'][1]*3600*fs)]
    data_1 = data[int(EPOCHS['low'][0]*3600*fs):int(EPOCHS['low'][1]*3600*fs)]
    data_2 = data[int(EPOCHS['high'][0]*3600*fs):int(EPOCHS['high'][1]*3600*fs)]

    # compute spectra
    freqs, spectra_0 = compute_spectrum(data_0, fs, nperseg=NPERSEG)
    freqs, spectra_1 = compute_spectrum(data_1, fs, nperseg=NPERSEG)
    freqs, spectra_2 = compute_spectrum(data_2, fs, nperseg=NPERSEG)

    # apply specparam
    model_0 = SpectralModel(**SPECPARAM_SETTINGS, aperiodic_mode=AP_MODE)
    model_0.fit(freqs, spectra_0, freq_range=SP_FIT_RANGE)

    model_1 = SpectralModel(**SPECPARAM_SETTINGS, aperiodic_mode=AP_MODE)
    model_1.fit(freqs, spectra_1, freq_range=SP_FIT_RANGE)

    model_2 = SpectralModel(**SPECPARAM_SETTINGS, aperiodic_mode=AP_MODE)
    model_2.fit(freqs, spectra_2, freq_range=SP_FIT_RANGE)

    # Plotting #################################################################
    # init figure and gridspec
    fig = plt.figure(figsize=(7.5, 2.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.5])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel a: Light stimulation, time series with time ranges annotated
    ax_a.plot(np.arange(data.shape[0]) / fs / 3600, data, color='k', lw=0.5)
    labels = ['control', 'UV (low)', 'UV (high)']
    for ii, (key, (start, end)) in enumerate(EPOCHS.items()):
        ax_a.axvspan(start, end, color=COLORS[ii], alpha=0.3, label=labels[ii])
    ax_a.set(xlabel='time (hours)', ylabel='voltage (\u03BCV)')
    ax_a.set_title('Light stimulation test')
    ax_a.set_xlim(0, 10)
    ax_a.legend()

    # Panel b: Light stimulation, spectra fits
    plot_sp_results(model_0, label='control', color=COLORS[0], ax=ax_b)
    plot_sp_results(model_1, label='UV (low)', color=COLORS[1], ax=ax_b)
    # plot_sp_results(model_2, label='UV (high)', color=COLORS[2], ax=ax_b)
    ax_b.loglog(model_2.freqs, 10**model_2.power_spectrum, color=COLORS[2], lw=2, alpha=1, label='UV (high)')
    ax_b.loglog(model_2.freqs, 10**model_2._ap_fit, color='k', linestyle='--', lw=1, alpha=1, label='model fits')
    ax_b.set_title('Power spectra')
    ax_b.legend()


    # beautify plot
    for ax in [ax_a, ax_b]:
        beautify_ax(ax)

    # add panel labels
    fig.text(0.03, 0.95, 'a', fontsize=12, fontweight='bold')
    fig.text(0.70, 0.95, 'b', fontsize=12, fontweight='bold')

    # save and show
    fig.savefig(f'{dir_output}/figure_4.png')

    # Print results ############################################################

    # print exponent
    print("Exponents:")
    print(f"  Control: {model_0.get_params('aperiodic', 'exponent'):0.2f}")
    print(f"  Treatment (low): {model_1.get_params('aperiodic', 'exponent'):0.2f}")
    print(f"  Treatment (high): {model_2.get_params('aperiodic', 'exponent'):0.2f}")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def load_data(fname):
    # import data
    data = pd.read_csv(fname, sep=r'\s+', header=None, index_col=0,
                       names=['voltage'])
    data = np.squeeze(data.values)

    # interpolate nan values 
    data = pd.Series(data)
    data = data.interpolate()
    
    return data.values


def plot_sp_results(model, label, color, ax):
    ax.loglog(model.freqs, 10**model.power_spectrum, c=color, lw=2, label=label)
    ax.loglog(model.freqs, 10**model._ap_fit, c='k', linestyle='--', lw=1)


if __name__ == "__main__":
    main()
