"""
Figure 3. Reanalysis of electrophysiological recordings from Mishra et al., 2024.
"""

# Imports - standard
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel, SpectralGroupModel
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

# Imports - custom
import sys
sys.path.append("code")
from info import FS, DATASET_PATH
from utils import get_start_time, print_time_elapsed, beautify_ax

# settings
NPERSEG = 2**12 # samples per segment for fft
COLORS = sns.color_palette('colorblind')
plt.style.use('mplstyle/default.mplstyle')


def main():

    # display progress
    print("\n\nGenerating Figure 3...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = "figures"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load data
    try:
        data_s = load_data(f"{DATASET_PATH}/Mycelium bioelectric native and light stimulated data and robot control/Long-term fungi recordings/Small plate_recording.txt")
        data_l = load_data(f"{DATASET_PATH}/Mycelium bioelectric native and light stimulated data and robot control/Long-term fungi recordings/Large plate recording.txt")
    except FileNotFoundError:
        print("Data not found. Please download the dataset from https://zenodo.org/records/12812074 and update the DATASET_PATH variable in code/info.py.")
        return

    # Analysis #################################################################

    # compute spectra for whole signals
    freqs_s, spectra_s = compute_spectrum(data_s, FS, nperseg=NPERSEG)
    freqs_l, spectra_l = compute_spectrum(data_l, FS, nperseg=NPERSEG)
    
    # match frequency range to other figures
    freq_mask = freqs_s > 0.078
    freqs_s = freqs_s[freq_mask]
    spectra_s = spectra_s[freq_mask]
    freqs_l = freqs_l[freq_mask]
    spectra_l = spectra_l[freq_mask]

    # parameterize spectra for whole recording
    model_s = SpectralModel()
    model_s.fit(freqs_s, spectra_s)
    exponent_s = model_s.get_params('aperiodic', 'exponent')

    model_l = SpectralModel()
    model_l.fit(freqs_l, spectra_l)
    exponent_l = model_l.get_params('aperiodic', 'exponent')

    # epoch data into 1 hour blocks
    duration = 60 * 60 * FS
    n_blocks = data_s.shape[0] // duration
    data = data_s[:n_blocks*duration]
    data = data.reshape(n_blocks, duration)

    # compute spectra and exponent for each block
    freqs, spectra = compute_spectrum(data, FS, nperseg=NPERSEG)
    model = SpectralGroupModel()
    model.fit(freqs, spectra)
    exponent = model.get_params('aperiodic', 'exponent')

    # compute rolling average of exponent (every day)
    n_days = 30
    exponent_mu = np.zeros(n_days)
    exponent_std = np.zeros(n_days)
    for ii in range(n_days):
        exponent_mu[ii] = np.mean(exponent[ii*24:(ii+1)*24])
        exponent_std[ii] = np.std(exponent[ii*24:(ii+1)*24])

    # fit sigmoid
    time_days = np.arange(n_days) + 1
    p0 = [max(exponent_mu), np.median(time_days), 1, min(exponent_mu)]
    popt, _ = curve_fit(sigmoid, time_days, exponent_mu, p0)
    sigmoid_fit = sigmoid(time_days, *popt)

    # Plotting #################################################################

    # init figure
    fig = plt.figure(figsize=(6.5, 2.34))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.6, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # A: plot session spectra
    ax1.loglog(freqs_s, spectra_s, color=COLORS[0], alpha=0.7, label='large')
    ax1.loglog(freqs_l, spectra_l, color=COLORS[1], alpha=0.7, label='large')
    ax1.set(xlabel='frequency (Hz)', ylabel='power (\u03BCV\u00b2/Hz)')
    ax1.legend()

    # B: plot exponent over time and sigmoid fit
    ax2.errorbar(time_days, exponent_mu, yerr=exponent_std, fmt='o', color='k',
                label='daily average') # data by day
    ax2.plot(time_days, sigmoid_fit, color='grey', label='sigmoid fit')
    ax2.set(xlabel='time (days)', ylabel='exponent')
    ax2.legend()

    # beautify plot
    for ax in [ax1, ax2]:
        beautify_ax(ax)

    # label plots
    ax1.set_title('Power spectral density')
    ax2.set_title('Spectral exponent')

    # add panel labels
    fig.text(0.01, 0.95, 'a', fontsize=12, fontweight='bold')
    fig.text(0.41, 0.95, 'b', fontsize=12, fontweight='bold')

    # save and show
    fig.savefig('figures/figure_3.png', dpi=600)

    # Print results ############################################################

    # print exponent for each plate
    print("\nExponent for whole recording:")
    print(f"  Small: {exponent_s:0.2f}")
    print(f"  Large: {exponent_l:0.2f}")

    # print parameters of sigmoid
    print("\nSigmoid fit parameters:")
    print(f"  R-squared: {np.corrcoef(exponent_mu, sigmoid_fit)[0, 1]:0.2f}")
    print(f"  L (maximum): {popt[0]} ({popt[3]+popt[0]:0.2f})")
    print(f"  k (slope): {popt[2]:0.2f}")
    print(f"  X_0 (inflection point): {popt[1]:0.2f}")
    print(f"  b (offset): {popt[3]:0.2f}")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def load_data(fname):
    # import data
    data = pd.read_csv(fname, sep='\s+', header=None, index_col=0,
                       names=['voltage'])
    data = np.squeeze(data.values)

    # interpolate nan values 
    data = pd.Series(data)
    data = data.interpolate()
    data = data.values
    print(data.shape)
    
    return data


def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)


if __name__ == "__main__":
    main()
