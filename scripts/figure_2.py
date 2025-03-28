"""
Figure 2: Electrophysiological recordings from several species of fungi.
"""

# imports - standard
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from neurodsp.spectral import compute_spectrum
from specparam import SpectralModel
import numpy as np
from scipy.stats import zscore
from matplotlib.gridspec import GridSpec

# imports - custom
import sys
sys.path.append("code")
from info import FS, SPECIES
from utils import get_start_time, print_time_elapsed, beautify_ax

# settings
NPERSEG = 128 # samples per segment for fft
TITLES = ('Herecium', 'Lentinus', 'Pleurotus')
COLORS = sns.color_palette('colorblind')
plt.style.use('mplstyle/default.mplstyle')


def main():

    # display progress
    print("\n\nGenerating Figure 2...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = "figures"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load data
    df = pd.read_csv('data/recordings.csv')

    # Analysis #################################################################

    # init
    psd = {}
    models = {}
    params_list = []

    # analyze each species
    for species in SPECIES[:3]:
        # compute power spectrum
        freqs, psd[species] = compute_spectrum(df[species], FS, nperseg=NPERSEG)

        # fit spectral model
        model = SpectralModel(aperiodic_mode='knee', verbose=False)
        model.fit(freqs, psd[species])
        models[species] = model
        params = model.get_params('aperiodic')
        temp = pd.Series({'species': species, 'offset': params[0],
                        'knee': params[1], 'exponent': params[2]})
        params_list.append(temp)
    params = pd.DataFrame(params_list)

    # normalize power
    psd_norm = {}
    fit_norm = {}
    for species in SPECIES[:3]:
        psd_norm[species] = psd[species] / np.sum(psd[species])
        fit_norm[species] = 10**models[species]._ap_fit / np.sum(10**models[species]._ap_fit)
        df[f"{species}_norm"] = zscore(df[species])
        
    # print results
    print(params)
    print(f"\nMean exponent: {params.loc[:2, 'exponent'].mean()}")

    # Plotting #################################################################

    # create figure 
    fig = plt.figure(figsize=(6.5, 2.34), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 0.6], 
                height_ratios=[1, 1, 1])
    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[1, 0])
    ax_a2 = fig.add_subplot(gs[2, 0])
    ax_b = fig.add_subplot(gs[:, 1])

    # A: time-series
    for i, (sp, title, ax) in enumerate(zip(SPECIES[:3], TITLES, 
                                            [ax_a0, ax_a1, ax_a2])):
        ax.plot(df['time'], df[f"{sp}_norm"], 
                c=COLORS[i], lw=1)
        ax.set_title(title)
    ax_a2.set_xlabel('time (s)')
    fig.text(-0.01, 0.20, "normalized voltage (z-score)", rotation=90, fontsize=10)

    for ax in [ax_a0, ax_a1]:
        ax.set_xticks([])

    # B: plot power spectrum and fit
    for sp, color in zip(SPECIES[:-1], COLORS):
        ax_b.loglog(freqs, psd_norm[sp], color=color, lw=2, alpha=1, label=sp)
        ax_b.plot(models[sp].freqs, fit_norm[sp], color=color, lw=2, 
                linestyle='--', alpha=0.5)
    ax_b.set(xlabel= 'frequency (Hz)', ylabel='normalized power (au)')
    ax_b.set_title('Power spectral density')
    ax_b.legend()

    # beautify
    for ax in [ax_a0, ax_a1, ax_a2, ax_b]:
        beautify_ax(ax)

    # add panel labels
    fig.text(0.01, 0.95, 'a', fontsize=12, fontweight='bold')
    fig.text(0.61, 0.95, 'b', fontsize=12, fontweight='bold')

    # save
    fig.savefig('figures/figure_2.png', bbox_inches='tight')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
