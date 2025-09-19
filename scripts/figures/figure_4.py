"""
Figure 3: Shared spectral signiture across kingdoms

A. Cartoon to represent kingdoms
B. Example time-series for each kingdom
C. Power spectra for each kingdom
D. Violin plot of exponent across kingdoms 
E. Violin plot of timescale across kingdoms 
"""

# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import f_oneway

import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import KINGDOMS
from utils import shift_signals, zscore
from plots import plot_spectra, beautify_ax

# settings
plt.style.use('mplstyle/default.mplstyle')
SHIFT = [8, 8, 5] # Shift signals for plotting (STDs)
N_SIGNALS = 10 # Number of signals to plot per kingdom
N_SAMPLES = {'fungi'   : 12000, 
             'plantae' : 36000, 
             'animalia': 1000} # Number of samples to plot


def main():

    # display progress
    print("\n\nGenerating Figure 4...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"figures/main_figures"
    os.makedirs(dir_output, exist_ok=True)

    # LOAD DATA ################################################################
    print("Importing data...")

    # init
    signals, time, spectra, freqs = {}, {}, {}, {}
    
    # loop through kingdoms
    for ii, kingdom in enumerate(KINGDOMS):
        # load signals
        signals_ii = np.load(f"data/epochs/{kingdom}/signals.npy")
        time_ii = np.load(f"data/epochs/{kingdom}/time.npy")
        signals[kingdom] = signals_ii[:N_SIGNALS, :N_SAMPLES[kingdom]]
        time[kingdom] = time_ii[:N_SAMPLES[kingdom]]

        # load spectra
        spectra[kingdom] = np.load(f"data/spectra/spectra_{kingdom}.npy")
        freqs[kingdom] = np.load(f"data/spectra/freqs_{kingdom}.npy")

    # load spectral parameters
    params = pd.read_csv("data/results/spectral_parameters.csv")


    # STATS ####################################################################
    print("\nRunning ANOVA...")
    # exponent
    exp_anova = f_oneway(params.loc[params['kingdom']=='animalia', 'exponent'],
                         params.loc[params['kingdom']=='plantae', 'exponent'],
                         params.loc[params['kingdom']=='fungi', 'exponent'])
    print(f"\tExponent ANOVA: F={exp_anova.statistic:.2f}, p={exp_anova.pvalue:.3e}")

    # timescale
    ts_anova = f_oneway(params.loc[params['kingdom']=='animalia', 'timescale'],
                        params.loc[params['kingdom']=='plantae', 'timescale'],
                        params.loc[params['kingdom']=='fungi', 'timescale'])
    print(f"\tTimescale ANOVA: F={ts_anova.statistic:.2f}, p={ts_anova.pvalue:.3e}")

    # average parameters
    print("\nParameter means:")
    print(params.groupby('kingdom').mean()[['exponent', 'timescale']])

    print("\nParameter stds:")
    print(params.groupby('kingdom').std()[['exponent', 'timescale']])

    # PLOT #####################################################################
    print("\nPlotting...")
    
    # create figure and nested gridspec
    fig = plt.figure(figsize=[6.5, 8], constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=3, nrows=4, 
                             width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 2])
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=spec[:3, 0])
    gs_de = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=spec[3, :],
                                             width_ratios=[0.2, 1, 0.1, 1, 0.2])

    # plot subplot a
    ax_a = fig.add_subplot(gs_a[0])
    ax_a.imshow(plt.imread("data/images/kingdom_cartoon.png"))
    ax_a.axis('off')
    fig.text(0.01, 0.9, 'Fungi', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.69, 'Plantae', va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.45, 'Animalia', va='center', rotation='vertical', fontsize=12)

    # loop through kingdoms 
    for ii, kingdom in enumerate(KINGDOMS):

        # plot subplot b
        ax_b = fig.add_subplot(spec[ii, 1])
        signals_i = prep_signal_for_plotting(signals[kingdom], std=SHIFT[ii])
        ax_b.plot(time[kingdom], signals_i.T, color='k', linewidth=0.5)
        ax_b.set(ylabel='voltage (z-score)')
        ax_b.set_yticks([])
        ax_b.spines['left'].set_visible(False)
        beautify_ax(ax_b)

        # plot subplot c
        ax_c = fig.add_subplot(spec[ii, 2])
        plot_spectra(spectra[kingdom], freqs[kingdom], ax=ax_c, color='k', title='')
        ax_c.set_xlabel('')
        beautify_ax(ax_c)

        # labels
        if ii == 0:
            ax_b.set_title('Voltage time-series')
            ax_c.set_title('Power spectra')
            ax_c.set_ylabel('power ($mV^2/Hz$)') # fungi data in mV 

        if ii == 2:
            ax_b.set_xlabel('time (s)')
            ax_c.set_xlabel('frequency (Hz)')

    # plot subplot d
    ax_d = fig.add_subplot(gs_de[1])
    sns.boxplot(data=params, x='kingdom', y='exponent', ax=ax_d, color='gray')
    labels =[label.get_text().capitalize() for label in ax_d.get_xticklabels()]
    ax_d.set_xticks(ax_d.get_xticks(), labels)
    ax_d.set(xlabel="kingdom", ylabel="exponent")
    ax_d.set_title('Exponent')

    # plot subplot e
    ax_e = fig.add_subplot(gs_de[3])
    sns.boxplot(data=params, x='kingdom', y='timescale', ax=ax_e, color='gray')
    labels =[label.get_text().capitalize() for label in ax_e.get_xticklabels()]
    ax_e.set_xticks(ax_e.get_xticks(), labels)
    ax_e.set(xlabel="kingdom", ylabel="timescale")
    ax_e.set_title('Timescale')
    ax_e.set_ylabel('timescale (s)')
    ax_e.set_yscale('log')

    # beautify
    for ax in [ax_d, ax_e]:
        beautify_ax(ax)

    # add panel labels
    fig.text(0.28, 0.98, 'a', fontsize=12, fontweight='bold')
    fig.text(0.68, 0.98, 'b', fontsize=12, fontweight='bold')
    fig.text(0.11, 0.31, 'c', fontsize=12, fontweight='bold')
    fig.text(0.57, 0.31, 'd', fontsize=12, fontweight='bold')

    # save figure
    fig.savefig(f"{dir_output}/figure_4.png")

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def prep_signal_for_plotting(signals, std):
    """
    Subtract mean, z-score and vertically shift signals for plotting.

    Parameters
    ----------
    signals : 2D array
        Signal to shift.
    std : int
        Number of standard deviations to shift.

    Returns
    -------
    signals : 2D array
        Shifted signal.
    """

    signals = zscore(signals)
    signals = shift_signals(signals, std=std)

    return signals


if __name__ == "__main__":
    main()
