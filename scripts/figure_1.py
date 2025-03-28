"""
Figure 1: Electrophysiological recording from fungal mycelial network.
"""

# imports - standard
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
from neurodsp.spectral import compute_spectrum
from matplotlib.gridspec import GridSpec

# imports - custom
import sys
sys.path.append("code")
from info import FS
from utils import get_start_time, print_time_elapsed, beautify_ax

# settings
SPECIES = 'herecium' # species to plot
NPERSEG = 64 # samples per segment for fft
N_SAMPLES = 3000 # number of samples to plot
COLORS = [sns.color_palette('colorblind')[0], 'grey'] # [mycelium, control]
plt.style.use('mplstyle/default.mplstyle')


def main():

    # display progress
    print("\n\nGenerating Figure 1...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = "figures"
    if not os.path.exists(dir_output): 
        os.makedirs(dir_output)

    # load data
    df = pd.read_csv('data/recordings.csv')

    # compute power spectrum
    freqs, psd = compute_spectrum(df[SPECIES], FS, nperseg=NPERSEG)
    _, psd_ctrl = compute_spectrum(df['control'], FS, nperseg=NPERSEG)

    # create figure
    fig = plt.figure(figsize=(6.5, 2.34), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.6], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # A: plot time-series
    titles = ('Mycelial network', 'Control')
    for i, (column, title, ax) in enumerate(zip(df.columns[1:], titles, 
                                                [ax1, ax2])):
        ax.plot(df['time'][:N_SAMPLES], df[column][:N_SAMPLES], c=COLORS[i])
        ax.set_ylabel("voltage (uV)")
        ax.set_title(title)
    ax2.set_xlabel('time (s)')

    # B: plot power spectrum
    ax3.loglog(freqs, psd, color=COLORS[0], lw=2, label='mycelium')
    ax3.loglog(freqs, psd_ctrl, color=COLORS[1], lw=2, label='control')
    ax3.set(xlabel= 'frequency (Hz)', ylabel='power (\u03BCV\u00b2/Hz)')
    ax3.set_title('Power spectrum')
    ax3.legend()

    # add panel labels
    fig.text(0.01, 0.95, 'a', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.50, 'b', fontsize=12, fontweight='bold')
    fig.text(0.62, 0.95, 'c', fontsize=12, fontweight='bold')

    # beautify axes
    for ax in [ax1, ax2, ax3]:
        beautify_ax(ax)

    # save
    fig.savefig('figures/figure_1.png')

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
