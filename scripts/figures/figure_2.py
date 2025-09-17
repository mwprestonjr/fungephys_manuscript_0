"""
Figure 1: Electrophysiological recording from fungal mycelial network.
"""

# imports - standard
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from statannotations.Annotator import Annotator
from specparam import SpectralModel, SpectralGroupModel
from scipy.stats import ttest_ind

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import FS
from utils import beautify_ax

# settings
plt.style.use('mplstyle/default.mplstyle')
COLORS = ['#9467bd', '#8c564b']

def main():

    # display progress
    print("\n\nGenerating Figure 1...")
    t_start = get_start_time()

    # identify / create directories
    dir_output = "figures/main_figures"
    os.makedirs(dir_output, exist_ok=True)

    # load data: time-series
    signals_f = np.load('data/epochs/fungi/signals.npy')
    signals_c = np.load('data/epochs/fungi/control.npy')
    time = np.load('data/epochs/fungi/time.npy')

    # z-score normalize time-series for plotting
    signals_fn = _zscore(signals_f)
    signals_cn = _zscore(signals_c)

    # load data: spectra
    spectra_f = np.load('data/spectra/spectra_fungi.npy')
    spectra_c = np.load('data/spectra/spectra_control.npy')
    freqs = np.load('data/spectra/freqs_fungi.npy')

    # apply specparam
    sgm_fk = SpectralGroupModel(aperiodic_mode='knee', max_n_peaks=0)
    sgm_fk.fit(freqs, spectra_f)

    sgm_ff = SpectralGroupModel(aperiodic_mode='fixed', max_n_peaks=0)
    sgm_ff.fit(freqs, spectra_f, freq_range=[.2, 5])

    sgm_cf = SpectralGroupModel(aperiodic_mode='fixed', max_n_peaks=0)
    sgm_cf.fit(freqs, spectra_c, freq_range=[.2, 5])

    # parameterize grand average spectra (for plotting)
    sgm_fk_ga = SpectralModel(aperiodic_mode='knee', max_n_peaks=0)
    sgm_fk_ga.fit(freqs, np.mean(spectra_f, axis=0))

    sgm_ff_ga = SpectralModel(aperiodic_mode='fixed', max_n_peaks=0)
    sgm_ff_ga.fit(freqs, np.mean(spectra_f, axis=0), freq_range=[.2, 5])

    sgm_cf_ga = SpectralModel(aperiodic_mode='fixed', max_n_peaks=0)
    sgm_cf_ga.fit(freqs, np.mean(spectra_c, axis=0), freq_range=[.2, 5])

    # print results
    print_results(sgm_fk, sgm_ff, sgm_cf)

    # plot figure
    plot_figure(time, signals_fn, signals_cn, freqs, spectra_f, spectra_c,
                sgm_fk, sgm_ff, sgm_cf, sgm_fk_ga, sgm_ff_ga, sgm_cf_ga)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


def _zscore(signals):
    signal_mean = np.mean(signals, axis=1, keepdims=True)
    signal_std = np.std(signals, axis=1, keepdims=True)

    return (signals - signal_mean) / signal_std


def print_results(sgm_fk, sgm_ff, sgm_cf):
    # print mean and std of exponent
    print("Spectra exponent (mean ± std):")
    print("\tFungi (knee):", np.mean(sgm_fk.get_params('aperiodic', 'exponent')), 
          np.std(sgm_fk.get_params('aperiodic', 'exponent')))
    print("\tFungi (fixed):", np.mean(sgm_ff.get_params('aperiodic', 'exponent')), 
          np.std(sgm_ff.get_params('aperiodic', 'exponent')))
    print("\tControl (fixed):", np.mean(sgm_cf.get_params('aperiodic', 'exponent')), 
          np.std(sgm_cf.get_params('aperiodic', 'exponent')))

    # print r-squared
    print("R-squared:")
    print("\tFungi (knee):", np.mean(sgm_fk.get_params('r_squared')))
    print("\tFungi (fixed):", np.mean(sgm_ff.get_params('r_squared')))
    print("\tControl (fixed):", np.mean(sgm_cf.get_params('r_squared')))

    # print ttest results
    ttest_ff = ttest_ind(sgm_ff.get_params('aperiodic', 'exponent'), 
                         sgm_cf.get_params('aperiodic', 'exponent'))
    ttest_fk = ttest_ind(sgm_fk.get_params('aperiodic', 'exponent'), 
                         sgm_cf.get_params('aperiodic', 'exponent'))
    print("T-test results, fungi (fixed) vs Control (fixed):")
    print(f"\tp={ttest_ff.pvalue} \n\tt={ttest_ff.statistic}")
    print("T-test results, fungi (knee) vs Control (fixed):")
    print(f"\tp={ttest_fk.pvalue} \n\tt={ttest_fk.statistic}")


def plot_figure(time, signals_fn, signals_cn, freqs, spectra_f, spectra_c,
                sgm_fk, sgm_ff, sgm_cf, sgm_fk_ga, sgm_ff_ga, sgm_cf_ga):

    # create figure and gridspec
    fig = plt.figure(figsize=[6.5, 6], constrained_layout=True)
    spec = gridspec.GridSpec(figure=fig, ncols=2, nrows=4, width_ratios=[2.25,1], 
                             height_ratios=[1,1,1,2.5])
    ax_a0 = fig.add_subplot(spec[1,0])
    ax_a1 = fig.add_subplot(spec[2,0])
    ax_b = fig.add_subplot(spec[1:3,1])
    ax_c = fig.add_subplot(spec[0,:])

    # panel a: time-series
    start_idx = 0
    n_samples =  FS['fungi'] * 60 * 60 * 1 # signals_f.shape[1] #
    time_ = np.arange(n_samples) / FS['fungi']
    for ii in range(signals_fn.shape[0]):
        ax_a0.plot(time_, signals_fn[ii, start_idx:start_idx+n_samples]+(6*ii), 
                linewidth=0.5, color=COLORS[0])

    for ii in range(signals_cn.shape[0]):
        ax_a1.plot(time_, signals_cn[ii, start_idx:start_idx+n_samples]+(6*ii), 
                linewidth=0.5, color=COLORS[1])

    for ax in (ax_a0, ax_a1):
        ax.set_yticks([])
    ax_a0.set_xticklabels([])
    ax_a0.set(ylabel="fungi")
    ax_a1.set(xlabel="time (s)\n", ylabel="control")
    for ax in (ax_a0, ax_a1):
        ax.set_xlim([time_[0], time_[-1]])

    # panel b: spectra
    plot_spectra(freqs[1:-1], spectra_f[:, 1:-1], ax_b, color=COLORS[0], label='fungi')
    plot_spectra(freqs[1:-1], spectra_c[:, 1:-1], ax_b, color=COLORS[1], label='control')
    ax_b.set(xlabel="frequency (Hz)", ylabel="power (\u03BCV^2/Hz)")
    ax_b.legend()

    # panel c: example signal
    n_samples = 6000
    ax_c.plot(time[:n_samples], signals_fn[4, :n_samples], color='k', linewidth=1)
    ax_c.set(xlabel="time (s)\n", ylabel="voltage (\u03BCV)")

    # create nested gridpec for bottom row (3 plots)
    spec = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=spec[3,:])
    ax_d = fig.add_subplot(spec[0,0])
    ax_e = fig.add_subplot(spec[0,1])
    ax_f = fig.add_subplot(spec[0,2])

    # panel d: spectra fit fungi - knee
    ax_d.loglog(freqs, np.mean(spectra_f, axis=0), color=COLORS[0], label='fungi')
    ax_d.plot(sgm_fk_ga.freqs, 10**sgm_fk_ga._ap_fit, color='k', linestyle='--', 
              label='model fit')
    ax_d.set(xlabel="frequency (Hz)", ylabel="power (\u03BCV^2/Hz)")
    ax_d.legend()

    # panel e: spectra fit - fixed
    ax_e.loglog(freqs, np.mean(spectra_f, axis=0), color=COLORS[0], label='fungi')
    ax_e.plot(sgm_ff_ga.freqs, 10**sgm_ff_ga._ap_fit, color='k', linestyle='--')
    ax_e.loglog(freqs, np.mean(spectra_c, axis=0), color=COLORS[1], label='control')
    ax_e.plot(sgm_cf_ga.freqs, 10**sgm_cf_ga._ap_fit, color='k', linestyle='--', 
              label='model fits')
    ax_e.set(xlabel="frequency (Hz)", ylabel="power (\u03BCV^2/Hz)")
    ax_e.legend()

    # panel f: exponent estimates
    data = [sgm_fk.get_params('aperiodic', 'exponent'), 
            sgm_ff.get_params('aperiodic', 'exponent'), 
            sgm_cf.get_params('aperiodic', 'exponent')]
    ax_f.boxplot(data, positions=[0,1,2], widths=0.6)
    ax_f.set_xticks([0,1,2], labels=['fungi\n(Lorentzian)', 'fungi\n(linear)', 
                                     'control\n(linear)'])
    ax_f.set_ylabel("exponent")

    # add statistical significance
    df = pd.DataFrame({
        'exponent': np.concatenate(data),
        'group': (['fungi_knee'] * len(data[0])) + (['fungi_fixed'] * len(data[1])) + (['control_fixed'] * len(data[2]))
    })
    pairs = [("fungi_fixed", "control_fixed"), ("fungi_knee", "control_fixed")]
    annot = Annotator(ax_f, pairs, data=df, x='group', y='exponent')
    annot.configure(test='t-test_ind', text_format='star', loc='inside', verbose=0)
    annot.apply_and_annotate()

    # beautify axes
    for ax in (ax_a0, ax_a1, ax_b, ax_c, ax_d, ax_e, ax_f):
        beautify_ax(ax)
    for ax in (ax_a0, ax_a1):
        ax.spines['left'].set_visible(False)

    # add panel titles
    ax_c.set_title("Example recording")
    ax_a0.set_title("Normalized voltage time-series")
    ax_b.set_title("Power spectra")
    ax_d.set_title("Lorentzian model fit")
    ax_e.set_title("Linear model fit")
    ax_f.set_title("Spectral exponent")

    # add panel labels
    fig.text(0.01, 0.97, 'a', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.74, 'b', fontsize=12, fontweight='bold')
    fig.text(0.70, 0.74, 'c', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.32, 'd', fontsize=12, fontweight='bold')
    fig.text(0.33, 0.32, 'e', fontsize=12, fontweight='bold')
    fig.text(0.66, 0.32, 'f', fontsize=12, fontweight='bold')

    # save
    fig.savefig('figures/figure_2.png')


def plot_spectra(freqs, spectra, ax, color=None, label=None):
    if color is None:
        color = 'k'
    mean_spectra = np.mean(spectra, axis=0)
    sem_spectra = np.std(spectra, axis=0) / np.sqrt(spectra.shape[0])
    if label is None:
        ax.loglog(freqs, mean_spectra, color=color)
    else:
        ax.loglog(freqs, mean_spectra, color=color, label=label)
    ax.fill_between(freqs, mean_spectra - sem_spectra, 
                    mean_spectra + sem_spectra, color=color, alpha=0.3)


if __name__ == "__main__":
    main()
