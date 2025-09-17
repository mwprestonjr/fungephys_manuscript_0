"""
Plotting functions.

"""

# impots
import numpy as np
import matplotlib.pyplot as plt


def beautify_ax(ax):
    """
    Beautify axis by removing top and right spines.
    """
    
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_spectra(spectra, freqs, ax=None, shade_sem=True, plot_each=False,
                 y_units='\u03BCV\u00b2/Hz', title=None, fname=None, **kwargs):
    
    """
    Plot power spectra. 

    Parameters
    ----------
    spectra : 2d array
        Power spectra [n_spectra x n_frequencies].
    freqs : 1d array
        Frequency values corresponding to PSD values.
    ax : matplotlib axis, optional
        Axis to plot on. The default is None.
    shade_sem : bool, optional
        Whether to shade SEM. The default is True.
    plot_each : bool, options
        Whether to plot each spectra. The default is False.
    y_units : str, optional
        Units for y-axis. The default is '\u03BCV\u00b2/Hz' (microvolts).
    title : str, optional
        Title for plot. The default is None.
    fname : str, optional
        File name to save figure. The default is None.
    **kwargs : dict
        Additional keyword arguments for plotting.

    Returns
    -------
    fig, ax : matplotlib figure and axis
        Figure and axis.
    """

    # check axis
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=[6,4])

    # check psds are 2d
    if not spectra.ndim == 2:
        raise ValueError('PSD must be 2d arrays.')
    
    # remove rows containing all nans
    spectra = spectra[~np.isnan(spectra).all(axis=1)]

    # plot mean spectra for each condition
    ax.loglog(freqs, np.mean(spectra, axis=0), **kwargs)
    
    # shade between SEM of spectra for each condition
    if shade_sem and not plot_each:
        lower = np.mean(spectra, 0) - (np.std(spectra, 0)/np.sqrt(spectra.shape[0]))
        upper = np.mean(spectra, 0) + (np.std(spectra, 0)/np.sqrt(spectra.shape[0]))
        ax.fill_between(freqs, lower, upper, alpha=0.5, edgecolor=None, **kwargs)

    if plot_each:
        for i_spec in range(len(spectra)):
            ax.loglog(freqs, spectra[i_spec], alpha=0.5)

    # set axes ticks and labels
    ax.set_ylabel(f'power ({y_units})')
    ax.set_xlabel('frequency (Hz)')

    # add title
    if title is None:
        ax.set_title('Power spectra')
    else:
        ax.set_title(title)

    # add grid
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5)
    
    # return
    if fname is not None:
        plt.savefig(fname)


