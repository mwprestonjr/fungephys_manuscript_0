"""
utility functions for analysis.
"""

# imports
import numpy as np
from specparam import SpectralGroupModel


def parameterize_spectra(spectra, freqs, specparam_settings=None, n_jobs=1, 
                         ap_mode='knee', freq_range=None):
    
    if specparam_settings is None:
        specparam_settings = {
            'peak_width_limits' :   [2, 20], # default : (0.5, 12.0)
            'min_peak_height'   :   0, # default : 0
            'max_n_peaks'       :   4, # default : inf
            'peak_threshold'    :   3} # default : 2.0

    sgm = SpectralGroupModel(**specparam_settings, aperiodic_mode=ap_mode, 
                             verbose=False)
    sgm.fit(freqs, spectra, n_jobs=n_jobs, freq_range=freq_range)
    knee = sgm.get_params('aperiodic', 'knee')
    exponent = sgm.get_params('aperiodic', 'exponent')
    timescale = knee_to_timescale(knee, exponent)
    results = sgm.to_df(0)
    results.insert(3, 'timescale', timescale)

    return results


def knee_to_timescale(knee, exponent):
    """
    Convert specparam knee parameter to timescale.

    Parameters
    ----------
    knee, exponent : 1D array
        Knee and exponent parameters from specparam.

    Returns
    -------
    timescale : 1D array
        Timescale in seconds.
    """
    
    knee_freq = knee_to_hz(knee, exponent) # convert to Hz
    timescale = 1 / (2 * np.pi * knee_freq) # convert to seconds
        
    return timescale


def knee_to_hz(knee, exponent):
    """
    Convert specparam knee parameter to Hz.

    Parameters
    ----------
    knee, exponent : float or array_like
        Knee and exponent parameters from specparam.

    Returns
    -------
    knee_hz : float or ndarray
        Knee in Hz.
    """

    # init
    knee = np.asarray(knee)
    exponent = np.asarray(exponent)

    # Avoid invalid values by masking
    valid = knee > 0
    knee_hz = np.full_like(knee, np.nan, dtype=np.float64)
    knee_hz[valid] = knee[valid] ** (1 / exponent[valid])
    
    return knee_hz if knee_hz.ndim > 0 else knee_hz.item()