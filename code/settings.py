"""
Settings.
"""

SPECPARAM_SETTINGS = {
    'peak_width_limits' :   [2, 20], # default : (0.5, 12.0) - recommends at least frequency resolution * 2
    'min_peak_height'   :   0, # default : 0
    'max_n_peaks'       :   2, # default : inf
    'peak_threshold'    :   3 # default : 2.0
}
N_JOBS = -1 # for parallelization
FREQ_RANGE = dict({'animalia': None,
                   'plantae': None,
                   'fungi': None}) # frequency range for parameterization
NPERSEG = 1024 # number of samples per segment for spectral analysis