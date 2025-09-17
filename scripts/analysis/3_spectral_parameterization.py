"""
3_spectral_parameterization.py
===================
Compute spectral parameters.

"""

# imports - standard
import os
import numpy as np
import pandas as pd

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from analysis import parameterize_spectra
from info import KINGDOMS
from settings import SPECPARAM_SETTINGS, N_JOBS, FREQ_RANGE


def main():

    # display progress
    t_start = get_start_time()
    print("\nSet-up complete!")

    # identify / create directories
    dir_output = f"data/results"
    os.makedirs(dir_output, exist_ok=True)

    # load data
    df_list = []
    for kingdom in KINGDOMS:
        print(f"\nProcessing: {kingdom}")
        
        # load data
        spectra = np.load(f"data/spectra/spectra_{kingdom}.npy")
        freqs = np.load(f"data/spectra/freqs_{kingdom}.npy")

        # parameterize spectra
        results = parameterize_spectra(spectra, freqs, n_jobs=N_JOBS,
                                       freq_range=FREQ_RANGE[kingdom],
                                       specparam_settings=SPECPARAM_SETTINGS)

        # store results
        results.insert(0, 'kingdom', kingdom)
        results.insert(1, 'channel_idx', np.arange(len(results)))
        df_list.append(results)

    # save data
    df = pd.concat(df_list, ignore_index=True)
    fname = f"{dir_output}/spectral_parameters.csv"
    df.to_csv(fname, index=False)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
