"""
2_spectral_analysis.py
===================
Compute spectral power.

"""

# imports - standard
import os
import numpy as np
from neurodsp.spectral import compute_spectrum

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import KINGDOMS, FS
from settings import NPERSEG


def main():
    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"data/spectra"
    os.makedirs(dir_output, exist_ok=True)

    # load data, compute spectra, and save
    for kindom in KINGDOMS:
        signals = np.load(f"data/epochs/{kindom}/signals.npy")
        freqs, spectra = compute_spectrum(signals, FS[kindom], nperseg=NPERSEG)

        np.save(f"{dir_output}/spectra_{kindom}.npy", spectra)
        np.save(f"{dir_output}/freqs_{kindom}.npy", freqs)

    # repeat for control recordings
    signals = np.load(f"data/epochs/fungi/control.npy")
    freqs, spectra = compute_spectrum(signals, FS['fungi'], nperseg=NPERSEG)
    np.save(f"{dir_output}/spectra_control.npy", spectra)
    np.save(f"{dir_output}/freqs_control.npy", freqs)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
