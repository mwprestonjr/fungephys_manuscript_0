"""
1c_epoch_animalia_data.py
===================
Epoch animal data.
Load data for each individual and join into array.

"""

# imports - standard
import os
from mne.io import read_raw_edf
import numpy as np

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import FS

# settings


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_input = "data/raw/animalia/Wakefulness_AllRegions"
    dir_output = f"data/epochs/animalia"
    os.makedirs(dir_output, exist_ok=True)

    # load data
    signal_list = []
    fnames = [f for f in os.listdir(dir_input) if f.endswith('.edf')]
    for fname in fnames:
        signal_list.append(read_raw_edf(f"{dir_input}/{fname}").get_data())
    signals = np.vstack(signal_list)

    # create time array
    time = np.arange(signals.shape[1]) / FS['animalia']

    # save data
    np.save(f"{dir_output}/signals.npy", signals)
    np.save(f"{dir_output}/time.npy", time)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
