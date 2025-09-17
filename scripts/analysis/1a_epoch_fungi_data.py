"""
1a_epoch_fungi_data.py
===================
Epoch fungi data.
Load data for each individual and join into array.

"""

# imports - standard
import os
import numpy as np

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import FS
from pico_utils import import_data

# settings
N_SAMPLES = 60 * 60 * 8 * FS['fungi'] + 1 # 8 hours
DRIFT_WINDOW_SIZE = 1000 # window size for drift removal

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"data/epochs/fungi"
    os.makedirs(dir_output, exist_ok=True)

    # epoch fungi data
    fnames = [f for f in os.listdir("data/raw/fungi/fungi") if f.endswith('.csv')]
    signal_list = []
    for i, fname in enumerate(fnames):
        signal_i, _ = import_data(f"data/raw/fungi/fungi/{fname}", 
                                 n_samples=N_SAMPLES,
                                 drift_window_size=DRIFT_WINDOW_SIZE)
        signal_list.append(signal_i)
    signals = np.vstack(signal_list)

    # epoch control data
    fnames = [f for f in os.listdir("data/raw/fungi/control") if f.endswith('.csv')]
    control_list = []
    for i, fname in enumerate(fnames):
        control_i, _ = import_data(f"data/raw/fungi/control/{fname}", 
                                   n_samples=N_SAMPLES,
                                   drift_window_size=DRIFT_WINDOW_SIZE)
        control_list.append(control_i)
    control = np.vstack(control_list)

    # create time array
    time = np.arange(N_SAMPLES) / FS['fungi']

    # save data
    np.save(f"{dir_output}/signals.npy", signals)
    np.save(f"{dir_output}/control.npy", control)
    np.save(f"{dir_output}/time.npy", time)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
