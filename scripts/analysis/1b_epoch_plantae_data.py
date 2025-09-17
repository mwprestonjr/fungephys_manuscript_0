"""
1b_epoch_plantae_data.py
===================
Epoch plant data.
Load data for each individual and join into array.

"""

# imports - standard
import os
import numpy as np
import pandas as pd

# imports - custom
import sys
sys.path.append("code")
from time_utils import get_start_time, print_time_elapsed
from info import FS

# settings
IDS = ['T1P1', 'T1P4', 'T1P5', 'T1P6', 'T1P7', 'T1P9', 'T1P11', 'T1P12', 
       'T1P14', 'T1P17', 'T1P18', 'T1P20']

def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"data/epochs/plantae"
    os.makedirs(dir_output, exist_ok=True)

    # load data for day and night
    signal_list = []
    for id in IDS:
        fname = f"data/raw/plantae/{id}_antes_day.txt"
        data_in = pd.read_csv(fname, sep='\t', header=None)
        signal_list.append(data_in.iloc[:,0].values.T)
    signals = np.vstack(signal_list)

    # create time array
    time = np.arange(signals.shape[1]) / FS['plantae']

    # save data
    np.save(f"{dir_output}/signals.npy", signals)
    np.save(f"{dir_output}/time.npy", time)

    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
