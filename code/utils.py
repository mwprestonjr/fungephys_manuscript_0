"""
Utility functions.
"""

# imports
from time import time


# Plotting utils ###############################################################

def beautify_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Time utils ###################################################################

def get_start_time():
    return time()


def print_time_elapsed(t_start, prefix=''):
    hour, min, sec = hour_min_sec(time() - t_start)
    print(f"{prefix}{hour} hour, {min} min, and {sec :0.1f} s")


def hour_min_sec(duration):
    """
    Convert duration in seconds to hours, minutes, and seconds.

    Parameters
    ----------
    duration : float
        Duration in seconds.

    Returns
    -------
    hours, mins, secs : int
        Duration in hours, minutes, and seconds.
    """

    hours = int(duration // 3600)
    mins = int(duration % 3600 // 60)
    secs = int(duration % 60)
    
    return hours, mins, secs
