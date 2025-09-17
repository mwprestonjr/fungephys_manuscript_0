



# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from time_utils import convert_seconds


def import_data(fname, n_samples=None, interp_nan=True, return_labels=False,
                drift_window_size=None, subtract_mean=False, 
                highpass_cutoff=None, verbose=True):
    """
    Import time series data from CSV file; writen for the output of PicoScope.
    The first column is assumed to be time, and subsequent columns are assumed 
    to be signal channels. By default, NaN values are interpolated linearly.
    There are a number of options for removing drift from the signals including
    high-pass filtering (e.g., 0.01 Hz) and a rolling mean subtraction.
    The return_labels flag can be set to return the channel labels.

    Parameters
    ----------
    fname : str
        Path to CSV file.
    n_samples : int, optional, default: None
        Number of samples to read. If None, the entire file is read.
    interp_nan : bool, optional, default: True
        Interpolate NaN values linearly.
    return_labels : bool, optional, default: False
        Return channel labels.
    end_time : float, optional, default: None
        End time in seconds. If None, the entire signal is used.
    drift_window_size : int, optional, default: None
        Window size for rolling mean to remove drift. If None, no drift removal
        is performed.
    subtract_mean : bool, optional, default: False
        Subtract mean from each signal.
    highpass_cutoff : float, optional, default: None
        High-pass filter cutoff freq in Hz. If None, no filtering is performed.
    verbose : bool, optional, default: True
        Print verbose output.

    Returns
    -------
    signals : 2d array
        Array of shape (n_channels, n_samples) containing the imported signals.
    time : 1d array
        Array of shape (n_samples,) containing the time vector.
    labels : list
        List of channel labels.

    Notes
    -----
    The rolling mean is computed using a convolution with a window of ones.
    This introduces edge effects, so the output signals are shorter than the
    input signals by drift_window_size - 1 samples.

    The high-pass filter is a zero-phase Butterworth filter applied using
    sosfiltfilt. The time vector is assumed to be evenly spaced, which is
    true for PicoScope data.

    Example
    -------
    signals, time = import_data('data.csv')
    signals, time, labels = import_data('data.csv', return_labels=True)

    """

    # import data
    if n_samples is None:
        df = pd.read_csv(fname)
    else:
        df = pd.read_csv(fname, nrows=n_samples)

    # convert HH:MM:SS to seconds
    df = df.rename(columns={df.columns[0]: 'time'})
    df['time'] = pd.to_timedelta(df['time'])
    df['time'] = pd.to_timedelta(df['time'].astype(str)).dt.total_seconds()

    # print info
    if verbose:
        total_time = df['time'].iloc[-1] - df['time'].iloc[0]
        day, hour, min, sec = convert_seconds(total_time)
        print(f"\tFilename: {fname}")
        print(f"\tDuration: {day} days, {hour} hours, {min} minutes, {sec} seconds")
        print(f"\tColumns: {df.columns.to_list()}")

    # interpolate NaN values
    if interp_nan:
        if verbose:
            n_nan = df.isna().sum().sum()
            total_values = df.shape[0] * df.shape[1]
            perc = n_nan / total_values * 100
            print(f"\tCleaning: {n_nan} NaN values were interpolated ({perc:0.0f}%)")

        df = df.interpolate(method='linear')

    # convert to numpy
    signals = df.iloc[:, 1:].to_numpy().T
    time = df['time'].to_numpy()
    labels = df.columns[1:]

    # remove drift by subtracting rolling average
    if drift_window_size is not None:
        signals = subtract_rolling_mean(signals, window_size=drift_window_size)
        time = time[:signals.shape[1]]
        if verbose:
            print(f"\tRemoving drift: rolling mean over {drift_window_size} samples")

    # high-pass filter to remove slow drifts
    if highpass_cutoff is not None:
        fs = 1 / (time[1] - time[0])
        signals = highpass_butter(signals, fs, cutoff=highpass_cutoff)
        if verbose:
            print(f"\tRemoving drift: highpass filter with cutoff at {highpass_cutoff} Hz")

    # subtract mean
    if subtract_mean:
        signals = signals - signals.mean(axis=1, keepdims=True)

    # return
    if return_labels:
        return signals, time, labels
    else:
        return signals, time
    

def subtract_rolling_mean(signals, window_size=1000):
    """Remove drift from signals by subtracting rolling mean."""

    ws = window_size
    signals_detrended = np.zeros((signals.shape[0], signals.shape[1] - ws + 1))
    for i in range(signals.shape[0]):
        rolling_mean = np.convolve(signals[i, :], np.ones(ws)/ws, mode='valid')
        signals_detrended[i, :] = signals[i, ws//2: -ws//2 + 1] - rolling_mean

    return signals_detrended


def highpass_butter(data, fs, cutoff=0.01, order=4, axis=-1):
    """
    Zero-phase Butterworth high-pass using sosfiltfilt.

    Parameters
    ----------
    data : np.ndarray
        Signal(s) to filter. Can be 1D or ND; filtering is along `axis`.
    fs : float
        Sampling rate in Hz.
    cutoff : float, optional
        High-pass cutoff in Hz (e.g., 0.01–0.05 Hz for drift removal).
    order : int, optional
        Butterworth order (4 is a good default).
    axis : int, optional
        Axis along which to filter.

    Returns
    -------
    np.ndarray
        Filtered data, same shape as input.
    """

    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    data = sosfiltfilt(sos, data, axis=axis)

    return data

