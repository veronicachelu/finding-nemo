# Load in packages
import numpy as np
import pandas as pd
import pynwb
from matplotlib import pyplot as plt
# autocorrelogram imports
from neo.core import SpikeTrain
from quantities import ms, s, Hz
from elephant.statistics import time_histogram, instantaneous_rate,  mean_firing_rate
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram

# Set the data root according to OS
import platform
from pathlib import Path

platstring = platform.platform()

if 'Darwin' in platstring:
    # macOS 
    data_root = Path("/Volumes/Brain2025/")
elif 'Windows'  in platstring:
    # Windows (replace with the drive letter of USB drive)
    data_root = Path("E:/")
elif ('amzn' in platstring):
    # then on CodeOcean
    data_root = Path("/data/")
else:
    # then your own linux platform
    # EDIT location where you mounted hard drive
    data_root = Path("/media/$USERNAME/Brain2025/")


# Load in a session from a NWB file
def process_nwb_metadata(
    path, 
    max_isi_violations=0.5, 
    max_amplitude_cutoff=0.1, 
    min_presence_ratio=0.95
):
    """
    Load a NWB file and return the session, units table, and stimuli information.
    Args:
        path (str): Path to the NWB file (excludes the data root).
        max_isi_violations (float): Maximum allowed ISI violations for units.
        max_amplitude_cutoff (float): Maximum allowed amplitude cutoff for units.
        min_presence_ratio (float): Minimum presence ratio for units.
    Returns:
        session (pynwb.NWBFile): The loaded NWB session.
        units_table (pd.DataFrame): DataFrame containing units information.
        stimuli (pd.DataFrame): DataFrame containing stimuli information.
        good_units (pd.DataFrame): DataFrame containing units that meet the QC criteria.
    """

    nwb_path = data_root / path
    session = pynwb.NWBHDF5IO(nwb_path).read()
    
    # Get stimuli information
    stimuli = session.intervals['Natural_Images_Lum_Matched_set_ophys_H_2019_presentations'].to_dataframe()

    # Get units table
    units_table = session.units.to_dataframe()
    electrodes_table = session.electrodes.to_dataframe()
    units_electrode_table = units_table.join(electrodes_table,on = 'peak_channel_id')

    # Filter units by QC criteria
    good_units = units_electrode_table[
        (units_electrode_table.isi_violations < max_isi_violations) &
        (units_electrode_table.amplitude_cutoff < max_amplitude_cutoff) &
        (units_electrode_table.presence_ratio > min_presence_ratio)
    ]
    assert len(good_units) > 0, "There are 0 units that meet the specified QC criteria in this session."

    return session, units_table, stimuli, good_units

def get_stim_window(
    spike_times,
    stim_times,
    pre_window=0.2,
    post_window=0.75,
):
    """
    Get spike counts around each stimulus.
    Args:
        spike_times (numpy.ndarray): Array of spike times from trial.
        stim_times (numpy.ndarray): Array of stimulus times from trial.
        pre_window (float): How far before the stimulus to look.
        post_window (float): How far after the stimulus to look.
    Returns:
        triggered_spike_times (numpy.ndarray): Array of stimulus-triggered spike times.
        triggered_stim_index (numpy.ndarray): Array of simului indices.
    """
    
    # Storage for data
    triggered_spike_times = []
    triggered_stim_index = []

    # Loop through the stimuli
    for i, stim_time in enumerate(stim_times):
        # Select spikes that fall within the time window around this stimulus
        mask = ((spike_times >= stim_time - pre_window) & 
                (spike_times < stim_time + post_window))
        
        # Align spike times to stimulus onset (0 = stimulus)
        trial_spikes = spike_times[mask] - stim_time

        triggered_spike_times.append(trial_spikes)
        triggered_stim_index.append(np.ones(len(trial_spikes))*i)

    # For plotting, we are going to want to concatenate these data into one big vector
    triggered_spike_times = np.concatenate(triggered_spike_times)
    triggered_stim_index = np.concatenate(triggered_stim_index)

    return triggered_spike_times, triggered_stim_index

def get_spike_counts_all(units_table, stim_times, bins):
    n_neurons = len(units_table.spike_times.values)
    spike_counts = np.empty((n_neurons, len(stim_times), len(bins)-1), dtype=int)
    for nn in range(n_neurons):
        spike_counts[nn, :, :] = get_binned_triggered_spike_counts_fast(
            units_table.spike_times.values[nn], stim_times, bins
        )
    return spike_counts


def get_spike_counts(
    spike_times,
    stim_times,
    stimuli,
    start=0,
    stop=0.35
):
    """
    Get spike counts in a window around each stimulus.
    Args:
        spike_times (numpy.ndarray): Array of spike times from trial.
        stim_times (numpy.ndarray): Array of stimulus times from trial.
        stimuli (pd.DataFrame): DataFrame containing stimuli information.
        start (float): How far before the stimulus to look.
        stop (float): How far after the stimulus to look.
    Returns:
        spike_count (numpy.ndarray): Array of spike counts within time window
        trial_id (numpy.ndarray): Indices of each unique stimuli
    """
    
    spike_count = []

    for i, stim_time in enumerate(stim_times):
        # Select spikes that fall within the time window around this stimulus
        mask = ((spike_times >= stim_time + start) & 
                (spike_times < stim_time + stop))
        
        # Count spikes in this bin
        spike_count.append(len(spike_times[mask]))
        
    spike_count = np.array(spike_count)
    _, trial_id = np.unique(stimuli.image_name.values, return_inverse= True)

    return spike_count, trial_id

# For poopulation decoding
def get_binned_triggered_spike_counts_fast(
    spike_times, 
    stim_times, 
    bins
):
    """
    Fast peri-stimulus time histogram using searchsorted.
    Args:
        spike_times (numpy.ndarray): Times of all spikes (e.g. in seconds).
        stim_times (numpy.ndarray): Times of stimulus onsets.
        bins (numpy.ndarray): Bin edges *relative* to stimulus
    Returns:
        counts (numpy.ndarray): Array of spike counts within time window, shape (n_trials, len(bins)-1)
    """
    # ensure numpy arrays
    spike_times = np.asarray(spike_times)
    stim_times = np.asarray(stim_times)
    bins = np.asarray(bins)

    spike_times = np.sort(spike_times)

    n_trials = stim_times.size
    n_bins = bins.size - 1
    counts = np.zeros((n_trials, n_bins), dtype=int)

    for i, stim in enumerate(stim_times):
        # compute the absolute edges for this trial
        edges = stim + bins
        # find the insertion indices for each edge
        idx = np.searchsorted(spike_times, edges, side='left')
        # differences between successive indices = counts per bin
        counts[i, :] = np.diff(idx)

    return counts

def apply_zscore(rates, axis=1):
    """
    Z-score firing rates.
    Args:
        rates (numpy.ndarray): Firing rates, shape (n_neurons, n_time_points)
        axis: Axis of rates across which statics are to be computed, default: 1 (across time points)
    Returns:
        rates_zscored (numpy.ndarray): Z-scored firing rates
    """
    return (rates - np.nanmean(rates, axis=axis, keepdims=True)) / np.nanstd(rates, axis=axis, keepdims=True)


def get_multi_area_firing_rates(area_packets,event_times,time_before_change=1.0,duration=2.5,bin_size=0.01):
    """
    Compute firing rates.
    Args:
        rates (numpy.ndarray): Firing rates, shape (n_neurons, n_time_points)
        axis: Axis of rates across which statics are to be computed, default: 1 (across time points)
    Returns:
        rates_zscored (numpy.ndarray): Z-scored firing rates
    """
    
    bins = np.arange(0, duration + bin_size, bin_size)
    
    results = []
    for r, pkt in enumerate(area_packets):
        name       = pkt['name']
        spikes_all = pkt['spikes']
        ridx       = int(np.clip(pkt.get('raster_idx', 0), 0, max(0, len(spikes_all)-1)))

        start_times = event_times - time_before_change  # align so x=0 is the event
        psths = []
        for st in spikes_all:
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)  # (trials, bins-1)
            rates = trial_counts.mean(axis=0) / bin_size  # Hz
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (units, bins-1)
    
        results.append({'area': name, 'pop_rates': pop_rates, 'bins': bins-time_before_change})
    
    return results


def build_trials_by(df, key_col, start_col="start_time", na_fill=None):
    t = df.loc[:, [start_col, key_col]].copy()
    if na_fill is not None:
        t[key_col] = t[key_col].fillna(na_fill)
    t = (t.rename(columns={key_col: "key"})
           .sort_values(["key", start_col], kind="mergesort")
           .reset_index(drop=True))

    codes, uniques = pd.factorize(t["key"])
    cmap = plt.get_cmap("tab10" if len(uniques) <= 10 else "tab20")
    color_map = {u: cmap(i % cmap.N) for i, u in enumerate(uniques)}
    trial_colors = t["key"].map(color_map).values

    k = t["key"].to_numpy()
    change_rows = np.flatnonzero(k[1:] != k[:-1]) + 1
    block_starts = np.r_[0, change_rows]
    block_ends = np.r_[change_rows, len(t)]
    block_mids = (block_starts + block_ends) / 2
    block_names = t["key"].iloc[block_starts].to_numpy()

    return t, trial_colors, change_rows, block_mids, block_names

def make_psth(spikes, start_times, window_dur, bin_size=0.001):
    # bins are absolute relative to each event (0..window)
    bins = np.arange(0, window_dur + bin_size, bin_size)
    # trial x bin counts
    trial_counts = get_binned_triggered_spike_counts_fast(spikes, start_times, bins)
    # average over trials, convert to Hz
    rates = trial_counts.mean(axis=0) / bin_size
    return rates, bins

def compute_population_psth(spike_times_by_unit, start_times, window_dur, bin_size=0.01):
    """
    Compute PSTHs for a population of units.
    Args:
        spike_times_by_unit (list of 1D arrays): spike times for each unit.
        start_times (1D array): event onset times.
        window_dur (float): seconds.
        bin_size (float): seconds.
    Returns:
        pop_rates (2D array): shape (n_units, n_bins).
        bins (1D array): bin edges (0..window_dur).
    """
    psths = []
    bins_ref = None
    for spikes in spike_times_by_unit:
        rates, bins = make_psth(spikes, start_times, window_dur, bin_size)
        psths.append(rates)
        bins_ref = bins if bins_ref is None else bins_ref
    return np.asarray(psths), bins_ref

def align_spikes_to_events(spikes, event_times, t_pre, t_post):
    """
    Align spikes to each event within [ -t_pre, +t_post ].
    Args:
        spikes (1D array): sorted spike times.
        event_times (1D array): event onsets.
        t_pre (float): seconds before event (positive).
        t_post (float): seconds after event (positive).
    Returns:
        aligned (list of 1D arrays): for each event, spike times relative to event.
    """
    aligned = []
    for t0 in event_times:
        start = t0 - t_pre
        stop  = t0 + t_post
        i0 = np.searchsorted(spikes, start)
        i1 = np.searchsorted(spikes, stop)
        aligned.append(spikes[i0:i1] - t0)
    return aligned


def build_area_packet(area_name, good_units_df, spike_times_all_units,
                      area_col='structure_acronym', sort_key=None, raster_unit_idx=0):
    area_units = good_units_df[good_units_df[area_col] == area_name]
    spike_times_by_unit = [spike_times_all_units[iu] for iu, _ in area_units.iterrows()]
    if sort_key is not None and sort_key in area_units.columns:
        order = np.argsort(area_units[sort_key].values)
        spike_times_by_unit = [spike_times_by_unit[i] for i in order]
    return dict(name=area_name, spikes=spike_times_by_unit, raster_idx=raster_unit_idx)


def _normalize_rates(X, method='zscore', axis=1, eps=1e-9):
    """
    Normalize firing-rate matrix X (units x time) along `axis`.
    method: 'zscore' | 'minmax' | 'none'
    """
    if method == 'none':
        return X
    X = np.asarray(X, float)
    if method == 'zscore':
        mu = X.mean(axis=axis, keepdims=True)
        sd = X.std(axis=axis, keepdims=True) + eps
        return (X - mu) / sd
    elif method == 'minmax':
        mn = X.min(axis=axis, keepdims=True)
        mx = X.max(axis=axis, keepdims=True)
        return (X - mn) / (mx - mn + eps)
    else:
        raise ValueError(f"unknown method: {method}")

def compute_unit_covariance(pop_rates, normalize='zscore'):
    """
    pop_rates: (n_units, n_bins) firing rates (Hz)
    Returns: (n_units, n_units) covariance matrix across time.
    """
    X = _normalize_rates(pop_rates, method=normalize, axis=1)
    # np.cov expects variables as rows (units) and observations as columns (time)
    # pop_rates is (units x time) already, so rowvar=True by default.
    return np.cov(X)


def compute_unit_covariance(pop_rates, normalize='zscore'):
    """
    pop_rates: (units, timebins) mean trial-binned firing rates per unit.
    normalize: None | 'zscore' | 'center' | 'l2'
    returns: (units, units) covariance
    """
    X = np.asarray(pop_rates)
    if X.ndim != 2:
        raise ValueError("pop_rates must be 2D (units x timebins)")
    if normalize == 'zscore':
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-9
        X = (X - mu) / sd
    elif normalize == 'center':
        X = X - X.mean(axis=1, keepdims=True)
    elif normalize == 'l2':
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    elif normalize is not None:
        raise ValueError(f"Unknown normalize '{normalize}'")
    return np.cov(X)

def make_group(df, key_col, *, label=None, start_col="start_time", t_pre=1., t_post=2.0, bin_size=0.1,
        smooth_kind="gaussian", smooth_value=0.5, na_fill=None, transform=None):
    gdf = df.copy()
    if transform is not None:
        gdf = transform(gdf)
    trials, colors, sep, mids, names = build_trials_by(gdf, key_col, start_col=start_col, na_fill=na_fill)
    return dict(trials=trials, colors=colors, sep=sep, mids=mids, names=names, label=(label or key_col), start_col=start_col,
                t_pre=t_pre, t_post=t_post, bin_size=bin_size, smooth_kind=smooth_kind, smooth_value=smooth_value)


def generate_autocorr_data(spiketrain, bin_ms, win_ms):
    win_bins = int(round(win_ms / bin_ms))
    bst = BinnedSpikeTrain([spiketrain], bin_size=bin_ms * ms)
    ac_sig, lag_bins = cross_correlation_histogram(bst, bst, window=[-win_bins, win_bins])
    ac = ac_sig.magnitude.flatten()
    lags = np.asarray(lag_bins, dtype=int)
    keep = lags != 0
    return lags[keep] * bin_ms, ac[keep]
