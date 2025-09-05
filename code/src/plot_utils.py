import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.collections import LineCollection 
import data_utils
import importlib
importlib.reload(data_utils)
from data_utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def create_raster(
    spike_times,
    stim_index,
    ax,
    size=1,
    color='black'
):
    """
    Create a raster plot.
    Args:
        spike_times (numpy.ndarray): Times of all spikes (e.g. in seconds).
        stim_index (numpy.ndarray): Times of stimulus onsets.
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        size (int): Size of scatter plot point.
        color (string): Color of scatter plot point.
    """
    ax.scatter(spike_times, stim_index, s=size,c=color)
    ax.set_xlabel('Time from stimulus (seconds)')
    ax.set_ylabel('Stim number (sorted)')
    ax.axvline([0],c = 'r')

def create_psth(
    spike_times,
    stim_index,
    ax,
    pre_window=0.2,     # How far before the stimulus should we look?
    post_window=0.75,   # How far after the stimulus should we look?
    bin_size=0.01,      # What size bins do we want for our PSTH?
    color='black',
    label=None
):
    """
    Create a peristimulus time histogram.
    Args:
        spike_times (numpy.ndarray): Times of all spikes (e.g. in seconds).
        stim_index (numpy.ndarray): Times of stimulus onsets.
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        pre_window (float): How far before the stimulus to look.
        post_window (float): How far after the stimulus to look.
        bin_size (float): Size of the bins for the PSTH.
        color (string): Color of scatter plot point.
        label (string): Label for plot group if generating a legend.
    """
    # Set up bins
    bins = np.arange(-pre_window,post_window+bin_size,bin_size) 
    bin_centers = bins[:-1] + bin_size/2  

    a, _ = np.histogram(spike_times, bins=bins)

    # Divide by # of trials, then bin size to get a rate estimate in Spikes/Sec = Hz
    a = a/np.max(stim_index)/bin_size
    ax.plot(bin_centers, a, c=color, label=label)
    ax.set_xlabel('Time from stimulus (seconds)')
    ax.set_ylabel('Spike Rate (Hz)')

def create_confusion_matrix(
    ax,
    y_pred,
    y_test,
):
    """
    Create a confusion matrix given predictions from a classifier.
    Args:
        ax (matplotlib.axes.Axes): Axes on which to generate plot.
        y_pred (numpy.ndarray): Predicted classes for each test datapoint.
        y_test (numpy.ndarray): Target outputs for test data.
    """ 
    im  = ax.imshow(confusion_matrix(y_true=y_test, y_pred=y_pred, normalize='pred'))
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    cbar = plt.colorbar(im)
    cbar.set_label('Fraction Guessed')
    

def plot_population_heatmap(pop_rates, bins, ax, pre_time, clim_percentiles=(0.1, 99.9), cmap='viridis'):
    """
    Heatmap of population PSTH (units x time).
    Args:
        pop_rates (2D array): (n_units, n_bins)
        bins (1D array): bin edges (0..window_dur)
        ax (Axes): target axes
        pre_time (float): seconds subtracted to center time axis at event (negative before event)
        clim_percentiles (tuple): robust color limits percentiles
        cmap (str): colormap
    """
    clims = [np.percentile(pop_rates, p) for p in clim_percentiles]
    im = ax.imshow(pop_rates, aspect='auto', cmap=cmap, vmin=clims[0], vmax=clims[1])
    # Mark event onset (time = 0) on heatmap
    # event_time = 0  # in seconds, relative to alignment
    # Find nearest bin index
    event_bin = np.argmin(np.abs(bins[:-1] - pre_time))  
    ax.axvline(event_bin, color='red', linestyle='--', lw=1.5, label='Event')

    # Optional: show a legend
    # ax.legend(loc='upper right')
    
    ax.set_title('Population PSTH (units × time)')
    ax.set_ylabel('Unit # (e.g., sorted by depth)')
    ax.set_xlabel('Time from event (s)')
    # Label x-axis in seconds relative to event (subtract pre_time)
    xticks = np.linspace(0, pop_rates.shape[1]-1, 6, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(bins[:-1][xticks] - pre_time, 2))
    return im


def plot_mean_psth(pop_rates, bins, ax, pre_time, color='k', label=None):
    """
    Mean PSTH across units.
    """
    ax.plot(bins[:-1] - pre_time, pop_rates.mean(axis=0), color=color, label=label)
    ax.axvline(0, ls='--', alpha=0.5)
    ax.set_title('Mean PSTH')
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Firing rate (Hz)')
    if label:
        ax.legend()


def plot_raster_from_aligned(aligned_spikes, ax, size=1, color='black', t_pre=None, t_post=None):
    """
    Raster from pre-aligned spike lists (one array per event).
    Args:
        aligned_spikes (list of 1D arrays): spike times relative to event.
        ax (Axes): target axes.
        size (float): marker size.
        color (str): marker color.
        t_pre, t_post (float | None): optional limits for x-axis.
    """
    # Flatten to x (times) and y (trial indices)
    xs, ys = [], []
    for trial_idx, rel_times in enumerate(aligned_spikes):
        if rel_times.size:
            xs.append(rel_times)
            ys.append(np.full(rel_times.shape, trial_idx))
    if xs:
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        ax.scatter(xs, ys, s=size, c=color)
    ax.axvline(0, c='r', lw=1)
    ax.set_title('Stimulus-aligned raster (single unit)')
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Trial #')
    if (t_pre is not None) and (t_post is not None):
        ax.set_xlim([-t_pre, t_post])


def plot_area_psth_and_raster(
    area_units_df,
    spike_times_by_unit,
    event_times,
    area_name='Area',
    time_before_change=1.0,
    duration=2.5,
    bin_size=0.01,
    raster_unit_idx=0,
    sort_key=None,
    figsize=(14, 4),
    cmap='viridis',
    clim_percentiles=(0.1, 99.9)
):
    """
    End-to-end population PSTH + mean PSTH + raster, aligned to the same events.

    Args:
        area_units_df (pd.DataFrame): metadata for units; used for optional sorting (e.g., by 'depth').
        spike_times_by_unit (list of 1D arrays): spike times per unit (sorted ascending).
        event_times (1D array): event onset times (seconds).
        area_name (str): label for titles.
        time_before_change (float): seconds before event to include (used for x-axis centering).
        duration (float): total window length per event passed to make_psth (seconds).
                          Note: the raster uses [-time_before_change, (duration - time_before_change)].
        bin_size (float): PSTH bin width (seconds).
        raster_unit_idx (int): which unit to show in the raster.
        sort_key (str | None): if provided and exists in area_units_df, rows are sorted by this column.
        figsize (tuple): figure size.
        cmap (str): heatmap colormap.
        clim_percentiles (tuple): robust color limits.
    Returns:
        fig, axes: the Matplotlib figure and axes (heatmap, mean psth, raster).
        pop_rates, bins: the computed population PSTH and bin edges.
    """
    # Optionally sort units (e.g., by depth); reorder spike_times accordingly
    if sort_key is not None and sort_key in area_units_df.columns:
        order = np.argsort(area_units_df[sort_key].values)
        spike_times_by_unit = [spike_times_by_unit[i] for i in order]
    else:
        order = np.arange(len(spike_times_by_unit))

    # Compute population PSTH over [0, duration], then shift x by 'time_before_change' when plotting
    pop_rates, bins = compute_population_psth(
        spike_times_by_unit,
        start_times=event_times - time_before_change,
        window_dur=duration,
        bin_size=bin_size
    )

    # Build figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [2, 1.2, 1.2]})

    # Heatmap
    im = plot_population_heatmap(pop_rates, bins, axes[0], pre_time=time_before_change,
                                 clim_percentiles=clim_percentiles, cmap=cmap)
    axes[0].set_title(f'Active Change Responses for {area_name}')
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('Firing rate (Hz)')

    # Mean PSTH
    plot_mean_psth(pop_rates, bins, axes[1], pre_time=time_before_change, color='k',
                   label=f'{area_name} (n={pop_rates.shape[0]})')

    # Raster (choose one unit to visualize)
    raster_unit = spike_times_by_unit[raster_unit_idx]
    t_pre  = time_before_change
    t_post = duration - time_before_change
    aligned = align_spikes_to_events(raster_unit, event_times, t_pre=t_pre, t_post=t_post)
    plot_raster_from_aligned(aligned, axes[2], size=2, color='black', t_pre=t_pre, t_post=t_post)

    fig.tight_layout()
    return fig, axes, (pop_rates, bins), order


def plot_unit_raster_aligned(unit_spike_times,
                             trial_starts,
                             trial_colors,
                             t_pre=0.5, t_post=1.0,
                             ax=None,
                             separators=None):
    """
    For each trial in time order, plot spikes within [start - t_pre, start + t_post]
    as a row in the raster. Color each row by that trial's image identity.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    for i, start in enumerate(trial_starts):
        # select spikes around trial
        mask = (unit_spike_times >= start - t_pre) & (unit_spike_times <= start + t_post)
        rel_spikes = unit_spike_times[mask] - start

        # scatter spikes for this trial
        ax.scatter(rel_spikes,
                   np.full_like(rel_spikes, i + 0.5, dtype=float),
                   s=2, c=[trial_colors[i]], marker='|', linewidths=0.8)

    # aesthetics
    ax.axvline(0, color='k', lw=0.8, alpha=0.8)  # stimulus onset
    if separators is not None:
        for r in separators:
            ax.axhline(r, color='0.8', lw=0.6, alpha=0.8)

    ax.set_ylim(0, len(trial_starts))
    ax.set_xlim(-t_pre, t_post)
    # ax.set_ylabel("Trials (time-ordered)")
    ax.set_xlabel("Time from stimulus")
    # ax.set_title("Stimulus-aligned raster (single unit) by image_name")

    return ax

def plot_multi_area_psth_and_raster(
    area_packets,
    event_times,
    t_pre=0.25,
    t_post=0.5,
    bin_size=0.01,
    normalize='zscore',
    cmap='viridis',
    clim_percentiles=(0.1, 99.9),
    figsize_per_row=(12, 3.6),
    hspace=0.5
):
    """
    Stack multiple areas as rows: [Heatmap | Mean PSTH | Raster].
    Reuses fast binning + raster helpers:
      - get_binned_triggered_spike_counts_fast
      - get_stim_window / create_raster

    area_packets: list of dicts like
        {'name': <str>, 'spikes': [np.array,...], 'raster_idx': <int>}
    event_times: 1D array of event onsets (seconds, absolute)
    """

    # ---- binning for PSTH (0..duration) ----
    duration = t_pre + t_post
    bins = np.arange(0, duration + bin_size, bin_size)
    n_areas = len(area_packets)

    fig_w, fig_h = figsize_per_row[0], figsize_per_row[1] * n_areas
    fig, axs = plt.subplots(
        nrows=n_areas, ncols=3, figsize=(fig_w, fig_h),
        gridspec_kw={'width_ratios': [2.0, 1.2, 1.2]},
        squeeze=False, sharex=False, sharey=False
    )
    

    results = []
    # event column index for heatmap (time=0 => bin ~ time_before_change)
    event_col = int(np.argmin(np.abs(bins[:-1] - t_pre)))

    for r, pkt in enumerate(area_packets):
        name       = pkt['name']
        spikes_all = pkt['spikes']
        ridx       = int(np.clip(pkt.get('raster_idx', 0), 0, max(0, len(spikes_all)-1)))

        # ---------- population PSTH using trial binning ----------
        start_times = event_times - t_pre  # align so x=0 is the event
        psths = []
        for st in spikes_all:
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)  # (trials, bins-1)
            rates = trial_counts.mean(axis=0) / bin_size  # Hz
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (units, bins-1)
        
        # ---------- if specified, z-score individual unit firing rates ----------
        plt_rate_label = 'Firing rate (Hz)'
        

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


        pop_rates = _normalize_rates(pop_rates, method=normalize, axis=1)

        # ---------- panels ----------
        ax_hm, ax_mean, ax_ras = axs[r, 0], axs[r, 1], axs[r, 2]

        # Heatmap
        vmin, vmax = [np.percentile(pop_rates, p) for p in clim_percentiles]
        im = ax_hm.imshow(pop_rates, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        # ticks in seconds relative to event (subtract pre)
        xticks = np.linspace(0, pop_rates.shape[1]-1, 6, dtype=int)
        ax_hm.set_xticks(xticks)
        ax_hm.set_xticklabels(np.round(bins[:-1][xticks] - t_pre, 2))
        ax_hm.set_ylabel('Unit # (e.g., sorted by depth)')
        ax_hm.set_xlabel('Time from event (s)')
        ax_hm.axvline(event_col, color='r', linestyle='--', lw=1)
        ax_hm.set_title(f'{name} population', loc='left', fontsize=10)
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cb.set_label(plt_rate_label)

        # Mean PSTH
        mean = np.nanmean(pop_rates, axis=0)
        std = np.nanstd(pop_rates, axis=0)
        time = bins[:-1] - t_pre

        ax_mean.plot(time, mean, color='k',
                    label=f'{name} (n={pop_rates.shape[0]})')
        ax_mean.fill_between(time, mean - std, mean + std,
                            color='k', alpha=0.2)  # transparent shading
        # ax_mean.plot(bins[:-1] - t_pre, np.nanmean(pop_rates, axis=0), color='k',
        #              label=f'{name} (n={pop_rates.shape[0]})')
        ax_mean.axvline(0, ls='--', color='r', lw=1)
        ax_mean.set_xlabel('Time from event (s)')
        ax_mean.set_ylabel(plt_rate_label)
        ax_mean.legend(frameon=False, fontsize=9)
        ax_mean.set_title('Mean PSTH', fontsize=10)

        # Raster (reuse helper)
        rtimes, rtrials = get_stim_window(spikes_all[ridx], event_times,
                                          pre_window=t_pre, post_window=t_post)
        create_raster(rtimes, rtrials, ax=ax_ras, size=2, color='black')
        ax_ras.set_xlim([-t_pre, t_post])
        ax_ras.set_title('Stimulus-aligned raster (single unit)', fontsize=10)

        results.append({'area': name, 'pop_rates': pop_rates, 'bins': bins})

    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)  # extra vertical space for row titles
    return fig, axs, results


def plot_covariance_matrix(cov, ax, title='Covariance Matrix',
                           cmap='magma', clim_percentiles=(1, 99)):
    """
    cov: (n_units, n_units) matrix
    """
    vmin, vmax = np.percentile(cov, clim_percentiles)
    im = ax.imshow(cov, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Unit'); ax.set_ylabel('Unit')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cov')
    return im


def plot_area_covariances(
    area_packets,
    event_times,
    time_before_change=1.0,
    duration=2.5,
    bin_size=0.01,
    normalize='zscore',
    cmap='viridis',
    figsize_per_row=(4.8, 3.6),
    hspace=0.6,
    share_colorbar=False
):
    """
    Build pop_rates from (spikes, event_times) and plot a covariance per area.

    Parameters
    ----------
    area_packets : list of dicts
        Each dict at least: {'name': str, 'spikes': list_of_1D_arrays_per_unit}
        e.g., spikes[i] are spike times (seconds, absolute) for unit i.
    event_times : 1D array of event onsets (seconds, absolute).
    time_before_change : float
        Seconds before event to include (pre-window). Alignment origin = event.
    duration : float
        Total window length (pre + post). Post = duration - time_before_change.
    bin_size : float
        Width of bins for PSTH-style trial binning.
    normalize : str or None
        Passed to compute_unit_covariance.
    cmap : str
        Matplotlib colormap name.
    share_colorbar : bool
        If True, use a global vmin/vmax across areas for comparability.

    Returns
    -------
    fig, axs, results
        results[i] = {
            'area': name,
            'pop_rates': (units x bins-1),
            'cov': (units x units),
            'bins': bins
        }
    """
    area_packets = list(area_packets)
    if len(area_packets) == 0:
        raise ValueError("area_packets is empty.")
    event_times = np.asarray(event_times)
    if event_times.ndim != 1 or event_times.size == 0:
        raise ValueError("event_times must be a non-empty 1D array.")

    # Bin edges for [0, duration]; align by shifting starts to (event - pre)
    bins = np.arange(0, duration + bin_size, bin_size)  # length B
    start_times = event_times - time_before_change       # trials

    # Build pop_rates per area using your fast helper
    results = []
    for pkt in area_packets:
        name = pkt.get('name', 'Area')
        spikes_all = pkt['spikes']  # list of 1D arrays (seconds)

        psths = []
        for st in spikes_all:
            # (trials, B-1) counts in each bin for each trial, aligned to event
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)
            rates = trial_counts.mean(axis=0) / bin_size  # Hz (units per bin / s)
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (units, B-1)

        cov = compute_unit_covariance(pop_rates, normalize=normalize)
        results.append({'area': name, 'pop_rates': pop_rates, 'cov': cov, 'bins': bins})

    # Optional shared color scale
    if share_colorbar:
        all_cov_vals = np.concatenate([r['cov'].ravel() for r in results])
        vmin, vmax = np.percentile(all_cov_vals, [1, 99])  # robust scale
    else:
        vmin = vmax = None

    # Plot one covariance per area (stacked)
    n = len(results)
    fig, axs = plt.subplots(n, 1,
                            figsize=(figsize_per_row[0], figsize_per_row[1]*n),
                            squeeze=False)

    for i, res in enumerate(results):
        ax = axs[i, 0]
        cov = res['cov']
        im = ax.imshow(cov, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{res['area']} — unit covariance ({normalize})", fontsize=10)
        ax.set_xlabel('Unit'); ax.set_ylabel('Unit')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Covariance')

    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)
    return fig, axs, results

def event_times_by_key(trials_df, start_col, time_before_change=0.25):
    """
    trials_df: has columns ['start_time','key'] where 'key' is image_name
    Returns dict: key -> np.array of (aligned) event start times
    """
    # keep original appearance order of keys
    key_groups = {}
    for k, g in trials_df.groupby("key", sort=False):
        key_groups[k] = g[start_col].to_numpy() - time_before_change
    return key_groups


def build_image_locked_rates(spikes_all, trials_df, *, start_col="start_time", 
t_pre=0.25, t_post=0.75,
 bin_size=0.01):
    """
    spikes_all: list[ndarray], each array is spike times (s) for one unit
    trials_df: DataFrame with ['start_time','key']
    Returns:
      pop_concat: (n_units, n_keys * n_bins) concatenated mean-rate vectors
      per_key_unit_rates: dict key -> (n_units, n_bins) mean rates per key
      key_slices: dict key -> slice into the concatenated vector
      keys: list of keys in the order used for concatenation
      bins: histogram bin edges used (relative, from 0..duration)
    """

    # bins are relative [0, duration] as your get_binned_* expects
    duration = t_pre + t_post
    bins = np.arange(0, duration + bin_size, bin_size)
    n_bins = len(bins) - 1

    key2starts = event_times_by_key(trials_df, start_col, time_before_change=t_pre)
    keys = list(key2starts.keys())

    # pre-alloc containers
    n_units = len(spikes_all)
    per_key_unit_rates = {k: np.zeros((n_units, n_bins), dtype=float) for k in keys}

    # compute mean rate per key per unit
    for ui, st in enumerate(spikes_all):
        for k in keys:
            starts = key2starts[k]
            # starts = starts - time_before_change
            # (n_trials, n_bins) counts
            trial_counts = get_binned_triggered_spike_counts_fast(st, starts, bins)
            mean_rate = trial_counts.mean(axis=0) / bin_size
            
            # mu = mean_rate.mean(axis=0, keepdims=True)
            # sd = mean_rate.std(axis=0, keepdims=True) + 1e-9
            # mean_rate = (mean_rate - mu) / sd
            per_key_unit_rates[k][ui, :] = mean_rate

    # concatenate in key order to get the “image-locked” vector per unit
    pop_concat = np.concatenate([per_key_unit_rates[k] for k in keys], axis=1)

    # handy slices so you know where each key lives inside the big vector
    key_slices = {}
    start = 0
    for k in keys:
        key_slices[k] = slice(start, start + n_bins)
        start += n_bins

    return pop_concat, per_key_unit_rates, key_slices, keys, bins


# def plot_area_covariances_with_eigs2(
#     area_packets,
#     G,
#     time_before_change=1.0,
#     duration=2.5,
#     bin_size=0.01,
#     normalize='zscore',
#     cmap='viridis',
#     figsize_per_row=(18, 3.6),   # widened for the extra column
#     hspace=0.8,
#     share_colorbar=False
# ):
#     """
#     For each area: [Covariance heatmap | Eigenvalue spectrum | First eigenvector | PCA (PC1–PC2 trajectory)].
#     """
#     decim = 2  # keep every 2nd bin
#     t_pre = 0.25
#     t_post = 1.
#     bins = np.arange(0, duration + bin_size, bin_size)
#     event_times = G["trials"]["start_time"].values
#     image_names = G["names"]
#     colors = G["colors"]
#     start_times = event_times - time_before_change
#     t_rel = bins[:-1] + bin_size/2.0 - time_before_change  # center-of-bin, relative to event
#     t_rel_ds = t_rel[::decim]

#     results = []
   
#     for pkt in area_packets:
#         spikes_all = pkt['spikes']
#         name = pkt.get('name', 'Area')
#         pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
#             spikes_all,
#             G["trials"],
#             duration=G["t_post"] - G["t_pre"],
#             bin_size=G["bin_size"],
#             time_before_change=G["t_pre"]
#         )
#         # plot_image_locked_matrix(pop_concat, keys, key_slices, cmap="viridis")

#         # --- Build population PSTHs (units x timebins)
#         # psths = []
#         # for st in spikes_all:
#         #     trial_counts = get_binned_triggered_spike_counts_fast(st, event_times, bins)  # (n_trials, n_timebins)
#         #     rates = trial_counts / bin_size                                   # (n_timebins,)
#         #     psths.append(rates.reshape(-1))
#         # pop_rates = np.asarray(psths)  # (n_units, n_timebins)

#         # --- Covariance across units (features), over time
#         cov = compute_unit_covariance(pop_concat, normalize=normalize)  # expects (n_units, n_timebins)
#         λ, eigvecs = np.linalg.eig(cov)

#         # --- PCA over timepoints (observations) with units as features
#         # shape to (n_timebins, n_units)
#         X = pop_concat.T
#         # standardize features (units) across time to avoid dominance by high-rate units
#         Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X) if X.shape[1] > 1 else X
#         pca = PCA(n_components=min(10, Xz.shape[1]))
#         scores = pca.fit_transform(Xz)  # (n_timebins, n_components)
#         evr = pca.explained_variance_ratio_


#         # win_bins = max(5, (int(round(win_s / (bin_size * decim))) // 2) * 2 + 1)  # odd >=5
#         # scores = savgol_filter(scores, window_length=win_bins, polyorder=3, axis=0)

#         results.append({
#             'area': name,
#             'pop_rates': pop_concat,
#             'cov': cov,
#             'eigvals': λ,
#             'eigvecs': eigvecs,
#             'pca_scores': scores,
#             'pca_evr': evr,
#             't_rel': t_rel,
#             't_rel_ds': t_rel_ds
#         })

#     # shared color scale
#     if share_colorbar:
#         all_cov_vals = np.concatenate([r['cov'].ravel() for r in results])
#         vmin, vmax = np.percentile(all_cov_vals, [1, 99])
#     else:
#         vmin = vmax = None

#     # --- Figure (now 4 columns)
#     n = len(results)
#     fig, axs = plt.subplots(n, 4,
#                             figsize=(figsize_per_row[0], figsize_per_row[1]*n),
#                             squeeze=False)

#     for i, res in enumerate(results):
#         # Covariance heatmap
#         ax_cov = axs[i, 0]
#         im = ax_cov.imshow(res['cov'], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
#         ax_cov.set_title(f"{res['area']} — covariance", fontsize=10)
#         ax_cov.set_xlabel('Unit'); ax_cov.set_ylabel('Unit')
#         cb = fig.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
#         cb.set_label('Covariance')

#         # Eigenvalue spectrum
#         ax_eig = axs[i, 1]
#         ax_eig.plot(np.sort(res['eigvals'])[::-1], marker='o', color='k')
#         ax_eig.set_title(f"{res['area']} — eigenvalue spectrum", fontsize=10)
#         ax_eig.set_xlabel('Component')
#         ax_eig.set_ylabel('Variance (λ)')

#         # First eigenvector
#         ax_vec = axs[i, 2]
#         ax_vec.plot(res['eigvecs'][:, 0], marker='o')
#         ax_vec.set_title(f"{res['area']} — first eigenvector", fontsize=10)
#         ax_vec.set_xlabel('Unit')
#         ax_vec.set_ylabel('Coeff.')

#         # PCA: PC1–PC2 trajectory over time
#         ax_pca = axs[i, 3]
#         scores = res['pca_scores']
#         if scores.shape[1] >= 2:
#             pts = np.column_stack([scores[:, 0], scores[:, 1]])
#             segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 2)
#             # draw time-ordered path
#             ax_pca.plot(scores[:, 0], scores[:, 1], '-o', ms=2)
#             # mark event time (t=0) if it falls within window
#             # NEW: color by time with LineCollection
#             lc = LineCollection(segs, cmap='viridis', array=res['t_rel_ds'][1:], linewidths=2)
#             ax_pca.add_collection(lc)
#             ax_pca.autoscale()
#             cbar = fig.colorbar(lc, ax=ax_pca, fraction=0.046, pad=0.04)
#             cbar.set_label('Time (s)')

#             # # NEW: sparse markers (~every 100 ms)
#             # mark_every = max(1, int(round(0.1 / (bin_size * decim))))
#             # ax_pca.plot(pts[::mark_every, 0], pts[::mark_every, 1], 'o', ms=3, color='k', alpha=0.6)

#             zero_idx = np.argmin(np.abs(res['t_rel']))
#             ax_pca.scatter(scores[zero_idx, 0], scores[zero_idx, 1], s=40, edgecolor='k', facecolor='none', label='event (t=0)')
#             ax_pca.set_title(f"{res['area']} — PCA traj (PC1 vs PC2)\nEVR: {res['pca_evr'][:2].sum():.2f}", fontsize=10)
#             ax_pca.set_xlabel('PC1'); ax_pca.set_ylabel('PC2')
#             ax_pca.legend(loc='best', fontsize=8, frameon=False)
#         else:
#             ax_pca.text(0.5, 0.5, "PCA<2 comps", ha='center', va='center')
#             ax_pca.set_axis_off()

#     fig.tight_layout()
#     fig.subplots_adjust(hspace=hspace)
#     return fig, axs, results


def plot_area_covariances_with_eigs(
    area_packets,
    event_times,
    time_before_change=1.0,
    duration=2.5,
    bin_size=0.01,
    normalize='zscore',
    cmap='RdBu_r',
    figsize_per_row=(18, 3.6),   # widened for the extra column
    hspace=0.8,
    share_colorbar=False
):
    """
    For each area: [Covariance heatmap | Eigenvalue spectrum | First eigenvector | PCA (PC1–PC2 trajectory)].
    """
    decim = 2  # keep every 2nd bin
   
    bins = np.arange(0, duration + bin_size, bin_size)
    start_times = event_times - time_before_change
    t_rel = bins[:-1] + bin_size/2.0 - time_before_change  # center-of-bin, relative to event
    t_rel_ds = t_rel[::decim]

    results = []
    for pkt in area_packets:
        name = pkt.get('name', 'Area')
        spikes_all = pkt['spikes']

        # --- Build population PSTHs (units x timebins)
        psths = []
        for st in spikes_all:
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)  # (n_trials, n_timebins)
            rates = trial_counts.mean(axis=0) / bin_size                                   # (n_timebins,)
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (n_units, n_timebins)

        # --- Covariance across units (features), over time
        cov = compute_unit_covariance(pop_rates, normalize=normalize)  # expects (n_units, n_timebins)
        λ, eigvecs = np.linalg.eig(cov)

        # --- PCA over timepoints (observations) with units as features
        # shape to (n_timebins, n_units)
        X = pop_rates.T
        # standardize features (units) across time to avoid dominance by high-rate units
        Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X) if X.shape[1] > 1 else X
        pca = PCA(n_components=min(10, Xz.shape[1]))
        scores = pca.fit_transform(Xz)  # (n_timebins, n_components)
        evr = pca.explained_variance_ratio_

        

        # win_bins = max(5, (int(round(win_s / (bin_size * decim))) // 2) * 2 + 1)  # odd >=5
        # scores = savgol_filter(scores, window_length=win_bins, polyorder=3, axis=0)

        results.append({
            'area': name,
            'pop_rates': pop_rates,
            'cov': cov,
            'eigvals': λ,
            'eigvecs': eigvecs,
            'pca_scores': scores,
            'pca_evr': evr,
            't_rel': t_rel,
            't_rel_ds': t_rel_ds
        })

    # shared color scale
    if share_colorbar:
        all_cov_vals = np.concatenate([r['cov'].ravel() for r in results])
        vmin, vmax = np.percentile(all_cov_vals, [1, 99])
    else:
        vmin = vmax = None

    # --- Figure (now 4 columns)
    n = len(results)
    fig, axs = plt.subplots(n, 4,
                            figsize=(figsize_per_row[0], figsize_per_row[1]*n),
                            squeeze=False)

    for i, res in enumerate(results):
        # Covariance heatmap
        ax_cov = axs[i, 0]
        im = ax_cov.imshow(res['cov'], aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax_cov.set_title(f"{res['area']} — covariance", fontsize=10)
        ax_cov.set_xlabel('Unit'); ax_cov.set_ylabel('Unit')
        cb = fig.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
        cb.set_label('Covariance')

        # Eigenvalue spectrum
        ax_eig = axs[i, 1]
        ax_eig.plot(np.sort(res['eigvals'])[::-1], marker='o', color='k')
        ax_eig.set_title(f"{res['area']} — eigenvalue spectrum", fontsize=10)
        ax_eig.set_xlabel('Component')
        ax_eig.set_ylabel('Variance (λ)')

        # First eigenvector
        ax_vec = axs[i, 2]
        ax_vec.plot(res['eigvecs'][:, 0], marker='o')
        ax_vec.set_title(f"{res['area']} — first eigenvector", fontsize=10)
        ax_vec.set_xlabel('Unit')
        ax_vec.set_ylabel('Coeff.')

        # PCA: PC1–PC2 trajectory over time
        ax_pca = axs[i, 3]
        scores = res['pca_scores']
        if scores.shape[1] >= 2:
            pts = np.column_stack([scores[:, 0], scores[:, 1]])
            segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 2)
            # draw time-ordered path
            ax_pca.plot(scores[:, 0], scores[:, 1], '-o', ms=2)
            # mark event time (t=0) if it falls within window
            # NEW: color by time with LineCollection
            lc = LineCollection(segs, cmap='viridis', array=res['t_rel_ds'][1:], linewidths=2)
            ax_pca.add_collection(lc)
            ax_pca.autoscale()
            cbar = fig.colorbar(lc, ax=ax_pca, fraction=0.046, pad=0.04)
            cbar.set_label('Time (s)')

            # # NEW: sparse markers (~every 100 ms)
            # mark_every = max(1, int(round(0.1 / (bin_size * decim))))
            # ax_pca.plot(pts[::mark_every, 0], pts[::mark_every, 1], 'o', ms=3, color='k', alpha=0.6)

            zero_idx = np.argmin(np.abs(res['t_rel']))
            ax_pca.scatter(scores[zero_idx, 0], scores[zero_idx, 1], s=40, edgecolor='k', facecolor='none', label='event (t=0)')
            ax_pca.set_title(f"{res['area']} — PCA traj (PC1 vs PC2)\nEVR: {res['pca_evr'][:2].sum():.2f}", fontsize=10)
            ax_pca.set_xlabel('PC1'); ax_pca.set_ylabel('PC2')
            ax_pca.legend(loc='best', fontsize=8, frameon=False)
        else:
            ax_pca.text(0.5, 0.5, "PCA<2 comps", ha='center', va='center')
            ax_pca.set_axis_off()

    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)
    return fig, axs, results


def plot_unit_raster_simple(unit_spike_times, trials, trial_colors, change_rows,
                            block_mids, block_names, t_pre=0.5, t_post=1., ax=None, start_col="start_time"):
    """Thin wrapper around your existing plot_unit_raster_aligned."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))
    ax = plot_unit_raster_aligned(
        unit_spike_times=np.asarray(unit_spike_times),
        trial_starts=trials[start_col].values,
        trial_colors=trial_colors,
        t_pre=t_pre, t_post=t_post,
        ax=ax,
        separators=change_rows
    )
    ax.set_yticks(block_mids)
    ax.set_yticklabels(block_names)
    return ax

def _gauss1d(x, sigma_bins):
    if (sigma_bins is None) or (sigma_bins <= 0): return x
    w = int(max(3, round(6 * sigma_bins))) | 1  # odd length
    c = w // 2
    g = np.exp(-0.5 * (np.arange(w) - c)**2 / (sigma_bins**2))
    g /= g.sum()
    return np.convolve(x, g, mode="same")

def _boxcar1d(x, win_bins):
    if (win_bins is None) or (win_bins <= 1): return x
    k = np.ones(int(win_bins)) / int(win_bins)
    return np.convolve(x, k, mode="same")
    
def plot_subpop_spikes(unit_spike_times, trials, colors, sep, mids, names,
                       t_pre=0.5, t_post=1.0, bin_size=0.1,
                       smooth_kind=None, smooth_value=None, ax=None, x_label="Time from stimulus (s)", start_col="start_time"):
    if ax is None:
        _, ax = plt.subplots()

    bins = np.arange(-t_pre, t_post + bin_size, bin_size)
    X = []
    for t0 in trials[start_col].values:
        rel = unit_spike_times - t0
        rel = rel[(rel >= -t_pre) & (rel <= t_post)]
        X.append(np.histogram(rel, bins=bins)[0])
    X = np.asarray(X)

    starts = np.r_[0, sep]; ends = np.r_[sep, len(trials)]
    t = bins[:-1]

    for s, e, name in zip(starts, ends, names):
        grp = X[s:e]
        m, sd = grp.mean(axis=0), grp.std(axis=0)

        if smooth_kind == "gaussian":
            m, sd = _gauss1d(m, smooth_value), _gauss1d(sd, smooth_value)
        elif smooth_kind == "boxcar":
            m, sd = _boxcar1d(m, smooth_value), _boxcar1d(sd, smooth_value)

        ax.plot(t, m, color=colors[s], label=name)
        ax.fill_between(t, m - sd, m + sd, color=colors[s], alpha=0.3)

    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Firing rate (Hz)')
    # ax.set_xlabel("Time from stimulus (s)"); ax.set_ylabel("Spikes")
    # ax.legend()
    return ax

def plot_area_grid_combined(
    area, good_units, spike_times, n_units, groupings,
    subplots=("raster", "mean±sd"),
    t_pre=0.5, t_post=1.0, bin_size=0.1,
    smooth_kind=None, smooth_value=None,
    pick="random", seed=0, figsize_per_row=(4, 2.5), sharex=True,
    extras=("autocorr",),                  
    acg_bin_ms=1.0, acg_win_ms=60.0,     
    acg_refractory_ms=2.0             
):
    """
    One grid: n_units rows × (len(groupings) * len(subplots)) cols.
    Each grouping can choose its own key_col and alignment start_col.
    """
    # --- pick units ---
    rng = np.random.default_rng(seed)
    area_units = good_units[good_units["structure_acronym"] == area].index.to_numpy()
    sel = rng.choice(area_units, size=min(n_units, len(area_units)), replace=False) if pick=="random" else area_units[:n_units]

    n_rows = len(sel)
    n_cols = len(subplots) * len(groupings) + len(extras)
    figsize = (figsize_per_row[0] * n_cols, figsize_per_row[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=False) 

    # if extra == "autocorr":
    # ax.get_shared_y_axes().remove(ax)  # <— frees this axis
    # plot_autocorrelogram(st, bin_ms=acg_bin_ms, win_ms=acg_win_ms,
    #                      refractory_ms=acg_refractory_ms, ax=ax)

    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    # --- grid fill ---
    for r, iu in enumerate(sel):
        st = spike_times[iu]
        for g, G in enumerate(groupings):
            for s, kind in enumerate(subplots):
                ax = axes[r, g*len(subplots) + s]
                if kind == "raster":
                    plot_unit_raster_simple(st, G["trials"], G["colors"], G["sep"], G["mids"], G["names"],
                                            t_pre=G["t_pre"], t_post=G["t_post"], ax=ax, start_col=G["start_col"])
                elif kind == "mean±sd":
                    plot_subpop_spikes(st, G["trials"], G["colors"], G["sep"], G["mids"], G["names"],
                                       t_pre=G["t_pre"], t_post=G["t_post"], bin_size=G["bin_size"],
                                       smooth_kind=G["smooth_kind"], smooth_value=G["smooth_value"], ax=ax, start_col=G["start_col"])

                if r == 0:
                    ax.set_title(f"{G['label']} • {kind}")
                if s == 0:
                    ax.set_ylabel(f"unit {iu}")

        base = len(groupings) * len(subplots)                   
        for e, extra in enumerate(extras):
            ax = axes[r, base + e]
            if extra == "autocorr":
                # pos = ax.get_position()          # break sharing by rebuilding the axis
                # fig.delaxes(ax)
                # ax = fig.add_axes(pos)
                ax = plot_autocorrelogram(st, bin_ms=acg_bin_ms, win_ms=acg_win_ms,
                refractory_ms=acg_refractory_ms, ax=ax)

                # plot_autocorrelogram(st, bin_ms=acg_bin_ms, win_ms=acg_win_ms,
                #                     refractory_ms=acg_refractory_ms, ax=ax)
                if r == 0:
                    ax.set_title(f"Autocorr • ±{int(acg_win_ms)} ms")
            else:
                ax.set_axis_off()
    plt.tight_layout()
    return fig, axes

def plot_autocorrelogram(st, *, bin_ms=1.0, win_ms=60.0, refractory_ms=2.0, ax=None):
    spk = SpikeTrain(st * s, t_start=st.min() * s, t_stop=st.max() * s + 1 *s)
    lags_ms, counts = generate_autocorr_data(spk, bin_ms + 1, win_ms)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.bar(lags_ms, counts, width=bin_ms, color='k', linewidth=0.3)
    ax.axvspan(-refractory_ms, refractory_ms, color='orange', alpha=0.2)
    ax.axvline(refractory_ms, color='black', linewidth=0.2)
    ax.axvline(-refractory_ms, color='black', linewidth=0.2)
    ax.set_xlabel('Time (ms)')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_image_locked_matrix_old(pop_concat, keys, key_slices, cmap="viridis"):
    """
    pop_concat: (n_units, n_keys*n_bins)
    keys: list of image keys (ordered)
    key_slices: dict mapping key -> slice into the concat axis
    """

    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(pop_concat, aspect="auto", cmap=cmap,
                   interpolation="nearest")
    bin_size = 0.01          # keep consistent with build_image_locked_rates
    time_before_change = 0.25
    onset_bin = 0#int(round(time_before_change / bin_size))
    # add color blocks or separators per image
    # give each key a color
    cm = plt.get_cmap("tab10")
    for i, k in enumerate(keys):
        sl = key_slices[k]
        color = cm(i % cm.N)

        # stimulus onset (one line per image block)  ⟵ ADD
        x0 = sl.start + onset_bin
        ax.axvline(x0, ymin=0, ymax=1, linewidth=1.2, color=color, alpha=0.9)

        # ax.add_patch(
        #     patches.Rectangle(
        #         (sl.start, -0.5),        # (x,y) lower left
        #         sl.stop - sl.start,      # width
        #         pop_concat.shape[0],     # height = n_units
        #         fill=False,
        #         edgecolor=color,
        #         linewidth=2
        #     )
        # )
        # optional: put the key name centered at the top
        xmid = (sl.start + sl.stop) / 2
        ax.text(xmid, -1, k, ha="center", va="bottom",
                rotation=90, fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Firing rate (Hz)")
    ax.set_xlabel("Concatenated time bins (by image)")
    ax.set_ylabel("Unit index")
    plt.tight_layout()
    plt.show()



def plot_image_locked_matrix(
    pop_concat, keys, key_slices, *,
    bin_size=0.01, time_before_change=0.25, title="name",
    cmap="viridis", block_cmap="tab10",
    figsize=(12, 6), annotate=True, draw_onset=True
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pop_concat, aspect="auto", cmap=cmap, interpolation="nearest")

    onset_bin = int(round(time_before_change / bin_size))
    cm = plt.get_cmap(block_cmap)
    for i, k in enumerate(keys):
        sl = key_slices[k]
        color = cm(i % cm.N)
        if draw_onset:
            ax.axvline(sl.start + onset_bin, ymin=0, ymax=1, lw=1.2, color=color, alpha=0.9)
        if annotate:
            ax.text((sl.start + sl.stop) / 2, -1, str(k), ha="center", va="bottom",
                    rotation=90, fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Firing rate (Hz)")
    ax.set(xlabel=f"Concatenated bins (bin={bin_size}s)", ylabel="Unit index")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax

# def plot_pca_trajs_3d_multi(areas, groups, start_cols, *, n_components=3, win=7,
#                             time_before_change=0.25, ncols=4,
#                             random_state=0, cmap_name=None):

def build_per_trial_unit_rates(spikes_all, trials_df, *, start_col="start_time",
                               t_pre=0.25, t_post=0.75, bin_size=0.01):
    """
    Returns: per_key_unit_trials (dict key -> list[(n_units, n_bins)]), keys, bins
    """
    import numpy as np
    duration = t_pre + t_post
    bins = np.arange(0, duration + bin_size, bin_size)
    n_bins = len(bins) - 1
    keys = list(trials_df['key'].drop_duplicates().values)

    per_key_unit_trials = {k: [] for k in keys}
    for k, g in trials_df.groupby('key', sort=False):
        starts = g[start_col].values - t_pre
        for st in starts:
            edges = st + bins
            counts = np.stack([np.histogram(u_spk, edges)[0] for u_spk in spikes_all], axis=0)
            per_key_unit_trials[k].append(counts / bin_size)
    return per_key_unit_trials, keys, bins

def plot_pca_trajs_3d_multi(areas, groups, start_cols,
                            *, n_components=3, win=7, t_pre=0.25, t_post=0.75,
                            ncols=4, random_state=0, cmap_name=None,
                            show_trials=True, trial_alpha=0.35,
                            trial_lw=0.7, mean_lw=2.2,
                            max_trials_per_key=None,
                            n_trial_overlays=4,
                            downsample_step=5,
                            per_key_trial_starts=None,
                            abs_time_cmap="viridis"):
    figs = {}
    mean_val_dict_per_area = {}
    for area, spikes_all in areas.items():
        trials = groups[area]
        start_col = start_cols[area]

        # build matrices
        pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
            spikes_all, trials, start_col=start_col,
            t_pre=t_pre, t_post=t_post, bin_size=0.01
        )
        per_key_unit_trials, _, _ = build_per_trial_unit_rates(
            spikes_all, trials, start_col=start_col,
            t_pre=t_pre, t_post=t_post, bin_size=0.01
        )

        fig, axes, mean_val_dict = plot_pca_trajs_3d_flat(
            pop_concat, per_key_unit_rates, keys, bins,
            n_components=n_components, win=win, ncols=ncols,  
            random_state=random_state, cmap_name=cmap_name,
            per_key_unit_trials=per_key_unit_trials,
            show_trials=show_trials, trial_alpha=trial_alpha,
            trial_lw=trial_lw, mean_lw=mean_lw,
            max_trials_per_key=max_trials_per_key,
            n_trial_overlays=n_trial_overlays,     downsample_step=downsample_step,          # <— forward
            per_key_trial_starts=per_key_trial_starts.get(area, {}) if per_key_trial_starts else None,
            abs_time_cmap=abs_time_cmap                      # <— forward
        )

        for k in mean_val_dict.keys():
            if k not in mean_val_dict_per_area.keys():
                mean_val_dict_per_area[k] = []
            mean_val_dict_per_area[k].append(mean_val_dict[k])

        fig.suptitle(area, fontsize=12)
        figs[area] = (fig, axes)
    return figs, mean_val_dict_per_area

def plot_pca_trajs_3d_multi_old(areas, groups, start_cols, *, 
    n_components=3, win=7, time_before_change=0.25, ncols=4, random_state=0, cmap_name=None,
    show_trials=True, trial_alpha=0.35, trial_lw=0.7, mean_lw=2.2,
    max_trials_per_key=None):

    """
    areas: dict area_name -> spikes_all (list of spike time arrays per unit)
    groups: dict area_name -> trials_df (with ['start_time','key'])
    """
    figs = {}
    for area, spikes_all in areas.items():
        trials = groups[area]
        start_col = start_cols[area]

        pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
            spikes_all, trials, start_col=start_col, t_pre=t_pre, t_post=t_post, bin_size=0.01,
        )
        per_key_unit_trials, keys2, bins2 = build_per_trial_unit_rates(
            spikes_all, trials, start_col=start_col,
            t_pre=t_pre, t_post=t_post, bin_size=0.01
        )

        # --- do PCA/plot just like before, but return fig, axes ---
        # fig, axes = plot_pca_trajs_3d_flat(
        #     pop_concat, per_key_unit_rates, keys, bins,
        #     n_components=n_components, win=win,
        #     ncols=ncols, time_before_change=time_before_change,
        #     random_state=random_state, cmap_name=cmap_name
        # )
        fig, axes = plot_pca_trajs_3d_flat(
            pop_concat, per_key_unit_rates, keys, bins,
            n_components=n_components, win=win,
            ncols=ncols, time_before_change=time_before_change,
            random_state=random_state, cmap_name=cmap_name,
            per_key_unit_trials=per_key_unit_trials,
            show_trials=show_trials, trial_alpha=trial_alpha,
            trial_lw=trial_lw, mean_lw=mean_lw,
            max_trials_per_key=max_trials_per_key
        )

        fig.suptitle(area, fontsize=12)
        figs[area] = (fig, axes)
    return figs

# def plot_pca_trajs_3d_flat(pop_concat, per_key_unit_rates, keys, bins,
#                       *, n_components=3, win=7, ncols=4,
#                       time_before_change=0.25, random_state=0, cmap_name=None,
#                       per_key_unit_trials=None, show_trials=True,
#                       trial_alpha=0.35, trial_lw=0.7, mean_lw=2.2,
#                       max_trials_per_key=None):
#     """
#     Plot smoothed 3D PCA trajectories: one subplot per key.

#     pop_concat: (n_units, n_keys*n_bins) matrix used to fit PCA (samples=time, features=units)
#     per_key_unit_rates: dict key -> (n_units, n_bins) mean rates
#     keys: ordered list of keys
#     bins: 1D array of bin edges (len = n_bins+1)
#     """
#     # --- global PCA on timebins x units ---
#     X = pop_concat.T
#     mu, sd = X.mean(0), X.std(0) + 1e-9
#     Xz = (X - mu) / sd
#     pca = PCA(n_components=n_components, random_state=random_state).fit(Xz)

#     def project(R):  # (n_units, n_bins) -> (n_bins, n_components)
#         return ((R.T - mu) / sd) @ pca.components_.T

#     # --- smooth per-key trajectories ---
#     ker = np.ones(win) / win
#     traj = {k: np.column_stack([np.convolve(project(per_key_unit_rates[k])[:, i], ker, "same")
#                                 for i in range(n_components)]) for k in keys}

#     # unified limits
#     allP = np.vstack(list(traj.values()))
#     lims = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]

#     # colors
#     cmap = plt.get_cmap(cmap_name or ('tab20' if len(keys) > 10 else 'tab10'))
#     key_color = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

#     # grid
#     # n = len(keys)
#     # nrows = (n + ncols - 1) // ncols
#     # fig = plt.figure(figsize=(4*ncols, 3.6*nrows))

#     # grid
#     n = len(keys)
#     nrows, ncols = 1, n          # <-- force one row
#     fig = plt.figure(figsize=(4*ncols, 3.6))   # <-- height for one row only

#     t_rel = bins[:-1] - time_before_change
#     onset_idx = int(np.argmin(np.abs(t_rel)))

#     # optional thin per-trial overlays
#     if show_trials and (per_key_unit_trials is not None) and (k in per_key_unit_trials):
#         trials_list = per_key_unit_trials[k]
#         if max_trials_per_key is not None:
#             trials_list = trials_list[:max_trials_per_key]
#         for Rt in trials_list:
#             Pt = ((Rt.T - mu) / sd) @ pca.components_.T
#             if win > 1:
#                 kker = np.ones(win) / win
#                 Pt = np.column_stack([np.convolve(Pt[:, i], kker, "same") for i in range(n_components)])
#             ax.plot(Pt[:, 0], Pt[:, 1], Pt[:, 2],
#                     lw=trial_lw, alpha=trial_alpha, color=key_color[k], zorder=1)


#     axes = {}
#     for i, k in enumerate(keys, 1):
#         ax = fig.add_subplot(nrows, ncols, i, projection="3d")
#         P = traj[k][:, :3]  # first 3 PCs for 3D
#         ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=mean_lw, color=key_color[k], alpha=0.95)
#         # ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=1.6, color=key_color[k], alpha=0.95)
#         ax.scatter(P[onset_idx, 0], P[onset_idx, 1], P[onset_idx, 2],
#                    s=28, edgecolor="k", facecolor="w", linewidth=0.6)
#         ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
#         ax.set_title(str(k), fontsize=9)
#         ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
#         axes[k] = ax

#     fig.tight_layout()
#     return fig, axes

def plot_pca_trajs_3d_flat_without_insets(pop_concat, per_key_unit_rates, keys, bins,
                      *, n_components=3, win=7, ncols=4,
                      t_pre=0.25, t_post=0.75, random_state=0, cmap_name=None,
                      per_key_unit_trials=None, show_trials=True,
                      trial_alpha=0.35, trial_lw=0.7, mean_lw=2.2,
                      n_trial_overlays=4, max_trials_per_key=None,
                      per_key_trial_starts=None,        # NEW: dict key -> list[float secs]
                      abs_time_cmap="viridis"):          # NEW

    # --- global PCA on timebins x units ---
    X = pop_concat.T
    mu, sd = X.mean(0), X.std(0) + 1e-9
    pca = PCA(n_components=n_components, random_state=random_state).fit((X - mu) / sd)

    def project(R):  # (n_units, n_bins) -> (n_bins, n_components)
        return ((R.T - mu) / sd) @ pca.components_.T

    # --- smooth per-key trajectories ---
    traj = {
        k: np.column_stack([np.convolve(project(per_key_unit_rates[k])[:, i], np.ones(win) / win, "same")
                            for i in range(n_components)])
        for k in keys
    }

    # unified limits (first 3 PCs)
    allP = np.vstack([traj[k][:, :3] for k in keys])
    # lims = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]

    # colors
    cmap = plt.get_cmap(cmap_name or ('tab10' if len(keys) > 10 else 'tab10'))
    key_color = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

    # grid (force one row if you want)
    n = len(keys)
    nrows, ncols = 1, n
    fig = plt.figure(figsize=(4*ncols, 3.6))

    t_rel = bins[:-1] - t_pre
    onset_idx = int(np.argmin(np.abs(t_rel)))

    axes = {}
    for i, k in enumerate(keys, 1):
        ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        P = traj[k][:, :3]
        # t_rel = bins[:-1] - time_before_change
        # onset_idx = int(np.argmin(np.abs(t_rel)))

        # If absolute times available, build a single global Normalizer
        abs_time_norm = None
        abs_cmap = get_cmap(abs_time_cmap)
        if show_trials and (per_key_trial_starts is not None):
            # collect min/max across ALL trials, ALL keys
            tmins, tmaxs = [], []
            for j in keys:
                starts = per_key_trial_starts.get(j, [])
                if len(starts) == 0: 
                    continue
                tmins.append(np.min(np.asarray(starts) + t_rel[0]))
                tmaxs.append(np.max(np.asarray(starts) + t_rel[-1]))
            if tmins and tmaxs:
                abs_time_norm = Normalize(vmin=float(np.min(tmins)), vmax=float(np.max(tmaxs)))
        
        # --- unified limits INCLUDING trials ---
        pts_for_lims = [traj[k][:, :3] for k in keys]  # mean per key

        # --- per-trial overlays: only a few, evenly spaced ---
        if show_trials and (per_key_unit_trials is not None) and (k in per_key_unit_trials):
            trials_list = per_key_unit_trials[k]
            if max_trials_per_key is not None:
                trials_list = trials_list[:max_trials_per_key]

            if len(trials_list) > 0 and n_trial_overlays > 0:
                n_show = min(n_trial_overlays, len(trials_list))
                # idxs = np.unique(np.linspace(0, len(trials_list)-1, n_show, dtype=int))

                quarters = [q for q in np.array_split(np.arange(len(trials_list)), n_trial_overlays) if len(q) > 0]
                for qidx in quarters:
                    # CHANGED: average trials within this quarter (n_trials_q, n_units, n_bins) -> (n_units, n_bins)
                    Rt = np.mean(np.stack([trials_list[i] for i in qidx], axis=0), axis=0)

                # for idx in idxs:
                    # Rt = trials_list[idx]                      # (n_units, n_bins)
                    Pt = ((Rt.T - mu) / sd) @ pca.components_.T
                    if win > 1:
                        Pt = np.column_stack([
                            np.convolve(Pt[:, j], np.ones(win)/(win), "same")
                            for j in range(n_components)
                        ])

                    # # --- color by absolute time across session ---
                    # if abs_time_norm is not None and (per_key_trial_starts is not None):
                    #     start_t = per_key_trial_starts[k][idx]  # absolute start (seconds)
                    #     abs_times = start_t + t_rel             # length n_bins
                    #     colors = abs_cmap(abs_time_norm(abs_times))

                    #     # draw as colored segments
                    #     segments = [list(zip(Pt[i:i+2,0], Pt[i:i+2,1], Pt[i:i+2,2]))
                    #                 for i in range(Pt.shape[0]-1)]
                    #     lc = Line3DCollection(segments, colors=colors[:-1],
                    #                         linewidth=trial_lw, alpha=1.0)
                    #     ax.add_collection3d(lc)
                    # else:
                    #     # fallback: single-color line
                    #     ax.plot(Pt[:, 0], Pt[:, 1], Pt[:, 2],
                    #             lw=trial_lw, alpha=0.9, color=key_color[k], zorder=1)
                    # CHANGED: color by mean absolute start time of the quarter (if available)
                    if (abs_time_norm is not None) and (per_key_trial_starts is not None):
                        mean_start = float(np.mean(per_key_trial_starts[k][qidx]))
                        abs_times = mean_start + t_rel
                        colors = abs_cmap(abs_time_norm(abs_times))

                        segments = [list(zip(Pt[i:i+2, 0], Pt[i:i+2, 1], Pt[i:i+2, 2]))
                                    for i in range(Pt.shape[0] - 1)]
                        lc = Line3DCollection(segments, colors=colors[:-1],
                                            linewidth=trial_lw, alpha=1.0)
                        ax.add_collection3d(lc)
                    else:
                        ax.plot(Pt[:, 0], Pt[:, 1], Pt[:, 2],
                                lw=trial_lw, alpha=0.9, color=key_color[k], zorder=1)

                    pts_for_lims.append(Pt[:, :3])

        allP = np.vstack(pts_for_lims) if pts_for_lims else np.zeros((1,3))   
        lims = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]
 
        # mean trajectory
        ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=mean_lw, color='k' if per_key_trial_starts else key_color[k], alpha=0.95) #key_color[k],

        # onset marker
        ax.scatter(P[onset_idx, 0], P[onset_idx, 1], P[onset_idx, 2],
                   s=28, edgecolor="k", facecolor="w", linewidth=0.6)

        # axes & labels
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_title(str(k), fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        axes[k] = ax

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_rel[0], vmax=t_rel[-1]))
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04, label="Time (s)")
    
    # before adding the colorbar
    fig.tight_layout(rect=[0, 0.12, 1, 1])  # leave 12% at the bottom

    # Shared colorbar for absolute time (if available)
    if abs_time_norm is not None:
        sm = plt.cm.ScalarMappable(cmap=abs_cmap, norm=abs_time_norm)
        sm.set_array([])
        # cbar = fig.colorbar(sm, ax=list(axes.values()),
        #                     fraction=0.02, pad=0.04)
        # cbar.set_label("Absolute time (s)")

        fig.subplots_adjust(bottom=0.18)
        cax = fig.add_axes([0.2, 0.10, 0.6, 0.03])
        fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("Absolute time (s)")


    # fig.tight_layout()
    return fig, axes


def plot_pca_trajs_3d_flat(pop_concat, per_key_unit_rates, keys, bins,
                      *, n_components=3, win=7, ncols=4, downsample_step=5,
                      t_pre=0.25, t_post=0.75, random_state=0, cmap_name=None,
                      per_key_unit_trials=None, show_trials=True,
                      trial_alpha=0.35, trial_lw=0.7, mean_lw=2.2,
                      n_trial_overlays=4, max_trials_per_key=None,
                      per_key_trial_starts=None,        # NEW: dict key -> list[float secs]
                      abs_time_cmap="viridis"):          # NEW

    # --- global PCA on timebins x units ---
    X = pop_concat.T
    mu, sd = X.mean(0), X.std(0) + 1e-9
    pca = PCA(n_components=n_components, random_state=random_state).fit((X - mu) / sd)

    def project(R):  # (n_units, n_bins) -> (n_bins, n_components)
        return ((R.T - mu) / sd) @ pca.components_.T

    # --- ADD near the top (after defining `project`) ---
    quarter_trajs = {}   # NEW: (key, quarter_idx) -> Ptd (n_out x n_components)

    # --- smooth (and optionally resample) per-key trajectories ---
    traj = {}
    onset_idx_ds_dict = {}
    t_rel = bins[:-1] - t_pre
    onset_idx = int(np.argmin(np.abs(t_rel)))

    for k in keys:
        Pk = project(per_key_unit_rates[k])[:, :n_components]
        dt = float(np.median(np.diff(bins)))    # seconds per sample
        Pks = smooth_traj(Pk, win, dt=dt, cut_hz=3., order=2)
        n_out = max(3, len(Pks)//downsample_step)
        Pkd = _resample_by_arclength(Pks, n_out)
        keep_idx = np.linspace(0, len(Pks)-1, n_out).astype(int)  # for onset mapping
        j = int(np.argmin(np.abs(keep_idx - onset_idx)))  # position in S3d
        onset_idx_ds = np.clip(j, 0, len(Pkd)-1)

        traj[k] = Pkd
        onset_idx_ds_dict[k] = onset_idx_ds

    # unified limits (first 3 PCs)
    allP = np.vstack([traj[k][:, :3] for k in keys])
    # lims = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]

    # colors
    cmap = plt.get_cmap(cmap_name or ('tab10' if len(keys) > 10 else 'tab10'))
    key_color = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

    # grid (force one row if you want)
    n = len(keys)
    # nrows, ncols = 1, n
    nrows, ncols = 2, n                        
    fig = plt.figure(figsize=(4*ncols, 6))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=[3, 2])


    axes = {}
    mean_val_dict = {}
    for i, k in enumerate(keys, 1):
        # ax = fig.add_subplot(nrows, ncols, i, projection="3d")
        ax = fig.add_subplot(gs[0, i-1], projection="3d")  # NEW: first row cell

        P = traj[k][:, :3]
        onset_idx_ds = onset_idx_ds_dict[k]
        # t_rel = bins[:-1] - time_before_change
        # onset_idx = int(np.argmin(np.abs(t_rel)))

        # If absolute times available, build a single global Normalizer
        abs_time_norm = None
        abs_cmap = get_cmap(abs_time_cmap)
        if show_trials and (per_key_trial_starts is not None):
            # collect min/max across ALL trials, ALL keys
            tmins, tmaxs = [], []
            for j in keys:
                starts = per_key_trial_starts.get(j, [])
                if len(starts) == 0: 
                    continue
                tmins.append(np.min(np.asarray(starts) + t_rel[0]))
                tmaxs.append(np.max(np.asarray(starts) + t_rel[-1]))
            if tmins and tmaxs:
                abs_time_norm = Normalize(vmin=float(np.min(tmins)), vmax=float(np.max(tmaxs)))
        
        # --- unified limits INCLUDING trials ---
        pts_for_lims = [traj[k][:, :3] for k in keys]  # mean per key

        # --- per-trial overlays: only a few, evenly spaced ---
        if show_trials and (per_key_unit_trials is not None) and (k in per_key_unit_trials):
            trials_list = per_key_unit_trials[k]
            if max_trials_per_key is not None:
                trials_list = trials_list[:max_trials_per_key]

            if len(trials_list) > 0 and n_trial_overlays > 0:
                n_show = min(n_trial_overlays, len(trials_list))
                # idxs = np.unique(np.linspace(0, len(trials_list)-1, n_show, dtype=int))

                # quarters = [q for q in np.array_split(np.arange(len(trials_list)), n_trial_overlays) if len(q) > 0]
                quarters = [q for q in np.array_split(np.arange(len(trials_list)), n_trial_overlays) if len(q) > 0]
                for qnum, qidx in enumerate(quarters):        
                    # CHANGED: average trials within this quarter (n_trials_q, n_units, n_bins) -> (n_units, n_bins)
                    Rt = np.mean(np.stack([trials_list[i] for i in qidx], axis=0), axis=0)
                    
                    Pt = ((Rt.T - mu) / sd) @ pca.components_.T
                    dt = float(np.median(np.diff(bins)))    # seconds per sample
                    Pts = smooth_traj(Pt, win, dt=dt, cut_hz=3., order=3)
                    n_out = max(3, len(Pt)//downsample_step)
                    Ptd = _resample_by_arclength(Pts, n_out)
                    quarter_trajs[(k, qnum)] = Ptd[:, :n_components]

                    keep_idx = np.linspace(0, len(Pts)-1, n_out).astype(int)  # for onset mapping
                    j = int(np.argmin(np.abs(keep_idx - onset_idx)))  # position in S3d
                    onset_idx_ds = np.clip(j, 0, len(Ptd)-1)

                    # if win > 1:
                    #     Pt = np.column_stack([
                    #         np.convolve(Pt[:, j], np.ones(win)/(win), "same")
                    #         for j in range(n_components)
                    #     ])

                   
                    if (abs_time_norm is not None) and (per_key_trial_starts is not None):
                        mean_start = float(np.mean(per_key_trial_starts[k][qidx]))
                        abs_times = mean_start + t_rel
                        colors = abs_cmap(abs_time_norm(abs_times))

                        segments = [list(zip(Ptd[i:i+2, 0], Ptd[i:i+2, 1], Ptd[i:i+2, 2]))
                                    for i in range(Ptd.shape[0] - 1)]
                        lc = Line3DCollection(segments, colors=colors[:-1],
                                            linewidth=trial_lw, alpha=1.0)
                        ax.add_collection3d(lc)
                         # onset marker
                        ax.scatter(Ptd[onset_idx_ds, 0], Ptd[onset_idx_ds, 1], Ptd[onset_idx_ds, 2],
                                s=20, edgecolor=colors[-1], facecolor=colors[-1], linewidth=0.6)

                        ax.scatter(Ptd[-1, 0], Ptd[-1, 1], Ptd[-1, 2],
                            s=50, color=colors[-1], edgecolor=colors[-1], zorder=5, marker='*')

                    else:
                        ax.plot(Ptd[:, 0], Ptd[:, 1], Ptd[:, 2],
                                lw=trial_lw, alpha=0.9, color=key_color[k], zorder=1)

                    pts_for_lims.append(Ptd[:, :3])
                    
        allP = np.vstack(pts_for_lims) if pts_for_lims else np.zeros((1,3))   
        lims = [(allP[:, i].min(), allP[:, i].max()) for i in range(3)]
 
        # mean trajectory
        ax.plot(P[:, 0], P[:, 1], P[:, 2], lw=mean_lw, color='k' if per_key_trial_starts else key_color[k], alpha=0.95) #key_color[k],

        
        # onset marker
        ax.scatter(P[onset_idx_ds, 0], P[onset_idx_ds, 1], P[onset_idx_ds, 2],
                   s=20, edgecolor='k', facecolor='k', linewidth=0.6)

        ax.scatter(P[-1, 0], P[-1, 1], P[-1, 2],
                        s=50, color='k', edgecolor='k', zorder=5, marker='*')

        # axes & labels
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_title(str(k), fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        axes[k] = ax
        # add vertical colorbar next to this subplot
        sm = plt.cm.ScalarMappable(cmap=abs_cmap, norm=abs_time_norm)
        #     sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, pad=0.2)
        cbar.set_label("time")

        # --- RSA over quarter-averaged trajectories; plots on second row ---
        q_trajs = [quarter_trajs[(k, q)] for q in range(n_trial_overlays) if (k, q) in quarter_trajs]
        if len(q_trajs) >= 2:
            X = np.vstack([P.reshape(-1) for P in q_trajs])
            X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)
            # rdm = 1.0 - np.corrcoef(X)
            from sklearn.metrics.pairwise import cosine_similarity

            rdm = cosine_similarity(X)   # (n_cond x n_cond), values ∈ [−1, 1]
            # rdm = 1.0 - sim              # cosine distance

            # ax_rdm = fig.add_subplot(gs[1, i-1])
            # im = ax_rdm.imshow(rdm, interpolation='nearest')
            # ax_rdm.set_title("Q1–Q{} RDM".format(len(q_trajs)), fontsize=8)
            # ticks = np.arange(len(q_trajs))
            # labels = [f"Q{j+1}" for j in ticks]
            # ax_rdm.set_xticks(ticks); ax_rdm.set_yticks(ticks)
            # ax_rdm.set_xticklabels(labels, rotation=0, fontsize=7)
            # ax_rdm.set_yticklabels(labels, fontsize=7)

             # --- replace this whole times-block ---
            # times = []
            # all_starts = per_key_trial_starts.get(k, [])
            # t0, t1 = t_rel[0], t_rel[-1]
            # mean_start = float(np.mean(per_key_trial_starts[k][qidx]))
            # times = mean_start + t_rel
            # q_bounds = np.linspace(t0, t1, n_trial_overlays+1)
            # times = [f"{0.5*(q_bounds[j]+q_bounds[j+1]):.2f}" 
            #         for j in range(len(q_bounds)-1)]
            
            ticks = np.arange(len(q_trajs))
            labels = [f"trial_grp_{j+1}" for j in ticks]

            ax_rdm = fig.add_subplot(gs[1, i-1])
            im = ax_rdm.imshow(rdm, interpolation='nearest', cmap="RdBu_r")  # use same cmap

            # mean of off-diagonal entries
            vals = rdm[np.triu_indices_from(rdm, k=1)]
            mean_val = np.mean(vals) if len(vals) > 0 else np.nan


            ax_rdm.set_title(f"RSA", fontsize=8)
            ticks = np.arange(len(q_trajs))
            ax_rdm.set_xticks(ticks); ax_rdm.set_yticks(ticks)
            ax_rdm.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax_rdm.set_yticklabels(labels, fontsize=7)

            # add colorbar
            cbar = fig.colorbar(im, ax=ax_rdm, orientation='vertical', fraction=0.05, pad=0.02)
            cbar.set_label("1 - r", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            mean_val_dict[k] = mean_val

            # # NEW: add a 1-column subplot on the right of the RDM row
            # ax_mean = ax_rdm.inset_axes([1.5, 0.1, 0.2, 0.8])  # x,y,w,h in axes fraction
            # ax_mean.bar([0], [mean_val], color="gray")
            # ax_mean.set_ylim(0, 2)          # since 1 - r ranges [0,2]
            # ax_mean.set_xticks([])
            # ax_mean.set_ylabel("Mean RSA", fontsize=7)
            # ax_mean.set_title("Avg", fontsize=7)
    
    # if quarter_trajs:
    #     labels = []
    #     X = []
    #     for (k, qnum), Ptd in sorted(quarter_trajs.items(), key=lambda x: (str(x[0][0]), x[0][1])):
    #         labels.append(f"{k}:Q{qnum+1}")
    #         X.append(Ptd.reshape(-1))
    #     X = np.vstack(X)  # (n_conditions, T*C)
    #     X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-9)  # z-score features
    #     sim = np.corrcoef(X)            # similarity
    #     rdm = 1.0 - sim                 # dissimilarity (RDM)

    #     ax_rdm = fig.add_subplot(gs[1, :])  # NEW: span full second row
    #     im = ax_rdm.imshow(rdm, interpolation='nearest')
    #     ax_rdm.set_title("Quarter-trajectory RSA (1 - r)")
    #     # ax_rdm.set_xticks(np.arange(len(labels)))
    #     # ax_rdm.set_yticks(np.arange(len(labels)))
    #     # ax_rdm.set_xticklabels(labels, rotation=60, ha='right', fontsize=7)
    #     # ax_rdm.set_yticklabels(labels, fontsize=7)
    #     ax_rdm.set_xticks([])
    #     ax_rdm.set_yticks([])
    #     # optional: add compact labels below plot
    #     ax_rdm.set_xlabel("Conditions (key × quarter)")
    #     ax_rdm.set_ylabel("Conditions (key × quarter)") 
    #     # fig.colorbar(im, ax=ax_rdm, orientation='vertical', fraction=0.02, pad=0.02)
    #     # --- tweak the colorbar ---
    #     fig.colorbar(im, ax=ax_rdm, orientation='vertical', fraction=0.015, pad=0.01, label="1 - r")


    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=t_rel[0], vmax=t_rel[-1]))
    # sm.set_array([])
    # fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04, label="Time (s)")
    
    # before adding the colorbar
    fig.tight_layout()#rect=[0, 0.12, 1, 1])  # leave 12% at the bottom

    # Shared colorbar for absolute time (if available)
    # if abs_time_norm is not None:
    #     sm = plt.cm.ScalarMappable(cmap=abs_cmap, norm=abs_time_norm)
    #     sm.set_array([])
    #     # cbar = fig.colorbar(sm, ax=list(axes.values()),
    #     #                     fraction=0.02, pad=0.04)
    #     # cbar.set_label("Absolute time (s)")

    #     fig.subplots_adjust(bottom=0.18)
    #     cax = fig.add_axes([0.2, 0.10, 0.6, 0.03])
    #     fig.colorbar(sm, cax=cax, orientation="horizontal").set_label("Absolute time (s)")


    # fig.tight_layout()
    return fig, axes, mean_val_dict

def _pca_inputs_for(area_packet, grouping, *, bin_size=0.01):
    spikes_all = area_packet["spikes"]
    t_pre = grouping.get("t_pre", 0.25)
    t_post = grouping.get("t_post", 0.5)
    duration = t_pre + t_post
    pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
        spikes_all, grouping["trials"], start_col=grouping["start_col"],  bin_size=bin_size,
         t_pre=t_pre, t_post=t_post
    )
    return pop_concat, per_key_unit_rates, keys, bins

def plot_pca_trajs_grid(area_packets, groupings, *, bin_size=0.01,  downsample_step=5,
                        win=7, marginals=False):


    if isinstance(area_packets, dict):
        items = list(area_packets.items())
    else:
        items = [(pkt.get("name", f"Area{i}"), pkt) for i, pkt in enumerate(area_packets)]

    area_names = [n for n,_ in items]
    group_labels = [g.get("label", "Group") for g in groupings]

    R, C = len(items), len(groupings)
    # fig, axes = plt.subplots(R, C, figsize=(4*C, 4*R), subplot_kw={"projection": "3d"})
    R, C = max(1, len(items)), len(groupings)
    fig, axes = plt.subplots(R, C, figsize=(4*C, 4*R), subplot_kw={"projection": "3d"})

    axes = np.atleast_2d(axes)
    handles = []
    labels = []
    for r, (area_name, pkt) in enumerate(items):
        for c, g in enumerate(groupings):
            ax = axes[r, c]
            pop_concat, per_key_unit_rates, keys, bins = _pca_inputs_for(pkt, g, bin_size=bin_size)
            title = group_labels[c] #if r == 0 else None
            t_pre = g.get("t_pre", 0.25)
            t_post = g.get("t_post", 0.5)
            _, ax = plot_pca_trajs_3d(pop_concat, per_key_unit_rates, keys, bins, t_pre=t_pre,
                                     win=win, ncols=4, ax=ax, title=title, 
                                     downsample_step=downsample_step,
                                     marginals=marginals)
            handles, labels = ax.get_legend_handles_labels()
            # Add legend only once per column (e.g. top row axis)
            if r == 0:
                ax.legend(handles, labels,
                        frameon=False, fontsize=8, ncol=3,
                        loc="upper center", bbox_to_anchor=(0.5, 1.25))
        
        axes[r,0].annotate(area_name, xy=(-0.1, 0.5), xycoords='axes fraction',
                       ha='right', va='center', rotation=90,
                       annotation_clip=False, fontsize=15)  
       
    fig.tight_layout(h_pad=2.0, w_pad=2.)
    
    return fig, axes

def _downsample_rows(X, *, target_points=None, step=None):
    n = X.shape[0]
    if step is None:
        step = max(1, int(np.ceil(n / max(1, target_points))))
    idx = np.arange(0, n, step)
    return X[idx], idx

def _downsample_rows_mean(X, *, step):
    chunks = np.array_split(X, int(np.ceil(X.shape[0]/step)), axis=0)
    Xd = np.vstack([c.mean(axis=0) for c in chunks])
    keep_idx = (np.rint(np.linspace(0, X.shape[0]-1, len(chunks)))).astype(int)
    return Xd, keep_idx

from scipy.signal import butter, filtfilt

def smooth_traj(P, w, *, dt=None, cut_hz=4.0, order=3):
    """
    Zero-phase low-pass per PC. If dt is given, uses cut_hz; else
    falls back to a boxcar of width w (for safety).
    """
    if dt is None:
        if w <= 1: 
            return P
        k = np.ones(int(w), dtype=float) / float(w)
        return np.column_stack([np.convolve(P[:, i], k, mode="same")
                                for i in range(P.shape[1])])

    nyq = 0.5 / dt
    b, a = butter(order, cut_hz / nyq, btype="low")
    Q = P.copy()
    for i in range(Q.shape[1]):
        Q[:, i] = filtfilt(b, a, Q[:, i], padtype="odd")
    return Q

def _resample_by_arclength(P, n_out):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0, np.cumsum(d)]
    s_new = np.linspace(0, s[-1], n_out)
    return np.column_stack([np.interp(s_new, s, P[:, j]) for j in range(P.shape[1])])


def plot_pca_trajs_3d(pop_concat, per_key_unit_rates, keys, bins, *, t_pre=0.25, downsample_step=5,
                      win=7, ncols=4, ax=None, title=None, marginals=False):
    """
    Plot 3D PCA trajectories for per-key population rates.

    pop_concat: (n_units, sum_k n_bins) concatenated mean-rate matrix (all keys)
    per_key_unit_rates: dict key -> (n_units, n_bins) mean rates per key
    keys: ordered list of keys to plot
    bins: histogram edges used (unused here, kept for API symmetry)
    win: smoothing window (moving average) over time bins
    ncols: kept for API compatibility (unused here)
    ax: optional existing 3D axis to draw into
    title: optional title for this panel
    marginals: if True, add small 2D projections (PC pairs) as insets
    """
  

    # --- fit PCA globally on all time bins (samples=time, features=units) ---
    X = pop_concat.T  # (time, units)
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xz = (X - mu) / sd

    n_samples, n_features = X.shape
    n_eff = min(3, n_samples, n_features)  # <= available dims
    if n_eff < 1:
        raise ValueError(f"Not enough data for PCA (samples={n_samples}, units={n_features}).")
    if n_eff < 3:
        print(f"[warn] reducing n_components to {n_eff} (samples={n_samples}, units={n_features}).")

    pca = PCA(n_components=3, random_state=0, svd_solver="full").fit(Xz)

    def project_to_pcs(R):
        # R: (n_units, n_bins) -> (n_bins, 3)
        Z = ((R.T - mu) / sd) @ pca.components_.T
        return Z
    
    # --- figure/axes ---
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure

    # --- colors ---
    cmap = plt.get_cmap("tab10")
    key_colors = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

    # relative time bins
    t_rel = bins[:-1] - t_pre
    onset_idx = int(np.argmin(np.abs(t_rel)))  # nearest bin to event onset


    # --- plot each key's smoothed 3D trajectory ---
    for k in keys:
        R = per_key_unit_rates[k]            # (n_units, n_bins)
        P3 = project_to_pcs(R)               # (n_bins, 3)
        # S3 = smooth_traj(P3, win)            # smoothed (n_bins, 3)
        dt = float(np.median(np.diff(bins)))    # seconds per sample
        S3 = smooth_traj(P3, win, dt=dt, cut_hz=3.0, order=3)

        # downsample
        # S3d, keep_idx = _downsample_rows(S3, step=downsample_step)
        n_out = max(3, len(S3)//downsample_step)
        S3d = _resample_by_arclength(S3, n_out)
        keep_idx = np.linspace(0, len(P3)-1, n_out).astype(int)  # for onset mapping
        
        ax.plot(S3d[:, 0], S3d[:, 1], S3d[:, 2], lw=1.0, alpha=0.95,
         label=str(k), color=key_colors[k])
        # mark the event onset with a scatter point

        # where does the onset land after decimation? closest kept index:
        # onset_idx_ds = np.argmin(np.abs(keep_idx - onset_idx))
        j = int(np.argmin(np.abs(keep_idx - onset_idx)))  # position in S3d
        onset_idx_ds = np.clip(j, 0, len(S3d)-1)

        # ax.plot(S3d[:, 0], S3d[:, 1], S3d[:, 2],
        #         lw=1.0, alpha=0.95, color=key_colors[k], label=str(k))

        ax.scatter(S3d[onset_idx_ds, 0], S3d[onset_idx_ds, 1], S3d[onset_idx_ds, 2],
                s=20, color=key_colors[k], edgecolor=key_colors[k], zorder=5)

        
        # --- add arrow at the end ---
        # pick last two points
        x0, y0, z0 = S3d[-2]
        x1, y1, z1 = S3d[-1]

        ax.scatter(S3d[-1, 0], S3d[-1, 1], S3d[-1, 2],
                s=50, color=key_colors[k], edgecolor=key_colors[k], zorder=5, marker='*')

       
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    if title:
        ax.set_title(title)

    # --- optional tiny marginal projections ---
    if marginals:
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax12 = inset_axes(ax, width="32%", height="26%", loc="upper left", borderpad=0.6)
            ax13 = inset_axes(ax, width="32%", height="26%", loc="upper right", borderpad=0.6)
            ax23 = inset_axes(ax, width="32%", height="26%", loc="lower left", borderpad=0.6)

            ax12.set_xlabel("PC1")
            ax12.set_ylabel("PC2")
            ax13.set_xlabel("PC1")
            ax13.set_ylabel("PC3")
            ax23.set_xlabel("PC2")
            ax23.set_ylabel("PC3")

            for k in keys:
                R = per_key_unit_rates[k]
                P3 = project_to_pcs(R)
                dt = float(np.median(np.diff(bins)))    # seconds per sample
                S3 = smooth_traj(P3, win, dt=dt, cut_hz=3.0, order=3)
                # S3 = smooth_traj(P3, win)
                # downsample
                # S3d, keep_idx = _downsample_rows_mean(S3, step=downsample_step)
                n_out = max(3, len(S3)//downsample_step)
                S3d = _resample_by_arclength(S3, n_out)
                keep_idx = np.linspace(0, len(P3)-1, n_out).astype(int)  # for onset mapping

                j = int(np.argmin(np.abs(keep_idx - onset_idx)))  # position in S3d
                onset_idx_ds = np.clip(j, 0, len(S3d)-1)

                c = key_colors[k]
                ax12.plot(S3d[:, 0], S3d[:, 1], lw=0.8, alpha=0.95, color=c)
                ax13.plot(S3d[:, 0], S3d[:, 2], lw=0.8, alpha=0.95, color=c)
                ax23.plot(S3d[:, 1], S3d[:, 2], lw=0.8, alpha=0.95, color=c)
                # onset markers (2D projections of the same event point)
                ax12.scatter(S3d[onset_idx_ds, 0], S3d[onset_idx_ds, 1], s=20, color=c, edgecolor=c, zorder=5)
                ax13.scatter(S3d[onset_idx_ds, 0], S3d[onset_idx_ds, 2], s=20, color=c, edgecolor=c, zorder=5)
                ax23.scatter(S3d[onset_idx_ds, 1], S3d[onset_idx_ds, 2], s=20, color=c, edgecolor=c, zorder=5)

                ax12.scatter(S3d[-1, 0], S3d[-1, 1], s=30, color=c, edgecolor=c, zorder=5, marker='*')
                ax13.scatter(S3d[-1, 0], S3d[-1, 2], s=30, color=c, edgecolor=c, zorder=5, marker='*')
                ax23.scatter(S3d[-1, 1], S3d[-1, 2], s=30, color=c, edgecolor=c, zorder=5, marker='*')

                # Arrow at the last segment
                # ax12.annotate("",
                #     xy=(S3d[-1, 0], S3d[-1, 1]),      # arrow head (last point)
                #     xytext=(S3d[-2, 0], S3d[-2, 1]),  # arrow tail (prev point)
                #     arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.95,lw=1.5, zorder=5)
                # )
                # ax13.annotate("",
                #     xy=(S3d[-1, 0], S3d[-1, 2]),      # arrow head (last point)
                #     xytext=(S3d[-2, 0], S3d[-2, 2]),  # arrow tail (prev point)
                #     arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.95,lw=1.5, zorder=5)
                # )
                # ax23.annotate("",
                #     xy=(S3d[-1, 1], S3d[-1, 2]),      # arrow head (last point)
                #     xytext=(S3d[-2, 1], S3d[-2, 2]),  # arrow tail (prev point)
                #     arrowprops=dict(arrowstyle="-|>", color=c, alpha=0.95,lw=1.5, zorder=5)
                # )

            for a in (ax12, ax13, ax23):
                a.set_xticks([]); a.set_yticks([])
                a.set_frame_on(True)
        except Exception:
            # keep it silent; main 3D plot still renders
            pass

    return (fig, ax) if created_fig else (None, ax)


def pca_trajs_3d_for(
    area_packet,
    stim_df,
    *,
    key_col="image_name",
    start_col="start_time",
    label=None,
    t_pre=0.25,
    t_post=0.5,
    bin_size=0.01,
    duration=0.75,
    smooth_kind="gaussian",
    smooth_value=0.5,
    win=7,
    ncols=4
):
    """
    One-stop shop: build grouping -> image-locked rates -> 3D PCA trajectories.
    Returns (fig, axes) from plot_pca_trajs_3d().
    """
    grp = make_group(
        stim_df,
        key_col=key_col,
        label=(label or key_col),
        start_col=start_col,
        t_pre=t_pre, t_post=t_post, bin_size=bin_size,
        smooth_kind=smooth_kind, smooth_value=smooth_value,
    )

    pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
        area_packet["spikes"],
        grp["trials"],                     # must have ['start_time','key']
        t_pre=t_pre,
        bin_size=bin_size,
        t_post=t_post
    )

    return plot_pca_trajs_3d(
        pop_concat,
        per_key_unit_rates,
        keys,
        bins,
        win=win,
        ncols=ncols
    )



def pca_trajs_3d_multi(areas_dict, stim_df, **kwargs):
    """
    areas_dict: {area_name: area_packet}
    Returns: {area_name: (fig, axes)}
    """
    out = {}
    for name, pkt in areas_dict.items():
        fig, axes = pca_trajs_3d_for(pkt, stim_df, label=name, **kwargs)
        out[name] = (fig, axes)
    return out



def plot_pca_trajs_for_area(area_packet, grouping, *, duration=0.75, bin_size=0.01, win=7, ncols=4):
    """
    grouping: an object returned by make_group(...) (your grp_image/grp_novelty/grp_outcome)
              must expose grouping["trials"] with ['start_time','key'].
    """
    spikes_all = area_packet["spikes"]
    t_pre = grouping.get("t_pre", 0.25)   # use whatever was set in make_group
    pop_concat, per_key_unit_rates, key_slices, keys, bins = build_image_locked_rates(
        spikes_all,
        grouping["trials"],
        start_col=grouping["start_col"],
        duration=duration,
        bin_size=bin_size,
        time_before_change=t_pre
    )
    return plot_pca_trajs_3d(pop_concat, per_key_unit_rates, keys, bins, win=win, ncols=ncols)


def pca_trajs_3d_multi(area_packets, groupings, **kwargs):
    """
    area_packets: dict{name->packet} OR list[packet] (expects pkt['name'])
    groupings: list of your make_group(...) outputs (e.g., [grp_image, grp_novelty, grp_outcome])
    Returns: dict{name -> dict{group_label -> (fig, axes)}}
    """
    if isinstance(area_packets, dict):
        items = area_packets.items()
    else:
        items = ((pkt.get("name", "Area"), pkt) for pkt in area_packets)

    out = {}
    for area_name, pkt in items:
        per_group = {}
        for g in groupings:
            label = g.get("label", "Group")
            per_group[label] = plot_pca_trajs_for_area(pkt, g, **kwargs)
        out[area_name] = per_group
    return out


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def _fit_pca(pop_concat, n_components=3, random_state=0):
    """Fit PCA on (units x all_timebins) matrix and return (pca, mu, sd)."""
    X = pop_concat.T                                # (timebins, units)
    mu = X.mean(0); sd = X.std(0) + 1e-9
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state).fit((X - mu) / sd)
    return pca, mu, sd

def _project(unit_by_time, pca, mu, sd):
    """Project (units x n_bins) onto previously fit PCs → (n_bins x n_pc)."""
    return ((unit_by_time.T - mu) / sd) @ pca.components_.T

def _smooth(M, win=None):
    """Moving-average smooth along time for each column of (T x D)."""
    if not win or win <= 1: return M
    k = np.ones(win) / win
    return np.column_stack([np.convolve(M[:,i], k, mode="same") for i in range(M.shape[1])])

def plot_pc_timeseries_by_key(pop_concat, per_key_unit_rates, keys, bins, *,
                              n_components=3, time_before_change=0.25,
                              win=None, sharey=True, figsize=(5, 2.6)):
    """
    Fit PCA on pop_concat, then plot PC1–3 time series per key.
    Returns (fig, axes_dict) where axes_dict[key] = Axes.
    """
    pca, mu, sd = _fit_pca(pop_concat, n_components=n_components)
    t_rel = bins[:-1] - float(time_before_change)

    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1]*n),
                             sharex=True, sharey=sharey)
    if n == 1: axes = [axes]

    axes_dict = {}
    for ax, k in zip(axes, keys):
        R = per_key_unit_rates[k]                # (n_units, n_bins)
        pcs = _smooth(_project(R, pca, mu, sd), win=win)  # (n_bins, n_pc)
        for i, lab, lw in [(0,"PC1",1.5),(1,"PC2",1.2),(2,"PC3",1.0)]:
            if i >= pcs.shape[1]: break
            ax.plot(t_rel, pcs[:, i], lw=lw, label=lab)
        ax.axvline(0, color='k', lw=0.8, alpha=0.6)
        ax.set_title(str(k))
        axes_dict[k] = ax

    axes[-1].set_xlabel("Time from onset (s)")
    for a in axes: a.set_ylabel("PC score")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    return fig, axes_dict

def plot_pc_timeseries_by_key_multi(area_data, *,
                                    n_components=3, time_before_change=0.25,
                                    win=None, ncols=4, sharey=True, figsize=(5, 2.6)):
    """
    area_data: dict[area] -> dict with keys:
        'pop_concat', 'per_key_unit_rates', 'keys', 'bins'
    Returns dict[area] -> (fig, axes_dict)
    """
    figs = {}
    for area, D in area_data.items():
        fig, axes_dict = plot_pc_timeseries_by_key(
            D['pop_concat'], D['per_key_unit_rates'], D['keys'], D['bins'],
            n_components=n_components, time_before_change=time_before_change,
            win=win, sharey=sharey, figsize=figsize
        )
        fig.suptitle(area)
        figs[area] = (fig, axes_dict)
    return figs

def plot_pc_timeseries_grid(area_data, *,
                            n_components=3, time_before_change=0.25,
                            win=None, figsize=(12, 8)):
    areas = list(area_data.keys())
    n_rows = len(areas)
    n_cols = len(next(iter(area_data.values()))['keys'])

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
    #                          sharex=True, sharey=True, squeeze=False)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 2.0),  # <-- scale dynamically
        sharex=False, sharey=False, squeeze=False
    )


    for r, area in enumerate(areas):
        D = area_data[area]
        pca, mu, sd = _fit_pca(D['pop_concat'], n_components=n_components)
        t_rel = D['bins'][:-1] - float(time_before_change)

        for c, k in enumerate(D['keys']):
            ax = axes[r, c]
            R = D['per_key_unit_rates'][k]
            pcs = _smooth(_project(R, pca, mu, sd), win=win)
            for i, lab, lw in [(0,"PC1",1.5),(1,"PC2",1.2),(2,"PC3",1.0)]:
                if i >= pcs.shape[1]: break
                ax.plot(t_rel, pcs[:, i], lw=lw, label=lab)
            ax.axvline(0, color='k', lw=0.8, alpha=0.6)
            ax.legend(frameon=False, fontsize=6, ncol=1, loc="upper left",
                 bbox_to_anchor=(1.01, 1.0))
            # if r == 0: 
            ax.set_title(str(k))
            axes[r,c].set_ylabel(f"{area}\nPC score")
            axes[r,c].set_xlabel(f"Time from onset (s)")

    # axes[-1, :].flat[-1].legend(frameon=False, fontsize=8, ncol=3)
    # fig.supxlabel("Time from onset (s)")
    # grab handles/labels from any Axes (e.g. first one)

    
    # handles, labels = axes[0,0].get_legend_handles_labels()

    # fig.legend(
    #     handles, labels,
    #     loc="upper center",       # or "lower center", "center right", etc.
    #     bbox_to_anchor=(0.5, -0.02),   # (x, y) in figure coords
    #     ncol=3, frameon=False, fontsize=8
    # )
    fig.tight_layout()
    return fig, axes
