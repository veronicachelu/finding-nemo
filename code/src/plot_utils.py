import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.collections import LineCollection 
import data_utils
import importlib
importlib.reload(data_utils)
from data_utils import *

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
    stim_ids,
    time_before_change=1.0,
    duration=2.5,
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
    event_col = int(np.argmin(np.abs(bins[:-1] - time_before_change)))

    for r, pkt in enumerate(area_packets):
        name       = pkt['name']
        spikes_all = pkt['spikes']
        ridx       = int(np.clip(pkt.get('raster_idx', 0), 0, max(0, len(spikes_all)-1)))

        # ---------- population PSTH using trial binning ----------
        start_times = event_times - time_before_change  # align so x=0 is the event
        psths = []
        for st in spikes_all:
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)  # (trials, bins-1)
            rates = trial_counts.mean(axis=0) / bin_size  # Hz
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (units, bins-1)
        
        # ---------- if specified, z-score individual unit firing rates ----------
        plt_rate_label = 'Firing rate (Hz)'
        pop_rates = _normalize_rates(pop_rates, method=normalize, axis=1)

        # ---------- panels ----------
        ax_hm, ax_mean, ax_ras = axs[r, 0], axs[r, 1], axs[r, 2]

        # Heatmap
        vmin, vmax = [np.percentile(pop_rates, p) for p in clim_percentiles]
        im = ax_hm.imshow(pop_rates, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        # ticks in seconds relative to event (subtract pre)
        xticks = np.linspace(0, pop_rates.shape[1]-1, 6, dtype=int)
        ax_hm.set_xticks(xticks)
        ax_hm.set_xticklabels(np.round(bins[:-1][xticks] - time_before_change, 2))
        ax_hm.set_ylabel('Unit # (e.g., sorted by depth)')
        ax_hm.set_xlabel('Time from event (s)')
        ax_hm.axvline(event_col, color='r', linestyle='--', lw=1)
        ax_hm.set_title(f'{name} population', loc='left', fontsize=10)
        cb = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cb.set_label(plt_rate_label)

        # Mean PSTH
        ax_mean.plot(bins[:-1] - time_before_change, np.nanmean(pop_rates, axis=0), color='k',
                     label=f'{name} (n={pop_rates.shape[0]})')
        ax_mean.axvline(0, ls='--', color='r', lw=1)
        ax_mean.set_xlabel('Time from event (s)')
        ax_mean.set_ylabel(plt_rate_label)
        ax_mean.legend(frameon=False, fontsize=9)
        ax_mean.set_title('Mean PSTH', fontsize=10)

        # Raster (reuse helper)
        t_pre, t_post = time_before_change, duration - time_before_change
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
    # from scipy.ndimage import gaussian_filter1d
    # from scipy.signal import savgol_filter
    # NEW: smooth PC scores in time (Savitzky–Golay)
    # win_s = 0.15  # 150 ms
    decim = 2  # keep every 2nd bin
    # Xz_ds = Xz[::decim]
   
    
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
    # ax.set_xlabel("Time from stimulus (s)"); ax.set_ylabel("Spikes")
    # ax.legend()
    return ax

def plot_area_grid_combined(
    area, good_units, spike_times, n_units, groupings,
    subplots=("raster", "mean±sd"),
    t_pre=0.5, t_post=1.0, bin_size=0.1,
    smooth_kind=None, smooth_value=None,
    pick="first", seed=0, figsize_per_row=(4, 2.5), sharex=True,
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
    ax.bar(lags_ms, counts, width=bin_ms, color='gray', linewidth=0.3)
    ax.axvspan(-refractory_ms, refractory_ms, color='orange', alpha=0.2)
    ax.axvline(refractory_ms, color='black', linewidth=0.2)
    ax.axvline(-refractory_ms, color='black', linewidth=0.2)
    ax.set_xlabel('Time (ms)')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax
