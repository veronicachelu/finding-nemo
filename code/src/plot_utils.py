import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
from data_utils import process_nwb_metadata, get_stim_window, get_spike_counts_all, get_binned_triggered_spike_counts_fast, apply_zscore

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
    # a = a/np.max(stim_index)/bin_size
    n_trials = int(rtrials.max() + 1) if 'rtrials' in locals() else 1  # or pass in n_trials
    a = a / max(1, n_trials) / bin_size

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


# ---------- Plotting helpers ----------

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


# ---------- Orchestrator ----------

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


def build_area_packet(area_name, good_units_df, spike_times_all_units,
                      area_col='structure_acronym', sort_key=None, raster_unit_idx=0):
    area_units = good_units_df[good_units_df[area_col] == area_name]
    spike_times_by_unit = [spike_times_all_units[iu] for iu, _ in area_units.iterrows()]
    if sort_key is not None and sort_key in area_units.columns:
        order = np.argsort(area_units[sort_key].values)
        spike_times_by_unit = [spike_times_by_unit[i] for i in order]
    return dict(name=area_name, spikes=spike_times_by_unit, raster_idx=raster_unit_idx)


def plot_multi_area_psth_and_raster(
    area_packets,
    event_times,
    time_before_change=1.0,
    duration=2.5,
    bin_size=0.01,
    zscore_pop=[],
    cmap='viridis',
    clim_percentiles=(0.1, 99.9),
    figsize_per_row=(12, 3.6),
    hspace=0.5
):
    """
    Stack multiple areas as rows: [Heatmap | Mean PSTH | Raster].
    Reuses your fast binning + raster helpers:
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

        # ---------- population PSTH using your fast trial binning ----------
        start_times = event_times - time_before_change  # align so x=0 is the event
        psths = []
        for st in spikes_all:
            trial_counts = get_binned_triggered_spike_counts_fast(st, start_times, bins)  # (trials, bins-1)
            rates = trial_counts.mean(axis=0) / bin_size  # Hz
            psths.append(rates)
        pop_rates = np.asarray(psths)  # (units, bins-1)
        
        # ---------- if specified, z-score individual unit firing rates ----------
        plt_rate_label = 'Firing rate (Hz)'
        if name in zscore_pop:
            pop_rates = apply_zscore(rates=pop_rates, axis=1)
            plt_rate_label = 'Z-scored firing rate (Hz)'

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

        # Raster (reuse your helpers)
        t_pre, t_post = time_before_change, duration - time_before_change
        rtimes, rtrials = get_stim_window(spikes_all[ridx], event_times,
                                          pre_window=t_pre, post_window=t_post)
        create_raster(rtimes, rtrials, ax=ax_ras, size=2, color='black')
        ax_ras.set_xlim([-t_pre, t_post])
        ax_ras.set_title('Stimulus-aligned raster (single unit)', fontsize=10)

        results.append({'area': name, 'pop_rates': pop_rates, 'bins': bins})

        # cleaner y labels on middle/right columns for lower rows
        if r > 0:
            ax_mean.set_ylabel('')
            ax_ras.set_ylabel('')

    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)  # extra vertical space for row titles
    return fig, axs, results
