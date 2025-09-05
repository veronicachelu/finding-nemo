import numpy as np
import pandas as pd

def select_good_units(session, snr=1, isi_violations=1, firing_rate=0.1, area_list = None):
    '''
    Create a df with good units by thresholding quality metrics and including recording sites.
    
    Parameters
    ----------
    - session: allenSDK ecephys session (from cache.get_ecephys_session)
    - snr: float. Minimum signal to noise ratio
    - isi_violations: float. Minimum inter spike interval violations
    - firing_rate: float. Minimun firing rate 
    
    Returns
    ----------
    pandas.DataFrame with all units that pass the threshold
    '''

    units = session.get_units()
    channels = session.get_channels()
    unit_channels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    good_unit_filter = ((unit_channels['snr']>snr)&
                        (unit_channels['isi_violations']<isi_violations)&
                        (unit_channels['firing_rate']>firing_rate))
    good_units = unit_channels.loc[good_unit_filter]

    if area_list is not None:
        good_units = good_units[good_units['structure_acronym'].isin(area_list)]

    return good_units

def get_first_n_stimulus_presentations(stimulus_presentations, image_name_list=None, n_start_after_change=0, n_presentations=None, select_same_length=False, exclude_initial_trials=0, experiment_mode='active'):
    # Filter: include only experiment_mode specified
    stimulus_presentations = stimulus_presentations[stimulus_presentations[experiment_mode]].reset_index()

    # Filter: exclude initial trials (e.g. auto-reward for active phase)
    first_valid_change = np.where(stimulus_presentations.iloc[exclude_initial_trials:].is_change)[0][0]
    stimulus_presentations = stimulus_presentations[stimulus_presentations.index >= exclude_initial_trials + first_valid_change].reset_index(drop=True)

    # If no image_name specified, select for first image presented
    if image_name_list is None:
        image_name_list = [stimulus_presentations['image_name'].iloc[0]]

    # For each change time, add 'n_after_change', and 'n_change_block'
    change_times = stimulus_presentations.loc[stimulus_presentations.is_change]    
    n_after_change = []
    n_change_block = []
    for ii, (idx_change, idx_next_change) in enumerate(zip(change_times.index[:-1], change_times.index[1:])):
        n_after_change.extend(list(np.arange(idx_next_change-idx_change)))
        n_change_block.extend([ii]*(idx_next_change-idx_change))

    # Cut last trial + add columns
    stimulus_presentations = stimulus_presentations[:idx_next_change]
    stimulus_presentations['n_after_change'] = n_after_change
    stimulus_presentations['n_change_block'] = n_change_block
    
    # If not n_presentations specific, use maximum
    if n_presentations is None:
        n_presentations = np.max(stimulus_presentations.n_after_change)

    # Make selections for all image_names
    n_select = np.arange(n_start_after_change,n_start_after_change + n_presentations)
    selection_all = []
    for image_name in image_name_list:

        # Select n_presentations flashes, starting from n_start_after_change flashes after every change to image_name
        selection_image_name = stimulus_presentations.loc[stimulus_presentations.n_after_change.isin(n_select) \
                                                            & (stimulus_presentations.image_name==image_name)]

        # If select_same_length is True, clip selected flashes
        if select_same_length:
            min_shared_length = np.min(selection_image_name[['n_change_block','n_after_change']].groupby(['n_change_block']).max().values)
            n_select_same_length = [nn for nn in n_select if nn<=min_shared_length]
            selection_image_name = selection_image_name[selection_image_name.n_after_change.isin(n_select_same_length)]
        
        selection_all.append(selection_image_name)

    return selection_all

def get_population_vectors(units, spike_times, event_start_times, event_end_times, time_before_event, time_after_event):
    population_vectors = []
    for event_start, event_end in zip(event_start_times, event_end_times):
        rates = []
        for ui in units.index:
            spike_times_ui = spike_times[ui]
            onset_window   = event_start - time_before_event
            offset_window  = event_end + time_after_event

            assert offset_window > onset_window

            spike_times_ui_window = spike_times_ui[(spike_times_ui > onset_window) \
                                                    & (spike_times_ui < offset_window)]
            rate_ui = len(spike_times_ui_window) / (offset_window - onset_window)
            rates.append(rate_ui)
        
        assert len(rates) == len(units)

        population_vectors.append(np.array(rates))
    
    return population_vectors

def get_euclidean_distance(population_vectors, mode='compare_first'):
    # modes: 'compare_first', 'compare_previous'
    dist_all = []
    if mode=='compare_first':
        pv_0 = population_vectors[0]
        for ii, pv_ii in enumerate(population_vectors):
            dist_i = np.linalg.norm(pv_ii - pv_0)
            dist_all.append(dist_i)
    elif mode=='compare_previous':
        for ii, (pv_ii, pv_jj) in enumerate(zip(population_vectors[1:], population_vectors[:-1])):
            dist_i = np.linalg.norm(pv_ii - pv_jj)
            dist_all.append(dist_i)

    return dist_all
    