import allensdk
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas as pd
import platform
import sys
import platform
import os
from os.path import join as pjoin
from pathlib import Path
from joblib import Parallel, delayed 
sys.path.insert(0, '/code/src')

def select_good_units(session, snr=1, isi_violations=1, firing_rate=0.1):
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
    good_units = good_units.sort_values('probe_vertical_position', ascending=False)     #sort y depth for later alignments

    return good_units



def get_population_vectors(units, spike_times, event_start_times, event_end_times, time_before_event, time_after_event):
    '''
    Extracts the firing rates of certain units arround specific task events

    Parameters
    ----------
    - units: pandas.DataFrame contianing the good_units
    - spike_times: dictionary. keys units, values spiketimes over the whole session
    - event_start_times: array. Times when the desired event starts
    - event_end_times: array. Times when the desired event starts 
    - time_before_event: float. Extra time before event_start
    - time_before_event: float. Extra time after event_start

    
    Returns
    ----------
    List of arrays, Ntask_events, Nunits
    '''

    population_vectors = []
    for event_start, event_end in zip(event_start_times, event_end_times):
        rates = []
        for ui in units.index: # loop over the selected units
            spike_times_ui = spike_times[ui]
            onset_window   = event_start - time_before_event
            offset_window  = event_end + time_after_event
            assert offset_window > onset_window
            spike_times_ui_window = spike_times_ui[(spike_times_ui > onset_window) \
                                                    & (spike_times_ui < offset_window)]
            rate_ui = len(spike_times_ui_window) / (offset_window - onset_window) # calculate the mean firing rate
            rates.append(rate_ui)
        assert len(rates) == len(units)
        population_vectors.append(np.array(rates))
    return population_vectors


def get_first_n_stimulus_presentations(stimulus_presentations, image_name_list=None, n_start_after_change=0, n_presentations=None, select_same_length=False, exclude_initial_trials=0, experiment_mode='active'):
    # Filter: include only experiment_mode specified
    stimulus_presentations = stimulus_presentations[stimulus_presentations[experiment_mode]].reset_index()

    # Filter: exclude initial trials (e.g. auto-reward for active phase)
    first_valid_change = np.where(stimulus_presentations.iloc[exclude_initial_trials:].is_change)[0][0]
    stimulus_presentations = stimulus_presentations[stimulus_presentations.index >= exclude_initial_trials + first_valid_change].reset_index(drop=True)

    # If no image_name specified, select for first image presented
    if image_name_list is None:
        image_name_list = [im_i for im_i in stimulus_presentations['image_name'].unique() if im_i!='omitted']

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


def save_all_population_vectors(sess, cache, ecephys_session_table, time_before_event, time_after_event, path, verbose=False):
        """
        Extract the populations
        """

        # Select the session
        session = cache.get_ecephys_session(sess)

        # Extract mouse_id
        subj= ecephys_session_table.loc[sess]['mouse_id']

        # Save stimulus presentations
        stimulus_presentations = session.stimulus_presentations
        stimulus_presentations.to_csv(os.path.join(path, f'stimulus_presentations_all_{sess}_{subj}.csv'))

        # mask = (stimulus_presentations_df['is_change'] == True) & (stimulus_presentations_df['active'] == True) # get only active phase changees and 4 flashes after
        # for k in range(1, 5): # 4 flashes after
        #     mask |= mask.shift(k, fill_value=False) # WARNING NOT ELECTING PROPERLY!
        # stimulus_presentations = stimulus_presentations_df.loc[mask]

        # Select events for population vector computation
        selection_all = get_first_n_stimulus_presentations(stimulus_presentations, 
                                                           image_name_list=None, 
                                                           n_start_after_change=0, 
                                                           n_presentations=5, 
                                                           select_same_length=True, 
                                                           exclude_initial_trials=3, 
                                                           experiment_mode='active')
        selection_stimuli = pd.concat(selection_all)
        selection_stimuli.to_csv(os.path.join(path, f'stimulus_presentations_selected_{sess}_{subj}.csv'))
    
        # Filter good units of the session
        good_units_df = select_good_units(session)
        good_units_df.to_csv(os.path.join(path, f'good_units_{sess}_{subj}.csv'))

        # Extract all spike times of the session
        spike_times_dic = session.spike_times

        if verbose==True:
            print('Starting...'+ str(sess))

        # Get starting and ending timestamps for each event (image flash)
        event_start_times = selection_stimuli.start_time.values
        event_end_times = selection_stimuli.end_time.values

        # get 1D array with pop vectors
        pop_vector= get_population_vectors(good_units_df, spike_times_dic, event_start_times, event_end_times,\
                                          time_before_event, time_after_event)  # List of arrays Ntask_events, Nunits
        pop_vector = np.array(pop_vector)  # n_events x n_units

        # store in an array 
        filename = os.path.join(path, f'population_vectors_{sess}_{subj}.npy')
        np.save(filename, pop_vector)

        if verbose==True:
            print('Finished!')

        return 'Success! :)'

if __name__=="__main__":

    ##### LOAD DATA FROM ALLEN SDK #####
    platstring = platform.platform()
    system = platform.system()
    if system == "Darwin":
        # macOS
        data_dir = "/Volumes/Brain2025/"
    elif system == "Windows":
        # Windows (replace with the drive letter of USB drive)
        data_dir = "E:/"
    elif "amzn" in platstring:
        # then on CodeOcean
        data_dir = "/data/"
    else:
        # then your own linux platform
        # EDIT location where you mounted hard drive
        data_dir = "/media/$USERNAME/Brain2025/"
    data_root = Path(data_dir)
    print('data directory set to', data_dir)

    # Visual Behavior Neuropixels 
    from allensdk.brain_observatory.behavior.behavior_project_cache.\
        behavior_neuropixels_project_cache \
        import VisualBehaviorNeuropixelsProjectCache

    cache_dir = '/root/capsule/data/'

    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
                cache_dir=cache_dir, use_static_cache=True)

    ### GET SESSIONS ###
    ecephys_session_table = cache.get_ecephys_session_table() 

    # Load good behavior session table
    good_session = pd.read_csv('/root/capsule/resources/good_session_ids.csv')
    good_session_ids = good_session['ecephys_session_id'].to_list()
    # good_session_ids = [1093638203, 109386780

    ### INIT PARAMS ###
    n_jobs = 1
    time_before_event= 0
    time_after_event = 0
    path = '/root/capsule/scratch/'

    ### ITERATE OVER SESSIONS ###
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(save_all_population_vectors)(
            sess, cache, ecephys_session_table, time_before_event, time_after_event, path, verbose=True)
        for sess in tqdm((good_session_ids), total=len(good_session_ids))) 

    print(results)