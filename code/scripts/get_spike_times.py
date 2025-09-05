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

# Ephys behavioral sessions data
ecephys_session_table = cache.get_ecephys_session_table() 


# Load good behavior session table
good_session = pd.read_csv('/root/capsule/resources/good_session_ids.csv')
good_session_ids = good_session['ecephys_session_id'].to_list()


# INIT PARAMS
n_jobs = 1
time_before_event= 0
time_after_event = 0
path = '/root/capsule/scratch/'



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
        
        mask = (stimulus_presentations_df['is_change'] == True) & (stimulus_presentations_df['active'] == True) # get only active phase changees and 4 flashes after
        for k in range(1, 5): # 4 flashes after
            mask |= mask.shift(k, fill_value=False)
        stimulus_presentations = stimulus_presentations_df.loc[mask]
        stimulus_presentations.to_csv(os.path.join(path, f'stimulus_presentations_{sess}_{subj}.csv'))

        # Filter good units of the session
        good_units_df = select_good_units(session)

        # Extract all spike times of the session
        spike_times_dic = session.spike_times

        if verbose==True:
            print('Starting...'+ str(sess))

       
        # Get starting and ending timestamps for each event (image flash)
        event_start_times = stimulus_presentations.start_time.values[:5]
        event_end_times =stimulus_presentations.end_time.values[:5]

        # get 1D array with pop vectors
        pop_vector= get_population_vectors(good_units_df, spike_times_dic, event_start_times, event_end_times,\
                                        time_before_event, time_after_event)  # List of arrays Ntask_events, Nunits
        pop_vector = np.array(pop_vector) 

        # store in an array 
        filename = os.path.join(path, f'population_vetors_{sess}_{subj}.npy')
        np.save(filename, pop_vector)

        if verbose==True:
            print('Finished!')

        return 'Success! :)'


# Run in parallel over subjects
results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(save_all_population_vectors)(
        sess, cache, ecephys_session_table, time_before_event, time_after_event, path, verbose=True)
    for sess in tqdm((good_session_ids), total=len(good_session_ids[:1])))  