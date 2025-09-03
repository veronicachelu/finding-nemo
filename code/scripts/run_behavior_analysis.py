import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd 
import numpy as np

import sys
sys.path.insert(0, '/code/src/')

import behavior_utils 
import behavior_plots 

if __name__=="__main__":

    ## Complete workflow ##
    reward_threshold    = 2
    thresh_perc_engaged = 0.6
    thresh_dur_engaged  = 0.6

    plot_performance_filtering = True
    verbose                    = True

    # Load Visual Behavior Neuropixels data
    from allensdk.brain_observatory.behavior.behavior_project_cache.\
        behavior_neuropixels_project_cache \
        import VisualBehaviorNeuropixelsProjectCache

    cache_dir = '/root/capsule/data/'

    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
                cache_dir=cache_dir, use_static_cache=True)

    # Extract relevant sessions based on task structure (ephys, 3uL, trained on G, familiar session first)
    ecephys_session_table_final = behavior_utils.extract_relevant_session_info(cache, table_type='ecephys_strict', verbose=verbose)

    # For debugging: reduce table
    ecephys_session_table_final = ecephys_session_table_final.iloc[:2]
    # ecephys_session_table_final.head()

    if verbose:
        print(f'Number of sessions included in distribution: {len(ecephys_session_table_final)}')

    # Extract performance metrics for relevant sessions 
    all_session_performance_summary = behavior_utils.get_performance_summary_all_sessions(cache, ecephys_session_table_final, reward_threshold=reward_threshold, verbose=verbose)

    # Merge with session table
    ecephys_session_table_final        = ecephys_session_table_final.reset_index()
    df_all_session_performance_summary = pd.DataFrame(all_session_performance_summary)
    ecephys_session_table_final        = ecephys_session_table_final.merge(df_all_session_performance_summary, on=['behavior_session_id'])

    # Filter sessions based on performance
    valid_sessions, valid_session_ids, _ = behavior_utils.filter_valid_sessions(ecephys_session_table_final, 
                                                                thresh_perc_engaged=thresh_perc_engaged, 
                                                                thresh_dur_engaged=thresh_dur_engaged, 
                                                                verbose=verbose)

    # Plot performance metrics + filtering
    if plot_performance_filtering:
        all_stats, all_label_stats = behavior_plots.plot_performance_hist(all_session_performance_summary)
        behavior_plots.plot_performance_filtering(all_stats, all_label_stats, thresh_perc_engaged, thresh_dur_engaged)

    # Save valid session ids
    ecephys_valid_session_ids = ecephys_session_table_final.loc[ecephys_session_table_final.behavior_session_id.isin(valid_session_ids), ['behavior_session_id','ecephys_session_id']]
    ecephys_valid_session_ids.to_csv('/root/capsule/resources/good_session_ids.csv')

    # Plot discarded sessions (not saved)
    all_session_id = ecephys_session_table_final.behavior_session_id.unique()

    for sess_id in all_session_id:
        if sess_id not in valid_session_ids:
            session = cache.get_behavior_session(sess_id)
            rolling_performance = session.get_rolling_performance_df()
            behavior_plots.plot_single_session_performance(rolling_performance, title=f'Invalid: session {sess_id}')

    # Plot some good sessions (not saved)
    num_example_sessions = 5
    example_valid_sessions = np.random.choice(valid_session_ids, num_example_sessions)
    for sess_id in all_session_id:
        if sess_id in example_valid_sessions:
            session = cache.get_behavior_session(sess_id)
            rolling_performance = session.get_rolling_performance_df()
            behavior_plots.plot_single_session_performance(rolling_performance, title=f'Valid: session {sess_id}')
        
    print('done')

