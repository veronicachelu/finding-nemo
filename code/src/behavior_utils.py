import numpy as np
import pandas as pd

def extract_relevant_session_info(cache, table_type='ecephys', verbose=False):
    # table_type: 'ecephys', 'ecephys_strict', 'behavior'
    
    if table_type == 'behavior':
        session_table = cache.get_behavior_session_table() 
        session_table_final = session_table[(session_table.session_type.str.startswith('EPHYS')) & 
                                            # (behavior_session_table.genotype=='wt/wt') & ## genotypes okay for ephys
                                            (session_table.session_type.str.contains('3uL')) ## 5 uL: not motivated long enough
                                            ]
    elif 'ecephys' in table_type:
        # Filter only for mice that are (i) trained on set G, (ii) tested on first G ('familiar'), then H ('novel').
        session_table = cache.get_ecephys_session_table() 
        session_table_final = session_table[((session_table.session_type.str.startswith('EPHYS')) 
                                                # & (behavior_session_table.genotype=='wt/wt')          ## genotypes okay for ephys
                                                & (session_table.session_type.str.contains('3uL')))     ## 5 uL: not motivated long enough
                                            & ((session_table.image_set=='G') 
                                                & (session_table.experience_level=='Familiar') 
                                                & (session_table.session_number==1)) 
                                            | ((session_table.image_set=='H') 
                                                & (session_table.experience_level=='Novel') 
                                                & (session_table.session_number==2))
                                            ]
        
        if 'strict' in table_type:
            # Get mice that only have both session types
            full_session_mice_idx = np.where(session_table_final['mouse_id'].value_counts()==2)[0]
            full_session_mice     = list(session_table_final['mouse_id'].value_counts().index[full_session_mice_idx])
            session_table_final   = session_table_final[session_table_final.mouse_id.isin(full_session_mice)]
    
    if verbose:
        print(f'Unique genotypes: {session_table_final.genotype.unique()}')
        print(f'Unique session types: {session_table_final[["session_type"]].value_counts()}')
        print(f'Number of sessions: {len(session_table_final)}')
        print(f'Number of mice: {session_table_final[["mouse_id"]].value_counts()}')
        
    return session_table_final


def get_perc_engaged(roll_perf, rthresh): 
    return len(roll_perf.loc[roll_perf['reward_rate']>rthresh]) / len(roll_perf)


def get_performance_summary(rolling_performance, reward_threshold=2,verbose=False):
    
    # Compute percentage spent in engaged state
    perc_engaged = get_perc_engaged(rolling_performance, reward_threshold)
    if verbose: print(f'Percentage engaged: {perc_engaged}')

    # Compute percentage spent in engaged state (before last disengagement)
    trial_last_disengagement = rolling_performance.loc[rolling_performance['reward_rate']>reward_threshold].index[-1]+1
    perc_engaged_strict = get_perc_engaged(rolling_performance.iloc[:trial_last_disengagement], reward_threshold)
    if verbose: print(f'Percentage engaged (strict): {perc_engaged_strict}')

    performance_summary = {'perc_engaged': perc_engaged,
                            'perc_engaged_strict': perc_engaged_strict,
                            'trial_last_disengagement': trial_last_disengagement,
                            'trial_number': len(rolling_performance)
                        }

    return performance_summary


def get_performance_summary_all_sessions(ecephys_session_table,verbose=False):
    
    all_session_performance_summary = []
    for ii, session_id in enumerate(ecephys_session_table.behavior_session_id.values):
        if verbose: 
            print(f'Loading session {ii}/{len(ecephys_session_table)}, id: {session_id}.')

        # Load session
        session = cache.get_behavior_session(session_id)

        # Extract performance summary
        rolling_performance = session.get_rolling_performance_df()
        performance_summary = get_performance_summary(rolling_performance, reward_threshold=2, verbose=verbose)
        performance_summary['behavior_session_id'] = session_id

        all_session_performance_summary.append(performance_summary)
    
    return all_session_performance_summary

def filter_behavior_sessions_performance():
    raise NotImplementedError

def extract_relevant_sessions():
    raise NotImplementedError
