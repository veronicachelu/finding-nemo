import sys
sys.path.insert('/code/src/')

if __name__=="__main__":

    ## Complete workflow ##
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Extract relevant sessions based on task structure (ephys, 3uL, trained on G, familiar session first)
    ecephys_session_table_final = extract_relevant_session_info(cache, table_type='ecephys_strict', verbose=True)
    # For debugging: reduce table
    # ecephys_session_table_final = ecephys_session_table_final.iloc[:2]
    # ecephys_session_table_final.head()

    print(f'Number of sessions included in distribution: {len(ecephys_session_table_final)}')

    # Extract performance metrics for relevant sessions 
    all_session_performance_summary = get_performance_summary_all_sessions(ecephys_session_table_final,verbose=True)
    
    # Merge with session table
    ecephys_session_table_final = ecephys_session_table_final.reset_index()
    df_all_session_performance_summary = pd.DataFrame(all_session_performance_summary)
    ecephys_session_table_final = ecephys_session_table_final.merge(df_all_session_performance_summary, on=['behavior_session_id'])

    # Filter behavior sessions based on performance

    # Load session data and merge with session table

    


