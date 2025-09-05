# Package imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os
# %matplotlib inline

import sys
sys.path.insert(0, '/code/src')

import euclidean_distances_utils as ed_utils
import euclidean_distances_plots as ed_plots

if __name__=="__main__":

    # Parameters
    areas_of_interest    = ['VISp','CA3']
    n_presentations      = None # 1
    select_same_length   = False # True
    comparison_mode      = 'distance_to_other'

    # Load Visual Behavior Neuropixels 
    from allensdk.brain_observatory.behavior.behavior_project_cache.\
        behavior_neuropixels_project_cache \
        import VisualBehaviorNeuropixelsProjectCache

    cache_dir = '/root/capsule/data/'

    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
                cache_dir=cache_dir, use_static_cache=True)


    # Load test sessions
    test_session_list, test_session_ecephys_id_list, test_session_experience_level_list = ed_utils.get_test_sessions(cache, test_session_behavior_id = 1093668878)

    # Run comparison
    f, axl = ed_plots.plot_all_areas(areas_of_interest, 
                                        test_session_list, 
                                        test_session_ecephys_id_list, 
                                        test_session_experience_level_list, 
                                        n_presentations=n_presentations, 
                                        select_same_length=select_same_length, 
                                        comparison_mode=comparison_mode,
                                        verbose=False)

    print('done')