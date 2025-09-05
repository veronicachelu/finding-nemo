import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/code/src')

import euclidean_distances_utils as ed_utils
import euclidean_distances_plots as ed_plots

def plot_dist_self_in_area(dist_to_first_all_list, 
                        images_list,
                        test_session_ecephys_id_list, 
                        test_session_experience_level_list, 
                        colors_images_list, 
                        shared_images_labels, 
                        area_of_interest):

    f1, axl1 = plt.subplots(1,2,figsize=(8,4), sharey=True)
    f2, axl2 = plt.subplots(1,2,figsize=(8,4), sharey=True)
    f = [f1, f2]
    axl = [axl1, axl2]

    window_size1 = 5
    window_size2 = 10

    for jj, (d_first_all, images, ecephys_id, exp_level, colors_images, shared_images) in enumerate(zip(dist_to_first_all_list, images_list, test_session_ecephys_id_list, test_session_experience_level_list, colors_images_list, shared_images_labels)): # iterate over sessions

        # Create arrays (fill nan values where lengths don't match)
        len_first_all = np.max([len(di) for di in d_first_all])
        arr_first_all = np.nan*np.ones((len(d_first_all),len_first_all))
        for i_di, di in enumerate(d_first_all):
            arr_first_all[i_di,:len(di)] = di
        mu_shared_first   = np.nanmean(arr_first_all[shared_images==1,:],axis=0)
        std_shared_first  = np.nanstd(arr_first_all[shared_images==1,:],axis=0)
        mu_sep_first      = np.nanmean(arr_first_all[shared_images==0,:],axis=0)
        std_sep_first     = np.nanstd(arr_first_all[shared_images==0,:],axis=0)

        ax = axl[0][jj]
        ax.plot(np.arange(len(mov_mean(mu_shared_first,window_size1))), mov_mean(mu_shared_first,window_size1), ls='--', c='r', alpha=0.6, label=f'shared')
        ax.plot(np.arange(len(mov_mean(mu_sep_first,window_size1))), mov_mean(mu_sep_first,window_size1), ls='--', c='k', alpha=0.6, label=f'unique')
        ax.set_ylabel(f'Euclidean distances to first ({exp_level})')
            
        colors_image_ids = matplotlib.cm.get_cmap('viridis', len(d_first_all))
        for ii, (d_first, image_name) in enumerate(zip(d_first_all, images)): # iterate over images
            # Plot distances to first
            ax = axl[1][jj]
            ax.scatter(np.arange(len(mov_mean(d_first,window_size2))), mov_mean(d_first,window_size2), s=8, c=colors_image_ids(ii), alpha=0.2)
            ax.plot(np.arange(len(mov_mean(d_first,window_size2))), mov_mean(d_first,window_size2), ls='--', c=colors_image_ids(ii), alpha=0.2, label=f'image {image_name}')
            ax.set_ylabel(f'Euclidean distances to first ({exp_level})')
        
    for axli in axl:
        for axi in axli:
            axi.legend(loc='lower right',fontsize='small')
            axi.spines[['top','right']].set_visible(False)
            axi.set_xlabel(f'Presentation time of image')

    f1.suptitle(f'Area: {area_of_interest}, Session ID: {ecephys_id} (shared / unique)')
    f2.suptitle(f'Area: {area_of_interest}, Session ID: {ecephys_id} (image ids)')

    return f, axl


def plot_dist_other_in_area(dist_to_first_allrefs, 
                            ref_images_list,
                            images_list,
                            test_session_ecephys_id_list, 
                            test_session_experience_level_list, 
                            colors_images_list, 
                            shared_images_labels, 
                            area_of_interest):

    f1, axl1 = plt.subplots(1,2,figsize=(8,4), sharey=True) # for nov/familiar
    f2, axl2 = plt.subplots(1,2,figsize=(8,4), sharey=True) # for individual images
    f = [f1, f2]
    axl = [axl1, axl2]

    window_size1 = 5
    window_size2 = 10

    for jj_ref, (ref_image, d_first_allsessions) in enumerate(zip(ref_images_list,dist_to_first_allrefs)):

        for jj_session, (d_first_allimages, images, ecephys_id, exp_level, colors_images, shared_images) in enumerate(zip(d_first_allsessions, images_list, test_session_ecephys_id_list, test_session_experience_level_list, colors_images_list, shared_images_labels)): # iterate over sessions

            # Transform d_first_allimages into arrays (fill nan values where lengths don't match)
            len_first_all = np.max([len(di) for di in d_first_allimages])
            arr_first_all = np.nan*np.ones((len(d_first_allimages),len_first_all))
            for i_di, di in enumerate(d_first_allimages):
                arr_first_all[i_di,:len(di)] = di
            mu_shared_first   = np.nanmean(arr_first_all[shared_images==1,:],axis=0)
            std_shared_first  = np.nanstd(arr_first_all[shared_images==1,:],axis=0)
            mu_sep_first      = np.nanmean(arr_first_all[shared_images==0,:],axis=0)
            std_sep_first     = np.nanstd(arr_first_all[shared_images==0,:],axis=0)

            ax = axl[0][jj_ref]
            ls = '--' if exp_level=='Familiar' else '-'
            ax.plot(np.arange(len(ed_utils.mov_mean(mu_shared_first,window_size1))), ed_utils.mov_mean(mu_shared_first,window_size1), ls=ls, c='r', alpha=0.6, lw=1.5, label=f'shared ({exp_level})')
            ax.plot(np.arange(len(ed_utils.mov_mean(mu_sep_first,window_size1))), ed_utils.mov_mean(mu_sep_first,window_size1), ls=ls, c='k', alpha=0.6, lw=1.5, label=f'unique ({exp_level})')
            ax.set_ylabel(f'Euclidean distance to {ref_image}')
                
            colors_image_ids = matplotlib.cm.get_cmap('viridis', len(d_first_allimages))
            for ii, (d_first, image_name) in enumerate(zip(d_first_allimages, images)): # iterate over images
                # Plot distances to first
                ax = axl[1][jj_ref]
                ls = '--' if exp_level=='Familiar' else '-'
                ax.plot(np.arange(len(ed_utils.mov_mean(d_first,window_size2))), ed_utils.mov_mean(d_first,window_size2), ls=ls, c=colors_image_ids(ii), alpha=0.2, lw=1.5, label=f'image {image_name}')
                ax.set_ylabel(f'Euclidean distance to {ref_image}')
            
    for axli in axl:
        for axi in axli:
            axi.legend(loc='lower right',fontsize='small')
            axi.spines[['top','right']].set_visible(False)
            axi.set_xlabel(f'Presentation time of image')

    f1.suptitle(f'Area: {area_of_interest}, Session ID: {ecephys_id} (shared / unique)')
    f2.suptitle(f'Area: {area_of_interest}, Session ID: {ecephys_id} (image ids)')
    # plt.savefig(f1, f'/results/{area_of_interest}_{ecephys_id}_binary.png')

    return f, axl


def plot_all_areas(areas_of_interest, 
                    test_session_list, 
                    test_session_ecephys_id_list, 
                    test_session_experience_level_list, 
                    n_presentations=None, 
                    select_same_length=False, 
                    comparison_mode='distance_to_self',
                    verbose=False):
    
    #### Get stimulus presentations and images ###
    stimulus_presentations_list = []
    images_list = []
    for test_session in test_session_list:
        # Get stimulus presentations
        stimulus_presentations = test_session.stimulus_presentations
        stimulus_presentations_list.append(stimulus_presentations)

        # Get all unique images
        images = [im for im in stimulus_presentations.image_name.unique() if not (im=='omitted' or str(im)=='nan')]
        if verbose: print(images)
        images_list.append(images)

    # Get which images are shared
    shared_images = list(set(images_list[0]).intersection(set(images_list[1])))
    shared_images_labels = np.array([[1 if im in shared_images else 0 for im in images] for images in images_list]) # 1: shared, 0: not shared
    if verbose: print(shared_images_labels)

    shared_sorting = [np.argsort(sil)[::-1] for sil in shared_images_labels]
    if verbose: print(shared_sorting)

    colors_images_list = [['r' if l_im==1 else 'k' for l_im in l_images] for l_images in shared_images_labels]
    if verbose: print(colors_images_list) 

    # Select sessions of interest
    selection_all_list = []
    for stimulus_presentations, images in zip(stimulus_presentations_list, images_list):
        selection_all = ed_utils.get_first_n_stimulus_presentations(stimulus_presentations, 
                                                                    image_name_list=images, 
                                                                    n_start_after_change=0, 
                                                                    n_presentations=n_presentations, 
                                                                    select_same_length=select_same_length, 
                                                                    exclude_initial_trials=3, 
                                                                    experiment_mode='active')

        selection_all_list.append(selection_all)
        if verbose: print(len(selection_all))

    ### Get good units and spike times for area of interest ###
    f_all = []; axl_all = []
    for area in areas_of_interest:
        good_units_list = []
        spike_times_list = []
        for test_session in test_session_list:
            # Get good units
            good_units = ed_utils.select_good_units(test_session, 
                                                    snr=1, 
                                                    isi_violations=1, 
                                                    firing_rate=0.1, 
                                                    area_list = [area])
            if verbose: print(len(good_units))
            good_units_list.append(good_units)

            # Get spike times
            spike_times = test_session.spike_times # spike times: dict with keys = units, values: spike times
            spike_times_list.append(spike_times)

        # Get population vectors for stimulus presentations of interest
        time_before_stimulus = 0
        time_after_stimulus  = 0

        population_vectors_all_list = []
        population_vectors_info_all_list = []
        for good_units, spike_times, selection_all in zip(good_units_list, spike_times_list, selection_all_list): # iterate over sessions
            population_vectors_all = []
            population_vectors_info_all = []
            for ii in range(len(selection_all)): # iterate over images
                population_vectors = ed_utils.get_population_vectors(units = good_units, 
                                                                    spike_times = spike_times, 
                                                                    event_start_times = selection_all[ii]['start_time'].values, 
                                                                    event_end_times = selection_all[ii]['end_time'].values, 
                                                                    time_before_event = time_before_stimulus, 
                                                                    time_after_event = time_after_stimulus)
                if verbose: print(f'Image {ii}: {len(population_vectors)}')

                population_vectors_info = pd.DataFrame({'image_name': selection_all[ii].image_name.values,
                                                        'n_change_block': selection_all[ii].n_change_block.values,
                                                        'n_after_change': selection_all[ii].n_after_change.values,
                                                        'population_vector_id': np.arange(len(population_vectors))})
                
                # population_vectors_info
                population_vectors_all.append(population_vectors)
                population_vectors_info_all.append(population_vectors_info)
            
            population_vectors_all_list.append(population_vectors_all)
            population_vectors_info_all_list.append(population_vectors_info_all)

        # Compute Euclidean distances for all selection events
        if comparison_mode=='distance_to_self':
            dist_to_first_all_list = []
            for pv_all, pvi_all in zip(population_vectors_all_list, population_vectors_info_all_list): # iterate over sessions
                dist_to_first_all = []
                for pv, pvi in zip(pv_all, pvi_all): # iterate over images
                    dist_to_first = ed_utils.get_euclidean_distance(pv, 
                                                                    mode='compare_first')
                    dist_to_first_all.append(dist_to_first)
                    if verbose: print(len(dist_to_first))
                
                dist_to_first_all_list.append(dist_to_first_all)

            ### Plot for area of interest ###
            f, axl = ed_plots.plot_dist_self_in_area(dist_to_first_all_list, 
                                                        images_list,
                                                        test_session_ecephys_id_list, 
                                                        test_session_experience_level_list, 
                                                        colors_images_list, 
                                                        shared_images_labels, 
                                                        area)

            f_all.append(f); axl_all.append(axl)
    
        elif comparison_mode=='distance_to_other':

            ref_image_list = shared_images
            
            dist_to_first_allrefs = []
            for ref_image in ref_image_list:

                dist_to_first_allsessions = []
                for pv_all, pvi_all, iml in zip(population_vectors_all_list, population_vectors_info_all_list, images_list): # iterate over sessions
                    dist_to_first_allimages = []
                    # Get population vectors for reference image
                    idx_ref = [idx_im for idx_im, im in enumerate(iml) if im==ref_image][0]
                    ref_pv  = pv_all[idx_ref]
                    ref_pvi = pvi_all[idx_ref]
                    for pv, pvi in zip(pv_all, pvi_all): # iterate over images
                        dist_to_first = ed_utils.get_euclidean_distance(pv, 
                                                                        ref_vectors=ref_pv, 
                                                                        mode='compare_ref')
                        dist_to_first_allimages.append(dist_to_first)
                        if verbose: print(len(dist_to_first))
                    
                    dist_to_first_allsessions.append(dist_to_first_allimages)

                dist_to_first_allrefs.append(dist_to_first_allsessions)
            
            ### Plot for area of interest ###
            f, axl = ed_plots.plot_dist_other_in_area(dist_to_first_allrefs, 
                                                        ref_image_list,
                                                        images_list,
                                                        test_session_ecephys_id_list, 
                                                        test_session_experience_level_list, 
                                                        colors_images_list, 
                                                        shared_images_labels, 
                                                        area)
            f_all.append(f); axl_all.append(axl)

    return f_all, axl_all
