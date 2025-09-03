import numpy as np
import matplotlib.pyplot as plt

def plot_performance_filtering(all_stats, all_label_stats, thresh_perc_engaged, thresh_dur_engaged):
    
    f, ax = plt.subplots(1,1,figsize=(4,4))
    ax.scatter(all_stats[2],all_stats[1],s=10,c=['r' if (as1 > thresh_perc_engaged and as2 > thresh_dur_engaged) else 'b' for as2, as1 in zip(all_stats[2],all_stats[1])])
    ax.axvline(x=thresh_dur_engaged, c='r', ls='--',label='threshold for inclusion')
    ax.axhline(y=thresh_perc_engaged, c='r', ls='--')
    ax.set_xlabel(all_label_stats[2])
    ax.set_ylabel(all_label_stats[1])
    ax.axhline(y=1, c='k', ls='--')
    ax.axvline(x=1, c='k', ls='--')
    ax.spines[['top','right']].set_visible(False)
    ax.legend()

    f.tight_layout()

def plot_performance_hist(all_session_performance_summary):
    all_perc = []
    all_perc_strict = []
    all_trial_disengage = []
    all_session_length = []
    for sess_summary in all_session_performance_summary:
        all_perc.append(sess_summary['perc_engaged'])
        all_perc_strict.append(sess_summary['perc_engaged_strict'])
        all_trial_disengage.append(sess_summary['trial_last_disengagement'])
        all_session_length.append(sess_summary['trial_number'])

    all_perc_disengage = list(np.array(all_trial_disengage) / np.array(all_session_length))

    f, axl = plt.subplots(1, 3, figsize=(12, 4))

    num_bins = 19
    all_stats       = [all_perc, all_perc_strict, all_perc_disengage]
    all_label_stats = ['% engaged / session', '% engaged / session (strict)', '% session before \n disengagement']
    all_bins        = [np.linspace(0, 1, num_bins)]*3
    filter_session  = '3*std'
    for i, (plot_stats, label_stats, bins) in enumerate(zip(all_stats, all_label_stats, all_bins)):
        ax = axl[i]
        mu = np.mean(plot_stats)
        std = np.std(plot_stats)
        ax.hist(plot_stats, bins=bins)
        ylim = ax.get_ylim()
        ax.axvline(x=mu, label=f'$\\mu$={np.round(mu,2)}, $\\sigma$={np.round(std,2)}', c='r', ls='--')
        ax.fill_betweenx(y=ylim, x1=mu-std, x2=mu+std, color='r', alpha=0.1)
        ax.set_ylim(ylim)
        ax.set_xlabel(label_stats)
        ax.set_ylabel('Count')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left')

    f.tight_layout()

    return all_stats, all_label_stats

def plot_single_session_performance(rolling_performance, title):
    f, ax = plt.subplots(1,1)
    ax.plot(rolling_performance['reward_rate'], 'k', label='reward rate (av. rew. / minute)')
    ax.plot(rolling_performance['hit_rate'], 'r', label='hit rate')
    ax.plot(rolling_performance['false_alarm_rate'], 'b', label='false alarm rate')
    ax.plot(rolling_performance['hit_rate'] + rolling_performance['false_alarm_rate'], 'grey', label='lick rate')
    ax.axhline(y=2,c='k', ls='--', label='Minimum reward rate')

    ax.set_xlabel('Trial ID')
    ax.set_ylabel('Rolling rate')
    ax.spines[['top','right']].set_visible(False)
    ax.legend()
    ax.set_title(title)