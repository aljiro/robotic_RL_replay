import numpy as np
import matplotlib.pyplot as plt

def plotLineSeries(results_replay, results_non_replay, options):
    means_non_replay = []
    means_replay = []
    medians_non_replay = []
    medians_replay = []
    std_devs_non_replay = []
    std_devs_replay = []
    percentiles_non_replay = []
    percentiles_replay = []

    # Computing the means and standard deviations
    for trial_no in range(len(results_non_replay)):
        means_non_replay.append(np.mean(results_non_replay[trial_no]))
        means_replay.append(np.mean(results_replay[trial_no]))
        std_devs_non_replay.append(np.std(results_non_replay[trial_no]))
        std_devs_replay.append(np.std(results_replay[trial_no]))

    # Getting moving averages from the data
    means_non_replay_mov_avg = []
    means_replay_mov_avg = []
    std_devs_non_replay_mov_avg = []
    std_devs_replay_mov_avg = []
    for i in range(30):
        if i == 0:
            means_non_replay_mov_avg.append((means_non_replay[0] + means_non_replay[1]) / 2)
            means_replay_mov_avg.append((means_replay[0] + means_replay[1]) / 2)
            std_devs_non_replay_mov_avg.append((std_devs_non_replay[0] + std_devs_non_replay[1]) / 2)
            std_devs_replay_mov_avg.append((std_devs_replay[0] + std_devs_replay[1]) / 2)
        elif i == 29:
            means_non_replay_mov_avg.append((means_non_replay[28] + means_non_replay[29]) / 2)
            means_replay_mov_avg.append((means_replay[28] + means_replay[29]) / 2)
            std_devs_non_replay_mov_avg.append((std_devs_non_replay[28] + std_devs_non_replay[29]) / 2)
            std_devs_replay_mov_avg.append((std_devs_replay[28] + std_devs_replay[29]) / 2)
        else:
            means_non_replay_mov_avg.append((means_non_replay[i - 1] + means_non_replay[i] + means_non_replay[i + 1])  / 3)
            means_replay_mov_avg.append((means_replay[i - 1] + means_replay[i] + means_replay[i + 1]) / 3)
            std_devs_non_replay_mov_avg.append((std_devs_non_replay[i - 1] + std_devs_non_replay[i] +
                                                std_devs_non_replay[i + 1]) / 3)
            std_devs_replay_mov_avg.append((std_devs_replay[i - 1] + std_devs_replay[i] + std_devs_replay[i + 1]) / 3)

    # plot averages

    plt.plot(np.arange(1, 31), means_replay_mov_avg, label='With Replay')
    plt.plot(np.arange(1, 31), means_non_replay_mov_avg, label='Without Replay')
    # plt.title('Best cases comparison')
    # plt.ylim(0, 60)
    # plt.xlim(1, 30)
    plt.xlabel('Trial No.', fontsize=14)
    plt.ylabel(options['ylabel'], fontsize=14)

    # plot standard deviations
    plt.fill_between(np.arange(1, 31), np.array(means_replay_mov_avg) - np.array(std_devs_replay_mov_avg),
                        np.array(means_replay_mov_avg) + np.array(std_devs_replay_mov_avg),
                        alpha=0.4)
    plt.fill_between(np.arange(1, 31), np.array(means_non_replay_mov_avg) - np.array(std_devs_non_replay_mov_avg),
                        np.array(means_non_replay_mov_avg) + np.array(std_devs_non_replay_mov_avg),
                        alpha=0.2)
    plt.legend(fontsize=14, loc = options['loc'])
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.xlim((1, 30))
    plt.ylim(options['ylim'])
    fig = plt.gcf()
    fig.set_size_inches(8.5, 6)
    plt.savefig(options['figname'])
    plt.show()


def plotBoxSeries(results_replay, results_non_replay, options):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x = np.arange(1, 31)
    idxs = slice(0, len(x), 3)
    w = 0.37
    bp1 = plt.boxplot( results_replay[idxs], 
                       positions = x[idxs] - w, 
                       widths=0.6, 
                       patch_artist=True, 
                       sym = '*' )
    bp2 = plt.boxplot( results_non_replay[idxs], 
                       positions = x[idxs] + w, 
                       widths=0.6, 
                       patch_artist=True, 
                       sym = '*' )


    for i in range(len(bp1['boxes'])):
        bp1['boxes'][i].set_facecolor(colors[0])
        bp1['medians'][i].set_color('black')

    for i in range(len(bp2['boxes'])):
        bp2['boxes'][i].set_facecolor(colors[1])
        bp2['medians'][i].set_color('black')

    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['With Replay', 'Without Replay'], fontsize=14, loc = options['loc'])

    plt.xlabel('Trial No.', fontsize=14)
    plt.ylabel(options['ylabel'], fontsize=14)
    plt.yticks(fontsize=13)
    plt.xticks(x[idxs], labels = [str(c) for c in x[idxs]], fontsize=13)
    fig = plt.gcf()
    fig.set_size_inches(10.5, 6)
    plt.show()

