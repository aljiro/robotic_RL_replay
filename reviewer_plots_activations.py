# Plotting the evolution of wall bumping and percentage of random 
# walks as a function of trials

#!/usr/bin/python3

'''
Plotting the best cases. For replay, this is tau_e = 0.04s, eta = 1; non-replay is tau_e = 1s, eta = 0.01
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

N_EXP = 3
results_non_replay = []
results_replay = []
means_non_replay = []
means_replay = []
medians_non_replay = []
medians_replay = []
std_devs_non_replay = []
std_devs_replay = []
percentiles_non_replay = []
percentiles_replay = []

# Loading the random walk data
gazebo = "exp_gazebo9"
with open("data/trial_times/%s/activations_NON_REPLAY_FULL.csv"%gazebo, newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (1 <= i < N_EXP):
			results_non_replay.append([float(j) for j in row[1:]])

with open("data/trial_times/%s/activations_WITH_REPLAY_FULL.csv"%gazebo, newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (1 <= i < N_EXP):
			results_replay.append([float(j) for j in row[1:]])


# Data is stored as a list of the values for each experiments. As in results_non_replay[experiment1_list,
# experiment2_list, ..., experiment40_list]. It is easier to have the data as a list of trial values first. So
# results_non_replay[trial1_list, trial2_list, ..., trial30_list]
results_change_up_non_replay = []
results_change_up_replay = []
for trial_no in range(len(results_non_replay[0])):
	trial_result_non_replay = []
	trial_result_replay = []
	for experiment_no in range(len(results_non_replay)):
		trial_result_non_replay.append(results_non_replay[experiment_no][trial_no])
		trial_result_replay.append(results_replay[experiment_no][trial_no])
	results_change_up_non_replay.append(trial_result_non_replay)
	results_change_up_replay.append(trial_result_replay)
results_non_replay = results_change_up_non_replay
results_replay = results_change_up_replay

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
plt.xlabel('Trial No.')
plt.ylabel('Average population vector magnitude')

# plot standard deviations
plt.fill_between(np.arange(1, 31), np.array(means_replay_mov_avg) - np.array(std_devs_replay_mov_avg),
                       np.array(means_replay_mov_avg) + np.array(std_devs_replay_mov_avg),
                       alpha=0.4)
plt.fill_between(np.arange(1, 31), np.array(means_non_replay_mov_avg) - np.array(std_devs_non_replay_mov_avg),
                       np.array(means_non_replay_mov_avg) + np.array(std_devs_non_replay_mov_avg),
                       alpha=0.2)
plt.legend()
plt.savefig("activations_comparison_%s.png"%gazebo)
plt.show()
