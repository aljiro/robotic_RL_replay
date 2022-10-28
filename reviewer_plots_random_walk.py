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
from plot_util import *


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

N_EXP = 39
gazebo = "best_gazebo7_correct"
# Loading the random walk data
with open("data/trial_times/%s/random_times_NON_REPLAY_FULL.csv"%gazebo, newline='') as file:
	data = csv.reader(file, delimiter=',')
	for i, row in enumerate(data):
		if (1 <= i < N_EXP):
			results_non_replay.append([float(j) for j in row[1:]])


with open("data/trial_times/%s/random_times_WITH_REPLAY_FULL.csv"%gazebo, newline='') as file:
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

options = {'ylabel': 'Number of times the random walk\nmodule was used',
		   'figname': "random_walk_comparison_%s.png"%gazebo,
		   'ylim': (0, 70), 'loc': 'best'}

plotLineSeries(results_replay, results_non_replay, options)
# plotBoxSeries(results_replay, results_non_replay, options)
