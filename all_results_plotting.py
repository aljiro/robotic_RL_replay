#!/usr/bin/python3

'''
Comparisons for all parameters with and without reverse replays.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import csv
import copy

# Storing the raw data
result_parameters = {'tau=0.04_eta=0.001': [],
	'tau=0.04_eta=0.01': [],
    'tau=0.04_eta=0.1': [],
    'tau=0.04_eta=1': [],
    'tau=0.04_eta=10': [],
	'tau=0.2_eta=0.001': [],
    'tau=0.2_eta=0.01': [],
    'tau=0.2_eta=0.1': [],
    'tau=0.2_eta=1': [],
    'tau=0.2_eta=10': [],
    'tau=1_eta=0.001': [],
    'tau=1_eta=0.002': [],
    'tau=1_eta=0.005': [],
    'tau=1_eta=0.01': [],
    'tau=1_eta=0.1': [],
    'tau=1_eta=1': [],
    'tau=1_eta=10': [],
    'tau=5_eta=0.001': [],
    'tau=5_eta=0.01': [],
    'tau=5_eta=0.1': [],
    'tau=5_eta=1': [],
    'tau=5_eta=10': [],}

results_full_nonreplay = copy.deepcopy(result_parameters)
results_full_replay = copy.deepcopy(result_parameters)

with open('data/trial_times/trial_times_NON_REPLAY_plot.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	step = 42
	for i, row in enumerate(data):
		if (2 <= i < step):
			results_full_nonreplay['tau=0.04_eta=0.001'].append([float(j) for j in row[1:]])
		if (step + 2 <= i < 2 * step):
			results_full_nonreplay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (2 * step + 2 <= i < 3 * step):
			results_full_nonreplay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (3 * step + 2 <= i < 4 * step):
			results_full_nonreplay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (4 * step + 2 <= i < 5 * step):
			results_full_nonreplay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])

		if (5 * step + 2 <= i < 6 * step):
			results_full_nonreplay['tau=0.2_eta=0.001'].append([float(j) for j in row[1:]])
		if (6 * step + 2 <= i < 7 * step):
			results_full_nonreplay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (7 * step + 2 <= i < 8 * step):
			results_full_nonreplay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (8 * step + 2 <= i < 9 * step):
			results_full_nonreplay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (9 * step + 2 <= i < 10 * step):
			results_full_nonreplay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])

		if (10 * step + 2 <= i < 11 * step):
			results_full_nonreplay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])
		if (11 * step + 2 <= i < 12 * step):
			results_full_nonreplay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (12 * step + 2 <= i < 13 * step):
			results_full_nonreplay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (13 * step + 2 <= i < 14 * step):
			results_full_nonreplay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (14 * step + 2 <= i < 15 * step):
			results_full_nonreplay['tau=1_eta=10'].append([float(j) for j in row[1:]])

		if (15 * step + 2 <= i < 16 * step):
			results_full_nonreplay['tau=5_eta=0.001'].append([float(j) for j in row[1:]])
		if (16 * step + 2 <= i < 17 * step):
			results_full_nonreplay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (17 * step + 2 <= i < 18 * step):
			results_full_nonreplay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (18 * step + 2 <= i < 19 * step):
			results_full_nonreplay['tau=5_eta=1'].append([float(j) for j in row[1:]])
		if (19 * step + 2 <= i < 20 * step):
			results_full_nonreplay['tau=5_eta=10'].append([float(j) for j in row[1:]])

with open('data/trial_times/trial_times_WITH_REPLAY_plot.csv', newline='') as file:
	data = csv.reader(file, delimiter=',')
	step = 42
	for i, row in enumerate(data):
		if (2 <= i < step):
			results_full_replay['tau=0.04_eta=0.001'].append([float(j) for j in row[1:]])
		if (step + 2 <= i < 2 * step):
			results_full_replay['tau=0.04_eta=0.01'].append([float(j) for j in row[1:]])
		if (2 * step + 2 <= i < 3 * step):
			results_full_replay['tau=0.04_eta=0.1'].append([float(j) for j in row[1:]])
		if (3 * step + 2 <= i < 4 * step):
			results_full_replay['tau=0.04_eta=1'].append([float(j) for j in row[1:]])
		if (4 * step + 2 <= i < 5 * step):
			results_full_replay['tau=0.04_eta=10'].append([float(j) for j in row[1:]])

		if (5 * step + 2 <= i < 6 * step):
			results_full_replay['tau=0.2_eta=0.001'].append([float(j) for j in row[1:]])
		if (6 * step + 2 <= i < 7 * step):
			results_full_replay['tau=0.2_eta=0.01'].append([float(j) for j in row[1:]])
		if (7 * step + 2 <= i < 8 * step):
			results_full_replay['tau=0.2_eta=0.1'].append([float(j) for j in row[1:]])
		if (8 * step + 2 <= i < 9 * step):
			results_full_replay['tau=0.2_eta=1'].append([float(j) for j in row[1:]])
		if (9 * step + 2 <= i < 10 * step):
			results_full_replay['tau=0.2_eta=10'].append([float(j) for j in row[1:]])

		if (10 * step + 2 <= i < 11 * step):
			results_full_replay['tau=1_eta=0.001'].append([float(j) for j in row[1:]])
		if (11 * step + 2 <= i < 12 * step):
			results_full_replay['tau=1_eta=0.01'].append([float(j) for j in row[1:]])
		if (12 * step + 2 <= i < 13 * step):
			results_full_replay['tau=1_eta=0.1'].append([float(j) for j in row[1:]])
		if (13 * step + 2 <= i < 14 * step):
			results_full_replay['tau=1_eta=1'].append([float(j) for j in row[1:]])
		if (14 * step + 2 <= i < 15 * step):
			results_full_replay['tau=1_eta=10'].append([float(j) for j in row[1:]])

		if (15 * step + 2 <= i < 16 * step):
			results_full_replay['tau=5_eta=0.001'].append([float(j) for j in row[1:]])
		if (16 * step + 2 <= i < 17 * step):
			results_full_replay['tau=5_eta=0.01'].append([float(j) for j in row[1:]])
		if (17 * step + 2 <= i < 18 * step):
			results_full_replay['tau=5_eta=0.1'].append([float(j) for j in row[1:]])
		if (18 * step + 2 <= i < 19 * step):
			results_full_replay['tau=5_eta=1'].append([float(j) for j in row[1:]])
		if (19 * step + 2 <= i < 20 * step):
			results_full_replay['tau=5_eta=10'].append([float(j) for j in row[1:]])

# Getting the averages and standard deviations for each
averages_nonreplay = copy.deepcopy(result_parameters)

averages_replay = copy.deepcopy(result_parameters)

std_dev_nonreplay = copy.deepcopy(result_parameters)

std_dev_replay = copy.deepcopy(result_parameters)

no_experiments = 40
for key in averages_nonreplay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(len(results_full_nonreplay[key])):
			average += results_full_nonreplay[key][experiment_no][trial_no]
		average /= no_experiments
		averages_nonreplay[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_full_nonreplay[key])):
			standard_dev += (results_full_nonreplay[key][experiment_no][trial_no] - averages_nonreplay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_nonreplay[key].append(standard_dev)

for key in averages_replay:
	for trial_no in range(20):
		average = 0
		for experiment_no in range(len(results_full_replay[key])):
			average += results_full_replay[key][experiment_no][trial_no]
		average /= no_experiments
		averages_replay[key].append(average)

		standard_dev = 0
		for experiment_no in range(len(results_full_replay[key])):
			standard_dev += (results_full_replay[key][experiment_no][trial_no] - averages_replay[key][trial_no])**2
		standard_dev = np.sqrt(standard_dev / no_experiments)
		std_dev_replay[key].append(standard_dev)


################################################################################

def generate_moving_average_data(tau_e, eta):
	time_const = 'tau=' + tau_e + '_'
	eta = 'eta=' + eta
	# calculate moving averages
	moving_average_nonreplay = np.zeros(20)
	moving_average_replay = np.zeros(20)
	moving_average_std_devs_nonreplay = np.zeros(20)
	moving_average_std_devs_replay = np.zeros(20)

	for i in range(20):
		if i == 0:
			moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
			                              averages_nonreplay[time_const + eta][i
			                                                                   + 1]
			moving_average_nonreplay[i] /= 2
			moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
			                                       std_dev_nonreplay[time_const +
			                                                         eta][i + 1]
			moving_average_std_devs_nonreplay[i] /= 2

			moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i + 1]
			moving_average_replay[i] /= 2
			moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
				i + 1]
			moving_average_std_devs_replay[i] /= 2
		elif i == 19:
			moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
			                              averages_nonreplay[time_const + eta][i
			                                                                   - 1]
			moving_average_nonreplay[i] /= 2
			moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
			                                       std_dev_nonreplay[time_const +
			                                                         eta][i - 1]
			moving_average_std_devs_nonreplay[i] /= 2

			moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i - 1]
			moving_average_replay[i] /= 2
			moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
				i - 1]
			moving_average_std_devs_replay[i] /= 2
		else:
			moving_average_nonreplay[i] = averages_nonreplay[time_const + eta][i] + \
			                              averages_nonreplay[time_const + eta][
				                              i - 1] + averages_nonreplay[time_const + eta][i + 1]
			moving_average_nonreplay[i] /= 3
			moving_average_std_devs_nonreplay[i] = std_dev_nonreplay[time_const + eta][i] + \
			                                       std_dev_nonreplay[time_const +
			                                                         eta][i - 1] + std_dev_nonreplay[time_const + eta][
				                                       i + 1]
			moving_average_std_devs_nonreplay[i] /= 3

			moving_average_replay[i] = averages_replay[time_const + eta][i] + averages_replay[time_const + eta][i - 1] + \
			                           averages_replay[time_const + eta][i + 1]
			moving_average_replay[i] /= 3
			moving_average_std_devs_replay[i] = std_dev_replay[time_const + eta][i] + std_dev_replay[time_const + eta][
				i - 1] + \
			                                    std_dev_replay[time_const + eta][i + 1]
			moving_average_std_devs_replay[i] /= 3

	return moving_average_replay, moving_average_nonreplay, moving_average_std_devs_replay, moving_average_std_devs_nonreplay

def produce_plots(list_tau_e, list_eta, comparison='eta', plt_show=False):
	'''
	produces the comparison plots
	:param tau_e: list of strings if using for comparison, else a single string
	:param eta: list of strings if using for comparison, else a single string
	:param comparison: determines whether the plots are comparing across vayring eta or tau. Default is eta.
	:param plt_show: bool
	:return: none
	'''

	if comparison == 'eta':
		tau_e = list_tau_e
		fig1, ax1 = plt.subplots(2, 2)

		eta = list_eta[0]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# print(tau_e, eta)
		# print(moving_average_replay)

		# plot averages
		ax1[0][0].plot(np.arange(1, 21), moving_average_replay)
		ax1[0][0].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[0][0].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[0][0].set_ylim(0, 60)
		ax1[0][0].set_xlim(1, 20)
		ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[0][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[0][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		eta = list_eta[1]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[0][1].plot(np.arange(1, 21), moving_average_replay)
		ax1[0][1].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[0][1].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[0][1].set_ylim(0, 60)
		ax1[0][1].set_xlim(1, 20)
		ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[0][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[0][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		eta = list_eta[2]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[1][0].plot(np.arange(1, 21), moving_average_replay)
		ax1[1][0].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[1][0].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[1][0].set_ylim(0, 60)
		ax1[1][0].set_xlim(1, 20)
		ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[1][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[1][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		eta = list_eta[3]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[1][1].plot(np.arange(1, 21), moving_average_replay, label='With Replay')
		ax1[1][1].plot(np.arange(1, 21), moving_average_nonreplay, label='Without Replay')
		ax1[1][1].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[1][1].set_ylim(0, 60)
		ax1[1][1].set_xlim(1, 20)
		ax1[1][1].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[1][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[1][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)
		ax1[1][1].legend()

		fig1.tight_layout()
		fig1.savefig('figs/replay_vs_non_replay_tau_e=' + tau_e + '.png')
		if plt_show == True:
			plt.show()

	elif comparison == 'tau':
		fig1, ax1 = plt.subplots(2, 2)
		eta = list_eta

		tau_e = list_tau_e[0]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# print(tau_e, eta)
		# print(moving_average_replay)

		# plot averages
		ax1[0][0].plot(np.arange(1, 21), moving_average_replay)
		ax1[0][0].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[0][0].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[0][0].set_ylim(0, 60)
		ax1[0][0].set_xlim(1, 20)
		ax1[0][0].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[0][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[0][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		tau_e = list_tau_e[1]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[0][1].plot(np.arange(1, 21), moving_average_replay)
		ax1[0][1].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[0][1].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[0][1].set_ylim(0, 60)
		ax1[0][1].set_xlim(1, 20)
		ax1[0][1].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[0][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[0][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		tau_e = list_tau_e[2]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[1][0].plot(np.arange(1, 21), moving_average_replay)
		ax1[1][0].plot(np.arange(1, 21), moving_average_nonreplay)
		ax1[1][0].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[1][0].set_ylim(0, 60)
		ax1[1][0].set_xlim(1, 20)
		ax1[1][0].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[1][0].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[1][0].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)

		tau_e = list_tau_e[3]
		moving_average_data = generate_moving_average_data(tau_e, eta)
		moving_average_replay = moving_average_data[0]
		moving_average_nonreplay = moving_average_data[1]
		moving_average_std_devs_replay = moving_average_data[2]
		moving_average_std_devs_nonreplay = moving_average_data[3]

		# plot averages
		ax1[1][1].plot(np.arange(1, 21), moving_average_replay, label='With Replay')
		ax1[1][1].plot(np.arange(1, 21), moving_average_nonreplay, label='Without Replay')
		ax1[1][1].set_title('$\\tau_e = $' + tau_e + 's, $\eta = $' + eta)
		ax1[1][1].set_ylim(0, 60)
		ax1[1][1].set_xlim(1, 20)
		ax1[1][1].set_xticklabels([1, 5, 10, 15, 20])

		# plot standard deviations
		ax1[1][1].fill_between(np.arange(1, 21), moving_average_replay - moving_average_std_devs_replay,
		                       moving_average_replay + moving_average_std_devs_replay,
		                       alpha=0.4)
		ax1[1][1].fill_between(np.arange(1, 21), moving_average_nonreplay - moving_average_std_devs_nonreplay,
		                       moving_average_nonreplay + moving_average_std_devs_nonreplay,
		                       alpha=0.2)
		ax1[1][1].legend()

		fig1.tight_layout()
		fig1.savefig('figs/replay_vs_non_replay_tau_e=' + tau_e + '.png')
		if plt_show == True:
			plt.show()

###############################################################################
# # Plots for tau_e = 0.04s
# tau_e = '0.04'
# list_eta = ['0.01', '0.1', '1', '10']
# produce_plots(tau_e, list_eta, plt_show=True)
#
# ###############################################################################
# # Plots for tau_e = 0.2s
# tau_e = '0.2'
# list_eta = ['0.01', '0.1', '1', '10']
# produce_plots(tau_e, list_eta, plt_show=True)
#
# ###############################################################################
# # Plots for tau_e = 1s
# tau_e = '1'
# list_eta = ['0.001', '0.01', '0.1', '1']
# produce_plots(tau_e, list_eta, plt_show=True)
#
# ###############################################################################
# # Plots for tau_e = 10s
# tau_e = '5'
# list_eta = ['0.01', '0.1', '1', '10']
# produce_plots(tau_e, list_eta, plt_show=True)
#
# ###############################################################################
# # Plots for eta = 0.001
# list_tau_e = ['0.04', '0.2', '1', '5']
# eta = '0.001'
# produce_plots(list_tau_e, eta, comparison='tau', plt_show=True)

###############################################################################
# Producing the bar charts
# setup the figure and axes

parameters = ['tau=0.04_eta=0.001',
	'tau=0.2_eta=0.001',
	'tau=1_eta=0.001',
	'tau=5_eta=0.001',
	'tau=0.04_eta=0.01',
	'tau=0.2_eta=0.01',
	'tau=1_eta=0.01',
	'tau=5_eta=0.01',
	'tau=0.04_eta=0.1',
	'tau=0.2_eta=0.1',
	'tau=1_eta=0.1',
	'tau=5_eta=0.1',
    'tau=0.04_eta=1',
	'tau=0.2_eta=1',
	'tau=1_eta=1',
	'tau=5_eta=1',
    'tau=0.04_eta=10',
    'tau=0.2_eta=10',
    'tau=1_eta=10',
    'tau=5_eta=10']

number_parameters = 20
ten_trial_average_replay = np.zeros(number_parameters)
ten_trial_average_dev_replay = np.zeros(number_parameters)
ten_trial_average_nonreplay = np.zeros(number_parameters)
ten_trial_average_dev_nonreplay = np.zeros(number_parameters)

for j, key in enumerate(parameters):
	# print(i, key)
	average_replay = 0
	average_dev_replay = 0
	average_nonreplay = 0
	average_dev_nonreplay = 0
	for i in range(10):
		average_replay += averages_replay[key][10 + i]
		average_dev_replay += std_dev_replay[key][10 + i]
		average_nonreplay += averages_nonreplay[key][10 + i]
		average_dev_nonreplay += std_dev_nonreplay[key][10 + i]
	average_replay /= 10
	average_dev_replay /= 10
	average_nonreplay /= 10
	average_dev_nonreplay /= 10
	ten_trial_average_replay[j] = average_replay
	ten_trial_average_dev_replay[j] = average_dev_replay
	ten_trial_average_nonreplay[j] = average_nonreplay
	ten_trial_average_dev_nonreplay[j] = average_dev_nonreplay

# 3D bar chart
# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# eta = np.array([0.001, 0.01, 0.1, 1, 10])
# tau_e = np.array([0.04, 0.2, 1, 5])
#
# # get the bar chart positions
# width = 0.5
# start = 0.0
# _x = np.array([0.6, 1.0,
#                1.6, 2.0,
#                2.6, 3.0,
#                3.6, 4.0,
#                4.6, 5.0,
#                ])
# _y = np.array([0.0, 0.5, 1.0, 1.5])
# _xx, _yy = np.meshgrid(_x, _y)
# x, y = _xx.ravel(), _yy.ravel()
#
#
# replay_data = ten_trial_average_replay
# nonreplay_data = ten_trial_average_nonreplay
#
# # putting the data together
# full_data = np.zeros(40)
# for i in range(40):
# 	if i % 2 == 0:
# 		full_data[i] = replay_data[int(i / 2)]
# 	else:
# 		full_data[i] = nonreplay_data[int(i / 2)]
#
#
# bottom = np.zeros(len(eta) * len(tau_e) * 2)
# width = depth = 0.4
# top = np.ones(len(eta) * len(tau_e) * 2)
#
#
#
# ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 0.5, 1, 1]))
# colors = ['blue', 'red'] * 20
# ax1.bar3d(x, y, bottom, width, depth, full_data, shade=True, color=colors)
#
#
#
# ax1.set_xticks(np.arange(len(eta)) + 1)
# ax1.set_yticks(_y + width / 2)
# ax1.set_xticklabels(eta)
# ax1.set_yticklabels(tau_e)
#
# ax1.set_xlabel('$\eta$')
# ax1.set_ylabel('$\\tau_e$')
# ax1.set_zlabel('Time (s)')
# for ii in range(0,360,1):
#         ax1.view_init(elev=10., azim=ii)
#         plt.savefig("figs/barchart/movie%d.png" % ii)


#2D bar charts
eta = np.array([0.001, 0.01, 0.1, 1, 10])
tau_e = np.array([0.04, 0.2, 1, 5])
replay_tau_0_04 = np.zeros(5)
nonreplay_tau_0_04 = np.zeros(5)
replay_tau_0_2 = np.zeros(5)
nonreplay_tau_0_2 = np.zeros(5)
replay_tau_1 = np.zeros(5)
nonreplay_tau_1 = np.zeros(5)
replay_tau_10 = np.zeros(5)
nonreplay_tau_10 = np.zeros(5)

replay_tau_0_04_dev = np.zeros((2,5))
nonreplay_tau_0_04_dev = np.zeros((2,5))
replay_tau_0_2_dev = np.zeros((2,5))
nonreplay_tau_0_2_dev = np.zeros((2,5))
replay_tau_1_dev = np.zeros((2,5))
nonreplay_tau_1_dev = np.zeros((2,5))
replay_tau_10_dev = np.zeros((2,5))
nonreplay_tau_10_dev = np.zeros((2,5))

for i in range(5):
	replay_tau_0_04[i] = ten_trial_average_replay[4 * i]
	nonreplay_tau_0_04[i] = ten_trial_average_nonreplay[4 * i]
	replay_tau_0_2[i] = ten_trial_average_replay[4 * i + 1]
	nonreplay_tau_0_2[i] = ten_trial_average_nonreplay[4 * i + 1]
	replay_tau_1[i] = ten_trial_average_replay[4 * i + 2]
	nonreplay_tau_1[i] = ten_trial_average_nonreplay[4 * i + 2]
	replay_tau_10[i] = ten_trial_average_replay[4 * i + 3]
	nonreplay_tau_10[i] = ten_trial_average_nonreplay[4 * i + 3]

	replay_tau_0_04_dev[1,i] = ten_trial_average_dev_replay[4 * i]
	nonreplay_tau_0_04_dev[1,i] = ten_trial_average_dev_nonreplay[4 * i]
	replay_tau_0_2_dev[1,i] = ten_trial_average_dev_replay[4 * i + 1]
	nonreplay_tau_0_2_dev[1,i] = ten_trial_average_dev_nonreplay[4 * i + 1]
	replay_tau_1_dev[1,i] = ten_trial_average_dev_replay[4 * i + 2]
	nonreplay_tau_1_dev[1,i] = ten_trial_average_dev_nonreplay[4 * i + 2]
	replay_tau_10_dev[1,i] = ten_trial_average_dev_replay[4 * i + 3]
	nonreplay_tau_10_dev[1,i] = ten_trial_average_dev_nonreplay[4 * i + 3]

x = np.arange(1, 6)
width = 0.4

fig, ax = plt.subplots(2, 2)

ax[0,0].title.set_text('$\\tau_e = 0.04s$')
ax[0,0].bar(x, replay_tau_0_04, width, yerr=replay_tau_0_04_dev)
ax[0,0].bar(x + width, nonreplay_tau_0_04, width, yerr=nonreplay_tau_0_04_dev)
ax[0,0].set_xticks([1 + width/2, 2 + width/2, 3 + width/2, 4 + width/2, 5 + width/2])
ax[0,0].set_xticklabels(eta)
ax[0,0].set_ylim(0, 60)
ax[0,0].set_xlabel('$\eta$')
ax[0,0].set_ylabel('t (s)')

ax[1,0].title.set_text('$\\tau_e = 0.2s$')
ax[1,0].bar(x, replay_tau_0_2, width, yerr=replay_tau_0_2_dev)
ax[1,0].bar(x + width, nonreplay_tau_0_2, width, yerr=nonreplay_tau_0_2_dev)
ax[1,0].set_xticks([1 + width/2, 2 + width/2, 3 + width/2, 4 + width/2, 5 + width/2])
ax[1,0].set_xticklabels(eta)
ax[1,0].set_ylim(0, 60)
ax[1,0].set_xlabel('$\eta$')
ax[1,0].set_ylabel('t (s)')

ax[0,1].title.set_text('$\\tau_e = 1s$')
ax[0,1].bar(x, replay_tau_1, width, yerr=replay_tau_1_dev, label='With Replay')
ax[0,1].bar(x + width, nonreplay_tau_1, width, yerr=nonreplay_tau_1_dev, label='Without Replay')
ax[0,1].set_xticks([1 + width/2, 2 + width/2, 3 + width/2, 4 + width/2, 5 + width/2])
ax[0,1].set_xticklabels(eta)
ax[0,1].legend()
ax[0,1].set_ylim(0, 60)
ax[0,1].set_xlabel('$\eta$')
ax[0,1].set_ylabel('t (s)')

ax[1,1].title.set_text('$\\tau_e = 10s$')
ax[1,1].bar(x, replay_tau_10, width, yerr=replay_tau_10_dev)
ax[1,1].bar(x + width, nonreplay_tau_10, width, yerr=nonreplay_tau_10_dev)
ax[1,1].set_xticks([1 + width/2, 2 + width/2, 3 + width/2, 4 + width/2, 5 + width/2])
ax[1,1].set_xticklabels(eta)
ax[1,1].set_ylim(0, 60)
ax[1,1].set_xlabel('$\eta$')
ax[1,1].set_ylabel('t (s)')

plt.show()
